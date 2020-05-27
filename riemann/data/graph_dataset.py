from math import floor
from typing import Iterator, Dict
from itertools import combinations, chain

import numpy as np
from graph_tool import Graph
from tqdm import tqdm
from scipy.special import comb

from .graph_data_batch import GraphDataBatch
from .graph_data_batch_iterator import GraphDataBatchIterator
from .pickle_manager import load_or_gen
from ..config.config_loader import get_config
from ..config.graph_sampling_config import GraphSamplingConfig
from ..graph_embedder import GraphEmbedder
from ..manifold_nns import ManifoldNNS


def comb_index(n, k):
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)),
                        int, count=count*k)
    return index.reshape(-1, k)

class GraphDataset:
    """
    Class for managing datasets with graph data
    """

    def __init__(self, name, edges, object_ids, weights, hidden_graph=None):
        """
        Params:
            name (str): unique string to name this dataset (for pickling and
                unpickling)
            edges (numpy.ndarray): numpy array of shape [num_edges, 2]
                containing the indices of nodes in all edges
            objects (List[str]): string object ids for all nodes
            weights (numpy.ndarray): numpy array of shape [num_edges]
                containing edge weights
            hidden_graph (GraphDataset): Graph data that should be excluded
                but not considered as negative edges. (i.e. train
                edges should not be in eval dataset but they shouldn't be
                counted as negatives either)
        """

        self.name = name
        self.edges = edges
        self.object_ids = np.asarray(object_ids)
        self.weights = weights
        self.hidden_graph = hidden_graph

        self.graph = Graph(directed=False)
        self.graph.add_vertex(len(object_ids))
        edge_weights = [[edge[0], edge[1], weight] for edge, weight in
                        zip(self.edges, self.weights)]
        self.weight_property = self.graph.new_edge_property("float")
        eprops = [self.weight_property]
        self.graph.add_edge_list(edge_weights, eprops=eprops)
        self.manifold_nns = None

    def gen_neighbor_data(self, verbose=True) -> Dict:
        """
        Generates the graph data needed to run the cython iterator
        Returns a dict with the neighbor data which will have values

        - 'non_empty_vertices' the indices of vertices which have edges
           emanating from them
        - 'all_graph_neighbors' a list of lists of ints such that the list of
          edges emanating from the vertex with index non_empty_vertices[i] is
          stored in all_graph_neighbors[i]
        - 'all_graph_weights' a list of lists of ints such that
          all_graph_weights[i][j] represents the weight of the connection in
          all_graph_neighbors[i][j]
        - 'N' number of nodes in the graph

        Parameters:
            verbose (bool): should graph loading be printed out
        """

        all_graph_neighbors = []
        all_graph_weights = []
        non_empty_vertices = []
        empty_vertices = []
        if verbose:
            iterator = tqdm(range(self.n_nodes()),
                            desc="Generating Neighbor Data", dynamic_ncols=True)
        else:
            iterator = range(self.n_nodes())

        for i in iterator:
            in_edges = self.graph.get_in_edges(i, [self.weight_property])
            out_edges = self.graph.get_out_edges(i, [self.weight_property])
            if in_edges.size + out_edges.size > 0:
                non_empty_vertices.append(i)
                if in_edges.size == 0:
                    all_graph_neighbors.append(out_edges[:, 1].astype(np.int64))
                    all_graph_weights.append(out_edges[:, 2].astype(np.float32))
                elif out_edges.size == 0:
                    all_graph_neighbors.append(in_edges[:, 1].astype(np.int64))
                    all_graph_weights.append(in_edges[:, 2].astype(np.float32))
                else:
                    all_graph_neighbors.append(
                        np.concatenate([in_edges[:, 0], out_edges[:, 1]]
                                       ).astype(np.int64))
                    all_graph_weights.append(
                        np.concatenate([in_edges[:, 2], out_edges[:, 2]]
                                      ).astype(np.float32))
            else:
                empty_vertices.append(i)

        # graph_neighbors = np.concatenate(all_graph_neighbors)
        # graph_neighbor_weights = np.concatenate(all_graph_weights)
        non_empty_vertices = np.array(non_empty_vertices, dtype=np.int64)
        empty_vertices = np.array(empty_vertices, dtype=np.int64)

        return {
            "all_graph_neighbors": all_graph_neighbors,
            "all_graph_weights": all_graph_weights,
            "non_empty_vertices": non_empty_vertices,
            "empty_vertices": empty_vertices,
            "N": self.n_nodes()
        }

    def add_manifold_nns(self, graph_embedder: GraphEmbedder):
        manifold = graph_embedder.get_manifold()
        data_points = graph_embedder.retrieve_nodes(self.n_nodes())
        self.manifold_nns = ManifoldNNS(data_points, manifold)

    def n_nodes(self) -> int:
        """
        Returns the number of nodes in the graph
        """
        return len(self.object_ids)

    def collapse_nodes(self, node_ids):
        all_new_edges = []
        for node_id in tqdm(node_ids, desc="Collapsing Nodes",
                            dynamic_ncols=True):
            in_edges = self.graph.get_in_edges(node_id, [self.weight_property])
            out_edges = self.graph.get_out_edges(node_id, [self.weight_property])
            neighbors = np.concatenate([out_edges[:,1:3], in_edges[:,0:3:2]])
            if neighbors.shape[0] > 1:
                neighbor_combos = \
                    neighbors[comb_index(neighbors.shape[0], 2)]
                neighbor_combos = \
                    neighbor_combos.reshape(neighbor_combos.shape[0], 4)
                new_edges = np.zeros((neighbor_combos.shape[0], 3))
                new_edges[:,:2] += neighbor_combos[:,0:3:2]
                new_edges[:,2] += (neighbor_combos[:,1] + \
                                   neighbor_combos[:,3])/4
                all_new_edges.append(new_edges)

        self.graph.add_edge_list(np.concatenate(all_new_edges), eprops=[self.weight_property])

        self.object_ids = np.delete(self.object_ids, np.array(node_ids))
        self.graph.remove_vertex(node_ids)


        edges_weights = self.graph.get_edges(eprops=[self.weight_property])
        edges = edges_weights[:,0:2]
        weights = edges_weights[:,2]
        self.edges = edges
        self.weights = weights

    def get_neighbor_iterator(self,
                              graph_sampling_config: GraphSamplingConfig,
                              data_fraction: float = 1,
                              ) -> Iterator[GraphDataBatch]:
        """
        Gets an efficient iterator of edge batches
        """
        neighbor_data = load_or_gen(f"GraphDataset.{self.name}",
                                    self.gen_neighbor_data)
        if self.hidden_graph is None:
            # GraphDataBatchIterator is defined in cython with these arguments.
            # noinspection PyArgumentList
            iterator = GraphDataBatchIterator(neighbor_data,
                                              graph_sampling_config)
            iterator.data_fraction = data_fraction

        else:
            hidden_neighbor_data = load_or_gen(
                f"GraphDataset.{self.hidden_graph.name}",
                self.hidden_graph.gen_neighbor_data)

            # GraphDataBatchIterator is defined in cython with these arguments.
            # noinspection PyArgumentList
            iterator = GraphDataBatchIterator(neighbor_data,
                                              graph_sampling_config, hidden_neighbor_data)
            iterator.data_fraction = data_fraction

        if self.manifold_nns is not None:
            sampling_config = get_config().sampling
            _, nns = \
                self.manifold_nns.knn_query_all(sampling_config.manifold_nn_k)

            all_manifold_neighbors = [nns[i][1:].astype(np.int64) for i in
                                      range(self.n_nodes())]
            iterator.refresh_manifold_nn(all_manifold_neighbors)

        return iterator

    @classmethod
    def make_train_eval_split(cls, name, edges, object_ids, weights):
        """
        Returns a tuple of a train eval split of the graph as defined in the
        data config.
        """

        data_config = get_config().data
        np.random.seed(data_config.split_seed)
        if data_config.split_by_edges:

            shuffle_order = np.arange(edges.shape[0])
            np.random.shuffle(shuffle_order)
            num_eval = floor(edges.shape[0] * data_config.split_size)
            eval_indices = shuffle_order[:num_eval]
            train_indices = shuffle_order[num_eval:]
            train_edges = edges[train_indices]
            train_weights = weights[train_indices]
            eval_edges = edges[eval_indices]
            eval_weights = weights[eval_indices]
        else:
            shuffle_order = np.arange(len(object_ids))
            np.random.shuffle(shuffle_order)
            num_eval = floor(len(object_ids) * data_config.split_size)
            eval_indices = shuffle_order[:num_eval]
            train_indices = shuffle_order[num_eval:]

            train_edges = []
            eval_edges = []
            train_weights = []
            eval_weights = []
            for edge, weight in zip(edges, weights):
                if edge[0] in eval_indices or edge[1] in eval_indices:
                    eval_edges.append(edge)
                    eval_weights.append(weight)
                else:
                    train_edges.append(edge)
                    train_weights.append(weight)

            train_edges = np.array(train_edges)
            eval_edges = np.array(eval_edges)
            train_weights = np.array(train_weights)
            eval_weights = np.array(eval_weights)


        train_data = GraphDataset(f"{name}_train_{data_config.split_seed}",
                                  train_edges, object_ids, train_weights)

        eval_data = GraphDataset(f"{name}_eval_{data_config.split_seed}",
                                 eval_edges, object_ids, eval_weights,
                                 hidden_graph=train_data)

        return train_data, eval_data
