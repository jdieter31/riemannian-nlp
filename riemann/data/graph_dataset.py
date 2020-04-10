from typing import Iterator, Dict
from .graph_data_batch import GraphDataBatch
from .graph_data_batch_iterator import GraphDataBatchIterator
from graph_tool.all import Graph
import numpy as np
from tqdm import tqdm
from .pickle_manager import load_or_gen
from ..config.graph_sampling_config import GraphSamplingConfig
from ..config.config_loader import get_config
from math import floor

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

        graph_neighbors = np.concatenate(all_graph_neighbors)
        graph_neighbor_weights = np.concatenate(all_graph_weights)
        non_empty_vertices = np.array(non_empty_vertices, dtype=np.int64)

        return {
            "all_graph_neighbors": all_graph_neighbors,
            "all_graph_weights": all_graph_weights,
            "non_empty_vertices": non_empty_vertices,
            "N": self.n_nodes()
        }

    def n_nodes(self) -> int:
        """
        Returns the number of nodes in the graph
        """
        return len(self.object_ids)

    def get_neighbor_iterator(self, 
                              graph_sampling_config: GraphSamplingConfig) -> Iterator[GraphDataBatch]:
        """
        Gets an efficient iterator of edge batches

        """
        neighbor_data = load_or_gen(f"GraphDataset.{self.name}",
                                    self.gen_neighbor_data)
        if self.hidden_graph is None:
            return GraphDataBatchIterator(neighbor_data, graph_sampling_config)
        else:
            hidden_neighbor_data = load_or_gen(
                f"GraphDataset.{self.hidden_graph.name}", 
                self.hidden_graph.gen_neighbor_data)

            return GraphDataBatchIterator(neighbor_data,
                graph_sampling_config, hidden_neighbor_data)

    @classmethod
    def make_train_eval_split(cls, name, edges, object_ids, weights):
        """
        Returns a tuple of a train eval split of the graph as defined in the
        data config.
        """

        data_config = get_config().data
        np.random.seed(data_config.split_seed)
        shuffle_order = np.arange(edges.shape[0])
        np.random.shuffle(shuffle_order)
        num_eval = floor(edges.shape[0] * data_config.split_size)
        eval_indices = shuffle_order[:num_eval]
        train_indices = shuffle_order[num_eval:]
        train_edges = edges[train_indices]
        train_weights = weights[train_indices]
        eval_edges = edges[eval_indices]
        eval_weights = weights[eval_indices]

        train_data = GraphDataset(f"{name}_train_{data_config.split_seed}",
                                  train_edges, object_ids, train_weights)

        eval_data = GraphDataset(f"{name}_eval_{data_config.split_seed}", 
                                 eval_edges, object_ids, eval_weights,
                                 hidden_graph=train_data)

        return train_data, eval_data
        




