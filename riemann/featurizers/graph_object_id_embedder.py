from abc import abstractmethod

import numpy as np
import torch

from ..data.graph_dataset import GraphDataset
from ..graph_embedder import GraphEmbedder


class GraphObjectIDEmbedder(GraphEmbedder):
    """
    Abstract class for a GraphEmbedder that makes embeddings based on object
    IDs from the graph dataset rather than just raw node data.
    """

    @abstractmethod
    def __init__(self, graph_dataset: GraphDataset):
        super().__init__()
        self.graph_dataset = graph_dataset

    @abstractmethod
    def embed_graph_data(self, node_ids: torch.Tensor, object_ids:
    np.ndarray) \
            -> torch.Tensor:
        """
        Embeds graph data based on nodes and object ids

        Params:
            node_ids (torch.Tensor): graph ids of nodes
            object_ids (numpy.ndarray): numpy array of str datatype containing
                the associated object ids to the graph nodes
        """

        raise NotImplementedError

    def embed_nodes(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Find all unique nodes and embed them using their associated object id
        from the graph dataset
        """
        unique_nodes, inverse_map = torch.unique(node_ids, return_inverse=True)

        graph_obj_id_list = self.graph_dataset.object_ids

        object_ids = graph_obj_id_list[unique_nodes.detach().numpy()]

        return self.embed_graph_data(unique_nodes, object_ids)[inverse_map]
