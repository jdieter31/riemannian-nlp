from abc import ABC, abstractmethod
from ..data.graph_dataset import GraphDataset
from .text_featurizer import TextFeaturizer
from ..graph_embedder import GraphEmbedder 
from typing import List
import torch

class GraphObjectIDFeaturizer(ABC, GraphEmbedder):
    """
    Abstract class for a GraphEmbedder that makes embeddings based on object
    IDs from the graph dataset rather than just raw node data
    """

    @abstract_method
    def __init__(self, graph_dataset: GraphDataset):
        self.graph_dataset = graph_dataset

    @abstract_method
    def embed_graph_data(self, node_ids: torch.Tensor, object_ids:
                         numpy.ndarray) \
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



