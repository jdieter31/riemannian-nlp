from .batching import DataBatch
import torch
from typing import Dict

class GraphDataBatch(DataBatch):
    """
    Class to contain a batch of graph neighbors
    """

    def __init__(self, vertices: torch.Tensor, neighbors: torch.Tensor,
                 train_distances: torch.Tensor,
                 additional_data: Dict=None):
        """
        Params:
            vertices (torch.Tensor long): indices of vertices should be of shape
                [batch_size, embedding_dim]
            neighbors (torch.Tensor long): indices of neighbors to vertices
                should be of shape [batch_size, num_neighbors, embedding_dim]
            train_distance (torch.Tensor float): Distances from main vertices
                to neighbors in whichever metric is being used as a "true"
                distance for training (i.e. graph, graph^2, etc) should be same
                size as neighbors
        """
        self.vertices = vertices
        self.neighbors = neighbors
        self.train_distances = train_distances

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Returns dictionary with entries: "vertices" "neighbors"
        "train_distances" which are used as explained in the constructor.
        """
        return {
            "vertices": self.vertices,
            "neighbors": self.neighbors,
            "train_distances": self.train_distances
        }

    @classmethod
    def get_data_type(cls) -> str:
        """
        This object is meant to store a batch of embedded graph data
        """
        return "embedded_graph"
