from abc import ABC, abstractmethod

class GraphEmbedder(ABC):
    @abstractmethod
    def embed_nodes(self, node_ids):
        """
        Produces embedding of graph based on input nodes

        Args:
            node_ids (long tensor): input node ids

        Returns:
            embedding (tensor): embedding of nodes
        """
        raise NotImplementedError

