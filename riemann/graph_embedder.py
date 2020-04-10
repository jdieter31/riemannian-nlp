from abc import ABC, abstractmethod
from torch.nn import Embedding
import torch
from .manifold_initialization import initialize_manifold_tensor
from .manifolds import RiemannianManifold
from .manifold_tensors import ManifoldParameter
from .config.manifold_config import ManifoldConfig

class GraphEmbedder(ABC):
    """
    Abstract class for any type of model that produces embeddings of 
    a graph
    """
    @abstractmethod
    def embed_nodes(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Produces embedding of graph based on input nodes

        Args:
            node_ids (long tensor): input node ids

        Returns:
            embedding (tensor): embedding of nodes
        """
        raise NotImplementedError

    def get_manifold(self) -> RiemannianManifold:
        """
        Returns the manifold that this GraphEmbedder embeds nodes into.
        Defaults to Euclidean if this method is not overwritten
        """

        return ManifoldConfig().get_manifold_instance()

    def get_losses(self) -> List[Callable[[DataBatch], torch.Tensor]]:
        """
        Gets additional losses that should be trained. This is where isometry
        losses are parameter regularization losses should go
        """

        return []

class ManifoldEmbedding(Embedding, GraphEmbedder):

    def __init__(
            self,
            manifold: RiemannianManifold,
            num_embeddings,
            embedding_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
            manifold_initialization=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)

        self.manifold = manifold
        self.params = [manifold, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse]
        self.weight = ManifoldParameter(self.weight.data, manifold=manifold)
        if manifold_initialization is not None:
            initialize_manifold_tensor(self.weight.data, self.manifold,
                                       manifold_initialization)
    
    def get_embedding_matrix(self):
        return self.weight.data

    def get_manifold(self): -> RiemannianManifold:
        return self.manifold

    def embed_nodes(self, node_ids):
        return self(node_ids)

