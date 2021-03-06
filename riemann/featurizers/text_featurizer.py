from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from ..manifolds import RiemannianManifold


class TextFeaturizer(ABC):
    """
    Abstract class for any type of model that produces embeddings of 
    text
    """

    @abstractmethod
    def embed_text(self, data: List[str]) -> List[Optional[torch.Tensor]]:
        """
        Produces embedding of text

        Args:
            data (List[str]): List of text objects

        Returns:
            embedding (List[Optional[tensor]]): embedding of text objects (if
                they can be embedded as represented by the optional)
        """
        raise NotImplementedError

    def get_manifold(self) -> RiemannianManifold:
        """
        Returns the manifold that this GraphEmbedder embeds nodes into.
        Defaults to Euclidean if this method is not overwritten
        """

        return ManifoldConfig().get_manifold_instance()
