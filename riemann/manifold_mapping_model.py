from typing import Callable, Optional

import torch
from torch import nn

from .featurizers.graph_text_featurizer import GraphObjectIDFeaturizer
from .featurizers.text_featurizer import TextFeaturizer
from .manifolds import RiemannianManifold


class ManifoldMappingModel(GraphObjectIDFeaturizer, TextFeaturizer):

    def __init__(self, graph_featurizer: GraphObjectIDFeaturizer,
                 manifold_mapping: nn.Module,
                 out_manifold: RiemannianManifold,
                 text_featurizer: TextFeaturizer = None,
                 text_to_node_func: Callable[[str], Optional[int]] = None):
        """
        Params:
            graph_featurizer (GraphObjectIDFeaturizer): Produces initial
                embeddings for graph nodes
            manifold_mapping (nn.Module): torch module that represents the
                mapping between manifolds
            out_manifold (RiemannianManifold): output manifold of
                manifold_mapping
            text_featurizer (TextFeaturizer): Produces initial embeddings for
                text
            text_to_node_func (Callable[[str], Optional[int]]): A function like
                this can be specified to determine if text data should be
                reinterpretted as node data (i.e. for retrofitting)
        """
        self.graph_featurizer = graph_featurizer
        self.text_featurizer = text_featurizer
        self.out_manifold = out_manifold
        self.manifold_mapping = manifold_mapping
        self.text_to_node_func = text_to_node_func

    def embed_graph_data(self, node_ids: torch.Tensor, object_ids: numpy.ndarray):
        input_embeddings = self.graph_featurizer.embed_graph_data(node_ids,
                                                                  object_ids)
        return self.manifold_mapping(input_embeddings)

    def embed_text(self, data: List[str]) -> List[Optional[torch.Tensor]]:
        # TODO
        pass

    def get_manifold(self) -> RiemannianManifold:
        return self.out_manifold
