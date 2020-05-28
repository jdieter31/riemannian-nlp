import io
import logging
from typing import Callable
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn

from .graph_object_id_embedder import GraphObjectIDEmbedder
from ..config.config_loader import get_config
from ..data.graph_data_batch import GraphDataBatch
from ..data.graph_dataset import GraphDataset
from ..device_manager import get_device
from ..losses.isometry_loss import isometry_loss
from ..manifold_initialization import initialize_manifold_tensor
from ..manifolds import RiemannianManifold
from ..optimizer_gen import register_parameter_group

logger = logging.getLogger(__name__)


class GraphObjectIDFeaturizerEmbedder(GraphObjectIDEmbedder):
    """
    A GraphEmbedder that embeds by first featurizing graph nodes and then
    appling a torch module.
    """

    def __init__(self, graph_dataset: GraphDataset,
                 featurizer: Callable[[np.ndarray, torch.Tensor], torch.Tensor],
                 model: nn.Module,
                 in_manifold: RiemannianManifold, in_dimension: int,
                 out_manifold: RiemannianManifold, out_dimension: int
                 ):
        """
        Params:
            graph_dataset  (GraphDataset): graph dataset this is embedding
            featurizer (Callable): function mapping from a numpy array of type
                str containing object ids to a torch tensor containing the
                featurizations of these objects
            model (nn.Module): torch model that maps from in_manifold to
                out_manifold
            in_manifold (RiemannianManifold): manifold containing
                featurizations of object ids
            out_manifold (RiemannianManifold): final manifold graph data will
                be embedded in 
            out_dimension (int): dimension of out_manifold
        """
        super(GraphObjectIDFeaturizerEmbedder, self).__init__(graph_dataset)
        self.featurizer = featurizer
        self.model = model
        self.in_manifold = in_manifold
        self.in_dimension = in_dimension
        self.out_manifold = out_manifold
        self.out_dimension = out_dimension

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

        in_values = self.featurizer(object_ids, node_ids)
        in_values = in_values.to(next(self.model.parameters()).device)

        out_values = self.model(in_values)
        return out_values

    def get_featurizer_graph_embedder(self) -> GraphObjectIDEmbedder:
        """
        Produces a embedder object that 
        """
        outer_class = self

        class FeaturizedGraphEmbedder(GraphObjectIDEmbedder):
            def __init__(self):
                super(FeaturizedGraphEmbedder,
                      self).__init__(outer_class.graph_dataset)

            def embed_graph_data(self, node_ids: torch.Tensor, object_ids: \
                    np.ndarray) -> torch.Tensor:
                in_values = outer_class.featurizer(object_ids, node_ids)
                in_values = in_values.to(next(outer_class.model.parameters()).device)
                return in_values

        return FeaturizedGraphEmbedder()

    def get_losses(self):
        loss_config = get_config().loss
        if loss_config.use_conformality_regularizer:
            def batch_isometry_loss(data_batch: GraphDataBatch):
                random_samples = loss_config.random_isometry_samples
                initialization = \
                    loss_config.random_isometry_initialization.get_initialization_dict()

                random_samples = torch.empty(random_samples,
                                             self.in_dimension,
                                             dtype=torch.float,
                                             device=get_device())
                initialize_manifold_tensor(random_samples, self.in_manifold,
                                           initialization)
                return isometry_loss(self.model, random_samples,
                                     self.in_manifold, self.out_manifold,
                                     self.out_dimension, loss_config.conformality
                                     )

            return [batch_isometry_loss]
        else:
            return []

    # region: serialization
    def _to_file(self, zf: ZipFile) -> None:
        super()._to_file(zf)

        with io.BytesIO() as buf:
            torch.save(self.model.state_dict(), buf)
            zf.writestr("model.state", buf.getvalue())

    def _from_file(self, zf: ZipFile) -> None:
        # Read Torch state from file
        super()._from_file(zf)
        with io.BytesIO(zf.read("model.state")) as buf:
            # make sure we start by placing things on cpu
            self.model.load_state_dict(torch.load(buf, map_location="cpu"))
    # endregion
