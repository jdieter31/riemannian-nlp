from ..data.graph_dataset import GraphDataset
from .text_featurizer import TextFeaturizer
from ..graph_embedder import GraphEmbedder 
from typing import List
from torch import nn
from ..manifolds import RiemannianManifold
from typing import Callable
from .graph_object_id_embedder import GraphObjectIDEmbedder
from ..losses.isometry_loss import isometry_loss
from ..data.graph_data_batch import GraphDataBatch
from ..config.config_loader import get_config
from ..device_manager import get_device
from ..manifold_initialization import initialize_manifold_tensor
from ..optimizer_gen import register_parameter_group
import torch
import numpy as np

class GraphObjectIDFeaturizerEmbedder(GraphObjectIDEmbedder):
    """
    A GraphEmbedder that embeds by first featurizing graph nodes and then
    appling a torch module.
    """

    def __init__(self, graph_dataset: GraphDataset, featurizer:
                 Callable[[np.ndarray, torch.Tensor], torch.Tensor], model: nn.Module,
                 in_manifold: RiemannianManifold, in_dimension: int,
                 out_manifold: RiemannianManifold, out_dimension: int,
                 isometry_loss: bool = True):
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
        self.isometry_loss = isometry_loss
        self.out_dimension = out_dimension
        """
        self.scale = torch.tensor(torch.FloatTensor([0]), requires_grad=True,
                                  device=next(self.model.parameters()).device)
        register_parameter_group([self.scale])
        """


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
        # in_values = in_values * torch.exp(self.scale)
        out_values = self.model(in_values)
        return out_values

    def get_losses(self):
        if self.isometry_loss:
            def batch_isometry_loss(data_batch: GraphDataBatch):
                loss_config = get_config().loss
                random_samples = loss_config.random_isometry_samples
                initialization = \
                    loss_config.random_isometry_initialization.get_initialization_dict()
                isometric = not loss_config.conformal

                random_samples = torch.empty(random_samples,
                                             self.in_dimension,
                                             dtype=torch.float,
                                             device=get_device())
                initialize_manifold_tensor(random_samples, self.in_manifold,
                                           initialization)
                return isometry_loss(self.model, random_samples,
                                     self.in_manifold, self.out_manifold,
                                     self.out_dimension, isometric)
                
            return [batch_isometry_loss]
        else:
            return []
