import torch.nn as nn
import torch
from .manifolds import EuclideanManifold

class ManifoldParameter(nn.Parameter):
    """PyTorch tensor with information about the manifold it is contained on."""

    def __new__(cls, data=None, requires_grad=True, manifold=EuclideanManifold(), lr_scale=1):
        if data is None:
            data = torch.Tensor()
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.manifold = manifold
        instance.lr_scale = lr_scale
        return instance
