import torch.nn as nn
import torch
from manifolds import RiemannianManifold
from manifold_tensors import ManifoldParameter
from embed_save import Savable
from torch.nn.init import orthogonal_

class ManifoldLayer(nn.Module):
    def __init__(
            self,
            in_manifold: RiemannianManifold,
            out_manifold: RiemannianManifold,
            in_dimension: int,
            out_dimension: int,
            log_base_init: torch.Tensor=None,
            exp_base_init: torch.Tensor=None,
            ortho_init=True,
            non_linear=False
            ):
        super(ManifoldLayer, self).__init__()
        self.in_manifold = in_manifold
        self.out_manifold = out_manifold
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        if log_base_init is not None:
            self.log_base = ManifoldParameter(log_base_init, manifold=in_manifold)
        else:
            self.log_base = ManifoldParameter(torch.Tensor(in_dimension), manifold=in_manifold)
        if exp_base_init is not None:
            self.exp_base = ManifoldParameter(exp_base_init, manifold=out_manifold)
        else:
            self.exp_base = ManifoldParameter(torch.Tensor(out_dimension), manifold=out_manifold)

        self.linear_layer = nn.Linear(in_dimension, out_dimension, bias=False)
        if ortho_init:
            orthogonal_(self.linear_layer.weight)
            with torch.no_grad():
                self.linear_layer.weight /= torch.sqrt(self.linear_layer.weight.new_tensor(in_dimension))
        self.relu = None
        if non_linear:
            self.relu = nn.ReLU()

    def forward(self, x):
        log_x = self.in_manifold.log(self.log_base, x)
        linear_out = self.linear_layer(log_x)
        if self.relu is not None:
            linear_out = self.relu(linear_out)
        exp_out = self.out_manifold.exp(self.exp_base, linear_out)
        return exp_out

    def get_save_data(self):
        return {
            'params': [self.in_manifold, self.out_manifold, self.in_dimension, self.out_dimension],
            'state_dict': self.state_dict()
        }

    @classmethod
    def from_save_data(cls, data):
        params = data['params']
        state_dict = data['state_dict']
        instance = ManifoldLayer(*params)
        instance.load_state_dict(state_dict)
        return instance
