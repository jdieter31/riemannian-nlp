import torch.nn as nn
import torch
from ..manifolds import RiemannianManifold
from ..manifold_tensors import ManifoldParameter
from ..embed_save import Savable
from torch.nn.init import orthogonal_
import math

class ManifoldLayer(nn.Module):
    def __init__(
            self,
            in_manifold: RiemannianManifold,
            out_manifold: RiemannianManifold,
            in_dimension: int,
            out_dimension: int,
            non_linearity=None,
            num_poles=3,
            log_base_init: torch.Tensor=None,
            exp_base_init: torch.Tensor=None,
            ortho_init=False,
            ):
        super(ManifoldLayer, self).__init__()
        self.in_manifold = in_manifold
        self.out_manifold = out_manifold
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.num_poles = num_poles
        if log_base_init is not None:
            self.log_base = ManifoldParameter(log_base_init, manifold=in_manifold, lr_scale=1)
        else:
            self.log_base = ManifoldParameter(torch.Tensor(num_poles, in_dimension), manifold=in_manifold, lr_scale=1)
        if exp_base_init is not None:
            self.exp_base = ManifoldParameter(exp_base_init, manifold=out_manifold, lr_scale=1)
        else:
            self.exp_base = ManifoldParameter(torch.Tensor(out_dimension), manifold=out_manifold, lr_scale=1)

        self.linear_layer = nn.Linear(in_dimension * num_poles, out_dimension, bias=False)
        if ortho_init:
            orthogonal_(self.linear_layer.weight)
        self.non_linearity = None
        self.non_linearity_name = non_linearity
        if non_linearity is not None:
            if non_linearity == "relu":
                self.non_linearity = nn.ReLU()
            elif non_linearity == "tanh":
                self.non_linearity = nn.Tanh()
            elif non_linearity == "tanhshrink":
                self.non_linearity = nn.Tanhshrink()
            elif non_linearity == "leakyrelu":
                self.non_linearity = nn.LeakyReLU()
            elif non_linearity == "elu":
                self.non_linearity = nn.ELU()

    def forward(self, x):
        x_expanded = x.unsqueeze(-2)
        log_x = self.in_manifold.log(self.log_base, x_expanded)
        # Scale by metric tensor (equivalent of lowering indices to get geodesic normal coordinates)
        # log_x = self.in_manifold.lower_indices(self.log_base, log_x)
        log_x_flattened = log_x.view(*x_expanded.size()[:-2], -1)
        linear_out = self.linear_layer(log_x_flattened)
        if self.non_linearity is not None:
            linear_out = self.non_linearity(linear_out)

        # Scale by metric tensor (raise indices)
        # linear_out = self.out_manifold.rgrad(self.exp_base, linear_out)
        exp_out = self.out_manifold.exp(self.exp_base, linear_out)
        return exp_out

    def get_save_data(self):
        return {
            'params': [self.in_manifold, self.out_manifold, self.in_dimension, self.out_dimension, self.non_linearity_name, self.num_poles],
            'state_dict': self.state_dict()
        }

    @classmethod
    def from_save_data(cls, data):
        params = data['params']
        state_dict = data['state_dict']
        instance = ManifoldLayer(*params)
        instance.load_state_dict(state_dict)
        return instance
