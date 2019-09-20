import torch
import torch.nn as nn
from manifolds import RiemannianManifold
from .manifold_layer import ManifoldLayer
from typing import List

class ManifoldNetwork(nn.Module):
    def __init__(
            self,
            manifold_seq: List[RiemannianManifold],
            dimension_seq: List[int],
            non_linearity,
            num_poles,
            log_base_inits: List[torch.Tensor],
            exp_base_inits: List[torch.Tensor],
            ):
        super(ManifoldNetwork, self).__init__()
        self.manifold_seq = manifold_seq
        self.dimension_seq = dimension_seq
        self.non_linearity = non_linearity
        self.num_poles = num_poles
        layer_list = []
        for i in range(len(manifold_seq) - 1):
            if i == len(manifold_seq) - 2:
                layer_list.append(ManifoldLayer(manifold_seq[i], manifold_seq[i+1], dimension_seq[i],
                    dimension_seq[i+1], None, num_poles, log_base_inits[i], exp_base_inits[i]))
            else:
                layer_list.append(ManifoldLayer(manifold_seq[i], manifold_seq[i+1], dimension_seq[i],
                    dimension_seq[i+1], non_linearity, num_poles, log_base_inits[i], exp_base_inits[i]))
        self.layers = nn.ModuleList(layer_list)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
    
    def get_save_data(self):
        return {
            'params': [self.manifold_seq, self.dimension_seq, self.non_linearity, self.num_poles],
            'state_dict': self.state_dict()
        }

    @classmethod
    def from_save_data(cls, data):
        params = data['params']
        params += [[None for _ in range(len(params[0]) - 1)], [None for _ in range(len(params[0]) - 1)]]
        instance = ManifoldNetwork(*params)
        instance.load_state_dict(data['state_dict'])
        return instance


            
