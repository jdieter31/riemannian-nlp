import torch
import torch.nn as nn
from manifolds import RiemannianManifold
from typing import List

class ManifoldNetwork(nn.Module):
    def __init__(
            self,
            manifold_seq: List[RiemannianManifold],
            dimension_seq: List[int],
            log_base_inits: List[torch.Tensor],
            exp_base_inits: List[torch.Tensor]
            ):
        super(ManifoldNetwork, self).__init__()

            
