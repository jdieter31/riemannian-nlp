from .manifold import RiemannianManifold
import torch
from typing import List
import numpy as np

class ProductManifold(RiemannianManifold):
    """Implemetation of a product manifold"""

    @classmethod
    def from_params(cls, params):
        if "submanifolds" in params:
            submanifolds = []
            submanifold_dims = []
            for i in range(len(params["submanifolds"])):
                submanifold = params["submanifolds"][i]
                name = submanifold["name"] 
                submanifold_dims.append(submanifold["dimension"])
                if "params" in submanifold:
                    submanifolds.append(RiemannianManifold.from_name_params(name, submanifold["params"]))
                else:
                    submanifolds.append(RiemannianManifold.from_name_params(name, None))

            curvature_scale = None
            if "curvature_scale" in params:
                curvature_scale = params["curvature_scale"]
                if len(curvature_scale) != len(submanifolds):
                   raise Exception("Curvature scale must be same dimension as parameters") 

            return ProductManifold(submanifolds, submanifold_dims, curvature_scale)

        raise Exception("Improper ProductManifold param layout")

    def __init__(self, submanifolds: List[RiemannianManifold], submanifold_dims: List[int], curvature_scale=None):
        """
        Args:
            submanifolds (List[RiemannianManifold]):  List of submanifolds
            submanifold_dims (List[int]): Dimensions of submanifolds
        """
        RiemannianManifold.__init__(self)
        self.submanifolds = submanifolds
        self.submanifold_dims = submanifold_dims
        self.curvature_scale = curvature_scale
        # Get slices for each submanifold for easy access
        slice_start = 0
        # List of slices for each submanifold
        self.slices = []
        for i in range(len(self.submanifolds)):
            dimension = self.submanifold_dims[i]
            slice_end = slice_start + dimension
            self.slices.append((slice_start, dimension))
            slice_start = slice_end

    def get_submanifold_value(self, x: torch.Tensor, submanifold: RiemannianManifold):
        """
        Gets the value of x in the repsective submanifold
        """
        return self.get_submanifold_value_index(x, self.submanifolds.index(submanifold))

    def get_submanifold_value_index(self, x: torch.Tensor, submanifold_index: int):
        """
        Gets the value of x in the respective submanifold given by the index
        """
        sub_slice = self.slices[submanifold_index]
        return x.narrow(-1, sub_slice[0], sub_slice[1])

    def initialize_from_submanifold_values(self, submanifold_values: List[torch.Tensor]):
        return torch.cat([submanifold_values[i] for i in range(len(self.submanifolds))], dim=-1)

    def retr_(self, x, u, indices=None):
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            sub_u = self.get_submanifold_value_index(u, i)
            self.submanifolds[i].retr_(sub_x, sub_u, indices)
        return x

    def retr(self, x, u, indices=None):
        submanifold_values = []
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            sub_u = self.get_submanifold_value_index(u, i)
            submanifold_values.append(self.submanifolds[i].retr(sub_x, sub_u, indices))

        return self.initialize_from_submanifold_values(submanifold_values)

    def dist(self, x, y, keepdim=False):
        sub_dists = []
        for i in range(len(self.submanifolds)):
            manifold = self.submanifolds[i]
            sub_x = self.get_submanifold_value_index(x, i)
            sub_y = self.get_submanifold_value_index(y, i)
            dist = manifold.dist(sub_x, sub_y, keepdim=False).unsqueeze(-1)
            if self.curvature_scale is not None:
                dist = dist / torch.exp(self.curvature_scale[i].to(dist.device) / 2)
            sub_dists.append(dist)
        stack = torch.cat(sub_dists, dim=-1)
        return torch.norm(stack, p=None, dim=-1, keepdim=keepdim)

    def exp(self, x, u):
        submanifold_values = []
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            sub_u = self.get_submanifold_value_index(u, i)
            exp_x_u = self.submanifolds[i].exp(sub_x, sub_u)
            submanifold_values.append(exp_x_u)

        return self.initialize_from_submanifold_values(submanifold_values)

    def log(self, x, y):
        submanifold_values = []
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            sub_y = self.get_submanifold_value_index(y, i)
            log_x_y = self.submanifolds[i].log(sub_x, sub_y)
            submanifold_values.append(log_x_y)

        return self.initialize_from_submanifold_values(submanifold_values)

    def proj(self, x, indices=None):
        submanifold_values = []
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            submanifold_values.append(self.submanifolds[i].proj(sub_x, indices))

        return self.initialize_from_submanifold_values(submanifold_values)

    def proj_(self, x, indices=None):
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            self.submanifolds[i].proj_(sub_x, indices)
        return x

    def rgrad(self, x, dx):
        submanifold_values = []
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            sub_dx = self.get_submanifold_value_index(dx, i)
            if self.curvature_scale is not None:
                sub_dx = sub_dx * torch.exp(self.curvature_scale[i].to(sub_dx.device))
            submanifold_values.append(self.submanifolds[i].rgrad(sub_x, sub_dx))

        return self.initialize_from_submanifold_values(submanifold_values)


    def lower_indices(self, x, dx):
        submanifold_values = []
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            sub_dx = self.get_submanifold_value_index(dx, i)
            if self.curvature_scale is not None:
                sub_dx = sub_dx / torch.exp(self.curvature_scale[i].to(sub_dx.device))
            submanifold_values.append(self.submanifolds[i].lower_indices(sub_x, sub_dx))

        return self.initialize_from_submanifold_values(submanifold_values)

    def rgrad_(self, x, dx):
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            sub_dx = self.get_submanifold_value_index(dx, i)
            if self.curvature_scale is not None:
                sub_dx *= torch.exp(self.curvature_scale[i].to(sub_dx.device))
            self.submanifolds[i].rgrad_(sub_x, sub_dx)
        return dx
    
    def tangent_proj_matrix(self, x):
        tangent_proj_matrix = torch.zeros(*x.size(), x.size()[-1], dtype=x.dtype, device=x.device)
        for i in range(len(self.submanifolds)):
            sub_slice = self.slices[i]
            sub_matrix = tangent_proj_matrix.narrow(-1, sub_slice[0], sub_slice[1]).narrow(-2, sub_slice[0], sub_slice[1])
            sub_matrix.copy_(self.submanifolds[i].tangent_proj_matrix(self.get_submanifold_value_index(x, i)))
        return tangent_proj_matrix

    def get_metric_tensor(self, x):
        metric_tensor = torch.zeros(*x.size(), x.size()[-1], dtype=x.dtype, device=x.device)
        for i in range(len(self.submanifolds)):
            sub_slice = self.slices[i]
            sub_matrix = metric_tensor.narrow(-1, sub_slice[0], sub_slice[1]).narrow(-2, sub_slice[0], sub_slice[1])
            sub_metric = self.submanifolds[i].get_metric_tensor(self.get_submanifold_value_index(x, i))
            if self.curvature_scale is not None:
                sub_metric = sub_metric / torch.exp(self.curvature_scale[i].to(sub_metric.device))
            sub_matrix.copy_(sub_metric)
        return metric_tensor

RiemannianManifold.register_manifold(ProductManifold)
