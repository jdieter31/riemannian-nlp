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
            tensor_shapes = []
            for i in range(len(params["submanifolds"])):
                submanifold = params["submanifolds"][i]
                name = submanifold["name"] 
                tensor_shapes.append(submanifold["tensor_shape"])
                if "params" in submanifold:
                    submanifolds.append(RiemannianManifold.from_name_params(name, submanifold["params"]))
                else:
                    submanifolds.append(RiemannianManifold.from_name_params(name, None))
            return ProductManifold(submanifolds, np.array(tensor_shapes))

        raise Exception("Improper ProductManifold param layout")

    def __init__(self, submanifolds: List[RiemannianManifold], tensor_shapes: np.array):
        """
        Args:
            submanifolds (List[RiemannianManifold]):  List of submanifolds
            tensor_shapes (np.array): List of required shapes for each submanifold (length of first dimension should be the same length as submanifolds)
        """
        RiemannianManifold.__init__(self)
        self.submanifolds = submanifolds
        self.tensor_shapes = tensor_shapes
        # Get slices for each submanifold for easy access
        slice_start = 0
        # List of slices for each submanifold
        self.slices = []
        for i in range(len(self.submanifolds)):
            shape = self.tensor_shapes[i]
            dimension = int(np.sum(shape))
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
        new_shape = x.size()
        new_shape = new_shape[0:-1]
        new_shape = list(new_shape) + list(self.tensor_shapes[submanifold_index])

        return x.narrow(-1, sub_slice[0], sub_slice[1]).view(new_shape)

    def initialize_from_submanifold_values(self, submanifold_values: List[torch.Tensor]):
        flattened_tensors = []
        new_shape = []
        for i in range(len(self.submanifolds)):
            submanifold_value = submanifold_values[i]
            submanifold = self.submanifolds[i]
            submanifold_shape = self.tensor_shapes[i]
            new_shape = list(submanifold_value.size())[:-len(submanifold_shape)] + [-1]
            flattened_tensors.append(submanifold_value.reshape(new_shape))

        return torch.cat(flattened_tensors, dim=-1)

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
            sub_dists.append(manifold.dist(sub_x, sub_y, keepdim=False).unsqueeze(-1))
        stack = torch.cat(sub_dists, dim=-1)
        return torch.norm(stack, p=None, dim=-1, keepdim=keepdim)

    def exp(self, x, u):
        submanifold_values = []
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            sub_u = self.get_submanifold_value_index(u, i)
            submanifold_values.append(self.submanifolds[i].exp(sub_x, sub_u))

        return self.initialize_from_submanifold_values(submanifold_values)

    def log(self, x, y):
        submanifold_values = []
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            sub_y = self.get_submanifold_value_index(y, i)
            submanifold_values.append(self.submanifolds[i].log(sub_x, sub_y))

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
            submanifold_values.append(self.submanifolds[i].rgrad(sub_x, sub_dx))

        return self.initialize_from_submanifold_values(submanifold_values)

    def rgrad_(self, x, dx):
        for i in range(len(self.submanifolds)):
            sub_x = self.get_submanifold_value_index(x, i)
            sub_dx = self.get_submanifold_value_index(dx, i)
            self.submanifolds[i].rgrad_(sub_x, sub_dx)
        return dx

RiemannianManifold.register_manifold(ProductManifold)
