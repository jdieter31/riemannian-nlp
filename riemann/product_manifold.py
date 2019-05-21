from typing import List

import torch

from geoopt.manifolds.base import Manifold

import numpy as np

class ProductManifold(Manifold):
    ndim = 1
    name = "ProductManifold"

    def __init__(self, submanifolds: List[Manifold], tensor_shapes: np.array):
        super(ProductManifold, self).__init__()
        self.submanifolds = submanifolds
        self.tensor_shapes_dict = {man : shape for man, shape in zip(self.submanifolds, tensor_shapes)}
        # Get slices for each submanifold for easy access

        slice_start = 0
        # Dictionary showing where submanifold values are stored in the tensor
        # Manifold -> (slice start index, dimension of manifold)
        self.slices = {}
        for manifold in self.submanifolds:
            shape = self.tensor_shapes_dict[manifold]
            dimension = int(np.sum(shape))
            slice_end = slice_start + dimension

            self.slices[manifold] = (slice_start, dimension)
            slice_start = slice_end

    # Extracts the value of x in a submanifold and is formed
    # into the correct shape for the submanifold
    def _get_submanifold_value(self, x: torch.Tensor, submanifold: Manifold):
        slice = self.slices[submanifold]
        new_shape = x.size()
        new_shape = new_shape[0:-1]
        new_shape = list(new_shape) + list(self.tensor_shapes_dict[submanifold])

        return x.narrow(-1, slice[0], slice[1]).reshape(new_shape)

    def _check_shape(self, x: torch.Tensor, name):
        dimension = sum([self.slices[manifold][1] for manifold in self.submanifolds])
        dim_is_ok = x.shape[-1] == dimension
        if not dim_is_ok:
            return False, "Not enough dimensions for `{}`".format(name)
        return True, None

    def _check_point_on_manifold(self, x: torch.Tensor, atol=1e-5, rtol=1e-5):
        for submanifold, man_slice in zip(self.submanifolds, self.slices):
            sub_result, reason = submanifold._check_point_on_manifold(self._get_submanifold_value(x, submanifold), atol, rtol)
            if not sub_result:
                return sub_result, "On submanifold " + submanifold.name + ":"
        return True, None

    def _check_vector_on_tangent(self, x: torch.Tensor, u, atol=1e-5, rtol=1e-5):
        for submanifold, man_slice in zip(self.submanifolds, self.slices):
            sub_result, reason = submanifold._check_vector_on_tangent(self._get_submanifold_value(x, submanifold), atol, rtol)
            if not sub_result:
                return sub_result, "On submanifold " + submanifold.name + ":"
        return True, None

    # Formats a lit of tensors of submanifold values into one tensor
    # in the correct format for the product manifold
    def _load_from_submanifold_values(self, submanifold_values: List[torch.tensor]):
        flattened_tensors = []
        new_shape = []
        for manifold, value in zip(self.submanifolds, submanifold_values):
            shape = self.tensor_shapes_dict[manifold]
            new_shape = list(value.size())[:-len(shape)] + [-1]
            flattened_tensors.append(value.reshape(new_shape))

        return torch.cat(flattened_tensors, dim=-1)

    def _projx(self, x: torch.Tensor):
        proj_vects = []
        for manifold in self.submanifolds:
            proj_vects.append(manifold.projx(self._get_submanifold_value(x, manifold)))
        return self._load_from_submanifold_values(proj_vects)

    def _proju(self, x: torch.Tensor, u: torch.Tensor):
        proj_vects = []
        for manifold in self.submanifolds:
            proj_vects.append(manifold.proju(self._get_submanifold_value(x, manifold),
                                             self._get_submanifold_value(u, manifold)))
        return self._load_from_submanifold_values(proj_vects)

    def _egrad2rgrad(self, x, u):
        grad_vects = []
        for manifold in self.submanifolds:
            grad_vects.append(manifold.egrad2rgrad(self._get_submanifold_value(x, manifold),
                                                   self._get_submanifold_value(u, manifold)))
        return self._load_from_submanifold_values(grad_vects)

    def _inner(self, x, u, v, keepdim):
        inner_vects = []
        for manifold in self.submanifolds:
            # The following is essentially the inner product with keepdim
            # False but it will add an extra dimension to do the last sum over
            inner_vects.append(manifold.inner(self._get_submanifold_value(x, manifold),
                                              self._get_submanifold_value(u, manifold),
                                              self._get_submanifold_value(v, manifold), keepdim=False).unsqueeze(-1))
        return torch.cat(inner_vects, dim=-1).sum(dim=-1, keepdim=keepdim)

    def _retr(self, x, u, t):
        retr_vects = []
        for manifold in self.submanifolds:
            retr_vects.append(manifold.retr(self._get_submanifold_value(x, manifold),
                                             self._get_submanifold_value(u, manifold), t))
        return self._load_from_submanifold_values(retr_vects)

    def _transp_follow_sub(self, x, v, u, t):
        transp_vects = []
        for manifold in self.submanifolds:
            transp_vects.append(manifold._transp_follow(self._get_submanifold_value(x, manifold),
                                             self._get_submanifold_value(v, manifold),
                                             u=self._get_submanifold_value(u, manifold), t=t))
        return self._load_from_submanifold_values(transp_vects)

    def _transp_follow(self, x, v, *more, u, t):
        if more:
            return tuple(self._transp_follow_sub(x, _v, u, t) for _v in (v,) + more)
        else:
            return self._transp_follow_sub(x, v, u, t)

    def _dist(self, x, y, keepdim):
        inner_vects = []
        for manifold in self.submanifolds:
            # The following is essentially the inner product with keepdim
            # False but it will add an extra dimension to do the last sum over
            inner_vects.append(manifold.dist(self._get_submanifold_value(x, manifold),
                                              self._get_submanifold_value(y, manifold), keepdim=False).unsqueeze(-1))
        stack = torch.cat(inner_vects, dim=-1)
        return torch.norm(stack, p=None, dim=-1, keepdim=keepdim)
