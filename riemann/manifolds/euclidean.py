from .manifold import RiemannianManifold
import torch

class EuclideanManifold(RiemannianManifold):
    '''
    Implementation of Euclidean Riemannian manifold with standard metric
    '''

    @classmethod
    def from_params(cls, params):
        return EuclideanManifold()

    def retr(self, x, u, indices=None):
        if indices is not None:
            return x.index_add(0, indices, u)
        return x + u

    def retr_(self, x, u, indices=None):
        if indices is not None:
            return x.index_add_(0, indices, u)
        return x.add_(u)
    
    def exp(self, x, u):
        return x + u

    def log(self, x, y):
        return y - x

    def dist(self, x, y, keepdim=False):
        dist = torch.norm(x - y, p=None, dim=-1, keepdim=keepdim)
        return dist

    def proj(self, x, indices=None):
        return x

    def proj_(self, x, indices=None):
        return x

    def rgrad(self, x, dx):
        return dx

    def rgrad_(self, x, dx):
        return dx

    def lower_indices(self, x, dx):
        return dx

    def tangent_proj_matrix(self, x):
        tangent_matrix = torch.eye(x.size()[-1], dtype=x.dtype, device=x.device)
        for i in range(len(x.size()) - 1):
            tangent_matrix.unsqueeze_(0)
        return tangent_matrix.expand(*x.size(), x.size()[-1])

    def get_metric_tensor(self, x):
        metric = torch.eye(x.size()[-1], dtype=x.dtype, device=x.device)
        for i in range(len(x.size()) - 1):
            metric.unsqueeze_(0)
        return metric.expand(*x.size(), x.size()[-1])

RiemannianManifold.register_manifold(EuclideanManifold)
