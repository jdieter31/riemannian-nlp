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

RiemannianManifold.register_manifold(EuclideanManifold)
