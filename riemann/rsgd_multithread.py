# Based on FBs code for RiemannianSGD since that works with Hogwild training and adapted to work
# with the implementation of Riemannian manifolds that we use

from torch.optim.optimizer import Optimizer, required
import torch
from manifold_tensors import ManifoldParameter
from manifolds import EuclideanManifold

class RiemannianSGD(Optimizer):

    def __init__(
            self,
            params,
            lr
    ):
        defaults = {
            'lr': lr,
        }
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None, **kwargs):
        """Performs a single optimization step.

        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    lr = lr or group['lr']
                    if isinstance(p, ManifoldParameter):
                        manifold = p.manifold
                    else:
                        manifold = EuclideanManifold()

                    d_p = p.grad.data
                    # Must only have sparse rows otherwise this will get messed up
                    if d_p.is_sparse:
                        d_p = d_p.coalesce()
                        indices = d_p._indices()[0]

                        manifold.rgrad_(p[indices], d_p._values())
                        manifold.retr_(p, d_p._values() * (-lr), indices=indices)
                    else:
                        d_p = p.grad.data
                        manifold.rgrad_(p.data, d_p)
                        manifold.retr_(p.data, d_p * (-lr))
                

        return loss
