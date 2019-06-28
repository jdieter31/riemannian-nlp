# Based on FBs code for RiemannianSGD since that works with Hogwild training and adapted to work
# with the implementation of Riemannian manifolds that we use

from torch.optim.optimizer import Optimizer, required
import torch

class RiemannianSGD(Optimizer):

    def __init__(
            self,
            params,
            lr,
            manifold
    ):
        defaults = {
            'lr': lr,
            'manifold': manifold
        }
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, **kwargs):
        """Performs a single optimization step.

        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    lr = group['lr']
                    manifold = group['manifold']

                    d_p = p.grad.data
                    # Must only have sparse rows otherwise this will get messed up
                    if d_p.is_sparse:
                        d_p = d_p.coalesce()
                        indices = d_p._indices()[0]
                        manifold.egrad2rgrad(p[indices], d_p._values())
                        manifold._retr(p, d_p._values(), -lr, indices=indices)
                    else:
                        d_p = p.grad.data
                        manifold.egrad2rgrad(p.data, d_p)
                        manifold._retr(p.data, d_p, -lr)
                

        return loss
