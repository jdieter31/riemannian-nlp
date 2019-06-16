# Based on FBs code for RiemannianSGD since that works with Hogwild training and adapted to work
# with the implementation of Riemannian manifolds that we use

from torch.optim.optimizer import Optimizer, required


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

        for group in self.param_groups:
            for p in group['params']:
                lr = group['lr']
                manifold = group['manifold']

                d_p = p.grad.data
                d_p = manifold.egrad2rgrad(p.data, d_p)
                manifold.retr(p.data, d_p, -lr)
                

        return loss
