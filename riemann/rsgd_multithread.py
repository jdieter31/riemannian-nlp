# Based on FBs code for RiemannianSGD since that works with Hogwild training and adapted to work
# with the implementation of Riemannian manifolds that we use

import torch
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from .manifold_tensors import ManifoldParameter
from .manifolds import EuclideanManifold


class RiemannianSGD(Optimizer):

    def __init__(
            self,
            params,
            lr=0.001,
            clip_grads=True,
            clip_val=1,
            adam_for_euc=True
    ):
        if adam_for_euc:
            riemann_params = []
            adam_params = []

            for param in params:
                if isinstance(param, ManifoldParameter) and not isinstance(param.manifold,
                                                                           EuclideanManifold):
                    riemann_params.append(param)
                else:
                    adam_params.append(param)

            self.adam_optimizer = torch.optim.Adam(adam_params, lr=lr, betas=(.9, .995), eps=1e-9)
            params = riemann_params

        defaults = {
            'lr': lr,
        }
        self.clip_grads = clip_grads
        self.clip_val = clip_val
        if adam_for_euc and len(riemann_params) == 0:
            self.all_adam = True
            super(RiemannianSGD, self).__init__(adam_params, defaults)
        else:
            self.all_adam = False
            super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, **kwargs):
        """
        Performs a single optimization step. Returns gradient norm
        """

        if self.adam_optimizer is not None:
            self.adam_optimizer.step()

        if self.all_adam:
            return

        norms = []
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    lr = group['lr']

                    if isinstance(p, ManifoldParameter):
                        manifold = p.manifold
                        lr *= p.lr_scale
                    else:
                        manifold = EuclideanManifold()

                    d_p = p.grad.data
                    # Must only have sparse rows otherwise this will get messed up

                    if d_p.is_sparse:
                        if d_p._nnz() == 0:
                            continue
                        d_p = d_p.coalesce()
                        indices = d_p._indices()[0]

                        norms.append(torch.flatten(d_p._values()).norm().cpu().detach())

                        manifold.rgrad_(p[indices], d_p._values())

                        if self.clip_grads:
                            if d_p._values().max() > self.clip_val or d_p._values().min() < -self.clip_val:
                                tqdm.write(
                                    f"Warning -- riemannian-gradients were clipped on {manifold} with max_val {d_p._values().abs().max()}")

                            d_p._values().clamp(-self.clip_val, self.clip_val)
                        manifold.retr_(p, d_p._values() * (-lr), indices=indices)
                    else:
                        d_p = p.grad.data
                        norms.append(torch.flatten(d_p).norm().cpu().detach())
                        manifold.rgrad_(p.data, d_p)

                        if self.clip_grads:
                            if d_p.max() > self.clip_val or d_p.min() < -self.clip_val:
                                tqdm.write(
                                    f"Warning -- gradients were clipped on {manifold} with max_val {d_p.abs().max()}")
                            d_p = d_p.clamp(-self.clip_val, self.clip_val)
                        manifold.retr_(p.data, d_p * (-lr))

        return float(torch.tensor(norms).norm().cpu().detach().numpy())

    def zero_grad(self):
        super().zero_grad()

        if self.adam_optimizer is not None:
            self.adam_optimizer.zero_grad()
