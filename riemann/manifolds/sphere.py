from .manifold import RiemannianManifold
import torch

# Determines when to use approximations when dividing by small values is a possibility
EPSILON = 1e-4

class GradClippedACos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + EPSILON,  1 - EPSILON)
        ctx.save_for_backward(x)
        dtype = x.dtype
        return torch.acos(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = - grad_output / ((1 - x ** 2) ** 0.5)
        return grad

def acos(x):
    return GradClippedACos.apply(x)

class SphericalManifold(RiemannianManifold):
    '''
    Implementation of Spherical Riemannian manifold with standard pullback metric
    '''

    @classmethod
    def from_params(cls, params):
        return SphericalManifold()

    def __str__(self):
        return "S"

    def proj(self, x, indices=None):
        if indices is not None:
            norm = x[indices].norm(dim=-1, keepdim=True)
            x_proj = x.clone()
            x_proj[indices] /= norm
        else:
            norm = x.norm(dim=-1, keepdim=True)
            out = x / norm
            out[norm.squeeze(-1) == 0] = (1 / x.size()[-1]) ** (1/2)
            return out

    def proj_(self, x, indices=None):
        if indices is not None:
            norm = x[indices].norm(dim=-1, keepdim=True)
            x[indices] /= norm
            return x
        else:
            norm = x.norm(dim=-1, keepdim=True)
            x.div_(norm)
            x[norm.squeeze(-1) == 0] = (1 / x.size()[-1]) ** (1/2)
            return x

    def retr(self, x, u, indices=None):
        if indices is not None:
            y = x.index_add(0, indices, u)
        else:
            y = x + u
        return self.proj(y, indices)

    def retr_(self, x, u, indices=None):
        if indices is not None:
            x.index_add_(0, indices, u)
        else:
            x = x.add_(u)
        return self.proj_(x, indices)

    def exp(self, x, u):
        # Ensure u is in tangent space
        #u = u - (x * u).sum(dim=-1, keepdim=True) * x

        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self.proj(x + u)
        cond = norm_u > EPSILON
        out = torch.where(cond, exp, retr)
        return out

    def log(self, x, y):
        u = y - x
        u = u - (x * u).sum(dim=-1, keepdim=True) * x
        dist = self.dist(x, y, keepdim=True)
        norm_u = u.norm(dim=-1, keepdim=True)
        cond = norm_u > EPSILON
        return torch.where(cond, u * dist / norm_u, u)

    def dist(self, x, y, keepdim=False):
        inner = (x * y).sum(-1, keepdim=keepdim)
        inner = inner.clamp(-.9999, .9999)
        return acos(inner)

    def rgrad(self, x, dx):
        return dx - (x * dx).sum(dim=-1, keepdim=True) * x

    def rgrad_(self, x, dx):
        return dx.sub_((x * dx).sum(dim=-1, keepdim=True) * x)

    def lower_indices(self, x, dx):
        return dx

    def tangent_proj_matrix(self, x):
        tangent_matrix = torch.eye(x.size()[-1], dtype=x.dtype, device=x.device)
        for i in range(len(x.size()) - 1):
            tangent_matrix.unsqueeze_(0)
        x_shape = x.size()
        x_reduced = x.view(-1, x.size()[-1])
        ortho_proj = torch.bmm(x_reduced.unsqueeze(-1), x_reduced.unsqueeze(-2)).view(*x.size(), x.size()[-1])
        return tangent_matrix.expand(*x.size(), x.size()[-1]) - ortho_proj

    def get_metric_tensor(self, x):
        metric = torch.eye(x.size()[-1], dtype=x.dtype, device=x.device)
        for i in range(len(x.size()) - 1):
            metric.unsqueeze_(0)
        return metric.expand(*x.size(), x.size()[-1])

RiemannianManifold.register_manifold(SphericalManifold)
