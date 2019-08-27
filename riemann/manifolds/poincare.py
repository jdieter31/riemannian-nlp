from .manifold import RiemannianManifold
import torch

# Defines how close to the boundary vectors can get
EPSILON = 1e-5
# Minimum norm to allow division by
MIN_NORM = 1e-9
MAX_NORM = 1e9

"""
Some miscelaneous math before the class definition
"""
def tanh(x):
    return x.clamp(-15, 15).tanh()


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / (1 - x ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(MIN_NORM).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / (1 + x ** 2) ** 0.5


def artanh(x):
    return Artanh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)

class PoincareBall(RiemannianManifold):
    '''
    Implementation of PoincareBall model of hyperbolic space
    '''
    @classmethod
    def from_params(cls, params):
        if params is not None and 'c' in params:
            return PoincareBall(params['c'])
        return PoincareBall()

    def __init__(self, c=1.0):
        super().__init__()
        self.c = c
        
    def proj(self, x, indices=None):
        if indices is not None:
            return x.index_copy(0, indices, self.proj_(x[indices]))

        maxnorm = (1 - EPSILON) / (self.c ** 0.5)
        x_shape = x.size()
        return x.view(-1, x.size()[-1]).renorm(2, 0, maxnorm).view(x_shape)

    def proj_(self, x, indices=None):
        if indices is not None:
            x.index_copy_(0, indices, self.proj_(x[indices]))
            return x
        
        maxnorm = (1 - EPSILON) / (self.c ** 0.5)
        x.view(-1, x.size()[-1]).renorm_(2, 0, maxnorm)
        return x

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor):
        """
        Performs the mobius addition of x and y
        """
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c ** 2 * x2 * y2
        return num / denom.clamp_min(MIN_NORM)

    def dist(self, x, y, keepdim=False):
        sqrt_c = self.c ** 0.5

        dist_c = artanh(
            sqrt_c * self.mobius_add(-x, y).norm(dim=-1, p=2, keepdim=keepdim)
        )
        return dist_c * 2 / sqrt_c

    def lambda_x(self, x: torch.Tensor, keepdim: bool = False):
        """
        Computes the conformal factor for a point on the ball
        """
        return 2 / (1 - self.c * x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(MIN_NORM)
        
    def exp(self, x, u):
        sqrt_c = self.c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        assert u_norm.max() < MAX_NORM
        second_term = tanh(sqrt_c / 2 * self.lambda_x(x, keepdim=True) * u_norm) * u / (sqrt_c * u_norm)
        gamma_1 = self.mobius_add(x, second_term)
        return gamma_1

    def retr(self, x, u, indices=None):
        if indices is not None:
            x = x.index_add(0, indices, u)
            return self.proj_(x, indices=indices)
        else:
            x = x + u
            return self.proj_(x)

    def retr_(self, x, u, indices=None):
        if indices is not None:
            x = x.index_add_(0, indices, u)
            return self.proj_(x, indices=indices)
        else:
            x = x.add_(u)
            return self.proj_(x)

    def log(self, x, y):
        sub = self.mobius_add(-x, y)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        lam = self.lambda_x(x, keepdim=True)
        sqrt_c = self.c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def rgrad(self, x, dx):
        return dx / (self.lambda_x(x, keepdim=True) ** 2)

    def rgrad_(self, x, dx):
        return dx.div_(self.lambda_x(x, keepdim=True) ** 2)

    def tangent_proj_matrix(self, x):
        tangent_matrix = torch.eye(x.size()[-1], dtype=x.dtype, device=x.device)
        for i in range(len(x.size()) - 1):
            tangent_matrix.unsqueeze_(0)
        return tangent_matrix.expand(*x.size(), x.size()[-1])

    def get_metric_tensor(self, x):
        metric = torch.eye(x.size()[-1], dtype=x.dtype, device=x.device)
        for i in range(len(x.size()) - 1):
            metric.unsqueeze_(0)
        metric = metric.expand(*x.size(), x.size()[-1])
        scaling_factor = self.lambda_x(x, keepdim=True)
        scaling_factor.unsqueeze_(-1)
        metric = metric * scaling_factor
        return metric

RiemannianManifold.register_manifold(PoincareBall)
