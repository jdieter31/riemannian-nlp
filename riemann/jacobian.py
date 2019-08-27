import torch
from torch.autograd.gradcheck import zero_gradients


def compute_jacobian(f, x, noutputs):
    """
    Computes jacobian of a function f

    Args:
        f: function to compute jacobian on
        x: input of shape batch_1 x ... x batch_n x in_dim
        noutputs: output dimension of f

    Returns:
        jacobian: tensor of shape batch_1 X batch_2 X ... X batch_n X out_dim X in_dim
        f(x): value of f(x)
    """

    x = x.unsqueeze(0)
    x_size = x.size()
    with torch.no_grad():
        x = x.repeat(noutputs, *[1 for _ in range(len(x.size()) - 1)])
    x.requires_grad_(True)
    y = f(x)
    grad_in = torch.eye(noutputs, dtype=x.dtype, device=x.device)
    for i in range(len(x.size()) - 2):
        grad_in.unsqueeze_(1)
    grad_in = grad_in.expand_as(y)
    jacobian = torch.autograd.grad(y, x, grad_outputs=grad_in, retain_graph=True, create_graph=True, only_inputs=True)[0]
    jacobian.requires_grad_(True)
    return jacobian.permute(*(list(range(1, len(jacobian.size()) - 1)) + [0, len(jacobian.size()) - 1])), y[0]
