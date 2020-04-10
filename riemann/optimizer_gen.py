from torch.optim.optimizer import Optimizer
from .rsgd_multithread import RiemannianSGD

optimizer = None

parameter_groups: List = []

def register_parameter_group(params, **optimizer_params):
    global parameter_groups

    param_dict = {
        "params": params
    }
    param_dict.update(optimizer_params)

    parameter_groups.append(param_dict)
    

def get_optimizer() -> Optimizer:
    """
    Gets the RSGD optimizer - make sure all parameter groups are registered
    with register_parameter_group before calling this
    """

    global obtimizer

    if optimizer is None:
        optimizer = RiemannianSGD(parameter_groups)

    return optimizer
        

