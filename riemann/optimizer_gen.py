from typing import List

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config.config_loader import get_config
from .rsgd_multithread import RiemannianSGD

optimizer = None
scheduler = None

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

    global optimizer
    global scheduler

    learning_config = get_config().learning

    if optimizer is None:
        optimizer = RiemannianSGD(parameter_groups, lr=learning_config.lr)
        scheduler = ReduceLROnPlateau(optimizer,
                                      patience=learning_config.patience,
                                      threshold=learning_config.threshold)

    return optimizer

def get_scheduler():
    global scheduler
    global optimizer

    learning_config = get_config().learning

    if optimizer is None:
        optimizer = RiemannianSGD(parameter_groups, lr=learning_config.lr)
        scheduler = ReduceLROnPlateau(optimizer,
                                      patience=learning_config.patience,
                                      threshold=learning_config.threshold)

    return scheduler
