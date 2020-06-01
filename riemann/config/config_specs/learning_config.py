from typing import List

from ..config import ConfigDict

CONFIG_NAME = "learning"


class LearningConfig(ConfigDict):
    """
    Configuration for learning rates and gradient norm balancing
    """
    grad_norm_lr: float = 0.05
    grad_norm_initial_refresh_rate: int = 200000
    grad_norm_alpha: float = 0.6

    loss_priority: List[float] = [0.5, 0.5]

    lr: float = 0.00005
    threshold: int = 50
    patience: int = 150
