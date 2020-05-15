from ..config import ConfigDict

CONFIG_NAME = "learning"

class LearningConfig(ConfigDict):
    """
    Configuration for learning rates and gradient norm balancing
    """
    grad_norm_lr: float = 0.05
    grad_norm_initial_refresh_rate: int = 200
    grad_norm_alpha: float = 1

    lr: float = 0.0002


