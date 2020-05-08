from ..config import ConfigDict

CONFIG_NAME = "learning"

class LearningConfig(ConfigDict):
    """
    Configuration for learning rates and gradient norm balancing
    """
    grad_norm_lr: float = 0.1
    grad_norm_initial_refresh_rate: int = 5
    grad_norm_alpha: float = 0.6

    lr: float = 0.01


