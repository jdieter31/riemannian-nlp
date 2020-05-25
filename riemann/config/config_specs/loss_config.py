from ..config import ConfigDict
import os
from ..manifold_initialization_config import ManifoldInitializationConfig

CONFIG_NAME = "loss"

class LossConfig(ConfigDict):
    """
    Configuration for loss functions
    """
    margin: float = 0.3

    random_isometry_samples: int = 30
    random_isometry_initialization: ManifoldInitializationConfig = \
        ManifoldInitializationConfig(
            default_params=[-1.0, 1.0] 
        )
    conformal: bool = True
    max_distortion: float = 4
