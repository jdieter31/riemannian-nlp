from ..config import ConfigDict
from ..manifold_config import ManifoldConfig

CONFIG_NAME = "general"


class GeneralConfig(ConfigDict):
    """
    General Configuration
    """
    n_epochs: int = 4000
    eval_every: int = 5
    gpu: int = 0
