from ..config import ConfigDict
from ..manifold_config import ManifoldConfig

CONFIG_NAME = "general"

class GeneralConfig(ConfigDict):
    """
    General Configuration
    """
    n_epochs: int = 1000
    eval_every: int = 5
    gpu: int = 0
    embed_manifold: ManifoldConfig = ManifoldConfig()
    embed_manifold_dim: int = 3
