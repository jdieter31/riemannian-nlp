from ..config import ConfigDict
from ..manifold_config import ManifoldConfig

CONFIG_NAME = "general"


class GeneralConfig(ConfigDict):
    """
    General Configuration
    """
    n_epochs: int = 40
    eval_every: int = 5
    gpu: int = 0
    embed_manifold: ManifoldConfig = ManifoldConfig(
        name="SphericalManifold"
    )
   
    embed_manifold_dim: int = 400
