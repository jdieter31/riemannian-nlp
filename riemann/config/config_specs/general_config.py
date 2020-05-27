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
    """
    embed_manifold: ManifoldConfig = ManifoldConfig(
        name="ProductManifold",
        params={
            "submanifolds": [
                {
                    "name": "SphericalManifold",
                    "dimension": 300
                },
                {
                    "name": "PoincareBall",
                    "dimension": 5
                }
            ]
        }
    )
    """

    embed_manifold: ManifoldConfig = ManifoldConfig(
        name="PoincareBall"
    )
   
    embed_manifold_dim: int = 305
