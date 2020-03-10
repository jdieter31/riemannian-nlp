from .config import ConfigDict
from typing import Dict

class ManifoldConfig(ConfigDict):
    """
    ConfigDict to specify a manifold and its properties
    """
    name: str = "RiemannianManifold"
    params: Dict = {}
