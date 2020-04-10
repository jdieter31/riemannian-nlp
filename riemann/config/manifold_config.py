from .config import ConfigDict
from typing import Dict
from ..manifolds.manifold import RiemannianManifold

class ManifoldConfig(ConfigDict):
    """
    ConfigDict to specify a manifold and its properties
    """
    name: str = "EuclideanManifold"
    params: Dict = {}

    def get_manifold_instance(self) -> RiemannianManifold:
        """
        Gets an instance of the manifold specified in this config
        """
        return RiemannianManifold.from_name_params(self.name, self.params)

