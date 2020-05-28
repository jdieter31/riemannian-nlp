import re
from typing import Dict

from .config import ConfigDict
from ..manifolds.manifold import RiemannianManifold


class ManifoldConfig(ConfigDict):
    """
    ConfigDict to specify a manifold and its properties
    """
    name: str = "EuclideanManifold"
    dimension: int = 0
    params: dict = {}

    def get_manifold_instance(self) -> RiemannianManifold:
        """
        Gets an instance of the manifold specified in this config
        """
        return RiemannianManifold.from_name_params(self.name, self.params)

    @classmethod
    def from_string(cls, spec) -> 'ManifoldConfig':
        pattern = re.compile(r"([ESH])([0-9]+)")
        short_forms = {
            "E": "EuclideanManifold",
            "S": "SphericalManifold",
            "H": "PoincareBall",
        }

        if "x" in spec:
            submanifolds, total_dim = [], 0
            for subspec in spec.split("x"):
                match = pattern.match(subspec)
                assert match is not None, f"Invalid spec {spec}"
                typ, dim = match.groups()

                submanifolds.append({
                    "name": short_forms[typ],
                    "dimension": int(dim)
                })
                total_dim += int(dim)
            return cls(name="ProductManifold", dimension=total_dim,
                       params={"submanifolds": submanifolds})
        else:
            match = pattern.match(spec)
            assert match is not None, f"Invalid spec {spec}"
            typ, dim = match.groups()
            return cls(name=short_forms[typ], dimension=int(dim), params={})
