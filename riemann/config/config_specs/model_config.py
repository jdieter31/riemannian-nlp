import os
from typing import List

from ..config import ConfigDict
from ..manifold_config import ManifoldConfig
from ..manifold_initialization_config import ManifoldInitializationConfig

CONFIG_NAME = "model"


class ModelConfig(ConfigDict):
    """
    Configuration for model component
    """
    intermediate_manifold: str = "E5"
    intermediate_layers: int = 2
    target_manifold: str = "H5"

    sparse: bool = True
    double_precision: bool = False
    manifold_initialization: ManifoldInitializationConfig = ManifoldInitializationConfig()
    nonlinearity: str = "elu"
    num_poles: int = 1

    @property
    def intermediate_manifolds(self) -> List[ManifoldConfig]:
        return [ManifoldConfig.from_string(self.intermediate_manifold)] * self.intermediate_layers

    @property
    def target_manifold_(self) -> ManifoldConfig:
        return ManifoldConfig.from_string(self.target_manifold)
