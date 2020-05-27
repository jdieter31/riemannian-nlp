from ..config import ConfigDict
from ..manifold_config import ManifoldConfig
from ..manifold_initialization_config import ManifoldInitializationConfig

import os
from typing import List

CONFIG_NAME = "model"

def get_latest_model(path="model/model"):
    i = 1
    while os.path.isfile(path + f"{i}.tch"):
        i += 1
    path += f"{i}.tch"
    return path

class ModelConfig(ConfigDict):
    """
    Configuration for model component
    """
    path: str = get_latest_model()
    model_type: str = "featurized_model_manifold_network"
    intermediate_manifolds: List[ManifoldConfig] = [ManifoldConfig(),
                                                    ManifoldConfig()]
    intermediate_dims: List[int] = [500, 400]
    sparse: bool = True
    double_precision: bool = False
    manifold_initialization: ManifoldInitializationConfig = \
            ManifoldInitializationConfig()
    nonlinearity: str = "elu"
    num_poles: int = 1
    train_isometry: bool = True
