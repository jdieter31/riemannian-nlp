from ..config import ConfigDict
from ..manifold_config import ManifoldConfig
from ..manifold_initialization_config import ManifoldInitializationConfig

import os
from typing import List

CONFIG_NAME = "model"

class ModelConfig(ConfigDict):
    """
    Configuration for model component
    """
    path: str = "models/model"
    i = 1
    while os.path.isfile(path + f"{i}.tch"):
        i += 1
    path += f"{i}.tch"
    model_type: str = "featurized_model_manifold_network"
    intermediate_manifolds: List[ManifoldConfig] = [ManifoldConfig()]        
    intermediate_manifold_gen_products: bool = None
    intermediate_dims: List[int] = [2700, 900]
    sparse: bool = True
    double_precision: bool = False
    manifold_initialization: ManifoldInitializationConfig = \
            ManifoldInitializationConfig()
    nonlinearity: bool = None
    num_poles: int = 1
    tries: int = 10
    num_layers: int = 2

    featurizer_name: str = "conceptnet"
    cn_vector_frame_file: str = "data/glove_w2v_merge.h5"
    input_manifold: str = "Euclidean"
