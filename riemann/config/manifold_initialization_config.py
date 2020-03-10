from .config import ConfigDict
from typing import Dict, List

class ManifoldInitializationConfig(ConfigDict):
    default_init_func: str = "_uniform" 
    default_params: List = [-0.001, 0.001]
    manifold_init_funcs: Dict[str, str] = {}
    manifold_params: Dict[str, List] = {}
