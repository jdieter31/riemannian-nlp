from .config import ConfigDict
from typing import Dict, List

class ManifoldInitializationConfig(ConfigDict):
    """
    Provides a configuration for the random initialization of a tensor on a
    manifold
    """

    default_init_func: str = "uniform_" 
    default_params: List = [-0.001, 0.001]
    manifold_init_funcs: Dict[str, str] = {}
    manifold_params: Dict[str, List] = {}

    def get_initialization_dict(self):
        """
        Puts this manifold initialization in a dictionary readable by
        riemann.manifold.manifold_initialization
        """

        initialization_dict = {
            "global": {
                "init_func": self.default_init_func,
                "params": self.default_params
            }
        }
        for manifold_name, init_func in self.manifold_init_funcs.items():
            initialization_dict[manifold_name] = {}
            initialization_dict[manifold_name]["init_func"] = init_func

        for manifold_name, params in self.manifold_params.items():
            intialization_dict[manifold_name]["params"] = params

        return initialization_dict
