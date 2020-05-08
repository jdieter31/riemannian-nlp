from ..config import ConfigDict
from ..manifold_config import ManifoldConfig

CONFIG_NAME = "eval"

class EvalConfig(ConfigDict):
    """
    Evaluation config
    """
    eval_link_pred: bool = False
    eval_reconstruction: bool = True
    link_pred_frequency: int = 3
    reconstruction_frequency: int = 3
    data_fraction: float = 1
