from ..config import ConfigDict

CONFIG_NAME = "eval"


class EvalConfig(ConfigDict):
    """
    Evaluation config
    """
    eval_link_pred: bool = True
    eval_reconstruction: bool = True
    make_visualization: bool = False
    visualization_frequency: int = 20
    link_pred_frequency: int = 50
    reconstruction_frequency: int = 50
    data_fraction: float = 1
