from ..config import ConfigDict
import os

CONFIG_NAME = "loss"

class LossConfig(ConfigDict):
    """
    Configuration for loss functions
    """
    margin: float = 0.01
