from ..config import ConfigDict
from ..graph_sampling_config import GraphSamplingConfig

CONFIG_NAME = "sampling"

class SamplingConfig(ConfigDict):
    """
    Configuration for neighborhood sampling
    """
    train_sampling_config: GraphSamplingConfig = GraphSamplingConfig()
    eval_sampling_config: GraphSamplingConfig = GraphSamplingConfig()
