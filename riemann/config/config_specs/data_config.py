from ..config import ConfigDict
import os

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "../../..")

CONFIG_NAME = "data"

class DataConfig(ConfigDict):
    """
    Configuration for data component
    """
    dataset_name: str = "simple"
    path: str = os.path.join(root_path, 
            "data/simple.csv")
    graph_data_type: str = "edge"
    graph_data_format: str = "hdf5"
    symmetrize: bool = False
    num_workers: int = 5
    delimiter: str = "\t"

    make_eval_split: bool = False
    split_seed: int = 14534432
    split_size: float = 0.25

    featurizer: str = "random"
