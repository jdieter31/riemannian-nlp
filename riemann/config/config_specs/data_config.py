from ..config import ConfigDict
import os

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "../../..")

CONFIG_NAME = "data"

class DataConfig(ConfigDict):
    """
    Configuration for data component
    """
    dataset_name: str = "nouns"
    path: str = os.path.join(root_path, 
            "data/nouns.csv")
    graph_data_type: str = "edge"
    graph_data_format: str = "hdf5"
    symmetrize: bool = False
    num_workers: int = 5
    delimiter: str = ","

    make_eval_split: bool = True
    split_seed: int = 14534432
    split_size: float = 0.2
    split_by_edges: bool = False

    featurizer: str = "wordnet"
