import os
from typing import Optional

from ..config import ConfigDict

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         "../../..")

CONFIG_NAME = "data"


class DataConfig(ConfigDict):
    """
    Configuration for data component
    """
    dataset_name: str = "mammals"
    train_path: str = os.path.join(root_path, 
            "data/mammals/mammals_train.csv")
    eval_path: str = os.path.join(root_path,
            "data/mammals/mammals_eval.csv")
    test_path: str = os.path.join(root_path,
            "data/mammals/mammals_test.csv")
    graph_data_type: str = "edge"
    graph_data_format: str = "hdf5"
    symmetrize: bool = False
    num_workers: int = 5
    delimiter: str = ","

    generate_eval_split: bool = False
    generate_test_set: bool = True
    full_path: str = os.path.join(root_path,
                                  "data/mammals.csv")
    split_seed: int = 14534482
    split_size: float = 0.1
    split_by_edges: bool = False

    featurizer: str = "wordnet"
    predefined_features_path: Optional[str] = None
