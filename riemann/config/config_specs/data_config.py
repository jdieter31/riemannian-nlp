from ..config import ConfigDict
import os

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")

CONFIG_NAME = "data"

class DataConfig(ConfigDict):
    """
    Configuration for data component
    """
    path: str = os.path.join(root_path, 
            "data/en_conceptnet_regularized_filtered.csv")
    graph_data_type: str = "edge"
    graph_data_format: str = "hdf5"
    symmetrize: bool = False
    num_workers: int = 5
    nn_workers: int = 25
    n_graph_neighbors: int = 10
    n_manifold_neighbors: int = 20
    n_rand_neighbors: int = 5
    batch_size: int = 2000
    manifold_nn_k: int = 30
    delimiter: str = "\t"

    make_eval_split: bool = False
    split_seed: int = 14534432
    split_size: float = 0.25
    eval_batch_size: int = 50
    n_eval_neighbors: int = 1000
    max_eval_graph_neighbors: int = 500
    eval_manifold_neighbors : int = 50
    eval_workers: int = 2
    eval_nn_workers: int = 1

    graph_data_file: str = os.path.join(root_path,
        "data/en_conceptnet_uri_filtered_gdata.pkl")
    gen_graph_data: bool = False

    # Valid values are conceptnet, wordnet
    object_id_to_feature_func: str = "id"
