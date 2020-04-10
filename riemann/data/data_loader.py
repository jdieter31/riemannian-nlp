from .graph_dataset import GraphDataset
from ..config.config_loader import get_config
from .graph_loader_utils import load_csv_edge_list, load_adjacency_matrix

training_data: GraphDataset = None
eval_data: GraphDataset = None

def get_training_data() -> GraphDataset:
    """
    Loads the training data or fetches it if already loaded
    """
    _load_data_if_needed()
    return training_data

def get_eval_data() -> GraphDataset:
    """
    Loads the eval data or fetches it if already loaded. Will return None if no
    train eval split is configured
    """
    _load_data_if_needed()
    return eval_data

def _load_data_if_needed():
    global training_data
    global eval_data

    if training_data is None:

        data_config = get_config().data

        if data_config.graph_data_type == "edge":
            idx, objects, weights = \
            load_csv_edge_list(data_config.path, data_config.symmetrize,
                           delimiter=data_config.delimiter)
        else:
            idx, objects, weights \
            = load_adjacency_matrix(data_config.path, data_config.graph_data_format,
                                    data_config.symmetrize)


        if data_config.make_eval_split:
            training_data, eval_data = \
            GraphDataset.make_train_eval_split(data_config.dataset_name, idx,
                                               objects, weights)
        else:
            training_data = GraphDataset(data_config.dataset_name, idx, objects,
                                      weights)


