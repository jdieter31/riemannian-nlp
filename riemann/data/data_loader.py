from .graph_dataset import GraphDataset
from .graph_loader_utils import load_csv_edge_list, load_adjacency_matrix
from ..config.config_loader import get_config

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
            if data_config.generate_eval_split:
                path = data_config.full_path
            else:
                path = data_config.train_path


            idx, objects, weights = \
                load_csv_edge_list(path, data_config.symmetrize,
                                   delimiter=data_config.delimiter)
        else:
            # TODO This needs to be updated for handling train eval splits
            idx, objects, weights \
                = load_adjacency_matrix(data_config.train_path, data_config.graph_data_format,
                                        data_config.symmetrize)

        if data_config.generate_eval_split:

            training_data, eval_data = \
                GraphDataset.make_train_eval_split(data_config.dataset_name, idx,
                                                   objects, weights)
        else:
            if data_config.eval_path is not None:
                eval_idx, eval_objects, eval_weights = \
                    load_csv_edge_list(data_config.eval_path,
                                    data_config.symmetrize,
                                    delimiter=data_config.delimiter)
                
                # Correct indexing of objects in eval_idx
                for edge in eval_idx:
                    for i in range(2):
                        if eval_objects[edge[i]] not in objects:
                            objects.append(eval_objects[edge[i]])
                            edge[i] = len(objects) - 1
                        else:
                            edge[i] = objects.index(eval_objects[edge[i]])
                training_data = GraphDataset(f"{data_config.dataset_name}_train",
                                          idx, objects, weights)
                eval_data = GraphDataset(f"{data_config.dataset_name}_eval",
                                          eval_idx, objects, eval_weights)
            else:

                training_data = GraphDataset(data_config.dataset_name, idx, objects,
                                            weights)
