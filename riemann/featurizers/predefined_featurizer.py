import numpy as np
import torch

from ..config.config_loader import get_config


def get_predefined_featurizer():
    """
    This featurizer loads static features from a given file.
    :param graph_dataset:
    :return:
    """
    path = get_config().data.predefined_features_path
    assert path is not None, "You must provide data.predefined_features_path"

    predefined_vectors = {}
    dim = None
    with open(path) as f:
        for line in f:
            object_id, *vector = line.split()
            if dim is None:
                dim = len(vector)
            else:
                assert len(vector) == dim, \
                    f"Expected dim {dim}, but found a vector of dim {len(vector)}"
            predefined_vectors[object_id] = torch.from_numpy(np.array([float(v) for v in vector],
                                                                      dtype=np.float32))

    def featurize(object_ids, node_ids):
        return torch.stack([predefined_vectors[id_] for id_ in object_ids.tolist()])

    return featurize, dim
