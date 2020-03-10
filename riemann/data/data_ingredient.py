import os
from ..config.config_loader import register_default_config
from ..config.config import ConfigDict
from math import floor

from .graph import load_edge_list, load_adjacency_matrix

from .graph_dataset import BatchedDataset

from ..config_loader import get_config

import numpy as np

def load_dataset():
    data_config = get_config().data    

    if graph_data_type == "edge":
        idx, objects, weights = load_edge_list(data_config.path, data_config.symmetrize, delimiter=data_config.delimiter)
    else:
        idx, objects, weights = load_adjacency_matrix(data_config.path, data_config.graph_data_format, data_config.symmetrize)

    # define a feature function
    if data_config.object_id_to_feature_func == "conceptnet":
        features = [' '.join(object_id.split('_')) for object_id in objects]
    elif data_config.object_id_to_feature_func == "wordnet":
        # placental_mammal.n.01 -> placental mammal
        features = [' '.join(object_id.split('.')[0].split('_')) for object_id in objects]
    elif data_config.object_id_to_feature_func == "id":
        # xyz -> xyz
        features = [object_id for object_id in objects]
    else:
        features = None

    if make_eval_split:
        np.random.seed(data_config.split_seed)
        shuffle_order = np.arange(idx.shape[0])
        np.random.shuffle(shuffle_order)
        num_eval = floor(idx.shape[0] * data_config.split_size)
        eval_indices = shuffle_order[:num_eval]
        train_indices = shuffle_order[num_eval:]
        train_idx = idx[train_indices]
        train_weights = weights[train_indices]
        eval_idx = idx[eval_indices]
        eval_weights = weights[eval_indices]

        train_data = BatchedDataset(
                train_idx,
                objects,
                train_weights,
                data_config.manifold,
                data_config.n_graph_neighbors,
                data_config.n_manifold_neighbors,
                data_config.n_rand_neighbors,
                data_config.batch_size,
                data_config.num_workers,
                data_config.nn_workers,
                data_config.manifold_nn_k,
                features,
                saved_data_file=data_config.graph_data_file,
                gen_data=data_config.gen_graph_data
                )


        eval_data = BatchedDataset.initialize_eval_dataset(
                train_data,
                eval_batch_size,
                data_config.n_eval_neighbors,
                data_config.max_eval_graph_neighbors,
                data_config.eval_workers, 
                data_config.eval_nn_workers, 
                manifold_neighbors=data_config.eval_manifold_neighbors,
                eval_edges=eval_idx, 
                eval_weights=eval_weights)

        return train_data, eval_data
    else: 
        train_data = BatchedDataset(
            idx,
            objects,
            weights,
            manifold,
            data_config.n_graph_neighbors,
            data_config.n_manifold_neighbors,
            data_config.n_rand_neighbors,
            data_config.batch_size,
            data_config.num_workers,
            data_config.nn_workers,
            data_config.manifold_nn_k,
            features,
            saved_data_file=data_config.graph_data_file, 
            gen_data=data_config.gen_graph_data)

        eval_data = BatchedDataset.initialize_eval_dataset(
                train_data,
                data_config.eval_batch_size,
                data_config.n_eval_neighbors,
                data_config.max_eval_graph_neighbors,
                data_config.eval_workers, 
                data_config.eval_nn_workers, 
                manifold_neighbors=data_config.eval_manifold_neighbors,
                saved_data_file=data_config.graph_data_file, 
                gen_data=data_config.gen_graph_data)

        return train_data, eval_data

def get_adjacency_dict(data):
    adj = {}
    for row in data.idx:
        x = row[0]
        y = row[1]
        if x in adj:
            adj[x].add(y)
        else:
            adj[x] = {y}
    return adj
