import os
from ..config.config_loader import register_default_config
from ..config.config import ConfigDict
from math import floor

from .graph import load_edge_list, load_adjacency_matrix

from .graph_dataset import BatchedDataset

from ..config_loader import global_config

import numpy as np



class DataConfig(ConfigDict):
    path = os.path.join(root_path, "data/en_conceptnet_regularized_filtered.csv")
    graph_data_type = "edge"
    graph_data_format = "hdf5"
    symmetrize = False
    num_workers = 5
    nn_workers = 25
    n_graph_neighbors = 10
    n_manifold_neighbors = 20
    n_rand_neighbors = 5
    batch_size = 2000
    manifold_nn_k = 30
    delimiter = "\t"

    make_eval_split = False
    split_seed = 14534432
    split_size = 0.25
    eval_batch_size = 50
    n_eval_neighbors = 1000
    max_eval_graph_neighbors = 500
    eval_manifold_neighbors = 50
    eval_workers = 2
    eval_nn_workers = 1

    graph_data_file = os.path.join(root_path, "data/en_conceptnet_uri_filtered_gdata.pkl")
    gen_graph_data = False

    # Valid values are conceptnet, wordnet
    object_id_to_feature_func = "id"

register_default_config("data", DataConfig)

'''
def load_dataset(
        manifold,
        graph_data_type,
        path,
        n_graph_neighbors,
        n_manifold_neighbors,
        n_rand_neighbors,
        batch_size,
        num_workers,
        nn_workers,
        symmetrize,
        graph_data_format,
        manifold_nn_k,
        delimiter,
        make_eval_split,
        split_seed,
        split_size,
        eval_batch_size,
        n_eval_neighbors,
        max_eval_graph_neighbors,
        eval_manifold_neighbors,
        eval_workers,
        eval_nn_workers,
        graph_data_file,
        gen_graph_data,
        object_id_to_feature_func=None):
'''
def load_dataset():
    data_config = global_config["data"]     

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
                n_graph_neighbors,
                n_manifold_neighbors,
                n_rand_neighbors,
                batch_size,
                num_workers,
                nn_workers,
                manifold_nn_k,
                features,
                saved_data_file=graph_data_file,
                gen_data=gen_graph_data
                )


        eval_data = BatchedDataset.initialize_eval_dataset(
                train_data,
                eval_batch_size,
                n_eval_neighbors,
                max_eval_graph_neighbors,
                eval_workers, 
                eval_nn_workers, 
                manifold_neighbors=eval_manifold_neighbors,
                eval_edges=eval_idx, 
                eval_weights=eval_weights)

        return train_data, eval_data
    else: 
        train_data = BatchedDataset(
            idx,
            objects,
            weights,
            manifold,
            n_graph_neighbors,
            n_manifold_neighbors,
            n_rand_neighbors,
            batch_size,
            num_workers,
            nn_workers,
            manifold_nn_k,
            features,
            saved_data_file=graph_data_file, 
            gen_data=gen_graph_data)

        eval_data = BatchedDataset.initialize_eval_dataset(
                train_data,
                eval_batch_size,
                n_eval_neighbors,
                max_eval_graph_neighbors,
                eval_workers, 
                eval_nn_workers, 
                manifold_neighbors=eval_manifold_neighbors,
                saved_data_file=graph_data_file, 
                gen_data=gen_graph_data)

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
