from ..config_spec import ConfigSpec

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")

class DataConfig(ConfigSpec)
    config_name = "data"

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
