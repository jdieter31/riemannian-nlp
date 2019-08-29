from sacred import Ingredient

from .graph import load_edge_list, load_adjacency_matrix

from .graph_dataset import BatchedDataset

data_ingredient = Ingredient("dataset")

@data_ingredient.config
def config():
    path = "data/mammals.csv"
    graph_data_type = "edge"
    graph_data_format = "hdf5"
    symmetrize = False
    num_workers = 3
    n_graph_neighbors = 25
    n_manifold_neighbors = 40
    n_rand_neighbors = 10
    batch_size = 1000
    manifold_nn_k = 100

    # placental_mammal.n.01 -> placental mammal
    # object_id_to_feature_func = lambda word_id : ' '.join(word_id.split('.')[0].split('_'))
    # object_id_to_feature_func = lambda word : ' '.join(word.split('_'))


@data_ingredient.capture
def load_dataset(
        manifold,
        graph_data_type,
        path,
        n_graph_neighbors,
        n_manifold_neighbors,
        n_rand_neighbors,
        batch_size,
        num_workers,
        symmetrize,
        graph_data_format,
        manifold_nn_k,
        object_id_to_feature_func=None):

    if graph_data_type == "edge":
        idx, objects, weights = load_edge_list(path, symmetrize)
    else:
        idx, objects, weights = load_adjacency_matrix(path, graph_data_format, symmetrize)
    features = None
    if object_id_to_feature_func is not None:
        features = [object_id_to_feature_func(object_id) for object_id in objects]

    return BatchedDataset(
            idx,
            objects,
            manifold,
            n_graph_neighbors,
            n_manifold_neighbors,
            n_rand_neighbors,
            batch_size,
            num_workers,
            manifold_nn_k,
            features)

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
