from sacred import Ingredient

from .graph import load_edge_list, load_adjacency_matrix

from .graph_dataset import BatchedDataset

data_ingredient = Ingredient("dataset")

@data_ingredient.config
def config():
    path = "data/mammals.csv"
    type = "edge"
    format = "hdf5"
    symmetrize = False
    num_workers = 3
    num_negs = 50
    batch_size = 1000
    sample_dampening = 0.75
    # placental_mammal.n.01 -> placental mammal
    object_id_to_feature_func = lambda word_id : ' '.join(word_id.split('.')[0].split('_'))
    # object_id_to_feature_func = lambda word : ' '.join(word.split('_'))

    sample_data = "graph_dist"

@data_ingredient.capture
def load_dataset(type, path, num_negs, batch_size, num_workers, symmetrize, format, burnin, sample_dampening, sample_data, object_id_to_feature_func=None):
    if type == "edge":
        idx, objects, weights = load_edge_list(path, symmetrize)
    else:
        idx, objects, weights = load_adjacency_matrix(path, format, symmetrize)
    features = None
    if object_id_to_feature_func is not None:
        features = [object_id_to_feature_func(object_id) for object_id in objects]

    return BatchedDataset(idx, objects, weights, num_negs, batch_size, num_workers, burnin, sample_dampening, features, sample_data)

def get_adjacency_dict(data):
    adj = {}
    sample_data = data.sample_data
    data.sample_data = "targets"
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}
    data.sample_data = sample_data
    return adj
