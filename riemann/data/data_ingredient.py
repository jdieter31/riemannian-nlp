from sacred import Ingredient

from data.graph import load_edge_list, load_adjacency_matrix

from data.graph_dataset import BatchedDataset

data_ingredient = Ingredient("dataset")

@data_ingredient.config
def config():
    path = "data/noun_closure.csv"
    type = "edge"
    format = "hdf5"
    symmetrize = False
    burnin = False
    num_workers = 1
    num_negs = 50
    batch_size = 50
    sample_dampening = 0.75

@data_ingredient.capture
def load_dataset(type, path,  num_negs, batch_size, num_workers, symmetrize, format, burnin, sample_dampening):
    if type == "edge":
        idx, objects, weights = load_edge_list(path, symmetrize)
    else:
        idx, objects, weights = load_adjacency_matrix(path, format, symmetrize)

    return BatchedDataset(idx, objects, weights, num_negs, batch_size, num_workers, burnin, sample_dampening)

def get_adjacency_dict(data):
    adj = {}
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}
    return adj
