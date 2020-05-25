import numpy as np
import torch
from ..data.graph_dataset import GraphDataset

def get_random_featurizer(graph_dataset: GraphDataset, dimension=2, grid_size=4):
    random_vectors = torch.empty((graph_dataset.n_nodes(), dimension),
                                 dtype=torch.float, device=torch.device('cpu'))

    """
    with torch.no_grad():
        # Initialize randomly on a grid

        torch.manual_seed(100324324)
        torch.nn.init.uniform_(random_vectors, a=-1)
        # random_vectors *= grid_size
        # random_vectors.round_()
        # random_vectors /= grid_size
    """
    
    np.random.seed(3243241)
    random_vectors = np.array([
     [(i - 2)/2 , (j - 2)/2] for j in range (4) for i in range(4)
    ])
    np.random.shuffle(random_vectors)

    """
    random_vectors = np.array([
        [0.4, -1],
        [-0.4, -1],
        [-1, 0.4],
        [1, 0.4]
    ])
    """

    random_vectors = torch.tensor(random_vectors, dtype=torch.float, device=torch.device('cpu'))


    def featurize(object_ids, node_ids):
        return random_vectors[node_ids]

    return featurize
    
     
