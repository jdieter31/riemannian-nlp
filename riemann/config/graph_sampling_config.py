from .config import ConfigDict


class GraphSamplingConfig(ConfigDict):
    """
    Provides a configuration for the neighbor sampling algorithm
    """

    """
    Max number of graph 1-neighbors to sample (rest will be substituted for
    random neighbors if there is less than the max number of 1-neighbors
    """
    n_graph_neighbors: int = 20

    """
    Number of manifold neighbors to sample
    """
    n_manifold_neighbors: int = 0

    """
    Number of neighbors to sample uniformly at random
    """
    n_rand_neighbors: int = 20

    batch_size: int = 50000

    """
    Number of workers to use when running the graph sampling algorithm
    """
    num_workers: int = 4
