from ..config.config_loader import get_config
from .random_featurizer import get_random_featurizer
from ..data.data_loader import get_training_data
from ..manifolds import EuclideanManifold

__featurizer = None
__dimension = None
__in_manifold = None

def get_featurizer():
    global __featurizer
    global __dimension
    global __in_manifold

    if __featurizer is None:
        featurizer_string = get_config().data.featurizer

        if featurizer_string == "random":
            data = get_training_data()
            __featurizer = get_random_featurizer(data)
            __dimension = 2
            __in_manifold = EuclideanManifold()
           
    return __featurizer, __dimension, __in_manifold
