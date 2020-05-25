from .random_featurizer import get_random_featurizer
from .wordnet_featurizer import get_wordnet_featurizer
from ..config.config_loader import get_config
from ..data.data_loader import get_training_data, get_eval_data
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
        elif featurizer_string == "wordnet":
            data = get_training_data()
            eval_data = get_eval_data()
            __featurizer, __dimension, __in_manifold = \
                get_wordnet_featurizer(data, eval_data)
        else:
            raise NotImplementedError(f"Unsupported featurizer {featurizer_string}")

    return __featurizer, __dimension, __in_manifold
