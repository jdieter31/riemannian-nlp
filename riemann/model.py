from .config.config_loader import get_config
from .data.data_loader import get_training_data
from .device_manager import get_device
from .featurizers.featurizer import get_featurizer
from .featurizers.graph_object_id_featurizer_embedder import GraphObjectIDFeaturizerEmbedder
from .graph_embedder import GraphEmbedder, ManifoldEmbedding
from .graph_embedding_utils import ManifoldEmbedding
from .manifold_initialization import *
from .neural import ManifoldNetwork
from .optimizer_gen import register_parameter_group

__model = None


def get_model() -> GraphEmbedder:
    """
    Loads graph embedding model based on model config
    """
    global __model

    data = get_training_data()

    if __model is None:
        model_config = get_config().model
        general_config = get_config().general

        embed_manifold = general_config.embed_manifold.get_manifold_instance()

        featurizer, featurizer_dim, featurizer_manifold = get_featurizer()

        if featurizer is None:
            __model = ManifoldEmbedding(
                embed_manifold,
                data.n_nodes(),
                general_config.embed_manifold_dim,
                sparse=model_config.sparse,
                manifold_initialization=
                model_config.manifold_initialization.get_initialization_dict())
            __model.to(get_device())
            register_parameter_group(__model.parameters())
        else:
            __model = GraphObjectIDFeaturizerEmbedder(
                data,
                featurizer,
                get_feature_model(),
                featurizer_manifold,
                featurizer_dim,
                embed_manifold,
                general_config.embed_manifold_dim,
                model_config.train_isometry
            )

    return __model


def get_feature_model():
    featurizer, featurize_dim, in_manifold = get_featurizer()

    general_config = get_config().general
    model_config = get_config().model
    manifold_out = general_config.embed_manifold.get_manifold_instance()
    manifold_out_dim = general_config.embed_manifold_dim

    manifold_initialization = \
        model_config.manifold_initialization.get_initialization_dict()

    intermediate_dims = model_config.intermediate_dims
    intermediate_manifolds = [m.get_manifold_instance() for m in
                              model_config.intermediate_manifolds]
    nonlinearity = model_config.nonlinearity

    device = get_device()

    log_base_inits = []
    log_base_inits.append(
        get_initialized_manifold_tensor(device, torch.float, [1, featurize_dim], in_manifold,
                                        manifold_initialization, requires_grad=True))
    exp_base_inits = []
    manifold_seq = [in_manifold]
    dimension_seq = [featurize_dim]
    for i in range(len(intermediate_manifolds)):
        intermediate_manifold = intermediate_manifolds[i]
        log_base_inits.append(
            get_initialized_manifold_tensor(device, torch.float, [1, intermediate_dims[i]],
                                            intermediate_manifold, manifold_initialization,
                                            requires_grad=True))
        exp_base_inits.append(
            get_initialized_manifold_tensor(device, torch.float, intermediate_dims[i],
                                            intermediate_manifold, manifold_initialization,
                                            requires_grad=True))
        manifold_seq.append(intermediate_manifold)
        dimension_seq.append(intermediate_dims[i])
    exp_base_inits.append(
        get_initialized_manifold_tensor(device, torch.float, manifold_out_dim, manifold_out,
                                        manifold_initialization, requires_grad=True))
    dimension_seq.append(manifold_out_dim)
    manifold_seq.append(manifold_out)
    featurized_model = ManifoldNetwork(manifold_seq, dimension_seq, nonlinearity, 1, log_base_inits,
                                       exp_base_inits)
    featurized_model.to(device)
    register_parameter_group(featurized_model.parameters())
    return featurized_model
