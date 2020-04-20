from .graph_embedding_utils import ManifoldEmbedding, FeaturizedModelEmbedding, get_canonical_glove_sentence_featurizer, get_cn_vector_featurizer
from .manifolds import RiemannianManifold, EuclideanManifold, SphericalManifold
from .manifold_initialization import *
from .neural import ManifoldLayer, ManifoldNetwork
from .embedding.conceptnet.formats import load_hdf
from .config.config_loader import get_config
from .graph_embedder import GraphEmbedder, ManifoldEmbedding
from .data.graph_dataset import GraphDataset
from .optimizer_gen import register_parameter_group
import random
import time
import torch
import os

__model = None

def get_model(data: GraphDataset) -> GraphEmbedder:
    """
    Loads graph embedding model based on model config
    """
    global __model

    if __model is None:
        model_config = get_config().model
        general_config = get_config().general

        embed_manifold = general_config.embed_manifold.get_manifold_instance()
        __model = ManifoldEmbedding(
            embed_manifold, 
            data.n_nodes(),
            general_config.embed_manifold_dim,
            sparse=model_config.sparse,
            manifold_initialization=
            model_config.manifold_initialization.get_initialization_dict())
        register_parameter_group(__model.parameters())

    return __model


# TODO(jdieter): is this function still necessary, or is it subsumed by `get_model`? It doesn't
#                seem to get called anywhere?
def gen_model(data, device, manifold_out, manifold_out_dim, model_type, sparse, double_precision, manifold_initialization, intermediate_manifolds, intermediate_dims, nonlinearity, num_poles, num_layers, intermediate_manifold_gen_products, featurizer_name, cn_vector_frame_file, input_manifold):
    model_config = get_config().model
    
    intermediate_manifolds = intermediate_manifolds[:num_layers]
    intermediate_dims = intermediate_dims[:num_layers]
    if intermediate_manifold_gen_products is not None:
        if intermediate_manifold_gen_products == "ProductManifold":
            intermediate_manifolds = [{
                "name": intermediate_manifold_gen_products,
                "params": {
                    "submanifolds": [
                        {
                            "name": "PoincareBall",
                            "dimension": dim//3
                        },
                        {
                            "name": "SphericalManifold",
                            "dimension": dim//3
                        },
                        {
                            "name": "EuclideanManifold",
                            "dimension": dim//3 + dim % 3
                        }
                    ]
                    }
                } for dim in intermediate_dims]
        else:
            intermediate_manifolds = [{
                "name": intermediate_manifold_gen_products,
                "params": None
                } for _ in intermediate_dims]

    torch_dtype = torch.double if double_precision else torch.float
    if num_layers == 0:
        model_type = "featurized_model_manifold_logistic"
    if model_type == "embedding":
        model = ManifoldEmbedding(manifold_out, len(data.objects), manifold_out_dim, sparse=sparse)
        initialize_manifold_tensor(model.weight.data, manifold_out, manifold_initialization)
    elif model_type == "featurized_model_manifold_logistic":
        features = data.features
        if featurizer_name == "conceptnet":
            frame = load_hdf(cn_vector_frame_file)
            featurizer, featurize_dim = get_cn_vector_featurizer(cn_vector_frame_file)
        else:
            featurizer, featurize_dim = get_canonical_glove_sentence_featurizer()
        if input_manifold == "Spherical":
            in_manifold = SphericalManifold()
        else:
            in_manifold = EuclideanManifold()
            
        log_base_init = get_initialized_manifold_tensor(device, torch_dtype, [num_poles, featurize_dim], in_manifold, manifold_initialization, requires_grad=True)
        exp_base_init = get_initialized_manifold_tensor(device, torch_dtype, manifold_out_dim, manifold_out, manifold_initialization, requires_grad=True)
        featurized_model = ManifoldLayer(in_manifold, manifold_out, featurize_dim, manifold_out_dim, None, num_poles, log_base_init, exp_base_init)
        model = FeaturizedModelEmbedding(featurized_model, features, in_manifold, manifold_out, manifold_out_dim, device=device, manifold_initialization=manifold_initialization, featurizer=featurizer, featurizer_dim=featurize_dim)
    elif model_type == "featurized_model_manifold_network":
        features = data.features
        if featurizer_name == "conceptnet":
            featurizer, featurize_dim = get_cn_vector_featurizer(cn_vector_frame_file)
        else:
            featurizer, featurize_dim = get_canonical_glove_sentence_featurizer()
        if input_manifold == "Spherical":
            in_manifold = SphericalManifold()
        else:
            in_manifold = EuclideanManifold()
        log_base_inits = []
        log_base_inits.append(get_initialized_manifold_tensor(device, torch_dtype, [num_poles, featurize_dim], in_manifold, manifold_initialization, requires_grad=True))
        exp_base_inits = []
        manifold_seq = [in_manifold]
        dimension_seq = [featurize_dim]
        for i in range(len(intermediate_manifolds)):
            intermediate_manifold = RiemannianManifold.from_name_params(intermediate_manifolds[i]["name"], intermediate_manifolds[i]["params"])
            log_base_inits.append(get_initialized_manifold_tensor(device, torch_dtype, [num_poles, intermediate_dims[i]], intermediate_manifold, manifold_initialization, requires_grad=True))
            exp_base_inits.append(get_initialized_manifold_tensor(device, torch_dtype, intermediate_dims[i], intermediate_manifold, manifold_initialization, requires_grad=True))
            manifold_seq.append(intermediate_manifold)
            dimension_seq.append(intermediate_dims[i])
        exp_base_inits.append(get_initialized_manifold_tensor(device, torch_dtype, manifold_out_dim, manifold_out, manifold_initialization, requires_grad=True))
        dimension_seq.append(manifold_out_dim)
        manifold_seq.append(manifold_out)
        featurized_model = ManifoldNetwork(manifold_seq, dimension_seq, nonlinearity, num_poles, log_base_inits, exp_base_inits)
        model = FeaturizedModelEmbedding(featurized_model, features, in_manifold, manifold_out, manifold_out_dim, device=device, manifold_initialization=manifold_initialization, featurizer=featurizer, featurizer_dim=featurize_dim)
    else:
        raise Exception("Improper model configuration")
    model = model.to(device)
    if double_precision:
        model = model.double()
    else:
        model = model.float()
    return model
