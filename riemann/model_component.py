from sacred import Ingredient
from graph_embedding_utils import ManifoldEmbedding, FeaturizedModelEmbedding, get_canonical_glove_sentence_featurizer
from manifolds import RiemannianManifold, EuclideanManifold
from manifold_initialization import *
from neural import ManifoldLayer, ManifoldNetwork
import random
import time
import torch
import os

model_ingredient = Ingredient("model")

@model_ingredient.config
def config():
    path = "models/model"
    i = 1
    while os.path.isfile(path + f"{i}.tch"):
        i += 1
    path += f"{i}.tch"
    model_type = "featurized_model_manifold_network"
    intermediate_manifolds = [
        {
            "name": "ProductManifold",
            "params": {
                "submanifolds": [
                    {
                        "name": "PoincareBall",
                        "tensor_shape": [10]
                    },

                    {
                        "name": "PoincareBall",
                        "tensor_shape": [20]
                    },

                    {
                        "name": "EuclideanManifold",
                        "tensor_shape": [20]
                    }
                ]
            }
        },
        {
            "name": "ProductManifold",
            "params": {
                "submanifolds": [
                    {
                        "name": "PoincareBall",
                        "tensor_shape": [10]
                    },

                    {
                        "name": "EuclideanManifold",
                        "tensor_shape": [20]
                    },

                    {
                        "name": "PoincareBall",
                        "tensor_shape": [20]
                    }
                ]
            }
        }
    ]
    intermediate_dims = [50, 50]
    sparse = True
    double_precision = False
    manifold_initialization = {
        'PoincareBall': {
            'init_func': 'uniform_',
            'params': [-0.001, 0.001]
        },
        'global': {
            'init_func': 'normal_',
            'params': [0, 0.001]
        }
    }
    tries = 10

@model_ingredient.capture
def gen_model(data, device, manifold_out, manifold_out_dim, model_type, sparse, double_precision, manifold_initialization, intermediate_manifolds, intermediate_dims):
    torch_dtype = torch.double if double_precision else torch.float
    if model_type == "embedding":
        model = ManifoldEmbedding(manifold_out, len(data.objects), manifold_out_dim, sparse=sparse)
        initialize_manifold_tensor(model.weight.data, manifold_out, manifold_initialization)
    elif model_type == "featurized_model_manifold_logistic":
        features = data.features
        featurizer, featurize_dim = get_canonical_glove_sentence_featurizer()
        in_manifold = EuclideanManifold()
        log_base_init = get_initialized_manifold_tensor(device, torch_dtype, featurize_dim, in_manifold, manifold_initialization, requires_grad=True)
        exp_base_init = get_initialized_manifold_tensor(device, torch_dtype, manifold_out_dim, manifold_out, manifold_initialization, requires_grad=True)
        featurized_model = ManifoldLayer(in_manifold, manifold_out, featurize_dim, manifold_out_dim, log_base_init, exp_base_init)
        model = FeaturizedModelEmbedding(featurized_model, features)
    elif model_type == "featurized_model_manifold_network":
        features = data.features
        featurizer, featurize_dim = get_canonical_glove_sentence_featurizer()
        in_manifold = EuclideanManifold()
        log_base_inits = []
        log_base_inits.append(get_initialized_manifold_tensor(device, torch_dtype, featurize_dim, in_manifold, manifold_initialization, requires_grad=True))
        exp_base_inits = []
        manifold_seq = [in_manifold]
        dimension_seq = [featurize_dim]
        for i in range(len(intermediate_manifolds)):
            intermediate_manifold = RiemannianManifold.from_name_params(intermediate_manifolds[i]["name"], intermediate_manifolds[i]["params"])
            log_base_inits.append(get_initialized_manifold_tensor(device, torch_dtype, intermediate_dims[i], intermediate_manifold, manifold_initialization, requires_grad=True))
            exp_base_inits.append(get_initialized_manifold_tensor(device, torch_dtype, intermediate_dims[i], intermediate_manifold, manifold_initialization, requires_grad=True))
            manifold_seq.append(intermediate_manifold)
            dimension_seq.append(intermediate_dims[i])
        exp_base_inits.append(get_initialized_manifold_tensor(device, torch_dtype, manifold_out_dim, manifold_out, manifold_initialization, requires_grad=True))
        dimension_seq.append(manifold_out_dim)
        manifold_seq.append(manifold_out)
        featurized_model = ManifoldNetwork(manifold_seq, dimension_seq, log_base_inits, exp_base_inits)
        model = FeaturizedModelEmbedding(featurized_model, features)
    else:
        raise Exception("Improper model configuration")
    model = model.to(device)
    if double_precision:
        model = model.double()
    else:
        model = model.float()
    return model

