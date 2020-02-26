from sacred import Ingredient
from .graph_embedding_utils import ManifoldEmbedding, FeaturizedModelEmbedding, get_canonical_glove_sentence_featurizer, get_cn_vector_featurizer
from .manifolds import RiemannianManifold, EuclideanManifold, SphericalManifold
from .manifold_initialization import *
from .neural import ManifoldLayer, ManifoldNetwork
from .embedding.conceptnet.formats import load_hdf
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
              "dimension": 225,
              "name": "PoincareBall"
            },
            {
              "dimension": 225,
              "name": "PoincareBall"
            },
            {
              "dimension": 225,
              "name": "PoincareBall"
            },
            {
              "dimension": 225,
              "name": "PoincareBall"
            },
            {
              "dimension": 225,
              "name": "SphericalManifold"
            },
            {
              "dimension": 225,
              "name": "SphericalManifold"
            },
            {
              "dimension": 225,
              "name": "SphericalManifold"
            },
            {
              "dimension": 225,
              "name": "SphericalManifold"
            },
            {
              "dimension": 900,
              "name": "EuclideanManifold"
            }
          ]
        }
      },
      {
        "name": "ProductManifold",
        "params": {
          "submanifolds": [
            {
              "dimension": 100,
              "name": "PoincareBall"
            },
            {
              "dimension": 100,
              "name": "PoincareBall"
            },
            {
              "dimension": 100,
              "name": "PoincareBall"
            },
            {
              "dimension": 100,
              "name": "SphericalManifold"
            },
            {
              "dimension": 100,
              "name": "SphericalManifold"
            },
            {
              "dimension": 100,
              "name": "SphericalManifold"
            },
            {
              "dimension": 300,
              "name": "EuclideanManifold"
            }
          ]
        }
      }
    ]
    intermediate_manifold_gen_products = None
    intermediate_dims = [2700, 900]
    sparse = True
    double_precision = False
    manifold_initialization = {
        # 'PoincareBall': {
        #     'init_func': 'uniform_',
        #     'params': [-0.001, 0.001]
        # },
        'global': {
            'init_func': 'uniform_',
            'params': [-0.003, 0.003]
        },
        'SphericalManifold': {
            'init_func': 'normal_',
            'params': [0, 1]
        }
    }
    nonlinearity = None
    num_poles = 1
    tries = 10
    num_layers = 2

    featurizer_name = "conceptnet"
    cn_vector_frame_file = "data/glove_w2v_merge.h5"
    input_manifold = "Euclidean"

@model_ingredient.capture
def gen_model(data, device, manifold_out, manifold_out_dim, model_type, sparse, double_precision, manifold_initialization, intermediate_manifolds, intermediate_dims, nonlinearity, num_poles, num_layers, intermediate_manifold_gen_products, featurizer_name, cn_vector_frame_file, input_manifold):
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

