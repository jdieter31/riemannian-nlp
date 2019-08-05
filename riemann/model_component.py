from sacred import Ingredient
from graph_embedding_utils import ManifoldEmbedding, FeaturizedModelEmbedding, get_canonical_glove_sentence_featurizer
from manifolds import RiemannianManifold, EuclideanManifold
from manifold_initialization import *
from neural import ManifoldLayer
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
    model_type = "featurized_model"
    sparse = True
    double_precision = True
    manifold_initialization = {
        'PoincareBall': {
            'init_func': 'uniform_',
            'params': [-0.001, 0.001]
        },
        'global': {
            'init_func': 'normal_'
        }
    }
    tries = 10

@model_ingredient.capture
def gen_model(data, device, manifold_out, manifold_out_dim, model_type, sparse, double_precision, manifold_initialization):
    if model_type == "embedding":
        model = ManifoldEmbedding(manifold_out, len(data.objects), manifold_out_dim, sparse=sparse)
        initialize_manifold_tensor(model.weight.data, manifold_out, manifold_initialization)
    elif model_type == "featurized_model":
        features = data.features
        featurizer, featurize_dim = get_canonical_glove_sentence_featurizer()
        in_manifold = EuclideanManifold()
        log_base_init = get_initialized_manifold_tensor(device, torch.double, featurize_dim, in_manifold, manifold_initialization, requires_grad=True)
        exp_base_init = get_initialized_manifold_tensor(device, torch.double, manifold_out_dim, manifold_out, manifold_initialization, requires_grad=True)
        featurized_model = ManifoldLayer(in_manifold, manifold_out, featurize_dim, manifold_out_dim, log_base_init, exp_base_init)
        model = FeaturizedModelEmbedding(featurized_model, features)
    else:
        raise Exception("Improper model configuration")
    model = model.to(device)
    if double_precision:
        model = model.double()
    else:
        model = model.float()
    return model

