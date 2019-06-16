import logging
import timeit
import torch

from geoopt import PoincareBall, Sphere, Euclidean, Product
from sacred import Experiment
from sacred.observers import FileStorageObserver

from manifold_embedding import ManifoldEmbedding

from data.data_ingredient import data_ingredient, load_dataset, get_adjacency_dict
from embed_save import save_ingredient, save
from embed_eval import evaluate, initialize_eval
from train import train
from manifold_initialization import initialization_ingredient, apply_initialization

from rsgd_multithread import RiemannianSGD

from torch.distributions import uniform

import numpy as np

import torch.multiprocessing as mp

ex = Experiment('Embed', ingredients=[data_ingredient, save_ingredient, initialization_ingredient])

ex.observers.append(FileStorageObserver.create("experiments"))

logger = logging.getLogger('Embeddings')
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

ex.logger = logger

@ex.config
def config():
    n_epochs = 400
    dimension = 5
    manifold_name = "Product"
    eval_every = 10
    gpu = -1
    train_threads = 4
    submanifold_names = ["PoincareBall", "Sphere"]
    double_precision = True
    submanifold_shapes = [[3], [2]]
    learning_rate = 0.3

@ex.capture
def get_embed_manifold(manifold_name, submanifold_names=None, submanifold_shapes=None):
    manifold = None
    if manifold_name == "Euclidean":
        manifold = Euclidean()
    elif manifold_name == "PoincareBall":
        manifold = PoincareBall()
    elif manifold_name == "Sphere":
        manifold = Sphere()
    elif manifold_name == "Product":
        submanifolds = [get_embed_manifold(name) for name in submanifold_names]
        manifold = Product(submanifolds, submanifold_shapes)
    return manifold

    
@ex.command
def embed(n_epochs, dimension, eval_every, gpu, train_threads, double_precision, learning_rate, _log):
    data = load_dataset()
    device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')

    log_queue = mp.Queue()
    initialize_eval(get_adjacency_dict(data), log_queue)
    
    manifold = get_embed_manifold()

    model = ManifoldEmbedding(
        manifold,
        len(data.objects),
        dimension
    )
    model = model.to(device)
    if double_precision:
        model = model.double()
    
    apply_initialization(model.weight, manifold) 
    model.weight.proj_()

    shared_params = {
        "manifold": manifold,
        "dimension": dimension,
        "objects": data.objects,
        "double_precision": double_precision
    }

    optimizer = RiemannianSGD(model.parameters(), lr=learning_rate, manifold=manifold)

    threads = []
    if train_threads > 1:
        model = model.share_memory()
        for i in range(train_threads):
            args = [device, model, data, optimizer, n_epochs, eval_every, shared_params, i, log_queue, _log]
            threads.append(mp.Process(target=train, args=args))
            threads[-1].start()
    else:
        args = [device, model, data, optimizer, n_epochs, eval_every, shared_params, 0, log_queue, _log]
        train_process = mp.Process(target=train, args=args)
        train_process.start()
        threads = [train_process]
    for thread in threads:
        thread.join()

    





