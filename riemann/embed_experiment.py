import logging
import timeit
import torch

from geoopt import PoincareBall, Euclidean, Sphere
from geoopt.optim import RiemannianAdam, RiemannianSGD
from sacred import Experiment
from sacred.observers import FileStorageObserver

from euclidean_manifold import EuclideanManifold
from manifold_embedding import ManifoldEmbedding

from data.data_ingredient import data_ingredient, load_dataset, get_adjacency_dict
from product_manifold import ProductManifold
from save_ingredient import save_ingredient, save
from eval_ingredient import evaluate, initialize

import numpy as np

ex = Experiment('Embed', ingredients=[data_ingredient, save_ingredient])

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
    n_epochs = 300
    dimension = 15
    manifold_name = "Product"
    eval_every = 10
    gpu = 0
    submanifold_names = ["Poincare", "Euclidean", "Sphere"]
    submanifold_shapes = [[5], [5], [5]]


def get_manifold_from_name(manifold_name):
    manifold = None
    if manifold_name == "Euclidean":
        manifold = EuclideanManifold()
    elif manifold_name == "Poincare":
        manifold = PoincareBall()
    elif manifold_name == "Sphere":
        manifold = Sphere()
    return manifold


@ex.command
def train(n_epochs, manifold_name, dimension, eval_every, gpu, submanifold_names, submanifold_shapes, _log):
    data = load_dataset()
    device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
    initialize(get_adjacency_dict(data), _log)

    if manifold_name == "Euclidean":
        manifold = EuclideanManifold()
    elif manifold_name == "Poincare":
        manifold = PoincareBall()
    elif manifold_name == "Sphere":
        manifold = Sphere()
    elif manifold_name == "Product":
        submanifolds = [get_manifold_from_name(name) for name in submanifold_names]
        manifold = ProductManifold(submanifolds, submanifold_shapes)

    model = ManifoldEmbedding(
        manifold,
        len(data.objects),
        dimension
    )
    model = model.to(device)

    shared_params = {
        "manifold": manifold,
        "dimension": dimension,
        "objects": data.objects
    }

    optimizer = RiemannianAdam(model.parameters())

    for epoch in range(n_epochs):
        batch_losses = []
        t_start = timeit.default_timer()
        for inputs, targets in data:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            loss = model.loss(inputs, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.cpu().detach().numpy())
            elapsed = timeit.default_timer() - t_start

        if epoch % eval_every == (eval_every - 1):
            mean_loss = float(np.mean(batch_losses))
            save_data = {
                'model': model.state_dict(),
                'epoch': epoch
            }
            save_data.update(shared_params)
            path = save(save_data)
            evaluate(epoch, elapsed, mean_loss, path)






