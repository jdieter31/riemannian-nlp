import logging
import timeit
import torch

from manifolds import RiemannianManifold, EuclideanManifold, SphericalManifold, ProductManifold, PoincareBall

from sacred import Experiment
from sacred.observers import FileStorageObserver

from data.data_ingredient import data_ingredient, load_dataset, get_adjacency_dict
from embed_save import save_ingredient
from embed_eval import eval_ingredient
import embed_eval
from model_component import model_ingredient, gen_model
from train import train
import logging_thread
from rsgd_multithread import RiemannianSGD

from lr_schedule import lr_schedule_ingredient, get_lr_scheduler, get_base_lr
from torch.distributions import uniform
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import numpy as np

import torch.multiprocessing as mp
from datetime import datetime

ex = Experiment('Embed', ingredients=[eval_ingredient, data_ingredient, save_ingredient, model_ingredient, lr_schedule_ingredient])

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
    n_epochs = 10000
    eval_every = 100
    gpu = 0
    train_threads = 1
    embed_manifold_name = "SphericalManifold"
    embed_manifold_dim = 600
    embed_manifold_params = {
       "submanifolds": [
            {
                "name" : "PoincareBall",
                "dimension" : 200
            },
            {
                "name" : "SphericalManifold",
                "dimension" : 200
            },
            {
                "name": "EuclideanManifold",
                "dimension" : 200
            }
        ]
    }
    sparse = True
    now = datetime.now()
    tensorboard_dir = f"runs/{embed_manifold_name}-{embed_manifold_dim}D"
    tensorboard_dir += now.strftime("-%m:%d:%Y-%H:%M:%S")
    loss_params = {
        "margin": 0.05,
        "discount_factor": 0.5
    }
    conformal_loss_params = {
        "weight": 0.1,
        "num_samples": 3,
        "isometric": False,
        "random_samples": 3,
        "random_init": {
            'global': {
                'init_func': 'normal_',
                'params': [0, 1]
            }
        },
        "update_every": 1
    }
    sample_neighbors_every = 5


@ex.command
def embed(
        n_epochs,
        eval_every,
        gpu,
        train_threads,
        sparse,
        tensorboard_dir,
        embed_manifold_name,
        embed_manifold_dim,
        embed_manifold_params,
        loss_params,
        conformal_loss_params,
        sample_neighbors_every,
        _log
        ):

    device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
    torch.set_num_threads(1)

    logging_thread.initialize(tensorboard_dir, _log)

    embed_manifold = RiemannianManifold.from_name_params(embed_manifold_name, embed_manifold_params)
    data = load_dataset(embed_manifold)
    embed_eval.initialize_eval(adjacent_list=get_adjacency_dict(data))

    model = gen_model(data, device, embed_manifold, embed_manifold_dim)
    if train_threads > 1:
        mp.set_sharing_strategy('file_system')
        model = model.share_memory()

    feature_manifold = RiemannianManifold.from_name_params("SphericalManifold", None)

    shared_params = {
        "manifold": embed_manifold,
        "objects": data.objects,
        "in_manifold": feature_manifold
    }
    
    optimizer = RiemannianSGD(model.parameters(), lr=get_base_lr())
    lr_scheduler = get_lr_scheduler(optimizer)
    
    threads = []
    if train_threads > 1:
        try:
            for i in range(train_threads):
                args = [device, model, embed_manifold, embed_manifold_dim, data, optimizer, loss_params, n_epochs, eval_every, sample_neighbors_every, lr_scheduler, shared_params, i, feature_manifold, conformal_loss_params]
                threads.append(mp.Process(target=train, args=args))
                threads[-1].start()

            for thread in threads:
                thread.join()
        finally:
            for thread in threads:
                try:
                    thread.close()
                except:
                    thread.terminate()
            embed_eval.close_thread(wait_to_finish=True)
            logging_thread.close_thread(wait_to_finish=True)

    else:
        args = [device, model, embed_manifold, embed_manifold_dim, data, optimizer, loss_params, n_epochs, eval_every, sample_neighbors_every, lr_scheduler, shared_params, 0, feature_manifold, conformal_loss_params]
        try:
            train(*args)
        finally:
            embed_eval.close_thread(wait_to_finish=True)
            logging_thread.close_thread(wait_to_finish=True)
    
if __name__ == '__main__':
    ex.run_commandline()

