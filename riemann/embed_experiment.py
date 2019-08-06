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

from rsgd_multithread import RiemannianSGD

from torch.distributions import uniform
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import numpy as np

import torch.multiprocessing as mp
from datetime import datetime

ex = Experiment('Embed', ingredients=[eval_ingredient, data_ingredient, save_ingredient, model_ingredient])

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
    n_epochs = 100
    eval_every = 10
    gpu = -1
    train_threads = 1
    embed_manifold_name = "PoincareBall"
    embed_manifold_dim = 5
    embed_manifold_params = None
    '''
    embed_manifold_params = {
       "submanifolds": [
            {
                "name" : "PoincareBall",
                "tensor_shape" : [10]
            },
            {
                "name" : "SphericalManifold",
                "tensor_shape" : [10]
            },
            {
                "name" : "EuclideanManifold",
                "tensor_shape" : [10]
            }
        ]
    }'''
    double_precision = True
    learning_rate = 0.003
    sparse = True
    burnin_num = 10
    burnin_lr_mult = 0.1
    burnin_neg_multiplier = 0.1
    now = datetime.now()
    tensorboard_dir = f"runs/{embed_manifold_name}-{embed_manifold_dim}D"
    use_plateau_lr_scheduler = False
    plateau_lr_scheduler_factor = 0.1
    plateau_lr_scheduler_patience = 2
    plateau_lr_scheduler_verbose = True
    plateau_lr_scheduler_threshold = 0.4
    plateau_lr_scheduler_min_lr = 0.1
    use_lr_scheduler = False 
    scheduled_lrs = [1] + list(np.geomspace(0.01, 10, num=40))
    scheduled_lr_epochs = [10] + [1 for _ in range(39)]
    use_lr_func = False 
    lr_func_name = "linear-values-[0.01, 10, 1]-epochs-[10, 30, 100]"
    def linear_func(epoch):
        if epoch < 10:
            return 1
        elif epoch < 30:
            return 0.01 * (30 - (epoch + 1))/20 + 10 * ((epoch + 1) - 10)/20
        elif epoch < 100:
            return 10 * (100 - (epoch + 1))/70 + 1 * ((epoch + 1) - 30)/70
        else:
            return 1
    lr_func = linear_func
    
    if use_lr_scheduler:
        if use_lr_func:
            tensorboard_dir += f"-LRFunc{lr_func_name}"
        else:
            tensorboard_dir += f"-LRSched{len(scheduled_lrs)}"
    else:
        tensorboard_dir += f"-LR{learning_rate}"
    tensorboard_dir += now.strftime("-%m:%d:%Y-%H:%M:%S")


@ex.command
def embed(n_epochs, eval_every, gpu, train_threads, learning_rate, burnin_num, burnin_lr_mult, burnin_neg_multiplier, sparse, tensorboard_dir, use_plateau_lr_scheduler, plateau_lr_scheduler_factor, plateau_lr_scheduler_patience, plateau_lr_scheduler_verbose, plateau_lr_scheduler_threshold, plateau_lr_scheduler_min_lr, use_lr_scheduler, scheduled_lrs, scheduled_lr_epochs, lr_func, use_lr_func, embed_manifold_name, embed_manifold_dim, embed_manifold_params, _log):
    data = load_dataset(burnin=burnin_num > 0)
    if burnin_num > 0:
        data.neg_multiplier = burnin_neg_multiplier

    device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
    torch.set_num_threads(1)

    log_queue = mp.Queue()
    embed_eval.initialize_eval(adjacent_list=get_adjacency_dict(data), log_queue_=log_queue, tboard_dir=tensorboard_dir)

    embed_manifold = RiemannianManifold.from_name_params(embed_manifold_name, embed_manifold_params)
    model = gen_model(data, device, embed_manifold, embed_manifold_dim)
    if train_threads > 1:
        mp.set_sharing_strategy('file_system')
        model = model.share_memory()

    shared_params = {
        "manifold": embed_manifold,
        "objects": data.objects
    }

    optimizer = RiemannianSGD(model.parameters(), lr=learning_rate)
    plateau_lr_scheduler = None
    lr_scheduler = None
    if use_plateau_lr_scheduler:
        plateau_lr_scheduler = ReduceLROnPlateau(
            optimizer,
            factor=plateau_lr_scheduler_factor,
            patience=plateau_lr_scheduler_patience,
            verbose=plateau_lr_scheduler_verbose,
            threshold=plateau_lr_scheduler_threshold,
            min_lr=plateau_lr_scheduler_min_lr
        )
    elif use_lr_scheduler:
        if not use_lr_func:
            global lrs 
            lrs = scheduled_lrs
            global epoch_sched
            epoch_sched = scheduled_lr_epochs
            def return_lr(epochs):
                i = 0
                sum_epochs = 0
                for i in range(len(epoch_sched)):
                    sum_epochs += epoch_sched[i]
                    if epochs < sum_epochs:
                        break
                if epochs >= sum_epochs:
                    i += 1

                return lrs[i]
            lr_func = return_lr
        lr_scheduler = LambdaLR(optimizer, lr_func)

    threads = []
    if train_threads > 1:
        try:
            for i in range(train_threads):
                args = [device, model, embed_manifold, data, optimizer, n_epochs, eval_every, learning_rate, burnin_num, burnin_lr_mult, shared_params, i, tensorboard_dir, log_queue, _log, plateau_lr_scheduler, lr_scheduler]
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

    else:
        args = [device, model, embed_manifold, data, optimizer, n_epochs, eval_every, learning_rate, burnin_num, burnin_lr_mult, shared_params, 0, tensorboard_dir, log_queue, _log, plateau_lr_scheduler, lr_scheduler]
        try:
            train(*args)
        finally:
            embed_eval.close_thread(wait_to_finish=True)

    
    while not log_queue.empty():
        msg = log_queue.get()
        _log.info(msg)

    
if __name__ == '__main__':
    ex.run_commandline()

