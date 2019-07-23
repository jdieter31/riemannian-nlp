import logging
import timeit
import torch

from geoopt import PoincareBall, Sphere, Euclidean, Product
from sacred import Experiment
from sacred.observers import FileStorageObserver

from manifold_embedding import ManifoldEmbedding

from data.data_ingredient import data_ingredient, load_dataset, get_adjacency_dict
from embed_save import save_ingredient, save, load
from embed_eval import eval_ingredient
import embed_eval
from train import train
from manifold_initialization import initialization_ingredient, apply_initialization

from rsgd_multithread import RiemannianSGD

from torch.distributions import uniform
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import numpy as np

import torch.multiprocessing as mp
from datetime import datetime

ex = Experiment('Embed', ingredients=[eval_ingredient, data_ingredient, save_ingredient, initialization_ingredient])

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
    dimension = 20
    manifold_name = "PoincareBall"
    eval_every = 10
    gpu = -1
    train_threads = 65
    submanifold_names = ["PoincareBall", "PoincareBall", "Euclidean"]
    double_precision = True
    submanifold_shapes = [[15], [15], [20]]
    learning_rate = 1
    sparse = True
    burnin_num = 10
    burnin_lr_mult = 0.01
    burnin_neg_multiplier = 0.1
    now = datetime.now()
    tensorboard_dir = f"runs/{manifold_name}-{dimension}D"
    if manifold_name == "Product":
        tensorboard_dir += f"-Subs[{','.join([sub_name for sub_name in submanifold_names])}]"
    use_plateau_lr_scheduler = False
    plateau_lr_scheduler_factor = 0.1
    plateau_lr_scheduler_patience = 2
    plateau_lr_scheduler_verbose = True
    plateau_lr_scheduler_threshold = 0.4
    plateau_lr_scheduler_min_lr = 0.1
    use_lr_scheduler = True
    scheduled_lrs = [1] + list(np.geomspace(0.01, 10, num=40)) 
    scheduled_lr_epochs = [10] + [1 for _ in range(39)]
    use_lr_func = True
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
        manifold = Product(submanifolds, np.array(submanifold_shapes))   
    return manifold
 
@ex.command
def cli_search():
    print("Loading model...")
    model, objects = load() 
    embeddings = model.weight.data
    while True:
        print("Input a word to search near neighbors (or type 'quit')")
        search_q = input("--> ")
        if search_q == "quit":
            return
        if not search_q in objects:
            print("Search query not found in embeddings!")
            continue
        k = -1
        while k<0:
            print("How many neighbors to list?")
            try:
                k = int(input("--> "))
            except:
                print("Must be valid integer")
        q_index = objects.index(search_q)
        dists = model.manifold.dist(embeddings[None, q_index], embeddings)
        sorted_dists, sorted_indices = dists.sort()
        sorted_objects = [objects[index] for index in sorted_indices]
        for i in range(k):
            print(f"{sorted_objects[i]} - dist: {sorted_dists[i]}")

@ex.command
def embed(n_epochs, dimension, eval_every, gpu, train_threads, double_precision, learning_rate, burnin_num, burnin_lr_mult, burnin_neg_multiplier, sparse, tensorboard_dir, use_plateau_lr_scheduler, plateau_lr_scheduler_factor, plateau_lr_scheduler_patience, plateau_lr_scheduler_verbose, plateau_lr_scheduler_threshold, plateau_lr_scheduler_min_lr, use_lr_scheduler, scheduled_lrs, scheduled_lr_epochs, lr_func, use_lr_func, _log):
    data = load_dataset(burnin=burnin_num > 0)
    if burnin_num > 0:
        data.neg_multiplier = burnin_neg_multiplier

    device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
    torch.set_num_threads(1)

    log_queue = mp.Queue()
    embed_eval.initialize_eval(adjacent_list=get_adjacency_dict(data), log_queue_=log_queue, tboard_dir=tensorboard_dir)
    
    manifold = get_embed_manifold()

    model = ManifoldEmbedding(
        manifold,
        len(data.objects),
        dimension,
        sparse=sparse
    )
    if train_threads > 1:
        mp.set_sharing_strategy('file_system')
        model = model.share_memory()

    model = model.to(device)
    if double_precision:
        model = model.double()
    else:
        model = model.float()
    
    apply_initialization(model.weight.data, manifold)
    with torch.no_grad():
        manifold._projx(model.weight.data)

    shared_params = {
        "manifold": manifold,
        "dimension": dimension,
        "objects": data.objects,
        "double_precision": double_precision
    }

    optimizer = RiemannianSGD(model.parameters(), lr=learning_rate, manifold=manifold)
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
                args = [device, model, data, optimizer, n_epochs, eval_every, learning_rate, burnin_num, burnin_lr_mult, shared_params, i, tensorboard_dir, log_queue, _log, plateau_lr_scheduler, lr_scheduler]
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
        args = [device, model, data, optimizer, n_epochs, eval_every, learning_rate, burnin_num, burnin_lr_mult, shared_params, 0, tensorboard_dir, log_queue, _log, plateau_lr_scheduler, lr_scheduler]
        try:
            train(*args)
        finally:
            embed_eval.close_thread(wait_to_finish=True)

    
    while not log_queue.empty():
        msg = log_queue.get()
        _log.info(msg)

    
if __name__ == '__main__':
    ex.run_commandline()

