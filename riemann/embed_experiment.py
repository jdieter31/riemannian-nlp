import logging
from datetime import datetime

import torch
import torch.multiprocessing as mp
import wandb

from . import embed_eval
from . import logging_thread
from .data.data_ingredient import load_dataset, get_adjacency_dict
from .embed_save import load_model
from .graph_embedding_utils import FeaturizedModelEmbedding
from .lr_schedule import get_lr_scheduler, get_base_lr, \
    get_fixed_embedding_lr
from .manifolds import RiemannianManifold
from .model_component import gen_model
from .rsgd_multithread import RiemannianSGD
from .train import train

# Initialize wandb dashboard
wandb.init(project="retrofitting-manifolds")

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


@ex.config
def config():
    n_epochs = 50
    eval_every = 100
    gpu = 0
    train_threads = 1
    embed_manifold_name = "ProductManifold"
    embed_manifold_dim = 500
    embed_manifold_params = {
        "submanifolds": [
            {
                "name": "PoincareBall",
                "dimension": 90
            },
            {
                "name": "EuclideanManifold",
                "dimension": 320
            },
            {
                "name": "SphericalManifold",
                "dimension": 90
            }
        ]
    }
    sparse = True
    now = datetime.now()
    tensorboard_dir = f"runs/{embed_manifold_name}-{embed_manifold_dim}D-{now:-%m:%d:%Y-%H:%M:%S}"
    loss_params = {
        "margin": 0.0001,
        "discount_factor": 0.5
    }
    conformal_loss_params = {
        "weight": 0.99,
        "num_samples": 15,
        "isometric": True,
        "random_samples": 15,
        "random_init": {
            'global': {
                'init_func': 'normal_',
                'params': [0, 0.06]
            }
        },
        "update_every": 1
    }
    sample_neighbors_every = 1
    resume_training = False


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
        resume_training,
        model,
        _log
):
    model_ingredient_data = model
    device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
    torch.set_num_threads(1)

    logging_thread.initialize(tensorboard_dir, _log)

    curvature_scale = [torch.nn.Parameter(torch.tensor(0.)), torch.nn.Parameter(torch.tensor(0.)),
                       torch.tensor(0., requires_grad=False)]
    embed_manifold_params = embed_manifold_params.copy()
    embed_manifold_params["curvature_scale"] = curvature_scale
    embed_manifold = RiemannianManifold.from_name_params(embed_manifold_name, embed_manifold_params)
    tensorboard_watch = {
        "hyper_scale": curvature_scale[0],
        "sphere_scale": curvature_scale[1]
    }
    data, eval_data = load_dataset(embed_manifold)
    embed_eval.initialize_eval(adjacent_list=get_adjacency_dict(data))
    if resume_training:
        model, save_data = load_model()
        model.to(device)
        if "features" in save_data:
            model = FeaturizedModelEmbedding(model, data.features, save_data["in_manifold"],
                                             embed_manifold, embed_manifold_dim, device=device)
    else:
        model = gen_model(data, device, embed_manifold, embed_manifold_dim)

    if train_threads > 1:
        mp.set_sharing_strategy('file_system')
        model = model.share_memory()

    if model_ingredient_data["input_manifold"] == "Spherical":
        feature_manifold = RiemannianManifold.from_name_params("SphericalManifold", None)
    else:
        feature_manifold = RiemannianManifold.from_name_params("EuclideanManifold", None)

    shared_params = {
        "manifold": embed_manifold,
        "dimension": embed_manifold_dim,
        "objects": data.objects,
        "in_manifold": feature_manifold
    }
    if hasattr(model,
               "get_additional_embeddings") and model.get_additional_embeddings() is not None:
        optimizer = RiemannianSGD([
            {'params': model.get_savable_model().parameters()},
            # {'params': model.main_deltas.parameters(), 'lr':300},
            # {'params': model.additional_deltas.parameters(), 'lr':300},
            # {'params': curvature_scale[:2], 'lr':0.001},
            {'params': model.get_additional_embeddings().parameters(),
             'lr': get_fixed_embedding_lr()}
        ], lr=get_base_lr(), adam_for_euc=False)
        # optimizer = RiemannianSGD(list(model.get_savable_model().parameters()) + list(model.get_additional_embeddings().parameters()) + curvature_scale[1:], lr=get_base_lr(), adam_for_euc=False)
    else:
        optimizer = RiemannianSGD([
            {'params': model.get_savable_model().parameters()}
            # {'params': curvature_scale[:2], 'lr':0.001}
        ], lr=get_base_lr(), adam_for_euc=False)
    lr_scheduler = get_lr_scheduler(optimizer)

    threads = []
    if train_threads > 1:
        try:
            for i in range(train_threads):
                args = [device, model, embed_manifold, embed_manifold_dim, data, optimizer,
                        loss_params, n_epochs, eval_every, sample_neighbors_every, lr_scheduler,
                        shared_params, i, feature_manifold, conformal_loss_params,
                        tensorboard_watch, eval_data]
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
            # embed_eval.close_thread(wait_to_finish=True)
            logging_thread.close_thread(wait_to_finish=True)

    else:
        args = [device, model, embed_manifold, embed_manifold_dim, data, optimizer, loss_params,
                n_epochs, eval_every, sample_neighbors_every, lr_scheduler, shared_params, 0,
                feature_manifold, conformal_loss_params, tensorboard_watch, eval_data]
        try:
            train(*args)
        finally:
            # embed_eval.close_thread(wait_to_finish=True)
            logging_thread.close_thread(wait_to_finish=True)


if __name__ == '__main__':
    ex.run_commandline()
