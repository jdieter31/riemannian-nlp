import torch.multiprocessing as mp
import torch

from . import embed_eval
from .embed_save import save_model
import timeit
import numpy as np
from math import ceil
from tqdm import tqdm
from .logging_thread import write_tensorboard
from .graph_embedding_utils import manifold_dist_loss_relu_sum, metric_loss
from .manifold_initialization import initialize_manifold_tensor
from .manifolds import EuclideanManifold
from .manifold_tensors import ManifoldParameter

import random

# This controls how frequently the whole batch is iterated over vs only
# {QUICK_EVAL_PERCENT} of the data
QUICK_EVAL_FREQUENCY = 10
QUICK_EVAL_PERCENT = 0.05
QUICK_EVAL_TRAIN_PERCENT = 0.025

def train(
        device,
        model,
        manifold,
        dimension,
        data,
        optimizer,
        loss_params,
        n_epochs,
        eval_every,
        sample_neighbors_every,
        lr_scheduler,
        shared_params,
        thread_number,
        feature_manifold,
        conformal_loss_params,
        tensorboard_watch={},
        eval_data=None
        ):

    batch_num = 0
    for epoch in range(1, n_epochs + 1):
        batch_losses = []
        if conformal_loss_params is not None:
            batch_conf_losses = []
        t_start = timeit.default_timer()
        if (epoch - 1) % sample_neighbors_every == 0 and thread_number == 0:
            optimizer.zero_grad()
            inputs = None
            graph_dists = None
            conf_loss = None
            loss = None
            import gc; gc.collect()
            torch.cuda.empty_cache()
            with torch.no_grad():
                model.to(device)
                nns = data.refresh_manifold_nn(model.get_embedding_matrix(), manifold, return_nns=True)
                if eval_data is not None:
                    eval_data.refresh_manifold_nn(model.get_embedding_matrix(), manifold, manifold_nns=nns)

            if epoch > 1:
                syn_acc, sem_acc = embed_eval.eval_analogy(model, manifold, nns)
                write_tensorboard('add_scalar', ['syn_acc', syn_acc, epoch - 1])
                write_tensorboard('add_scalar', ['sem_acc', sem_acc, epoch - 1])

            import gc; gc.collect()
            torch.cuda.empty_cache()

        data_iterator = tqdm(data) if thread_number == 0 else data

        for batch in data_iterator:

            if batch_num % eval_every == 0 and thread_number == 0:
                mean_loss = 0 # float(np.mean(batch_losses)) use to eval every batch setting this to zero as its only used for printing output
                savable_model = model.get_savable_model()
                save_data = {
                    'epoch': epoch
                }
                if data.features is not None:
                    save_data["features"] = data.features
                    if model.get_additional_embeddings() is not None:
                        save_data["additional_embeddings_state_dict"] = model.get_additional_embeddings().state_dict()
                    if hasattr(model, "main_deltas"):
                        save_data["main_deltas_state_dict"] = model.main_deltas.state_dict()
                    if hasattr(model, "additional_deltas"):
                        save_data["additional_deltas_state_dict"] = model.additional_deltas.state_dict()
                    save_data["deltas"] = model.deltas

                save_data.update(shared_params)
                path = save_model(savable_model, save_data)
                elapsed = 0 # Used to eval every batch setting this to zero as its only used for printing output
                embed_eval.evaluate(batch_num, elapsed, mean_loss, path)
                if eval_data is not None:
                    with torch.no_grad():
                        hitsat10 = 0
                        rank_sum = 0
                        rec_rank_sum = 0
                        num_ranks = 0

                        if QUICK_EVAL_FREQUENCY > 0 and batch_num % (eval_every * QUICK_EVAL_FREQUENCY) == 0:
                            eval_data.data_fraction = 1
                            total_eval = True
                        else:
                            eval_data.data_fraction = QUICK_EVAL_PERCENT
                            total_eval = False
                        eval_data.compute_train_ranks = False

                        for eval_batch in tqdm(eval_data):
                            inputs, graph_dists = eval_batch
                            inputs = inputs.to(device)
                            graph_dists = graph_dists.to(device)
                            
                            input_embeddings = model(inputs)

                            sample_vertices = input_embeddings.narrow(1, 1, input_embeddings.size(1)-1)
                            main_vertices = input_embeddings.narrow(1, 0, 1).expand_as(sample_vertices)
                            manifold_dists = manifold.dist(main_vertices, sample_vertices)

                            sorted_indices = manifold_dists.argsort(dim=-1)
                            manifold_dists_sorted = torch.gather(manifold_dists, -1, sorted_indices)
                            n_neighbors = (graph_dists < 2).sum(dim=-1)
                            batch_nums, neighbor_ranks = (sorted_indices < n_neighbors.unsqueeze(1)).nonzero(as_tuple=True)
                            neighbor_ranks += 1

                            adjust_indices = torch.arange(n_neighbors.max())
                            neighbor_adjustements = torch.cat([adjust_indices[:n_neighbors[i]] for i in range(n_neighbors.size(0))])
                            neighbor_ranks -= neighbor_adjustements.to(device)
                            neighbor_ranks = neighbor_ranks.float()
                            rec_ranks = 1 / neighbor_ranks 
                            hitsat10 += (neighbor_ranks <= 10).sum().cpu().numpy()
                            rank_sum += neighbor_ranks.sum().cpu().numpy()
                            rec_rank_sum += rec_ranks.sum().cpu().numpy()
                            num_ranks += neighbor_ranks.size(0)

                        mean_rank = rank_sum / num_ranks
                        mean_rec_rank = rec_rank_sum / num_ranks
                        hitsat10 = hitsat10 / num_ranks

                        postfix = "_approx"
                        if total_eval:
                            postfix = ""

                        write_tensorboard('add_scalar', [f'mean_rank{postfix}', mean_rank, batch_num])
                        write_tensorboard('add_scalar', [f'mean_rec_rank{postfix}', mean_rec_rank, batch_num])
                        write_tensorboard('add_scalar', [f'hitsat10{postfix}', hitsat10, batch_num])

                        if eval_data.is_eval:
                            hitsat10 = 0
                            rank_sum = 0
                            rec_rank_sum = 0
                            num_ranks = 0
                            
                            eval_data.data_fraction = QUICK_EVAL_TRAIN_PERCENT
                            eval_data.compute_train_ranks = True
                            for eval_batch in tqdm(eval_data):
                                inputs, graph_dists = eval_batch
                                inputs = inputs.to(device)
                                graph_dists = graph_dists.to(device)
                                
                                input_embeddings = model(inputs)

                                sample_vertices = input_embeddings.narrow(1, 1, input_embeddings.size(1)-1)
                                main_vertices = input_embeddings.narrow(1, 0, 1).expand_as(sample_vertices)
                                manifold_dists = manifold.dist(main_vertices, sample_vertices)

                                sorted_indices = manifold_dists.argsort(dim=-1)
                                manifold_dists_sorted = torch.gather(manifold_dists, -1, sorted_indices)
                                n_neighbors = (graph_dists < 2).sum(dim=-1)
                                batch_nums, neighbor_ranks = (sorted_indices < n_neighbors.unsqueeze(1)).nonzero(as_tuple=True)
                                neighbor_ranks += 1

                                adjust_indices = torch.arange(n_neighbors.max())
                                neighbor_adjustements = torch.cat([adjust_indices[:n_neighbors[i]] for i in range(n_neighbors.size(0))])
                                neighbor_ranks -= neighbor_adjustements.to(device)
                                neighbor_ranks = neighbor_ranks.float()
                                rec_ranks = 1 / neighbor_ranks 
                                hitsat10 += (neighbor_ranks <= 10).sum().cpu().numpy()
                                rank_sum += neighbor_ranks.sum().cpu().numpy()
                                rec_rank_sum += rec_ranks.sum().cpu().numpy()
                                num_ranks += neighbor_ranks.size(0)

                            mean_rank = rank_sum / num_ranks
                            mean_rec_rank = rec_rank_sum / num_ranks
                            hitsat10 = hitsat10 / num_ranks

                            write_tensorboard('add_scalar', [f'mean_rank_train', mean_rank, batch_num])
                            write_tensorboard('add_scalar', [f'mean_rec_rank_train', mean_rec_rank, batch_num])
                            write_tensorboard('add_scalar', [f'hitsat10_train', hitsat10, batch_num])

            conf_loss = None
            delta_loss = None
            inputs, graph_dists = batch
            inputs = inputs.to(device)
            graph_dists = graph_dists.to(device)
            optimizer.zero_grad()
            
            rand_val = random.random()
            optimizing_model = False
            if hasattr(model, "get_additional_embeddings"):
                if rand_val > 0.7: 
                    optimizing_model = False
                    optimizing_deltas = False
                    model.deltas = False
                    for p in model.parameters():
                        p.requires_grad = False
                    for p in model.get_additional_embeddings().parameters():
                        p.requires_grad = True
                    if model.deltas:
                        for p in model.main_deltas.parameters():
                            p.requires_grad = False
                        if hasattr(model, "additional_deltas"):
                            for p in model.additional_deltas.parameters():
                                p.requires_grad = False
                else:
                    optimizing_model = True 
                    optimizing_deltas = False
                    model.deltas = False
                    for p in model.parameters():
                        p.requires_grad = True
                    for p in model.get_additional_embeddings().parameters():
                        p.requires_grad = False
                    if model.deltas:
                        for p in model.main_deltas.parameters():
                            p.requires_grad = False
                        if hasattr(model, "additional_deltas"):
                            for p in model.additional_deltas.parameters():
                                p.requires_grad = False
            '''
            else:
                optimizing_model = False
                optimizing_deltas = True
                model.deltas = True
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.get_additional_embeddings().parameters():
                    p.requires_grad = False
                if model.deltas:
                    for p in model.main_deltas.parameters():
                        p.requires_grad = True
                    if hasattr(model, "additional_deltas"):
                        for p in model.additional_deltas.parameters():
                            p.requires_grad = True
            '''
            loss = manifold_dist_loss_relu_sum(model, inputs, graph_dists, manifold, **loss_params)

            if optimizing_model and hasattr(model, 'embedding_model') and conformal_loss_params is not None and epoch % conformal_loss_params["update_every"] == 0:
                main_inputs = inputs.narrow(1, 0, 1).squeeze(1).clone().detach()
                perm = torch.randperm(main_inputs.size(0))
                idx = perm[:conformal_loss_params["num_samples"]]
                main_inputs = main_inputs[idx]
                conf_loss = metric_loss(model, main_inputs, feature_manifold, manifold, dimension,
                        isometric=conformal_loss_params["isometric"], random_samples=conformal_loss_params["random_samples"],
                        random_init=conformal_loss_params["random_init"])

            if hasattr(model, 'main_deltas') and optimizing_deltas:
                main_inputs = inputs.view(inputs.shape[0], -1)
                vals = model.main_deltas(model.index_map[main_inputs][model.index_map[main_inputs] >= 0])
                mean_deltas = torch.mean(torch.norm(vals, dim=-1))
                delta_loss = 800 * torch.mean(torch.norm(vals, dim=-1) ** 2)

            total_loss = None
            if conformal_loss_params is not None and conf_loss is not None:
                total_loss = (1 - conformal_loss_params["weight"]) * loss + conformal_loss_params["weight"] * conf_loss
                if delta_loss is not None:
                    #   total_loss += delta_loss
                    pass
                total_loss.backward()
            else:
                if conformal_loss_params is not None:
                    scaled_loss = (1 - conformal_loss_params["weight"]) * loss
                else:
                    scaled_loss = loss

                if delta_loss is not None:
                    scaled_loss += delta_loss
                scaled_loss.backward()


            optimizer.step()
            batch_losses.append(loss.cpu().detach().numpy())
            if thread_number == 0:
                write_tensorboard('add_scalar', ['minibatch_loss', float(batch_losses[-1]), batch_num])
                if total_loss is not None:
                    write_tensorboard('add_scalar', ['minibatch_total_loss', total_loss.cpu().detach().numpy(), batch_num])

                if delta_loss is not None:
                    write_tensorboard('add_scalar', ['minibatch_delta_loss', delta_loss.cpu().detach().numpy(), batch_num])
                    write_tensorboard('add_scalar', ['minibatch_delta_mean', mean_deltas.cpu().detach().numpy(), batch_num])

                for name, value in tensorboard_watch.items():
                    write_tensorboard('add_scalar', [name, value.cpu().detach().numpy(), batch_num])

            if conf_loss is not None:
                batch_conf_losses.append(conf_loss.cpu().detach().numpy())
                if thread_number == 0:
                    write_tensorboard('add_scalar', ['minibatch_conf_loss', float(batch_conf_losses[-1]), batch_num])


            elapsed = timeit.default_timer() - t_start
            batch_num += 1



        mean_loss = float(np.mean(batch_losses))
        if thread_number == 0:
            if conformal_loss_params is not None and len(batch_conf_losses) > 0:
                mean_conf_loss = float(np.mean(batch_conf_losses))
                metric_loss_type = "isometric" if conformal_loss_params["isometric"] else "conformal"
                write_tensorboard('add_scalar', [f'batch_{metric_loss_type}_loss', mean_conf_loss, epoch])
            write_tensorboard('add_scalar', ['batch_loss', mean_loss, epoch])
            write_tensorboard('add_scalar', ['learning_rate', lr_scheduler.get_lr()[0], epoch])

        lr_scheduler.step()
