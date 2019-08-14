import torch.multiprocessing as mp
import torch

import embed_eval
from embed_save import save_model
import timeit
import numpy as np
from tqdm import tqdm
from logging_thread import write_tensorboard
from graph_embedding_utils import manifold_dist_loss, manifold_dist_loss_kl, manifold_dist_loss_relu_sum

def train(
        device,
        model,
        manifold,
        data,
        optimizer,
        n_epochs,
        eval_every,
        lr_scheduler,
        burnin_num,
        shared_params,
        thread_number,
        ):

    for epoch in range(1, n_epochs + 1):
        data.burnin = False
        if epoch <= burnin_num:
            data.burnin = True

        batch_losses = []
        t_start = timeit.default_timer()
        data_iterator = tqdm(data) if thread_number == 0 else data

        for batch in data_iterator:
            if data.sample_data == "targets":
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                loss = manifold_dist_loss(model, inputs, targets, manifold)
            elif data.sample_data == "graph_dist":
                inputs, graph_dists = batch
                inputs = inputs.to(device)
                graph_dists = graph_dists.to(device)
                optimizer.zero_grad()
                loss = manifold_dist_loss_relu_sum(model, inputs, graph_dists, manifold)

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.cpu().detach().numpy())
            elapsed = timeit.default_timer() - t_start

        if (epoch == 1 or epoch % eval_every == 0 or epoch == n_epochs) and thread_number == 0:
            mean_loss = float(np.mean(batch_losses))
            savable_model = model.get_savable_model()
            save_data = {
                'embedding_matrix': model.get_embedding_matrix(),
                'epoch': epoch
            }
            save_data.update(shared_params)
            path = save_model(savable_model, save_data)
            embed_eval.evaluate(epoch, elapsed, mean_loss, path)

        mean_loss = float(np.mean(batch_losses))
        if thread_number == 0:
            write_tensorboard('add_scalar', ['batch_loss', mean_loss, epoch])
            write_tensorboard('add_scalar', ['learning_rate', lr_scheduler.get_lr()[0], epoch])

        lr_scheduler.step()
