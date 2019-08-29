import torch.multiprocessing as mp
import torch

import embed_eval
from embed_save import save_model
import timeit
import numpy as np
from tqdm import tqdm
from logging_thread import write_tensorboard
from graph_embedding_utils import manifold_dist_loss, manifold_dist_loss_kl, manifold_dist_loss_relu_sum, metric_loss
from manifold_initialization import initialize_manifold_tensor

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
        lr_scheduler,
        shared_params,
        thread_number,
        feature_manifold,
        conformal_loss_params
        ):

    for epoch in range(1, n_epochs + 1):

        batch_losses = []
        if conformal_loss_params is not None:
            batch_conf_losses = []
        t_start = timeit.default_timer()
        data.refresh_manifold_nn(model.get_embedding_matrix(), manifold)
        data_iterator = tqdm(data) if thread_number == 0 else data

        for batch in data_iterator:
            conf_loss = None
            inputs, graph_dists = batch
            inputs = inputs.to(device)
            graph_dists = graph_dists.to(device)
            optimizer.zero_grad()
            loss = manifold_dist_loss_relu_sum(model, inputs, graph_dists, manifold, **loss_params)
            if hasattr(model, 'embedding_model') and conformal_loss_params is not None and epoch % conformal_loss_params["update_every"] == 0:
                main_inputs = inputs.narrow(1, 0, 1).squeeze(1)
                perm = torch.randperm(main_inputs.size(0))
                idx = perm[:conformal_loss_params["num_samples"]]
                main_inputs = main_inputs[idx]
                conf_loss = metric_loss(model, main_inputs, feature_manifold, manifold, dimension,
                        isometric=conformal_loss_params["isometric"], random_samples=conformal_loss_params["random_samples"],
                        random_init=conformal_loss_params["random_init"])

            if conformal_loss_params is not None and conf_loss is not None:
                total_loss = loss + conformal_loss_params["weight"] * conf_loss
                total_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            batch_losses.append(loss.cpu().detach().numpy())
            if conf_loss is not None:
                batch_conf_losses.append(conf_loss.cpu().detach().numpy())

            elapsed = timeit.default_timer() - t_start

        if (epoch == 1 or epoch % eval_every == 0 or epoch == n_epochs) and thread_number == 0:
            mean_loss = float(np.mean(batch_losses))
            savable_model = model.get_savable_model()
            save_data = {
                'epoch': epoch
            }
            if data.features is not None:
                save_data["features"] = data.features

            save_data.update(shared_params)
            path = save_model(savable_model, save_data)
            embed_eval.evaluate(epoch, elapsed, mean_loss, path)

        mean_loss = float(np.mean(batch_losses))
        if thread_number == 0:
            if conformal_loss_params is not None and len(batch_conf_losses) > 0:
                mean_conf_loss = float(np.mean(batch_conf_losses))
                metric_loss_type = "isometric" if conformal_loss_params["isometric"] else "conformal"
                write_tensorboard('add_scalar', [f'batch_{metric_loss_type}_loss', mean_conf_loss, epoch])
            write_tensorboard('add_scalar', ['batch_loss', mean_loss, epoch])
            write_tensorboard('add_scalar', ['learning_rate', lr_scheduler.get_lr()[0], epoch])

        lr_scheduler.step()
