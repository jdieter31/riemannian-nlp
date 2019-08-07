import torch.multiprocessing as mp
import torch

import embed_eval
from embed_save import save_model
import timeit
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from graph_embedding_utils import manifold_dist_loss

def train(
        device,
        model,
        manifold,
        data,
        optimizer,
        n_epochs,
        eval_every,
        lr,
        burnin_num,
        burnin_lr_mult,
        shared_params,
        thread_number,
        tensorboard_dir,
        log_queue,
        log,
        plateau_lr_scheduler=None,
        lr_scheduler=None
        ):

    if thread_number == 0:
        tensorboard_writer = SummaryWriter(tensorboard_dir)

    for epoch in range(1, n_epochs + 1):
        data.burnin = False
        learning_rate = None
        if epoch <= burnin_num:
            data.burnin = True
            learning_rate = lr * burnin_lr_mult

        batch_losses = []
        t_start = timeit.default_timer()
        data_iterator = tqdm(data) if thread_number == 0 else data

        for inputs, targets in data_iterator:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = manifold_dist_loss(model, inputs, targets, manifold)
            loss.backward()
            optimizer.step(lr=learning_rate)
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
            tensorboard_writer.add_scalar('batch_loss', mean_loss, epoch)
            if lr_scheduler is not None:
                tensorboard_writer.add_scalar('learning_rate', lr_scheduler.get_lr()[0], epoch)
            tensorboard_writer._get_file_writer().flush()

            # Output log if main thread
            while not log_queue.empty():
                msg = log_queue.get()
                log.info(msg)

        if plateau_lr_scheduler is not None and epoch > burnin_num:
            plateau_lr_scheduler.step(mean_loss)
        elif lr_scheduler is not None:
            lr_scheduler.step()


    if thread_number == 0:
        tensorboard_writer.close() 
