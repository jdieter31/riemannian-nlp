import torch.multiprocessing as mp

from embed_eval import evaluate
from embed_save import save
import timeit
import numpy as np
from tqdm import tqdm

def train(
        device,
        model,
        data,
        optimizer,
        n_epochs,
        eval_every,
        shared_params,
        thread_number,
        log_queue,
        log
        ):
    

    for epoch in range(n_epochs):
        batch_losses = []
        t_start = timeit.default_timer()
        data_iterator = tqdm(data) if thread_number == 0 else data
        
        for inputs, targets in data_iterator:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            loss = model.loss(inputs, targets)
            loss.backward()

            optimizer.step()

            batch_losses.append(loss.cpu().detach().numpy())
            elapsed = timeit.default_timer() - t_start

        if epoch % eval_every == (eval_every - 1) and thread_number == 0:
            mean_loss = float(np.mean(batch_losses))
            save_data = {
                'model': model.state_dict(),
                'epoch': epoch
            }
            save_data.update(shared_params)
            path = save(save_data)
            evaluate(epoch, elapsed, mean_loss, path)
        if thread_number == 0: 
            # Output log if main thread
            while not log_queue.empty():
                msg = log_queue.get()
                log.info(msg)

