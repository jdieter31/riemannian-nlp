from sacred import Ingredient
import torch.multiprocessing as mp

import json
import torch

from data.graph import eval_reconstruction

from embed_save import load_model

from tensorboard_thread import write_tensorboard

eval_ingredient = Ingredient('evaluation')

eval_queue = None
log_queue = None
process = None

def async_eval(adj, log_queue, num_workers):
    global eval_queue
    if eval_queue is None:
        return
    finished = False

    while True:
        if finished and eval_queue.empty():
            return

        temp = eval_queue.get()
        if temp is None:
            return
        
        if temp == "finish":
            finished=True
            continue

        epoch, elapsed, loss, path = temp
        model, save_data = load_model(path)
        objects = save_data["objects"]
        manifold = save_data["manifold"]
        embeddings = save_data["embedding_matrix"]
        meanrank, maprank = eval_reconstruction(adj, embeddings, manifold.dist, workers=num_workers)

        write_tensorboard('add_scalar', ['mean_rank', meanrank, epoch])
        write_tensorboard('add_scalar', ['map_rank', maprank, epoch])
        write_tensorboard('add_embedding', [embeddings.cpu().detach().numpy(), objects, None, epoch, f"epoch-{epoch}"])

        lmsg = {
            'epoch': epoch,
            'elapsed': elapsed,
            'loss': loss,
            'mean_rank': meanrank,
            'map_rank': maprank
        }
        log_queue.put(f"Stats: {json.dumps(lmsg)}")

@eval_ingredient.config
def config():
    eval_workers = 5

@eval_ingredient.capture
def initialize_eval(eval_workers, adjacent_list, log_queue_):
    global log_queue
    log_queue = log_queue_
    global eval_queue
    eval_queue = mp.Queue()
    global process
    process = mp.Process(target=async_eval, args=(adjacent_list, log_queue_, eval_workers))
    process.start()

@eval_ingredient.capture
def evaluate(epoch, elapsed, loss, path):
    global eval_queue
    eval_queue.put((epoch, elapsed, loss, path))

def close_thread(wait_to_finish=False):
    global process
    if process:
        try:
            if wait_to_finish:
                global eval_queue
                if eval_queue is not None:
                    eval_queue.put("finish")
                process.join()
            else:
                process.close()
        except:
            process.terminate()

