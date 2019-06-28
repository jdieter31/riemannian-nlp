from sacred import Ingredient
import torch.multiprocessing as mp

import json
import torch

from data.graph import eval_reconstruction
from manifold_embedding import ManifoldEmbedding

eval_ingredient = Ingredient('evaluation')

eval_queue = None
log_queue = None


def async_eval(adj, log_queue):
    global eval_queue
    if eval_queue is None:
        return

    while True:
        temp = eval_queue.get()
        if temp is None:
            return

        if not eval_queue.empty():
            continue

        epoch, elapsed, loss, path = temp
        params = torch.load(path, map_location='cpu')
        objects = params["objects"]
        dimension = params["dimension"]
        double_precision = params["double_precision"]
        manifold = params["manifold"]

        model = ManifoldEmbedding(
            manifold,
            len(objects),
            dimension
        )
        model.to(torch.device('cpu'))
        if double_precision:
            model.double()
        model.load_state_dict(params["model"])
        embeddings = model.weight.data
        meanrank, maprank = eval_reconstruction(adj, embeddings, manifold.dist)
        lmsg = {
            'epoch': epoch,
            'elapsed': elapsed,
            'loss': loss,
            'mean_rank': meanrank,
            'map_rank': maprank
        }
        log_queue.put(f"Stats: {json.dumps(lmsg)}")

@eval_ingredient.capture
def initialize_eval(adjacent_list, log_queue_):
    global log_queue
    log_queue = log_queue_
    global eval_queue
    eval_queue = mp.Queue()
    process = mp.Process(target=async_eval, args=(adjacent_list, log_queue_))
    process.start()

@eval_ingredient.capture
def evaluate(epoch, elapsed, loss, path):
    global eval_queue
    eval_queue.put((epoch, elapsed, loss, path))

