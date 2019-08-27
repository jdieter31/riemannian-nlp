from sacred import Ingredient
import torch.multiprocessing as mp

import json
import torch

from data.graph import eval_reconstruction

from embed_save import load_model

from logging_thread import write_tensorboard, write_log
from embedding_evaluation.process_benchmarks import process_benchmarks
from graph_embedding_utils import get_canonical_glove_sentence_featurizer, FeaturizedModelEmbedding

from scipy import stats

eval_ingredient = Ingredient('evaluation')

eval_queue = None
process = None

def async_eval(adj, benchmarks_to_eval, num_workers):
    global eval_queue
    if eval_queue is None:
        return
    finished = False

    if len(benchmarks_to_eval) > 0:
        benchmarks = process_benchmarks()
    featurizer, featurizer_dim = get_canonical_glove_sentence_featurizer()
    graph_embedding_model = None

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
        if "features" in save_data:
            if graph_embedding_model is None:
                graph_embedding_model = FeaturizedModelEmbedding(model, save_data["features"], featurizer=featurizer, featurizer_dim=featurizer_dim)
            else:
                graph_embedding_model.embedding_model = model
        else:
            graph_embedding_model = model

        embeddings = graph_embedding_model.get_embedding_matrix()
        write_tensorboard('add_embedding', [embeddings.cpu().detach().numpy(), objects, None, epoch, f"epoch-{epoch}"])

        meanrank, maprank = eval_reconstruction(adj, embeddings, manifold.dist, workers=num_workers)

        benchmark_results = {}
        for benchmark in benchmarks_to_eval:
            featurize = lambda w: embeddings.new_tensor(featurizer(w))
            dist_func = lambda w1, w2: - manifold.dist(model(w1), model(w2))
            rho = eval_benchmark_batch(benchmarks[benchmark], featurize, dist_func) 
            benchmark_results[f"{benchmark}_rho"] = rho
            write_tensorboard('add_scalar', [f"{benchmark}_rho", rho, epoch])

        write_tensorboard('add_scalar', ['mean_rank', meanrank, epoch])
        write_tensorboard('add_scalar', ['map_rank', maprank, epoch])

        lmsg = {
            'epoch': epoch,
            'elapsed': elapsed,
            'loss': loss,
            'mean_rank': meanrank,
            'map_rank': maprank
        }
        lmsg.update(benchmark_results)
        write_log(f"Stats: {json.dumps(lmsg)}")

def eval_benchmark(benchmark, dist_func):
    gold_list = []
    model_dist = []
    for (word1, word2), gold_score in benchmark.items():
        gold_list.append(gold_score)
        model_dist.append(dist_func(word1, word2))
    return stats.spearmanr(model_dist, gold_list)[0]

def eval_benchmark_batch(benchmark, featurizer, dist_func):
    gold_list = []
    w1_feature_list = []
    w2_feature_list = []
    for (word1, word2), gold_score in benchmark.items():
        gold_list.append(gold_score)
        w1_feature_list.append(featurizer(word1))
        w2_feature_list.append(featurizer(word2))
    w1_features = torch.stack(w1_feature_list)
    w2_features = torch.stack(w2_feature_list)
    dists = dist_func(w1_features, w2_features).detach().numpy()
    return stats.spearmanr(dists, gold_list)[0]

@eval_ingredient.config
def config():
    eval_workers = 5
    benchmarks = ['usf', 'men_dev', 'vis_sim', 'sem_sim']

@eval_ingredient.capture
def initialize_eval(eval_workers, adjacent_list, benchmarks):
    global eval_queue
    eval_queue = mp.Queue()
    global process
    process = mp.Process(target=async_eval, args=(adjacent_list, benchmarks, eval_workers))
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

