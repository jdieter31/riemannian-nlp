from sacred import Ingredient
import torch.multiprocessing as mp

import json
import torch

from data.graph import eval_reconstruction

from embed_save import load_model

from logging_thread import write_tensorboard, write_log
from embedding_evaluation.process_benchmarks import process_benchmarks
from graph_embedding_utils import get_canonical_glove_word_featurizer, FeaturizedModelEmbedding
from manifold_nns import compute_pole_batch
from torch.nn.functional import cosine_similarity

from scipy import stats

eval_ingredient = Ingredient('evaluation')

eval_queue = None
process = None

def async_eval(adj, benchmarks_to_eval, num_workers, eval_mean_rank, tboard_projector):
    global eval_queue
    if eval_queue is None:
        return
    finished = False

    if len(benchmarks_to_eval) > 0:
        benchmarks = process_benchmarks()
    featurizer, featurizer_dim = None, 0 
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
        dimension = save_data["dimension"]
        in_manifold = save_data["in_manifold"]
        if "features" in save_data:
            if featurizer is None:
                featurizer, featurizer_dim = get_canonical_glove_word_featurizer()
            if graph_embedding_model is None:
                graph_embedding_model = FeaturizedModelEmbedding(model, save_data["features"], in_manifold, manifold, dimension, featurizer=featurizer, featurizer_dim=featurizer_dim)
                if "additional_embeddings_state_dict" in save_data:
                    graph_embedding_model.get_additional_embeddings().load_state_dict(save_data["additional_embeddings_state_dict"])
                if "main_deltas_state_dict" in save_data:
                    graph_embedding_model.main_deltas.load_state_dict(save_data["main_deltas_state_dict"])
                if "additional_deltas_state_dict" in save_data:
                    graph_embedding_model.additional_deltas.load_state_dict(save_data["additional_deltas_state_dict"])
                if "deltas" in save_data:
                    graph_embedding_model.deltas = save_data["deltas"]
            else:
                graph_embedding_model.embedding_model = model
                if "additional_embeddings_state_dict" in save_data:
                    graph_embedding_model.get_additional_embeddings().load_state_dict(save_data["additional_embeddings_state_dict"])
                if "main_deltas_state_dict" in save_data:
                    graph_embedding_model.main_deltas.load_state_dict(save_data["main_deltas_state_dict"])
                if "additional_deltas_state_dict" in save_data:
                    graph_embedding_model.additional_deltas.load_state_dict(save_data["additional_deltas_state_dict"])
                if "deltas" in save_data:
                    graph_embedding_model.deltas = save_data["deltas"]
        else:
            graph_embedding_model = model
        if eval_mean_rank or tboard_projector:
            with torch.no_grad():
                embeddings = graph_embedding_model.get_embedding_matrix()
        if tboard_projector:
            write_tensorboard('add_embedding', [embeddings.cpu().detach().numpy(), objects, None, epoch, f"epoch-{epoch}"])


        lmsg = {
            'epoch': epoch,
            'elapsed': elapsed,
            'loss': loss,
        }
        

        if eval_mean_rank:
            meanrank, maprank = eval_reconstruction(adj, embeddings, manifold.dist, workers=num_workers)
            lmsg.update({
                'mean_rank': meanrank,
                'map_rank': maprank
            })

        if featurizer is not None:
            benchmark_results = {}
            #poles = compute_pole_batch(embeddings, manifold)

            for benchmark in benchmarks_to_eval:
                #featurize = lambda w: in_manifold.proj(torch.FloatTensor(featurizer(w))) if featurizer(w) is not None else in_manifold.proj(torch.zeros(embeddings.size(-1)))
                #dist_func = lambda w1, w2: pole_log_cosine_sim(model(w1), model(w2), manifold, poles)
                
                dist_func = lambda w1, w2: - manifold.dist(graph_embedding_model.forward_featurize(w1), graph_embedding_model.forward_featurize(w2))
                with torch.no_grad():
                    rho = eval_benchmark(benchmarks[benchmark], dist_func)
                benchmark_results[f"{benchmark}_rho"] = rho
                write_tensorboard('add_scalar', [f"{benchmark}_rho", rho, epoch])
            
            lmsg.update(benchmark_results)

        if eval_mean_rank:
            write_tensorboard('add_scalar', ['mean_rank', meanrank, epoch])
            write_tensorboard('add_scalar', ['map_rank', maprank, epoch]) 
        write_log(f"Stats: {json.dumps(lmsg)}")


def pole_log_cosine_sim(w1, w2, manifold, poles):
    w1 = w1.unsqueeze(1).expand(w1.size(0), poles.size(0), w1.size(-1))
    w2 = w2.unsqueeze(1).expand_as(w1)
    poles = poles.unsqueeze(0).expand_as(w1)
    log_w1 = manifold.log(poles, w1)
    log_w2 = manifold.log(poles, w2)
    cosine_sim = cosine_similarity(log_w1, log_w2, dim=-1)
    return cosine_sim.mean(-1)

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
    benchmarks = ['men_dev']
    eval_mean_rank = False
    tboard_projector = False

@eval_ingredient.capture
def initialize_eval(eval_workers, adjacent_list, benchmarks, eval_mean_rank, tboard_projector):
    global eval_queue
    eval_queue = mp.Queue()
    global process
    process = mp.Process(target=async_eval, args=(adjacent_list, benchmarks, eval_workers, eval_mean_rank, tboard_projector))
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

