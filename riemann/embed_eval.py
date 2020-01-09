from sacred import Ingredient
import torch.multiprocessing as mp

import json
import torch

from .data.graph import eval_reconstruction

from .embed_save import load_model

from .logging_thread import write_tensorboard, write_log
from embedding_evaluation.process_benchmarks import process_benchmarks
from .graph_embedding_utils import get_canonical_glove_word_featurizer, FeaturizedModelEmbedding
from .manifold_nns import compute_pole_batch, ManifoldNNS
from .manifolds.schilds_ladder import schilds_ladder
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

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
            meanrank, mrr, hitsat10 = eval_reconstruction(adj, adj, embeddings, manifold.dist, workers=1, progress=True)
            lmsg.update({
                'mean_rank': meanrank,
                'mrr': mrr,
                'hits@10': hitsat10
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
            write_tensorboard('add_scalar', ['mrr', mrr, epoch])
            write_tensorboard('add_scalar', ['hits@10', hitsat10, epoch])
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
        model_dist.append(dist_func(word1, word2)[0].cpu().detach().numpy())
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


def load_analogy(path):
    syn_analogies = []
    sem_analogies = []
    sem = True
    with open(path) as f:
        line = f.readline()
        while line:
            if line.startswith(":"):
                if line.startswith(": gram"):
                    sem = False
            else:
                if sem:
                    sem_analogies.append(line.split())
                else:
                    syn_analogies.append(line.split())
            line = f.readline()
    return syn_analogies, sem_analogies

syn_analogies, sem_analogies = load_analogy("./data/google_analogy/questions-words.txt")

def eval_analogy(model, manifold, nns=None):

    with torch.no_grad():
        embedding_matrix = model.get_embedding_matrix().cpu()
        features = [feature.lower() for feature in model.features_list]
        feature_set = set(features)
        extra_vocab = []
        extra_vecs = []
        analogies_not_in_vocab = [0, 0]
        skip_analogies = []
        
        for analogy_set in [syn_analogies, sem_analogies]:
            for analogy in tqdm(analogy_set):
                for word in analogy:
                    if word.lower() not in feature_set:
                        if model.featurizer(word.lower()) is None:
                            skip_analogies.append(analogy)
                            break

                        extra_vocab.append(word.lower())
                        feature_set.add(word.lower())
                        extra_vecs.append(model.forward_featurize(word.lower()).cpu().squeeze(0))
                            
        vocab = features + extra_vocab
        vocab_dict = {vocab[i] : i for i in range(len(vocab))}
        embedding_matrix = torch.cat([embedding_matrix, torch.stack(extra_vecs)])

        if nns is None:
            manifold_nns = ManifoldNNS(embedding_matrix, manifold)
        else:
            manifold_nns = nns
            nns.add_vectors(torch.stack(extra_vecs))

        results = []
        print("Evaluating analogy")
        for analogy_set in [syn_analogies, sem_analogies]:

            a1_vecs = []
            a2_vecs = []
            b1_vecs = []
            
            a1_words = []
            a2_words = []
            b1_words = []
            b2_words = []

            for analogy in tqdm(analogy_set):
                if analogy in skip_analogies:
                    continue 
                a1_vecs.append(embedding_matrix[vocab_dict[analogy[0].lower()]])
                a2_vecs.append(embedding_matrix[vocab_dict[analogy[1].lower()]])
                b1_vecs.append(embedding_matrix[vocab_dict[analogy[2].lower()]])
                a1_words.append(analogy[0].lower())
                a2_words.append(analogy[1].lower())
                b1_words.append(analogy[2].lower())
                b2_words.append(analogy[3].lower())

            a1_vecs = torch.stack(a1_vecs)
            a2_vecs = torch.stack(a2_vecs)
            b1_vecs = torch.stack(b1_vecs)
        
            da = manifold.log(a1_vecs, a2_vecs)
            db = schilds_ladder(a1_vecs, b1_vecs, da, manifold)
            b2_vecs = manifold.exp(b1_vecs, db)

            _, nns = manifold_nns.knn_query_batch_vectors(b2_vecs, k=100)       
            embeddings = embedding_matrix[nns]
            b2_vecs_expanded = b2_vecs.unsqueeze(-2).expand_as(embeddings)
            dists = manifold.dist(b2_vecs_expanded, embeddings)
            sorted_dists, indices = torch.sort(dists)
            right = 0
            for i in range(dists.size(0)):
                j = 0
                feature = vocab[nns[i][indices[i][j]]].lower()
                while (feature == a1_words[i]) or (feature == a2_words[i]) or (feature == b1_words[i]):
                    j += 1
                    feature = vocab[nns[i][indices[i][j]]].lower()
                if feature == b2_words[i]:
                    right += 1
            
            results.append(right/len(analogy_set))
        return results[0], results[1]

@eval_ingredient.config
def config():
    eval_workers = 5
    benchmarks = []
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

