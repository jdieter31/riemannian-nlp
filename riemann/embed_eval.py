from sacred import Ingredient
import torch.multiprocessing as mp

import os
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
from .embedding.conceptnet import standardized_uri

from scipy import stats


eval_ingredient = Ingredient('evaluation')

benchmark_set = {}

@eval_ingredient.capture
def eval_wordsim_benchmarks(graph_embedding_model, manifold, benchmarks, dist_func, device='cpu', iteration=0):
    benchmark_results = {}
    global benchmark_set
    if dist_func == "manifold_cosine":
        with torch.no_grad():
            embeddings = graph_embedding_model.get_embedding_matrix()
        poles = compute_pole_batch(embeddings, manifold)
        poles = poles.to(device)

    print("\nEvalutating Benchmarks")
    for benchmark in benchmarks:

                #featurize = lambda w: in_manifold.proj(torch.FloatTensor(featurizer(w)).to(device)) if featurizer(w) is not None else in_manifold.proj(torch.zeros(embeddings.size(-1)).to(device))
        if dist_func == "manifold_cosine":
            dist_func = lambda w1, w2: pole_log_cosine_sim(graph_embedding_model.forward_featurize(standardized_uri("en", w1)), graph_embedding_model.forward_featurize(standardized_uri("en", w2)), manifold, poles)
        else:
            dist_func = lambda w1, w2: - manifold.dist(graph_embedding_model.forward_featurize(standardized_uri("en", w1)), graph_embedding_model.forward_featurize(standardized_uri("en", w2)))
        # dist_func = lambda w1, w2: - manifold.dist(graph_embedding_model.forward_featurize(w1), graph_embedding_model.forward_featurize(w2))

        # featurizer = lambda w: graph_embedding_model.forward_featurize(w)
        # featurizer = lambda w: torch.as_tensor([graph_embedding_model.featurizer(w)], dtype=graph_embedding_model.input_embedding.weight.dtype, device=device)

        # dist_func = lambda w1, w2: - manifold.dist(graph_embedding_model.embedding_model(w1), graph_embedding_model.embedding_model(w2))

        with torch.no_grad():
            rho = eval_benchmark(benchmark_set[benchmark], dist_func)
            # rho = eval_benchmark_batch(benchmark_set[benchmark], featurizer, dist_func)
        benchmark_results[f"{benchmark}_rho"] = rho
        write_tensorboard('add_scalar', [f"{benchmark}_rho", rho, iteration])


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
    for (word1, word2), gold_score in tqdm(benchmark.items()):
        gold_list.append(gold_score)
        model_dist.append(dist_func(word1, word2)[0].cpu().detach().numpy())
    return stats.spearmanr(model_dist, gold_list)[0]

def eval_benchmark_batch(benchmark, featurizer, dist_func):
    gold_list = []
    w1_feature_list = []
    w2_feature_list = []
    for (word1, word2), gold_score in tqdm(benchmark.items()):
        gold_list.append(gold_score)
        w1_feature_list.append(featurizer(word1))
        w2_feature_list.append(featurizer(word2))
    w1_features = torch.stack(w1_feature_list)
    w2_features = torch.stack(w2_feature_list)
    dists = dist_func(w1_features, w2_features).cpu().detach().numpy()
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


root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
syn_analogies, sem_analogies = None, None

def eval_analogy(model, manifold, nns=None):
    global syn_analogies, sem_analogies
    if syn_analogies is None or sem_analogies is None:
        syn_analogies, sem_analogies = load_analogy(os.path.join(root_path, "data/google_analogy/questions-words.txt"))

    with torch.no_grad():
        embedding_matrix = model.get_embedding_matrix().cpu()
        features = [standardized_uri("en", feature) for feature in model.features_list]
        feature_set = set(features)
        extra_vocab = []
        extra_vecs = []
        analogies_not_in_vocab = [0, 0]
        skip_analogies = []
        
        for analogy_set in [syn_analogies, sem_analogies]:
            for analogy in tqdm(analogy_set):
                for word in analogy:
                    if standardized_uri("en", word) not in feature_set:
                        if model.featurizer(standardized_uri("en", word)) is None:
                            skip_analogies.append(analogy)
                            break

                        extra_vocab.append(standardized_uri("en", word))
                        feature_set.add(standardized_uri("en", word))
                        extra_vecs.append(model.forward_featurize(standardized_uri("en", word)).cpu().squeeze(0))
                            
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
                a1_vecs.append(embedding_matrix[vocab_dict[standardized_uri("en", analogy[0])]])
                a2_vecs.append(embedding_matrix[vocab_dict[standardized_uri("en", analogy[1])]])
                b1_vecs.append(embedding_matrix[vocab_dict[standardized_uri("en", analogy[2])]])
                a1_words.append(standardized_uri("en", analogy[0]))
                a2_words.append(standardized_uri("en", analogy[1]))
                b1_words.append(standardized_uri("en", analogy[2]))
                b2_words.append(standardized_uri("en", analogy[3]))

            a1_vecs = torch.stack(a1_vecs)
            a2_vecs = torch.stack(a2_vecs)
            b1_vecs = torch.stack(b1_vecs)
        
            da = manifold.log(a1_vecs, a2_vecs)
            db = schilds_ladder(a1_vecs, b1_vecs, da, manifold)
            b2_vecs = manifold.exp(b1_vecs, db)

            _, nns = manifold_nns.knn_query_batch_vectors(b2_vecs, k=60)       
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
                    if j >= 60:
                        break
                    feature = vocab[nns[i][indices[i][j]]].lower()
                if feature == b2_words[i]:
                    right += 1
            
            results.append(right/len(analogy_set))
        return results[0], results[1]

@eval_ingredient.config
def config():
    eval_workers = 5
    # benchmarks = ['men_full', 'men_dev', 'men_test', 'ws353', 'rw', 'simlex', "simlex-q1", "simlex-q2", "simlex-q3", "simlex-q4", "mturk771"]
    benchmarks = ['men_full', 'men_dev', 'ws353', 'rw', "mturk771"]
    dist_func = "manifold_cosine"
    eval_mean_rank = False
    tboard_projector = False

@eval_ingredient.capture
def initialize_eval(eval_workers, adjacent_list, benchmarks, eval_mean_rank, tboard_projector):
    global benchmark_set
    if len(benchmarks) > 0:
        benchmark_set = process_benchmarks()
    

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

