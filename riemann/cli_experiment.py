from sacred import Experiment
import torch

from embed_save import save_ingredient, load_model
from graph_embedding_utils import get_canonical_glove_sentence_featurizer

ex = Experiment('cli', ingredients=[save_ingredient])

@ex.config
def config():
    neighbors=False

@ex.command
def cli_search(neighbors):
    print("Loading model...")
    model, extra_data = load_model()
    model = model.double()
    objects = extra_data["objects"]
    embeddings = extra_data["embedding_matrix"]
    embeddings = embeddings.double()
    manifold = extra_data["manifold"]
    if not neighbors:
        featurizer, _ = get_canonical_glove_sentence_featurizer()
    while True:
        print("Input a word to search near neighbors (or type 'quit')")
        search_q = input("--> ")
        if search_q == "quit":
            return
        if neighbors:
            if not search_q in objects:
                print("Search query not found in embeddings!")
                continue
        k = -1
        while k < 0:
            print("How many neighbors to list?")
            try:
                k = int(input("--> "))
            except:
                print("Must be valid integer")
        if neighbors:
            q_index = objects.index(search_q)
            dists = manifold.dist(embeddings[None, q_index], embeddings)
        else:
            dists = manifold.dist(model(torch.tensor(featurizer(search_q), dtype=torch.double)).unsqueeze(0), embeddings)
            
        sorted_dists, sorted_indices = dists.sort()
        sorted_objects = [objects[index] for index in sorted_indices]
        for i in range(k):
            print(f"{sorted_objects[i]} - dist: {sorted_dists[i]}")

if __name__ == '__main__':
    ex.run_commandline()
