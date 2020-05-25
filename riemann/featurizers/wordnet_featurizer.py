from ..embedding.sentence_embedder import GloveSentenceEmbedder
from ..embedding.word_embedding import Glove
from ..embedding.core_nlp import SimpleSentence
from ..data.graph_dataset import GraphDataset
from ..manifolds import SphericalManifold
import torch
import numpy as np
from tqdm import tqdm

def get_wordnet_featurizer(graph_dataset: GraphDataset, eval_dataset:
                           GraphDataset=None):
    sentence_embedder = GloveSentenceEmbedder.canonical()
    
    embeddings = []
    bad_nodes = []
    for i, o_id in enumerate(graph_dataset.object_ids):
        sentence = ' '.join(o_id.split('.')[0].split('_')) 
        ssentence = SimpleSentence.from_text(sentence)
        embedding = sentence_embedder.embed(ssentence)

        if np.any(embedding):
            embeddings.append(embedding)
        else:
            bad_nodes.append(i)

    graph_dataset.collapse_nodes(bad_nodes)
    if eval_dataset is not None:
        eval_dataset.collapse_nodes(bad_nodes)
    """
    deleted = 0
    for bad_node in tqdm(bad_nodes, desc="Collapsing nodes w/o features",
                         dynamic_ncols=True):
        graph_dataset.collapse_node(i - deleted)
        if eval_dataset is not None:
            eval_dataset.collapse_node(i - deleted)
        deleted += 1
    """

    vectors = torch.tensor(np.array(embeddings), dtype=torch.float, device=torch.device('cpu'))

    def featurize(object_ids, node_ids):
        return vectors[node_ids]

    return featurize, sentence_embedder.dim, SphericalManifold()




