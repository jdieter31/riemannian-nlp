import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn.functional import cross_entropy
from manifolds import RiemannianManifold
from manifold_tensors import ManifoldParameter
from typing import Dict, List
from embedding import GloveSentenceEmbedder
from embedding import SimpleSentence
from tqdm import tqdm
from embed_save import Savable
import numpy as np

def manifold_dist_loss(model: nn.Module, inputs: torch.Tensor,
        targets: torch.Tensor, manifold: RiemannianManifold):
        """
        :param model: model that takes in graph indices and outputs embeddings
        :param inputs: Tensor of shape [batch_size, 2 + number of negative examples]
            Input to train on. The second dimension contains 2 + number of negative
            examples as the first contains the word itself, followed by one positive
            example and the rest of the negative examples
        :param targets: Tensor of shape batch_size
            Gves the index of where the positive example is in inputs out of
            all of the negative examples trained on
        :return: scalar
        """
        embeddings = model(inputs)
        relations = embeddings.narrow(1, 1, embeddings.size(1) - 1)
        word = embeddings.narrow(1, 0, 1).expand_as(relations)
        dists = manifold.dist(word, relations)

        # Minimize distance between words that are positive examples
        return cross_entropy(-dists, targets)


class ManifoldEmbedding(Embedding, Savable):

    def __init__(
            self,
            manifold: RiemannianManifold,
            num_embeddings,
            embedding_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)

        self.manifold = manifold
        self.params = [manifold, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse]
        self.weight = ManifoldParameter(self.weight.data, manifold=manifold)
    
    def get_embedding_matrix(self):
        return self.weight.data

    def get_save_data(self):
        return {
            'state_dict': self.state_dict(),
            'params': self.params
        }

    @classmethod
    def from_save_data(cls, data):
        params = data["params"]
        state_dict = data["state_dict"]
        embedding = ManifoldEmbedding(*params)
        embedding.load_state_dict(state_dict)
        return embedding

    def get_savable_model(self):
        return self

class FeaturizedModelEmbedding(nn.Module):
    def __init__(self, embedding_model: nn.Module, features_list, featurizer=None, featurizer_dim=0):
        super(FeaturizedModelEmbedding, self).__init__()
        self.embedding_model = embedding_model
        if featurizer is None:
            featurizer, featurizer_dim = get_canonical_glove_sentence_featurizer()
        self.featurizer = featurizer
        self.featurizer_dim = featurizer_dim
        self.input_embedding = get_featurized_embedding(features_list, featurizer, featurizer_dim)

    def forward(self, x):
        return self.embedding_model(self.input_embedding(x))

    def forward_featurize(self, feature):
        featurized = self.embedding_model(self.featurizer(feature))
        return featurized

    def get_embedding_matrix(self):
        out = self.embedding_model(self.input_embedding.weight.data)
        return out


    def get_savable_model(self):
        return self.embedding_model

def get_canonical_glove_sentence_featurizer():
    embedder = GloveSentenceEmbedder.canonical()
    return lambda sent : embedder.embed(SimpleSentence.from_text(sent)), embedder.dim

def get_featurized_embedding(features: List, featurizer, featurizer_dim, dtype=torch.double):
    embeddings_list = np.empty((len(features), featurizer_dim))
    print("Processing features of dataset...") 
    for i in tqdm(range(len(features))):
        embeddings_list[i] = featurizer(features[i])
    return Embedding.from_pretrained(torch.as_tensor(np.array(embeddings_list)))

