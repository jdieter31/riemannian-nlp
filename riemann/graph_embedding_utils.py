import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn.functional import cross_entropy, kl_div, log_softmax, softmax, relu_
from manifolds import RiemannianManifold
from manifold_tensors import ManifoldParameter
from typing import Dict, List
from embedding import GloveSentenceEmbedder
from embedding import SimpleSentence
from tqdm import tqdm
from embed_save import Savable
import numpy as np

EPSILON = 1e-9

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

def manifold_dist_loss_kl(model: nn.Module, inputs: torch.Tensor, train_distances: torch.Tensor, manifold: RiemannianManifold):
    """ Gives a loss function defined by the KL divergence of the distribution given by the manifold distance verses the provide train_distances
    Args:
        model (nn.Module): model that takes in graph indices and ouptuts embeddings
        inputs (torch.Tensor): LongTensor of shape [batch_size, num_samples+1] giving the indices of the vertices to be trained with the first vertex in each element of the batch being the main vertex and the others being samples
        train_distances (torch.Tensor): floating point tensor of shape [batch_size, num_samples] containing the training distances from the input vertex to the sampled vertices
        manifold (RiemannianManifold): Manifold that model embeds vertices into

    Returns:
        kl_div (scalar): KL Divergence of sampled distributions
    """

    input_embeddings = model(inputs)
    sample_vertices = input_embeddings.narrow(1, 1, input_embeddings.size(1)-1)
    main_vertices = input_embeddings.narrow(1,0,1).expand_as(sample_vertices)
    manifold_dists = manifold.dist(main_vertices, sample_vertices)
    manifold_dists_log = (manifold_dists + 0.01).log()
    manifold_dist_distrib = log_softmax(-manifold_dists, -1)
    train_dists_log = (train_distances + 0.01).log()
    train_distrib = softmax(-train_distances, -1)
    return kl_div(manifold_dist_distrib, train_distrib, reduction="batchmean")

def manifold_dist_loss_relu_sum(model: nn.Module, inputs: torch.Tensor, train_distances: torch.Tensor, manifold: RiemannianManifold, margin=0.01, discount_factor=0.9):
    input_embeddings = model(inputs)

    sample_vertices = input_embeddings.narrow(1, 1, input_embeddings.size(1)-1)
    main_vertices = input_embeddings.narrow(1, 0, 1).expand_as(sample_vertices)
    manifold_dists = manifold.dist(main_vertices, sample_vertices)

    sorted_indices = train_distances.argsort(dim=-1)
    manifold_dists_sorted = torch.gather(manifold_dists, -1, sorted_indices)
    manifold_dists_sorted.add_(EPSILON).log_()
    diff_matrix_shape = [manifold_dists.size()[0], manifold_dists.size()[1], manifold_dists.size()[1]]
    row_expanded = manifold_dists_sorted.unsqueeze(2).expand(*diff_matrix_shape)
    column_expanded = manifold_dists_sorted.unsqueeze(1).expand(*diff_matrix_shape)
    diff_matrix = row_expanded - column_expanded + margin

    train_dists_sorted = torch.gather(train_distances, -1, sorted_indices)
    train_row_expanded = train_dists_sorted.unsqueeze(2).expand(*diff_matrix_shape)
    train_column_expanded = train_dists_sorted.unsqueeze(1).expand(*diff_matrix_shape)
    diff_matrix_train = train_row_expanded - train_column_expanded
    masked_diff_matrix = torch.where(diff_matrix_train == 0, diff_matrix_train, diff_matrix)
    masked_diff_matrix.triu_()
    relu_(masked_diff_matrix)
    masked_diff_matrix = masked_diff_matrix.mean(-1)
    order_scale = torch.arange(0, masked_diff_matrix.size()[1], device=masked_diff_matrix.device, dtype=masked_diff_matrix.dtype)
    order_scale = (torch.ones_like(order_scale) * discount_factor).pow(order_scale)
    order_scale = order_scale.unsqueeze_(0).expand_as(masked_diff_matrix) 
    masked_diff_matrix *= order_scale
    loss = masked_diff_matrix.sum(-1).mean()
    return loss

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

