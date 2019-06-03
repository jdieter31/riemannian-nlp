import torch
from geoopt import ManifoldParameter
from geoopt.manifolds import Manifold
from torch.nn.functional import cross_entropy


class ManifoldEmbedding(torch.nn.Embedding):

    def __init__(self, manifold: Manifold, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None ):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm,
                 norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)

        self.manifold = manifold
        self.weight = ManifoldParameter(data=self.weight, manifold=manifold)
        self.weight.proj_()

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        :param inputs: Tensor of shape [batch_size, 2 + number of negative examples]
            Input to train on. The second dimension contains 2 + number of negative
            examples as the first contains the word itself, followed by one positive
            example and the rest of the negative examples
        :param targets: Tensor of shape batch_size
            Gives the index of where the positive example is in inputs out of
            all of the negative examples trained on
        :return: scalar
        """
        embeddings = self(inputs)
        relations = embeddings.narrow(1, 1, embeddings.size(1) - 1)
        word = embeddings.narrow(1, 0, 1).expand_as(relations)
        dists = self.manifold.dist(word, relations)
        # Minimize distance between words that are positive examples
        return cross_entropy(-dists, targets)
