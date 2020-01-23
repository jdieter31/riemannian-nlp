import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn.functional import cross_entropy, kl_div, log_softmax, softmax, relu_, mse_loss, cosine_similarity, relu
from .manifolds import RiemannianManifold, EuclideanManifold
from .manifold_tensors import ManifoldParameter
from typing import Dict, List
from .embedding import GloveSentenceEmbedder
from .embedding import SimpleSentence
from .embedding import Glove
from tqdm import tqdm
from .embed_save import Savable
from .jacobian import compute_jacobian
import numpy as np
from .manifold_initialization import initialize_manifold_tensor, get_initialized_manifold_tensor
from torch.autograd import Function
from math import ceil
from tqdm import tqdm

EPSILON = 1e-9

def manifold_dist_loss_relu_sum(model: nn.Module, inputs: torch.Tensor, train_distances: torch.Tensor, manifold: RiemannianManifold, margin=0.01, discount_factor=0.9):
    """
    See write up for details on this loss function -- encourages embeddings to preserve graph topology
    Parameters:
        model (nn.Module): model that takes in graph indices and outputs embeddings in output manifold space
        inputs (torch.Tensor): LongTensor of shape [batch_size, num_samples+1] giving the indices of 
            the vertices to be trained with the first vertex in each element of the batch being the 
            main vertex and the others being samples
        train_distances (torch.Tensor): floating point tensor of shape [batch_size, num_samples]
            containing the training distances from the input vertex to the sampled vertices
        manifold (RiemannianManifold): Manifold that model embeds vertices into

    Returns:
        pytorch scalar: Computed loss
    """
    
    input_embeddings = model(inputs)

    sample_vertices = input_embeddings.narrow(1, 1, input_embeddings.size(1)-1)
    main_vertices = input_embeddings.narrow(1, 0, 1).expand_as(sample_vertices)
    manifold_dists = manifold.dist(main_vertices, sample_vertices)

    sorted_indices = train_distances.argsort(dim=-1)
    manifold_dists_sorted = torch.gather(manifold_dists, -1, sorted_indices)
    #manifold_dists_sorted.add_(EPSILON).log_()
    # manifold_dists_sorted = manifold_dists_sorted ** 2
    manifold_dists_sorted = (manifold_dists_sorted + EPSILON).log()
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
    masked_diff_matrix = masked_diff_matrix.sum(-1)
    loss = masked_diff_matrix.sum(-1).mean()
    return loss

def metric_loss(model: nn.Module, input_embeddings: torch.Tensor, in_manifold: RiemannianManifold, out_manifold: RiemannianManifold, out_dimension: int, isometric=False, random_samples=0, random_init = None):
    """
    See write up for details on this loss functions -- encourages model to be isometric or to be conformal
    Parameters:
        model (nn.Module): model that takes in embeddings in original space and outputs embeddings in output manifold space
        input_embeddings (torch.Tensor): tensor of shape [batch_size, embedding_sim] with embeddings in original space
        in_manifold (RiemannianManifold): RiemannianManifold object characterizing original space
        out_manifold (RiemannianManifold): RiemannianManifold object characterizing output space
        out_dimension (int): dimension of tensors in out_manifold
        isometric (bool): The function will be optimized to be isometric if True, conformal if False. Riemannian distance
            on the manifold of PD matrices is used to optimized the metrics if isometric and cosine distance between
            the flattened metric matrices is used if conformal
        random_samples (int): Number of randomly generated samples to use in addition to provided input_embeddings
        random_init (dict): Parameters to use for random generation of samples - use format described in manifold_initialization

    Returns:
        pytorch scalar: computed loss
    """
    input_embeddings = model.map_to_input_embeddings(input_embeddings)
    if random_samples > 0:
        random_samples = torch.empty(random_samples, input_embeddings.size()[1], dtype=input_embeddings.dtype, device=input_embeddings.device)
        initialize_manifold_tensor(random_samples, in_manifold, random_init)
        input_embeddings = torch.cat([input_embeddings, random_samples])

    model = model.embedding_model
    jacobian, model_out = compute_jacobian(model, input_embeddings, out_dimension)
    jacobian = jacobian.clamp(-1, 1)
    tangent_proj_out = out_manifold.tangent_proj_matrix(model_out)
    jacobian_shape = jacobian.size()
    tangent_proj_out_shape = tangent_proj_out.size()
    tangent_proj_out_batch = tangent_proj_out.view(-1, tangent_proj_out_shape[-2], tangent_proj_out_shape[-1])
    jacobian_batch = jacobian.view(-1, jacobian_shape[-2], jacobian_shape[-1])

    tangent_proj_in = in_manifold.tangent_proj_matrix(input_embeddings)
    proj_eigenval, proj_eigenvec = torch.symeig(tangent_proj_in, eigenvectors=True)
    first_nonzero = (proj_eigenval > 1e-3).nonzero()[0][1]
    significant_eigenvec = proj_eigenvec.narrow(-1, first_nonzero, proj_eigenvec.size()[-1] - first_nonzero)
    significant_eigenvec_shape = significant_eigenvec.size()
    significant_eigenvec_batch = significant_eigenvec.view(-1, significant_eigenvec_shape[-2], significant_eigenvec_shape[-1])
    metric_conjugator = torch.bmm(torch.bmm(tangent_proj_out_batch, jacobian_batch), significant_eigenvec_batch)
    metric_conjugator_t = torch.transpose(metric_conjugator, -2, -1)
    out_metric = out_manifold.get_metric_tensor(model_out)
    out_metric_shape = out_metric.size()
    out_metric_batch = out_metric.view(-1, out_metric_shape[-2], out_metric_shape[-1])
    pullback_metric = torch.bmm(torch.bmm(metric_conjugator_t, out_metric_batch), metric_conjugator)
    in_metric = in_manifold.get_metric_tensor(input_embeddings)
    in_metric_shape = in_metric.size()
    in_metric_batch = in_metric.view(-1, in_metric_shape[-2], in_metric_shape[-1])
    sig_eig_t = torch.transpose(significant_eigenvec_batch, -2, -1)
    in_metric_reduced = torch.bmm(torch.bmm(sig_eig_t, in_metric_batch), significant_eigenvec_batch)
    in_metric_flattened = in_metric_batch.view(in_metric_reduced.size()[0], -1)
    pullback_flattened = pullback_metric.view(pullback_metric.size()[0], -1)
    
    if not isometric:
        in_metric_reduced = in_metric_reduced / in_metric_reduced.norm(dim=(-2, -1), keepdim=True)
        pullback_metric = pullback_metric / pullback_metric.norm(dim=(-2, -1), keepdim=True)

    # if isometric:
    rd = riemannian_divergence(in_metric_reduced, pullback_metric)
    rd_scaled = torch.sqrt(rd)
    loss = rd_scaled.mean()

    '''
    else:
        loss = -torch.mean(cosine_similarity(pullback_flattened, in_metric_flattened, -1))
    '''

    return loss

def riemannian_divergence(matrix_a: torch.Tensor, matrix_b: torch.Tensor):
    matrix_a_inv = torch.inverse(matrix_a)
    ainvb = torch.bmm(matrix_a_inv, matrix_b)
    eigenvalues, _ = torch.symeig(ainvb, eigenvectors=True)
    log_eig = torch.log(eigenvalues)
    return (log_eig * log_eig).sum(dim=-1)

def closest_pd_matrix(matrix: torch.Tensor):
    eigenvalues, vectors = torch.symeig(matrix, eigenvectors=True)
    eigenvalues = relu(eigenvalues)
    diag_vals = torch.diag_embed(eigenvalues, offset=0, dim1=-2, dim2=-1)
    nearest_psd = torch.bmm(torch.bmm(vectors, diag_vals), vectors.transpose(-1, -2))
    if eigenvalues.min() < 1e-3:
        offset = 1e-3 * torch.eye(nearest_psd.size()[-1], device=nearest_psd.device, dtype=nearest_psd.dtype).unsqueeze(0).expand_as(nearest_psd)
        return nearest_psd + offset
    else:
        return nearest_psd

def log_det_divergence(matrix_a: torch.Tensor, matrix_b: torch.Tensor):
    ab_product = torch.bmm(matrix_a, matrix_b)
    ab_sum = (matrix_a + matrix_b) / 2
    logdet_sum = torch.logdet(ab_sum)
    logdet_product = (torch.logdet(ab_product) / 2)
    divergence = logdet_sum - logdet_product
    return divergence

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
    def __init__(self, embedding_model: nn.Module, features_list, in_manifold, out_manifold, out_dim, featurizer=None, featurizer_dim=0, dtype=torch.float, device=None, manifold_initialization=None, deltas=False):
        super(FeaturizedModelEmbedding, self).__init__()
        self.embedding_model = embedding_model
        self.embedding_model.to(device)
        self.features_list = features_list
        if featurizer is None:
            featurizer, featurizer_dim = get_canonical_glove_word_featurizer()
        self.featurizer = featurizer
        self.featurizer_dim = featurizer_dim
        self.out_manifold = out_manifold
        self.input_embedding, self.index_map = get_featurized_embedding(features_list, featurizer, featurizer_dim, dtype=dtype, device=device)
        in_manifold.proj_(self.input_embedding.weight)
        self.deltas = deltas
        if deltas:
            self.main_deltas = Embedding(torch.sum(self.index_map >= 0), out_dim)
            with torch.no_grad():
                self.main_deltas.weight /= 100

        self.additional_embeddings = None
        self.additional_index_map = None
        num_non_featurized = torch.sum(self.index_map < 0)
        if num_non_featurized > 0:
            self.additional_embeddings = ManifoldEmbedding(in_manifold, num_non_featurized, featurizer_dim)
            self.additional_embeddings.to(device)
            if deltas:
                self.additional_deltas = Embedding(num_non_featurized, out_dim)
                with torch.no_grad():
                    self.additional_deltas.weight /= 100

            if manifold_initialization is not None:
                with torch.no_grad():
                    num_perms = ceil(int(num_non_featurized) / self.input_embedding.weight.size(0))
                    for i in range(num_perms):
                        start_index = self.input_embedding.weight.size(0) * i
                        end_index = min(self.input_embedding.weight.size(0) * (i + 1), num_non_featurized)
                        perm = torch.randperm(self.input_embedding.weight.size(0), device=device)[:end_index - start_index]
                        self.additional_embeddings.weight.data[start_index : end_index] = self.input_embedding.weight[perm]
                    tangent_space = EuclideanManifold()
                    if self.additional_embeddings.weight.size(0) > 1000:
                        blocks = 50
                    else:
                        blocks = 1
                    block_size = ceil(self.additional_embeddings.weight.size(0) / blocks)
                    for i in tqdm(range(blocks)):
                        start_index = i * block_size
                        end_index = min((i + 1) * block_size, self.additional_embeddings.weight.size(0))
                        vector_offset = get_initialized_manifold_tensor(device, dtype, [end_index - start_index] + list(self.additional_embeddings.weight.size()[1:]), tangent_space, {
                                'global': {
                                    'init_func': 'normal_',
                                    'params': [-0.003, 0.003]
                                }
                            }, False)
                        self.additional_embeddings.weight[start_index:end_index] = in_manifold.retr(self.additional_embeddings.weight[start_index:end_index], vector_offset)

            self.additional_index_map = torch.zeros_like(self.index_map) - 1
            self.additional_index_map[self.index_map < 0] = torch.arange(0, num_non_featurized, dtype=self.index_map.dtype, device=device)
            if manifold_initialization is not None:
                print("Computing initialization of multi-word vertices...")
                vector_avg = None
                avg_count = 0
                done_processing = False
                word_index = 0
                average_counts = {}
                with torch.no_grad():
                    while not done_processing:
                        new_vector_indices = []
                        new_vector_tensors = []
                        new_average_indices = []
                        new_average_tensors = []
                        new_average_counts = []
                        indices = (self.index_map.clone() < 0).nonzero().squeeze(-1).cpu().detach().numpy()
                        for i in indices:
                            feature = features_list[i]
                            if len(feature.split(' ')) > max(1, word_index):  
                                word = feature.split(' ')[word_index]
                                featurized = self.featurizer(word)
                                if featurized is not None:
                                    featurized = featurized
                                    if i not in average_counts:
                                        new_vector_indices.append(i)
                                        new_vector_tensors.append(featurized)
                                        average_counts[i] = 1
                                    else:
                                        new_average_indices.append(i)
                                        new_average_tensors.append(featurized)
                                        average_counts[i] += 1
                                        new_average_counts.append(average_counts[i])

                        if len(new_vector_indices) > 0:
                            new_vector_indices = torch.LongTensor(new_vector_indices)
                            new_vector_indices = new_vector_indices.to(device)
                            new_vector_tensors = self.additional_embeddings.weight.new_tensor(new_vector_tensors)
                            in_manifold.proj_(new_vector_tensors)
                            self.additional_embeddings.weight.data[self.additional_index_map[new_vector_indices]] = new_vector_tensors

                        if len(new_average_indices) > 0:
                            new_average_indices = torch.LongTensor(new_average_indices)
                            new_average_indices = new_average_indices.to(device)
                            new_average_tensors = self.additional_embeddings.weight.new_tensor(new_average_tensors)
                            new_average_counts = self.additional_embeddings.weight.new_tensor(new_average_counts)
                            in_manifold.proj_(new_average_tensors)
                            original_average_tensors = self.additional_embeddings.weight.data[self.additional_index_map[new_average_indices]]
                            log_mu_x = in_manifold.log(original_average_tensors, new_average_tensors)
                            weighted = log_mu_x / new_average_counts.unsqueeze(-1)
                            self.additional_embeddings.weight.data[self.additional_index_map[new_average_indices]] = in_manifold.exp(original_average_tensors, weighted)
                        
                        if len(new_vector_indices) == 0 and len(new_average_indices) == 0:
                            done_processing = True

                        word_index += 1
                    single_average_indices = torch.LongTensor([index for index in average_counts if average_counts[index] == 1])
                    single_average_indices.to(device)

                    vector_offset = get_initialized_manifold_tensor(device, dtype,
                            self.additional_embeddings.weight.data[self.additional_index_map[single_average_indices]].size(),
                            tangent_space, {
                            'global': {
                                'init_func': 'normal_',
                                'params': [-0.003, 0.003]
                            }
                        }, False)
                    self.additional_embeddings.weight.data[self.additional_index_map[single_average_indices]] = in_manifold.retr(
                            self.additional_embeddings.weight.data[self.additional_index_map[single_average_indices]], vector_offset)


        if device is not None:
            self.to(device)

    def forward(self, x):
        out = self.embedding_model(self.map_to_input_embeddings(x))
        if self.deltas:
            out = self.out_manifold.exp(out, self.map_to_deltas(x))
        return out

    def forward_featurize(self, feature):
        if feature in self.features_list:
            in_tensor = torch.LongTensor([self.features_list.index(feature)])
            in_tensor = in_tensor.to(self.input_embedding.weight.device)
            return self.forward(in_tensor)
        feature = self.featurizer(feature)
        if feature is None:
            in_tensor = torch.LongTensor([0])
            in_tensor = in_tensor.to(self.input_embedding.weight.device)
            return self.forward(in_tensor)

        featurized = torch.as_tensor([feature], dtype=self.input_embedding.weight.dtype, device=self.input_embedding.weight.device)
        return self.embedding_model(featurized)

    def map_to_input_embeddings(self, x):
        if self.additional_embeddings is not None:
            return torch.where(self.index_map[x].unsqueeze(-1) > -1, self.input_embedding(self.index_map[x].clamp(min=0)), self.additional_embeddings(self.additional_index_map[x].clamp(min=0)))
        else:
            return self.input_embedding(x)

    def map_to_deltas(self, x):
        if self.additional_deltas is not None:
            return torch.where(self.index_map[x].unsqueeze(-1) > -1, self.main_deltas(self.index_map[x].clamp(min=0)), self.additional_deltas(self.additional_index_map[x].clamp(min=0)))
        else:
            return self.main_deltas(x)


    def get_embedding_matrix_input(self, num_blocks=50):
        block_size = ceil(self.input_embedding.weight.data.size(0) / num_blocks)
        out_blocks = []
        print("Processing embedding matrix...")
        for i in tqdm(range(num_blocks)):
            start_index = i * block_size
            end_index = min((i + 1) * block_size, self.input_embedding.weight.data.size(0))
            out_blocks.append(self.embedding_model(self.input_embedding.weight.data[start_index:end_index]).cpu())
        out = torch.cat(out_blocks)
        return out

    def get_embedding_matrix(self, num_blocks=50):
        block_size = ceil(self.index_map.size(0) / num_blocks)
        out_blocks = []
        print("Processing embedding matrix...")
        for i in tqdm(range(num_blocks)):
            start_index = i * block_size
            end_index = min((i + 1) * block_size, self.index_map.size(0))
            out_blocks.append(self.forward(torch.arange(start_index, end_index, dtype=torch.long, device=self.index_map.device)).cpu())
        out = torch.cat(out_blocks)
        return out

    def get_savable_model(self):
        return self.embedding_model

    def get_additional_embeddings(self):
        return self.additional_embeddings


def get_canonical_glove_sentence_featurizer():
    embedder = GloveSentenceEmbedder.canonical()
    return lambda sent : embedder.embed(SimpleSentence.from_text(sent), l2_normalize=False), embedder.dim

def get_canonical_glove_word_featurizer():
    glove = Glove.canonical()
    def get_glove_or_none(string):
        idx = glove.lookup_word(string.lower())
        if idx >= glove.token_mapper.mapped_output_size():
            return glove.get_embedding_at_index(idx)
        else:
            return None

    return lambda w: get_glove_or_none(w), glove.embedding_dim

def get_featurized_embedding(features: List, featurizer, featurizer_dim, dtype=torch.float, device=None, verbose=True):
    embeddings_list = np.empty((len(features), featurizer_dim))
    index_map = np.empty((len(features)), dtype=np.int64)
    iterator = range(len(features))
    count = 0
    if verbose:
        print("Processing features of dataset...") 
        iterator = tqdm(iterator)
    for i in iterator:
        featurized = featurizer(features[i])
        if featurized is not None:
            embeddings_list[count] = featurizer(features[i])
            index_map[i] = count
            count += 1
        else:
            index_map[i] = -1
    embeddings_list = embeddings_list[0:count]
    return Embedding.from_pretrained(torch.as_tensor(np.array(embeddings_list), dtype=dtype, device=device)), torch.as_tensor(index_map, dtype=torch.long, device=device)

