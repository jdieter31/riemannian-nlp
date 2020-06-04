from math import ceil, sqrt

import faiss
import numpy as np
import torch
from tqdm import tqdm

from .config.config_loader import get_config
from .manifolds import RiemannianManifold


class ManifoldNNS:
    def __init__(self, data_points: torch.Tensor, manifold: RiemannianManifold,
                 samples_for_pole: int = 10000):
        self.manifold = manifold
        self.compute_index(data_points, samples_for_pole)

    def compute_index(self, data_points: torch.Tensor, samples_for_pole: int = 10000):
        data_points = data_points.cpu()
        if samples_for_pole == 0:
            samples_for_pole = data_points.size(0)
        perm = torch.randperm(data_points.size(0))
        idx = perm[:min(samples_for_pole, perm.size(0))]
        self.pole = compute_pole(data_points[idx], self.manifold)

        tqdm.write("Creating nns index")
        ivf_size = 2 ** (ceil(4 * sqrt(data_points.size(0)) - 1)).bit_length()
        index_flat = faiss.index_factory(data_points.size(-1),
                                         "Flat") #f"PCAR16,IVF{ivf_size},SQ4")

        general_config = get_config().general
        _use_gpu: bool = general_config.gpu >= 0
        # make it into a gpu index
        if _use_gpu:
            res = faiss.StandardGpuResources()
            # make it into a gpu index
            self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)

            params = faiss.GpuParameterSpace()
        else:
            self.index = index_flat
            params = faiss.ParameterSpace()

        # params.set_index_parameter(self.index, 'nprobe', 100)
        params.initialize(self.index)

        num_blocks = 200
        block_size = ceil(data_points.size(0) / num_blocks)
        num_blocks = ceil(data_points.size(0) / block_size)
        self.data_embedding = data_points
        pole_batch = self.pole.unsqueeze(0).expand_as(data_points[:block_size])

        for i in tqdm(range(num_blocks), desc="Euclidean Project",
                      dynamic_ncols=True):
            start_index = i * block_size
            end_index = min((i + 1) * block_size, data_points.size(0))
            self.data_embedding[start_index:end_index] = self.manifold.log(
                pole_batch[0: end_index - start_index], data_points[start_index:end_index])

        tqdm.write("Training Index")
        train_size = int(20 * sqrt(data_points.size(0)))
        perm = torch.randperm(data_points.size(0))
        train_points = \
            self.data_embedding.cpu().detach()[perm[:train_size]].numpy()

        self.index.train(train_points)
        tqdm.write("Adding Vectors to Index")
        self.index.add(self.data_embedding.cpu().detach().numpy())

    def knn_query_batch_vectors(self, data, k=10, log_space=False):
        pole_batch = self.pole.unsqueeze(0).expand_as(data)
        if log_space:
            data_embedding = data.cpu().detach().numpy()
        else:
            data_embedding = self.manifold.log(pole_batch, data).cpu().detach().numpy()
        return self.index.search(data_embedding, k)

    def add_vectors(self, data):
        pole_batch = self.pole.unsqueeze(0).expand_as(data)
        data_embedding = self.manifold.log(pole_batch, data).cpu().detach().numpy()
        self.index.add(data_embedding)

    def knn_query_batch_indices(self, indices, k=10):
        return self.knn_query_batch_vectors(self.data_embedding[indices], k, log_space=True)

    def knn_query_all(self, k=10):
        block_size = self.data_embedding.size()[0] // 3
        num_blocks = ceil(self.data_embedding.size()[0] / block_size)
        dists, nns = None, None
        for i in tqdm(range(num_blocks), desc="knn_query", dynamic_ncols=True):
            start_index = i * block_size
            end_index = min((i + 1) * block_size, self.data_embedding.size()[0])
            block_dists, block_nns = self.knn_query_batch_indices(
                torch.arange(start_index, end_index,
                             dtype=torch.long, device=self.data_embedding.device), k)
            if dists is None:
                dists, nns = block_dists, block_nns
            else:
                dists = np.concatenate((dists, block_dists))
                nns = np.concatenate((nns, block_nns))
        return dists, nns


def compute_pole(data_samples: torch.Tensor, manifold: RiemannianManifold):
    running_pole = data_samples[0].clone()
    for i in range(data_samples.size()[0] - 1):
        log_mu_x = manifold.log(running_pole, data_samples[i + 1])
        weighted = log_mu_x / (i + 2)
        running_pole = manifold.exp(running_pole, weighted)
    return running_pole


def compute_pole_batch(data: torch.Tensor, manifold: RiemannianManifold, samples_per_pole=1000,
                       num_poles=15):
    permuted_data = data.new_empty(
        [num_poles, min(samples_per_pole, data.size(0)), data.size()[-1]])
    for i in range(num_poles):
        perm = torch.randperm(data.size(0))
        idx = perm[:min(samples_per_pole, perm.size(0))]
        permuted_data[i] = data[idx]
    running_poles = permuted_data[:, 0, :].clone()
    for i in range(permuted_data.size()[1] - 1):
        log_mu_x = manifold.log(running_poles, permuted_data[:, i + 1, :])
        weighted = log_mu_x / (i + 2)
        running_poles = manifold.exp(running_poles, weighted)
    return running_poles
