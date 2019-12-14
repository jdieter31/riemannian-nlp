import nmslib
import numpy
import torch
from manifolds import RiemannianManifold
from tqdm import tqdm
from math import ceil

class ManifoldNNS:
    def __init__(self, data_points: torch.Tensor, manifold: RiemannianManifold, samples_for_pole: int=10000):
        self.manifold = manifold
        self.compute_index(data_points, samples_for_pole)

    def compute_index(self, data_points: torch.Tensor, samples_for_pole: int=10000):
        if samples_for_pole == 0:
            samples_for_pole = data_points.size(0)
        perm = torch.randperm(data_points.size(0))
        idx = perm[:min(samples_for_pole, perm.size(0))]
        self.pole = compute_pole(data_points[idx], self.manifold)
        num_blocks = 50
        block_size = ceil(data_points.size(0) / num_blocks)
        print("Projecting to Euclidean space for nns:")

        self.index = nmslib.init(method="hnsw", space="l2")
        self.data_embedding = data_points

        pole_batch = self.pole.unsqueeze(0).expand_as(data_points[:block_size])

        for i in tqdm(range(num_blocks)):
            start_index = i * block_size
            end_index = min((i + 1) * block_size, data_points.size(0))
            self.data_embedding[start_index:end_index] = self.manifold.log(pole_batch[0: end_index-start_index], data_points[start_index:end_index])
            import gc ; gc.collect()

        self.index.addDataPointBatch(self.data_embedding.cpu().detach().numpy())
        print("Computing nns index:")
        self.index.createIndex({'post': 0}, print_progress=True)

        
    def knn_query_batch_vectors(self, data, k=10, num_threads=4, log_space=False):
        pole_batch = self.pole.unsqueeze(0).expand_as(data)
        if log_space:
            data_embedding = data.cpu().detach().numpy()
        else:
            data_embedding = self.manifold.log(pole_batch, data).cpu().detach().numpy()
        return self.index.knnQueryBatch(data_embedding, k, num_threads)

    def knn_query_batch_indices(self, indices, k=10, num_threads=4):
        return self.knn_query_batch_vectors(self.data_embedding[indices], k, num_threads, log_space=True)

    def knn_query_all(self, k=10, num_threads=4):
        return self.knn_query_batch_indices(
            torch.arange(0, self.data_embedding.size()[0],
                        dtype=torch.long, device=self.data_embedding.device), k, num_threads)


def compute_pole(data_samples: torch.Tensor, manifold: RiemannianManifold):
    running_pole = data_samples[0].clone()
    for i in range(data_samples.size()[0] - 1):
        log_mu_x = manifold.log(running_pole, data_samples[i + 1])
        weighted = log_mu_x / (i + 2)
        running_pole = manifold.exp(running_pole, weighted)
    return running_pole

def compute_pole_batch(data: torch.Tensor, manifold: RiemannianManifold, samples_per_pole=1000, num_poles=15):
    permuted_data = data.new_empty([num_poles, min(samples_per_pole, data.size(0)), data.size()[-1]])
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


