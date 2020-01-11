# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cimport numpy as npc
cimport cython

import numpy as np
import torch
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libc.math cimport pow
from libc.stdlib cimport rand, RAND_MAX, malloc, free
from libc.stdio cimport printf
import threading
import queue
import copy
from graph_tool.all import Graph
from ..manifold_nns import ManifoldNNS
from tqdm import tqdm
from math import ceil

# Thread safe random number generation.  libcpp doesn't expose rand_r...
cdef unsigned long rand_r(unsigned long* seed) nogil:
    seed[0] = (seed[0] * 1103515245) % <unsigned long>pow(2, 32) + 12345
    return seed[0] % RAND_MAX

cdef class BatchedDataset:
    cdef public list objects
    cdef public list features
    cdef public object weights
    cdef public object idx
    cdef public object manifold
    cdef public int n_graph_neighbors, n_manifold_neighbors, n_rand_neighbors, N, N_non_trivial, N_non_trivial_train, batch_size, current, manifold_nn_k, num_workers, nn_workers
    cdef public float data_fraction
    cdef public bool is_eval
    cdef public bool compute_train_ranks
    cdef object queue
    cdef object graph
    cdef object train_graph
    cdef long[:] graph_neighbors
    cdef long[:] graph_neighbors_indices
    cdef long[:] train_graph_neighbors
    cdef long[:] train_graph_neighbors_indices
    cdef float[:] graph_neighbor_weights
    cdef float[:] train_graph_neighbor_weights
    cdef long[:] manifold_neighbors
    cdef long[:] manifold_neighbors_indices
    cdef long[:] perm
    cdef long[:] non_empty_indices
    cdef long[:] non_empty_indices_train
    cdef long[:,:] graph_neighbor_permutations
    cdef unordered_map[int, int] graph_neighbor_difs
    cdef list threads

    def __cinit__(self, idx, objects, weights, manifold, n_graph_neighbors, n_manifold_neighbors, n_rand_neighbors, 
            batch_size, num_workers, nn_workers, manifold_nn_k=60, features=None, train_edges=None, train_weights=None):
        self.idx = idx
        self.objects = objects
        self.manifold = manifold
        self.weights = weights
        self.n_graph_neighbors = n_graph_neighbors
        self.n_manifold_neighbors = n_manifold_neighbors
        self.n_rand_neighbors = n_rand_neighbors
        self.N = len(objects)
        self.batch_size = batch_size
        self.features = features
        self.queue = queue.Queue(maxsize=num_workers)
        self.graph = Graph(directed=False)
        self.graph.add_vertex(self.N)
        idx_weights = [[edge[0], edge[1], weight] for edge, weight in zip(self.idx, self.weights)]
        weight_property = self.graph.new_edge_property("float")
        eprops = [weight_property]  
        self.graph.add_edge_list(idx_weights, eprops=eprops)
        self.manifold_nn_k = manifold_nn_k
        self.num_workers = num_workers
        self.nn_workers = nn_workers
        self.data_fraction = 1
        print("Processing graph neighbors...")
        all_graph_neighbors = []
        all_graph_weights = []
        non_empty_vertices = []
        empty = 0
        for i in tqdm(range(self.N)):
            in_edges = self.graph.get_in_edges(i, [weight_property])
            out_edges = self.graph.get_out_edges(i, [weight_property])
            if in_edges.size + out_edges.size == 0:
                empty += 1
            else:
                non_empty_vertices.append(i)
                if in_edges.size == 0:
                    all_graph_neighbors.append(out_edges[:, 1].astype(np.int64))
                    all_graph_weights.append(out_edges[:, 2].astype(np.float32))
                elif out_edges.size == 0:
                    all_graph_neighbors.append(in_edges[:, 1].astype(np.int64))
                    all_graph_weights.append(in_edges[:, 2].astype(np.float32))
                else:
                    all_graph_neighbors.append(np.concatenate([in_edges[:, 0], out_edges[:, 1]]).astype(np.int64))
                    all_graph_weights.append(np.concatenate([in_edges[:, 2], out_edges[:, 2]]).astype(np.float32))
            
        self.non_empty_indices = np.array(non_empty_vertices, dtype=np.int64)
        self.N_non_trivial = self.N - empty

        list_size, index_size = self.get_init_size_1d_memview_numpy_list(all_graph_neighbors)
        self.graph_neighbors = np.concatenate(all_graph_neighbors)
        self.graph_neighbor_weights = np.concatenate(all_graph_weights)
        self.graph_neighbors_indices = np.empty([index_size], dtype=np.int64)
        self.numpy_list_to_1d_memview(all_graph_neighbors, self.graph_neighbors, self.graph_neighbors_indices)
        
        if train_edges is not None:
            self.is_eval = True 
            self.compute_train_ranks = False

            self.train_graph = Graph(directed=False)
            self.train_graph.add_vertex(self.N)
            train_idx_weights = [[edge[0], edge[1], weight] for edge, weight in zip(train_edges, train_weights)]
            train_weight_property = self.train_graph.new_edge_property("float")
            eprops = [train_weight_property]  
            self.train_graph.add_edge_list(train_idx_weights, eprops=eprops)

            print("Processing train graph neighbors...")
            all_graph_neighbors = []
            all_graph_weights = []
            non_empty_vertices = []
            empty = 0
            for i in tqdm(range(self.N)):
                in_edges = self.train_graph.get_in_edges(i, [train_weight_property])
                out_edges = self.train_graph.get_out_edges(i, [train_weight_property])
                if in_edges.size + out_edges.size == 0:
                    empty += 1
                else:
                    non_empty_vertices.append(i)
                    if in_edges.size == 0:
                        all_graph_neighbors.append(out_edges[:, 1].astype(np.int64))
                        all_graph_weights.append(out_edges[:, 2].astype(np.float32))
                    elif out_edges.size == 0:
                        all_graph_neighbors.append(in_edges[:, 1].astype(np.int64))
                        all_graph_weights.append(in_edges[:, 2].astype(np.float32))
                    else:
                        all_graph_neighbors.append(np.concatenate([in_edges[:, 0], out_edges[:, 1]]).astype(np.int64))
                        all_graph_weights.append(np.concatenate([in_edges[:, 2], out_edges[:, 2]]).astype(np.float32))
                
            self.non_empty_indices_train = np.array(non_empty_vertices, dtype=np.int64)
            self.N_non_trivial_train = self.N - empty

            list_size, index_size = self.get_init_size_1d_memview_numpy_list(all_graph_neighbors)
            self.train_graph_neighbors = np.concatenate(all_graph_neighbors)
            self.train_graph_neighbor_weights = np.concatenate(all_graph_weights)
            self.train_graph_neighbors_indices = np.empty([index_size], dtype=np.int64)
            self.numpy_list_to_1d_memview(all_graph_neighbors, self.train_graph_neighbors, self.train_graph_neighbors_indices)
        else:
            self.is_eval = False



    @classmethod
    def initialize_eval_dataset(cls, train_dataset, eval_batch_size, n_eval_neighbors, max_graph_neighbors,
            eval_workers, eval_nn_workers, manifold_neighbors=0, eval_edges=None, eval_weights=None):
        if eval_edges is None:
            return BatchedDataset(train_dataset.idx, train_dataset.objects, train_dataset.weights, train_dataset.manifold,
                    max_graph_neighbors, manifold_neighbors, n_eval_neighbors - max_graph_neighbors, eval_batch_size, eval_workers,
                    eval_nn_workers, features=train_dataset.features)
        else:
            return BatchedDataset(eval_edges, train_dataset.objects, eval_weights, train_dataset.manifold,
                    max_graph_neighbors, manifold_neighbors, n_eval_neighbors - max_graph_neighbors, eval_batch_size, eval_workers,
                    eval_nn_workers, features=train_dataset.features, train_edges=train_dataset.idx, train_weights=train_dataset.weights)
    
    def get_init_size_1d_memview_numpy_list(self, numpy_list):
        list_size = sum(array.shape[0] for array in numpy_list)
        index_size = len(numpy_list) + 1
        return list_size, index_size

    cpdef numpy_list_to_1d_memview(self, numpy_list, long[:] memview, long[:] indices):
        cdef int current = 0
        cdef int i = 0
        while i < len(numpy_list):
            indices[i] = current
            current = current + numpy_list[i].shape[0]
            i = i + 1
        indices[i] = current

    def refresh_manifold_nn(self, manifold_embedding, manifold, return_nns=False, manifold_nns=None):
        if manifold_nns is None:
            manifold_nns = ManifoldNNS(manifold_embedding, manifold)
        print("\nQuerying near neighbors...")
        _, nns = manifold_nns.knn_query_all(self.manifold_nn_k, self.nn_workers)
        print("Processing near neighbor data...")
        all_manifold_neighbors = [nns[i][1:].astype(np.int64) for i in range(self.N)]
        list_size, index_size = self.get_init_size_1d_memview_numpy_list(all_manifold_neighbors)
        self.manifold_neighbors = np.concatenate(all_manifold_neighbors)
        self.manifold_neighbors_indices = np.empty([index_size], dtype=np.int64)
        self.numpy_list_to_1d_memview(all_manifold_neighbors, self.manifold_neighbors, self.manifold_neighbors_indices)
        print("Finished processing near neighbor data.")
        if return_nns:
            return manifold_nns

    cpdef initialize_graph_perms(self):
        graph_neighbor_permutations = []
        difs = []
        self.graph_neighbor_difs.clear()
        cdef int i = 0
        if self.is_eval:
            n_non_trivial = self.N_non_trivial_train if self.compute_train_ranks else self.N_non_trivial
            graph_neighbors_indices = self.train_graph_neighbors_indices if self.compute_train_ranks else self.graph_neighbors_indices
        else:
            n_non_trivial = self.N_non_trivial
            graph_neighbors_indices = self.graph_neighbors_indices
        while i < n_non_trivial:
            num_neighbors = graph_neighbors_indices[i + 1] - graph_neighbors_indices[i]
            dif = num_neighbors - self.n_graph_neighbors
            if dif > 0 and dif not in difs:
                difs.append(dif)
                self.graph_neighbor_difs[dif] = len(difs) - 1
            i = i+1

        for dif in difs:
            graph_neighbor_permutations.append(np.random.permutation(dif + self.n_graph_neighbors)[:self.n_graph_neighbors])

        if len(graph_neighbor_permutations) > 0:
            self.graph_neighbor_permutations = np.stack(graph_neighbor_permutations)

    def __iter__(self):
        n_non_trivial = self.N_non_trivial_train if self.is_eval and self.compute_train_ranks else self.N_non_trivial
        self.perm = np.random.permutation(n_non_trivial)
        self.initialize_graph_perms()
        self.current = 0
        self.threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker, args=(i,))
            t.start()
            self.threads.append(t)
        return self

    cpdef _worker(self, i):
        cdef long [:,:] memview
        cdef float [:,:] graph_memview
        cdef long count
        cdef int current
        
        n_non_trivial = self.N_non_trivial_train if self.is_eval and self.compute_train_ranks else self.N_non_trivial
        while self.current < ceil(n_non_trivial * self.data_fraction):
            current = self.current
            self.current += self.batch_size
            ix = torch.LongTensor(self.batch_size, 1 + self.n_graph_neighbors + self.n_manifold_neighbors + self.n_rand_neighbors)
            graph_dists = torch.zeros((self.batch_size, self.n_graph_neighbors + self.n_manifold_neighbors + self.n_rand_neighbors))
            graph_dists += 2
            memview = ix.numpy()
            graph_memview = graph_dists.numpy()
            with nogil:
                count = self._getbatch(current, memview, graph_memview)
            if count < self.batch_size:
                ix = ix.narrow(0, 0, count)
                graph_dists = graph_dists.narrow(0, 0, count)
            self.queue.put((ix, graph_dists))

        self.queue.put(i)

    def iter(self):
        return self.__iter__()

    def __len__(self):
        n_non_trivial = self.N_non_trivial_train if self.is_eval and self.compute_train_ranks else self.N_non_trivial
        return int(np.ceil(float(ceil(self.data_fraction * n_non_trivial)) / self.batch_size))

    def __next__(self):
        return self.next()

    def next(self):
        '''
        Python visible function for indexing the dataset.  This first
        allocates a tensor, and then modifies it in place with `_getitem`
        Args:
            idx (int): index into the dataset
        '''
        size = self.queue.qsize()
        if size == 0 and all([not(t.is_alive()) for t in self.threads]):
            # No more items in queue and we've joined with all worker threads
            raise StopIteration
        item = self.queue.get()
        if isinstance(item, int):
            self.threads[item].join()  # Thread `item` is finished, join with it...
            return self.next()  # try again...
        return item

    cdef public long _getbatch(self, int i, long[:,:] vertices, float[:,:] graph_dists) nogil:
        '''
        Fast internal C method for indexing the dataset/negative sampling
        Args:
            i (int): Index into the dataset
            vertices (long [:,:]) - A C memoryview of the vertices tensor that we will
                return to Python
            graph_dists (float [:,:]) - A C memoryview of the graph_dists tensor that we will
                return to Python
        '''
        cdef int vertex, extra_rand_samples, total_graph_samples, new_vertex
        cdef int j, k, l, current_index
        cdef int neighbors_index, neighbors_length, size_dif, permutation_index, train_neighbors_index, train_neighbors_length
        cdef unordered_set[long] excluded_samples
        cdef unsigned long seed

        seed = i
        j = 0
        while j < self.batch_size and i + j < self.perm.shape[0]:
            ntries = 0
            if self.is_eval and self.compute_train_ranks:
                vertex = self.non_empty_indices_train[self.perm[i + j]]
                vertices[j, 0] = vertex
                neighbors_index = self.train_graph_neighbors_indices[self.perm[i + j]]
                neighbors_length = self.train_graph_neighbors_indices[self.perm[i + j] + 1] - neighbors_index
            else:
                vertex = self.non_empty_indices[self.perm[i + j]]
                vertices[j, 0] = vertex
                neighbors_index = self.graph_neighbors_indices[self.perm[i + j]]
                neighbors_length = self.graph_neighbors_indices[self.perm[i + j] + 1] - neighbors_index
            k = 0

            extra_rand_samples = 0
            if neighbors_length < self.n_graph_neighbors:
                extra_rand_samples = self.n_graph_neighbors - neighbors_length
            total_graph_samples = self.n_graph_neighbors - extra_rand_samples
            excluded_samples = unordered_set[long]()
            excluded_samples.insert(vertex)
            if (not self.is_eval) or (not self.compute_train_ranks):
                k = 0
                while k < neighbors_length:
                    excluded_samples.insert(self.graph_neighbors[neighbors_index + k])
                    k = k + 1
            if self.is_eval:
                train_neighbors_index = self.train_graph_neighbors[self.perm[i + j]]
                train_neighbors_length = self.train_graph_neighbors_indices[self.perm[i + j] + 1] - train_neighbors_index
                k = 0
                while k < neighbors_length:
                    excluded_samples.insert(self.train_graph_neighbors[train_neighbors_index + k])
                    k = k + 1

            k = 1
            size_dif = neighbors_length - self.n_graph_neighbors
            while k < 1+total_graph_samples:
                if size_dif > 0:
                    permutation_index = self.graph_neighbor_difs[size_dif]
                    if self.is_eval and self.compute_train_ranks:
                        vertices[j, k] = self.train_graph_neighbors[neighbors_index + self.graph_neighbor_permutations[permutation_index][k - 1]]
                        graph_dists[j,k - 1] = 1 + 1 / (self.train_graph_neighbor_weights[neighbors_index + self.graph_neighbor_permutations[permutation_index][k - 1]] + 2)
                    else:
                        vertices[j, k] = self.graph_neighbors[neighbors_index + self.graph_neighbor_permutations[permutation_index][k - 1]]
                        graph_dists[j,k - 1] = 1 + 1 / (self.graph_neighbor_weights[neighbors_index + self.graph_neighbor_permutations[permutation_index][k - 1]] + 2)
                else:
                    if self.is_eval and self.compute_train_ranks:
                        vertices[j, k] = self.train_graph_neighbors[neighbors_index + k - 1]
                        graph_dists[j,k - 1] = 1 + 1 / (self.train_graph_neighbor_weights[neighbors_index + k - 1] + 2)
                    else:
                        vertices[j, k] = self.graph_neighbors[neighbors_index + k - 1]
                        graph_dists[j,k - 1] = 1 + 1 / (self.graph_neighbor_weights[neighbors_index + k - 1] + 2)
                k = k + 1
                current_index = k
            if self.n_manifold_neighbors > 0:
                neighbors_index = self.manifold_neighbors_indices[vertex]
                neighbors_length = self.manifold_neighbors_indices[vertex + 1] - neighbors_index
                k = 0
                l = 0
                while k < self.n_manifold_neighbors and l < neighbors_length:
                    new_vertex = self.manifold_neighbors[neighbors_index + l]
                    if excluded_samples.find(new_vertex) == excluded_samples.end():
                        vertices[j, k + current_index] = new_vertex
                        k = k + 1
                        excluded_samples.insert(new_vertex)
                    l = l + 1
                current_index = current_index + k
                if k < self.n_manifold_neighbors:
                    extra_rand_samples = extra_rand_samples + self.n_manifold_neighbors - k
            k = 0
            while k < self.n_rand_neighbors + extra_rand_samples:
                new_vertex = <long>(<double>rand_r(&seed) / <double>RAND_MAX * self.N)
                if excluded_samples.find(new_vertex) == excluded_samples.end():
                    vertices[j, k + current_index] = new_vertex
                    k = k + 1
                    excluded_samples.insert(new_vertex)
                
            j = j + 1
        return j

