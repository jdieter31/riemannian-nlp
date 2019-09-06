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
from graph_tool.all import Graph
from manifold_nns import ManifoldNNS
from tqdm import tqdm

# Thread safe random number generation.  libcpp doesn't expose rand_r...
cdef unsigned long rand_r(unsigned long* seed) nogil:
    seed[0] = (seed[0] * 1103515245) % <unsigned long>pow(2, 32) + 12345
    return seed[0] % RAND_MAX

cdef class BatchedDataset:
    cdef public list objects
    cdef public list features
    cdef public object idx
    cdef object manifold
    cdef int n_graph_neighbors, n_manifold_neighbors, n_rand_neighbors, N, batch_size, current, manifold_nn_k, num_workers
    cdef object queue
    cdef object graph
    cdef long[:] graph_neighbors
    cdef long[:] graph_neighbors_indices
    cdef long[:] manifold_neighbors
    cdef long[:] manifold_neighbors_indices
    cdef long[:] perm
    cdef long[:,:] graph_neighbor_permutations
    cdef list threads

    def __cinit__(self, idx, objects, manifold, n_graph_neighbors, n_manifold_neighbors, n_rand_neighbors, 
            batch_size, num_workers, manifold_nn_k=60, features=None):
        self.idx = idx
        self.objects = objects
        self.manifold = manifold
        self.n_graph_neighbors = n_graph_neighbors
        self.n_manifold_neighbors = n_manifold_neighbors
        self.N = len(objects)
        self.batch_size = batch_size
        self.features = features
        self.queue = queue.Queue(maxsize=num_workers)
        self.graph = Graph(directed=False)
        self.graph.add_edge_list(idx)
        self.manifold_nn_k = manifold_nn_k
        self.num_workers = num_workers
        print("Processing graph neighbors...")
        all_graph_neighbors = [self.graph.get_all_neighbors(i) for i in tqdm(range(self.N))]
        list_size, index_size = self.get_init_size_1d_memview_numpy_list(all_graph_neighbors)
        self.graph_neighbors = np.concatenate(all_graph_neighbors)
        self.graph_neighbors_indices = np.empty([index_size], dtype=np.int64)
        self.numpy_list_to_1d_memview(all_graph_neighbors, self.graph_neighbors, self.graph_neighbors_indices)
    
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

    def refresh_manifold_nn(self, manifold_embedding, manifold):
        manifold_nns = ManifoldNNS(manifold_embedding, manifold)
        print("\nQuerying near neighbors...")
        nns = manifold_nns.knn_query_all(self.manifold_nn_k, self.num_workers)
        print("Processing near neighbor data...")
        all_manifold_neighbors = [nns[i][0][1:].astype(np.int64) for i in range(self.N)]
        list_size, index_size = self.get_init_size_1d_memview_numpy_list(all_manifold_neighbors)
        self.manifold_neighbors = np.concatenate(all_manifold_neighbors)
        self.manifold_neighbors_indices = np.empty([index_size], dtype=np.int64)
        self.numpy_list_to_1d_memview(all_manifold_neighbors, self.manifold_neighbors, self.manifold_neighbors_indices)
        print("Finished processing near neighbor data.")

    cpdef initialize_graph_perms(self):
        graph_neighbor_permutations = []
        cdef int max_dif = 0
        cdef int i = 0
        while i < self.N:
            num_neighbors = self.graph_neighbors_indices[i + 1] - self.graph_neighbors_indices[i]
            dif = num_neighbors - self.n_graph_neighbors
            if dif > max_dif:
                max_dif = dif
            i = i+1
        
        for i in range(max_dif):
            graph_neighbor_permutations.append(np.random.permutation(i + 1 + self.n_graph_neighbors)[:self.n_graph_neighbors])

        if len(graph_neighbor_permutations) > 0:
            self.graph_neighbor_permutations = np.stack(graph_neighbor_permutations)

    def __iter__(self):
        self.perm = np.random.permutation(self.N)
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

        while self.current < self.N:
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
        return int(np.ceil(float(self.N) / self.batch_size))

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
        cdef int neighbors_index, neighbors_length, size_dif
        cdef unordered_set[long] excluded_samples
        cdef unsigned long seed

        seed = i
        j = 0
        while j < self.batch_size and i + j < self.perm.shape[0]:
            ntries = 0
            vertex = self.perm[i + j]
            vertices[j, 0] = vertex
            neighbors_index = self.graph_neighbors_indices[vertex]
            neighbors_length = self.graph_neighbors_indices[vertex + 1] - neighbors_index
            k = 0

            extra_rand_samples = 0
            if neighbors_length < self.n_graph_neighbors:
                extra_rand_samples = self.n_graph_neighbors - neighbors_length
            total_graph_samples = self.n_graph_neighbors - extra_rand_samples
            excluded_samples = unordered_set[long]()
            k = 0
            while k < neighbors_length:
                excluded_samples.insert(self.graph_neighbors[neighbors_index + k])
                k = k + 1
            k = 1
            size_dif = neighbors_length - self.n_graph_neighbors
            while k < 1+total_graph_samples:
                if size_dif > 0:
                    vertices[j, k] = self.graph_neighbors[neighbors_index + self.graph_neighbor_permutations[size_dif - 1][k - 1]]
                else:
                    vertices[j, k] = self.graph_neighbors[neighbors_index + k - 1]
                graph_dists[j,k - 1] = 1
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

