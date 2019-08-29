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
    cdef long[:] graph_neighbor_permutations
    cdef long[:] graph_neighbor_permutations_indices
    cdef list threads

    def __cinit__(self, idx, objects, manifold, n_graph_neighbors, n_manifold_neighbors, n_rand_neighbors, 
            batch_size, num_workers, manifold_nn_k=60, features=None):
        '''
        Create a dataset for training Hyperbolic embeddings.  Rather than
        allocating many tensors for individual dataset items, we instead
        produce a single batch in each iteration.  This allows us to do a single
        Tensor allocation for the entire batch and filling it out in place.
        Args:
            idx (ndarray[ndims=2]):  Indexes of objects corresponding to co-occurrence.
                I.E. if `idx[0, :] == [4, 19]`, then item 4 co-occurs with item 19
            weights (ndarray[ndims=1]): Weights for each co-occurence.  Corresponds
                to the number of times a pair co-occurred.  (Equal length to `idx`)
            nnegs (int): Number of negative samples to produce with each positive
            objects (list[str]): Mapping from integer ID to hashtag string
            nnegs (int): Number of negatives to produce with each positive
            batch_size (int): Size of each minibatch
            num_workers (int): Number of threads to use to produce each batch
            burnin (bool): ???
            features (list[any]): Features for each vertex in the graph
            sample_data (str): Either 'targets' or 'graph_dist'. If 'targets' batches will give
                a tensor showing which of the samples is the targeted positive example. If
                'graph_dist' batches will give a tensor 
        '''
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
        all_graph_neighbors = [self.graph.get_all_neighbors(i) for i in range(self.N)]
        list_size, index_size = self.get_init_size_1d_memview_numpy_list(all_graph_neighbors)
        self.graph_neighbors = np.empty([list_size], dtype=np.int64)
        self.graph_neighbors_indices = np.empty([index_size], dtype=np.int64)
        self.numpy_list_to_1d_memview(all_graph_neighbors, self.graph_neighbors, self.graph_neighbors_indices)
    
    def get_init_size_1d_memview_numpy_list(self, numpy_list):
        list_size = sum(array.shape[0] for array in numpy_list)
        index_size = len(numpy_list) + 1
        return list_size, index_size

    cpdef numpy_list_to_1d_memview(self, numpy_list, long[:] memview, long[:] indices):
        np.copyto(np.asarray(memview), np.concatenate(numpy_list))
        cdef int current = 0
        cdef int i = 0
        while i < len(numpy_list):
            indices[i] = current
            current = current + numpy_list[i].shape[0]
            i = i + 1
        indices[i] = current

    cdef long[:] access_1d_memview_list(self, int i, long[:] memview, long[:] indices) nogil:
        return memview[indices[i]:indices[i+1]]

    def refresh_manifold_nn(self, manifold_embedding, manifold):
        manifold_nns = ManifoldNNS(manifold_embedding, manifold)
        nns = manifold_nns.knn_query_all(self.manifold_nn_k, self.num_workers)
        all_manifold_neighbors = [nns[i][0][1:].astype(np.int64) for i in range(self.N)]
        list_size, index_size = self.get_init_size_1d_memview_numpy_list(all_manifold_neighbors)
        self.manifold_neighbors = np.empty([list_size], dtype=np.int64)
        self.manifold_neighbors_indices = np.empty([index_size], dtype=np.int64)
        self.numpy_list_to_1d_memview(all_manifold_neighbors, self.manifold_neighbors, self.manifold_neighbors_indices)

    cpdef initialize_graph_perms(self):
        graph_neighbor_permutations = []
        for i in range(self.N):
            graph_neighbors = self.access_1d_memview_list(i, self.graph_neighbors, self.graph_neighbors_indices)
            total_neighbors = min(graph_neighbors.shape[0], self.n_graph_neighbors)
            graph_neighbor_permutations.append(np.random.permutation(graph_neighbors)[:total_neighbors])

        list_size, index_size = self.get_init_size_1d_memview_numpy_list(graph_neighbor_permutations)
        self.graph_neighbor_permutations = np.empty([list_size], dtype=np.int64)
        self.graph_neighbor_permutations_indices = np.empty([index_size], dtype=np.int64)
        self.numpy_list_to_1d_memview(graph_neighbor_permutations, self.graph_neighbor_permutations, self.graph_neighbor_permutations_indices)

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
        cdef long[:] neighbors
        cdef unordered_set[long] excluded_samples
        cdef unsigned long seed

        seed = i
        j = 0
        while j < self.batch_size and i + j < self.perm.shape[0]:
            ntries = 0

            vertex = self.perm[i + j]
            vertices[j, 0] = vertex
            neighbors = self.access_1d_memview_list(vertex, self.graph_neighbors, self.graph_neighbors_indices)
            k = 0

            extra_rand_samples = 0
            if neighbors.shape[0] < self.n_graph_neighbors:
                extra_rand_samples = self.n_graph_neighbors - neighbors.shape[0]
            total_graph_samples = self.n_graph_neighbors - extra_rand_samples
            excluded_samples = unordered_set[long]()
            k = 0
            while k < neighbors.shape[0]:
                excluded_samples.insert(neighbors[k])
                k = k + 1
            neighbors = self.access_1d_memview_list(vertex, self.graph_neighbor_permutations, self.graph_neighbor_permutations_indices)
            k = 1
            while k < 1+total_graph_samples:
                vertices[j, k] = neighbors[k - 1]
                graph_dists[j,k - 1] = 1
                k = k + 1
            current_index = k
            neighbors = self.access_1d_memview_list(vertex, self.manifold_neighbors, self.manifold_neighbors_indices)
            k = 0
            l = 0
            while k < self.n_manifold_neighbors and l < neighbors.shape[0]:
                new_vertex = neighbors[l]
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

