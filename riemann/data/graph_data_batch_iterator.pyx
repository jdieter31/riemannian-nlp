# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import queue
import threading
from math import ceil

import numpy as np
import torch
from libc.math cimport

pow
from libc.stdlib cimport

RAND_MAX
from libcpp cimport

bool
from libcpp.unordered_map cimport

unordered_map
from libcpp.unordered_set cimport

unordered_set

from .graph_data_batch import GraphDataBatch

# Thread safe random number generation.  libcpp doesn't expose rand_r...
cdef unsigned long rand_r(unsigned long* seed) nogil:
    seed[0] = (seed[0] * 1103515245) % <unsigned long>pow(2, 32) + 12345
    return seed[0] % RAND_MAX

cdef class GraphDataBatchIterator:
    """
    Efficient graph neighbor sampler. Works as configured by
    GraphSamplingConfig. Ensure to call refresh_manifold_nn()
    before iterating if n_manifold_neighbors is not zero to avoid
    iterating over uninitialized data.
    """

    cdef public int n_graph_neighbors, n_manifold_neighbors, n_rand_neighbors, N, N_non_trivial, N_non_trivial_train, batch_size, current, num_workers
    cdef public float data_fraction
    cdef public bool is_eval
    cdef object queue
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
    cdef unordered_set[long] empty_vertices
    cdef long[:] non_empty_indices_train
    cdef long[:,:] graph_neighbor_permutations
    cdef unordered_map[int, int] graph_neighbor_difs
    cdef list threads

    def __cinit__(self,
                  neighbor_data,
                  graph_sampling_config,
                  train_neighbor_data=None,
                 ):
        """
        Params:
            neighbor_data (Dict): neighbor data dictionary as produced from a
                GraphDataset object's gen_graph_data()
            graph_sampling_config (GraphSamplingConfig): Configuration for
                algorithm
            train_neighbor_data (Dict): neighbor data for the train portion of
                the dataset (for ensuring hidden positives are not used as
                negatives during evaluation)
        """

        self.N = neighbor_data["N"]
        self.batch_size = min(graph_sampling_config.batch_size, self.N)
        self.queue = queue.Queue(maxsize=graph_sampling_config.num_workers)
        self.num_workers = graph_sampling_config.num_workers
        self.data_fraction = 1
               
        self.non_empty_indices = neighbor_data["non_empty_vertices"]
        self.N_non_trivial = len(self.non_empty_indices)
        
        all_graph_neighbors = neighbor_data["all_graph_neighbors"]
        all_graph_weights = neighbor_data["all_graph_weights"]
        self.graph_neighbors = np.concatenate(all_graph_neighbors)
        self.graph_neighbor_weights = np.concatenate(all_graph_weights)
        self.graph_neighbors_indices = \
            np.empty([len(all_graph_neighbors) + 1], dtype=np.int64)
        self.get_1d_index_list(all_graph_neighbors,
                               self.graph_neighbors_indices)
        

        self.empty_vertices = unordered_set[long]()
        if train_neighbor_data is not None:
            self.is_eval = True 
            self.n_graph_neighbors = min(self.N - 1,
                                        graph_sampling_config.n_graph_neighbors)

            max_train_neighbors = max([graph_neighbors.shape[0] for graph_neighbors in train_neighbor_data["all_graph_neighbors"]])
                  
            max_neighbors = max([0, max([graph_neighbors.shape[0] for  graph_neighbors in all_graph_neighbors]) - self.n_graph_neighbors])
            self.n_manifold_neighbors = min(self.N - 1 - self.n_graph_neighbors - max_neighbors - max_train_neighbors,
                                            graph_sampling_config.n_manifold_neighbors)

            self.n_rand_neighbors = min(self.N - 1 - self.n_graph_neighbors -
                                        self.n_manifold_neighbors - max_train_neighbors - max_neighbors,
                                        graph_sampling_config.n_rand_neighbors)

            self.non_empty_indices_train = \
                train_neighbor_data["non_empty_vertices"]
            self.N_non_trivial_train = len(self.non_empty_indices_train)

            all_graph_neighbors = train_neighbor_data["all_graph_neighbors"]
            all_graph_weights = train_neighbor_data["all_graph_weights"]
            self.train_graph_neighbors = np.concatenate(all_graph_neighbors)
            self.train_graph_neighbor_weights = \
                                    np.concatenate(all_graph_weights)
            self.train_graph_neighbors_indices = \
                    np.empty([len(self.non_empty_indices_train) + 1],
                             dtype=np.int64)
            self.get_1d_index_list(all_graph_neighbors,
                                   self.train_graph_neighbors_indices)
        else:
            self.is_eval = False


            self.n_graph_neighbors = min(self.N_non_trivial - 1,
                                        graph_sampling_config.n_graph_neighbors)

            max_neighbors = max([0, max([graph_neighbors.shape[0] for  graph_neighbors in all_graph_neighbors]) - self.n_graph_neighbors])
            self.n_manifold_neighbors = min(self.N_non_trivial - 1 - self.n_graph_neighbors - max_neighbors,
                                            graph_sampling_config.n_manifold_neighbors)
            self.n_rand_neighbors = min(self.N_non_trivial - 1 - self.n_graph_neighbors -
                                        self.n_manifold_neighbors - max_neighbors,
                                        graph_sampling_config.n_rand_neighbors)
            k = 0
            while k < neighbor_data["empty_vertices"].shape[0]:
                self.empty_vertices.insert(neighbor_data["empty_vertices"][k])
                k = k + 1
    
    def get_init_size_1d_memview_numpy_list(self, numpy_list):
        list_size = sum(array.shape[0] for array in numpy_list)
        index_size = len(numpy_list) + 1
        return list_size, index_size

    cpdef get_1d_index_list(self, numpy_list, long[:] indices):
        cdef int current = 0
        cdef int i = 0
        while i < len(numpy_list):
            indices[i] = current
            current = current + numpy_list[i].shape[0]
            i = i + 1
        indices[i] = current

    def refresh_manifold_nn(self, all_manifold_neighbors):
        """
        Refreshes the manifold near neighbor samples. If n_manifold_neighbors
        is not zero this must be ran before __iter__ to ensure uninitialized
        data is not sampled.
        
        Params:
            manifold_neighbors (list of numpy long): array of shape [nodes,
                num_neighbors] where manifold_neighbors[i] is a sorted list of
                the closest near neighbors to the vertex with index i
        """

        """
        if manifold_nns is None:
            manifold_nns = ManifoldNNS(manifold_embedding, manifold)
        print("\nQuerying near neighbors...")
        _, nns = manifold_nns.knn_query_all(self.manifold_nn_k, self.nn_workers)
        print("Processing near neighbor data...")
        all_manifold_neighbors = [nns[i][1:].astype(np.int64) for i in range(self.N)]
        """

        self.manifold_neighbors = np.concatenate(all_manifold_neighbors)
        self.manifold_neighbors_indices = \
            np.empty([len(all_manifold_neighbors) + 1], dtype=np.int64)
        self.get_1d_index_list(all_manifold_neighbors,
                               self.manifold_neighbors_indices)
        print("Finished processing near neighbor data.")

    cpdef initialize_graph_perms(self):
        graph_neighbor_permutations = []
        difs = []
        self.graph_neighbor_difs.clear()
        cdef int i = 0
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
        n_non_trivial = self.N_non_trivial
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
        
        n_non_trivial = self.N_non_trivial
        while self.current < ceil(n_non_trivial * self.data_fraction):
            current = self.current
            self.current += self.batch_size
            ix = torch.LongTensor(self.batch_size, 1 + self.n_graph_neighbors + self.n_manifold_neighbors + self.n_rand_neighbors)
            graph_dists = torch.zeros((self.batch_size, self.n_graph_neighbors + self.n_manifold_neighbors + self.n_rand_neighbors))
            graph_dists += 2
            memview = ix.numpy()
            graph_memview = graph_dists.numpy()
            count = self._getbatch(current, memview, graph_memview)
            if count < self.batch_size:
                ix = ix.narrow(0, 0, count)
                graph_dists = graph_dists.narrow(0, 0, count)
            neighbors = ix.narrow(1, 1, ix.size(1) - 1)
            vertices = ix.narrow(1, 0, 1).squeeze(1)
            batch = GraphDataBatch(vertices, neighbors, graph_dists)
            
            self.queue.put(batch)

        self.queue.put(i)

    def iter(self):
        return self.__iter__()

    def __len__(self):
        n_non_trivial = self.N_non_trivial
        return int(np.ceil(float(
            ceil(self.data_fraction * n_non_trivial)) / self.batch_size))

    def __next__(self):
        return self.next()

    def next(self):
        '''
        Python visible function for indexing the dataset.  This first
        allocates a tensor, and then modifies it in place with `_getitem`
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

    cdef public long _getbatch(self, int i, long[:,:] vertices, float[:,:] graph_dists):
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
            vertex = self.non_empty_indices[self.perm[i + j]]
            vertices[j, 0] = vertex
            neighbors_index = self.graph_neighbors_indices[self.perm[i + j]]
            neighbors_length = self.graph_neighbors_indices[self.perm[i + j] + 1] - neighbors_index

            extra_rand_samples = 0
            if neighbors_length < self.n_graph_neighbors:
                extra_rand_samples = self.n_graph_neighbors - neighbors_length
            total_graph_samples = self.n_graph_neighbors - extra_rand_samples
            excluded_samples = unordered_set[long]()
            excluded_samples.insert(vertex)

            k = 0
            while k < neighbors_length:
                excluded_samples.insert(self.graph_neighbors[neighbors_index + k])
                k = k + 1
            if self.is_eval:
                train_neighbors_index = self.train_graph_neighbors_indices[self.perm[i + j]]
                train_neighbors_length = self.train_graph_neighbors_indices[self.perm[i + j] + 1] - train_neighbors_index
                k = 0

                while k < train_neighbors_length:
                    excluded_samples.insert(self.train_graph_neighbors[train_neighbors_index + k])
                    k = k + 1
            k = 1
            size_dif = neighbors_length - self.n_graph_neighbors
            while k < 1+total_graph_samples:
                if size_dif > 0:
                    permutation_index = self.graph_neighbor_difs[size_dif]
                    vertices[j, k] = self.graph_neighbors[neighbors_index + self.graph_neighbor_permutations[permutation_index][k - 1]]
                    graph_dists[j,k - 1] = 1 + 1 / (self.graph_neighbor_weights[neighbors_index + self.graph_neighbor_permutations[permutation_index][k - 1]] + 2)
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
                    if excluded_samples.find(new_vertex) == excluded_samples.end() and \
                       self.empty_vertices.find(new_vertex) == self.empty_vertices.end():
                        vertices[j, k + current_index] = new_vertex
                        k = k + 1
                        excluded_samples.insert(new_vertex)
                    l = l + 1
                current_index = current_index + k
                if k < self.n_manifold_neighbors:
                    extra_rand_samples = extra_rand_samples + self.n_manifold_neighbors - k
            k = 0
            while k < self.n_rand_neighbors + extra_rand_samples:
                if self.is_eval:
                    new_vertex = <long>(<double>rand_r(&seed) / <double>RAND_MAX * self.N)
                else:
                    new_vertex = self.non_empty_indices[<long>(<double>rand_r(&seed) / <double>RAND_MAX * self.N_non_trivial)]


                if excluded_samples.find(new_vertex) == excluded_samples.end():
                    vertices[j, k + current_index] = new_vertex
                    k = k + 1
                    excluded_samples.insert(new_vertex)
                
            j = j + 1
        return j

