#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict as ddict
import pandas
import numpy as np
from numpy.random import choice
import torch as th
from torch import nn
from torch.utils.data import Dataset as DS
from sklearn.metrics import average_precision_score
from multiprocessing.pool import ThreadPool
from functools import partial
import h5py
from tqdm import tqdm
import random


def load_adjacency_matrix(path, format='hdf5', symmetrize=False):
    if format == 'hdf5':
        with h5py.File(path, 'r') as hf:
            return {
                'ids': hf['ids'].value.astype('int'),
                'neighbors': hf['neighbors'].value.astype('int'),
                'offsets': hf['offsets'].value.astype('int'),
                'weights': hf['weights'].value.astype('float'),
                'objects': hf['objects'].value
            }
    elif format == 'csv':
        df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')

        if symmetrize:
            rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
            df = pandas.concat([df, rev])

        idmap = {}
        idlist = []

        def convert(id):
            if id not in idmap:
                idmap[id] = len(idlist)
                idlist.append(id)
            return idmap[id]
        df.loc[:, 'id1'] = df['id1'].apply(convert)
        df.loc[:, 'id2'] = df['id2'].apply(convert)

        groups = df.groupby('id1').apply(lambda x: x.sort_values(by='id2'))
        counts = df.groupby('id1').id2.size()

        ids = groups.index.levels[0].values
        offsets = counts.loc[ids].values
        offsets[1:] = np.cumsum(offsets)[:-1]
        offsets[0] = 0
        neighbors = groups['id2'].values
        weights = groups['weight'].values
        return {
            'ids' : ids.astype('int'),
            'offsets' : offsets.astype('int'),
            'neighbors': neighbors.astype('int'),
            'weights': weights.astype('float'),
            'objects': np.array(idlist)
        }
    else:
        raise RuntimeError(f'Unsupported file format {format}')


def load_edge_list(path, symmetrize=False, delimiter=","):
    df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c', sep=delimiter)
    # df = pandas.read_csv(path, usecols=['id1', 'id2'], engine='c', sep=" ")
    df["weight"] = 1

    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
        df = pandas.concat([df, rev])
    idx, objects = pandas.factorize(df[['id1', 'id2']].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    weights = df.weight.values.astype('float')
    return idx, objects.tolist(), weights

def reconstruction_worker(adj, train_adj, lt, distfn, objects, progress=False, samples=0, uniform_neg_samples=0, neg_samples=10000):
    ranksum = nranks = ap_scores = iters = recranksum = hitsat10 = 0
    labels = np.empty(lt.size(0))
    if samples > 0:
        objects = np.random.choice(objects, samples)
    if neg_samples > 0:
        neg_p = np.zeros(lt.size(0))
        for node, neighbors in train_adj.items():
            for neighbor in neighbors:
                neg_p[neighbor] += 1
        neg_p /= neg_p.sum()

    for object in tqdm(objects) if progress else objects:
        labels.fill(0)
        neighbors = np.array(list(adj[object]))
        train_neighbors = np.array(list(train_adj[object]))
        
        neg_pos_samples = lt[neighbors]
        if uniform_neg_samples > 0:
            # Add uniformly sampled negative samples
            uniform_p = np.ones(lt.size(0))
            uniform_p[object] = 0
            uniform_p[neighbors] = 0
            uniform_p[train_neighbors] = 0
            uniform_p /= uniform_p.sum()
            uniform_samples = lt[np.random.choice(lt.size(0), size=uniform_neg_samples, p=uniform_p)]
            neg_pos_samples = th.cat([neg_pos_samples, uniform_samples])
        if neg_samples > 0:
            # Add negative samples sampled according to frequency in training data
            neg_p_no_neighbors = neg_p.copy()
            neg_p_no_neighbors[object] = 0
            neg_p_no_neighbors[neighbors] = 0
            neg_p_no_neighbors[train_neighbors] = 0
            neg_p_no_neighbors /= neg_p_no_neighbors.sum()
            train_neg_samples = lt[np.random.choice(lt.size(0), size=neg_samples, p=neg_p_no_neighbors)]
            neg_pos_samples = th.cat([neg_pos_samples, train_neg_samples])

        dists = distfn(lt[None, object], lt)
        sorted_dists, sorted_idx = dists.sort()
        ranks, = np.where(np.in1d(sorted_idx.detach().cpu().numpy(), np.arange(neighbors.shape[0])))
        # The above gives us the position of the neighbors in sorted order.  We
        # want to count the number of non-neighbors that occur before each neighbor
        ranks += 1
        N = ranks.shape[0]
        # Simpler way to cancel out other near neighbors
        ranks -= np.arange(N)
        rec_ranks = np.ones(N) / ranks

        recranksum += rec_ranks.sum()
        hitsat10 += (ranks <= 10).sum()
        ranksum += ranks.sum()
        nranks += ranks.shape[0]
        labels[neighbors] = 1
        # No need to compute ap for now
        # ap_scores += average_precision_score(labels, -dists.detach().cpu().numpy())
        iters += 1
    return float(ranksum), float(recranksum), float(hitsat10), nranks, ap_scores, iters


def eval_reconstruction(adj, train_adj, lt, distfn, workers=1, progress=False, samples=0, uniform_neg_samples=0, neg_samples=10000):
    '''
    Reconstruction evaluation.  For each object, rank its neighbors by distance
    Args:
        adj (dict[int, set[int]]): Adjacency list mapping objects to its neighbors
        lt (torch.Tensor[N, dim]): Embedding table with `N` embeddings and `dim`
            dimensionality
        distfn ((torch.Tensor, torch.Tensor) -> torch.Tensor): distance function.
        workers (int): number of workers to use
    '''
    objects = np.array(list(adj.keys()))
    if workers > 1:
        with ThreadPool(workers) as pool:
            f = partial(reconstruction_worker, adj, train_adj, lt, distfn, samples=samples//workers, uniform_neg_samples=uniform_neg_samples, neg_samples=neg_samples)
            results = pool.map(f, np.array_split(objects, workers))
            results = np.array(results).sum(axis=0).astype(float)
    else:
        results = reconstruction_worker(adj, train_adj, lt, distfn, objects, progress, samples, uniform_neg_samples, neg_samples)
    return float(results[0]) / results[3], float(results[1]) / results[3], float(results[2]) / results[3] 
