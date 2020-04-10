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

def load_csv_edge_list(path: str, symmetrize: bool=False, delimiter:str =","):
    """
    Loads a graph from a csv file with headers id1, id2, weight

    Params:
        path (str): Csv file of graph data
        symmetrize (bool): Should the data be symmetrized (i.e. transform
            a directed graph such that every edge has an opposite edge
            pointing the other direction
        delimiter: Delimiter for csv file

    Returns tuple of:
        edges (numpy.ndarray): numpy array of shape [num_edges, 2] containing
            the indices of nodes in all edges
        objects (List[str]): string object ids for all nodes as listed in the
            csv file
        weights (numpy.ndarray): numpy array of shape [num_edges] containing
            edge weights as listed in csv file
    """

    df = pandas.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c', sep=delimiter)

    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
        df = pandas.concat([df, rev])
    idx, objects = pandas.factorize(df[['id1', 'id2']].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    weights = df.weight.values.astype('float')
    return idx, objects.tolist(), weights
