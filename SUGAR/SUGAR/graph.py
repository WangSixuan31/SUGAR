# /env sugar-env
"""
@author: Sixuan Wang
"""

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad

import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import pairwise, pairwise_distances
from sklearn.neighbors import kneighbors_graph, NearestNeighbors, radius_neighbors_graph

from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops

from .utils import *

import warnings
warnings.filterwarnings('ignore')

import gc

############################################
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


############################################
def construct_graph(expr, k=10, metric="euclidean"):
    n = expr.shape[0]

    if sp.issparse(expr):
        expr = expr.toarray()
    else:
        expr = np.asarray(expr, np.float32)

    nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric, algorithm='brute', n_jobs=1).fit(expr)
    idx = nbrs.kneighbors(expr, return_distance=False)

    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = idx[:, 1:].ravel().astype(np.int32)

    del nbrs
    gc.collect()

    data = np.ones(n * k, dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    adj.eliminate_zeros()
    
    return adj



############################################
def get_graph_data_loader(adata, edge_index, spatial_edge_index=None, batch_size=256, worker_init_fn=None, generator=None):
    if sp.issparse(adata.X):
        dataa = adata.X.toarray()
    else:
        dataa = np.array(adata.X)
    
    x = torch.tensor(dataa, dtype=torch.float32)
    pos = torch.tensor(adata.obsm.get('spatial_normalized', np.zeros((adata.shape[0], 2))), dtype=torch.float32)

    if 'label' in adata.obs:
        label = torch.tensor(adata.obs['label'].values, dtype=torch.long)
    else:
        label = None

    data = Data(x=x, edge_index=edge_index, pos=pos, y=label)
    if spatial_edge_index is not None:
        data.spatial_edge_index = spatial_edge_index 

    return DataLoader([data], batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, generator=generator)




def construct_graph_data(sc_adata, st_adata, batch_size=256, seed=2026):
    g = torch.Generator()
    g.manual_seed(seed)
    
    # scRNA
    sc_adj_gene = construct_graph(sc_adata.obsm['X_pca'], k=6) 
    sc_edge_index, _ = from_scipy_sparse_matrix(sc_adj_gene)

    # ST
    st_adj_gene = construct_graph(st_adata.obsm['X_pca'], k=6) 
    st_edge_index, _ = from_scipy_sparse_matrix(st_adj_gene)

    # DataLoader
    sc_loader = get_graph_data_loader(sc_adata, sc_edge_index, batch_size=batch_size, worker_init_fn=seed_worker,
        generator=g)
    st_loader = get_graph_data_loader(st_adata, st_edge_index, batch_size=batch_size, worker_init_fn=seed_worker,
        generator=g)
    return sc_loader, st_loader



