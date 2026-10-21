# /env sugar-env
"""
@author: Sixuan Wang
"""

from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import scanpy as sc

import scipy
import scipy.sparse as sp
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist,pdist
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from esda.moran import Moran
from esda.geary import Geary
from skimage.metrics import structural_similarity as ssim
from libpysal.weights import KNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric import seed_everything

import time
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

import os
import warnings
warnings.filterwarnings('ignore')

####################################
def init_model(net):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        net.cuda()
    return net


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # 锁定 cuBLAS  workspace
    os.environ['TORCH_USE_DETERMINISTIC_ALGORITHMS'] = '1'
    os.environ['FORCE_DETERMINISTIC_CALLBACKS'] = '1'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True, warn_only=False)

    if hasattr(torch.cuda, 'matmul'):
        torch.cuda.matmul.allow_tf32 = False
    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


########################################################################
def pearson_corr(x, y):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)

    covariance = torch.sum(x_centered * y_centered)
    
    x_std = torch.sqrt(torch.sum(x_centered ** 2))
    y_std = torch.sqrt(torch.sum(y_centered ** 2))

    # epsilon = 1e-8
    correlation = covariance / (x_std * y_std)

    return correlation

def cal_dist(coord, normalize: bool = True):
    if not isinstance(coord, torch.Tensor):
        coord = torch.tensor(coord, dtype=torch.float32)
        
    n = len(coord)
    dist_list = []
    for i in tqdm(range(n - 1)):
        point_i = coord[i]
        points_j = coord[i+1:]
        dist_tmp = torch.norm(points_j - point_i, dim=1)
        dist_list.append(dist_tmp)
    if len(dist_list) > 0:
        dist = torch.cat(dist_list)
    else:
        dist = torch.tensor([])
    if normalize:
        max_dist = torch.max(dist)
        if max_dist > 0:
            dist = dist / max_dist   
    return dist


def cal_dist_group(sc_adata, group_key, select_group):
    coord_all = sc_adata.obsm['spatial_normalized_pred']

    group_df = sc_adata.obs[group_key]
    group_df = group_df.reset_index()
    group_df[group_key] = group_df[group_key].astype(str)
    group = list(np.unique(group_df[group_key]))
    assert select_group in group, 'Please select the correct group!'

    select_idx = group_df[group_df[group_key] == select_group].index
    selsct_coord = coord_all[select_idx]

    # dist_all = pd.DataFrame(columns=['dist', 'group'])
    dist_all = []
    for g in group:
        print('Calculating all cell pairs between', select_group, 'and', g, '...')
        dist_group = []
        tmp_idx = group_df[group_df[group_key] == g].index
        tmp_coord = coord_all[tmp_idx]

        if g == select_group:
            for i in range(len(tmp_coord)):
                xi, yi = tmp_coord[i, :]
                for j in range(len(selsct_coord)):
                    if i >= j:
                        continue
                    xj, yj = selsct_coord[j, :]
                    dist_tmp = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    dist_group.append(dist_tmp)
        else:
            for i in range(len(tmp_coord)):
                xi, yi = tmp_coord[i, :]
                for j in range(len(selsct_coord)):
                    xj, yj = selsct_coord[j, :]
                    dist_tmp = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    dist_group.append(dist_tmp)

        dist_all.append(dist_group)

    return dist_all

def hitnumber_evaluation(pred_coord, true_coord, k=10):
    if isinstance(pred_coord, torch.Tensor):
        pred_coord = pred_coord.detach().cpu().numpy()
    if isinstance(true_coord, torch.Tensor):
        true_coord = true_coord.detach().cpu().numpy()

    nn_pred = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(pred_coord)
    _, pred_knn_indices = nn_pred.kneighbors(pred_coord)
    pred_knn_indices = pred_knn_indices[:, 1:]  
    
    nn_true = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(true_coord)
    _, true_knn_indices = nn_true.kneighbors(true_coord)
    true_knn_indices = true_knn_indices[:, 1:]  

    hit_numbers_array = (pred_knn_indices[:, :, None] == true_knn_indices[:, None, :]).sum(axis=(1, 2))
    hit_rate = hit_numbers_array.mean()
    return hit_rate, hit_numbers_array.tolist()

######################################################################## 
def jaccard_similarity(pred_coord, true_coord, k):

    if isinstance(pred_coord, torch.Tensor):
        pred_coord = pred_coord.detach().cpu().numpy()
    if isinstance(true_coord, torch.Tensor):
        true_coord = true_coord.detach().cpu().numpy()

    dist_pred = cdist(pred_coord, pred_coord)
    dist_true = cdist(true_coord, true_coord)
    pred_knn_indices = np.argsort(dist_pred, axis=1)[:, 1:k + 1]
    true_knn_indices = np.argsort(dist_true, axis=1)[:, 1:k + 1]
    jaccard_scores = []

    for i in range(len(pred_coord)):
        pred_neighbors = set(pred_knn_indices[i])
        true_neighbors = set(true_knn_indices[i])
        intersection = pred_neighbors.intersection(true_neighbors)
        union = pred_neighbors.union(true_neighbors)
        jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0.0
        jaccard_scores.append(jaccard_index)

    average_jaccard = np.mean(jaccard_scores)
    return average_jaccard


def calculate_ssim(coords_true,
                   coords_pred,
                   expr,
                   grid_size=16,
                   sigma=2,
                   win_size=11,
                   interpolation="linear",
                   use_gaussian_smoothing=True,
                   align_coordinates=True):

    if align_coordinates:
        coords_true = (coords_true - coords_true.min(axis=0)) / \
                      (coords_true.max(axis=0) - coords_true.min(axis=0) + 1e-8)
        coords_pred = (coords_pred - coords_pred.min(axis=0)) / \
                      (coords_pred.max(axis=0) - coords_pred.min(axis=0) + 1e-8)

    x_min = min(coords_true[:, 0].min(), coords_pred[:, 0].min())
    x_max = max(coords_true[:, 0].max(), coords_pred[:, 0].max())
    y_min = min(coords_true[:, 1].min(), coords_pred[:, 1].min())
    y_max = max(coords_true[:, 1].max(), coords_pred[:, 1].max())

    grid_x, grid_y = np.mgrid[
        x_min:x_max:complex(grid_size),
        y_min:y_max:complex(grid_size)
    ]

    img_true = griddata(
        coords_true,
        expr,
        (grid_x, grid_y),
        method=interpolation,
        fill_value=np.nan
    )

    img_pred = griddata(
        coords_pred,
        expr,
        (grid_x, grid_y),
        method=interpolation,
        fill_value=np.nan
    )

    nan_mask = np.isnan(img_true) | np.isnan(img_pred)
    img_true[nan_mask] = 0
    img_pred[nan_mask] = 0

    if use_gaussian_smoothing:
        img_true = gaussian_filter(img_true, sigma=sigma)
        img_pred = gaussian_filter(img_pred, sigma=sigma)

    def normalize(img):
        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)
        img = np.clip(img, p1, p99)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    img_true = normalize(img_true)
    img_pred = normalize(img_pred)

    score = ssim(
        img_true,
        img_pred,
        data_range=1.0,
        win_size=win_size,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )

    return score


####################################
def nearest_neighbor_labels(z_sc, z_st, k=5):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(z_sc.detach().cpu().numpy())
    distances, indices = nn.kneighbors(z_st.detach().cpu().numpy())

    pseudo_labels = torch.zeros(z_st.size(0), z_sc.size(0), device=z_sc.device)
    for i in range(z_st.size(0)):
        pseudo_labels[i, indices[i]] = 1  # Mark the first k nearest neighbors as positive samples
    return pseudo_labels


############################################
def compute_jsd(adata, label_col='celltype', k=20):
    coords_true = adata.obsm['spatial']
    coords_pred = adata.obsm['spatial_pred']
    cell_types = adata.obs[label_col].values
    
    unique_types = np.unique(cell_types)
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    n_types = len(unique_types)
    
    # 2. Find Neighbors
    nn_true = NearestNeighbors(n_neighbors=k).fit(coords_true)
    nn_pred = NearestNeighbors(n_neighbors=k).fit(coords_pred)
    indices_true = nn_true.kneighbors(coords_true, return_distance=False)
    indices_pred = nn_pred.kneighbors(coords_pred, return_distance=False)

    all_jsd = []
    for i in range(len(adata)):
        # True
        t_dist = np.zeros(n_types)
        for t in cell_types[indices_true[i]]:
            t_dist[type_to_idx[t]] += 1
        t_dist /= k
        
        # Predicted
        p_dist = np.zeros(n_types)
        for t in cell_types[indices_pred[i]]:
            p_dist[type_to_idx[t]] += 1
        p_dist /= k
        
        # JS Divergence
        js_val = jensenshannon(t_dist, p_dist, base=2) ** 2
        all_jsd.append(js_val)
    
    all_jsd = np.array(all_jsd)

    overall_avg = np.mean(all_jsd)

    type_summary = {}
    for ct in unique_types:
        mask = (cell_types == ct)
        type_summary[ct] = all_jsd[mask].mean()
    
    df_type = pd.DataFrame.from_dict(type_summary, orient='index', columns=['Avg_JSD'])
    
    return overall_avg, df_type.sort_values(by='Avg_JSD')



##################################################################
def draw_spot_pies_no_bg(coords, prop_df, ax, radius=0.4, thresh=0.01, color_dict=None):
    cell_types = prop_df.columns

    if color_dict is None:
        colors = plt.cm.tab20(np.linspace(0, 1, len(cell_types)))
        color_dict = {ct: c for ct, c in zip(cell_types, colors)}

    for i in range(len(coords)):
        x, y = coords[i, 0], coords[i, 1]
        props = prop_df.iloc[i, :].values

        mask = props > thresh
        if not np.any(mask):
            continue
            
        filtered_props = props[mask]
        filtered_colors = [color_dict[ct] for ct in cell_types[mask]]

        ax.pie(
            filtered_props, 
            center=(x, y), 
            colors=filtered_colors, 
            radius=radius,
            wedgeprops={'linewidth': 0} 
        )

    ax.set_xlim(coords[:, 0].min() - 1, coords[:, 0].max() + 1)
    ax.set_ylim(coords[:, 1].min() - 1, coords[:, 1].max() + 1)
    ax.set_aspect('equal')
    ax.axis('off') 

####################################
def benchmark_scalability(algorithm_func, adata, sample_sizes=[2000, 5000, 9146]):
    results = []
    
    for n in sample_sizes:
        print(f"Testing scale: {n} cells...")

        if n < adata.shape[0]:
            idx = np.random.choice(adata.shape[0], n, replace=False)
            sub_adata = adata[idx].copy()
        else:
            sub_adata = adata.copy()
            
        # 2. Record
        start_time = time.time()
        mem_peak = memory_usage((algorithm_func, (sub_adata,)), interval=0.1, max_usage=True)
        end_time = time.time()
        
        results.append({
            'Cells': n,
            'Runtime_Sec': end_time - start_time,
            'Peak_Memory_MB': mem_peak
        })
        
    return pd.DataFrame(results)


#######################################################################################################
def focal_loss(logits, labels, gamma=2, alpha=None):
    ce_loss = F.cross_entropy(logits, labels, reduction='mean',
                              label_smoothing=0.1, weight=alpha)
    pt = torch.exp(-ce_loss)
    return ((1 - pt) ** gamma * ce_loss).mean()
    
def fuse_fixed_alpha(probs_cls, probs_m, alpha=0.2):
    probs_joint = alpha * probs_cls + (1 - alpha) * probs_m
    max_p, idx = torch.max(probs_joint, dim=1)
    return idx.cpu().numpy(), max_p.cpu().numpy()

def idx_to_labels(final_idx, categories):
    label_map = {i: c for i, c in enumerate(categories)}
    return [label_map[i] for i in final_idx]

