# /env sugar-env
"""
@author: Sixuan Wang
"""

import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYCHARM_MATPLOTLIB_GUI'] = 'false'

import torch.optim as optim
from sklearn.metrics import adjusted_rand_score, mean_squared_error, mean_absolute_error
from scipy.spatial.distance import pdist
import math

from .preprocessing import *
from .models import *
from .utils import *
from .graph import *

import warnings
warnings.filterwarnings('ignore')


#######################################################################################################
# Infer: spatial positioning
def infer_spatial_coordinates_with_uncertainty(M_unbalanced, st_coords, k=6, 
                                               split_factor=30.0, 
                                               resolution='low_resolution'):

    if resolution == 'high_resolution':
        split_factor = 15.0  # 

    device = M_unbalanced.device
    st_coords_f32 = st_coords.to(torch.float32).to(device)

    raw_entropy = -torch.sum(M_unbalanced * torch.log(M_unbalanced + 1e-8), dim=1)
    norm_ent = raw_entropy / (math.log(M_unbalanced.size(1)) + 1e-8)
    power = 2.0 - norm_ent
    P = torch.pow(M_unbalanced.clamp(min=0), power.unsqueeze(-1))
    P = P / (P.sum(dim=1, keepdim=True) + 1e-8)

    topk_probs, topk_indices = torch.topk(P, k=k, dim=1)
    w = topk_probs / (topk_probs.sum(dim=1, keepdim=True) + 1e-8)
    coords_k = st_coords_f32[topk_indices]                    # [N, k, 2]

    with torch.no_grad():
        n_s = min(1000, st_coords_f32.size(0))
        idx = torch.randperm(st_coords_f32.size(0), device=device)[:n_s]
        d = torch.cdist(st_coords_f32[idx], st_coords_f32[idx])
        d = d + torch.eye(n_s, device=device) * 1e9
        global_spot_dist = d.min(dim=1).values.median()

    pairwise = torch.cdist(coords_k, coords_k)                # [N, k, k]
    medoid_idx = pairwise.sum(dim=2).argmin(dim=1)            # [N]
    arangeN = torch.arange(coords_k.size(0), device=device)
    anchor = coords_k[arangeN, medoid_idx].unsqueeze(1)       # [N, 1, 2]

    dist_to_anchor = torch.norm(coords_k - anchor, dim=-1)    # [N, k]
    keep = dist_to_anchor < split_factor * global_spot_dist
    keep[arangeN, medoid_idx] = True

    w_kept = w * keep.float()
    w_kept = w_kept / (w_kept.sum(dim=1, keepdim=True) + 1e-8)
    pred_coords = (w_kept.unsqueeze(-1) * coords_k).sum(dim=1)

    exp_sq = (w_kept * (coords_k ** 2).sum(dim=-1)).sum(dim=1)
    spatial_std = torch.sqrt(torch.clamp(exp_sq - (pred_coords ** 2).sum(dim=1), min=0.0) + 1e-8)
    confidence = w[:, 0
    return pred_coords, confidence, spatial_std, raw_entropy



#######################################################################################################
# Infer: spot deconvolution
def infer_deconvolute_st(M_unbalanced, sc_cell_types, num_classes=None, size_factors=None):
    device = M_unbalanced.device
    
    if sc_cell_types.dim() == 1:
        if num_classes is None:
            num_classes = sc_cell_types.max().item() + 1
        L_sc = F.one_hot(sc_cell_types, num_classes=num_classes).to(torch.float32).to(device)
    else:
        L_sc = sc_cell_types.to(torch.float32).to(device)
        num_classes = L_sc.shape[1]

    M_col_norm = M_unbalanced / (M_unbalanced.max(dim=0, keepdim=True)[0] + 1e-8)
    M_filtered = torch.where(M_col_norm > 0.2, M_unbalanced, torch.zeros_like(M_unbalanced))

    spot_celltype_scores = torch.matmul(M_filtered.t(), L_sc)
    if size_factors is not None:
        spot_celltype_scores = spot_celltype_scores / (size_factors + 1e-8)

    row_sums = spot_celltype_scores.sum(dim=1, keepdim=True)
    spot_proportions = spot_celltype_scores / (row_sums + 1e-8)

    spot_dominant_type = torch.argmax(spot_proportions, dim=1)
    spot_entropy = -torch.sum(spot_proportions * torch.log(spot_proportions + 1e-8), dim=1)
    
    return spot_proportions, spot_dominant_type, spot_entropy

