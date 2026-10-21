# /env sugar-env
"""
@author: Sixuan Wang
"""

import timeit
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from torch_geometric.nn import knn_graph

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

from .preprocessing import *
from .utils import *

import warnings
warnings.filterwarnings('ignore')


#######################################################################################################
# Evaluate: spatial positioning
def evaluate(sc_adata, compute_jsd_flag=False, label_col=None):    

    if 'is_mapped' in sc_adata.obs:
        valid_mask = sc_adata.obs['is_mapped'].values
        if valid_mask.sum() == 0:
            print("Warning: No cells were mapped confidently!")
            return
        sc_adata_eval = sc_adata[valid_mask].copy()
    else:
        sc_adata_eval = sc_adata
        
    pred_pos_sc = sc_adata_eval.obsm['spatial_pred']
    spatial_coords_sc = sc_adata_eval.obsm['spatial']
    
    pred_dist_sc = cal_dist(pred_pos_sc, normalize = True)
    true_dist_sc = cal_dist(spatial_coords_sc, normalize = True)
    dist_pcc = round(pearsonr(pred_dist_sc, true_dist_sc)[0], 3)
    coord_pcc = pearson_corr(pred_pos_sc, spatial_coords_sc)
    print(f"For PCC: {coord_pcc.item():.4f}, ", f"PCC of pair distance: {dist_pcc:.4f}")

    individual_distances = np.linalg.norm(pred_pos_sc - spatial_coords_sc, axis=1)  
    sc_adata_eval.obs['pred_error_dist'] = individual_distances
    mse = mean_squared_error(spatial_coords_sc, pred_pos_sc)
    mae = mean_absolute_error(spatial_coords_sc, pred_pos_sc)
    std_dist = np.std(individual_distances) # std
    print(f"MSE: {mse:.4f}, ", f"MAE: {mae:.4f}(±{std_dist:.4f} SD)")

    hit_rate_10, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 10)
    hit_rate_20, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 20)
    hit_rate_50, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 50)
    hit_rate_100, _ = hitnumber_evaluation(pred_pos_sc, spatial_coords_sc, k = 100)
    print(f"Hit_rate (10-20-50-100): {hit_rate_10:.0f}, {hit_rate_20:.0f}, {hit_rate_50:.0f}, {hit_rate_100:.0f}")
   
    # Optional：Label transfer
    if 'cluster' in sc_adata.obs.columns and 'cluster_pred' in sc_adata.obs.columns:
        ARI_sc = adjusted_rand_score(sc_adata_eval.obs['cluster'], sc_adata_eval.obs['cluster_pred'])
        print(f'Adjusted Rand Index (ARI_sc): {ARI_sc:.4f}')

    if compute_jsd_flag:
        jsd_result = compute_jsd(sc_adata_eval, label_col=label_col)
        avg_jsd, detail_df = jsd_result
        print(f"Average JSD: {avg_jsd:.4f}")
        print("--- Detailed JSD per Cell Type ---")
        print(detail_df)

    return sc_adata, sc_adata_eval



#######################################################################################################
# Evaluate: spot deconvolution
def evaluate_deconvolution(pred_proportions, true_proportions=None):
    if isinstance(pred_proportions, pd.DataFrame):
        pred_proportions = pred_proportions.values
    
    pred_proportions = pred_proportions / (pred_proportions.sum(axis=1, keepdims=True) + 1e-8)
    
    results = {}
    
    if true_proportions is not None:
        if isinstance(true_proportions, pd.DataFrame):
            true_proportions = true_proportions.values
        true_proportions = true_proportions / (true_proportions.sum(axis=1, keepdims=True) + 1e-8)
        
        mse = np.mean((pred_proportions - true_proportions) ** 2)
        results['RMSE'] = np.sqrt(mse)
        
        spot_corrs = []
        for i in range(pred_proportions.shape[0]):
            if np.std(pred_proportions[i]) > 1e-6 and np.std(true_proportions[i]) > 1e-6:
                r, _ = pearsonr(pred_proportions[i], true_proportions[i])
                if not np.isnan(r):
                    spot_corrs.append(r)
        results['Mean_Pearson_Spot'] = np.mean(spot_corrs) if spot_corrs else np.nan
        
        # celltype_corrs = []
        # for j in range(pred_proportions.shape[1]):
        #     if np.std(pred_proportions[:, j]) > 1e-6 and np.std(true_proportions[:, j]) > 1e-6:
        #         r, _ = pearsonr(pred_proportions[:, j], true_proportions[:, j])
        #         if not np.isnan(r):
        #             celltype_corrs.append(r)
        # results['Mean_Pearson_Celltype'] = np.mean(celltype_corrs) if celltype_corrs else np.nan
        
        jsd_list = []
        for i in range(pred_proportions.shape[0]):
            jsd = jensenshannon(pred_proportions[i], true_proportions[i])
            if not np.isnan(jsd):
                jsd_list.append(jsd)
        results['Mean_JSD'] = np.mean(jsd_list) if jsd_list else np.nan
        
    else:
        entropy = -np.sum(pred_proportions * np.log(pred_proportions + 1e-8), axis=1)
        results['Mean_Entropy'] = np.mean(entropy)
        results['Median_Entropy'] = np.median(entropy)
        
        active_celltypes_per_spot = np.sum(pred_proportions > 0.05, axis=1)
        results['Mean_Active_Celltypes'] = np.mean(active_celltypes_per_spot)

    return results

