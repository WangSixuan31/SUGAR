# /env sugar-env
"""
@author: Sixuan Wang
"""

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
import muon as mu  
from .utils import *

import warnings
warnings.filterwarnings('ignore')

############################################
def center_spatial_coordinates(adata, key='spatial', new_key='spatial_centered'):

    if key in adata.obsm:
        spatial_coords = adata.obsm[key]
        centered_coords = spatial_coords - spatial_coords.mean(axis=0)
        adata.obsm[new_key] = centered_coords
    else:
        adata.obsm[new_key] = np.zeros((adata.shape[0], 2)) 

def normalized_spatial_coordinates(adata, key='spatial_centered', new_key='spatial_normalized', range=(0, 1)):

    if key in adata.obsm:
        centered_coords = adata.obsm[key]
        min_vals = centered_coords.min(axis=0)
        max_vals = centered_coords.max(axis=0)
        range_span = max_vals - min_vals
        range_span[range_span == 0] = 1

        normalized_coords = (centered_coords - min_vals) / range_span
        normalized_coords = normalized_coords * (range[1] - range[0]) + range[0]
        adata.obsm[new_key] = normalized_coords
    else:
        adata.obsm[new_key] = np.zeros((adata.shape[0], 2))   


############################################
def preprocess(sc_adata, st_adata,  
               sc_min_genes: int = 200, sc_min_cells: int = 5,
               st_min_genes: int = 200, st_min_cells: int = 5,
               normalize: bool = True, log1p: bool = True,     
               flavor: str = 'seurat_v3', 
               hvg_features: int = 3000, svg_features: int = 3000,
               select_gene: str = 'union',
               n_pcs: int = 30,
               seed: int = 2026):


    sc_adata.var_names_make_unique()
    st_adata.var_names_make_unique()

    sc_adata.raw = sc_adata 
    st_adata.raw = st_adata

    sc.pp.filter_cells(sc_adata, min_genes=sc_min_genes) 
    sc.pp.filter_genes(sc_adata, min_cells=sc_min_cells)
    sc.pp.filter_cells(st_adata, min_genes=st_min_genes)
    sc.pp.filter_genes(st_adata, min_cells=st_min_cells)
    
    if normalize:
        sc.pp.normalize_total(sc_adata, target_sum=1e4)
        sc.pp.normalize_total(st_adata, target_sum=1e4)
    if log1p:
        sc.pp.log1p(sc_adata)  
        sc.pp.log1p(st_adata)  

    n_top_genes_sc = min(hvg_features, sc_adata.shape[1])
    sc.pp.highly_variable_genes(sc_adata, flavor=flavor, n_top_genes=n_top_genes_sc) 

    n_top_genes_st = min(hvg_features, st_adata.shape[1])
    sc.pp.highly_variable_genes(st_adata, flavor=flavor, n_top_genes=n_top_genes_st)

    sc_hvg = sc_adata.var['highly_variable'][sc_adata.var['highly_variable'] == True].index
    st_hvg = st_adata.var['highly_variable'][st_adata.var['highly_variable'] == True].index

    sq.gr.spatial_neighbors(st_adata, coord_type="generic")
    sq.gr.spatial_autocorr(st_adata, mode="moran") 
    st_svg = st_adata.uns["moranI"]["I"].sort_values(ascending=False).head(svg_features).index
    
    if select_gene == 'intersection':
        inter_hvg = set(sc_hvg).intersection(set(st_hvg))
        inter_gene = set(inter_hvg).intersection(set(st_svg))
        
    elif select_gene == 'union':
        sc_gene = set(sc_adata.var_names)
        st_gene = set(st_adata.var_names)
        common_gene = set(sc_gene).intersection(set(st_gene))

        inter_svg = set(common_gene).intersection(set(st_svg))
        inter_hvg = set(sc_hvg).intersection(set(st_hvg))
        inter_gene = set(inter_hvg).union(set(inter_svg))

    final_gene_list = sorted(list(inter_gene))

    sc_adata = sc_adata[:, final_gene_list]
    st_adata = st_adata[:, final_gene_list]

    sc.pp.scale(sc_adata, max_value=10)
    sc.pp.scale(st_adata, max_value=10)

    sc.tl.pca(sc_adata, n_comps=n_pcs, use_highly_variable=True, svd_solver='arpack', random_state=seed)
    sc.tl.pca(st_adata, n_comps=n_pcs, use_highly_variable=True, svd_solver='arpack', random_state=seed)
   
    if 'spatial' in sc_adata.obsm:
        center_spatial_coordinates(sc_adata)
        normalized_spatial_coordinates(sc_adata)
    else:
        print("Spatial information not found in sc_adata.obsm!")
    center_spatial_coordinates(st_adata)
    normalized_spatial_coordinates(st_adata)

    print("sc_adata:", sc_adata)
    print("st_adata:", st_adata)
    print("Data have been pre-processed!")

    return sc_adata, st_adata



############################################
# def restore_spatial_coordinates(sc_adata,  st_adata,
#                                 spatial_normalized_pred, range=(0, 1),
#                                 centered_key='spatial_centered', original_key='spatial'):
#     centered_coords = st_adata.obsm[centered_key]
#     min_vals = centered_coords.min(axis=0)
#     max_vals = centered_coords.max(axis=0)

#     range_span = range[1] - range[0]
#     centered_restored = (spatial_normalized_pred - range[0]) * (max_vals - min_vals) / range_span + min_vals

#     spatial_mean = st_adata.obsm[original_key].mean(axis=0)
#     spatial_restored = centered_restored + spatial_mean
#     sc_adata.obsm['spatial_pred'] = spatial_restored
#     return sc_adata

