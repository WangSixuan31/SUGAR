# /env sugar-env
"""
@author: Sixuan Wang
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.nn import knn_graph

import timeit
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

 from scipy.stats import median_abs_deviation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, mean_squared_error, mean_absolute_error
from itertools import cycle
import copy
from tqdm import tqdm

from .preprocessing import *
from .models import *
from .utils import *
from .graph import *
from .transfer import *

import warnings
warnings.filterwarnings('ignore')

import gc


#######################################################################################################
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0 
     
        else:
            self.counter += 1 
            if self.counter >= self.patience:
                self.early_stop = True



#######################################################################################################
class SpaceReconstructor:
    def __init__(self, 
                 EncoderClass=Encoder, 
                 DecoderClass=Decoder, 
                 GAEClass=GraphAutoEncoder, 
                 MappingClass=UnbalancedMappingModel,
                 device: str = 'cuda:1', 
                 seed: int = 2026):
        self.Encoder = EncoderClass
        self.Decoder = DecoderClass
        self.GAE = GAEClass
        self.Mapping = MappingClass
        self.device = device
        self.seed = seed
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

        self.sc_embed_model = None
        self.st_embed_model = None
        self.mapping_model = None

    def _get_merged_config(self, n_feature, n_labels, custom_config=None):
        setup_seed(self.seed) 
        config = {'encoder': {'in_dims': n_feature, 
                              'hidden_dims': [256, 64],
                              'dropout': [0.1, 0],
                              'latent_dim': 64},
                  'decoder': {'latent_dim': 64, 
                              'hidden_dims': [256, 64], 
                              'output_dims': n_feature,
                              'dropout': [0.1, 0]},          
                  'mapping': {'input_dim': 64, 
                              'hidden_dims': [256, 128],
                              'dropout': [0.1, 0],
                              'eps':0.1, 'tau_sc':0.5, 'tau_st':0.5,
                              'prior_lambda': 10}}

        if custom_config:
            for module, params in custom_config.items():
                if module in config:
                    config[module].update(params)
        return config

    def fit(self, sc_adata, st_adata, 
            lr: float = 0.001,
            w_cl: float = 0.2, 
            w_cycle: float = 0.1,
            filter_mapped=False, 
            filter_strategy="mad",
            resolution='low_resolution',
            class_alpha: float = 0.2,
            epoch_num: int = 1000, 
            batch_size: int = 512, 
            log_epoch: int = 500,
            custom_config: dict = None,
            type_to_region: dict = None,     # prior
            sc_type_col: str = None,         # scRNA-seq
            st_region_col: str = None):      # ST

        setup_seed(self.seed) 
        n_feature = sc_adata.shape[1]
        n_labels = len(st_adata.obs['label'].unique()) if 'label' in st_adata.obs else 0

        self.region_penalty = None 
        if type_to_region is not None:
            sc_types = sc_adata.obs[sc_type_col].astype(str).values
            st_regions = st_adata.obs[st_region_col].astype(str).values
            N_sc, N_st = len(sc_types), len(st_regions)
            self.region_penalty = torch.zeros(N_sc, N_st, device=self.device)
            st_arr = np.asarray(st_regions)
            for i, t in enumerate(sc_types):
                allowed = type_to_region.get(t)
                if allowed is not None:
                    bad = ~np.isin(st_arr, allowed)
                    self.region_penalty[i, torch.tensor(bad, device=self.device)] = 1.0

        model_params = self._get_merged_config(n_feature, n_labels, custom_config)
        latent_dim = model_params['encoder']['latent_dim']

        self.embed_model = self.GAE(encoder=self.Encoder(**model_params['encoder']), 
                                    decoder=self.Decoder(**model_params['decoder']),
                                    latent_dim=latent_dim, n_labels=n_labels).to(self.device)

        self.mapping_model = self.Mapping(**model_params['mapping']).to(self.device)

        sc_loader, st_loader = construct_graph_data(sc_adata, st_adata, batch_size=batch_size)

        scaler = GradScaler()
        total_start_time = timeit.default_timer()

        optimizer_gae = optim.AdamW(self.embed_model.parameters(), lr=lr, weight_decay=1e-4, fused=True)
        optimizer_mapping = optim.AdamW(self.mapping_model.parameters(), lr=lr, weight_decay=1e-5, fused=True)

        early_stopping = EarlyStopping(patience=1000, min_delta=1e-3) #(patience=100, min_delta=1e-4)
        best_state_dicts = None

        pbar = tqdm(range(epoch_num))

        for epoch in pbar:
            self.embed_model.train()
            self.mapping_model.train()
            epoch_loss = 0.0
            sc_iter = iter(sc_loader)
            st_iter = iter(st_loader)
            
            if len(st_loader) >= len(sc_loader):
                main_loader = st_loader
                is_st_main = True
            else:
                main_loader = sc_loader
                is_st_main = False

            for batch_idx, main_data in enumerate(main_loader):
                if is_st_main:
                    st_data = main_data
                    try:
                        sc_data = next(sc_iter)
                    except StopIteration:
                        sc_iter = iter(sc_loader) 
                        sc_data = next(sc_iter)
                else:
                    sc_data = main_data
                    try:
                        st_data = next(st_iter)
                    except StopIteration:
                        st_iter = iter(st_loader) 
                        st_data = next(st_iter)

                x_sc, edge_sc = sc_data.x.to(self.device), sc_data.edge_index.to(self.device)
                x_st, edge_st = st_data.x.to(self.device), st_data.edge_index.to(self.device)
                pos_st = st_data.pos.to(self.device)
                y_st = st_data.y.long().to(self.device) 

                optimizer_gae.zero_grad(set_to_none=True) 
                optimizer_mapping.zero_grad(set_to_none=True)
                
                with autocast():       
                    z_sc, x_recon_sc, _ = self.embed_model(x_sc, edge_sc) 
                    z_st, x_recon_st, logits_st = self.embed_model(x_st, edge_st)

                    z_sc_norm = F.normalize(z_sc, p=2, dim=1)
                    z_st_norm = F.normalize(z_st, p=2, dim=1)

                    with autocast(enabled=False):
                        M = self.mapping_model(z_sc_norm.float(), z_st_norm.float(),
                                              region_penalty=self.region_penalty)   
                    
                    loss_recon = F.mse_loss(x_recon_sc, x_sc) + F.mse_loss(x_recon_st, x_st) 
                    loss_cls_st = focal_loss(logits_st, y_st.view(-1))

   
                    sc_confidence = M.sum(dim=1, keepdim=True)
                    st_confidence = M.sum(dim=0, keepdim=True) 
                    M_norm_st = M / (st_confidence + 1e-8) 
                    M_norm_sc = M / (sc_confidence + 1e-8)

                    pseudo_z_st = torch.matmul(M_norm_st.t(), z_sc_norm)   
                    loss_mapping_z = F.mse_loss(z_st_norm, pseudo_z_st)

                    pseudo_x_st = torch.matmul(M_norm_st.t(), x_sc)  
                    loss_st_per_spot = torch.mean((pseudo_x_st - x_st) ** 2, dim=1)
                    loss_mapping_x = torch.sum(loss_st_per_spot * st_confidence.squeeze().detach()) / (torch.sum(st_confidence.detach()) + 1e-8)
                    
                    cycle_x_sc = torch.matmul(M_norm_sc, pseudo_x_st)
                    cosine_sim = F.cosine_similarity(cycle_x_sc, x_sc, dim=-1)
                    loss_cycle = (1.0 - cosine_sim).mean() 
                    del pseudo_x_st

                    with torch.no_grad():
                        n_samples = min(pos_st.size(0), 1000)
                        sample_idx = torch.randperm(pos_st.size(0), device=self.device,
                                                    generator=self.generator)[:n_samples]
                        sample_dist = torch.cdist(pos_st[sample_idx], pos_st[sample_idx])
                        bandwidth = sample_dist.median() * 0.5 + 1e-8
                        del sample_dist
                    cell_confidence = sc_confidence.squeeze().detach()
                    mean_conf = cell_confidence.mean()
                    confidence_mask = cell_confidence > mean_conf
                    if confidence_mask.sum() > 0:
                        z_sc_conf = z_sc_norm[confidence_mask]
                        m_row_conf = M_norm_sc[confidence_mask].detach()
                        N_conf = z_sc_conf.size(0)
                        N_st = pos_st.size(0)
                        chunk_size = 512
                        cl = 0.0
                        for i in range(0, N_conf, chunk_size):
                            end_i = min(i + chunk_size, N_conf)
                            z_chunk = z_sc_conf[i:end_i]
                            m_chunk = m_row_conf[i:end_i].float()
                            logits_chunk = torch.matmul(z_chunk, z_st_norm.t())
                            log_probs_chunk = F.log_softmax(logits_chunk, dim=1)
                            target_chunk = torch.zeros((end_i - i, N_st),
                                                       device=self.device, dtype=torch.float32)
                            with torch.no_grad():
                                for j in range(0, N_st, chunk_size):
                                    end_j = min(j + chunk_size, N_st)
                                    dist_j = torch.cdist(pos_st, pos_st[j:end_j])   # [n_st, chunk]
                                    sp_aff_j = torch.exp(-(dist_j / bandwidth) ** 2)
                                    sp_aff_j[sp_aff_j < 1e-2] = 0.0
                                    target_chunk[:, j:end_j] = torch.matmul(m_chunk, sp_aff_j)
                                    del dist_j, sp_aff_j
                                target_probs_chunk = target_chunk / (target_chunk.sum(dim=1, keepdim=True) + 1e-8)
                                del target_chunk
                            cl = cl - (target_probs_chunk * log_probs_chunk.float()).sum()
                            del logits_chunk, log_probs_chunk, target_probs_chunk
                        cl = cl / N_conf
                    else:
                        cl = torch.tensor(0.0, device=self.device)
         
                    all_src, all_dst = edge_sc[0], edge_sc[1]
                    with torch.no_grad():
                        w_all = F.cosine_similarity(z_sc_norm[all_src], z_sc_norm[all_dst]).clamp(min=0)
                        if all_src.size(0) > 2048:
                            idx = torch.multinomial(w_all + 1e-8, 2048, replacement=False, generator=self.generator)
                            src_nodes, dst_nodes = all_src[idx], all_dst[idx]  
                        else:
                            src_nodes, dst_nodes = all_src, all_dst
                        entropy = torch.special.entr(M_norm_sc.clamp(min=1e-8)).sum(dim=1)
                        lam = (entropy / (math.log(M_norm_sc.size(1)) + 1e-8)).detach()
                        unc = 1.0 - 0.5 * (lam[src_nodes] + lam[dst_nodes])
                    prob_src = M_norm_sc[src_nodes]
                    prob_dst = M_norm_sc[dst_nodes]
                    edge_loss = F.smooth_l1_loss(prob_src, prob_dst, reduction='none').sum(dim=1)
                    loss_ssp = torch.mean(unc * edge_loss)
                        

                    loss_mapping = loss_mapping_z + loss_mapping_x + loss_ssp + w_cl * cl + w_cycle * loss_cycle 
                    total_loss = loss_recon + loss_cls_st + loss_mapping 
                    
                    del cell_confidence, M_norm_sc, M_norm_st

                scaler.scale(total_loss).backward()
                scaler.step(optimizer_gae)
                scaler.step(optimizer_mapping)
                scaler.update()      

                del z_sc, z_st, z_sc_norm, z_st_norm, x_recon_sc, x_recon_st, logits_st
                del x_sc, x_st, pos_st
                del sc_data, st_data, main_data

                epoch_loss += total_loss.detach().item()
                
            if epoch % 20 == 0 and epoch > 0:
                gc.collect()
            
            if epoch % 500 == 0 and epoch > 0:
                for g in optimizer_gae.param_groups: g['lr'] *= 0.5
                for g in optimizer_mapping.param_groups: g['lr'] *= 0.5
                
            avg_loss = epoch_loss/(batch_idx+1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

            early_stopping.step(avg_loss)
            if early_stopping.counter == 0:
                best_state_dicts = {
                    'embed': {k: v.cpu().clone() for k, v in self.embed_model.state_dict().items()},
                    'mapping': {k: v.cpu().clone() for k, v in self.mapping_model.state_dict().items()},}
            if early_stopping.early_stop:
                print(f"\nStop Epoch {epoch}，Best loss: {early_stopping.best_loss:.4f}")
                break
                    
        if best_state_dicts is not None:
            self.embed_model.load_state_dict(best_state_dicts['embed'])
            self.mapping_model.load_state_dict(best_state_dicts['mapping'])

        self._predict_and_map(sc_adata, st_adata, sc_loader, st_loader, 
                              filter_mapped, alpha=class_alpha, resolution=resolution)
        total_time = timeit.default_timer() - total_start_time
      
        return sc_adata, st_adata, self.embed_model, self.mapping_model
    
    def _predict_and_map(self, sc_adata, st_adata, sc_loader, st_loader, 
                         filter_mapped=False, filter_strategy="mad",
                         alpha=0.2, resolution='low_resolution'):
        setup_seed(self.seed) 
        self.embed_model.eval()
        self.mapping_model.eval()
        
        z_sc_list, logits_sc_list = [], []
        z_st_list, logits_st_list = [], []
        st_coords = torch.tensor(st_adata.obsm['spatial'], dtype=torch.float32, device=self.device)
        target_col = 'cluster' if 'cluster' in st_adata.obs else 'label'

        with torch.inference_mode():
            for d in sc_loader:
                z, _, lg = self.embed_model(d.x.to(self.device), d.edge_index.to(self.device))
                z_sc_list.append(z)
                logits_sc_list.append(lg.cpu())
                
            for d in st_loader:
                z, _, lg = self.embed_model(d.x.to(self.device), d.edge_index.to(self.device))
                z_st_list.append(z)
                logits_st_list.append(lg.cpu())
        
        Z_sc = torch.cat(z_sc_list, dim=0)
        Z_st = torch.cat(z_st_list, dim=0)
        Logits_sc = torch.cat(logits_sc_list, dim=0)
        
        sc_adata.obsm['embeddings'] = Z_sc.cpu().numpy()
        st_adata.obsm['embeddings'] = Z_st.cpu().numpy()

        with torch.inference_mode():
            n_sc = Z_sc.size(0)
            shuffle_idx = torch.randperm(n_sc, device=self.device, generator=self.generator)
            reverse_idx = torch.argsort(shuffle_idx)
            Z_sc_shf = Z_sc[shuffle_idx]

            if self.region_penalty is not None:     
                penalty_shf = self.region_penalty[shuffle_idx]
            else:
                penalty_shf = None
                
            M_list = []
            chunk_size = 5000
            for i in range(0, n_sc, chunk_size):
                chunk = Z_sc_shf[i : i + chunk_size]
                pen_chunk = penalty_shf[i : i + chunk_size] if penalty_shf is not None else None
                m_part = self.mapping_model(chunk, Z_st, region_penalty=pen_chunk)
                M_list.append(m_part.cpu())
                
            M_final = torch.cat(M_list, dim=0)[reverse_idx.cpu()]
        
        sc_adata.obsm['mapping_matrix'] = M_final.numpy()

        conf_m = M_final.sum(dim=1, keepdim=True)
        P_norm = M_final / (conf_m + 1e-8)

        categories = st_adata.obs[target_col].astype('category').cat.categories
        num_cls = len(categories)
        st_lab_idx = torch.tensor(st_adata.obs[target_col].astype('category').cat.codes.values, dtype=torch.long)
        st_lab_idx = torch.clamp(st_lab_idx, min=0)
        one_hot_st = F.one_hot(st_lab_idx, num_classes=num_cls).float()
        probs_m = torch.matmul(P_norm, one_hot_st)
        probs_cls = torch.softmax(Logits_sc, dim=1)

        final_idx, final_probs = fuse_fixed_alpha(probs_cls, probs_m, alpha=alpha)
        label_map = {i: c for i, c in enumerate(categories)}
        sc_adata.obs['cluster_pred'] = [label_map[i] for i in final_idx]
        sc_adata.obs['cluster_prob'] = final_probs

        coords, _, s_std, ent = infer_spatial_coordinates_with_uncertainty(P_norm, 
                                                                           st_coords.cpu(),
                                                                           resolution=resolution)
        sc_adata.obsm['spatial_pred'] = coords.numpy()
        sc_adata.obs['mapping_confidence'] = conf_m.numpy().squeeze()
        sc_adata.obs['spatial_uncertainty'] = s_std.numpy()
        sc_adata.obs['mapping_entropy'] = ent.numpy()

        
        if filter_mapped:
            conf = sc_adata.obs["mapping_confidence"].to_numpy()
            if filter_strategy == "mad":
                threshold = max(np.median(conf) - 2 * median_abs_deviation(conf), 0.0)
                is_mapped = conf > threshold            
            elif filter_strategy == "confidence80":
                threshold = np.quantile(conf, 0.20)
                is_mapped = conf >= threshold            
            elif filter_strategy == "confidence_entropy80":
                entropy = sc_adata.obs["mapping_entropy"].to_numpy()
                is_mapped = ((conf >= np.quantile(conf, 0.20)) &(entropy <= np.quantile(entropy, 0.80)))            
            sc_adata.obs["is_mapped"] = is_mapped
        else:
            sc_adata.obs["is_mapped"] = True

    
        if 'celltype' in sc_adata.obs:
            celltype_series = sc_adata.obs['celltype'].astype('category')
            celltype_codes = torch.tensor(celltype_series.cat.codes.values, dtype=torch.long, device=M_final.device)
            
            sc_X_raw = sc_adata.raw.X if sc_adata.raw is not None else sc_adata.X
            total_counts_per_cell = np.asarray(sc_X_raw.sum(axis=1)).squeeze()
            sc_adata.obs['total_counts'] = total_counts_per_cell
            mean_size_per_ct = sc_adata.obs.groupby('celltype')['total_counts'].mean().values
            size_factors = mean_size_per_ct / np.median(mean_size_per_ct)
            size_factors_tensor = torch.tensor(size_factors, dtype=torch.float32, device=M_final.device)

            spot_proportions, spot_dominant_type, spot_entropy = infer_deconvolute_st(
                M_final, celltype_codes, size_factors=size_factors_tensor)
            
            st_adata.obsm['spot_proportions_pred'] = spot_proportions.cpu().numpy()
            st_adata.obs['spot_entropy'] = spot_entropy.cpu().numpy()
            dominant_type_codes = spot_dominant_type.cpu().numpy()
            st_adata.obs['spot_dominant_type_pred'] = pd.Categorical.from_codes(dominant_type_codes, 
                                                                            categories=celltype_series.cat.categories)

            cell_types_list = celltype_series.cat.categories.tolist()
            st_adata.uns['cell_types'] = cell_types_list


