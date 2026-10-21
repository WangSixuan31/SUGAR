# /env sugar-env
"""
@author: Sixuan Wang
"""
import os
os.environ['FORCE_DETERMINISTIC_CALLBACKS'] = '1'
os.environ['TORCH_CUDNN_DETERMINISTIC'] = '1'
import ot
from pykeops.torch import LazyTensor

from torch.autograd import Function
from torch_geometric.nn import GCNConv, SAGEConv#, GATConv,GATv2Conv,InnerProductDecoder
from torch.nn import Sequential, Linear, BatchNorm1d, LayerNorm, Dropout, Sigmoid, ReLU, LeakyReLU
from torch.utils.checkpoint import checkpoint 

from .utils import *



#####################################  GAE  #####################################
class Encoder(nn.Module):
    def __init__(self, in_dims, hidden_dims, dropout, latent_dim=64):
        super(Encoder, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.act = nn.LeakyReLU(inplace=True) 
        self.drop_rates = dropout if isinstance(dropout, list) else [dropout] * len(hidden_dims)
        
        current_in_dim = in_dims
        for i, h_dim in enumerate(hidden_dims):
            self.conv_layers.append(GCNConv(current_in_dim, h_dim))
            self.batch_norms.append(nn.BatchNorm1d(h_dim))
            if current_in_dim != h_dim:
                self.residual_projs.append(nn.Linear(current_in_dim, h_dim, bias=False))
            else:
                self.residual_projs.append(nn.Identity())
            current_in_dim = h_dim

        self.head_z = nn.Linear(sum(hidden_dims), latent_dim)

    def forward(self, x, edge_index):
        h = x
        layer_outputs = []
        for i, conv in enumerate(self.conv_layers):
            identity = self.residual_projs[i](h)
            h = conv(h, edge_index)
            h = self.batch_norms[i](h)
            h = self.act(h + identity)
            if i < len(self.drop_rates):
                h = F.dropout(h, p=self.drop_rates[i], training=self.training)
            layer_outputs.append(h)
        h_jk = torch.cat(layer_outputs, dim=1)  
        z = self.head_z(h_jk)
        return z
        


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dims, dropout):
        super(Decoder, self).__init__()
        layers = []
        curr_dim = latent_dim
        
        if isinstance(dropout, (float, int)):
            dropout = [dropout] * len(hidden_dims)
            
        for h_dim, drop_p in zip(reversed(hidden_dims), reversed(dropout)):
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) #BatchNorm1d
            layers.append(nn.LeakyReLU(inplace=True)) 
            if drop_p > 0.0:
                layers.append(nn.Dropout(drop_p))
                
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dims))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)



class GraphAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, latent_dim, n_labels):
        super(GraphAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 128),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Dropout(0.1),
                                        nn.Linear(128, n_labels))

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_recon = self.decoder(z)
        logits = self.classifier(z)
        return z, x_recon, logits



#####################################  UOT  #####################################
def build_prior_penalty(sc_type_labels, st_region_labels,
                        type_to_region, penalty_value=1.0, 
                        device='cuda'):
    N_sc, N_st = len(sc_type_labels), len(st_region_labels)
    penalty = torch.zeros(N_sc, N_st, device=device)
    st_arr = np.asarray(st_region_labels)
    for i, t in enumerate(sc_type_labels):
        allowed = type_to_region.get(t)
        if allowed is None:
            continue
        bad = ~np.isin(st_arr, allowed)
        penalty[i, torch.tensor(bad, device=device)] = penalty_value
    return penalty



class UnbalancedMappingModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, 
                 eps=0.1, tau_sc=0.5, tau_st=0.5,
                 prior_lambda=10): 
        super(UnbalancedMappingModel, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]),
                                 nn.LayerNorm(hidden_dims[0]),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Dropout(dropout[0] if isinstance(dropout, list) else dropout),
                                 nn.Linear(hidden_dims[0], hidden_dims[1]),)

        nn.init.orthogonal_(self.net[0].weight)
        nn.init.orthogonal_(self.net[-1].weight)

        self.eps = eps
        self.tau_sc = tau_sc       
        self.tau_st = tau_st  
        self.prior_lambda = prior_lambda  

    def project(self, z):
        h = self.net(z)
        return F.normalize(h, p=2, dim=1, eps=1e-8)

    def forward(self, z_sc, z_st, region_penalty=None):
        h_sc = self.project(z_sc)
        h_st = self.project(z_st)

        cost_matrix = 1.0 - torch.matmul(h_sc, h_st.t())
        if region_penalty is not None:
            cost_matrix = cost_matrix + region_penalty * self.prior_lambda   # prior

        N_sc, N_st = h_sc.shape[0], h_st.shape[0]
        a = torch.ones(N_sc, device=h_sc.device) / N_sc
        b = torch.ones(N_st, device=h_st.device) / N_st

        def run_ot_checkpoint(cost):
            return ot.unbalanced.sinkhorn_unbalanced(a, b, cost, 
                                                     reg=self.eps, 
                                                     reg_m=[self.tau_sc, self.tau_st],    
                                                     numItermax=10,
                                                     method='sinkhorn')
        cost_matrix = cost_matrix.float()
        M = checkpoint(run_ot_checkpoint, cost_matrix, use_reentrant=False)
        return M


