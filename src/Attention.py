""""
Title: Attention.py
Author: Han Tong
Date: 2024-07-26
Python Version: Python 3.11.3
Description: All attention model we use
"""
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax, dropout_adj, add_self_loops, remove_self_loops
from torch_scatter import scatter
import warnings
warnings.filterwarnings('ignore')

from config import get_config
from load_data import *

config = get_config()
CHECK_ALL = config['CHECK_ALL']

    
'''
Models
'''
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads, concat, dropout, init0=False, linear=True):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_features, out_features, heads, concat=concat, add_self_loops=True, bias=True)
        self.batch_norm = nn.BatchNorm1d(out_features * heads if concat else out_features)
        self.activation = nn.ReLU()
        self.dropout_rate = dropout
        self.linear = linear
        if linear:
            self.Linear = nn.Linear(out_features * heads if concat else out_features, out_features)
        
        if init0 is True:
            # Initialize the lower half of gat_conv.lin_src.weight to zeros
            lower_tri_indices = range(int(in_features/2), in_features)
            self.gat_conv.lin_src.weight.data[:, lower_tri_indices] = 0.0

    def forward(self, x, edge_index):
        # only training step will drop edge
        edge_index, _ = dropout_adj(edge_index, p=self.dropout_rate, force_undirected=True, training=self.training)        
        out = self.gat_conv(x, edge_index)
        out = self.batch_norm(out)
        out = self.activation(out)
        if self.linear:
            out = self.Linear(out)
        return out
    

class inst_encoder(nn.Module):
    def __init__(self, config):
        super(inst_encoder, self).__init__()
        self.device = config['DEVICE']
        if config['Decoder'] is True:
            self.GAT = GATLayer(config['hidden_features'], config['hidden_features'], config['heads'], True, config['drop_p'], init0=False, linear=False)
            # self.Linear = nn.Linear(2*config['hidden_features'], config['out_dim'])
            self.Linear = nn.Linear(config['hidden_features'], config['out_dim'])
            self.X = nn.Parameter(torch.eye(config['hidden_features']))
        else:
            if config['path_origin'] == 'align_NA':
                self.inst = torch.nn.ModuleList([GATLayer(config['num_features'], config['hidden_features'], config['heads'], True, config['drop_p']) for i in range(config['num_inst'])]) 
            else:
                self.GAT_together = GATLayer(2 * config['hidden_features'], config['hidden_features'], config['heads'], True, config['drop_p'], init0=True, linear=False) 
            if config['path_origin'] is None:
                self.Linear = nn.Linear(2*config['hidden_features'], config['rmax'])
            elif config['path_origin'] != 'align_NA':
                self.Linear = nn.Linear(2*config['hidden_features'], config['out_dim'] - config['rmax'])
        
    def align_loss(self, new_sppmi_list, config):
        num_inst = config['num_inst']
        loss = 0
        # weights = torch.tensor(config['inst_weight'], device='cuda')
        for i in range(num_inst):  # num_inst must be bigger than 1
            for j in range(num_inst):
                loss += torch.norm(new_sppmi_list[i][config['inst_row'][i],:] - new_sppmi_list[j][config['inst_row'][i],:], 'fro')
                # loss += 1/(weights[i]) * (torch.norm(new_sppmi_list[i][config['inst_row'][i],:] - new_sppmi_list[j][config['inst_row'][i],:], 'fro'))
        print(f"align_loss: {loss * config['scale_align']}")
        return loss * config['scale_align']


    
    def forward(self, sppmi_list = None, sap_emb=None, edge_index=None, encoder_emb=None, out_1=None, config=None):
        if config['Decoder'] is True:
            # tmp = self.GAT(encoder_emb, edge_index)
            # tmp = self.Linear(tmp)
            tmp = self.Linear(encoder_emb)
            decoder_emb = tmp / torch.norm(tmp, dim=1, keepdim=True) 
            # decoder_emb = torch.matmul(encoder_emb, self.X)
            return decoder_emb
         
        if config['path_origin'] == "align_NA":
            all_emb_list = []
            # Now we need to align sppmi emb together, and store this embedding
            for i in range(config['num_inst']): 
                inst_emb = self.inst[i](sppmi_list[i], edge_index)                
                # # tmp
                # mask = torch.zeros_like(inst_emb, device=self.device)
                # mask[config['inst_row'][i]] = 1
                # inst_emb = inst_emb * mask

                all_emb_list.append(inst_emb)
            
            # weights = torch.tensor(config['inst_weight'], device='cuda')
            # all_emb = sum(w * emb for w, emb in zip(weights, all_emb_list))
            all_emb = torch.sum(torch.stack(all_emb_list), dim=0)
            out_1 = all_emb / torch.norm(all_emb, dim=1, keepdim=True)
            
            if self.training:
                # get aligned loss
                align_loss_term = self.align_loss(all_emb_list, config)
                if config['CHECK_ALL']:
                    print(f'align_loss_term={align_loss_term}')
                return config['scale_align'] * align_loss_term, out_1
            return out_1
            
        # concatenate with sapbert embedding, to build simi/rela gat embedding
        if config['path_origin'] is None:
            out_2 = torch.concat((sap_emb, out_1), dim=1)
        else:
            out_2 = torch.concat((out_1, sap_emb), dim=1)
            
        # get unified representation
        uni_tmp = self.GAT_together(out_2, edge_index)
        uni = self.Linear(uni_tmp)
        uni = uni / torch.norm(uni, dim=1, keepdim=True)
        return uni
    
    
'''
Loss Functions
'''
def custom_loss(my_objects, x, now_index, device, name_all, config, TYP1=False):  
    '''
    Return 5 parts loss
    1. hierarchy loss
    2. Local lab to Loinc Loss
    3. Related Pairs Loss
    4. SIM_NO_HIE Loss 
    5. SPPMI pos&neg Loss
    '''

    def loss_term(temp_objects, x, now_index, AA=config['AA'], BB=config['BB'], lambd=config['lambd']):
        '''
        In 1, 2, 3, 4 parts of loss, the formats are all like
        \begin{equation}
        \begin{aligned}
           \mathcal{L}_{i}  & = \frac{1}{\alpha} \log \Bigg ( 1 + \frac{1}{|\widetilde{\mathcal{P}}_{1i} |} 
           \sum_{j \in \widetilde{\mathcal{P}}_{1i} }e^{-\alpha (\mathbf{Z}_i^T \mathbf{Z}_j - \lambda)} \Bigg) \\
           & + \frac{1}{\beta}\log \Bigg ( 1 + \frac{1}{|\widetilde{\mathcal{N}}_{1i}|} 
           \sum_{j \in \widetilde{\mathcal{N}}_{1i} }e^{\beta (\mathbf{Z}_i^T \mathbf{Z}_j - \lambda)} \Bigg)  
        \end{aligned}
        \end{equation}
        '''
        loss1 = (1 / AA) * sum(torch.log(1 + (1 / len(temp_objects[i].sampled_set1)) * sum(torch.exp(- AA * (torch.dot(x[i, :], x[j, :]) - lambd)) for j in temp_objects[i].sampled_set1)) for i in range(len(now_index)) if len(temp_objects[i].sampled_set1) > 0)
        loss2 = (1 / BB) * sum(torch.log(1 + (1 / len(temp_objects[i].sampled_set2)) * sum(torch.exp(BB * (torch.dot(x[i, :], x[j, :]) - lambd)) for j in temp_objects[i].sampled_set2)) for i in range(len(now_index)) if len(temp_objects[i].sampled_set2) > 0)
        return loss1, loss2


    def calculate_loss(loss_type, my_objects, x, now_index, name_all, scale, DIFF=True,  AA=config['AA'], BB=config['BB'], lambd=config['lambd']):
        '''
        Aggragate the first 4 parts Loss
        '''            
        if loss_type == 'hierarchy':
            if CHECK_ALL:
                print('Hierarchy loss:')
                
            set1 = [my_objects[i].same_par for i in now_index]
            set2 = [my_objects[i].same_gra for i in now_index]
            
        elif loss_type == 'local_to_loinc':
            if CHECK_ALL:
                print('Local lab To Loinc loss:')
                
            set1 = [my_objects[i].P_local for i in now_index]
            set2 = [my_objects[i].N_local for i in now_index]
            
        elif loss_type == 'related_pairs':
            if CHECK_ALL:
                print('Related Pairs Loss:')
                
            set2 = [find_same_type(i, name_all) for i in now_index] 
            set1 = [set(my_objects[i].rel) if isinstance(my_objects[i].rel, np.ndarray) 
                    else my_objects[i].rel for i in now_index]

        elif loss_type == 'similar_no_hie_pairs':
            if CHECK_ALL:
                print('Similar Pairs (Not Hierarchy) Loss:')
            set1 = [my_objects[i].sim_no_hie for i in now_index]
            set2 = [find_same_type(i, name_all) for i in now_index]  
            
        else:
            raise ValueError("Invalid loss_type. Supported values are 'one_one','hierarchy','local_to_loinc', 'related_pairs','similar_no_hie_pairs'.")

        temp_objects = origin_term_temp(name_temp=get_values(name_all[now_index]), set1=set1, set2=set2, DIFF=DIFF)

        result = loss_term(temp_objects, x, now_index, AA=AA, BB=BB, lambd=lambd)
        result = [scale * r for r in result]
        P_LOSS = result[0]
        N_LOSS = result[1]
        
        if CHECK_ALL:
            print("Positive_Loss_{} = {:.4f}".format(loss_type, P_LOSS))
            print("Negative_Loss_{} = {:.4f}".format(loss_type, N_LOSS))
        return P_LOSS, N_LOSS
    
    if TYP1:    
        P_LOSS_hie, N_LOSS_hie = calculate_loss('hierarchy', my_objects, x, now_index, name_all, config['scale_hie'])
        P_LOSS_OTOL, N_LOSS_OTOL = calculate_loss('local_to_loinc', my_objects, x, now_index, name_all, config['scale_OTOL'], DIFF=False)
        P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE = calculate_loss('similar_no_hie_pairs', my_objects, x, now_index, name_all, config['scale_SIM_NO_HIE'])   

        return P_LOSS_hie, N_LOSS_hie, P_LOSS_OTOL, N_LOSS_OTOL, P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE
    
    else:
        if CHECK_ALL:
            print('Relative Loss:')  
        P_LOSS_REL, N_LOSS_REL = calculate_loss('related_pairs', my_objects, x, now_index, name_all, config['scale_REL'])
        return P_LOSS_REL, N_LOSS_REL

    
def decoder_loss(decoder_emb, sppmi_list, config):
    now_inst = config['Decoder_inst']
    sppmi = sppmi_list[now_inst]
    inst_index = config['inst_row'][now_inst]
    sppmi = sppmi[inst_index]
    loss = torch.norm(decoder_emb - sppmi, 'fro')
    return loss


    
def sppmi_edge_loss(x, edge_pos, edge_neg, config):
    # Determine the device based on the input tensor 'x'
    device = x.device
    max_num = x.shape[0]

    # Compute positive edge scores
    emb_start_pos = x[edge_pos[0, :]]
    emb_end_pos = x[edge_pos[1, :]]
    score_pos = torch.sum(emb_start_pos * emb_end_pos, dim=-1)

    # Compute negative edge scores
    emb_start_neg = x[edge_neg[0, :]]
    emb_end_neg = x[edge_neg[1, :]]
    score_neg = torch.sum(emb_start_neg * emb_end_neg, dim=-1)

    # Initialize node aggregates on the same device as x
    node_agg_pos = torch.zeros(x.shape[0], device=device)
    node_agg_neg = torch.zeros(x.shape[0], device=device)
    
    # Loop through positive and negative edges
    for src, tgt, score in zip(edge_pos[0], edge_pos[1], score_pos):
        node_agg_pos[src] += torch.exp(- config['AA'] * (score - config['lambd']))

    for src, tgt, score in zip(edge_neg[0], edge_neg[1], score_neg):
        node_agg_neg[src] += torch.exp(config['BB'] * (score - config['lambd']))
   
    pos_counts = torch.bincount(edge_pos[:,0], minlength=max_num)
    # print(len(pos_counts))
    neg_counts = torch.bincount(edge_neg[:,0], minlength=max_num)
    # print(len(neg_counts))
    pos_mask = pos_counts != 0
    neg_mask = neg_counts != 0
    
    # # Calculate the log term per node for positive and negative scores
    # log_term_pos = (1 / config['AA']) * torch.log(1 + node_agg_pos[pos_mask]/pos_counts[pos_mask])
    # log_term_neg = (1 / config['BB']) * torch.log(1 + node_agg_neg[neg_mask]/neg_counts[neg_mask])
    log_term_pos = (1 / config['AA']) * torch.log(1 + node_agg_pos[pos_mask])
    log_term_neg = (1 / config['BB']) * torch.log(1 + node_agg_neg[neg_mask])

    # Sum log terms across all nodes to compute the final loss
    loss = log_term_pos.sum() + log_term_neg.sum()
    
    if config['CHECK_ALL']: 
        print(f"sppmi_pos_loss: {config['scale_sppmi'] * log_term_pos.sum()}")
        print(f"sppmi_neg_loss: {config['scale_sppmi'] * log_term_neg.sum()}")
    
    return config['scale_sppmi'] * log_term_pos.sum(), config['scale_sppmi'] * log_term_neg.sum()
