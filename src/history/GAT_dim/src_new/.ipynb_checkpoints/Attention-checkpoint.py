""""
Title: Attention.py
Author: Han Tong
Date: 2023-12-06
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
# from torch_scatter import scatter
import warnings
warnings.filterwarnings('ignore')

from config import get_config
from load_data import *
config = get_config()
CHECK_ALL = config['CHECK_ALL']

'''
h^{T}Ah Attention
'''
class GATLayer_A(nn.Module):
    '''
    \begin{align}
        h_i^k = W^k x_i
        e_{ij} &=  h_i^T A h_j  \\
        \alpha_{ij} &= \frac{exp(e_{ij})}{\sum_{k \in \mathcal{N}_i}exp(e_{ik})}\\
        h_i' &=||_{k=1}^K \sigma(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k  h_j)
    \end{align}   

    \begin{align}
        e_{ij} &=  h_i^T VV^T h_j  \\
        \alpha_{ij} &= \frac{exp(e_{ij})}{\sum_{k \in \mathcal{N}_i}exp(e_{ik})}\\
        h_i' &=||_{k=1}^K \sigma(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k h_j)
    \end{align} 

    :param low_dim is dimension of V if not 0
    '''
    def __init__(self, in_features, out_features, K, low_dim=0, residual=False):
        super(GATLayer_A, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(in_features, out_features)) for _ in range(K)])
        self.LOW_DIM = low_dim
        self.batch_norm = nn.BatchNorm1d(out_features * K)
        self.dropout = nn.Dropout(config['drop_p'])
        self.activation = nn.ReLU()
        if low_dim:
            self.V = nn.ParameterList([nn.Parameter(torch.Tensor(out_features, low_dim)) for _ in range(K)])
        else:
            self.A = nn.Parameter(torch.Tensor(out_features, out_features))
            nn.init.xavier_uniform_(self.A.data)
        
        self.residual = residual
        
        if str(self.residual).lower() != "false":
            self.res_connection = nn.Linear(in_features, K * out_features)
            nn.init.xavier_uniform_(self.res_connection.weight)
        
        for k in range(K):
            nn.init.xavier_uniform_(self.W[k].data)
            if low_dim:
                nn.init.xavier_uniform_(self.V[k].data)

    def forward(self, h, edge_index):
        H_new = []
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        
        if self.LOW_DIM:
            for k in range(self.K):
                h_k = torch.mm(h, self.W[k]) # [num_nodes, out_features]
                h_proj = h_k @ self.V[k] @ self.V[k].transpose(0,1) # [num_nodes, out_features]
                row, col = edge_index
                alpha = (h_proj[row] * h_k[col]).sum(dim=-1, keepdim=True)  # [num_edges, 1]
                
                if config['Truncate']:
                    alpha = retain_topk_alpha(alpha, edge_index, config['truncate_mask'])
                
                alpha = softmax(alpha, edge_index[0])  # [num_edges, 1]
                
                if CHECK_ALL:
                    print(':::::::ATTENTION EXAMPLE:::::::')
                    print(alpha[[np.where(edge_index[0,:].cpu().detach() == 7400)[0]]])
                
                h_weighted = h_k[col] * alpha.view(-1, 1)
                h_new = scatter(src=h_weighted, index=row, dim=0, reduce="add")
                H_new.append(h_new)
                
                h_proj2 = h_k @ self.V[k] @ self.V[k].transpose(0,1) # [num_nodes, in_features]
                
        else:
            for k in range(self.K):
                h_k =  torch.mm(h, self.W[k]) # [num_nodes, out_features]
                h_proj = h_k @ self.A  # [num_nodes, in_features]
                row, col = edge_index
                alpha = (h_proj[row] * h_k[col]).sum(dim=-1, keepdim=True)  # [num_edges, 1]
                if config['Truncate']:
                    alpha = retain_topk_alpha(alpha, edge_index, config['truncate_mask'])
                alpha = softmax(alpha, edge_index[0])  # [num_edges, 1]
                h_weighted = h_k[col] * alpha.view(-1, 1)
                h_new = scatter(src=h_weighted, index=row, dim=0, reduce="add")
                H_new.append(h_new)
                h_proj2 = h_k @ self.A  # [num_nodes, out_features]

            if CHECK_ALL:
                print(':::::::ATTENTION EXAMPLE:::::::')
                print(alpha[[np.where(edge_index[0,:].cpu().detach() == 7400)[0]]])
        
        H_new = torch.cat(H_new, dim=1)  # [num_nodes, K * out_features]

        if str(self.residual).lower() != "false":
            res = self.res_connection(h)
            H_new = H_new + res
            print('res!')
        
        # H_new = self.batch_norm(H_new)
        H_new = self.dropout(H_new)
        H_new = self.activation(H_new)
       
        return H_new, alpha

    
class GAT_A(nn.Module):
    def __init__(self, config, SIMI=True):
        super(GAT_A, self).__init__()
        self.gat1 = GATLayer_A(config['num_features'], config['hidden_features'], config['K'], config['low_dim'], residual=config['res'])
        self.gat2 = GATLayer_A(config['hidden_features'] * config['K'], config['hidden_features'], config['K'], config['low_dim'], residual=config['res'])
        if SIMI:
            self.linear = nn.Linear(config['K'] * config['hidden_features'], config['rmax'])
        else:
            self.linear = nn.Linear(config['K'] * config['hidden_features'], config['out_dim'] - config['rmax'])
        
    def forward(self, x, edge_index):
        x, alpha1 = self.gat1(x, edge_index)
        x, alpha2 = self.gat2(x, edge_index)
        x = self.linear(x)
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x, alpha1, alpha2

    
'''
Naive Attention
'''
    
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads, concat, dropout, residual=False):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_features, out_features, heads, concat=concat, add_self_loops=True, bias=True)
        self.batch_norm = nn.BatchNorm1d(out_features * heads)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.residual = residual
        
        if str(self.residual).lower() != "false":
            self.res_connection = nn.Linear(in_features, out_features * heads)
            nn.init.xavier_uniform_(self.res_connection.weight)

    def forward(self, x, edge_index):
        out, attention_weights = self.gat_conv(x, edge_index, return_attention_weights=True)
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.activation(out)
        
        if str(self.residual).lower() != "false":
            res = self.res_connection(x)
            out = out + res
            print('res!')
            
        return out, attention_weights

    
class GATModel(nn.Module):
    def __init__(self, config, SIMI=True):
        super(GATModel, self).__init__()
        self.gat1 = GATLayer(config['num_features'], config['hidden_features'], 2, True, config['drop_p'], residual=config['res'])
        self.gat2 = GATLayer(2 * config['hidden_features'], 1 * config['hidden_features'], 3, True, config['drop_p'], residual=config['res'])
        if SIMI:
            self.linear = nn.Linear(3 * config['hidden_features'], config['rmax']) 
        else:
            self.linear = nn.Linear(3 * config['hidden_features'], config['out_dim'] - config['rmax'])

    def forward(self, x, edge_index):
        x1, attention1 = self.gat1(x, edge_index)
        x2, attention2 = self.gat2(x1, edge_index)
        x_out = self.linear(x2)
        x_out = x_out / torch.norm(x_out, dim=1, keepdim=True) 
        return x_out, attention1, attention2

    
'''
Combine the related and similar model
'''
class SandR_Model(nn.Module):
    def __init__(self, config, ONLY_SIMI = config['ONLY_SIMI']):
        super(SandR_Model, self).__init__()
        if config['ATTENTION_TYPE'] == "Naive":
            if config['path_origin'] is None:
                self.S_Model = GATModel(config, SIMI=True)
            if str(config['ONLY_SIMI']).lower() == 'false':
                self.R_Model = GATModel(config, SIMI=False)
        else:
            if config['path_origin'] is None:
                self.S_Model = GAT_A(config, SIMI=True)
            if str(config['ONLY_SIMI']).lower() == 'false':
                self.R_Model = GAT_A(config, SIMI=False)
            
    def forward(self, x, edge_index):
        if str(config['ONLY_SIMI']).lower() == 'false':
            x_rel_part, _, _ = self.R_Model(x, edge_index)
        if config['path_origin'] is None:
            x_sim, _, _ = self.S_Model(x, edge_index)
            if str(config['ONLY_SIMI']).lower() == 'false':
                return x_sim, x_rel_part
            else:
                return x_sim
        else:
            return x_rel_part

'''
Total model
'''

class SparseLayer(nn.Module):
    def __init__(self, input_dim, output_dim, mask, origin_weight=None):
        super(SparseLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.mask = mask
        if origin_weight is not None:
            assert origin_weight.shape == (output_dim, input_dim), "The shape of original weight must match with weight."
            self.weight.data = origin_weight.clone()
        else:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 
        
    def forward(self, input):
        # Multiply weight with mask to enforce sparsity
        sparse_weight = self.weight * self.mask
        return torch.mm(sparse_weight, input)
    
    
class ExtendedGAT(nn.Module):
    def __init__(self, SandR):
        super(ExtendedGAT, self).__init__()
        self.origin_model = SandR
        self.sparseLayer = SparseLayer(config['num_nodes'], config['num_latent'], mask, origin_weight)

    def forward(self, x, edge_index, FROZEN = config['FROZEN']):
        if FROZEN is False:
            x = self.origin_model(x, edge_index)
        x_new = self.sparseLayer(x)
        if FROZEN is False:
            return x, x_new
        return x_new

    
'''
Loss Functions
'''
def custom_loss(my_objects, x, x_rel, now_index, device, name_all,COS_origin_sap=None, 
                COS_origin_svd=None, scale_one_one=config['scale_one_one'], 
                scale_hie=config['scale_hie'], scale_OTOL=config['scale_OTOL'], 
                scale_LTOL=config['scale_LTOL'], scale_REL=config['scale_REL'], 
                scale_SIM_NO_HIE=config['scale_SIM_NO_HIE'], 
                scale_LSAP=config['scale_LSAP'], scale_LSVD=config['scale_LSVD'], 
                ORIGIN=True, rmax = config['rmax']):  
    '''
    Return 8 parts loss
    1. one-one loss
    2. hierarchy loss
    3. Other lab to Loinc Loss
    4. Local lab to Loinc Loss
    5. Related Pairs Loss
    6. SIM_NO_HIE Loss
    7. Sapbert Embedding Loss
    8. SVD PPMI Embedding Loss
    '''

    def loss_term(temp_objects, x, now_index, AA=config['AA'], BB=config['BB'], lambd=config['lambd']):
        '''
        In 1, 2, 3, 4, 5, 6 parts of loss, the formats are all like

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
        loss2 = (1 / BB) * sum(torch.log(1 + (1 / len(temp_objects[i].sampled_set2)) * sum(torch.exp(BB * (torch.dot(x[i, :], x[j, :]) - lambd)) 
                                                                                  for j in temp_objects[i].sampled_set2)) for i in range(len(now_index)) if len(temp_objects[i].sampled_set2) > 0)
        return loss1, loss2


    def calculate_loss(loss_type, my_objects, x, now_index, name_all, scale, DIFF=True,  AA=config['AA'], BB=config['BB'], lambd=config['lambd'], ORIGIN=True):

        '''
        Aggragate the first 5 parts Loss
        '''

        if loss_type == 'one_one':
            if CHECK_ALL:
                print('One-one loss:')
                
            set1 = [my_objects[i].one_one for i in now_index]
            set2 = [my_objects[i].same_par if len(my_objects[i].same_par) > 0 else my_objects[i].same_gra 
                    if len(my_objects[i].same_gra) > 0 else find_same_type(i, name_all) for i in now_index]
            
        elif loss_type == 'hierarchy':
            if CHECK_ALL:
                print('Hierarchy loss:')
                
            set1 = [my_objects[i].same_par for i in now_index]
            set2 = [my_objects[i].same_gra for i in now_index]
            
        elif loss_type == 'other_to_loinc':
            if CHECK_ALL:
                print('Other To Loinc loss:')
                
            set1 = [my_objects[i].P_other for i in now_index]
            set2 = [my_objects[i].N_other for i in now_index]
            
        elif loss_type == 'local_to_loinc':
            if CHECK_ALL:
                print('Local To Loinc loss:')
                
            set1 = [my_objects[i].P_local for i in now_index]
            set2 = [my_objects[i].N_local for i in now_index]
            
        elif loss_type == 'related_pairs':
            
            if CHECK_ALL:
                print('Related Pairs Loss:')
            set2 = [find_same_type(i, name_all, INST=ORIGIN) for i in now_index] # within inst
            set1 = [set(my_objects[i].rel) if isinstance(my_objects[i].rel, np.ndarray) 
                    else my_objects[i].rel for i in now_index]

        elif loss_type == 'similar_no_hie_pairs':
            if CHECK_ALL:
                print('Similar Pairs (Not Hierarchy) Loss:')
            set1 = [my_objects[i].sim_no_hie for i in now_index]
            set2 = [find_same_type(i, name_all, INST=ORIGIN) for i in now_index] # within inst     
            
        else:
            raise ValueError("Invalid loss_type. Supported values are 'one_one','hierarchy','other_to_loinc','local_to_loinc', 'related_pairs','similar_no_hie_pairs'.")

        temp_objects = origin_term_temp(name_temp=get_values(name_all[now_index]), set1=set1, set2=set2, DIFF=DIFF)

        result = loss_term(temp_objects, x, now_index, AA=AA, BB=BB, lambd=lambd)
        result = [scale * r for r in result]
        P_LOSS = result[0]
        N_LOSS = result[1]
        
        if CHECK_ALL:
            print("Positive_Loss_{} = {:.4f}".format(loss_type, P_LOSS))
            print("Negative_Loss_{} = {:.4f}".format(loss_type, N_LOSS))
        return P_LOSS, N_LOSS


    def calculate_loss2(COS_origin, emb_now, now_index, scale, device):
        '''
        calculate sapbert/SPPMI embedding cosine similarity loss
        '''
        now_index_tensor = torch.tensor(now_index) 
        COS_origin1 = torch.index_select(COS_origin, 0, now_index_tensor)
        COS_origin2 = torch.index_select(COS_origin1, 1, now_index_tensor)
        COS_origin2 = COS_origin2.to(device)

        x1_now = torch.index_select(emb_now, 0, now_index_tensor.to(device))                
        COS_now = torch.matmul(x1_now, x1_now.transpose(0, 1))
        COS_now = COS_now.to(device)
        pos_loss1 = scale *  torch.norm(COS_now - COS_origin2, p="fro")
        
        if CHECK_ALL:
            print("pos_loss1 = {:.4f}".format(pos_loss1))  

        # sample another len(now_index) rows to compute the cosine similarity 
        sample_range = set(range(config['num_nodes']))- set(now_index)
        now_index2 = np.random.choice(list(sample_range), len(now_index), replace=False)
        now_index2_tensor = torch.tensor(now_index2)

        COS_origin3 = torch.index_select(COS_origin, 0, now_index2_tensor)
        COS_origin4 = torch.index_select(COS_origin3, 1, now_index2_tensor)
        COS_origin4 = COS_origin4.to(device)
        
        x2_now = torch.index_select(emb_now, 0, now_index2_tensor.to(device))
        COS_now2 = torch.matmul(x2_now, x2_now.transpose(0, 1))
        COS_now2 = COS_now2.to(device)
        pos_loss2 = scale * torch.norm(COS_now2 - COS_origin4, p="fro")
        
        if CHECK_ALL:
            print("pos_loss2 = {:.4f}".format(pos_loss2))
        return pos_loss1, pos_loss2     
    

    # Call the function to compute first 5 types of losses
    if x is not None:
        if ORIGIN is True:
            P_LOSS_one_one, N_LOSS_one_one = calculate_loss('one_one', my_objects, x, now_index, name_all, scale_one_one)
        else:
            P_LOSS_one_one = None
            N_LOSS_one_one = None
            
        P_LOSS_hie, N_LOSS_hie = calculate_loss('hierarchy', my_objects, x, now_index, name_all, scale_hie,ORIGIN=ORIGIN)
        P_LOSS_OTOL, N_LOSS_OTOL = calculate_loss('other_to_loinc', my_objects, x, now_index, name_all, scale_OTOL, DIFF=False)
        P_LOSS_LTOL, N_LOSS_LTOL = calculate_loss('local_to_loinc', my_objects, x, now_index, name_all, scale_LTOL, DIFF=False)
        P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE = calculate_loss('similar_no_hie_pairs', my_objects, x, now_index, name_all, scale_SIM_NO_HIE, ORIGIN=ORIGIN)   
        
        if ORIGIN is True:
            if CHECK_ALL:
                print('Sapbert embedding Cosine Loss:')
            P_LOSS_SAP1, P_LOSS_SAP2 = calculate_loss2(COS_origin_sap, x, now_index, scale_LSAP, device)
    
    else:
        P_LOSS_one_one = None
        N_LOSS_one_one = None
        P_LOSS_hie = None
        N_LOSS_hie = None
        P_LOSS_OTOL = None
        N_LOSS_OTOL = None
        P_LOSS_LTOL = None
        N_LOSS_LTOL = None
        P_LOSS_SIM_NO_HIE = None
        N_LOSS_SIM_NO_HIE = None
        P_LOSS_SAP1 = None
        P_LOSS_SAP2 = None
        
    if x_rel is not None:
        P_LOSS_REL, N_LOSS_REL = calculate_loss('related_pairs', my_objects, x_rel, now_index, name_all, scale_REL, ORIGIN=ORIGIN)
        if ORIGIN is True:
            if CHECK_ALL:
                print('SVD embedding Cosine Loss:')  
            P_LOSS_SVD1, P_LOSS_SVD2 = calculate_loss2(COS_origin_svd, x_rel/torch.sqrt(torch.tensor(2, dtype=torch.float32)), now_index, scale_LSVD, device)
        else:
            P_LOSS_REL = None
            N_LOSS_REL = None
            
    else:
        P_LOSS_REL = None
        N_LOSS_REL = None
        P_LOSS_SVD1 = None
        P_LOSS_SVD2 = None    
  
    return P_LOSS_one_one, N_LOSS_one_one, P_LOSS_hie, N_LOSS_hie, P_LOSS_OTOL, N_LOSS_OTOL, P_LOSS_LTOL, N_LOSS_LTOL, P_LOSS_REL, N_LOSS_REL, P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE, P_LOSS_SAP1, P_LOSS_SAP2, P_LOSS_SVD1, P_LOSS_SVD2
  

# def sppmi_edge_loss(x, edge_pos, edge_neg):
#     # Get embeddings for positive and negative edges
#     emb_start_pos = x[edge_pos[0, :]]
#     emb_end_pos = x[edge_pos[1, :]]
#     emb_start_neg = x[edge_neg[0, :]]
#     emb_end_neg = x[edge_neg[1, :]]

#     # Calculate score for positive and negative edges
#     score_pos = torch.sum(emb_start_pos * emb_end_pos, dim=-1)
#     score_neg = torch.sum(emb_start_neg * emb_end_neg, dim=-1)

#     # Calculate loss for positive and negative edges
#     # loss_pos = nn.functional.relu(1 - score_pos).sum()
#     # loss_pos = torch.log(1 + torch.exp(- score_pos)).sum()
#     loss_pos = (1 / config['AA']) * torch.log(1 + torch.exp(- (config['AA'] * score_pos - config['lambd']))).sum()
#     # loss_pos = 10 * F.binary_cross_entropy_with_logits(score_pos, torch.ones_like(score_pos))
#     # loss_neg = nn.functional.relu(score_neg).sum()
#     # loss_neg = torch.log(1 + torch.exp(score_neg)).sum()
#     # loss_neg = F.binary_cross_entropy_with_logits(score_neg, torch.zeros_like(score_neg))
#     loss_neg = (1 / config['BB']) * torch.log(1 + torch.exp(config['BB'] * score_neg - config['lambd'])).sum()
#     # Combine the losses
#     loss = loss_pos + loss_neg
#     print(f'sppmi_pos_loss: {loss_pos}')
#     print(f'sppmi_neg_loss: {loss_neg}')
#     # return loss
#     return loss

def sppmi_edge_loss(x, edge_pos, edge_neg, config):
    # Compute positive edge scores
    emb_start_pos = x[edge_pos[0, :]]
    emb_end_pos = x[edge_pos[1, :]]
    score_pos = torch.sum(emb_start_pos * emb_end_pos, dim=-1)

    # Compute negative edge scores
    emb_start_neg = x[edge_neg[0, :]]
    emb_end_neg = x[edge_neg[1, :]]
    score_neg = torch.sum(emb_start_neg * emb_end_neg, dim=-1)

    # Aggregate scores by node for positive and negative edges
    node_agg_pos = torch.zeros(x.shape[0])
    node_agg_neg = torch.zeros(x.shape[0])
    
    # Assuming edge_pos and edge_neg are 2D arrays where each column is an edge, 
    # with the first row being the source and the second being the target
    for src, tgt, score in zip(edge_pos[0], edge_pos[1], score_pos):
        node_agg_pos[src] += torch.exp(- config['AA'] * (score - config['lambd']))
        node_agg_pos[tgt] += torch.exp(- config['AA'] * (score - config['lambd']))

    for src, tgt, score in zip(edge_neg[0], edge_neg[1], score_neg):
        node_agg_neg[src] += torch.exp(config['BB'] * (score - config['lambd']))
        node_agg_neg[tgt] += torch.exp(config['BB'] * (score - config['lambd']))

    # Calculate the log term per node for positive and negative scores
    log_term_pos = (1 / config['AA']) * torch.log(1 + node_agg_pos)
    print(node_agg_neg)
    log_term_neg = (1 / config['BB']) * torch.log(1 + node_agg_neg)

    # Sum log terms across all nodes to compute the final loss
    loss = log_term_pos.sum() + log_term_neg.sum()

    print(f'sppmi_pos_loss: {log_term_pos.sum()}')
    print(f'sppmi_neg_loss: {log_term_neg.sum()}')

    return loss