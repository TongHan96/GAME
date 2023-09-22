""""
Title: Attention.py
Author: Han Tong
Date: 2023-09-22
Python Version: Python 3.11.3
Description: All attention model we use
"""

import torch
from torch_geometric.utils import softmax
from torch_scatter import scatter
import numpy as np
import math
import torch.nn as nn
import warnings
from config import get_config
warnings.filterwarnings('ignore')
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from load_data import *
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops,softmax, remove_self_loops
from torch_scatter import scatter
config = get_config()
# print(f'In Attention{config}')
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
        
        if self.residual:
            self.res_connection = nn.Linear(in_features, K * out_features)
            nn.init.xavier_uniform_(self.res_connection.weight)
        
        for k in range(K):
            nn.init.xavier_uniform_(self.W[k].data)
            if low_dim:
                nn.init.xavier_uniform_(self.V[k].data)

    def forward(self, h, edge_index, edge_attention):
        H_new = []
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        alpha2_all = []
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
                    print(alpha[[np.where(edge_index[0,:].cpu().detach() == 2771)[0]]])
                
                
                h_weighted = h_k[col] * alpha.view(-1, 1)
                h_new = scatter(src=h_weighted, index=row, dim=0, reduce="add")
                H_new.append(h_new)
                
                h_proj2 = h_k @ self.V[k] @ self.V[k].transpose(0,1) # [num_nodes, in_features]
                row, col = edge_attention
                alpha2 = (h_proj2[row] * h_k[col]).sum(dim=-1, keepdim=True)  # [num_edges, 1]
                alpha2_all.append(alpha2)
        
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
                row, col = edge_attention
                alpha2 = (h_proj2[row] * h_k[col]).sum(dim=-1, keepdim=True)  # [num_edges, 1]
                # print(alpha2.shape)
                alpha2_all.append(alpha2)

            if CHECK_ALL:
                print(':::::::ATTENTION EXAMPLE:::::::')
                print(alpha[[np.where(edge_index[0,:].cpu().detach() == 2771)[0]]])
        
        H_new = torch.cat(H_new, dim=1)  # [num_nodes, K * out_features]

        if self.residual:
            res = self.res_connection(h)
            H_new = H_new + res
        
        # H_new = self.batch_norm(H_new)
        # H_new = self.dropout(H_new)
        H_new = self.activation(H_new)
        # print(torch.cat(alpha2_all, dim=1).shape)
        
        return H_new, alpha, torch.cat(alpha2_all, dim=1).t()

    
class GAT_A(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, K, low_dim=config['low_dim']):
        super(GAT_A, self).__init__()
        self.gat1 = GATLayer_A(in_features, hidden_features, K, low_dim, residual=config['res'])
        self.gat2 = GATLayer_A(hidden_features * K, hidden_features, K, low_dim, residual=config['res'])
        self.linear = nn.Linear(K * hidden_features, out_features)
        
    def forward(self, x, edge_index, edge_attention):
        x, alpha1, rank_attention1 = self.gat1(x, edge_index, edge_attention)
        x, alpha2, rank_attention2 = self.gat2(x, edge_index, edge_attention)
        x = self.linear(x)
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x, alpha1, alpha2, rank_attention1, rank_attention2

    
'''
Naive Attention
'''

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value


class GATModel(nn.Module):
    
    def __init__(self, in_features, hidden_features, dropout=config['drop_p'], out_dim=config['all_dim'], MASK=False):
        super(GATModel, self).__init__()
        self.gat1 = GATLayer(in_features, hidden_features, 2, True, dropout)
        self.gat2 = GATLayer(2 * hidden_features, 1 * hidden_features, 2, True, dropout)
        self.linear = nn.Linear(2 * hidden_features, out_dim) 
        self.reset_parameters(MASK)
        
    def reset_parameters(self, MASK):
        temp = self.gat1.gat_conv.lin_src_new.weight
        mask = generate_mask(temp, generate_tuples())
        if MASK is True:
            new_weight = temp.clone().masked_fill(mask == 0, 0)
            self.gat1.gat_conv.lin_src_new.weight = nn.Parameter(new_weight)        
        self.gat1.gat_conv.lin_src_new.weight.mask = mask

        temp2 = self.gat1.gat_conv.att_src_new
        mask2 = generate_mask(temp2[0], [(slice(0,2), generate_slices(heads=1)[0])])
        if MASK is True:
            new_weight = temp2.clone().masked_fill(mask2 == 0, 0)
            self.gat1.gat_conv.att_src_new = nn.Parameter(new_weight)
        self.gat1.gat_conv.att_src_new.mask = mask2.unsqueeze(0)   
        
        temp2_ = self.gat1.gat_conv.att_dst_new
        mask2_ = generate_mask(temp2_[0], [(slice(0,2), generate_slices(heads=1)[0])])
        if MASK is True:
            new_weight = temp2_.clone().masked_fill(mask2_ == 0, 0)
            self.gat1.gat_conv.att_dst_new = nn.Parameter(new_weight)
        self.gat1.gat_conv.att_dst_new.mask = mask2_.unsqueeze(0)   

        temp3 = self.gat2.gat_conv.lin_src_new.weight
        mask3 = generate_mask(temp3, [(slice(0,config['hidden_features']), slice(config['hidden_features'], 2*config['hidden_features'])),
                                      (slice(0,config['hidden_features']), slice(3*config['hidden_features'], 4*config['hidden_features'])),
                                      (slice(2*config['hidden_features'], 3*config['hidden_features']), slice(config['hidden_features'], 2*config['hidden_features'])),
                                      (slice(2*config['hidden_features'], 3*config['hidden_features']), slice(3*config['hidden_features'], 4*config['hidden_features'])),
                                      (slice(config['hidden_features'], 2*config['hidden_features']), slice(0,temp3.shape[1])),
                                      (slice(3*config['hidden_features'],4*config['hidden_features']), slice(0,temp3.shape[1]))])
        new_weight = temp3.clone().masked_fill(mask3 == 0, 0)
        self.gat2.gat_conv.lin_src_new.weight = nn.Parameter(new_weight)
        self.gat2.gat_conv.lin_src_new.weight.mask = mask3
        # always 0 part: mask2
        self.gat2.gat_conv.lin_src_new.weight.mask2 = generate_mask(temp3, [(slice(0,config['hidden_features']), slice(config['hidden_features'], 2*config['hidden_features'])),
                                      (slice(0,config['hidden_features']), slice(3*config['hidden_features'], 4*config['hidden_features'])),
                                      (slice(2*config['hidden_features'], 3*config['hidden_features']), slice(config['hidden_features'], 2*config['hidden_features'])),
                                      (slice(2*config['hidden_features'], 3*config['hidden_features']), slice(3*config['hidden_features'], 4*config['hidden_features']))])
        
        temp4 = self.gat2.gat_conv.att_src_new
        mask4 = generate_mask(temp4[0], [(slice(0,2), generate_slices(heads=1)[0])])
        if MASK is True:
            new_weight = temp4.clone().masked_fill(mask4 == 0, 0)
            self.gat2.gat_conv.att_src_new = nn.Parameter(new_weight)
        self.gat2.gat_conv.att_src_new.mask = mask4.unsqueeze(0)   
        
        temp4_ = self.gat2.gat_conv.att_dst_new
        mask4_ = generate_mask(temp4_[0], [(slice(0,2), generate_slices(heads=1)[0])])
        if MASK is True:
            new_weight = temp4_.clone().masked_fill(mask4_==0, 0)
            self.gat2.gat_conv.att_dst_new = nn.Parameter(new_weight)
        self.gat2.gat_conv.att_dst_new.mask = mask4_.unsqueeze(0)   

        temp5 = self.linear.weight
        mask5 = generate_mask(temp5, [(slice(config['out_dim'], config['all_dim']), slice(0,config['K'] * config['hidden_features2'])), (slice(0,config['out_dim']), slice(config['hidden_features'], 2*config['hidden_features'])), (slice(0,config['out_dim']), slice(3*config['hidden_features'], 4*config['hidden_features']))])
        if MASK is True:
            new_weight= temp5.clone().masked_fill(mask5==0, 0)
            self.linear.weight = nn.Parameter(new_weight)
        self.linear.weight.mask = mask5 
        print(mask5.shape)
        self.linear.weight.mask2= generate_mask(temp5, [(slice(0,config['out_dim']), slice(config['hidden_features'], 2*config['hidden_features'])), (slice(0,config['out_dim']), slice(3*config['hidden_features'], 4*config['hidden_features']))])   
    

    def reset_parameters_part(self):
        for name, param in self.named_parameters():
            print(name)
            print(param)
            if hasattr(param, 'mask'):
                param_copy = param.clone()
                updated_param = torch.nn.init.xavier_uniform_(torch.empty_like(param))
                updated_param[param.mask == 1] = param_copy[param.mask == 1]
                print(updated_param)
                exec_str = f"self.{name} = nn.Parameter(updated_param)"
                exec(exec_str)
                param.requires_grad = True
                
            if hasattr(param, "mask2"):
                param_copy = param.clone()
                new_weight = param_copy.clone().masked_fill(param.mask2 == 0, 0)
                exec_str = f"self.{name} = nn.Parameter(new_weight)"
                exec(exec_str)
                
                       

    def forward(self, x, edge_index, edge_attention):
        x1, attention1, rank_attention1 = self.gat1(x, edge_index, edge_attention) 
        x2, attention2, rank_attention2 = self.gat2(x1, edge_index, edge_attention)
        x_out = self.linear(x2)
        x_out = x_out / torch.norm(x_out, dim=1, keepdim=True)
        return x_out, attention1, attention2, rank_attention1, rank_attention2


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads, concat, dropout):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_features, out_features, heads, concat=concat)
        self.batch_norm = nn.BatchNorm1d(out_features * heads)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()


    def forward(self, x, edge_index, edge_attention):
        # edge_index is now edges; edge_attention is GAT4 score pairs
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out, attention_weights = self.gat_conv(x, edge_index, return_attention_weights=True)
        rank_attention = attention_weights[1]

        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.activation(out)

        if CHECK_ALL:
            # print(':::::::ATTENTION EXAMPLE:::::::')
            now_edges = attention_weights[0][:,np.where(edge_index[0,:].cpu().detach() == 2771)[0]] 
            temp = find_mark_of_now_edges(now_edges, edge_attention, true_rank)
            index = np.where(temp != -1)[0]
            print(temp[index])
            print(attention_weights[1][np.where(edge_index[1,:].cpu().detach() == 2771)[0],:][index,0])
            print(f"SPEARMAN = {compute_spearman(temp[index], attention_weights[1][np.where(edge_index[1,:].cpu().detach() == 2771)[0],:][index,0])}")
        
        return out, attention_weights, rank_attention

        

class GATConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        fill_value: Union[float, Tensor, str] = 'mean',
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.fill_value = fill_value
        
        self.lin_src_new = nn.Linear(in_channels, heads * out_channels , bias=False)
        self.att_src_new = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst_new = Parameter(torch.empty(1, heads, out_channels))
        self.reset_parameters()

        
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src_new.reset_parameters()
        glorot(self.att_src_new)
        glorot(self.att_dst_new)


    def forward(self, x: Union[Tensor, OptPairTensor], 
                edge_index: Adj, size: Size = None,
                return_attention_weights=None):
        
        # IF BEGIN_place==0, then this is the first gat layer; 
        # ELSE, this is the second gat layer;
        ## BEGIN_place is the length for one head original embedding. Default:config['hidden_features']
        
        H, C = self.heads, self.out_channels
        
        x_src_new = x_dst_new = self.lin_src_new(x).view(-1, H, C)
        x_new = (x_src_new, x_dst_new)
        alpha_src_new = (x_src_new * self.att_src_new).sum(dim=-1)
        alpha_dst_new = (x_dst_new * self.att_dst_new).sum(dim=-1)
        alpha_new = (alpha_src_new, alpha_dst_new)
        alpha_new = self.edge_updater(edge_index, alpha=alpha_new, edge_attr=None)
        out_new = self.propagate(edge_index, x=x_new, alpha=alpha_new, size=size)
        
        if self.concat:
            out = out_new.view(-1, self.heads * self.out_channels)

        else:
            out = out_new.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            return out, (edge_index, alpha_new)
        else:
            return out


    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

    
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
    def __init__(self):
        super(ExtendedGAT, self).__init__()
        if config['ATTENTION_TYPE'] == "A":
            self.origin_model = GAT_A(in_features=config['num_features'], hidden_features=config['hidden_features'], out_features=config['out_dim'], K=config['K'])
        else:
            self.origin_model = GATModel(config['num_features'], config['hidden_features'])
        self.sparseLayer = SparseLayer(config['num_nodes'], config['num_latent'], mask, origin_weight)

    def forward(self, x, edge_index, FROZEN = config['FROZEN']):
        if FROZEN is False:
            x, attention1, attention2 = self.origin_model(x, edge_index)
        x_new = self.sparseLayer(x)
        if FROZEN is False:
            return x, x_new, attention1, attention2
        return x_new

    
'''
Loss Functions
'''
def custom_loss(my_objects, x, now_index, device, name_all, edge_attention, 
                rank_attention1, rank_attention2, true_rank, COS_origin_sap=None, 
                COS_origin_svd=None, scale_one_one=config['scale_one_one'], 
                scale_hie=config['scale_hie'], scale_OTOL=config['scale_OTOL'], 
                scale_LTOL=config['scale_LTOL'], scale_REL=config['scale_REL'], 
                scale_SIM_NO_HIE=config['scale_SIM_NO_HIE'], 
                scale_LSAP=config['scale_LSAP'], scale_LSVD=config['scale_LSVD'], 
                scale_attention=config['scale_attention'], ORIGIN=True, ONLY_SIMI=False):  
    '''
    Return 8(9) parts loss
    1. one-one loss
    2. hierarchy loss
    3. Other lab to Loinc Loss
    4. Local lab to Loinc Loss
    5. Related Pairs Loss
    6. SIM_NO_HIE Loss
    7. Sapbert Embedding Loss
    8. SVD PPMI Embedding Loss
    9. （select） Knowledge Guided Attention    
    '''

    def loss_term(temp_objects, x, now_index, AA=config['AA'], BB=config['BB'], lambd=config['lambd'], rmax=config['rmax']):
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
        x1 = x[:,0:rmax]
        loss1 = (1 / AA) * sum(torch.log(1 + (1 / len(temp_objects[i].sampled_set1)) * sum(torch.exp(- AA * (torch.dot(x1[i, :], x1[j, :]) - lambd)) for j in temp_objects[i].sampled_set1)) for i in range(len(now_index)) if len(temp_objects[i].sampled_set1) > 0)
        loss2 = (1 / BB) * sum(torch.log(1 + (1 / len(temp_objects[i].sampled_set2)) * sum(torch.exp(BB * (torch.dot(x1[i, :], x1[j, :]) - lambd)) 
                                                                                  for j in temp_objects[i].sampled_set2)) for i in range(len(now_index)) if len(temp_objects[i].sampled_set2) > 0)
        return loss1, loss2


    def calculate_loss(loss_type, my_objects, x, now_index, name_all, scale, DIFF=True,  AA=config['AA'], BB=config['BB'], lambd=config['lambd'], ORIGIN=True, rmax=config["rmax"]):

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
            rmax = config['out_dim']

        elif loss_type == 'similar_no_hie_pairs':
            if CHECK_ALL:
                print('Similar Pairs (Not Hierarchy) Loss:')
            set1 = [my_objects[i].sim_no_hie for i in now_index]
            set2 = [find_same_type(i, name_all, INST=ORIGIN) for i in now_index] # within inst     
            
        else:
            raise ValueError("Invalid loss_type. Supported values are 'one_one','hierarchy','other_to_loinc','local_to_loinc', 'related_pairs','similar_no_hie_pairs'.")

        temp_objects = origin_term_temp(name_temp=get_values(name_all[now_index]), set1=set1, set2=set2, DIFF=DIFF)

        result = loss_term(temp_objects, x, now_index, AA=AA, BB=BB, lambd=lambd, rmax=rmax)
        result = [scale * r for r in result]
        P_LOSS = result[0]
        N_LOSS = result[1]
        
        if CHECK_ALL:
            print("Positive_Loss_{} = {:.4f}".format(loss_type, P_LOSS))
            print("Negative_Loss_{} = {:.4f}".format(loss_type, N_LOSS))
        return P_LOSS, N_LOSS


    def calculate_loss2(COS_origin, emb_now, now_index, scale):
        '''
        calculate sapbert/SPPMI embedding cosine similarity loss
        '''
        now_index_tensor = torch.tensor(now_index).to(device) 
        COS_origin1 = torch.index_select(COS_origin, 0, now_index_tensor)
        COS_origin2 = torch.index_select(COS_origin1, 1, now_index_tensor)
        COS_origin2 = COS_origin2.to(device)

        x1_now = torch.index_select(emb_now, 0, now_index_tensor)                
        COS_now = torch.matmul(x1_now, x1_now.transpose(0, 1))
        COS_now = COS_now.to(device)
        pos_loss1 = scale *  torch.norm(COS_now - COS_origin2, p="fro")
        
        if CHECK_ALL:
            print("pos_loss1 = {:.4f}".format(pos_loss1))  

        # sample another len(now_index) rows to compute the cosine similarity 
        sample_range = set(range(config['num_nodes']))- set(now_index)
        now_index2 = np.random.choice(list(sample_range), len(now_index), replace=False)
        now_index2_tensor = torch.tensor(now_index2).to(device)

        COS_origin3 = torch.index_select(COS_origin, 0, now_index2_tensor)
        COS_origin4 = torch.index_select(COS_origin3, 1, now_index2_tensor)
        x2_now = torch.index_select(x, 0, now_index2_tensor)

        COS_now2 = torch.matmul(x2_now, x2_now.transpose(0, 1))
        COS_now2 = COS_now2.to(device)
        COS_origin4 = COS_origin4.to(device)
        pos_loss2 = scale_LSAP * torch.norm(COS_now2 - COS_origin4, p="fro")
        if CHECK_ALL:
            print("pos_loss2 = {:.4f}".format(pos_loss2))
        return pos_loss1, pos_loss2     
    
                             
    # def calculate_loss3(edge_attention, rank_attention1, rank_attention2, true_rank, scale):
    #     '''
    #     calculate embedding similarity-rank loss
    #     '''
    #     loss= pairwise_loss(edge_attention, rank_attention1.t(), rank_attention2.t(), true_rank)
    #     if CHECK_ALL:
    #         print('loss_rank = {:4f}'.format(loss))
    #     return scale * loss
    

    # Call the function to compute first 5 types of losses
    if ORIGIN:
        P_LOSS_one_one, N_LOSS_one_one = calculate_loss('one_one', my_objects, x, now_index, name_all, scale_one_one)
        
    P_LOSS_hie, N_LOSS_hie = calculate_loss('hierarchy', my_objects, x, now_index, name_all, scale_hie,ORIGIN=ORIGIN)
    P_LOSS_OTOL, N_LOSS_OTOL = calculate_loss('other_to_loinc', my_objects, x, now_index, name_all, scale_OTOL, DIFF=False)
    P_LOSS_LTOL, N_LOSS_LTOL = calculate_loss('local_to_loinc', my_objects, x, now_index, name_all, scale_LTOL, DIFF=False)
    if ONLY_SIMI is False:
        P_LOSS_REL, N_LOSS_REL = calculate_loss('related_pairs', my_objects, x, now_index, name_all, scale_REL, ORIGIN=ORIGIN, rmax=config['all_dim'])
    P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE = calculate_loss('similar_no_hie_pairs', my_objects, x, now_index, name_all, scale_SIM_NO_HIE, ORIGIN=ORIGIN)                           

    # Call the function to compute next 2 types of losses
    
    if CHECK_ALL:
        print('Sapbert embedding Cosine Loss:')
    
    if ORIGIN:
        P_LOSS_SAP1, P_LOSS_SAP2 = calculate_loss2(COS_origin_sap, x, now_index, scale_LSAP)
    
    if CHECK_ALL:
        print('SVD embedding Cosine Loss:')
        
    if ORIGIN:    
        P_LOSS_SVD1, P_LOSS_SVD2 = calculate_loss2(COS_origin_svd, x, now_index, scale_LSVD)
    
    # # Call the function to compute similarity-rank lossed
    # LOSS_attention = calculate_loss3(edge_attention, rank_attention1, rank_attention2, true_rank, scale_attention)
    LOSS_attention = 0
    
    if ORIGIN:
        if ONLY_SIMI:
            return P_LOSS_one_one, N_LOSS_one_one, P_LOSS_hie, N_LOSS_hie, P_LOSS_OTOL, N_LOSS_OTOL, P_LOSS_LTOL, N_LOSS_LTOL, P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE, P_LOSS_SAP1, P_LOSS_SAP2, P_LOSS_SVD1, P_LOSS_SVD2, LOSS_attention
        else:
            return P_LOSS_one_one, N_LOSS_one_one, P_LOSS_hie, N_LOSS_hie, P_LOSS_OTOL, N_LOSS_OTOL, P_LOSS_LTOL, N_LOSS_LTOL, P_LOSS_REL, N_LOSS_REL, P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE, P_LOSS_SAP1, P_LOSS_SAP2, P_LOSS_SVD1, P_LOSS_SVD2, LOSS_attention
    
    return P_LOSS_hie, N_LOSS_hie, P_LOSS_OTOL, N_LOSS_OTOL, P_LOSS_LTOL, N_LOSS_LTOL, P_LOSS_REL, N_LOSS_REL, P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE