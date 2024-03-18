# pylint: skip-file

"""
Title: load_data.py
Author: Han Tong
Date: 2023-09-02
Python Version: Python 3.11.3
Description: Load all data we need in this file

"""

# !pip install torch_geometric
# !pip install tensorflow
# !pip install keras
# !pip install zipfile
# !pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html

import numpy as np
import pandas as pd
import warnings
import torch
import re
import os

os.chdir('/root/current_code/GAT_model_9_1/src')

from config import get_config
import sys
warnings.filterwarnings('ignore')
import torch_geometric as tg
from torch.utils.data import DataLoader
from utils import split_set, weighted_sum_x, get_latent_related, get_parent
from data_structure import *

config = get_config()

# the original embedding we need
datax = pd.read_csv("https://han-attention.s3.amazonaws.com/input/emb/emb_Z0_825.csv")

typeofZ = datax.iloc[:, 0]
index1 = [i for i, x in enumerate(typeofZ) if re.search("LOINC", x)]  # loinc
index2 = [i for i, x in enumerate(typeofZ) if re.search("RXNORM", x)]  # RxNorm
index3 = [i for i, x in enumerate(typeofZ) if re.search("PheCode", x)]  # Phecode
index_other = [i for i, x in enumerate(typeofZ) if re.search("Other lab", x)]  # Other lab
index = np.hstack((index_other, index1))

# the name and descriptions we need
name_all = datax.iloc[:, 0]
name_MGB = name_all.iloc[config['MGB_index']] # MGB codes
name_VA = name_all.iloc[config['VA_index']] # VA codes
name_UP = name_all.iloc[config['UP_index']] # UPMC codes

dict_MGB = name_all[config['MGB_index']]
dict_VA = name_all[config['VA_index']]
dict_UP = name_all[config['UP_index']]

# the name of latent nodes
unique_name = pd.read_csv('https://han-attention.s3.amazonaws.com/input/name_desc/unique_name.csv')
unique_name = unique_name['x'].values
name_new = unique_name[np.setdiff1d(np.arange(len(unique_name)), [i for i, x in enumerate(unique_name) if re.search("^LOINC:(?!LP)", x)])]

# the frequency we need
freq_all = pd.read_csv('https://han-attention.s3.amazonaws.com/input/latent_node/freq_all.csv')
freq_all = freq_all.iloc[:,1]

# the hierarchy of loinc, rxnorm and phecode we need.
hie_loinc_rxn_phe = pd.read_csv('https://han-attention.s3.amazonaws.com/input/Hierarchy/hie_loinc_rxn_phe_9_2.csv')

# Other Lab map to LOINC (gpt4)
P_OTOL = pd.read_csv('https://han-attention.s3.amazonaws.com/input/Other%26Local-Loinc/OtherToLoinc_Positive.csv')
N_OTOL = pd.read_csv('https://han-attention.s3.amazonaws.com/input/Other%26Local-Loinc/OtherToLoinc_Negative.csv')

# Local Lab map to LOINC (gpt4)
P_LTOL = pd.read_csv('https://han-attention.s3.amazonaws.com/input/Other%26Local-Loinc/LocalToLoinc_Positive.csv')
N_LTOL = pd.read_csv('https://han-attention.s3.amazonaws.com/input/Other%26Local-Loinc/LocalToLoinc_Negative.csv')

if config['latent']:

    P_OTOL_LP = get_parent(P_OTOL, list(np.arange(1,P_OTOL.shape[1])), DROP=False)
    N_OTOL_LP = get_parent(N_OTOL, list(np.arange(1,N_OTOL.shape[1])), DROP=False)
    P_LTOL_LP = get_parent(P_LTOL, list(np.arange(1,P_LTOL.shape[1])), DROP=False)
    N_LTOL_LP = get_parent(N_LTOL, list(np.arange(1,N_LTOL.shape[1])), DROP=False)

# Related pairs we have
REL_pairs = pd.read_csv('https://han-attention.s3.amazonaws.com/input/similar_related_pairs/related_pairs_0902.csv')

# no-hie similar pairs we have
SIM_no_hie_pairs = pd.read_csv('https://han-attention.s3.amazonaws.com/input/similar_related_pairs/similar_pairs_no_hie_9_2.csv')

# train hie-similar pairs we have
val_sim_pairs = pd.read_csv('https://han-attention.s3.amazonaws.com/input/similar_related_pairs/similar_pairs_hie_val_9_2.csv')

# initialize the train, validation and test set of related pairs
# train_rel_pairs, val_rel_pairs, test_rel_pairs = split_set(dict_MGB=dict_MGB, dict_VA=dict_VA, dict_UP=dict_UP, REL_pairs=REL_pairs, scale=[0.5,0.3], ADJ_ONLY=False)
# np.save('/root/current_code/GAT_model_9_1/input/rel_pairs.npy', [train_rel_pairs, val_rel_pairs, test_rel_pairs])
train_rel_pairs,  val_rel_pairs, test_rel_pairs = np.load('/root/current_code/GAT_model_9_1/input/rel_pairs.npy', allow_pickle=True)

if config['latent']:

    train_rel_pairs_LP, val_rel_pairs_LP, test_rel_pairs_LP = get_latent_related(train_rel_pairs, val_rel_pairs, test_rel_pairs)

# initialize the train, validation and test set of similar no hie pairs
# train_sim_no_hie_pairs, val_sim_no_hie_pairs, test_sim_no_hie_pairs = split_set(dict_MGB=dict_MGB, dict_VA=dict_VA, dict_UP=dict_UP, REL_pairs=SIM_no_hie_pairs, scale=[0.5,0.3], ADJ_ONLY=False)
# np.save('/root/current_code/GAT_model_9_1/input/sim_no_hie_pairs.npy', [train_sim_no_hie_pairs, val_sim_no_hie_pairs, test_sim_no_hie_pairs])
train_sim_no_hie_pairs, val_sim_no_hie_pairs, test_sim_no_hie_pairs = np.load('/root/current_code/GAT_model_9_1/input/sim_no_hie_pairs.npy', allow_pickle=True)

if config['latent']:

    train_sim_no_hie_pairs_LP, val_sim_no_hie_pairs_LP, test_sim_no_hie_pairs_LP = get_latent_related(train_sim_no_hie_pairs,  val_sim_no_hie_pairs, test_sim_no_hie_pairs)

# edges = pd.read_csv('https://han-attention.s3.amazonaws.com/input/adj/adj_train_9_2.csv')  # (63413,2)
# edges['Var1'] = edges['Var1'] - 1
# edges['Var2'] = edges['Var2'] - 1
# edges = torch.tensor(np.array(edges))
# np.save('/root/current_code/GAT_model_9_1/input/edges.npy', np.transpose(edges))

# edges_rel = split_set(dict_MGB=dict_MGB, dict_VA=dict_VA, dict_UP=dict_UP, combine=False, train_pairs=train_rel_pairs) # (75990,2) 

# edges_sim = split_set(dict_MGB=dict_MGB, dict_VA=dict_VA, dict_UP=dict_UP, combine=False, train_pairs=train_sim_no_hie_pairs) # (132555,2)

# np.save('/root/current_code/GAT_model_9_1/input/edges_rel.npy', edges_rel.transpose())
# np.save('/root/current_code/GAT_model_9_1/input/edges_sim.npy', edges_sim.transpose())

# np.save('/root/current_code/GAT_model_9_1/input/edges.npy', edges)
edges = np.load('/root/current_code/GAT_model_9_1/input/edges.npy', allow_pickle=True)
edges_rel = np.load('/root/current_code/GAT_model_9_1/input/edges_rel.npy', allow_pickle=True)
edges_sim = np.load('/root/current_code/GAT_model_9_1/input/edges_sim.npy', allow_pickle=True)

ALL_sim_val_pairs =  pd.concat([val_sim_no_hie_pairs, val_sim_pairs], ignore_index=True)

if config['latent']:

    ALL_sim_val_pairs_LP =  get_parent(ALL_sim_val_pairs, [0,1])
    rel_index_LP = get_index(train_rel_pairs_LP, name_new)
    sim_no_hie_index_LP = get_index(train_sim_no_hie_pairs_LP, name_new)

rel_index = get_index(train_rel_pairs, name_all)
sim_no_hie_index = get_index(train_sim_no_hie_pairs, name_all)

# generate my_objects using hie_train
# Origin_term can take half an hour. We can load my_objects that have been generated
# my_objects = origin_term(name_all, P_OTOL, N_OTOL, P_LTOL, N_LTOL, hie_loinc_rxn_phe, 
#                          train_rel_pairs, train_sim_no_hie_pairs, rel_index, sim_no_hie_index)
# np.save('/root/current_code/GAT_model_9_1/input/my_objects.npy', my_objects)

my_objects = np.load('/root/current_code/GAT_model_9_1/input/my_objects.npy', allow_pickle=True)
 
if config['latent']:
    # my_objects2 = origin_term(name_new, P_OTOL_LP, N_OTOL_LP, P_LTOL_LP, N_LTOL_LP, hie_loinc_rxn_phe, train_rel_pairs_LP, train_sim_no_hie_pairs_LP, rel_index_LP, sim_no_hie_index_LP, INST=False)
    # np.save('/root/current_code/GAT_model_9_1/input/my_objects2.npy', my_objects2)
    my_objects2 = np.load('/root/current_code/GAT_model_9_1/input/my_objects2.npy', allow_pickle=True)
   

# COS similarity of original sapbert embeddings and svd-PPMI embeddings
features_torch = torch.from_numpy(datax.iloc[:, 1:1537].to_numpy())#.cuda()  # runing on GPU to accelerate
COS_origin_sap = torch.mm(features_torch[:, 0:768], features_torch[:, 0:768].t())  
COS_origin_svd = torch.mm(features_torch[:, 768:1536], features_torch[:, 768:1536].t())  

# process embedding x into tensor
x_tensor = torch.tensor(datax.iloc[:, 1:1537].to_numpy(), dtype=torch.float32, requires_grad=True)
data = tg.data.Data(x=x_tensor, edge_index=torch.tensor(edges, dtype=torch.long))

mask = np.zeros((config['num_latent'], config['num_nodes']))

for i in range(config['num_latent']):
    idx = id_map([name_new[i]], name_all)
    mask[i, idx] = 1

# Convert the mask to a torch tensor
mask = torch.tensor(mask, dtype=torch.float32)
origin_weight = create_weight_matrix(name_all, name_new, freq_all)