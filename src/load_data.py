# pylint: skip-file

"""
Title: load_data.py
Author: Han Tong
Date: 2024-04-07
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
import pickle
import warnings
import torch
import re
import os

from config import set_config, get_config
config = get_config()
os.chdir(f'{config["path"]}/src')

import sys
warnings.filterwarnings('ignore')
import torch_geometric as tg
from torch.utils.data import DataLoader
from utils import *
from data_structure import *

# load the inst row index
data = np.load(f'{config["input_dir"]}/name_desc/inst_row.npz')
keys = data.files  # This will give you a list of all keys in the .npz file
config['inst_row'] = [data[key] for key in keys]

# the original embedding we need
sppmi_list = torch.load(f'{config["input_dir"]}/emb/inst_emb.pth')
sap_emb = torch.load(f'{config["input_dir"]}/emb/sap_emb.pth')
coder_emb = torch.load(f'{config["input_dir"]}/emb/coder_emb.pth')

# the name of latent nodes
name_all = pd.read_csv(f'{config["input_dir"]}/name_desc/name_desc_all_GPT4.csv')
name_all = name_all.iloc[:,0].values
unique_name = pd.read_csv(f'{config["input_dir"]}/name_desc/unique_name_desc.csv')
unique_name = unique_name.iloc[:,0].values

# the hierarchy of loinc, rxnorm and phecode we need.
hie_loinc_rxn_phe = pd.read_csv(f'{config["input_dir"]}/Hierarchy/hie_train_0127.csv')

# Other Lab map to LOINC (gpt4)
P_OTOL = pd.read_csv(f'{config["input_dir"]}/Other&Local-Loinc/OtherToLoinc_Positive.csv')
N_OTOL = pd.read_csv(f'{config["input_dir"]}/Other&Local-Loinc/OtherToLoinc_Negative.csv')

# Local Lab map to LOINC (gpt4)
P_LTOL = pd.read_csv(f'{config["input_dir"]}/Other&Local-Loinc/LocalToLoinc_Positive.csv')
N_LTOL = pd.read_csv(f'{config["input_dir"]}/Other&Local-Loinc/LocalToLoinc_Negative.csv')

# Related pairs we have
REL_pairs = pd.read_csv(f'{config["input_dir"]}/similar_related_pairs/related_pairs_0127.csv')

# no-hie similar pairs we have
SIM_no_hie_pairs = pd.read_csv(f'{config["input_dir"]}/similar_related_pairs/similar_pairs_no_hie_0127.csv')

# train hie-similar pairs we have
val_sim_pairs = pd.read_csv(f'{config["input_dir"]}/similar_related_pairs/similar_pairs_hie_val_0127.csv')

# # initialize the train, validation and test set of related pairs
# train_rel_pairs, val_rel_pairs, test_rel_pairs = split_train_set(unique_name, REL_pairs=REL_pairs, scale=[0.5,0.3])
# drug_side_pairs = pd.read_csv('/home/doz128/GAME_0122/GNN/supp_code/drug_side_detect/input/drug_side_effect_0127.csv')
# with open(f"{config['path']}/input/rel_pairs_drug_0204.pkl", 'wb') as f:
#     pickle.dump([train_rel_pairs, val_rel_pairs, test_rel_pairs, drug_side_pairs], f)
# rel_edges = np.row_stack([match(train_rel_pairs.iloc[:,0].values, unique_name), match(train_rel_pairs.iloc[:,1].values, unique_name)])
# np.save(f"{config['path']}/input/edges_rel.npy", rel_edges)

with open(f"{config['input_dir']}/similar_related_pairs/rel_pairs_drug.pkl", 'rb') as f:
    train_rel_pairs, val_rel_pairs, test_rel_pairs, drug_side_pairs = pickle.load(f)

# # initialize the train, validation and test set of similar no hie pairs
# train_sim_no_hie_pairs, val_sim_no_hie_pairs, test_sim_no_hie_pairs = split_train_set(unique_name, REL_pairs=SIM_no_hie_pairs, scale=[0.5,0.3])
# with open(f"{config['path']}/input/sim_no_hie_pairs_0127.pkl", 'wb') as f:
#     pickle.dump([train_sim_no_hie_pairs, val_sim_no_hie_pairs, test_sim_no_hie_pairs], f)
# sim_edges = np.row_stack([match(train_sim_no_hie_pairs.iloc[:,0].values, unique_name), match(train_sim_no_hie_pairs.iloc[:,1].values, unique_name)])
# np.save(f"{config['path']}/input/edges_sim_no_hie.npy", sim_edges)

train_sim_no_hie_pairs, val_sim_no_hie_pairs, test_sim_no_hie_pairs = np.load(f"{config['input_dir']}/similar_related_pairs/sim_no_hie_pairs.pkl", allow_pickle=True)

edges = torch.tensor(np.load(f"{config['input_dir']}/edges/edges.npy", allow_pickle=True))
same_desc_edge = torch.tensor(np.load(f"{config['input_dir']}/edges/edges_same_desc.npy", allow_pickle=True))

edges_rel = torch.tensor(np.load(f"{config['input_dir']}/edges/edges_rel.npy", allow_pickle=True))
edges_sim = torch.tensor(np.load(f"{config['input_dir']}/edges/edges_sim_no_hie.npy", allow_pickle=True))

edges_sppmi = torch.tensor(np.load(f"{config['input_dir']}/edges/edges_sppmi.npy", allow_pickle=True))
pos_sppmi = torch.tensor(np.load(f"{config['input_dir']}/edges/pos_sppmi.npy", allow_pickle=True))
neg_sppmi = torch.tensor(np.load(f"{config['input_dir']}/edges/neg_sppmi.npy", allow_pickle=True))

all_pos_sppmi = torch.cat([edges_sppmi, pos_sppmi], dim=1)
ALL_sim_val_pairs =  pd.concat([val_sim_no_hie_pairs, val_sim_pairs], ignore_index=True)

rel_index = get_index(train_rel_pairs, unique_name)
sim_no_hie_index = get_index(train_sim_no_hie_pairs, unique_name)

# # generate my_objects using hie_train
# # Origin_term can take half an hour. We can load my_objects that have been generated
# my_objects_new = origin_loss_set(unique_name, P_OTOL, N_OTOL, P_LTOL, N_LTOL, hie_loinc_rxn_phe, 
#                           train_rel_pairs, train_sim_no_hie_pairs, rel_index, sim_no_hie_index)

# np.save(f"{config['path']}/input/my_objects_0130.npy", my_objects_new)

my_objects = np.load(f"{config['input_dir']}/edges/my_objects.npy", allow_pickle=True)