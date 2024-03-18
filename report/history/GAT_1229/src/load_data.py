# pylint: skip-file

"""
Title: load_data.py
Author: Han Tong
Date: 2023-12-29
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

from config import get_config
config = get_config()
os.chdir(f"{config['path']}/src")
import sys
warnings.filterwarnings('ignore')
import torch_geometric as tg
from torch.utils.data import DataLoader
from utils import *
from data_structure import *

# the original embedding we need
mgb_emb, va_emb, upmc_emb = torch.load(f'/home/doz128/GAME/input/inst_emb.pth')

# the name of latent nodes
unique_name = pd.read_csv(f'{config["input_dir"]}/name_desc/unique_name_desc.csv')
unique_name = unique_name.iloc[:,0].values

# the hierarchy of loinc, rxnorm and phecode we need.
hie_loinc_rxn_phe = pd.read_csv(f'{config["input_dir"]}/Hierarchy/hie_train_9_17.csv')

# Other Lab map to LOINC (gpt4)
P_OTOL = pd.read_csv(f'{config["input_dir"]}/Other&Local-Loinc/OtherToLoinc_Positive.csv')
N_OTOL = pd.read_csv(f'{config["input_dir"]}/Other&Local-Loinc/OtherToLoinc_Negative.csv')

# Local Lab map to LOINC (gpt4)
P_LTOL = pd.read_csv(f'{config["input_dir"]}/Other&Local-Loinc/LocalToLoinc_Positive.csv')
N_LTOL = pd.read_csv(f'{config["input_dir"]}/Other&Local-Loinc/LocalToLoinc_Negative.csv')


# Related pairs we have
REL_pairs = pd.read_csv(f'{config["input_dir"]}/similar_related_pairs/related_pairs_1218.csv')

# no-hie similar pairs we have
SIM_no_hie_pairs = pd.read_csv(f'{config["input_dir"]}/similar_related_pairs/similar_pairs_no_hie_9_17.csv')

# train hie-similar pairs we have
val_sim_pairs = pd.read_csv(f'{config["input_dir"]}/similar_related_pairs/similar_pairs_hie_val_9_18.csv')

# # initialize the train, validation and test set of related pairs
# train_rel_pairs, val_rel_pairs, test_rel_pairs = split_train_set(unique_name, REL_pairs=REL_pairs, scale=[0.5,0.3])
# with open(f"{config['path']}/input/rel_pairs_1229.pkl", 'wb') as f:
#     pickle.dump([train_rel_pairs, val_rel_pairs, test_rel_pairs], f)
with open(f"{config['path']}/input/rel_pairs_1229.pkl", 'rb') as f:
    train_rel_pairs, val_rel_pairs, test_rel_pairs = pickle.load(f)

# # initialize the train, validation and test set of similar no hie pairs
# train_sim_no_hie_pairs, val_sim_no_hie_pairs, test_sim_no_hie_pairs = split_train_set(unique_name, REL_pairs=SIM_no_hie_pairs, scale=[0.5,0.3])
# with open(f"/home/doz128/GAME/input/sim_no_hie_pairs_1229.pkl", 'wb') as f:
#     pickle.dump([train_sim_no_hie_pairs, val_sim_no_hie_pairs, test_sim_no_hie_pairs], f)
train_sim_no_hie_pairs, val_sim_no_hie_pairs, test_sim_no_hie_pairs = np.load(f"{config['path']}/input/sim_no_hie_pairs_1229.pkl", allow_pickle=True)

edges = np.load(f"{config['path']}/input/edges_1229.npy", allow_pickle=True)
edges_rel = np.load(f"{config['path']}/input/edges_rel_1229.npy", allow_pickle=True)
edges_sim = np.load(f"{config['path']}/input/edges_sim_1229.npy", allow_pickle=True)

edges_sppmi = np.load(f"{config['path']}/input/edges_sppmi_1229.npy", allow_pickle=True)
pos_sppmi = np.load(f"{config['path']}/input/pos_sppmi_1229.npy", allow_pickle=True)
neg_sppmi = np.load(f"{config['path']}/input/neg_sppmi_1229.npy", allow_pickle=True)
ALL_sim_val_pairs =  pd.concat([val_sim_no_hie_pairs, val_sim_pairs], ignore_index=True)

rel_index = get_index(train_rel_pairs, unique_name)
sim_no_hie_index = get_index(train_sim_no_hie_pairs, unique_name)

# # generate my_objects using hie_train
# # Origin_term can take half an hour. We can load my_objects that have been generated
# my_objects_new = origin_loss_set(unique_name, P_OTOL, N_OTOL, P_LTOL, N_LTOL, hie_loinc_rxn_phe, 
#                           train_rel_pairs, train_sim_no_hie_pairs, rel_index, sim_no_hie_index)

# np.save(f"{config['path']}/input/my_objects_1229.npy", my_objects_new)

my_objects = np.load(f"{config['path']}/input/my_objects_1229.npy", allow_pickle=True)