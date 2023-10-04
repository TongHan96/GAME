""""
Title: config.py
Author: Han Tong
Date: 2023-10-04
Python Version: Python 3.11.3
Description: All parameter we need
"""

import numpy as np

def set_config(new_config):
    global config
    config = new_config

def get_config():
    return config

config = {
    'num_nodes': 24682, 
    'num_features': 1536,
    'num_latent': 11609,
    'MGB_index': np.arange(0, 6917), 
    'VA_index': np.arange(6917, 13538), 
    'UP_index': np.arange(13538, 24682), 
    'hidden_features': 1024, 
    'epochs': 3, 
    'base_lr':  1e-3, 
    'batch_size': 256, 
    'gamma': 0.8, 
    'scale_hie':1, 
    'scale_OTOL':50, 
    'scale_LTOL':5, 
    'scale_REL': 5, 
    'scale_LSAP': 1e-2, 
    'scale_LSVD' : 1e-2, 
    'scale_SIM_NO_HIE': 1,
    'scale_attention':1e-3,
    'SEED': 1, 
    'res': False,
    'rmax': 256, 
    'latent': False,
    'K': 2,
    'out_dim': 768, 
    'drop_p': 0.0,
    'CHECK_ALL': False,
    'truncate_mask': -100,
    'ATTENTION_TYPE' : 'Naive',
    'has_origin_model': False,
    'want_TOP1' : None,
    'want_TOP20' : None,
    'AA' : 1,
    'BB' : 5,
    'epochs':3,
    'lambd' : 0.5,
    'scale_one_one' : 10,
    'low_dim' : None,
    'Truncate' : None,
    'path_origin' : None,
    'path': '/root/current_code/GAT_model_10_4', 
    'FROZEN' : None
}  