""""
Title: config.py
Author: Han Tong
Date: 2023-12-29
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
    'num_features': 1536,
    'num_union': 11607,
    'hidden_features': 1024, 
    'epochs': 3, 
    'base_lr':  1e-3, 
    'batch_size': 256, 
    'gamma': 0.8, 
    'scale_hie':1, 
    'scale_OTOL':50, 
    'scale_REL': 5, 
    'scale_SIM_NO_HIE': 1,
    'scale_sppmi':1e-2,
    'SEED': 1, 
    'rmax': 256, 
    'K': 2,
    'out_dim': 768, 
    'drop_p': 0.0,
    'CHECK_ALL': False,
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
    'path': '/home/doz128/GAME', 
    'input_dir': '/n/data1/hsph/biostat/celehs/lab/doz128/GAME/input'
}  