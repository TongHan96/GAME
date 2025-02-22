""""
Title: config.py
Author: Han Tong
Date: 2025-02-21
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
    'drop_p': 0.1,
    'base_lr':  1e-4, 
    'AA' : 1.0,
    'BB' : 5.0,
    'lambd' : 0.5,
    'scale_one_one' : 10,
    'scale_hie': 1, 
    'scale_sppmi': 0.1,
    'scale_OTOL':30, 
    'scale_REL': 5, 
    'scale_align': 1,
    'rmax': 256,
    'hidden_features': 768, 
    'Decoder': False,
    'Decoder_inst': None,
    'path': '/home/doz128/GAME', # chg to wk path
    'input_dir': '/n/data1/hsph/biostat/celehs/lab/Han/GAME_final/input', # chg to file path
    'path_origin' : None,
    'epochs': 500, 
    'CHECK_ALL': False,
    'DEVICE': 'cuda:0',
    'num_inst': 7,
    'api_key': None,
    
    'out_dim': 768, 
    'num_features': 768,
    'num_union': 50738, 
    'batch_size': 256, 
    'gamma': 0.99, 
    'scale_SIM_NO_HIE': 10,
    'SEED': 1, 
    'heads': 2, 
}  
