import numpy as np


config = {
    'num_nodes': 24682, 
    'num_features': 1536,
    # 'num_features': 768,
    'num_latent': 11609,
    'MGB_index': np.arange(0, 6917), 
    'VA_index': np.arange(6917, 13538), 
    'UP_index': np.arange(13538, 24682), 
    'hidden_features': 1024, 
    'hidden_features2': 2048,
    'res': False,
    
    'epochs': 1, 
    'base_lr':  1e-3, 
    'batch_size': 128, 
    'gamma': 0.5, 
    
    'scale_hie':1, 
    'scale_OTOL':50, 
    'scale_LTOL':5, 
    'scale_REL': 5, 
    'scale_LSAP': 1, 
    'scale_LSVD' : 1, 
    'scale_SIM_NO_HIE': 1,
    'scale_attention':1e-3,
    
    'SEED': 1, 
    'rmax': 256, 
    'latent': False,
    'K': 2,
    'out_dim': 256, 
    'all_dim': 768,
    'drop_p': 0.1,
    'CHECK_ALL': False,
    'truncate_mask': -100,
    'ATTENTION_TYPE' : None,
    'has_origin_model': None,
    'origin_start_time': None,
    
    'want_TOP1' : None,
    'want_TOP20' : None,
    'AA' : None,
    'BB' : None,
    'lambd' : None,
    'scale_one_one' : None,
    'low_dim' : None,
    'Truncate' : None,
    'path_origin' : True,
    'FROZEN' : None,
    'epochs' : None
}  

def set_config(new_config):
    global config
    config = new_config

def get_config():
    return config


