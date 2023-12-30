# pylint: skip-file

"""
Title: main.py
Author: Han Tong
Date: 2023-12-29
Python Version: Python 3.11.3
Description: main file of our attention model
"""

import torch
from torch_geometric.utils import to_undirected
from torch_geometric.utils import add_self_loops
import warnings
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
warnings.filterwarnings('ignore')
import gc
import argparse
import random
import numpy as np   
from config import set_config
import argparse
import pdb


def update_config_from_args():
    from config import set_config, get_config
    
    config = get_config()

    parser = argparse.ArgumentParser(description="GAT Training Script")
    parser.add_argument("--want_TOP1", type=float, default=70.2,
                        help="Top1 performance threshold for storing model and embedding. Default: 70.2.")
    parser.add_argument("--drop_out", type=float, default=0.0,
                        help="Parameter drop_out prob. Default: 0.0.") 
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Parameter learning rate. Default: 1e-6.")    
    parser.add_argument("--AA", type=float, default=1.0,
                        help="Parameter AA. Default: 1.0")
    parser.add_argument("--BB", type=float, default=5.0,
                        help="Parameter BB. Default: 5.0")
    parser.add_argument("--lambd", type=float, default=0.5,
                        help="Parameter lambd. Default: 0.5.")
    parser.add_argument("--scale_one_one", type=int, default=1,
                        help="Parameter scale_one_one. Default: 1.")
    parser.add_argument("--scale_hie", type=int, default=1,
                        help="Parameter scale_hie. Default: 1.")
    parser.add_argument("--scale_sppmi", type=float, default=1e-2,
                        help="Parameter scale_sppmi. Default: 1e-2.")
    parser.add_argument("--scale_OTOL", type=int, default=50,
                        help="Parameter scale_one_one. Default: 50.")
    parser.add_argument("--scale_REL", type=int, default=5,
                        help="Parameter scale_REL. Default: 5.")   
    parser.add_argument("--rmax", type=int, default=256,
                        help="Parameter r_max we use for similarity. Default: 256.")  
    parser.add_argument("--path", type=str, default=config['path'],
                        help="Specify the path parameter.")
    parser.add_argument("--input_dir", type=str, default=config['input_dir'],
                        help="Specify the path parameter for input data.")
    parser.add_argument("--path_origin", type=str, default=config['path_origin'],
                        help='Train from the initial model and embedding path_origin is not None. Default: None.')
    parser.add_argument("--epochs", type=int, default=3,
                    help='Total Epochs. Default: 3.')
    parser.add_argument("--CHECK_ALL", type=str, default=False,
                    help='whether to check attention or not. Default: False.')
    parser.add_argument("--DEVICE", type=str, default='cuda:0',
                    help='Use GPU or CPU. Default: cuda:0.')

    args = parser.parse_args()

    config['want_TOP1'] = args.want_TOP1
    config['base_lr'] = args.lr
    config['drop_p'] = args.drop_out   
    config['AA'] = args.AA
    config['BB'] = args.BB
    config['lambd'] = args.lambd
    config['scale_one_one'] = args.scale_one_one
    config['scale_hie'] = args.scale_hie
    config['scale_sppmi'] = args.scale_sppmi
    config['scale_OTOL'] = args.scale_OTOL
    config['scale_REL'] = args.scale_REL
    config['rmax'] = args.rmax
    config['input_dir'] = args.input_dir
    config['path'] = args.path
    config['path_origin'] = args.path_origin
    config['epochs'] = args.epochs
    config['CHECK_ALL'] = args.CHECK_ALL
    config['DEVICE'] = args.DEVICE
    
    seed = config['SEED']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  
    set_config(config)
        

def main(config): 
    # load data
    from load_data import mgb_emb, va_emb, upmc_emb, unique_name, val_rel_pairs, ALL_sim_val_pairs, my_objects
    
    if config['path_origin'] is not None:
        print('load origin embedding!')
        x_sim = torch.load(f"{config['path']}/output/{config['path_origin']}/sim_emb.pth")

    # device
    device = torch.device(config['DEVICE'])
    
    mgb_emb = mgb_emb.to(device)
    va_emb =  va_emb.to(device)
    upmc_emb = upmc_emb.to(device)
    
    # load GAT model
    model_all = inst_encoder(config)
    model_all = model_all.to(device)

    optimizer0 = optim.SGD(model_all.parameters(), lr=config['base_lr'])
    # scheduler0 = CustomExponentialLR(optimizer0, gamma=config['gamma'], min_lr=1e-5)
    
    if config['path_origin'] is None:
        edge_index = torch.cat((torch.tensor(edges), torch.tensor(edges_sim)), dim=1)
    else:
        edge_index = torch.cat((torch.cat((torch.tensor(edges_rel), torch.tensor(edges_sppmi)), dim=1), torch.tensor(pos_sppmi)), dim=1).to(device)
        
    undirected_edge_index = to_undirected(edge_index).to(device)
    best_PRE_0 = -float('inf')
    
    # train part
    for epoch in range(1, config['epochs']+1):
        print('------------------------Epoch: {:03d}-----------------------'.format(epoch))
    
        now_time = time.time()
        optimizer0.zero_grad()
        
        for name, param in model_all.named_parameters():
            print('Name: ', name)
            print('Values: ', param.data)
            break
        
        if config['path_origin'] is None:
            # get new embedding
            x_sim = model_all(mgb_emb, va_emb, upmc_emb, undirected_edge_index)
            # loss function
            P_LOSS_hie, N_LOSS_hie, P_LOSS_OTOL, N_LOSS_OTOL, P_LOSS_LTOL, N_LOSS_LTOL, P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE = custom_loss(my_objects, x_sim, list(range(config['num_union'])), device, unique_name, config)
            loss0 = P_LOSS_hie + N_LOSS_hie + P_LOSS_OTOL + N_LOSS_OTOL + P_LOSS_LTOL + N_LOSS_LTOL + P_LOSS_SIM_NO_HIE + N_LOSS_SIM_NO_HIE
            loss=[my_item(P_LOSS_hie), my_item(N_LOSS_hie), 
                    my_item(P_LOSS_OTOL), my_item(N_LOSS_OTOL), 
                    my_item(P_LOSS_LTOL), my_item(N_LOSS_LTOL), 
                    my_item(P_LOSS_SIM_NO_HIE), my_item(N_LOSS_SIM_NO_HIE)]
            # evaluate
            PRE_new, AUC_new = test(x_sim, unique_name, config, similar_pairs=ALL_sim_val_pairs, PRE=True, AUC=True, AUC_type=True)
            # write
            write_file(epoch, 0, config, loss=loss, pre=PRE_new, SIM_AUC=AUC_new[0][0])
            case_store = (PRE_new[0] >= best_PRE_0) & (PRE_new[0] >= config['want_TOP1']) 
            if PRE_new[0] > best_PRE_0:
                best_PRE_0 = PRE_new[0]
            
        else:
            # get new embedding
            x_rel_part = model_all(mgb_emb, va_emb, upmc_emb, undirected_edge_index)
            x_detached = x_sim.detach()
            x_rel = torch.cat((x_detached, x_rel_part), dim=1)
            # loss function
            P_REL, N_REL = custom_loss(my_objects, x_rel, list(range(config['num_union'])), device, unique_name, config)
            loss_sppmi = config['scale_sppmi'] * sppmi_edge_loss(x_rel, pos_sppmi, neg_sppmi, config)
            loss = [my_item(P_REL), my_item(N_REL)]
            loss0 = P_REL + N_REL + loss_sppmi
            # evaluate
            AUC_new = test(x_rel, unique_name, config, related_pairs=val_rel_pairs, PRE=False, AUC=True, AUC_type=True)
            # write
            write_file(epoch, 0, config, loss=loss, REL_AUC=AUC_new[0][0][0])
            
            rel_all = sum_(AUC_new[0][0])
            print(f'AUC = {rel_all}')
            case_store = (rel_all >= best_PRE_0)
            if case_store:
                best_PRE_0 = rel_all
        
        # update
        loss0.backward()
        optimizer0.step()
        # scheduler0.step()
        
        print('EPOCH: {:03d}     LOSS: {:.4f}'.format(epoch, loss0))
        # Store the embedding or not
        
        if case_store:
            if config['path_origin'] is None:
                torch.save(x_sim, f"{config['path']}/output/{start_time}/sim_emb.pth")
                torch.save(model_all.state_dict(), f"{config['path']}/output/{start_time}/model_sim.pth")   
            else:
                torch.save(x_rel, f"{config['path']}/output/{start_time}/rel_emb.pth")
                torch.save(model_all.state_dict(), f"{config['path']}/output/{start_time}/model_rel.pth")   
        
        # record time
        end_time = time.time()
        time_elapsed = end_time - now_time
        print(f"Epoch {epoch+1} of {config['epochs']} took {time_elapsed:.2f}s")
        
        # clean cache
        torch.cuda.empty_cache()
        gc.collect()
            
if __name__ == "__main__":
    
    update_config_from_args()
    
    from utils import *
    from utils import append_to_csv
    from data_structure import *
    from evaluate import *
    from Attention import *
    config = get_config()
        
    main(config)