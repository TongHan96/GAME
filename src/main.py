# pylint: skip-file

"""
Title: main.py
Author: Han Tong
Date: 2024-02-10
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
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"

def update_config_from_args():
    from config import set_config, get_config
    
    config = get_config()

    parser = argparse.ArgumentParser(description="GAMEw Training Script")
    parser.add_argument("--want_TOP1", type=float, default=77,
                        help="Top1 performance threshold for storing model and embedding. Default: 77.")
    parser.add_argument("--want_TOP20", type=float, default=93,
                        help="Top1 performance threshold for storing model and embedding. Default: 93.")
    parser.add_argument("--EDGE_ALL", type=str, default='True',
                        help="Whether to use all edges to train similarity. Default: True.")
    parser.add_argument("--drop_out", type=float, default=0.0,
                        help="Parameter drop_out prob. Default: 0.0.") 
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Parameter learning rate. Default: 1e-3.")    
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
    parser.add_argument("--scale_sppmi", type=float, default=100,
                        help="Parameter scale_sppmi. Default: 100.")
    parser.add_argument("--scale_OTOL", type=int, default=50,
                        help="Parameter scale_one_one. Default: 50.")
    parser.add_argument("--scale_REL", type=int, default=5,
                        help="Parameter scale_REL. Default: 5.")   
    parser.add_argument("--scale_align", type=int, default=1,
                        help="Parameter scale_REL. Default: 1.")  
    parser.add_argument("--rmax", type=int, default=256,
                        help="Parameter r_max we use for similarity. Default: 256.")  
    parser.add_argument("--hidden_features", type=int, default=768,
                        help="Parameter r_max we use for similarity. Default: 768.")  
    parser.add_argument("--path", type=str, default=config['path'],
                        help="Specify the path parameter.")
    parser.add_argument("--input_dir", type=str, default=config['input_dir'],
                        help="Specify the path parameter for input data.")
    parser.add_argument("--path_origin", type=str, default=config['path_origin'],
                        help='Get aligned sppmi if embedding path_origin is align_NA. Else Train from the initial model and embedding path_origin is not None. Default: None.')
    parser.add_argument("--epochs", type=int, default=3,
                    help='Total Epochs. Default: 3.')
    parser.add_argument("--CHECK_ALL", type=str, default=False,
                    help='whether to check attention or not. Default: False.')
    parser.add_argument("--DEVICE", type=str, default='cuda:0',
                    help='Use GPU or CPU. Default: cuda:0.')
    parser.add_argument("--num_inst", type=int, default=4,
                    help='The number of institutions. Default: 4.')

    args = parser.parse_args()

    config['num_inst'] = args.num_inst
    config['want_TOP1'] = args.want_TOP1
    config['want_TOP20'] = args.want_TOP20
    config['EDGE_ALL'] = args.EDGE_ALL
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
    config['scale_align'] = args.scale_align
    config['rmax'] = args.rmax
    config['hidden_features'] = args.hidden_features
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
    
    from load_data import sppmi_list, sap_emb, unique_name, val_rel_pairs, ALL_sim_val_pairs, edges_sppmi, my_objects, pos_sppmi, neg_sppmi, edges, edges_sim, edges_rel, same_desc_edge
    
    # load data to device
    device = torch.device(config['DEVICE'])
    if config['path_origin'] == "align_NA":
        sppmi_list = [inst_emb.to(device) for inst_emb in sppmi_list]
        
    else:
        # we have get aligned sppmi emb, stored in  .../align_sppmi folder
        print('load aligned sppmi embedding!')
        out_1 =  torch.load(f"{config['path']}/output/align_sppmi/align_sppmi.pth", map_location=device)
        out_1 = out_1.detach()
        out_1 = out_1.to(device)
    
    if (config['path_origin'] != 'align_NA') & (config['path_origin'] is not None):
        # simi embedding will be fixed duing related training process
        print('load similarity embedding!')
        x_sim_trained = torch.load(f"{config['path']}/output/{config['path_origin']}/sim_emb.pth", map_location=device)
        x_sim_trained = x_sim_trained.detach()
        x_sim_trained = x_sim_trained.to(device)
        
    sap_emb = sap_emb.to(device)
    
    # load GAT model
    model_all = inst_encoder(config)
    model_all = model_all.to(device)
    print(model_all)
    optimizer0 = optim.SGD(model_all.parameters(), lr=config['base_lr'])
    scheduler0 = CustomExponentialLR(optimizer0, gamma=config['gamma'], min_lr=5e-7)
    
    # load edges
    ## edges: code mapping + hierarchy
    ## edges_sim: non-hierarchy edges sim
    ## same_desc_edge: the codes having same descriptions
    ## edges_rel: edges_rel
    ## edges_sppmi: common edges from 4 insts
    ## pos_sppmi: uncommon edges selected by GPT4
    
    edge_all_sim = torch.cat((torch.cat((edges, edges_sim), dim=1), same_desc_edge), dim = 1)
    # edge_all_rel = torch.cat((torch.cat((torch.cat((torch.cat((edges_rel, edges_sppmi), dim=1), pos_sppmi), dim=1), same_desc_edge), dim = 1), pos_coder),dim=1)  # tmp!!
    edge_all_rel = torch.cat((torch.cat((torch.cat((edges_rel, edges_sppmi), dim=1), pos_sppmi), dim=1), same_desc_edge), dim = 1)
    
    if config['path_origin'] == "align_NA":
        record = -float('inf')
        loss_agg = 0
        edge_index = torch.cat((edge_all_sim, edge_all_rel), dim = 1)
    
    elif config['path_origin'] is None:
        record = -float('inf')
        if config['EDGE_ALL'] == 'True':
            edge_index = torch.cat((edge_all_sim, edge_all_rel), dim = 1) ## TMP!
        else:
            edge_index = edge_all_sim
        
    else:
        record = -float('inf')
        record2 = -float('inf')
        if config['EDGE_ALL'] == 'True':
            edge_index = torch.cat((edge_all_sim, edge_all_rel), dim = 1)
        else:
            edge_index = edge_all_rel
        # edge_index = torch.cat((torch.cat((torch.cat((torch.cat((edges_rel, edges_sppmi), dim=1), pos_sppmi), dim=1), same_desc_edge), dim = 1), edge_all_sim),dim=1)
        # edge_index = torch.cat((torch.cat((torch.cat((edges_rel, edges_sppmi), dim=1), pos_sppmi), dim=1), same_desc_edge), dim = 1)
        # edge_index = torch.cat((edge_all_sim, edge_all_rel), dim = 1) ## TMP!

    edge_index = remove_duplicate_edge(edge_index)       
    print(edge_index.shape)
    undirected_edge_index = to_undirected(edge_index).to(device)
    
    # begin training
    for epoch in range(1, 1+config['epochs']):
        now_time = time.time()
        case_store = False
        optimizer0.zero_grad()
        model_all.train()
        
        # align sppmi case
        if config['path_origin'] == "align_NA":
            align_loss_term, x_sim = model_all(sppmi_list=sppmi_list, sap_emb=sap_emb, edge_index=undirected_edge_index, config=config)
            # P_sppmi, N_sppmi = sppmi_edge_loss(x_sim, pos_sppmi, neg_sppmi, config)
            # P_coder, N_coder = sppmi_edge_loss(x_sim, pos_coder, neg_coder, config)
            loss0 = align_loss_term
            loss_agg += loss0
            loss = [my_item(loss0)]
        
        # simi embedding training case
        elif config['path_origin'] is None:
            x_sim = model_all(sap_emb=sap_emb, out_1=out_1, edge_index=undirected_edge_index, config=config)
            P_LOSS_hie, N_LOSS_hie, P_LOSS_OTOL, N_LOSS_OTOL, P_LOSS_LTOL, N_LOSS_LTOL, P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE = custom_loss(my_objects, x_sim, list(range(config['num_union'])), device, unique_name, config)
            loss0 = P_LOSS_hie + N_LOSS_hie + P_LOSS_OTOL + N_LOSS_OTOL + P_LOSS_LTOL + N_LOSS_LTOL + P_LOSS_SIM_NO_HIE + N_LOSS_SIM_NO_HIE            
            loss = [my_item(P_LOSS_hie), my_item(N_LOSS_hie), 
                    my_item(P_LOSS_OTOL), my_item(N_LOSS_OTOL), 
                    my_item(P_LOSS_LTOL), my_item(N_LOSS_LTOL), 
                    my_item(P_LOSS_SIM_NO_HIE), my_item(N_LOSS_SIM_NO_HIE)]
       
        # rela embedding training case
        else:
            x_rel_part = model_all(sap_emb=sap_emb, out_1=out_1, edge_index=undirected_edge_index, config=config)
            x_rel = torch.cat((x_sim_trained, x_rel_part), dim=1) # concat fixed simi embedding
            P_REL, N_REL = custom_loss(my_objects, x_rel, list(range(config['num_union'])), device, unique_name, config)
            P_sppmi, N_sppmi = sppmi_edge_loss(x_rel, pos_sppmi, neg_sppmi, config)
            # P_coder, N_coder = sppmi_edge_loss(x_rel, pos_coder, neg_coder, config)
            # loss0 = P_REL + N_REL + P_sppmi + N_sppmi + P_coder + N_coder
            # loss = [my_item(P_REL), my_item(N_REL), my_item(P_sppmi), my_item(N_sppmi), my_item(P_coder), my_item(N_coder)]  # tmp!!
            
            loss0 = P_REL + N_REL + P_sppmi + N_sppmi
            loss = [my_item(P_REL), my_item(N_REL), my_item(P_sppmi), my_item(N_sppmi), 0,0]  # tmp!!
            
        # update
        loss0.backward()
        optimizer0.step()
        scheduler0.step()
        torch.cuda.empty_cache()
        
        # evaluate 
        if epoch % 10 == 1:
            model_all.eval()
            if config['path_origin'] == 'align_NA':
                print(f'loss in 10 epochs = {np.round(my_item(loss_agg),3)}') 
                x_sim_test = model_all(sppmi_list, sap_emb=sap_emb, edge_index=undirected_edge_index, config=config)
                PRE_new, AUC_new, AUC_new2, AUC_new3 = test(x_sim_test, unique_name, config, similar_pairs=ALL_sim_val_pairs, related_pairs=val_rel_pairs, drug_side_pairs=drug_side_pairs, PRE=True, AUC=True, AUC_type=True)
                write_file(epoch, 0, config, loss=loss, pre=PRE_new, SIM_AUC=AUC_new[0][0], REL_AUC=AUC_new2[0][0], DRUG_AUC=AUC_new3[0][0])
                # feature selection and evalution
                emb_all = list([sap_emb, coder_emb, sppmi_list[0], sppmi_list[1], sppmi_list[2], sppmi_list[3], x_sim_test.cpu().detach()])

                new_corr = feature_selection_every_epoch(emb_all, start_time, epoch,RECORD=record)
                print(f'Corr: {np.round(new_corr,3)}')

                # whether to break training and store model
                case_store = (new_corr > record)
                if case_store:
                    if epoch > 1:
                        record = new_corr
                elif epoch > 50:
                    break
                loss_agg = 0
            
            elif config['path_origin'] is None:
                x_sim_test = model_all(sap_emb=sap_emb, out_1=out_1, edge_index=undirected_edge_index, config=config)
                PRE_new, AUC_new, AUC_new3 = test(x_sim_test, unique_name, config, similar_pairs=ALL_sim_val_pairs, drug_side_pairs=drug_side_pairs, PRE=True, AUC=True, AUC_type=True)
                write_file(epoch, 0, config, loss=loss, pre=PRE_new, SIM_AUC=AUC_new[0][0], DRUG_AUC=AUC_new3[0][0])

                emb_all = list([sap_emb, coder_emb, sppmi_list[0], sppmi_list[1], sppmi_list[2], sppmi_list[3], x_sim_test.cpu().detach()])
                new_corr = feature_selection_every_epoch(emb_all, start_time, epoch)
                case_store = (PRE_new[3] > record) # tmp!!
                if case_store:
                    record = PRE_new[3]
            
            else:
                x_rel_part_test = model_all(sap_emb=sap_emb, out_1=out_1, edge_index=undirected_edge_index, config=config)
                x_rel_test = torch.cat((x_sim_trained, x_rel_part_test), dim=1)
                AUC_new, AUC_new3 = test(x_rel_test, unique_name, config, related_pairs=val_rel_pairs, drug_side_pairs=drug_side_pairs, PRE=False, AUC=True, AUC_type=True)   
                rel_all = sum_(AUC_new[0])
                write_file(epoch, 0, config, loss=loss, REL_AUC=AUC_new[0][0], DRUG_AUC=AUC_new3[0][0])
                emb_all = list([sap_emb, coder_emb, sppmi_list[0], sppmi_list[1], sppmi_list[2], sppmi_list[3], x_rel_test.cpu().detach()])
                new_corr = feature_selection_every_epoch(emb_all, start_time, epoch)
                print(f'AUC = {np.round(rel_all,3)}')
                print(f'Corr: {np.round(new_corr,3)}')
                case_store = (rel_all > record) 
                case_store2 = (new_corr > record2)
                if case_store:
                    record = rel_all
                if case_store2:
                    record2 = new_corr
                case_store = case_store | case_store2
        
        else:
            write_file(epoch, 0, config, loss=loss)                  
        
        # Store the embedding or not
        # if store embedding, we need to evaluate features selection
        if case_store:
            if config['path_origin'] == "align_NA":
                torch.save(x_sim_test, f"{config['path']}/output/{start_time}/align_sppmi.pth")
                torch.save(model_all.state_dict(), f"{config['path']}/output/{start_time}/model_align.pth")  
            
            elif config['path_origin'] is None:
                torch.save(x_sim_test, f"{config['path']}/output/{start_time}/sim_emb.pth")
                torch.save(model_all.state_dict(), f"{config['path']}/output/{start_time}/model_sim.pth")   
                emb = pd.DataFrame(x_sim_test.cpu().detach().numpy())
                emb.to_csv(f"{config['path']}/output/{start_time}/sim_emb.csv", index=None)
                
            else:
                torch.save(x_rel_test, f"{config['path']}/output/{start_time}/rel_emb.pth")
                torch.save(model_all.state_dict(), f"{config['path']}/output/{start_time}/model_rel.pth") 
                emb = pd.DataFrame(x_rel_test.cpu().detach().numpy())
                emb.to_csv(f"{config['path']}/output/{start_time}/rel_emb.csv", index=None)

        # record time
        end_time = time.time()
        time_elapsed = end_time - now_time
        print('EPOCH: {:03d}     LOSS: {:.4f}     ELAPSED: {:.2f}s'.format(epoch, loss0, time_elapsed))
        
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