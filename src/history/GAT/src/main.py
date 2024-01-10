# pylint: skip-file

"""
Title: main.py
Author: Han Tong
Date: 2023-08-29
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
    
    parser.add_argument("--ATTENTION_TYPE", type=str, default='Naive', 
                        help="Attention type for GAT. Choices: 'Naive' or 'A'. Default: 'Naive'.")
    parser.add_argument("--want_TOP1", type=float, default=76.2,
                        help="Top1 performance threshold for storing model and embedding. Default: 76.2.")
    parser.add_argument("--drop_out", type=float, default=0.0,
                        help="Parameter drop_out prob. Default: 0.0.") 
    parser.add_argument("--has_origin_model", type=str, default=False,
                        help="Whether loading original model or not. Default: False.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Parameter learning rate. Default: 5e-4.")    
    parser.add_argument("--AA", type=int, default=1,
                        help="Parameter AA. Default: 1.")
    parser.add_argument("--BB", type=int, default=5,
                        help="Parameter BB. Default: 5.")
    parser.add_argument("--lambd", type=float, default=0.5,
                        help="Parameter lambd. Default: 0.5.")
    parser.add_argument("--scale_one_one", type=int, default=1,
                        help="Parameter scale_one_one. Default: 1.")
    parser.add_argument("--scale_hie", type=int, default=1,
                        help="Parameter scale_hie. Default: 1.")
    parser.add_argument("--res", type=str, default=True,
                        help='Whether use resnet not not. Default: True.')
    parser.add_argument("--scale_OTOL", type=int, default=50,
                        help="Parameter scale_one_one. Default: 50.")
    parser.add_argument("--low_dim", type=int, default=0,
                        help="Parameter low_dim. Default: 0.")
    parser.add_argument("--Truncate", type=int, default=0,
                        help="Parameter Truncate. Default: 0.")
    parser.add_argument("--path_origin", type=str, default=None,
                        help='Train from the initial model and embedding path_origin is not None. Default: None.')
    parser.add_argument("--FROZEN", type=str, default=False,
                        help='Whether the original model be trained not not. Default: False.')
    parser.add_argument("--epochs", type=int, default=3,
                    help='Total Epochs. Default: 3.')
    parser.add_argument("--CHECK_ALL", type=str, default=False,
                    help='whether to check attention or not. Default: False.')
    parser.add_argument("--latent", type=str, default=False,
                help='whether to generate latent nodes or not. Default: False.')
    


    args = parser.parse_args()
    
    config['ATTENTION_TYPE'] = args.ATTENTION_TYPE
    config['want_TOP1'] = args.want_TOP1
    config['has_origin_model'] = args.has_origin_model
    config['AA'] = args.AA
    config['base_lr'] = args.lr
    config['drop_p'] = args.drop_out
    config['BB'] = args.BB
    config['lambd'] = args.lambd
    config['scale_one_one'] = args.scale_one_one
    config['low_dim'] = args.low_dim
    config['Truncate'] = args.Truncate
    config['path_origin'] = args.path_origin
    config['latent'] = args.latent
    config['FROZEN'] = args.FROZEN
    config['epochs'] = args.epochs
    config['scale_OTOL'] = args.scale_OTOL
    config['res'] = args.res
    config['CHECK_ALL'] = args.CHECK_ALL

    seed = config['SEED']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  
    set_config(config)
        
        
def main(ATTENTION_TYPE, want_TOP1, want_TOP20): 
    best_PRE_0 = -float('inf')
    FROZEN = config['FROZEN']
    
    if str(config['has_origin_model']).lower() != "false":
        x_origin = torch.load(f"{config['path_origin']}/Final_emb_1.pth")
    else:
        x_origin = x_tensor
    print(x_origin.shape)
    # original test
    PRE_origin, PRE_0_origin, RELA_MGB_AUC_origin, RELA_VA_AUC_origin, RELA_UP_AUC_origin, SIMI_MGB_origin, SIMI_VA_origin, SIMI_UP_origin = test(x_origin, name_all, related_pairs= val_rel_pairs, similar_pairs= ALL_sim_val_pairs, PRE = True, AUC = True, AUC_type = True, LEVEL=[0,1])
    
    MGB_AUC_origin = [RELA_MGB_AUC_origin, SIMI_MGB_origin]
    VA_AUC_origin = [RELA_VA_AUC_origin, SIMI_VA_origin]
    UP_AUC_origin = [RELA_UP_AUC_origin, SIMI_UP_origin]    
    
    
    if str(config['latent']).lower() != 'false':
        x_new = weighted_sum_x(x_origin, name_all, name_new, freq_all, NORMALIZE=True)
        PRE_new, RELA_AUC_new, SIMI_AUC_new = test(x_new, name_new, related_pairs= val_rel_pairs_LP, similar_pairs= ALL_sim_val_pairs_LP, PRE = True, AUC = True, AUC_type = True, LATENT=True)
        
        write_file(0, 0, pre = PRE_origin, pre0 = PRE_0_origin, pre_latent = PRE_new, MGB_AUC1 = MGB_AUC_origin[0][0], VA_AUC1 = VA_AUC_origin[0][0], UP_AUC1 = UP_AUC_origin[0][0], MGB_AUC0 = MGB_AUC_origin[1][0], VA_AUC0 = VA_AUC_origin[1][0], UP_AUC0 = UP_AUC_origin[1][0], REL_AUC = RELA_AUC_new[0], SIM_AUC = SIMI_AUC_new[0], begin = True)
        
    else:
        write_file(0, 0, pre = PRE_origin, pre0 = PRE_0_origin, MGB_AUC1 = MGB_AUC_origin[0][0], VA_AUC1 = VA_AUC_origin[0][0], UP_AUC1 = UP_AUC_origin[0][0], MGB_AUC0 = MGB_AUC_origin[1][0], VA_AUC0 = VA_AUC_origin[1][0], UP_AUC0 = UP_AUC_origin[1][0], begin = True)
            

    sampler = MySampler(unique_name, name_all, pd.Series(name_new, name='V1'))  
    
    # device
    device = torch.device('cpu')
    x_origin = x_origin.to(device)
    
    # load GAT model
    if ATTENTION_TYPE == 'A':
        model = GAT_A(in_features=config['num_features'], hidden_features=config['hidden_features'],
                      out_features=config['out_dim'], K=config['K'], low_dim=config['low_dim'])
    else:
        model = GATModel(config['num_features'], config['hidden_features'])
        
    model = model.to(device)
    print(model)
    if config['has_origin_model'] is True:
        model.load_state_dict(torch.load(f"{config['path_origin']}/model_1.pth"))
    if str(config['latent']).lower() != 'false':
        extended_model = ExtendedGAT()
        original_weights = model.state_dict()
        extended_model.origin_model.load_state_dict(original_weights)
        print(extended_model)
        extended_model = extended_model.to(device)
        # print(extended_model.sparseLayer.weight)

    if str(config['latent']).lower() != 'false':
        optimizer = optim.SGD(extended_model.parameters(), lr=config['base_lr'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['base_lr'])
        
    scheduler = CustomExponentialLR(optimizer, gamma=config['gamma'], min_lr=1e-5)

    # clean cache
    torch.cuda.empty_cache()
    gc.collect()

    # train the model
    for epoch in range(1, config['epochs']+1):
        print('------------------------Epoch: {:03d}-----------------------'.format(epoch))
        n_batchs = sampler.__len__()
        for i in range(n_batchs):
            optimizer.zero_grad()
            
            edge_index = data.edge_index.to(device)
            edge_rel = torch.tensor(sample_cols(edges_rel, 0.5))
            edge_sim = torch.tensor(sample_cols(edges_sim, 0.5))
            edge_index = torch.cat((torch.cat((edge_index, edge_rel), dim=1), edge_sim), dim=1)
            print(edge_index.shape)
            
            undirected_edge_index = to_undirected(edge_index)
            if str(config['latent']).lower() != 'false':
                ind, ind_new = sampler.__iter__()
                batch_indices_new = list(ind_new)
            else:
                ind = sampler.__iter__()
            batch_indices = list(ind)
            print(x_tensor.shape)
            
            if str(config['latent']).lower() == 'false':
                x1, attention1, attention2 = model(x_tensor, undirected_edge_index)
                
            else:
                if FROZEN is False:
                    x1, x1_new, attention1, attention2 = extended_model(x_tensor, undirected_edge_index)
                else:
                    x1_new = extended_model(x_origin, undirected_edge_index)
            
            append_to_csv('attention1', attention1)
            append_to_csv('attention2', attention2)
            
            if FROZEN is False:
                (P_LOSS_one_one, N_LOSS_one_one, 
                P_LOSS_hie, N_LOSS_hie, 
                P_LOSS_OTOL, N_LOSS_OTOL, 
                P_LOSS_LTOL, N_LOSS_LTOL, 
                P_LOSS_REL, N_LOSS_REL, 
                P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE,
                P_LOSS_SAP1, P_LOSS_SAP2, 
                P_LOSS_SVD1, P_LOSS_SVD2) = custom_loss(my_objects, x1, batch_indices, device, name_all, COS_origin_sap, COS_origin_svd)

                loss = (P_LOSS_one_one + N_LOSS_one_one + P_LOSS_hie + N_LOSS_hie + 
                P_LOSS_OTOL + N_LOSS_OTOL + P_LOSS_LTOL + N_LOSS_LTOL + 
                P_LOSS_REL + N_LOSS_REL + P_LOSS_SIM_NO_HIE + N_LOSS_SIM_NO_HIE + P_LOSS_SAP1 
                + P_LOSS_SAP2 + P_LOSS_SVD1 + P_LOSS_SVD2)
            
            if str(config['latent']).lower() != 'false':
                (P_LOSS_hie_LP, N_LOSS_hie_LP, 
                P_LOSS_OTOL_LP, N_LOSS_OTOL_LP, 
                P_LOSS_LTOL_LP, N_LOSS_LTOL_LP, 
                P_LOSS_REL_LP, N_LOSS_REL_LP, 
                P_LOSS_SIM_NO_HIE_LP, N_LOSS_SIM_NO_HIE_LP) = custom_loss(my_objects2, x1_new, batch_indices_new,  device, name_new, ORIGIN=False)
            
                loss2 = (P_LOSS_hie_LP + N_LOSS_hie_LP + P_LOSS_OTOL_LP + 
                         N_LOSS_OTOL_LP + P_LOSS_LTOL_LP + N_LOSS_LTOL_LP + P_LOSS_REL_LP +
                         N_LOSS_REL_LP + P_LOSS_SIM_NO_HIE_LP + N_LOSS_SIM_NO_HIE_LP)
                
                if FROZEN is False:
                    print('batch_num: {:03d}     LOSS: {:.4f}     new_LOSS: {:.4f}'.format(i, loss, loss2))
                    loss_all = loss + loss2
                else:
                    print('batch_num: {:03d}     new_LOSS: {:.4f}'.format(i, loss2))
                    loss_all = loss2
                    
                loss_all.backward()
            
            else:
                print('batch_num: {:03d}     LOSS: {:.4f}'.format(i, loss))
                loss.backward()

            optimizer.step()
            scheduler.step()
            
            # original test
            if FROZEN is False:
                PRE_origin, PRE_0_origin, RELA_MGB_AUC_origin, RELA_VA_AUC_origin, RELA_UP_AUC_origin, SIMI_MGB_origin, SIMI_VA_origin, SIMI_UP_origin = test(x1, name_all, related_pairs= val_rel_pairs, similar_pairs= ALL_sim_val_pairs, PRE = True, AUC = True, AUC_type = True, LEVEL=[0,1])

                MGB_AUC_origin = [RELA_MGB_AUC_origin, SIMI_MGB_origin]
                VA_AUC_origin = [RELA_VA_AUC_origin, SIMI_VA_origin]
                UP_AUC_origin = [RELA_UP_AUC_origin, SIMI_UP_origin]    

            if str(config['latent']).lower() != 'false':

                PRE_new, RELA_AUC_new, SIMI_AUC_new = test(x1_new, name_new, related_pairs= val_rel_pairs_LP, similar_pairs= ALL_sim_val_pairs_LP, PRE = True, AUC = True, AUC_type = True, LATENT=True)
                
                if FROZEN is False:
                    write_file(0, 0, 
                        loss=[my_item(P_LOSS_one_one), my_item(N_LOSS_one_one), 
                        my_item(P_LOSS_hie), my_item(N_LOSS_hie), 
                        my_item(P_LOSS_OTOL), my_item(N_LOSS_OTOL), 
                        my_item(P_LOSS_LTOL), my_item(N_LOSS_LTOL), 
                        my_item(P_LOSS_REL), my_item(N_LOSS_REL), 
                        my_item(P_LOSS_SIM_NO_HIE), my_item(N_LOSS_SIM_NO_HIE),
                        my_item(P_LOSS_SAP1), my_item(P_LOSS_SAP2), 
                        my_item(P_LOSS_SVD1), my_item(P_LOSS_SVD2)], 
                        loss2=[my_item(P_LOSS_hie_LP), my_item(N_LOSS_hie_LP),
                               my_item(P_LOSS_OTOL_LP), my_item(N_LOSS_OTOL_LP),
                               my_item(P_LOSS_LTOL_LP), my_item(N_LOSS_LTOL_LP),
                               my_item(P_LOSS_REL_LP), my_item(N_LOSS_REL_LP),
                               my_item(P_LOSS_SIM_NO_HIE_LP), my_item(N_LOSS_SIM_NO_HIE_LP)
                        ],       
                        pre = PRE_origin, pre0 = PRE_0_origin, pre_latent = PRE_new, 
                        MGB_AUC1 = MGB_AUC_origin[0][0], VA_AUC1 = VA_AUC_origin[0][0], UP_AUC1 = UP_AUC_origin[0][0], MGB_AUC0 = MGB_AUC_origin[1][0], VA_AUC0 = VA_AUC_origin[1][0], UP_AUC0 = UP_AUC_origin[1][0], REL_AUC = RELA_AUC_new[0], SIM_AUC = SIMI_AUC_new[0])
                    
                else:
                    write_file(0, 0, 
                    loss2=[my_item(P_LOSS_hie_LP), my_item(N_LOSS_hie_LP),
                           my_item(P_LOSS_OTOL_LP), my_item(N_LOSS_OTOL_LP),
                           my_item(P_LOSS_LTOL_LP), my_item(N_LOSS_LTOL_LP),
                           my_item(P_LOSS_REL_LP), my_item(N_LOSS_REL_LP),
                           my_item(P_LOSS_SIM_NO_HIE_LP), my_item(N_LOSS_SIM_NO_HIE_LP)
                    ],       
                    pre = PRE_origin, pre0 = PRE_0_origin, pre_latent = PRE_new, 
                    REL_AUC = RELA_AUC_new[0], SIM_AUC = SIMI_AUC_new[0]) 

            else:
                write_file(0, 0, 
                    loss=[my_item(P_LOSS_one_one), my_item(N_LOSS_one_one), 
                    my_item(P_LOSS_hie), my_item(N_LOSS_hie), 
                    my_item(P_LOSS_OTOL), my_item(N_LOSS_OTOL), 
                    my_item(P_LOSS_LTOL), my_item(N_LOSS_LTOL), 
                    my_item(P_LOSS_REL), my_item(N_LOSS_REL), 
                    my_item(P_LOSS_SIM_NO_HIE), my_item(N_LOSS_SIM_NO_HIE),
                    my_item(P_LOSS_SAP1), my_item(P_LOSS_SAP2), 
                    my_item(P_LOSS_SVD1), my_item(P_LOSS_SVD2)], 
                    pre = PRE_origin, pre0 = PRE_0_origin, 
                    MGB_AUC1 = MGB_AUC_origin[0][0], VA_AUC1 = VA_AUC_origin[0][0], UP_AUC1 = UP_AUC_origin[0][0], MGB_AUC0 = MGB_AUC_origin[1][0], VA_AUC0 = VA_AUC_origin[1][0], UP_AUC0 = UP_AUC_origin[1][0])
            
            # print(extended_model.sparseLayer.weight)
            
            # if (PRE_origin[0] > want_TOP1) we will store the model and embedding
            if str(config['latent']).lower() != 'false':
                case_store =  (PRE_new[0] >= best_PRE_0) & (PRE_new[0] >= want_TOP1)
                if PRE_new[0] > best_PRE_0:
                    best_PRE_0 = PRE_new[0]
            else:
                case_store = (PRE_origin[0] >= best_PRE_0) & (PRE_origin[0] >= want_TOP1) 
                if PRE_origin[0] > best_PRE_0:
                    best_PRE_0 = PRE_origin[0]
            
            if case_store:
                if FROZEN is False:
                    torch.save(x1, f'/root/current_code/GAT_model_9_1/output/{start_time}/Final_emb_1.pth')
                    torch.save(model.state_dict(), 
                               f'/root/current_code/GAT_model_9_1/output/{start_time}/model_1.pth')   
                if str(config['latent']).lower() != 'false':
                    torch.save(x1_new, 
                               f'/root/current_code/GAT_model_9_1/output/{start_time}/Final_emb_2.pth')
                    torch.save(extended_model.state_dict(), 
                               f'/root/current_code/GAT_model_9_1/output/{start_time}/model_2.pth')
                
        print(f'Epoch{epoch} has finished...')
        scheduler.step()    
        if FROZEN is False:
            torch.save(x1, f'/root/current_code/GAT_model_9_1/output/{start_time}/Final_emb_epoch{epoch}_1.pth')
            torch.save(model.state_dict(), 
                       f'/root/current_code/GAT_model_9_1/output/{start_time}/model_epoch{epoch}_1.pth')   
        if str(config['latent']).lower() != 'false':
            torch.save(x1_new, 
                       f'/root/current_code/GAT_model_9_1/output/{start_time}/Final_emb_epoch{epoch}_2.pth')
            torch.save(extended_model.state_dict(), 
                       f'/root/current_code/GAT_model_9_1/output/{start_time}/model_epoch{epoch}_2.pth')        

if __name__ == "__main__":
    
    update_config_from_args()
    
    from utils import *
    from utils import append_to_csv
    from data_structure import *
    from load_data import x_tensor, name_all, val_rel_pairs, data, COS_origin_sap, COS_origin_svd, unique_name, my_objects
    if str(config['latent']).lower() != 'false':
        from load_data import freq_all, name_new, ALL_sim_val_pairs, ALL_sim_val_pairs_LP, val_rel_pairs_LP, my_objects2
    from evaluate import *
    from Attention import *
        
    main(config['ATTENTION_TYPE'], config['want_TOP1'], config['want_TOP20'])