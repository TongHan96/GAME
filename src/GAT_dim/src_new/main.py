# pylint: skip-file

"""
Title: main.py
Author: Han Tong
Date: 2023-10-04
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
    parser.add_argument("--want_TOP1", type=float, default=70.2,
                        help="Top1 performance threshold for storing model and embedding. Default: 70.2.")
    parser.add_argument("--drop_out", type=float, default=0.0,
                        help="Parameter drop_out prob. Default: 0.0.") 
    parser.add_argument("--has_origin_model", type=str, default=False,
                        help="Whether loading original model or not. Default: False.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Parameter learning rate. Default: 5e-4.")    
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
    parser.add_argument("--res", type=str, default=False,
                        help='Whether use resnet not not. Default: False.')
    parser.add_argument("--scale_OTOL", type=int, default=50,
                        help="Parameter scale_one_one. Default: 50.")
    parser.add_argument("--low_dim", type=int, default=0,
                        help="Parameter low_dim. Default: 0.")
    parser.add_argument("--rmax", type=int, default=256,
                        help="Parameter r_max we use for similarity. Default: 256.")  
    parser.add_argument("--Truncate", type=int, default=0,
                        help="Parameter Truncate. Default: 0.")
    parser.add_argument("--path", type=str,
                        help="Specify the path parameter.")
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
    parser.add_argument("--ONLY_SIMI", type=str, default=False,
                    help="whether loss is only similarity or contain relatedness. Default: False.")

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
    config['rmax'] = args.rmax
    config['Truncate'] = args.Truncate
    config['path'] = args.path
    config['path_origin'] = args.path_origin
    config['latent'] = args.latent
    config['FROZEN'] = args.FROZEN
    config['epochs'] = args.epochs
    config['scale_OTOL'] = args.scale_OTOL
    config['res'] = args.res
    config['CHECK_ALL'] = args.CHECK_ALL
    config['ONLY_SIMI'] = args.ONLY_SIMI

    seed = config['SEED']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  
    set_config(config)
        
        
def main(ATTENTION_TYPE, want_TOP1, want_TOP20): 
    
    from load_data import x_tensor, name_all, val_rel_pairs, data, COS_origin_sap, COS_origin_svd, unique_name, name_new, ALL_sim_val_pairs, datax, my_objects
    if str(config['latent']).lower() != 'false':
        from load_data import freq_all, name_new, ALL_sim_val_pairs, ALL_sim_val_pairs_LP, val_rel_pairs_LP, my_objects2
    best_PRE_0 = -float('inf')
    
    if str(config['latent']).lower() == 'true':
        print('load rel embedding!')
        x_origin = torch.load(f"{config['path']}/output/{config['path_origin']}/rel_emb_1.pth")
        
    elif config['path_origin'] is not None:
        print('load origin embedding!')
        x_sim = x_origin = torch.load(f"{config['path']}/output/{config['path_origin']}/sim_emb_1.pth")
        # x_tensor = torch.tensor(datax.iloc[:, 769:1537].to_numpy(), dtype=torch.float32, requires_grad=True)
    else:
        x_origin = x_tensor

    # print(x_tensor)
    PRE_origin,  RELA_MGB_AUC_origin, RELA_VA_AUC_origin, RELA_UP_AUC_origin, SIMI_MGB_origin, SIMI_VA_origin, SIMI_UP_origin = test(x_origin, name_all, related_pairs= val_rel_pairs, similar_pairs= ALL_sim_val_pairs, PRE = True, AUC = True, AUC_type = True, LEVEL=[1], ORIGIN_PACK = None)
    MGB_AUC_origin = [RELA_MGB_AUC_origin, SIMI_MGB_origin]
    VA_AUC_origin = [RELA_VA_AUC_origin, SIMI_VA_origin]
    UP_AUC_origin = [RELA_UP_AUC_origin, SIMI_UP_origin]  
    ORIGIN_PACK = [PRE_origin, RELA_MGB_AUC_origin, RELA_VA_AUC_origin, RELA_UP_AUC_origin, SIMI_MGB_origin, SIMI_VA_origin, SIMI_UP_origin]

    # original test        
    write_file(0, 0, pre = PRE_origin,MGB_AUC1 = MGB_AUC_origin[0][0], VA_AUC1 = VA_AUC_origin[0][0], UP_AUC1 = UP_AUC_origin[0][0], MGB_AUC0 = MGB_AUC_origin[1][0], VA_AUC0 = VA_AUC_origin[1][0], UP_AUC0 = UP_AUC_origin[1][0], begin = True)

    sampler = MySampler(unique_name, name_all, pd.Series(name_new, name='V1'))  
    
    # device
    device = torch.device('cpu')
    x_tensor = x_tensor.to(device)
    
    # load GAT model
    model_all = SandR_Model(config)
    model_all = model_all.to(device)
        
    print(model_all)
    
    if str(config['latent']).lower() != 'false':
        extended_model = ExtendedGAT(model_all)
        print(extended_model)
        extended_model = extended_model.to(device)
        optimizer0 = optim.SGD(extended_model.parameters(), lr=config['base_lr'])
        scheduler0 = CustomExponentialLR(optimizer0, gamma=config['gamma'], min_lr=1e-5)

    elif config['path_origin'] is None:
        optimizer1 = optim.SGD(model_all.S_Model.parameters(), lr=config['base_lr'])
        scheduler1 = CustomExponentialLR(optimizer1, gamma=config['gamma'], min_lr=1e-5)
        
    elif str(config['ONLY_SIMI']).lower() == 'false':
        optimizer2 = optim.SGD(model_all.R_Model.parameters(), lr=config['base_lr'])
        scheduler2 = CustomExponentialLR(optimizer2, gamma=config['gamma'], min_lr=1e-5)
    
    # clean cache
    torch.cuda.empty_cache()
    gc.collect()
    
    if config['path_origin'] is None:
        edge_index = data.edge_index.to(device)
        edge_index = torch.cat((torch.cat((torch.tensor(edge_index), torch.tensor(edges_rel)), dim=1), torch.tensor(edges_sim)), dim=1)
    else:
        edge_index = torch.cat((torch.tensor(edges_rel), torch.tensor(edges_sppmi)), dim=1).to(device)

    undirected_edge_index = to_undirected(edge_index)
    
    for epoch in range(1, config['epochs']+1):
        print('------------------------Epoch: {:03d}-----------------------'.format(epoch))
        n_batchs = sampler.__len__()
        for i in range(n_batchs):            
            ind = sampler.__iter__()
            batch_indices = list(ind)
            
            if str(config['latent']).lower() != 'false':
                
                if str(config['FROZEN']).lower() == 'true':
                    x_new = extended_model(x_origin, undirected_edge_index)
                
                else:
                    x_rel, x_new = extended_model(x_origin, undirected_edge_index)
                
            elif str(config['ONLY_SIMI']).lower() == 'false':
                
                if config['path_origin'] is None:
                    x_sim, x_rel_part = model_all(x_tensor, undirected_edge_index)
  
                else:
                    # print(x_tensor.shape)
                    x_rel_part = model_all(x_tensor, undirected_edge_index)
                
                x_detached = x_sim.detach()
                x_rel = torch.cat((x_detached, x_rel_part), dim=1)

            else:
                x_sim = model_all(x_origin, undirected_edge_index) 
            
            if str(config['latent']).lower() == 'false':
                (P_LOSS_one_one, N_LOSS_one_one, 
                P_LOSS_hie, N_LOSS_hie, 
                P_LOSS_OTOL, N_LOSS_OTOL, 
                P_LOSS_LTOL, N_LOSS_LTOL, 
                P_LOSS_REL, N_LOSS_REL, 
                P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE,
                P_LOSS_SAP1, P_LOSS_SAP2, 
                P_LOSS_SVD1, P_LOSS_SVD2) = custom_loss(my_objects, x_sim if config['path_origin'] is None else None, x_rel if str(config['ONLY_SIMI']).lower() == 'false' else None, batch_indices, device, name_all, COS_origin_sap, COS_origin_svd)
                if str(config['ONLY_SIMI']).lower() == 'false':
                    loss_sppmi = config['scale_attention'] * sppmi_edge_loss(x_rel, edges_sppmi, neg_sppmi)
                
            else:                            
                if str(config['FROZEN']).lower() is False:
                    (P_LOSS_one_one, N_LOSS_one_one, 
                    P_LOSS_hie, N_LOSS_hie, 
                    P_LOSS_OTOL, N_LOSS_OTOL, 
                    P_LOSS_LTOL, N_LOSS_LTOL, 
                    P_LOSS_REL, N_LOSS_REL, 
                    P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE,
                    P_LOSS_SAP1, P_LOSS_SAP2, 
                    P_LOSS_SVD1, P_LOSS_SVD2, LOSS_attention) = custom_loss(my_objects2, x1, batch_indices, device, name_all, edge_attention, rank_attention1, rank_attention2, true_rank, COS_origin_sap, COS_origin_svd)

                    loss0F = (P_LOSS_one_one + N_LOSS_one_one + P_LOSS_hie + N_LOSS_hie + 
                    P_LOSS_OTOL + N_LOSS_OTOL + P_LOSS_LTOL + N_LOSS_LTOL + 
                    P_LOSS_REL + N_LOSS_REL + P_LOSS_SIM_NO_HIE + N_LOSS_SIM_NO_HIE + P_LOSS_SAP1 
                    + P_LOSS_SAP2 + P_LOSS_SVD1 + P_LOSS_SVD2 + LOSS_attention)
                    
                else:
                    (_, _, P_LOSS_hie_LP, N_LOSS_hie_LP, 
                    P_LOSS_OTOL_LP, N_LOSS_OTOL_LP, 
                    P_LOSS_LTOL_LP, N_LOSS_LTOL_LP, 
                    P_LOSS_REL_LP, N_LOSS_REL_LP, 
                    P_LOSS_SIM_NO_HIE_LP, N_LOSS_SIM_NO_HIE_LP, _, _, _, _) = custom_loss(my_objects2, x_new, x_new, batch_indices_new,  device, name_new, ORIGIN=False)
                    
                    loss0 = (P_LOSS_hie_LP + N_LOSS_hie_LP + P_LOSS_OTOL_LP + 
                     N_LOSS_OTOL_LP + P_LOSS_LTOL_LP + N_LOSS_LTOL_LP + P_LOSS_REL_LP +
                     N_LOSS_REL_LP + P_LOSS_SIM_NO_HIE_LP + N_LOSS_SIM_NO_HIE_LP)
                    
                optimizer0.backward()
                loss0.backward()
                optimizer0.step()
                scheduler0.step()
                print('batch_num: {:03d}     LOSS: {:.4f}'.format(i, loss0+loss0F if str(config['FROZEN']).lower() is False else loss0))
                
                PRE_new, RELA_AUC_new, SIMI_AUC_new = test(x_new, name_new, related_pairs= val_rel_pairs_LP, similar_pairs= ALL_sim_val_pairs_LP, PRE = True, AUC = True, AUC_type = True)
                
                if str(config['FROZEN']).lower() is False:
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
                               my_item(P_LOSS_SIM_NO_HIE_LP), my_item(N_LOSS_SIM_NO_HIE_LP)],       
                        pre = PRE_origin,  pre_latent = PRE_new, 
                        MGB_AUC1 = MGB_AUC_origin[0][0], VA_AUC1 = VA_AUC_origin[0][0], UP_AUC1 = UP_AUC_origin[0][0], MGB_AUC0 = MGB_AUC_origin[1][0], VA_AUC0 = VA_AUC_origin[1][0], UP_AUC0 = UP_AUC_origin[1][0], REL_AUC = RELA_AUC_new[0], SIM_AUC = SIMI_AUC_new[0])
                
                else:
                    write_file(0, 0, 
                    loss2=[my_item(P_LOSS_hie_LP), my_item(N_LOSS_hie_LP),
                           my_item(P_LOSS_OTOL_LP), my_item(N_LOSS_OTOL_LP),
                           my_item(P_LOSS_LTOL_LP), my_item(N_LOSS_LTOL_LP),
                           my_item(P_LOSS_REL_LP), my_item(N_LOSS_REL_LP),
                           my_item(P_LOSS_SIM_NO_HIE_LP), my_item(N_LOSS_SIM_NO_HIE_LP)
                    ],       
                    pre = PRE_origin,  pre_latent = PRE_new, 
                    REL_AUC = RELA_AUC_new[0], SIM_AUC = SIMI_AUC_new[0])    
                    
                case_store = (PRE_new[0] >= best_PRE_0) & (PRE_new[0] >= want_TOP1) 
                
                if PRE_new[0] > best_PRE_0:
                    best_PRE_0 = PRE_new[0]
                
                if case_store:
                    if str(config['FROZEN']).lower() is False:
                        torch.save(x_rel, f"{config['path']}/output/{start_time}/latent_emb_1.pth")
                        torch.save(extended_model.state_dict(), f"{config['path']}/output/{start_time}/latent_model_1.pth")   
                    
                    torch.save(x_new, f"{config['path']}/output/{start_time}/latent_emb_2.pth")
                    torch.save(extended_model.state_dict(), f"{config['path']}/output/{start_time}/latent_model_2.pth")
                
                continue
                
            '''
            loss1.back_ward()  # Compute the gradient for loss1
            optimizer1.step()   # Update weights based on the gradient of loss1
            optimizer1.zero_grad()  # Set the parameter gradients to zero
            loss2.back_ward()  # Compute the gradient for loss2
            optimizer2.step()   # Update weights based on the gradient of loss2
            '''
            
            if str(config['latent']).lower() == 'false':
                
                if config['path_origin'] is None:
                    loss1 = P_LOSS_one_one + N_LOSS_one_one + P_LOSS_hie + N_LOSS_hie + P_LOSS_OTOL + N_LOSS_OTOL + P_LOSS_LTOL + N_LOSS_LTOL + P_LOSS_SIM_NO_HIE + N_LOSS_SIM_NO_HIE + P_LOSS_SAP1 + P_LOSS_SAP2

                if str(config['ONLY_SIMI']).lower() == 'false':   
                    loss2 = P_LOSS_REL + N_LOSS_REL + P_LOSS_SVD1 + P_LOSS_SVD2 + loss_sppmi 
                    # loss2 = P_LOSS_REL + N_LOSS_REL + P_LOSS_SVD1 + P_LOSS_SVD2

                if config['path_origin'] is None:
                    print('Update sim!')
                    optimizer1.zero_grad()
                    loss1.backward()
                    optimizer1.step()
                    scheduler1.step()

                if str(config['ONLY_SIMI']).lower() == 'false':   
                    print('Update rel!')
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()   
                    scheduler2.step()

                if config['path_origin'] is not None: 
                    print('batch_num: {:03d}     LOSS: {:.4f}'.format(i, loss2))

                elif str(config['ONLY_SIMI']).lower() == 'true':
                    print('batch_num: {:03d}     LOSS: {:.4f}'.format(i, loss1))

                else:
                    print('batch_num: {:03d}     LOSS: {:.4f}'.format(i, loss1 + loss2))

                PRE_origin, RELA_MGB_AUC_origin, RELA_VA_AUC_origin, RELA_UP_AUC_origin, SIMI_MGB_origin, SIMI_VA_origin, SIMI_UP_origin = test(x_rel if str(config['ONLY_SIMI']).lower() == 'false' else x_sim, name_all, related_pairs= val_rel_pairs, similar_pairs= ALL_sim_val_pairs, PRE = True, AUC = True, AUC_type = True, LEVEL=[1], ORIGIN_PACK=ORIGIN_PACK if config['path_origin'] is not None else None)

                MGB_AUC_origin = [RELA_MGB_AUC_origin, SIMI_MGB_origin]
                VA_AUC_origin = [RELA_VA_AUC_origin, SIMI_VA_origin]
                UP_AUC_origin = [RELA_UP_AUC_origin, SIMI_UP_origin]   

                write_file(epoch, i, 
                    loss=[my_item(P_LOSS_one_one), my_item(N_LOSS_one_one), 
                    my_item(P_LOSS_hie), my_item(N_LOSS_hie), 
                    my_item(P_LOSS_OTOL), my_item(N_LOSS_OTOL), 
                    my_item(P_LOSS_LTOL), my_item(N_LOSS_LTOL), 
                    my_item(P_LOSS_REL), my_item(N_LOSS_REL), 
                    my_item(P_LOSS_SIM_NO_HIE), my_item(N_LOSS_SIM_NO_HIE),
                    my_item(P_LOSS_SAP1), my_item(P_LOSS_SAP2), 
                    my_item(P_LOSS_SVD1), my_item(P_LOSS_SVD2)], 
                    pre = PRE_origin, 
                    MGB_AUC1 = MGB_AUC_origin[0][0], VA_AUC1 = VA_AUC_origin[0][0], UP_AUC1 = UP_AUC_origin[0][0], MGB_AUC0 = MGB_AUC_origin[1][0], VA_AUC0 = VA_AUC_origin[1][0], UP_AUC0 = UP_AUC_origin[1][0])

                # if (PRE_origin[0] > want_TOP1) we will store the model and embedding
                if config['path_origin'] is None:
                    case_store = (PRE_origin[0] >= best_PRE_0) & (PRE_origin[0] >= want_TOP1) 
                    if PRE_origin[0] > best_PRE_0:
                        best_PRE_0 = PRE_origin[0]

                    if case_store:
                        torch.save(x_sim, f"{config['path']}/output/{start_time}/sim_emb_1.pth")
                        if str(config['ONLY_SIMI']).lower() == 'false':   
                            torch.save(x_rel, f"{config['path']}/output/{start_time}/rel_emb_1.pth")
                        torch.save(model_all.state_dict(), 
                                   f"{config['path']}/output/{start_time}/model_1.pth")   
                else:
                    print(f'MGB_REL:{sum_(RELA_MGB_AUC_origin)}    VA_REL:{sum_(RELA_VA_AUC_origin)}    UP_REL:{sum_(RELA_UP_AUC_origin)}')
                    rel_all = sum([sum_(RELA_MGB_AUC_origin),sum_(RELA_VA_AUC_origin),sum_(RELA_UP_AUC_origin)])/3
                    case_store = (rel_all >= best_PRE_0) 

                    if case_store:
                        best_PRE_0 = rel_all
                        torch.save(x_rel, f"{config['path']}/output/{start_time}/rel_emb_1.pth")
                        torch.save(model_all.state_dict(), f"{config['path']}/output/{start_time}/model_2.pth")  
            
        print(f'Epoch{epoch} has finished...')  
        
        if str(config['latent']).lower() != 'false':
            if config['path_origin'] is None:
                torch.save(x_sim, f"{config['path']}/output/{start_time}/sim_emb_epoch{epoch}_1.pth")
                if str(config['ONLY_SIMI']).lower() == 'false':   
                    torch.save(x_rel, f"{config['path']}/output/{start_time}/rel_emb_epoch{epoch}_1.pth")
            else:
                torch.save(x_rel, f"{config['path']}/output/{start_time}/rel_emb_epoch{epoch}_1.pth")
            torch.save(model_all.state_dict(), f"{config['path']}/output/{start_time}/model_epoch{epoch}_2.pth")

            


if __name__ == "__main__":
    
    
    update_config_from_args()
    
    from utils import *
    from utils import append_to_csv
    from data_structure import *
    from evaluate import *
    from Attention import *

        
    main(config['ATTENTION_TYPE'], config['want_TOP1'], config['want_TOP20'])