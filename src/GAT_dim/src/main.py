# pylint: skip-file

"""
Title: main.py
Author: Han Tong
Date: 2023-09-22
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
torch.autograd.set_detect_anomaly(True)

def update_config_from_args():
    from config import set_config, get_config
    
    config = get_config()

    parser = argparse.ArgumentParser(description="GAT Training Script")
    
    parser.add_argument("--ATTENTION_TYPE", type=str, default='Naive', 
                        help="Attention type for GAT. Choices: 'Naive' or 'A'. Default: 'Naive'.")
    parser.add_argument("--want_TOP1", type=float, default=70.2,
                        help="Top1 performance threshold for storing model and embedding. Default: 76.2.")
    parser.add_argument("--drop_out", type=float, default=0.1,
                        help="Parameter drop_out prob. Default: 0.1.") 
    parser.add_argument("--has_origin_model", type=str, default=False,
                        help="Whether loading original model or not. Default: False.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Parameter learning rate. Default: 1e-4.")    
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
    parser.add_argument("--res", type=str, default=False,
                        help='Whether use resnet not not. Default: False.')
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
    parser.add_argument("--epochs", type=int, default=1,
                        help='Total Epochs. Default: 1.')
    parser.add_argument("--CHECK_ALL", type=str, default=False,
                        help='whether to check attention or not. Default: False.')
    parser.add_argument("--latent", type=str, default=False,
                        help='whether to generate latent nodes or not. Default: False.')
    parser.add_argument('--hidden_features', type=int, default=1024,
                        help = "Parameter hidden_features dimension. Default: 1024.")
    parser.add_argument('--scale_attention', type=float, default=1e-3,
                        help = "Parameter scale attention_loss. Default: 1e-3. ")
    parser.add_argument('--edge_all', type=str, default=False,
                        help = "Whether to use related information to update similar embedding part. Default: False.")
    parser.add_argument('--rmax', type=int, default=256,
                        help = "Parameter define similar embedding dimension. Default: 256.")
    parser.add_argument('--origin_start_time', type=str, default=None,
                        help = "Parameter that whether to import the original similar model or not. Dafault: None.")
                        
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
    config['origin_start_time'] = args.origin_start_time
    config['scale_OTOL'] = args.scale_OTOL
    config['res'] = args.res
    config['CHECK_ALL'] = args.CHECK_ALL
    config['hidden_features'] = args.hidden_features
    config['rmax'] = args.rmax
    config['out_dim'] = args.rmax
    config['edge_all'] = args.edge_all
                        

    seed = config['SEED']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  
    set_config(config)
  

# def update_model(loss, optimizer, retain_graph=False):
#     optimizer.zero_grad()
#     loss.backward(retain_graph=retain_graph)
#     optimizer.step()
        
def main(ATTENTION_TYPE, want_TOP1, want_TOP20): 
    best_PRE_0 = -float('inf')
    FROZEN = config['FROZEN']
    x_origin = x_tensor
    device = torch.device('cpu')
    x_origin = x_origin.to(device)
    edge_index = data.edge_index.to(device)
    edge_index1 = torch.cat((torch.tensor(edge_index), torch.tensor(edges_sim)), dim=1)
    edge_index2 = torch.cat((edge_index1, torch.tensor(edges_rel)), dim=1)
    undirected_edge_index1 = to_undirected(edge_index1)
    undirected_edge_index2 = to_undirected(edge_index2)
    if config['edge_all']:
        undirected_edge_index = undirected_edge_index2
    else:
        undirected_edge_index = undirected_edge_index1
    
    if config['origin_start_time'] is None:
        # original test
        PRE_origin, PRE_0_origin, RELA_MGB_AUC_origin, RELA_VA_AUC_origin, RELA_UP_AUC_origin, SIMI_MGB_origin, SIMI_VA_origin, SIMI_UP_origin = test(x_origin, name_all, related_pairs= val_rel_pairs, similar_pairs= ALL_sim_val_pairs, PRE = True, AUC = True, AUC_type = True, LEVEL=[0,1])

        MGB_AUC_origin = [RELA_MGB_AUC_origin, SIMI_MGB_origin]
        VA_AUC_origin = [RELA_VA_AUC_origin, SIMI_VA_origin]
        UP_AUC_origin = [RELA_UP_AUC_origin, SIMI_UP_origin]    

        write_file(0, 0, pre = PRE_origin, pre0 = PRE_0_origin, MGB_AUC1 = MGB_AUC_origin[0][0], VA_AUC1 = VA_AUC_origin[0][0], UP_AUC1 = UP_AUC_origin[0][0], MGB_AUC0 = MGB_AUC_origin[1][0], VA_AUC0 = VA_AUC_origin[1][0], UP_AUC0 = UP_AUC_origin[1][0], begin = True, ONLY_SIMI=True)
            
        sampler = MySampler(unique_name, name_all, pd.Series(name_new, name='V1'))  

        # load GAT model
        if ATTENTION_TYPE == 'A':
            model = GAT_A(in_features=config['num_features'], hidden_features=config['hidden_features2'],
                          out_features=config['all_dim'], K=config['K'], low_dim=config['low_dim'])
        else:
            model = GATModel(config['num_features'], config['hidden_features2'], MASK=True)

        model = model.to(device)
        print(model)
        
        optimizer = optim.SGD(model.parameters(), lr=config['base_lr'])
        scheduler = CustomExponentialLR(optimizer, gamma=config['gamma'], min_lr=1e-5)
        
        # clean cache
        torch.cuda.empty_cache()
        gc.collect()
        
        config['origin_start_time'] = start_time
        # train the model, first step: similarity
        for epoch in range(1, config['epochs']+1):
            print('------------------------Epoch: {:03d}-----------------------'.format(epoch))
            n_batchs = sampler.__len__()
            for i in range(n_batchs):
                optimizer.zero_grad()
                ind = sampler.__iter__()
                batch_indices = list(ind)

                x1, attention1, attention2, rank_attention1, rank_attention2 = model(x_tensor, undirected_edge_index, edge_attention)

                append_to_csv('attention1', attention1)
                append_to_csv('attention2', attention2)

                (P_LOSS_one_one, N_LOSS_one_one, 
                P_LOSS_hie, N_LOSS_hie, 
                P_LOSS_OTOL, N_LOSS_OTOL, 
                P_LOSS_LTOL, N_LOSS_LTOL, 
                P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE,
                P_LOSS_SAP1, P_LOSS_SAP2, 
                P_LOSS_SVD1, P_LOSS_SVD2, 
                LOSS_attention) = custom_loss(my_objects, x1[:,0:config['rmax']], batch_indices, device, name_all, edge_attention, rank_attention1, rank_attention2, true_rank, COS_origin_sap, COS_origin_svd, ONLY_SIMI=True)

                loss = (P_LOSS_one_one + N_LOSS_one_one + 
                        P_LOSS_hie + N_LOSS_hie + 
                        P_LOSS_OTOL + N_LOSS_OTOL + 
                        P_LOSS_LTOL + N_LOSS_LTOL + 
                        P_LOSS_SIM_NO_HIE + N_LOSS_SIM_NO_HIE + 
                        P_LOSS_SAP1 + P_LOSS_SAP2 + 
                        P_LOSS_SVD1 + P_LOSS_SVD2 + 
                        LOSS_attention)
                
                print('batch_num: {:03d}     LOSS: {:.4f}'.format(i, loss))
                loss.backward() 
                mask_tensor(model.gat1.gat_conv.lin_src_new.weight, model.gat1.gat_conv.lin_src_new.weight.mask)
                mask_tensor(model.gat1.gat_conv.att_dst_new, model.gat1.gat_conv.att_dst_new.mask)
                mask_tensor(model.gat1.gat_conv.att_src_new, model.gat1.gat_conv.att_src_new.mask)
                mask_tensor(model.gat2.gat_conv.lin_src_new.weight, model.gat2.gat_conv.lin_src_new.weight.mask)
                mask_tensor(model.gat2.gat_conv.att_dst_new, model.gat2.gat_conv.att_dst_new.mask)
                mask_tensor(model.gat2.gat_conv.att_src_new, model.gat2.gat_conv.att_src_new.mask)
                mask_tensor(model.linear.weight, model.linear.weight.mask)
                optimizer.step()
                scheduler.step()
                print('----------------check all parameters----------------')
                print(model.gat1.gat_conv.lin_src_new.weight)
                print(model.gat1.gat_conv.att_dst_new)
                print(model.gat1.gat_conv.att_src_new)
                print(model.gat2.gat_conv.lin_src_new.weight)
                print(model.gat2.gat_conv.att_dst_new)
                print(model.gat2.gat_conv.att_src_new)
                print(model.linear.weight)
                print('------------------check embedding-------------------')
                print(x1.shape)
                print(x1)
                PRE_origin, PRE_0_origin, RELA_MGB_AUC_origin, RELA_VA_AUC_origin, RELA_UP_AUC_origin, SIMI_MGB_origin, SIMI_VA_origin, SIMI_UP_origin = test(x1, name_all, related_pairs= val_rel_pairs, similar_pairs= ALL_sim_val_pairs, PRE = True, AUC = True, AUC_type = True, LEVEL=[0,1])

                MGB_AUC_origin = [RELA_MGB_AUC_origin, SIMI_MGB_origin]
                VA_AUC_origin = [RELA_VA_AUC_origin, SIMI_VA_origin]
                UP_AUC_origin = [RELA_UP_AUC_origin, SIMI_UP_origin]    

                write_file(epoch, i, 
                    loss=[my_item(P_LOSS_one_one), my_item(N_LOSS_one_one), 
                    my_item(P_LOSS_hie), my_item(N_LOSS_hie), 
                    my_item(P_LOSS_OTOL), my_item(N_LOSS_OTOL), 
                    my_item(P_LOSS_LTOL), my_item(N_LOSS_LTOL), 
                    my_item(P_LOSS_SIM_NO_HIE), my_item(N_LOSS_SIM_NO_HIE),
                    my_item(P_LOSS_SAP1), my_item(P_LOSS_SAP2), 
                    my_item(P_LOSS_SVD1), my_item(P_LOSS_SVD2),
                    my_item(LOSS_attention)], 
                    pre = PRE_origin, pre0 = PRE_0_origin, 
                    MGB_AUC1 = MGB_AUC_origin[0][0], VA_AUC1 = VA_AUC_origin[0][0], UP_AUC1 = UP_AUC_origin[0][0], 
                    MGB_AUC0 = MGB_AUC_origin[1][0], VA_AUC0 = VA_AUC_origin[1][0], UP_AUC0 = UP_AUC_origin[1][0],
                    ONLY_SIMI=True)

                # if (PRE_origin[0] > want_TOP1) we will store the model and embedding
                case_store = (PRE_origin[0] >= best_PRE_0) & (PRE_origin[0] >= want_TOP1) 
                if PRE_origin[0] > best_PRE_0:
                    best_PRE_0 = PRE_origin[0]
                if case_store:
                    torch.save(x1, f'/root/current_code/GAT_model_9_22/output/{start_time}/Final_emb_1.pth')
                    torch.save(model.state_dict(), 
                               f'/root/current_code/GAT_model_9_22/output/{start_time}/model_1.pth') 

            print(f'SIMI Epoch{epoch} has finished...')
            scheduler.step()    
            torch.save(x1, f'/root/current_code/GAT_model_9_22/output/{start_time}/Final_emb_epoch{epoch}_1.pth')
            torch.save(model.state_dict(),
                       f'/root/current_code/GAT_model_9_22/output/{start_time}/model_epoch{epoch}_1.pth') 

    # load GAT model
    if ATTENTION_TYPE == 'A':
        model = GAT_A(in_features=config['num_features'], hidden_features=config['hidden_features2'],
                      out_features=config['all_dim'], K=config['K'], low_dim=config['low_dim'])
    else:
        model = GATModel(config['num_features'], config['hidden_features2'])

    print('************************original model************************')
    print(model)   
    model.load_state_dict(torch.load(f"/root/current_code/GAT_model_9_22/output/{config['origin_start_time']}/model_1.pth"))
    model.reset_parameters_part()
    print('----------------check all parameters----------------')
    print(model.gat1.gat_conv.lin_src_new.weight)
    print(model.gat1.gat_conv.att_dst_new)
    print(model.gat1.gat_conv.att_src_new)
    print(model.gat2.gat_conv.lin_src_new.weight)
    print(model.gat2.gat_conv.att_dst_new)
    print(model.gat2.gat_conv.att_src_new)
    print(model.linear.weight)
    x_sim = torch.load(f"/root/current_code/GAT_model_9_22/output/{config['origin_start_time']}/Final_emb_1.pth")
    PRE_origin, PRE_0_origin, RELA_MGB_AUC_origin, RELA_VA_AUC_origin, RELA_UP_AUC_origin, SIMI_MGB_origin, SIMI_VA_origin, SIMI_UP_origin = test(x_sim[:,0:config['rmax']], name_all, related_pairs= val_rel_pairs, similar_pairs= ALL_sim_val_pairs, PRE = True, AUC = True, AUC_type = True, LEVEL=[0,1], rmax=config['rmax'])
    MGB_AUC_origin = [RELA_MGB_AUC_origin, SIMI_MGB_origin]
    VA_AUC_origin = [RELA_VA_AUC_origin, SIMI_VA_origin]
    UP_AUC_origin = [RELA_UP_AUC_origin, SIMI_UP_origin] 
    
    write_file(0, 0, pre = PRE_origin, pre0 = PRE_0_origin, 
        MGB_AUC1 = MGB_AUC_origin[0][0], VA_AUC1 = VA_AUC_origin[0][0], UP_AUC1 = UP_AUC_origin[0][0], 
        MGB_AUC0 = MGB_AUC_origin[1][0], VA_AUC0 = VA_AUC_origin[1][0], UP_AUC0 = UP_AUC_origin[1][0], 
        begin = True)

    sampler = MySampler(unique_name, name_all, pd.Series(name_new, name='V1'))  
    optimizer = optim.SGD(model.parameters(), lr=config['base_lr'])
    scheduler = CustomExponentialLR(optimizer, gamma=config['gamma'], min_lr=1e-5)
    
    # train the model, second step: relatedness
    for epoch in range(1, config['epochs']+1):       
        undirected_edge_index = undirected_edge_index2
        
        print('------------------------Epoch: {:03d}-----------------------'.format(epoch))
        n_batchs = sampler.__len__()

        for i in range(n_batchs):
            optimizer.zero_grad()
            ind = sampler.__iter__()
            batch_indices = list(ind)
            x1, attention1, attention2, rank_attention1, rank_attention2 = model(x_tensor, undirected_edge_index, edge_attention)
            append_to_csv('attention1', attention1)
            append_to_csv('attention2', attention2)

            (P_LOSS_one_one, N_LOSS_one_one, 
            P_LOSS_hie, N_LOSS_hie, 
            P_LOSS_OTOL, N_LOSS_OTOL, 
            P_LOSS_LTOL, N_LOSS_LTOL, 
            P_LOSS_REL, N_LOSS_REL,
            P_LOSS_SIM_NO_HIE, N_LOSS_SIM_NO_HIE,
            P_LOSS_SAP1, P_LOSS_SAP2, 
            P_LOSS_SVD1, P_LOSS_SVD2, 
            LOSS_attention) = custom_loss(my_objects, x1, batch_indices, device, name_all, edge_attention, rank_attention1, rank_attention2, true_rank, COS_origin_sap, COS_origin_svd)

            loss = (P_LOSS_one_one + N_LOSS_one_one + 
                    P_LOSS_hie + N_LOSS_hie + 
                    P_LOSS_OTOL + N_LOSS_OTOL + 
                    P_LOSS_LTOL + N_LOSS_LTOL + 
                    P_LOSS_REL + N_LOSS_REL + 
                    P_LOSS_SIM_NO_HIE + N_LOSS_SIM_NO_HIE + 
                    P_LOSS_SAP1 + P_LOSS_SAP2 + 
                    P_LOSS_SVD1 + P_LOSS_SVD2 + 
                    LOSS_attention)
            print('batch_num: {:03d}     LOSS: {:.4f}'.format(i, loss))
            loss.backward() 
            mask_tensor(model.gat2.gat_conv.lin_src_new.weight, model.gat2.gat_conv.lin_src_new.weight.mask2)
            mask_tensor(model.linear.weight, model.linear.weight.mask2)
            optimizer.step()
            scheduler.step()
            
            print('----------------check all parameters----------------')
            print(model.gat1.gat_conv.lin_src_new.weight)
            print(model.gat1.gat_conv.att_dst_new)
            print(model.gat1.gat_conv.att_src_new)
            print(model.gat2.gat_conv.lin_src_new.weight)
            print(model.gat2.gat_conv.att_dst_new)
            print(model.gat2.gat_conv.att_src_new)
            print(model.linear.weight)
            print('------------------check embedding-------------------')
            print(x1.shape)
            print(x1)
            
            PRE_origin, PRE_0_origin, RELA_MGB_AUC_origin, RELA_VA_AUC_origin, RELA_UP_AUC_origin, SIMI_MGB_origin, SIMI_VA_origin, SIMI_UP_origin = test(x1, name_all, related_pairs= val_rel_pairs, similar_pairs= ALL_sim_val_pairs, PRE = True, AUC = True, AUC_type = True, LEVEL=[0,1])

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
                my_item(P_LOSS_SVD1), my_item(P_LOSS_SVD2),
                my_item(LOSS_attention)], 
                pre = PRE_origin, pre0 = PRE_0_origin, 
                MGB_AUC1 = MGB_AUC_origin[0][0], VA_AUC1 = VA_AUC_origin[0][0], UP_AUC1 = UP_AUC_origin[0][0], MGB_AUC0 = MGB_AUC_origin[1][0], VA_AUC0 = VA_AUC_origin[1][0], UP_AUC0 = UP_AUC_origin[1][0])

            # if (PRE_origin[0] > want_TOP1) we will store the model and embedding
            case_store = (PRE_origin[0] >= best_PRE_0) & (PRE_origin[0] >= want_TOP1) 
            if PRE_origin[0] > best_PRE_0:
                best_PRE_0 = PRE_origin[0]
            if case_store:
                torch.save(x1, f'/root/current_code/GAT_model_9_22/output/{start_time}/Final_emb_1_with_rel.pth')
                torch.save(model.state_dict(), 
                            f'/root/current_code/GAT_model_9_22/output/{start_time}/model_1_with_rel.pth')   


        print(f'Related Epoch{epoch} has finished...')
        torch.save(x1, f'/root/current_code/GAT_model_9_22/output/{start_time}/Final_emb_epoch{epoch}_1_with_rel.pth')
        torch.save(model.state_dict(),
                    f'/root/current_code/GAT_model_9_22/output/{start_time}/model_epoch{epoch}_1_with_rel.pth') 


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