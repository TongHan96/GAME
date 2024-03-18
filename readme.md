# README

## Overview
This README provides instructions for running the training process in three parts to align institutions using PPMI embedding, obtain similarity embedding using similarity edges and loss function, and acquire relatedness embedding using relatedness edges and loss functions.

## Part 1: Aligning Institutions with PPMI Embedding

**Python Code:**
```python
import os
os.chdir('/home/doz128/GAME_align/src')  # Change directory to your working path
%run main.py --want_TOP1 62.0 --path '/home/doz128/GAME_align' --epochs 100 --lr 1e-5 --drop_out 0.3 --DEVICE 'cpu' --path_origin 'align_NA'
```

<!-- **examples output: 2024-01-09 07:34:23** -->
## Part 2: Obtaining Similarity Embedding

**Python Code:**
```python
import os
os.chdir('/home/doz128/GAME_align/src')  # Change directory to your working path
%run main.py --want_TOP1 62.0 --path '/home/doz128/GAME_align' --epochs 500 --lr 2e-5 --AA 0.5 --BB 0.5 --lambd 0.9 --drop_out 0.3 --DEVICE 'cuda:0'
```
<!-- 2024-01-09 08:02:22 : sap+align+1GAT
2024-01-09 08:14:40 : align+sap+1GAT
2024-01-09 08:21:53 : sap+align+2GAT
2024-01-09 08:23:00 : align+sap+2GAT 
2024-01-09 09:00:34 : align+sap+1GAT+edge_all-->


<!-- MAKE INITIAL SPPMI PART WEIGHT TO BE 0 sap+align+gat
2024-01-09 21:37:11 : 1e-5 EDGE_ALL 0.5,0.5,0.9 0.99
2024-01-09 21:59:28 : 1e-6 EDGE_ALL 0.5,0.5,0.9 0.99
2024-01-09 22:46:32 : 1e-6 EDEG_SIM 0.5,0.5,0.9 0.99
2024-01-10 01:02:27 : 1e-6 EDGE_SIM 0.5,0.5,0.8 0.99
2024-01-10 01:14:32 : 5e-6 EDGE_SIM 0.5,0.5,0.9 0.95
2024-01-10 01:19:51 : 5e-6 EDGE_ALL 0.5,0.5,0.9 0.95
2024-01-10 01:22:25 : 5e-6 EDGE_SIM 1,1,0.9 0.95
2024-01-10 01:23:59 : 5e-6 EDGE_ALL 1,1,0.9 0.95 -->

<!-- 2024-01-10 01:43:19 : 5e-6 EDGE_REL 0.5,0.5,0.9,0.95 -->



## Part 3: Acquiring Relatedness Embedding

**Python Code:**
```python
import os
os.chdir('/home/doz128/GAME_align/src')  # Change directory to your working path
%run main.py --want_TOP1 62.0 --path '/home/doz128/GAME_align' --epochs 500 --lr 2e-5 --AA 0.5 --BB 0.5 --lambd 0.9 --drop_out 0.3 --DEVICE 'cuda:1' --path_origin "trained_sim"
```

**Additional Configuration:**
- `want_TOP1`: Set the desired TOP1 value.
- `path`: Specify the project path.
- `epochs`: Define the number of training epochs.
- `lr`: Set the learning rate.
- `AA` and `BB`: Set parameters for relatedness embedding.
- `lambd`: Set the lambda value for the loss function.
- `drop_out`: Specify the dropout rate.
- `DEVICE`: Choose the GPU device ('cuda:0' or 'cuda:1' or other options).
- `path_origin`: Specify the path to the trained similarity model.

Please make sure to configure the parameters according to your specific needs before running the code.