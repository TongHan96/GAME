# README

## Overview
This README provides instructions for running the training process in both encoder part and decoder part.

In the encoder part, there are three parts. To align institutions using PPMI embedding, obtain similarity embedding using similarity edges and loss function, and acquire relatedness embedding using relatedness edges and loss functions.

<p align="center">
  <img src="https://github.com/TongHan96/GAME/blob/main/records/pic/alg.drawio.png" alt="alg" title="alg" width="1000"/>
</p>

## Part 1: Aligning Institutions with PPMI Embedding

**Python Code:**
```python
import os
os.chdir('/home/doz128/GAME_model1/src')  # Change directory to your working path
%run main.py --want_TOP1 62.0 --path '/home/doz128/GAME_model1/' --epochs 500 --drop_out 0.5 --scale_sppmi 0.1 --lr 1e-4 --hidden_features 768 --DEVICE 'cuda' --EGDE_ALL True --path_origin 'align_NA'
```

## Part 2: Obtaining Similarity Embedding

**Python Code:**
```python
import os
os.chdir('/home/doz128/GAME_model1/src')  # Change directory to your working path
%run main.py --want_TOP1 62.0 --path '/home/doz128/GAME_model1/' --epochs 500 --drop_out 0.5 --lr 5e-6 --hidden_features 768 --DEVICE 'cuda' --EDGE_ALL True
```

## Part 3: Acquiring Relatedness Embedding

**Python Code:**
```python
import os
os.chdir('/home/doz128/GAME_model1/src')  # Change directory to your working path
%run main.py --want_TOP1 62.0 --path '/home/doz128/GAME_model1/' --epochs 500 --drop_out 0.5 --scale_sppmi 0.1 --lr 5e-6 --hidden_features 768 --DEVICE 'cuda' --EDGE_ALL False --path_origin '2024-02-24 07:09:59'
```

## GAME Training Script Configuration

This section outlines the various configuration options available for the GAME Training Script. These options allow for customizing the training process, model performance thresholds, and other operational parameters.

**Command-Line Arguments:**

- `--want_TOP1`: Top1 performance threshold for storing model and embedding. *Default: 77*.
- `--want_TOP20`: Top20 performance threshold for storing model and embedding. *Default: 93*.
- `--EDGE_ALL`: Whether to use all edges to train similarity. Accepts 'True' or 'False'. *Default: True*.
- `--drop_out`: Dropout probability for training. *Default: 0.0*.
- `--lr`: Learning rate for the optimizer. *Default: 0.001 (1e-3)*.
- `--AA`: Parameter AA, related to the model's architecture or training process. *Default: 1.0*.
- `--BB`: Parameter BB, also related to the model's architecture or training process. *Default: 5.0*.
- `--lambd`: Parameter lambda for the loss function. *Default: 0.5*.
- `--scale_one_one`: Scaling factor for one-one alignment. *Default: 1*.
- `--scale_hie`: Scaling factor for hierarchical alignment. *Default: 1*.
- `--scale_sppmi`: Scaling factor for SPPMI (Shifted Positive Pointwise Mutual Information). *Default: 100*.
- `--scale_OTOL`: Scaling factor for one-to-one alignment. *Default: 50*.
- `--scale_REL`: Scaling factor for relevance alignment. *Default: 5*.
- `--scale_align`: Scaling factor for alignment. *Default: 1*.
- `--rmax`: Maximum radius for similarity calculations. *Default: 256*.
- `--hidden_features`: Number of hidden features in the model. *Default: 768*.
- `--path`: Specify the path for the project or model. Utilizes `config['path']` by default.
- `--input_dir`: Specify the directory for input data. Utilizes `config['input_dir']` by default.
- `--path_origin`: Specify the origin path for alignment or model training. *Default: config['path_origin']*.
- `--epochs`: Total number of epochs for training. *Default: 3*.
- `--CHECK_ALL`: Option to check all attention mechanisms during training. Accepts 'True' or 'False'. *Default: False*.
- `--DEVICE`: Specify the device for training, such as 'cuda:0' for GPU. *Default: 'cuda:0'*.
- `--num_inst`: Specify the number of institutions for training. In our project, we use MGB, VA, UPMC and BCH institutional data. *Default: 4*.

**Additional Configuration:**

- `want_TOP1` and `want_TOP20`: Set the desired performance thresholds for model validation and storage.
- `path`, `input_dir`, and `path_origin`: These parameters help specify various paths critical for the training process, including data input and model initialization.
- `epochs`, `lr`, `AA`, `BB`, `lambd`, and `drop_out`: These settings allow fine-tuning the training process, including how long the training runs and how the model learns.
- `DEVICE`: Configures the hardware to be used for training, allowing for GPU acceleration if available.
- `scale_*`: Various scaling factors used to adjust the influence of different components of the model or training process.

Use these configurations to customize the GAME training script to suit your specific needs and hardware setup.
