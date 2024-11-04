# GAME: Self-supervised Graph Alignment for Multi-institutional Collaboration with Electronic Health Records Data
## Overview of the GAME Training Process
This README offers a concise guide for implementing the GAME training process, which is designed to align multi-institutional codes within Electronic Health Records (EHR) data through a sophisticated knowledge graph system. Here's a simplified overview to facilitate understanding and implementation:


<p align="center">
  <img src="https://github.com/TongHan96/GAME/blob/main/report/pic/alg.png" alt="alg" title="alg" width="1000"/>
</p>

GAME comprises three main steps to align institutional embeddings from $M$ instititutions to the shared space.

1. **Aligning Institutions with PPMI Embedding**:
   - Utilizes Positive Pointwise Mutual Information (PPMI) embeddings, denoted as $V_i, i \in 1,2,..,M$, to align different institutions. After applying the institutional graph attention network, we obtain $Y_i, i \in 1,2,..,M$, and then we can establish a common representation $Y$ that captures co-occurrence probabilities. This step is essential for identifying similarities and relationships across institutions.

2. **Generating Main Embedding**:
   - Main embeddings are generated using edges and a loss function, where edges denote connections based on similarity and relatedness between entities. The initial input embedding is the concatenation of SAPBERT embedding $\mathbf{X}$ and the institutional aligned embedding $\mathbf{Y}$ obtained in the first step. With a default configuration of $rmax=256$, indicating a 256-dimensional embedding, \mathbf{Z}_{\mathcal{S}}, a multi-similarity loss function optimizes the embeddings. This dimensionality is adequate for positioning similar entities closely in the embedding space.

3. **Creating Relatedness Tails Embedding**:
   - This step involves generating relatedness tails embeddings $\mathbf{Z}_{\mathcal{R}}$ using all edges, then concatenating them with the main embedding to form the final embedding ($out_dim=768$), $\mathbb{Z}$. The main embedding efficiently handles similarity tasks, whereas the final embedding addresses more complex relatedness tasks.

## Implementing the Process


### Part 1: Aligning Institutions with PPMI Embedding

**Python Code:**
```python
import os
os.chdir('~/GAME/src')  # Change directory to your working path
%run main.py --path '~/GAME/' --epochs 500 --drop_out 0.1 --scale_sppmi 0.1 --lr 1e-4 --hidden_features 768 --DEVICE 'cuda' --EDGE_ALL --path_origin 'align_NA' --api_key '********'  # Change api_key to your openai api_key
```

### Part 2: Obtaining Similarity Embedding

**Python Code:**
```python
import os
os.chdir('~/GAME/src')  # Change directory to your working path
%run main.py --path '~/GAME/' --rmax 256 --epochs 500 --scale_OTOL 70 --drop_out 0.5 --scale_sppmi 0.1 --lr 1e-6 --hidden_features 768 --DEVICE 'cuda' --EDGE_ALL --api_key '********' --align_path 'align_step'  # Change api_key to your openai api_key
```

### Part 3: Acquiring Relatedness Embedding

**Python Code:**
```python
import os
os.chdir('~/GAME/src')  # Change directory to your working path
%run main.py --path '~/GAME/' --epochs 500 --drop_out 0.5 --EDGE_ALL --scale_sppmi 0.1 --lr 1e-6 --hidden_features 768 --DEVICE 'cuda' --path_origin 'sim_step' --align_path 'align_step' --api_key '********'   # Change api_key to your openai api_key
```

## GAME Training Script Configuration

This section outlines the various configuration options available for the GAME Training Script. These options allow for customizing the training process, model performance thresholds, and other operational parameters.


**Command-Line Arguments:**

*Basic Settings*
- `--num_inst`: Specifies the number of institutions for training. In your project, you have 7 institutions. *Default: 7*.
- `--path`: Sets the project or model path. Defaults to `config['path']`.
- `--input_dir`: Defines the directory for input data. Defaults to `config['input_dir']`.
- `--path_origin`: Sets the original path for alignment or model training. *Default: config['path_origin']*.

*Used in Loss*
- `--AA`: Parameter AA, pertaining to the model's architecture or training process. *Default: 1.0*.
- `--BB`: Parameter BB, also related to the model's architecture or training process. *Default: 5.0*.
- `--lambd`: The lambda parameter for the loss function. *Default: 0.5*.

*Used in Loss Scales*
- `--scale_one_one`: Scaling factor for one-to-one alignment. *Default: 10*.
- `--scale_hie`: Scaling factor for hierarchical alignment. *Default: 1*.
- `--scale_sppmi`: Scaling factor for SPPMI (Shifted Positive Pointwise Mutual Information). *Default: 0.1*.
- `--scale_OTOL`: Scaling factor for one-to-one alignment. *Default: 50*.
- `--scale_REL`: Scaling factor for relevance alignment. *Default: 5*.
- `--scale_align`: Scaling factor for alignment. *Default: 1*.

*Used to Determine Dimension*
- `--rmax`: Maximum radius for similarity calculations. *Default: 256*.
- `--out_dim`: Output dimension of the final embedding in the model. *Default: 768*.
- `--hidden_features`: Number of hidden features in the model. *Default: 768*.

*Training Process*
- `--EDGE_ALL`: Indicates whether to use all edges for training similarity. Accepts 'True' or 'False'. *Default: False*.
- `--drop_p`: Dropout probability during training. *Default: 0.0*.
- `--base_lr`: Learning rate for the optimizer. *Default: 1e-4*.
- `--epochs`: Total number of epochs for training. *Default: 3*.
- `--DEVICE`: Specifies the device for training, such as 'cuda:0' for GPU. *Default: 'cuda:0'*.

*Check and Store*
- `--CHECK_ALL`: Option to check all attention mechanisms during training. Accepts 'True' or 'False'. *Default: False*.
Use these configurations to customize the GAME training script to suit your specific needs and environments setup.
