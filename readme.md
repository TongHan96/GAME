

# GAME: Self-supervised Graph Alignment for Multi-institutional Collaboration with Electronic Health Records Data

## Overview of the GAME Training Process

This README provides a concise guide for implementing the GAME training process, designed to align multi-institutional codes within Electronic Health Records (EHR) data using a sophisticated knowledge graph system. The following is a simplified overview to facilitate understanding and implementation:

<p align="center">
  <img src="https://github.com/TongHan96/GAME/blob/main/report/pic/alg.png" alt="Algorithm" title="Algorithm" width="1000"/>
</p>

GAME comprises three main steps to align institutional embeddings from $M$ institutions into a shared space.

1. **Aligning Institutions with PPMI Embedding**:
   - Utilizes Positive Pointwise Mutual Information (PPMI) embeddings, denoted as $V_i, i \in 1, 2, \ldots, M$, to align different institutions. After applying the institutional graph attention network, we obtain $Y_i, i \in 1, 2, \ldots, M$, which helps establish a common representation $Y$ that captures co-occurrence probabilities. This step is crucial for identifying similarities and relationships across institutions.

2. **Generating Main Embeddings**:
   - Main embeddings are generated using edges and a loss function, where edges represent connections based on similarity and relatedness between entities. The initial input embedding is a concatenation of the SAPBERT embedding $\mathbf{X}$ and the aligned institutional embedding $\mathbf{Y}$ obtained in the first step. By default, $r_{\text{max}}=256$, indicating a 256-dimensional embedding. The multi-similarity loss function optimizes the embeddings, enabling similar entities to be positioned closely in the embedding space.

3. **Creating Relatedness Tail Embeddings**:
   - This step generates relatedness tail embeddings $\mathbf{Z}_{\mathcal{R}}$ using all edges, which are then concatenated with the main embedding to form the final embedding (out_dim = $768$), $\mathbb{Z}$. The main embedding is well-suited for similarity tasks, while the final embedding is designed to handle more complex relatedness tasks.

## Implementing the Process

### Part 1: Aligning Institutions with PPMI Embedding

**Python Code:**
```python
import os
os.chdir('~/GAME/src')  # Change the directory to your working path
%run main.py --path '~/GAME/' --epochs 500 --drop_out 0.1 --scale_sppmi 0.1 --lr 1e-4 --hidden_features 768 --DEVICE 'cuda' --EDGE_ALL --path_origin 'align_NA' --api_key '********'  # Replace api_key with your OpenAI API key
```

### Part 2: Obtaining the Similarity Embedding

**Python Code:**
```python
import os
os.chdir('~/GAME/src')  # Change the directory to your working path
%run main.py --path '~/GAME/' --rmax 256 --epochs 500 --scale_OTOL 30 --drop_out 0.5 --scale_sppmi 0.1 --lr 1e-6 --hidden_features 768 --DEVICE 'cuda' --EDGE_ALL --api_key '********' --align_path 'align_step'  # Set align_path to the folder containing the align step output.
```

### Part 3: Acquiring the Relatedness Embedding

**Python Code:**
```python
import os
os.chdir('~/GAME/src')  # Change the directory to your working path
%run main.py --path '~/GAME/' --epochs 500 --drop_out 0.5 --EDGE_ALL --scale_sppmi 0.1 --lr 1e-6 --hidden_features 768 --DEVICE 'cuda' --api_key '********' --path_origin 'sim_step' --align_path 'align_step'  # Set path_origin to the folder containing the similarity step output.
```


## GAME Training Script Configuration

This section outlines the various configuration options available for the GAME Training Script. These options allow for customizing the training process, model performance thresholds, and other operational parameters.




**Command-Line Arguments:**

*Basic Settings*
- `--num_inst`: Specifies the number of institutions for training. For this project, the default is *7*.
- `--path`: Sets the path to the project or model. The default is `config['path']`.
- `--input_dir`: Defines the directory containing input data. The default is `config['input_dir']`.
- `--path_origin`: Determines the training step:
  - If set to `align_NA`, trains the alignment PPMI step.
  - If set to `None`, trains the similarity step.
  - If set to a non-`None` value (the path to the similarity embedding), trains the relatedness step. 
  - *Default*: `config['path_origin']`.
- `--align_path`: Specifies the path to the pre-trained aligned SPPMI embeddings for similarity and relatedness steps. *Default*: `None`.

*Loss Function Parameters*
- `--AA`: A model-specific parameter related to the architecture or training process. *Default*: `1.0`.
- `--BB`: Another model-specific parameter related to the architecture or training process. *Default*: `5.0`.
- `--lambd`: The lambda parameter for the loss function. *Default*: `0.5`.

*Scaling Factors for Loss Components*
- `--scale_hie`: Scaling factor for hierarchical alignment loss. *Default*: `1`.
- `--scale_OTOL`: Scaling factor for one-to-one alignment loss. *Default*: `50`.
- `--scale_REL`: Scaling factor for relevance alignment loss. *Default*: `5`.
- `--scale_sppmi`: Scaling factor for PPMI feature selection loss. *Default*: `0.1`.
- `--scale_align`: Scaling factor for alignment loss. *Default*: `1`.

*Dimensionality Specifications*
- `--rmax`: Maximum dimensionality for similarity embeddings. *Default*: `256`.
- `--out_dim`: Output dimensionality of the final embedding. *Default*: `768`.
- `--hidden_features`: Number of hidden features in the model. *Default*: `768`.

*Training Configuration*
- `--EDGE_ALL`: Specifies whether to use all edges or only similar/related edges for training. Accepts `True` or `False`. *Default*: `False`.
- `--drop_out`: Dropout probability used during training. *Default*: `0.0`.
- `--lr`: Learning rate for the optimizer. *Default*: `1e-4`.
- `--epochs`: Number of epochs for training. *Default*: `3`.
- `--DEVICE`: Device used for training, e.g., 'cuda:0' for GPU. *Default*: `'cuda:0'`.

*Evaluation and Debugging*
- `--CHECK_ALL`: Option to enable checking all attention mechanisms during training. Accepts `True` or `False`. *Default*: `False`.

*API Configuration*
- `--api_key`: OpenAI API key used for scoring the results. *Default*: `None`.

