# Representation Learning to Advance Multi-institutional Studies with Electronic Health Record Data

This repository contains the implementation of **GAME (Graph-based Alignment for Multi-institutional Embeddings)**, a framework designed to align multi-institutional codes within Electronic Health Records (EHR) data using a sophisticated knowledge graph system. Below is a detailed guide to help you understand and implement the GAME training process.

---

## Overview of the GAME Training Process

GAME comprises three main steps to align institutional embeddings from *M* institutions into a shared space. The process is illustrated below:

<p align="center">
  <img src="https://github.com/TongHan96/GAME/blob/main/pic/alg.png" alt="GAME Algorithm" title="GAME Algorithm" width="1000"/>
</p>

### **1. Aligning Institutions with PPMI Embedding**
- **Objective**: Align different institutions using Positive Pointwise Mutual Information (PPMI) embeddings.
- **Input**: PPMI embeddings \( V_i \) for each institution \( i \in \{1, 2, \ldots, M\} \).
- **Process**: Apply an institutional graph attention network to obtain aligned embeddings \( Y_i \).
- **Output**: A common representation \( Y \) that captures co-occurrence probabilities across institutions.

### **2. Generating Main Embeddings**
- **Objective**: Generate main embeddings using edges and a loss function.
- **Input**: Concatenation of SAPBERT embedding \( \mathbf{X} \) and aligned institutional embedding \( \mathbf{Y} \).
- **Process**: Optimize embeddings using a multi-similarity loss function.
- **Output**: A 256-dimensional embedding (\( r_{\text{max}} = 256 \)) suitable for similarity tasks.

### **3. Creating Relatedness Tail Embeddings**
- **Objective**: Generate relatedness tail embeddings \( \mathbf{Z}_{\mathcal{R}} \) for complex relatedness tasks.
- **Input**: All edges and main embeddings.
- **Process**: Concatenate relatedness tail embeddings with the main embedding.
- **Output**: A final 768-dimensional embedding \( \mathbb{Z} \).

---

## Repository Structure

The repository is organized as follows:

```terminal
GAME/
├── output/                  # Contains training results and outputs
├── readme.md               # This README file
├── src/                    # Source code for the GAME framework
│   ├── Attention.py        # Graph attention network implementation
│   ├── config.py           # Configuration settings
│   ├── data_structure.py   # Data structures and utilities
│   ├── evaluate.py         # Evaluation scripts
│   ├── load_data.py        # Data loading utilities
│   ├── main.py             # Main training script
│   └── utils.py            # Utility functions
└── supp_code/              # Supplementary code and downstream tasks
    └── feature_selection/  # Feature selection results and inputs
        ├── input/          # GPT scores for feature selection
        └── score_all/      # Feature selection results (GPT + cosine similarities)
```

---

## Implementation Guide

### **Step 1: Aligning Institutions with PPMI Embedding**

**Python Code:**
```python
import os
os.chdir('~/GAME/src')  # Change to the source directory
%run main.py --path '~/GAME/' --drop_out 0.1 --lr 1e-4 --DEVICE 'cuda' --path_origin 'align_NA' --api_key '********'  # Replace with your OpenAI API key
```

### **Step 2: Obtaining the Similarity Embedding**

**Python Code:**
```python
import os
os.chdir('~/GAME/src')
%run main.py --path '~/GAME/' --lr 1e-6 --drop_out 0.5 --DEVICE 'cuda' --align_path '[your_align_step_result_folder]' --api_key '********'  # Replace with your OpenAI API key
```

### **Step 3: Acquiring the Relatedness Embedding**

**Python Code:**
```python
import os
os.chdir('~/GAME/src')
%run main.py --drop_out 0.5 --lr 1e-6 --DEVICE 'cuda' --path_origin '[your_sim_step_result_folder]' --align_path '[your_align_step_result_folder]' --api_key '********'  # Replace with your OpenAI API key
```

---

## GAME Training Script Configuration

The GAME training script supports the following command-line arguments for customization:

### **Basic Settings**
- `--num_inst`: Number of institutions for training (default: `7`).
- `--path`: Path to the project or model (default: `config['path']`).
- `--input_dir`: Directory containing input data (default: `config['input_dir']`).
- `--path_origin`: Specifies the training step:
  - `align_NA`: Train the alignment PPMI step.
  - `None`: Train the similarity step.
  - Non-`None` value: Train the relatedness step (default: `config['path_origin']`).
- `--align_path`: Path to pre-trained aligned SPPMI embeddings (default: `None`).

### **Loss Function Parameters**
- `--AA`: Model-specific parameter (default: `1.0`).
- `--BB`: Model-specific parameter (default: `5.0`).
- `--lambd`: Lambda parameter for the loss function (default: `0.5`).

### **Scaling Factors for Loss Components**
- `--scale_hie`: Scaling factor for hierarchical alignment loss (default: `1`).
- `--scale_OTOL`: Scaling factor for one-to-one alignment loss (default: `50`).
- `--scale_REL`: Scaling factor for relevance alignment loss (default: `5`).
- `--scale_sppmi`: Scaling factor for PPMI feature selection loss (default: `0.1`).
- `--scale_align`: Scaling factor for alignment loss (default: `1`).

### **Dimensionality Specifications**
- `--rmax`: Maximum dimensionality for similarity embeddings (default: `256`).
- `--out_dim`: Output dimensionality of the final embedding (default: `768`).
- `--hidden_features`: Number of hidden features in the model (default: `768`).

### **Training Configuration**
- `--drop_out`: Drop edge probability during training (default: `0.0`).
- `--lr`: Learning rate for the optimizer (default: `1e-4`).
- `--epochs`: Number of training epochs (default: `3`).
- `--DEVICE`: Device for training, e.g., `'cuda:0'` for GPU (default: `'cuda:0'`).

### **Evaluation and Debugging**
- `--CHECK_ALL`: Enable checking all attention mechanisms during training (default: `False`).

### **API Configuration**
- `--api_key`: OpenAI API key for scoring results (default: `None`).

---

