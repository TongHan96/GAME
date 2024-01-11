# Result Summary

## Changes
1. **Sapbert/coder embedding**: generated from their descriptions of their descriptions+LP code descriptions
2. We have also changed the **accuracy calculation method** to use both LOINC and LP codes.
3. The **rotation mode for sppmi** has changed: VA and UPMC revolve around MGB.
4. The **construction of Latent Nodes** has changed: Use concatenate of three inst. embeddings; Use rotation to pad the embeddings for missing codes in some inst. Then we use them to do feature selection.
5. Use **edges generated from sppmi emb** in related training part. **Really helpful for feature selection!**


## Code Mapping Job
### Sapbert embedding
**Accuracy Rates:**  
| Measure | short | dict| full | gat(256) |
| --- | --- |--- | --- | --- |
| TOP1 |52.7%| 55.6%|61.1% | 75.6% |
| TOP5 | 67.0%| 71.0%|79.6% | 85.2% |
| TOP10 |71.3%| 74.9%| 83.3% | 87.6% |
| TOP20 |75.1%| 78.4%| 86.3% | 89.9% |



 
## Feature Selection
### GPT4 Scoring & Cos Sim Rank Correlation:
#### RA
| Method          | 768*3 Correlation| 768 Correlation|
|-----------------|-------|-------|
| coder           | 0.303 | 0.287|
| sapbert         | 0.189 | 0.180|
| sppmi           | 0.592 | 0.580|
| gat(sap)        | 0.500 | 0.526|


#### CHF
| Method          | 768*3 Correlation| 768 Correlation|
|-----------------|-------|-------|
| coder           | 0.443 | 0.451|
| sapbert         | 0.565 | 0.569|
| sppmi           | 0.750 | 0.749|
| gat(sap)        | 0.759 | 0.763|

#### Depression

| Method          | 768*3 Correlation| 768 Correlation|
|-----------------|-------|-------|
| coder           | 0.302 | 0.301|
| sapbert         | -0.003 | 0.007|
| sppmi           | 0.724 | 0.718|
| gat(sap)        | 0.805 | 0.800|

#### Type 1 Diabetes

| Method          | 768*3 Correlation| 768 Correlation|
|-----------------|-------|-------|
| coder           | 0.427 | 0.426|
| sapbert         | 0.407 | 0.412|
| sppmi           | 0.469 | 0.497|
| gat(sap)        | 0.528 | 0.510|

## Detect Drug Side Effect (Using Concatenate Emb) 
| Method          | AUC (768)  | After Supervised (768) | Before Reduct Dimension (768*3)  |
|-----------------|-------|-------------------|--------------------------|
| coder           | 0.540 | 0.810             ||
| sapbert         | 0.578 | 0.797             ||
| sppmi           | 0.515 | 0.751             | 0.568, 0.744|
| GAT             | 0.625 | 0.770             | 0.626, 0.827| 


## Detect Drug Side Effect (Using MGB part Emb) 
| Method          | AUC (768)  | After Supervised (768) |
|-----------------|-------|-------------------|
| coder           | 0.541 | 0.810             |
| sapbert         | 0.578 | 0.793             |
| sppmi           | 0.563 | 0.751             |
| GAT             | 0.623 | 0.811             |


## Detect Drug Side Effect (dim = 768)

| Method                   | MGB   | VA    | UPMC  |
|--------------------------|-------|-------|-------|
| Coder                  | 0.544 | 0.546 | 0.541|
| Coder + supervised     | 0.821 |  0.814 |0.831|
| ---                      | ---   | ---   | ---   |
| SAPBERT                  | 0.579 | 0.578 | 0.577 |
| SAPBERT + supervised     | 0.806 | 0.794 | 0.810 |
| ---                      | ---   | ---   | ---   |
| SVD                      | 0.573 | 0.545 | 0.558 |
| SVD + supervised         | 0.763 | 0.745 | 0.727 |
| ---                      | ---   | ---   | ---   |
| SAP_gat                  | 0.624 | 0.624 | 0.627 |
| SAP_gat + supervised     | 0.827 | 0.807 | 0.813 |
<!--| ---                      | ---   | ---   | ---   |
| SAP_gat_Attention_h^{T}Bh                  | 0.618 | 0.613 | 0.619 |
| SAP_gat_Attention_h^{T}Bh + supervised     | 0.813 | 0.796 | 0.794 |
| ---                      | ---   | ---   | ---   |
| SAP_gat_Attention_low1                  | 0.635 | 0.630 | 0.633 |
| SAP_gat_Attention_low1 + supervised     | 0.816 | 0.803 | 0.801 |
| ---                      | ---   | ---   | ---   |
| SAP_gat_Attention_low2                  | 0.622 | 0.621 | 0.623 |
| SAP_gat_Attention_low2 + supervised     | 0.808 | 0.793 | 0.801 |
| ---                      | ---   | ---   | ---   |
| SAP_gat_no_sppmi                  | 0.636 | 0.628 | 0.631 |
| SAP_gat_no_sppmi + supervised     | 0.834 | 0.819 | 0.812 |
| ---                      | ---   | ---   | ---   |
| SAP_gat_no_sppmi_edge    | 0.629 | 0.622 | 0.627 |
| SAP_gat_no_sppmi_edge  + supervised | 0.836 | 0.820 | 0.813 |-->


## Feature Selection
### GPT4 Scoring & Cos Sim Rank Correlation:




#### RA
| Method          | Correlation|
|-----------------|-------|
| coder           | 0.203 |
| sapbert         | 0.022 |
| sppmi           | 0.445 |
| GAME        | 0.441 |

#### CHF
| Method          | Correlation|
|-----------------|-------|
| coder           | 0.189 |
| sapbert         | 0.351 |
| sppmi           | 0.571 |
| game      | 0.623 |

#### Depression

| Method          | Correlation|
|-----------------|-------|
| coder           | 0.124 |
| sapbert         | -0.311 |
| sppmi           | 0.636 |
| gat(coder)      | 0.764 |

#### Type 1 Diabetes
| Method          | Correlation|
|-----------------|-------|
| coder           | 0.329 |
| sapbert         | 0.336 |
| sppmi           | 0.306 |
| coder      | 0.361 |


## Detect Drug Side Effect
| Method          | AUC   | After Supervised  |
|-----------------|-------|-------------------|
| coder           | 0.540 | 0.810             |
| sapbert         | 0.578 | 0.782             |
| sppmi           | 0.570 | 0.744             |
| GAT             | 0.636 | 0.834             |


## Detect Drug Side Effect
| Method          | AUC   | After Supervised  |
|-----------------|-------|-------------------|
| aligned sppmi   | 0.568 | 0.810           |
| sapbert         | 0.578 | 0.782             |
| coder           | 0.539 | 0.816             |
| sppmi           | 0.570 | 0.744             |
| GAT sim 256 (edge_all)  | 0.598 | 0.693             |
| GAT rel only 512 (edge_all)  | 0.460 | 0.628        |
| GAT all (edge_all+edge_all)  | 0.550 | 0.745        |
| GAT rel only 512 (edge_rel)  | 0.515 |  0.572      |
| GAT all (edge_all+edge_rel)  | 0.585 |  0.743      |


## Detect Relatedness and Similarity

### MGB
| Method          | Related AUC   | Similarity AUC  |
|-----------------|-------|-------------------|
| coder           | 0.714 | 0.756             |
| sapbert         | 0.661 | 0.746             |
| sppmi           | 0.784 | 0.871             |
| GAT             | 0.906 | 0.928             |

### VA
| Method          | Related AUC   | Similarity AUC  |
|-----------------|-------|-------------------|
| coder           | 0.668 | 0.809             |
| sapbert         | 0.641 | 0.739             |
| sppmi           | 0.750 | 0.837             |
| GAT             | 0.889 | 0.915             |

### UPMC
| Method          | Related AUC   | Similarity AUC  |
|-----------------|-------|-------------------|
| coder           | 0.707 | 0.745             |
| sapbert         | 0.636 | 0.743             |
| sppmi           | 0.679 | 0.792             |
| GAT             | 0.914 | 0.910             |




