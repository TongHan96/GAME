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


### Sapbert+LP embedding
**Accuracy Rates:**  
| Measure | short | dict| full | gat(256) |
| --- | --- |--- | --- | --- |
| TOP1 |53.3%| 56.9%|62.3% | 75.1% |
| TOP5 | 66.8%| 71.3%|78.8% | 85.0% |
| TOP10 |70.6%| 74.9%| 81.9% | 87.7% |
| TOP20 |73.3%| 77.4%| 83.9% | 89.6% |


### Coder embedding
**Accuracy Rates:**  
| Measure | short | dict| full | gat(256) |
| --- | --- |--- | --- | --- |
| TOP1 |53.8%| 54.9%| 57.2% | 77.1% |
| TOP5 | 72.4%|73.7%| 77.5% | 86.2% |
| TOP10 |78.9%  |79.9%| 85.9% | 88.3% |
| TOP20 |83.3%| 84.6%| 91.1% | 90.2% |

### Coder+LP embedding
**Accuracy Rates:**  
| Measure | short | dict| full | gat(256) |
| --- | --- |--- | --- | --- |
| TOP1 |58.2%| 59.5%| 63.1% | 76.6% |
| TOP5 | 74.9%|75.6%| 80.7% | 86.2% |
| TOP10 |80.8%  |81.7%| 86.3% | 88.4% |
| TOP20 |85.1%| 86.2%| 91.2% | 90.6% |

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

## Detect Drug Side Effect
| Method          | AUC   | After Supervised  |
|-----------------|-------|-------------------|
| coder           | 0.540 | 0.810             |
| sapbert         | 0.578 | 0.797             |
| sppmi           | 0.525 | 0.751             |
| GAT             | 0.633 | 0.834             |

<!-- ## Feature Selection
### GPT4 Scoring & Cos Sim Rank Correlation:
#### RA
| Method          | Correlation|
|-----------------|-------|
| coder           | 0.276 |
| sapbert         | 0.176 |
| sppmi           | 0.586 |
| gat(coder)      | 0.443 |
| gat(sap)        | 0.468 |
|-----------------|-------|
| coder_LP        | 0.284 |
| sapbert_LP      | 0.174 |
| gat(coder_LP)   | 0.354 |
| gat(sap_LP)     | 0.464 |

#### CHF
| Method          | Correlation|
|-----------------|-------|
| coder           | 0.341 |
| sapbert         | 0.473 |
| sppmi           | 0.683 |
| gat(coder)      | 0.690 |
| gat(sap)        | 0.707 |
|-----------------|-------|
| coder_LP        | 0.350 |
| sapbert_LP      | 0.479 |
| gat(coder_LP)   | 0.667 |
| gat(sap_LP)     | 0.698 |

#### Depression

| Method          | Correlation|
|-----------------|-------|
| coder           | 0.132 |
| sapbert         | -0.115 |
| sppmi           | 0.626 |
| gat(coder)      | 0.772 |
| gat(sap)        | 0.765 |
|-----------------|-------|
| coder_LP        | 0.133 |
| sapbert_LP      | -0.138 |
| gat(coder_LP)   | 0.768 |
| gat(sap_LP)     | 0.754 |

#### Type 1 Diabetes
| Method          | Correlation|
|-----------------|-------|
| coder           | 0.384 |
| sapbert         | 0.374 |
| sppmi           | 0.403 |
| gat(coder)      | 0.410 |
| gat(sap)        | 0.464 |
|-----------------|-------|
| coder_LP        | 0.380 |
| sapbert_LP      | 0.392 |
| gat(coder_LP)   | 0.406 |
| gat(sap_LP)     | 0.460 |

## Detect Drug Side Effect
| Method          | AUC   | After Supervised  |
|-----------------|-------|-------------------|
| coder           | 0.540 | 0.810             |
| sapbert         | 0.578 | 0.782             |
| sppmi           | 0.570 | 0.744             |
| GAT             | 0.636 | 0.834             |-->




## Detect Relatedness and Similarity

### MGB
| Method          | Related AUC   | Similarity AUC  |
|-----------------|-------|-------------------|
| coder           | 0.689 | 0.767             |
| sapbert         | 0.685 | 0.745             |
| sppmi           | 0.775 | 0.871             |
| GAT             | 0.901 | 0.927             |

### VA
| Method          | Related AUC   | Similarity AUC  |
|-----------------|-------|-------------------|
| coder           | 0.656 | 0.809             |
| sapbert         | 0.676 | 0.737             |
| sppmi           | 0.721 | 0.836             |
| GAT             | 0.904 | 0.915             |

### UPMC
| Method          | Related AUC   | Similarity AUC  |
|-----------------|-------|-------------------|
| coder           | 0.681 | 0.747             |
| sapbert         | 0.671 | 0.742             |
| sppmi           | 0.679 | 0.793             |
| GAT             | 0.913 | 0.913             |
