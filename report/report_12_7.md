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


