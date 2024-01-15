# Result Summary

## Findings
1. align sppmi itself is good enough for most of works except code mapping
2. change of align sppmi is not good for drug side effect detection (like using a GNN or weighted add)
3. problem: how to combine these two embeddings (if do not use all edges in simi training part, feature selection of concatenate emb is bad).


## Code Mapping Job
### Sapbert embedding
**Accuracy Rates:**  
| Measure | short | dict| full | gat(256) | now(256 all edges)|now(256 sim edges)|
| --- | --- |--- | --- | --- | --- |--- |
| TOP1 |52.7%| 55.6%|61.1% | 75.6% | 77.9%|78.1%|
| TOP5 | 67.0%| 71.0%|79.6% | 85.2% |89.2%|88.7%|
| TOP10 |71.3%| 74.9%| 83.3% | 87.6% |91.3%|91.2%|
| TOP20 |75.1%| 78.4%| 86.3% | 89.9% |92.9%|92.8%|

 
## Detect Drug Side Effect
| Method          | AUC   | After Supervised  |
|-----------------|-------|-------------------|
| aligned sppmi (256)  | 0.571 | 0.788        |
| aligned sppmi (512)  | 0.565 | 0.809        |
| aligned sppmi (768)  | 0.568 | 0.810        |
| sapbert         | 0.578 | 0.792             |
| coder           | 0.539 | 0.821             |
| sppmi           | 0.570 | 0.744             |
| GAT + GAT  | 0.585 |  0.743  |
| GAT + align  | 0.587 |  0.824  |




## Feature Selection
### GPT4 Scoring & Cos Sim Rank Correlation:

|  | sap | coder | svd_MGB | svd_VA | svd_UP | align256 | align512 | align768 | gat+gat|gat+align512|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |--- |--- |
| RA | 0.001 | 0.191 | 0.433 | 0.415 | 0.146 | 0.444 | 0.417 | 0.455 |0.382|0.401|
| CHF | 0.382 | 0.291 | 0.586 | 0.56 | 0.147 | 0.628 | 0.612 | 0.629 |0.607|0.632|
| Depression | -0.222 | 0.099 | 0.641 | 0.405 | 0.226 | 0.755 | 0.769 | 0.762 |0.777|0.786|
| Type 1 diabetes | 0.166 | 0.264 | 0.135 | 0.187 | 0.127 | 0.291 | 0.321 | 0.273 |0.314|0.293|

**After using Gat in rel part, feature selection does not do beeter than use the column combine of sim part and align 512 directly.**


<!----## Detect Relatedness and Similarity
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
| GAT             | 0.914 | 0.910             |-->




