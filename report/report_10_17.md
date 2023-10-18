# Result Summary

## Changes
1. **Updated the sapbert/coder embedding** for leaf LOINC codes to their corresponding LP code embedding.
2. We have also changed the **accuracy calculation method** to only focus on LP code.
3. The **rotation mode for sppmi** has changed: VA and UPMC revolve around MGB.
4. Use **edges generated from sppmi emb** in related training part. **Really helpful for feature selection!**


## Accuracy Calculation Method:
By focusing solely on LP codes, **TOP1 accuracy tends to be slightly lower, while TOP20 accuracy tends to be slightly higher**. This trend makes sense because:
- The exact matches have become rarer and more challenging (hence, the drop in TOP1).
- The overall comparison objects have decreased (hence, the rise in TOP20).

## Accuracy Results:
### 1. SAPBERT:
With this new embedding approach (leaf LOINC sapbert equals LP code), the accuracy for sapbert decreased a bit, and the increase in accuracy with training was not substantial.

**Accuracy Rates:**  
| Measure | Sap | Sap+Procrutes| Sap+gat | SAP+gat+old_eval| Old gat (256) | Old gat (768) |
| --- | --- |--- | --- | --- | --- | --- |
| TOP1 |59.1%| 66.0%|71.6% | 74.8% | 75.4% | 76.9% |
| TOP5 | 77.7%|79.9%|82.5% | 83.7% | 85.8% | 87.3% |
| TOP10 |80.7%  |82.0%| 85.6% | 85.5% | 88.4% | 89.6% |
| TOP20 |83.1%| 84.0%|88.1% | 87.5% | 90.0% | 91.8% |

### 2. CODER:
The accuracy is generally higher for the coder compared to sapbert.

**Accuracy Rates:**  
| Measure | Coder | Coder+old_eval |
| --- | --- | --- |
| TOP1 | 76.6% | 76.6% |
| TOP5 | 86.7% | 85.9% |
| TOP10 | 89.1% | 88.0% |
| TOP20 | 91.1% | 89.6% |

## AUC:
**Similarity (256):**  
| | MGB | VA | UPMC |
| --- | --- | --- | --- |
| SAPBERT | 0.887 | 0.904 | 0.893 |
| CODER | 0.896 | 0.898 | 0.894 |

**Related (768):**  
| | MGB | VA | UPMC |
| --- | --- | --- | --- |
| SAPBERT | 0.904 | 0.889 | 0.901 |
| CODER | 0.908 | 0.902 | 0.915 |
| CODER + sppmi_edge | 0.933 | 0.921 | 0.936 |


## Biobank Data & Cos Sim Rank Correlation:

### **SPEARMAN PheCode:741.1**

| With ... code | coder_gat_sppmi_edge | coder_gat | sap_gat_old | coder | sap | svd |
|------------|----------------|-----|---------|-------|-----|-----|
| all        | 0.25           | 0.20| 0.23    | 0.04  | 0.06| 0.26|
| PheCode    | 0.35           | 0.22| 0.23    | 0.19  | 0.03| 0.35|
| RXNORM     | 0.24           | 0.21| 0.22    | -0.08 | 0.05| 0.39|
| LOINC      | 0.10           | 0.11| 0.19    | 0.00  | 0.05| 0.03|
| CCS-PCS    | 0.36           | 0.24| 0.28    | 0.22  | 0.13| 0.44|

### **SPEARMAN PheCode:428.1**

| With ... code | coder_gat_sppmi_edge | coder_gat | sap_gat_old | coder | sap | svd |
|------------|----------------|-----|---------|-------|-----|-----|
| all        | 0.28           | 0.25| 0.26    | 0.15  | 0.10| 0.42|
| PheCode    | 0.40           | 0.28| 0.26    | 0.29  | 0.10| 0.49|
| RXNORM     | 0.36           | 0.33| 0.33    | 0.01  | 0.06| 0.50|
| LOINC      | 0.27           | 0.29| 0.28    | 0.12  | 0.11| 0.30|
| CCS-PCS    | 0.26           | 0.22| 0.28    | 0.30  | 0.15| 0.51|

## GPT4 Scoring & Cos Sim Rank Correlation:

| Metric          | Score |
|-----------------|-------|
| coder           | 0.230 |
| sapbert         | 0.140 |
| sppmi           | 0.653 |
| sap_gat_old            | 0.325 |
| coder_gat             | 0.329 |
| coder_gat_sppmi_edge  | 0.407 |

## Drug side effect


| Method                   | MGB   | VA    | UPMC  |
|--------------------------|-------|-------|-------|
| SAPBERT                  | 0.579 | 0.579 | 0.577 |
| SAPBERT + supervised     | 0.807 | 0.802 | 0.791 |
| ---                      | ---   | ---   | ---   |
| SVD                      | 0.573 | 0.545 | 0.558 |
| SVD + supervised         | 0.762 | 0.740 | 0.710 |
| ---                      | ---   | ---   | ---   |
| SAP_gat                  | 0.626 | 0.616 | 0.634 |
| SAP_gat + supervised     | 0.832 | 0.829 | 0.831 |
| ---                      | ---   | ---   | ---   |
| CODER_gat                | 0.632 | 0.632 | 0.628 |
| CODER_gat + supervised   | 0.850 | 0.846 | 0.848 |





