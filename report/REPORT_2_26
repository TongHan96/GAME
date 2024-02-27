# Report

| Records           | Align           | Align edges   |Sim edges   | Rel edges   | Drug side| Feature selection| Accuracy|
|-------------------|-----------------|---------------|------------|-------------|-----------|-----------------|----------|
|2024-02-24 20:51:13| with sap + concat sap| all  no coder | all no coder    |rel no coder| 0.59(?)|0.404,0.637,0.734,0.317|76.7,88.9,90.7,92.5|
|2024-02-24 20:50:12| without sap + concat sap | all  no coder |all no coder | rel no coder|0.582(?)|0.434,0.646,0.739,0.342|76.6,89.5,91.0,92.9|


# Drug side Benchmark

| All methods   | AUC   | Supervised AUC |
|---------------|-------|----------------|
| sapbert       | 0.676 | 0.830          |
| coder         | 0.557 | 0.866         |
| MGB sppmi     | 0.551 | 0.694          |
| VA sppmi      | 0.528 | 0.676          |
| UPMC sppmi    | 0.544 | 0.716          |
| BCH sppmi     | 0.545 |           |


# Drug indi Benchmark

| All methods   | AUC   | Supervised AUC |
|---------------|-------|----------------|
| sapbert       | 0.716 | 0.779          |
| coder         |0.766 | 0.837           |
| MGB sppmi     | 0.753 | 0.753          |
| VA sppmi      | 0.732 | 0.751          |
| UPMC sppmi    | 0.769 | 0.775          |
| BCH sppmi     | 0.602 | 0.603          |

# Feature selection Benchmark

| Desease           | SAP   | Coder | SVD_MGB | SVD_VA | SVD_UP | SVD_BCH | GAME  |
|---------------------|-------|-------|---------|--------|--------|---------|-------|
| RA                  | 0.058 | 0.12  | 0.367   | 0.334  | 0.159  | 0.136   | 0.434 |
| CHF                 | 0.333 | 0.239 | 0.564   | 0.519  | 0.263  | 0.336   | 0.646 |
| Depression          | -0.213| 0.129 | 0.611   | 0.43   | 0.277  | 0.36    | 0.739 |
| Type 1 diabetes     | 0.134 | 0.278 | 0.117   | 0.184  | 0.153  | -0.075  | 0.342 |

# Accuracy Benchmark

| Measure | sap(short name) | sap(dict replaced name)| sap(full name) |
| --- | --- |--- | --- |
| TOP1 |52.7%| 55.6%|61.1% |
| TOP5 | 67.0%| 71.0%|79.6% |
| TOP10 |71.3%| 74.9%| 83.3% |
| TOP20 |75.1%| 78.4%| 86.3% |
