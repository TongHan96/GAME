# Report

| Records           | Align           | Align edges   |Sim edges   | Rel edges   | Drug side| Feature selection| Accuracy|
|-------------------|-----------------|---------------|------------|-------------|-----------|-----------------|----------|
|2024-02-19 08:00:18| with sap + concat sap| all   | all    |rel no coder| 0.698(0.835)|0.404,0.637,0.734,0.317|76.7,88.9,90.7,92.5|
|2024-02-20 20:24:39| without sap + concat sap | all   |all | rel no coder|0.697(0.830)|0.434,0.646,0.739,0.342|76.6,89.5,91.0,92.9|
|2024-02-18 04:13:48| without sap + concat sap | all   |all  | all        |0.685(0.818)|0.419,0.648,0.746,0.364|76.6,89.5,91.0,92.9|


# Drug side Benchmark
 | | AUC | supervised AUC|
| sapbert         | 0.576 | 0.805             |
| coder           | 0.539 | 0.826             |
| MGB sppmi           | 0.563 | 0.762          |
| VA sppmi           | 0.545 | 0.740           |
| UPMC sppmi           | 0.558 | 0.710        |
| BCH sppmi           | 0.532 | 0.729        |

# Feature selection Benchmark

| Condition           | SAP   | Coder | SVD_MGB | SVD_VA | SVD_UP | SVD_BCH | GAME  |
|---------------------|-------|-------|---------|--------|--------|---------|-------|
| RA                  | 0.058 | 0.12  | 0.367   | 0.334  | 0.159  | 0.136   | 0.434 |
| CHF                 | 0.333 | 0.239 | 0.564   | 0.519  | 0.263  | 0.336   | 0.646 |
| Depression          | -0.213| 0.129 | 0.611   | 0.43   | 0.277  | 0.36    | 0.739 |
| Type 1 diabetes     | 0.134 | 0.278 | 0.117   | 0.184  | 0.153  | -0.075  | 0.342 |

# Accuracy



