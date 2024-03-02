# t-sne msthod

1. We utilize Alzheimer's Disease (AD) data from UPMC and MGB. The case group consists of individuals diagnosed within 0-2 years, while the control group includes individuals from 10-5 years before diagnosis. From each group, we select 5000 data points for t-SNE analysis.
2. UPMC dataL Shanshan has already cleaned the data (rm some patients), using `UPMC_AD_survival_analysis_df_1.RData`; while MGB data remain no cleaning
3. The weighted embedding process involves using the cosine similarity of each code with the AD code (PheCode: 290.11) to weight the code embedding
 
 ```math
tf_{i,j} = \text{Frequency of \(i^{th}\) code in \(j^{th}\) patient's record}
```
 ```math
idf_{i,i} = log(\# patient/ (1+ \# \text{patient with code i}))
```
 ```math
tf\_idf = tf*idf
```
 ```math
pt\_emb = tf\_idf*code\_emb
```
 ```math
pt\_emb\_wgt = tf\_idf*diag(weight)*code\_emb
```


# t-sne results

## Single inst

### UPMC
<p align="center">
  <img src="https://github.com/TongHan96/GAME/blob/main/report/pic/upmc_wgt_5_0.0.png" alt="upmc_wgt" title="Weighted" width="400"/>
  <em>upmc weighted</em>
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://github.com/TongHan96/GAME/blob/main/report/pic/upmc_5_0.5.png" alt="upmc" title="Origin" width="400"/>
  <em>upmc</em>
</p>

### MGB
<p align="center">
  <img src="https://github.com/TongHan96/GAME/blob/main/report/pic/mgb_wgt_30_0.1_768.png" alt="MGB_wgt" title="Weighted" width="400"/>
  <em>MGB weighted</em>
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://github.com/TongHan96/GAME/blob/main/report/pic/mgb_30_0.5_768.png" alt="MGB" title="Origin" width="400"/>
  <em>MGB</em>
</p>

## Both inst
<p align="center">
  <img src="https://github.com/TongHan96/GAME/blob/main/report/pic/upmc_mgb_wgt_30_0.5.png" alt="U&M" title="U&M_Weighted" width="330"/>
  <img src="https://github.com/TongHan96/GAME/blob/main/report/pic/upmc_mgb_wgt.png" alt="U&M" title="U&M" width="330"/>
  <img src="https://github.com/TongHan96/GAME/blob/main/report/pic/upmc_mgb_wgt_inst.png" alt="U" title="U" width="330"/>
</p>
