# GLM Evaluation Report (Vertex-level)
**Subject:** sub-03  
**Task:** interdms  
**Session:** ses-001  
**Run:** run-1

## Model Specs
- # regressors (K): 161
- # task regressors: 128
- # time points after tmask (T_kept): 207
- AR(1) rho: 0.000
- Estimator: OLS

## Vertex Sampling
- Total vertices in run: 64984
- Sampled vertices for plots: 1000

## Fit (Sampled Vertices)
- Mean R²: 0.921
- Median R²: 0.933
- R² std: 0.050
- R² range: [0.731, 0.992]
- Vertices with R² > 0.0: 1000/1000 (100.0%)
- Vertices with R² > 0.1: 1000/1000 (100.0%)
- Vertices with R² > 0.2: 1000/1000 (100.0%)

## Task Beta Consistency (full vs saved task slice)
- Mean |Δ| over sampled vertices: 0
- Max  |Δ| over sampled vertices: 0

## Extremely Negative R²
- Threshold: R² < -1.0
- Count: 0 / 60263 (0.00%)