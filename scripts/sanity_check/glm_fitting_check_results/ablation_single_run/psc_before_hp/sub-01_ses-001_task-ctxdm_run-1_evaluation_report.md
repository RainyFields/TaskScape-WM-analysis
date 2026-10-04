# GLM Evaluation Report (Vertex-level)
**Subject:** sub-01  
**Task:** ctxdm  
**Session:** ses-001  
**Run:** run-1

## Model Specs
- # regressors (K): 141
- # task regressors: 120
- # time points after tmask (T_kept): 183
- AR(1) rho: 0.311
- Estimator: WLS_AR1

## Vertex Sampling
- Total vertices in run: 64984
- Sampled vertices for plots: 1000

## Fit (Sampled Vertices)
- Mean R²: 0.012
- Median R²: 0.011
- R² std: 0.009
- R² range: [0.002, 0.132]
- Vertices with R² > 0.0: 1000/1000 (100.0%)
- Vertices with R² > 0.1: 2/1000 (0.2%)
- Vertices with R² > 0.2: 0/1000 (0.0%)

## Task Beta Consistency (full vs saved task slice)
- Mean |Δ| over sampled vertices: 0
- Max  |Δ| over sampled vertices: 0

## Extremely Negative R²
- Threshold: R² < -1.0
- Count: 12 / 58794 (0.02%)