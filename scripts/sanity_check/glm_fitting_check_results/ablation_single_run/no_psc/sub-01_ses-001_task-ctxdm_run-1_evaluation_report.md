# GLM Evaluation Report (Vertex-level)
**Subject:** sub-01  
**Task:** ctxdm  
**Session:** ses-001  
**Run:** run-1

## Model Specs
- # regressors (K): 141
- # task regressors: 120
- # time points after tmask (T_kept): 183
- AR(1) rho: 0.342
- Estimator: WLS_AR1

## Vertex Sampling
- Total vertices in run: 64984
- Sampled vertices for plots: 1000

## Fit (Sampled Vertices)
- Mean R²: 0.609
- Median R²: 0.633
- R² std: 0.187
- R² range: [-2.050, 0.922]
- Vertices with R² > 0.0: 991/1000 (99.1%)
- Vertices with R² > 0.1: 989/1000 (98.9%)
- Vertices with R² > 0.2: 979/1000 (97.9%)

## Task Beta Consistency (full vs saved task slice)
- Mean |Δ| over sampled vertices: 0
- Max  |Δ| over sampled vertices: 0

## Extremely Negative R²
- Threshold: R² < -1.0
- Count: 96 / 58794 (0.16%)