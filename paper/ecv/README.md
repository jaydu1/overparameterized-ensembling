# ECV

Extrapolated cross-validation for randomized ensembles based on out-of-bag estimates.
This repository contains code for reproducing results in the paper ``Extrapolated cross-validation for randomized ensembles'' [[arXiv]](https://arxiv.org/abs/2302.13511).
The goal is to tune both the ensemble size $M$ and subsample/bootstrap size $k$.
The experiments demonstrate the utility of ECV on bagging (bootstrap aggregating) and subagging (subsample aggregating) with various predictors:

- ridgeless/ridge predictor,
- lassoless/lasso predictor,
- kNN regressor,
- Logistic regression,
- decision tree (for building random forests).



## Scripts

- Section 5.1
	- `run_iso.py` obtains ECV-based OOB estimates.
	- `run_iso_risk.py` obtains empirical out-of-sample error for comparison using a finer grid.
- Section 5.2
	- `run_iso_cv.py` evaluates the ensembles with ECV-tuned parameters $(\hat{M},\hat{k})$.
- Section 5.3
	- `run_ar1_cv_tree.py` evaluates the random forests with ECV-tuned ensemble size $\hat{M}$.
- Section 6
	- `convert_data.R` converts `pbmc_multimodal.h5seurat` obtained from [Seurat](https://satijalab.org/seurat/articles/multimodal_reference_mapping.html) to `pbmc_count.h5`.
	- `df_split.csv` splits training and test sets.
	- `run_real_data.py`

- Utility functions for running the experiments: 
	- `compute_risk.py` fits models, obtains estimates, and computes the risks.
	- `fixed_point_sol.py` computes quantities related to the fixed-point equations. 
	- `generate_data.py` generate simulated data used in Section 5.

- Figures: `Visualization.ipynb`