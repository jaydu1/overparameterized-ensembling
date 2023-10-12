---
layout: page
show_in_menu: false
title: "Extrapolated cross-validation for randomized ensembles"
permalink: /ecv/
---


# Abstract

Ensemble methods such as bagging and random forests are ubiquitous in various fields, from finance to genomics. Despite their prevalence, the question of the efficient tuning of ensemble parameters has received relatively little attention. This paper introduces a cross-validation method, ECV (Extrapolated Cross-Validation), for tuning the ensemble and subsample sizes in randomized ensembles. Our method builds on two primary ingredients: initial estimators for small ensemble sizes using out-of-bag errors and a novel risk extrapolation technique that leverages the structure of prediction risk decomposition. By establishing uniform consistency of our risk extrapolation technique over ensemble and subsample sizes, we show that ECV yields $\delta$-optimal (with respect to the oracle-tuned risk) ensembles for squared prediction risk. Our theory accommodates general ensemble predictors, only requires mild moment assumptions, and allows for high-dimensional regimes where the feature dimension grows with the sample size. As a practical case study, we employ ECV to predict surface protein abundances from gene expressions in single-cell multiomics using random forests. In comparison to sample-split cross-validation and $K$-fold cross-validation, ECV achieves higher accuracy avoiding sample splitting. At the same time, its computational cost is considerably lower owing to the use of the risk extrapolation technique. Additional numerical results validate the finite-sample accuracy of ECV for several common ensemble predictors under a computational constraint on the maximum ensemble size.



# Code


The experiments demonstrate the utility of ECV on bagging (bootstrap aggregating) and subagging (subsample aggregating) with various predictors:

- ridgeless/ridge predictor,
- lassoless/lasso predictor,
- kNN regressor,
- Logistic regression,
- decision tree (for building random forests).

The code for reproducing results of this paper is available at [Github](https://github.com/jaydu1/ecv).

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