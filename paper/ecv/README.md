# Abstract

Ensemble methods such as bagging and random forests are ubiquitous in various fields, from finance to genomics. Despite their prevalence, the question of the efficient tuning of ensemble parameters has received relatively little attention. This paper introduces a cross-validation method, ECV (Extrapolated Cross-Validation), for tuning the ensemble and subsample sizes in randomized ensembles. Our method builds on two primary ingredients: initial estimators for small ensemble sizes using out-of-bag errors and a novel risk extrapolation technique that leverages the structure of prediction risk decomposition. By establishing uniform consistency of our risk extrapolation technique over ensemble and subsample sizes, we show that ECV yields $\delta$-optimal (with respect to the oracle-tuned risk) ensembles for squared prediction risk. Our theory accommodates general ensemble predictors, only requires mild moment assumptions, and allows for high-dimensional regimes where the feature dimension grows with the sample size. As a practical case study, we employ ECV to predict surface protein abundances from gene expressions in single-cell multiomics using random forests. In comparison to sample-split cross-validation and $K$-fold cross-validation, ECV achieves higher accuracy avoiding sample splitting. At the same time, its computational cost is considerably lower owing to the use of the risk extrapolation technique. Additional numerical results validate the finite-sample accuracy of ECV for several common ensemble predictors under a computational constraint on the maximum ensemble size.


# Code

Extrapolated cross-validation for randomized ensembles based on out-of-bag estimates.
This repository contains code for reproducing results in the paper ``Extrapolated cross-validation for randomized ensembles'' [[arXiv]](https://arxiv.org/abs/2302.13511).
The goal is to tune both the ensemble size $M$ and subsample/bootstrap size $k$.
The experiments demonstrate the utility of ECV on bagging (bootstrap aggregating) and subagging (subsample aggregating) with various predictors:

- Ridgeless/Ridge predictor,
- Lassoless/Lasso predictor,
- kNN regressor,
- Logistic regression,
- Decision tree (for building random forests).



## Scripts

Below is the summary of the scripts. The details about command line arguments is described in each script separately.

- Estimation
	- (Section 5.1) `ex1_risk_estimate.py` obtains ECV-based OOB estimates and empirical risks.
- Prediction
	- (Section 5.2) `ex2_risk_cv.py` evaluates the ECV estimates for tuning $(M,k)$.
	- (Section 5.2) `ex2_risk.py` evaluates the empirical risk of ensembles with different $(M,k)$.
- Classification
	- (Section S6.3) `ex3_classification.py` compares ECV and K-fold CV on imbalanced binary classification problems.
- Random forests
	- (Section 5.3) `ex4_rf_path.py` evaluates ECV on random forests.
	- (Section S6.1) `ex4_rf_M0.py` conducts sensitivity analysis of the hyperparameter $M_0$.
	- (Section S6.2) `ex4_rf_rho.py` conducts sensitivity analysis of the correlative parameter of the AR(1) covariance.
	- (Section S6.3) `ex4_rf_cov.py` conducts sensitivity analysis of covariance structure.
	- (Section S6.4) `ex4_rf_mtry.py` tunes both the observation and feature subsampling ratios.
- Real data (Section 6)
	- `convert_data.R` converts `pbmc_multimodal.h5seurat` (see the next subsection about how to obtain the data) to `pbmc_count.h5`.
	- `df_split.csv` splits training and test sets.
	- `run_real_data.py`
- Utility functions for running the experiments: 
	- `compute_risk.py` fits models, obtains estimates, and computes the risks.
	- `generate_data.py` generate simulated data used in Sections 5, 6, and S6.
- Figures: `Visualization.ipynb`


## Data

The readers can follow the [Seurat tutorial](https://satijalab.org/seurat/articles/multimodal_reference_mapping.html) to download the preprocessed dataset `pbmc_multimodal.h5seurat`.

# Dependencies

The requirement of Python packages for running the Python scripts are as follows:


Package | version
---|---
numpy | 1.23.5
pandas | 2.0.0
scipy | 1.10.1
h5py | 3.1.0
scikit-learn | 1.2.2
tqdm | 4.65.0


To load and convert Seurat object (`convert_data.R`), the following R packages are required:

Package | version
---|---
Seurat | 4.3.0.1
SeuratDisk | 0.0.0.9020
hdf5r | 1.3.7
