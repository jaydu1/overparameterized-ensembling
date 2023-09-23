# Equivalence of ridge ensembles and consistency of GCV


## Scripts

- Section 2 Figure 1
	- `run_gcv_the_equiv.py` computes the theoretical GCV and risk asymptotics for plotting.

	
- Section 3.1 Figure 2
	- `run_gcv_the.py` computes theoretical GCV asymptotics.
	- `run_gcv_estimate.py` computes empirical GCV estimates.

	
- Section 3.2 Figure 3
	- `run_gcv_opt.py` computes the empirical risk of large ridgeless ensemble.
	- `run_gcv_the_lam.py` computes the theoretical risk of optimal ridge predictors. 
	
- Section 3.2 Figure 4 GCV for general M
	- `run_gcv_correct_est.py` computes empirical GCV estimates.
	- `run_gcv_correct_the.py` computes theoretical curves.
	
	
- Section 4 Figure 5
	- `convert_data.R` converts `pbmc_multimodal.h5seurat` obtained from [Seurat](https://satijalab.org/seurat/articles/multimodal_reference_mapping.html) to `pbmc_count.h5`.
	- `df_split.csv` splits training and test sets.
	- `run_gcv_real_data.py`

- Functions for runing the experiments: 
	- `compute_risk.py` fits models, obtains estimates, and computes the risks.
	- `fixed_point_sol.py` computes quantities related to the fixed-point equations. 
	- `generate_data.py` generate simulated isotopic and non-isotopic data.
