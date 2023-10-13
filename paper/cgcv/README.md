# Corrected generalized cross-validation for finite ensembles of penalized estimators
This repository includes Python code that can reproduce all the simulation results presented in both the main text and the supplementary material of the submitted paper.

## Description:
Description of folders:

1. **Gaussian**: Implement the numerical experiments on non-Gaussian models as described in Section 4 of the paper. 
The files inside this folder are described as follows. 
    * `run_ridge.py` : run ridge simulation, save results.

    * `run_net.py` : run elastic net simulation, save results.

    * `run_lasso.py` : run Lasso simulation, save results.

    * `plots.ipynb` : after executing the above scripts, generate figures using the saved results.

2. **Non-Gaussian**: Implement the numerical experiments on non-Gaussian models as described in Section 5 of the paper. 
The files inside this folder are described as follows.

    * `generate_data.py` : generate simulated data.

    * `compute_risk.py` : fit models and compute the risk and estimates.

    * `run_simu_lam.py` : run ridge, Lasso, and Elastic Net simulations with different penalty lambda.

    * `run_simu_psi.py` : run ridge, Lasso, and Elastic Net simulations with different subsample aspect ratio psi.

    * `Plot.ipynb` : after executing the above scripts, generate figures using the saved results.
