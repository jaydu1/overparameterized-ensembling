---
layout: page
show_in_menu: false
title: "Corrected generalized cross-validation for finite ensembles of penalized estimators"
permalink: /cgcv/
---

# Abstract

Generalized cross-validation (GCV) is a widely-used method for estimating the squared out-of-sample prediction risk that employs a scalar degrees of freedom adjustment (in a multiplicative sense) to the squared training error. In this paper, we examine the consistency of GCV for estimating the prediction risk of arbitrary ensembles of penalized least squares estimators. We show that GCV is inconsistent for any finite ensemble of size greater than one. Towards repairing this shortcoming, we identify a correction that involves an additional scalar correction (in an additive sense) based on degrees of freedom adjusted training errors from each ensemble component. The proposed estimator (termed CGCV) maintains the computational advantages of GCV and requires neither sample splitting, model refitting, or out-of-bag risk estimation. The estimator stems from a finer inspection of ensemble risk decomposition and two intermediate risk estimators for the components in this decomposition. We provide a non-asymptotic analysis of the CGCV and the two intermediate risk estimators for ensembles of convex penalized estimators under Gaussian features and a linear response model. In the special case of ridge regression, we extend the analysis to general feature and response distributions using random matrix theory, which establishes model-free uniform consistency of CGCV.

# Code


The code for reproducing results of this paper is available at [Github](https://github.com/kaitan365/CorrectedGCV/tree/main).
This repository includes Python code that can reproduce all the simulation results presented in both the main text and the supplementary material of the submitted paper.



## Scripts


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