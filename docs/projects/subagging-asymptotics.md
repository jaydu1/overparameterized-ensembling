---
layout: page
disable_anchors: true
show_in_menu: false
title: "Implicit regularization paths of weighted neural representations"
permalink: /weighted-neural/
---


# Abstract

We characterize the squared prediction risk of ensemble estimators obtained through subagging (subsample bootstrap aggregating) regularized M-estimators and construct a consistent estimator for the risk. Specifically, we consider a heterogeneous collection of $M\geq 1$ regularized M-estimators, each trained with (possibly different) subsample sizes, convex differentiable losses, and convex regularizers. We operate under the proportional asymptotics regime, where the sample size $n$, feature size $p$, and subsample sizes $k_m$ for $m\in[M]$ all diverge with fixed limiting ratios $n/p$ and $k_m/n$. Key to our analysis is a new result on the joint asymptotic behavior of correlations between the estimator and residual errors on overlapping subsamples, governed through a (provably) contractible nonlinear system of equations. Of independent interest, we also establish convergence of trace functionals related to degrees of freedom in the non-ensemble setting (with $M=1$) along the way, extending previously known cases for square loss and ridge, lasso regularizers.
When specialized to homogeneous ensembles trained with a common loss, regularizer, and subsample size, the risk characterization sheds some light on the implicit regularization effect due to the ensemble and subsample sizes $(M,k)$. For any ensemble size $M$, optimally tuning subsample size yields sample-wise monotonic risk. For the full-ensemble estimator (when $M\rightarrow\infty$), the optimal subsample size $k^\ast$ tends to be in the overparameterized regime $(k^*\leq\min\{n,p\})$, when explicit regularization is vanishing. Finally, joint optimization of subsample size, ensemble size, and regularization can significantly outperform regularizer optimization alone on the full data (without any subagging).

## Scripts for computing theoretical and empirical risks




# Code

The code for reproducing results of this paper is available at [Github](https://github.com/jaydu1/overparameterized-ensembling/tree/main/paper/subagging-asymptotics).

## Scripts

### Simulation
- Lasso    
    - Risk of lasso and optimal lasso ensemble (Figures 4, 5 and 10):
        - `run_lasso_opt.py`    
    - Risk of full lasso ensemble (Figures 6 and 11):
        - `run_lasso_equiv.py`
    - Risk of optimal lasso ensemble (Figure 7):
        - `run_lasso_opt_2.py`
    - Fixed-point quantities of lassoless (Figure 8):
        - `run_lassoless.py`
    - Empirical risk of lassoless ensemble (Figure 9):
        - `run_lasso_emp.py`
- Huber
    - Risk of full unregularized Huber ensemble (Figure 12):
        - `run_huber.py`
    - Risk of l1-regularized Huber and optimal l1-regularized Huber ensemble (Figures 3):
        - `run_huber_l1_opt.py`
    - Risk of full l1-regularized Huber ensemble (Figures 2, 8, 13 and 14):
        - `run_huber_l1_emp.py`
        - `run_huber_l1_equiv.py`
- Utility functions
    - `compute_risk.py`
    - `generate_data.py`   
- Visualization
    - The figures can be reproduced with the Jupyter Notebook `Plot.ipynb`.



## Computation details

All the experiments are run on Ubuntu 22.04.4 LTS using 12 cores and 128 GB of RAM.

The estimated time to run all experiments is roughly 12 hours.

## Dependencies

Package | Version
--- | ---
h5py | 3.1.0
joblib | 1.4.0
matplotlib | 3.4.3
numpy | 1.20.3
pandas | 1.3.3
python | 3.8.12
scikit-learn | 1.3.2
sklearn_ensemble_cv | 0.2.3
scipy | 1.10.1
statsmodels | 0.13.5
tqdm | 4.62.3

