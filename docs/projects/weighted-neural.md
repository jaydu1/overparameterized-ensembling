---
layout: page
disable_anchors: true
show_in_menu: false
title: "Implicit regularization paths of weighted neural representations"
permalink: /weighted-neural/
---


# Abstract

We investigate the implicit regularization of (observation) weighting pretrained features and derive a path of equivalence connecting different weighting matrices and ridge regularization with matching effective degrees of freedom. 
For the special case of subsampling without replacement, our results apply to both random features and kernel features, resolving recent conjectures in Patil and Du (2023). 
We also obtain a risk decomposition for an ensemble of weighted estimators and demonstrate that the risks are equivalent along the path for the full ensembles. 
For tuning in practice, we develop an efficient cross-validation method and apply it to subsampled pretrained representations across several models (e.g., ResNet-50) and datasets (e.g., CIFAR-100) to validate our theoretical results.


# Code

The code for reproducing results of this paper is available at [Github](https://github.com/jaydu1/overparameterized-ensembling/tree/main/paper/weighted-neural).

## Scripts

### Simulation
- `run_equiv_estimator.py` examines the equivalence of the degrees of freedom and the linear projections of ensemble estimators on simulated RMT features.
- `run_equiv_feature.py` examines the degrees of freedom equivalence on simulated RMT, random features, kernel features.
- `run_equiv_risk.py` examines the risk equivalence of ensemble estimators on simulated data and computes ECV risk estimates.

### Real data

- `real_data.ipynb` get pretrained features from ResNet on real datasets. One should first clone Github repo [empirical-ntks](https://github.com/aw31/empirical-ntks) to the local filesystem. 
- `run_real_data_df.py` examines the risk equivalence of ensemble estimators on real data.
- `run_real_data_risk.py` examines the risk equivalence of ensemble estimators on real data.
- `run_real_data_tuning.py` examines the corrected and extrapolated genearlized cross-validation method on real data.

### Plot
- `plot.ipynb` The jupyter notebook plots all the figures based on results produced by previous scripts.

## Computation details

All the experiments are run on Pittsburgh Supercomputing Center Bridge-2 RM-partition using 48 cores.

The estimated time to run all experiments is roughly 6~24 hours for each script.

