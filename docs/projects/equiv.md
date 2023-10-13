---
layout: page
disable_anchors: true
show_in_menu: false
title: "Generalized equivalences between subsampling and ridge regularization"
permalink: /equiv/
---


# Abstract

We establish precise structural and risk equivalences between subsampling and ridge regularization for ensemble ridge estimators. Specifically, we prove that linear and quadratic functionals of subsample ridge estimators, when fitted with different ridge regularization levels $\lambda$ and subsample aspect ratios $\psi$, are asymptotically equivalent along specific paths in the $(\lambda, \psi)$-plane (where $\psi$ is the ratio of the feature dimension to the subsample size). Our results only require bounded moment assumptions on feature and response distributions and allow for arbitrary joint distributions. Furthermore, we provide a datadependent method to determine the equivalent paths of $(\lambda, \psi)$. An indirect implication of our equivalences is that optimally-tuned ridge regression exhibits a monotonic prediction risk in the data aspect ratio. This resolves a recent open problem raised by Nakkiran et al. under general data distributions and a mild regularity condition that maintains regression hardness through linearized signal-to-noise ratios.


# Code

The code for reproducing results of this paper is available at [Github](https://github.com/jaydu1/overparameterized-bagging/tree/main/paper/equiv).

## Scripts

- Section 3:
	- Figure1 1: `run_equiv_estimator.py` computes the linear projections of ensemble estimators on simulated data.

- Section 4:
	- Figures 2 and F5: `run_equiv_risk.py` computes generalized quadratic risks on simulated data.

- Real data:
	- Figure 3: `run_equiv_cifar.py` computes the empirical estimates on CIFAR-10.
	- Figure F6: `run_equiv_real_data.py` computes the empirical estimates on CIFAR-10, MNIST, and USPS.

- Extensions:
	- Random features regression (Figure 4): `run_equiv_random_feature.py`
	- Kernel regression (Figure F7): `run_equiv_kernel.py`


- The jupyter notebook plots all the figures based on results produced by previous scripts.

## Computation details

All the experiments are run on Pittsburgh Supercomputing Center Bridge-2 RM-partition using 48 cores.

The estimated time to run all experiments is roughly 6 hours for each script.