---
layout: page
disable_anchors: false
title: "sklearn_ensemble_cv"
permalink: /sklearn-ensemble-cv/
---



[![PyPI](https://img.shields.io/pypi/v/sklearn_ensemble_cv?label=pypi)](https://pypi.org/project/sklearn-ensemble-cv)
[![PyPI-Downloads](https://img.shields.io/pepy/dt/sklearn_ensemble_cv)](https://pepy.tech/project/sklearn_ensemble_cv)

# Ensemble-cross-validation


`sklearn_ensemble_cv` is a Python module ([[Github]](https://github.com/jaydu1/ensemble-cross-validation/)) for performing accurate and efficient ensemble cross-validation methods from various [projects](https://jaydu1.github.io/overparameterized-ensembling/).


## Features
- The module builds on `scikit-learn`/`sklearn` to provide the most flexibility on various base predictors.
- The module includes functions for creating ensembles of models, training the ensembles using cross-validation, and making predictions with the ensembles. 
- The module also includes utilities for evaluating the performance of the ensembles and the individual models that make up the ensembles.


```python
from sklearn.tree import DecisionTreeRegressor
from sklearn_ensemble_cv import ECV

# Hyperparameters for the base regressor
grid_regr = {    
    'max_depth':np.array([6,7], dtype=int), 
    }
# Hyperparameters for the ensemble
grid_ensemble = {
    'max_features':np.array([0.9,1.]),
    'max_samples':np.array([0.6,0.7]),
}

# Build 50 trees and get estimates until 100 trees
res_ecv, info_ecv = ECV(
    X_train, y_train, DecisionTreeRegressor, grid_regr, grid_ensemble, 
    M=50, M_max=100, return_df=True
)
```

It currently supports bagging- and subagging-type ensembles under square loss.
The hyperparameters of the base predictor are listed at [`sklearn.tree.DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) and the hyperparameters of the ensemble are listed at [`sklearn.ensemble.BaggingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html).
Using other sklearn Regressors (`regr.is_regressor = True`) as base predictors is also supported.

# Cross-validation methods

This project is currently in development. More CV methods will be added shortly.

- [x] split CV
- [x] K-fold CV
- [x] ECV
- [x] GCV
- [x] CGCV
- [x] CGCV non-square loss
- [ ] ALOCV

# Usage


Check out Jupyter Notebooks in the [tutorials](https://github.com/jaydu1/ensemble-cross-validation/blob/main/tutorials) folder:

Name | Description
---|---
[basics.ipynb](https://github.com/jaydu1/ensemble-cross-validation/blob/main/tutorials/basics.ipynb) | Basics about how to apply ECV/CGCV on risk estimation and hyperparameter tuning for ensemble learning. 
[cgcv_l1_huber.ipynb](https://github.com/jaydu1/ensemble-cross-validation/blob/main/tutorials/cgcv_l1_huber.ipynb) | Custom CGCV for M-estimator: l1-regularized Huber ensembles. 
[multitask.ipynb](https://github.com/jaydu1/ensemble-cross-validation/blob/main/tutorials/multitask.ipynb) | Apply ECV on risk estimation and hyperparameter tuning for multi-task ensemble learning.

The code is tested with `scikit-learn == 1.3.1`.

The [document](https://jaydu1.github.io/overparameterized-ensembling/sklearn-ensemble-cv/docs/index) is available.

The module can be installed via PyPI:
```cmd
pip install sklearn-ensemble-cv
```