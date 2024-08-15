# Implicit regularization paths of weighted neural representations

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

