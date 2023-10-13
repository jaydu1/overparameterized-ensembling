# Generalized equivalences between subsampling and ridge regularization


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

