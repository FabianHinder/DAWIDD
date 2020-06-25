# Dynamic Adapting Window Independence Drift Detection (DAWIDD)

This repository contains the implementation of the methods proposed in the paper [Towards non-parametric drift detection via Dynamic Adapting Window Independence Drift Detection (DAWIDD)](Paper.pdf) by Fabian Hinder, André Artelt and Barbara Hammer (accepted at ICML 2020)
- The *Dynamic Adapting Window Independence Drift Detection (DAWIDD)* is implemented in [DAWIDD.py](DAWIDD.py). If your want to use a different/custom test for independence, you have to overwrite the method `test_independence`.
- The Hellinger-Distance-Drift-Detection-Method (HDDDM) is implemented in [HDDDM.py](HDDDM.py).
- The experiments for comparing different drift detection methods are implemented in [experiments_driftdetectors.py](experiments_driftdetectors.py).

## Requirements

- Python >= 3.6
- Packages as listed in [REQUIREMENTS.txt](REQUIREMENTS.txt)

## Third party components

- [kernel_two_sample_test.py](https://github.com/emanuele/kernel_two_sample_test/blob/master/kernel_two_sample_test.py) is taken from [GitHub](https://github.com/emanuele/kernel_two_sample_test) and implements the kernel two-sample tests as in Gretton et al 2012 (JMLR).


## How to cite

You can cite the version on TODO.
