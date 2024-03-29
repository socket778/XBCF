# Accelerated Bayesian Causal Forests (XBCF)

## About

This package implements the Accelerated Bayesian Causal Forests approach for conditional average treatment effect estimation; the manuscript is available [here](https://math.la.asu.edu/~prhahn/XBCF.pdf). This approach builds on the methodology behind Bayesian Causal Forests outlined in [Hahn et al.](https://projecteuclid.org/euclid.ba/1580461461) (2020) and incorporates several improvements to Bayesian Additive Regression Trees implemented by [He et al.](http://proceedings.mlr.press/v89/he19a.html) (2019).

This package is based on the source code of the [XBART](https://github.com/JingyuHe/XBART) package and was originally developed as a branch of that repository.

## Installation


### R
To install the package, run from the R console:

```
library(devtools)

install_github("socket778/XBCF")
```

### Python

To install XBCF from PyPI use `pip install xbcausalforest`
