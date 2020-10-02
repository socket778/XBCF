# XBCF python package

## Installation

### From PyPI

To install XBCF from PyPI use `pip install xbcausalforest`

### From source

#### linux/MacOS

For general installation run `./build_py.sh -d`.

If you are making changes to the C++ files in \xbart please download SWIG.
Once installed, run `./build_py.sh -s -d`

## Example
```{r code}
sweeps = 40
burn = 15
p_categorical = #depends on your data
model = XBCF(
    num_sweeps=sweeps,
    burnin=burn,
    max_depth=250,
    num_trees_pr=30,
    num_trees_trt=10,
    num_cutpoints=20,
    Nmin=1,
    p_categorical_pr=p_categorical,
    p_categorical_trt=p_categorical,
    tau_pr=0.6 * np.var(y) / 30,
    tau_trt=0.1 * np.var(y) / 10,
    no_split_penality="auto",
    parallel=True,
)
```
X and X1 are data (2darrays) used for the treatment and prognostic terms respectively.
For proper data processing please make sure to have categorical variables be the last columns of your data.

y is the outcome variable (1darray).

z is the binary treatment assignment (1darray).

```{r code2}
fit = model.fit(X, X1, y, z)

b = fit.b.transpose()

tauhats = sdy * obj.tauhats * (b[1] - b[0])
tauhats_mean = np.mean(tauhats[:, (burn) : (sweeps - 1)], axis=1)
```
The array tauhats_mean contains individual-level treatment effect estimates averaged over sweeps.