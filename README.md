# Multi-group Gaussian process (MGGP)

The multi-group Gaussian process (MGGP) is a generalization of a traditional GP to the setting in which observations are partitioned into a set of known subgroups.

## Installation

The MGGP software can be installed with `pip`:

`pip install multigroupGP`

## Usage

Given an `n x p` matrix `X` of explanatory variables and an `n`-vector `y` containing responses, a GP can be fit as follows:

```python
from multigroupGP import GP, rbf_kernel
gp = GP(kernel=rbf_kernel)
gp.fit(X, y)
```

After fitting, predictions can be made for a test matrix `Xtest` as follows:

```python
gp.predict(Xtest)
```

## Example
