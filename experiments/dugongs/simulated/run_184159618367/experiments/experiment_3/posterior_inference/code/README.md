# Fitting Code for Log-Log Power Law Model

This directory contains the code used to fit the Log-Log Power Law model to the data.

## Files

### `loglog_model.stan`
Stan model specification for the power law model:
- Model: log(Y) ~ Normal(α + β×log(x), σ)
- Includes proper log-likelihood computation for LOO-CV
- Generated quantities for posterior predictive checks

**Note**: This file was prepared for Stan/CmdStanPy but was not used due to system constraints (CmdStan not available).

### `fit_model_pymc.py`
PyMC implementation used for actual fitting:
- Loads and log-transforms data
- Fits model using NUTS sampler (4 chains, 2000 iterations)
- Generates diagnostic plots
- Saves ArviZ InferenceData with log-likelihood

**This is the implementation that was actually used.**

### `fit_model.py`
CmdStanPy implementation (not used):
- Similar to PyMC version but uses Stan backend
- Prepared for potential use but superseded by PyMC version

## Usage

To refit the model:

```bash
export PYTHONPATH="/tmp/agent-home/.local/lib/python3.13/site-packages:$PYTHONPATH"
python fit_model_pymc.py
```

Or use the simplified inline version documented in the convergence report.

## Dependencies

- PyMC 5.26.1
- ArviZ 0.22.0
- NumPy, Pandas, Matplotlib, SciPy
- PyTensor 2.35.1

## Model Specification

```
Priors:
  α ~ Normal(0.6, 0.3)
  β ~ Normal(0.12, 0.05)
  σ ~ HalfCauchy(0, 0.05)

Likelihood:
  log(Y_i) ~ Normal(α + β×log(x_i), σ)

Equivalent on original scale:
  Y_i = exp(α) × x_i^β × ε_i
  where log(ε_i) ~ Normal(0, σ)
```

## Outputs

All outputs are saved to parent directories:
- `../diagnostics/` - NetCDF file, parameter summaries, convergence report
- `../plots/` - Diagnostic visualizations
- `../inference_summary.md` - Comprehensive results report
