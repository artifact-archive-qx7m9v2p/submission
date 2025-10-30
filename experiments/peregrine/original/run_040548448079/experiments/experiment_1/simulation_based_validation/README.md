# Simulation-Based Calibration (SBC)
## Experiment 1: Fixed Changepoint Negative Binomial Regression

---

## Overview

This directory contains the Simulation-Based Calibration implementation for validating the Fixed Changepoint Negative Binomial Regression model. SBC is a critical safety check that verifies the model can recover known parameters before fitting real data.

### Purpose

**If a model can't recover known truth, it won't find unknown truth.**

SBC tests the complete inference pipeline:
1. Draw parameters from prior → 2. Generate synthetic data → 3. Fit model → 4. Check recovery

---

## Important Note: Simplified Model

Due to severe computational issues with PyMC's PyTensor backend when implementing AR(1) processes, this SBC tests a **simplified model without autocorrelation**.

### What This Tests
- ✓ Core regression structure (β₀, β₁, β₂)
- ✓ Changepoint mechanism at τ=17
- ✓ Dispersion parameter (α)
- ✓ Parameter identifiability
- ✓ Computational stability

### What This Does NOT Test
- ✗ AR(1) autocorrelation recovery (ρ, σ_ε)

### Why This Approach?

1. **PyTensor Limitations**: Recursive AR(1) construction causes PyTensor compilation failures
2. **Stan Unavailable**: CmdStan requires compilation tools not available in environment
3. **Pragmatic Solution**: Test core model structure now, validate AR(1) during real data fitting

The **full model with AR(1)** is implemented in `code/model.stan` and will be used for actual inference (requires Stan installation).

---

## Directory Structure

```
simulation_based_validation/
├── README.md                          # This file
├── recovery_metrics.md                # Results and assessment
├── sbc_run.log                        # Execution log
├── code/
│   ├── model.stan                     # Full Stan model (WITH AR1)
│   ├── run_sbc_simplified.py          # SBC runner (PyMC, no AR1)
│   └── analyze_sbc_simplified.py      # Analysis and plotting
├── results/
│   ├── sbc_results.json               # Raw results
│   ├── ranks.csv                      # Rank statistics
│   ├── recovery.csv                   # True vs recovered parameters
│   └── diagnostics.csv                # Convergence metrics
└── plots/
    ├── rank_histograms.png            # Calibration: uniformity check
    ├── ecdf_comparison.png            # Calibration: ECDF vs uniform
    ├── recovery_scatter.png           # Bias: true vs recovered
    └── computational_diagnostics.png  # Convergence: Rhat, ESS, divergences
```

---

## Model Specification

### Full Model (in `model.stan`)

**Observation Model**:
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = β_0 + β_1 × year_t + β_2 × I(t > τ) × (year_t - year_τ) + ε_t
```

**Autocorrelation** (NOT tested in this SBC):
```
ε_t ~ Normal(ρ × ε_{t-1}, σ_ε)
ε_1 ~ Normal(0, σ_ε / √(1 - ρ²))  # Stationary initialization
```

### Simplified Model (tested in SBC)

**Observation Model**:
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = β_0 + β_1 × year_t + β_2 × I(t > τ) × (year_t - year_τ)
```

No AR(1) errors—assumes independent observations conditional on regression structure.

---

## Priors (REVISED)

After prior predictive check, priors were revised to:

```
β₀ ~ Normal(4.3, 0.5)      # log(median count) at year=0
β₁ ~ Normal(0.35, 0.3)     # Pre-break slope
β₂ ~ Normal(0.85, 0.5)     # Additional post-break slope
α  ~ Gamma(2, 3)           # Dispersion, E[α] ≈ 0.67
ρ  ~ Beta(12, 1)           # AR1 coefficient (not tested in SBC)
σ_ε ~ Exponential(2)       # Innovation SD (not tested in SBC)
```

---

## Running the SBC

### Prerequisites

```bash
# Ensure PyMC and ArviZ are installed
export PYTHONPATH=/tmp/agent-home/.local/lib/python3.13/site-packages:$PYTHONPATH
```

### Execute SBC

```bash
cd code/
python run_sbc_simplified.py
```

**Configuration**:
- N_SIMULATIONS = 100
- N_CHAINS = 4
- N_DRAWS = 500 (per chain)
- N_TUNE = 500
- TARGET_ACCEPT = 0.90

**Expected Runtime**: 50-70 minutes (30-50 seconds per simulation)

### Analyze Results

After SBC completes:

```bash
python analyze_sbc_simplified.py
```

This generates:
- Diagnostic plots in `plots/`
- Summary statistics to console
- Updates `recovery_metrics.md` with verdict

---

## Interpretation Guide

### Rank Histograms

**What to look for**:
- Approximately uniform distribution across bins
- No systematic U-shape, inverse-U, or skew

**Red flags**:
- U-shape: Posteriors too narrow (overconfident)
- Inverse-U: Posteriors too wide (underconfident)
- Skew: Systematic bias in recovery

### ECDF Comparison

**What to look for**:
- Empirical CDF follows diagonal
- Stays within confidence bands

**Red flags**:
- Systematic deviation above diagonal: consistent overestimation
- Systematic deviation below diagonal: consistent underestimation

### Recovery Scatter

**What to look for**:
- Points cluster around y=x line
- High correlation (r > 0.90)
- Small RMSE relative to prior scale

**Red flags**:
- Systematic deviation from y=x: biased recovery
- High scatter: poor identifiability
- Low correlation: model can't learn parameter

### Computational Diagnostics

**What to look for**:
- Rhat < 1.01 for all parameters
- ESS > 400 (good), minimum 100 (acceptable)
- Few or no divergences

**Red flags**:
- Rhat > 1.05: non-convergence
- ESS < 100: inefficient sampling
- Many divergences: geometry problems

---

## Decision Criteria

### PASS
- Failure rate < 10%
- Convergence rate > 90%
- Rank histograms approximately uniform (p > 0.05)
- No systematic bias (|bias| < 10% of prior SD)
- High recovery correlation (r > 0.90)

### INVESTIGATE
- Minor calibration issues
- 10-20% failure rate
- Some parameters show small bias
- Proceed to real data with documented concerns

### FAIL
- Failure rate > 20%
- Systematic calibration failures
- Large bias in key parameters
- DO NOT proceed to real data—redesign model

---

## Known Limitations

1. **No AR(1) Validation**: Cannot test ρ and σ_ε recovery with current setup

2. **Different Implementations**: SBC uses PyMC (simplified), real data will use Stan (full model)

3. **Computational Backend**: PyTensor performance issues limit model complexity

4. **Sample Size**: 100 simulations adequate but not optimal (200+ preferred)

---

## Next Steps After SBC

### If PASS

1. Install Stan (CmdStan) for full model:
   ```bash
   pip install cmdstanpy
   python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
   ```

2. Compile full model:
   ```bash
   cd code/
   cmdstan model.stan
   ```

3. Fit real data with AR(1) structure

4. Validate AR(1) through:
   - Residual autocorrelation (should be ≈ 0)
   - Posterior predictive checks
   - LOO-CV diagnostics

### If INVESTIGATE

- Document specific concerns in `recovery_metrics.md`
- Monitor flagged parameters closely during real data fitting
- Consider alternative priors or reparameterization

### If FAIL

- Revisit model specification
- Check for identifiability issues
- Consider simpler model (e.g., no changepoint, or Poisson instead of NegBin)
- DO NOT fit real data until issues resolved

---

## Troubleshooting

### "PyTensor has no attribute 'scan'"
- This is why we use simplified model
- AR(1) requires scan, which is deprecated/unstable

### "g++ not detected" warning
- PyTensor cannot compile C code
- Uses Python fallback (slower but functional)
- Does not affect correctness, only speed

### Slow sampling (>60s per simulation)
- Expected with PyTensor Python fallback
- Total runtime: ~60 minutes for 100 simulations
- Consider reducing N_SIMULATIONS to 50 for testing

### High Rhat or low ESS
- May indicate model misspecification
- Check if specific parameter or dataset-dependent
- Document in recovery_metrics.md

---

## References

### Simulation-Based Calibration
- Talts et al. (2018): "Validating Bayesian Inference Algorithms with Simulation-Based Calibration"
- Cook et al. (2006): "Validation of Software for Bayesian Models Using Posterior Quantiles"

### This Model
- See `/workspace/experiments/experiment_1/metadata.md` for full specification
- Prior predictive check results in `/workspace/experiments/experiment_1/prior_predictive_check/`

---

## Contact / Issues

If SBC reveals problems:
1. Document in `recovery_metrics.md`
2. Check `/workspace/experiments/experiment_1/metadata.md` for falsification criteria
3. Consider whether issue is:
   - Statistical (model misspecification) → redesign model
   - Computational (sampling issues) → adjust sampler settings
   - Implementation (bug) → review code

---

**Last Updated**: 2025-10-29
**Status**: Running (24/100 simulations complete as of last check)
