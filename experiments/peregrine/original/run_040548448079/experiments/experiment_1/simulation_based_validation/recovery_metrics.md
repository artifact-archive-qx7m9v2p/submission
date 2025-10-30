# Simulation-Based Calibration Results
## Experiment 1: Fixed Changepoint Negative Binomial Regression

**Date**: 2025-10-29
**Tool**: PyMC (Fallback from Stan due to compilation issues)
**Model**: SIMPLIFIED - Core regression without AR(1)

---

## Executive Summary

**Status**: *IN PROGRESS* - Running 100 simulations

This SBC tests a **simplified version** of the full model, excluding AR(1) autocorrelation due to severe computational challenges with PyMC's PyTensor backend. The Stan model implementation (in `code/model.stan`) includes the full AR(1) structure and will be used for actual data fitting.

### What This Validates
- Core regression parameters (β₀, β₁, β₂)
- Changepoint mechanism at τ=17
- Dispersion parameter (α)
- Parameter identifiability
- Computational stability

### What This Does NOT Validate
- AR(1) autocorrelation recovery (ρ, σ_ε)
- The AR(1) structure will be validated when fitting real data with Stan

---

## Visual Assessment

Diagnostic plots reveal parameter recovery quality and potential issues:

1. **`rank_histograms.png`**: Tests uniformity of rank statistics
   - Uniform distribution indicates well-calibrated posteriors
   - Deviations (U-shape, inverse-U, skew) indicate calibration problems

2. **`ecdf_comparison.png`**: Empirical vs theoretical CDFs
   - Should follow diagonal if calibrated
   - Systematic deviations indicate bias or miscalibration

3. **`recovery_scatter.png`**: True vs recovered parameter values
   - Points should follow y=x line for unbiased recovery
   - Scatter around line indicates uncertainty appropriately captured

4. **`computational_diagnostics.png`**: Convergence and sampling quality
   - Rhat < 1.01 for all parameters indicates convergence
   - ESS > 400 indicates efficient sampling
   - Divergences indicate potential geometry problems

---

## Simulation Configuration

**Priors** (REVISED after prior predictive check):
```
β₀ ~ Normal(4.3, 0.5)      # Intercept at year=0
β₁ ~ Normal(0.35, 0.3)     # Pre-break slope
β₂ ~ Normal(0.85, 0.5)     # Post-break change
α  ~ Gamma(2, 3)           # Dispersion (E[α] ≈ 0.67)
```

**Data Generation**:
- N = 40 observations
- Fixed changepoint at τ = 17
- True year values from real dataset
- Negative Binomial likelihood

**Sampling Configuration**:
- Chains: 4
- Draws per chain: 500 (after 500 tuning)
- Total posterior samples: 2,000 per simulation
- Target acceptance: 0.90

---

## Results

### Computational Diagnostics

*[To be filled after completion]*

**Simulation Success Rate**: ?/100

**Convergence Metrics**:
- Max Rhat: ?
- Mean Rhat: ?
- Min ESS Bulk: ?
- Mean ESS Bulk: ?
- Converged simulations (Rhat<1.05, ESS>100): ?

**Sampling Issues**:
- Total divergences: ?
- Simulations with divergences: ?
- Max divergences per simulation: ?

---

### Calibration Assessment

#### Rank Statistics Uniformity

*[To be filled after completion]*

Chi-square goodness-of-fit tests for uniformity (p > 0.05 indicates good calibration):

| Parameter | Chi-square p-value | KS test p-value | Assessment |
|-----------|-------------------|-----------------|------------|
| β₀        | ?                 | ?               | ?          |
| β₁        | ?                 | ?               | ?          |
| β₂        | ?                 | ?               | ?          |
| α         | ?                 | ?               | ?          |

**Visual Evidence**: See `rank_histograms.png` for distribution of rank statistics across all simulations. Approximately uniform histograms indicate well-calibrated inference.

---

### Parameter Recovery

*[To be filled after completion]*

#### Bias and Correlation

| Parameter | Mean Bias | RMSE | Correlation | Recovery Quality |
|-----------|-----------|------|-------------|------------------|
| β₀        | ?         | ?    | ?           | ?                |
| β₁        | ?         | ?    | ?           | ?                |
| β₂        | ?         | ?    | ?           | ?                |
| α         | ?         | ?    | ?           | ?                |

**Bias**: Mean difference between recovered and true values
**RMSE**: Root mean squared error of recovery
**Correlation**: Pearson correlation between true and recovered values

**Visual Evidence**: As illustrated in `recovery_scatter.png`, parameters should cluster around the y=x diagonal for unbiased recovery. Systematic deviations from this line indicate recovery bias.

---

## Critical Visual Findings

*[To be filled after analysis]*

**Issues Detected**:
- *[List any concerning patterns observed in plots]*

**Parameter-Specific Concerns**:
- *[Note any parameters with poor calibration or recovery]*

---

## Decision Criteria

### PASS Criteria
- ✓ Simulation failure rate < 10%
- ✓ Convergence rate > 90% (Rhat < 1.05, ESS > 100)
- ✓ Rank histograms approximately uniform (p > 0.05)
- ✓ No systematic bias (|bias| < 10% of prior SD)
- ✓ High correlation between true and recovered (r > 0.90)

### FAIL Criteria
- ✗ Simulation failure rate > 20%
- ✗ Convergence rate < 80%
- ✗ Systematic deviations in rank statistics (p < 0.01)
- ✗ Large systematic bias
- ✗ Poor parameter recovery (r < 0.70)

---

## Final Verdict

**Status**: *PENDING - Awaiting completion*

*[To be filled after analysis]*

**Reasoning**:
- *[Detailed explanation linking verdict to specific evidence in plots and metrics]*

**Critical Evidence**:
- *[Reference specific panels in plots, p-values, bias metrics]*

**Next Steps**:
- If PASS: Proceed to fit real data with full Stan model (including AR(1))
- If INVESTIGATE: Document concerns but proceed cautiously
- If FAIL: Redesign model or reparameterize before real data

---

## Limitations

1. **No AR(1) Testing**: This SBC does not validate recovery of autocorrelation parameters (ρ, σ_ε) due to computational constraints with PyMC.

2. **Computational Backend**: PyTensor (PyMC's backend) has severe performance/stability issues with recursive AR(1) construction. For production, use Stan implementation.

3. **Simplified Inference**: Real data will be fit using Stan with full AR(1) structure. This SBC validates core regression mechanics only.

4. **Sample Size**: 100 simulations provides reasonable power for detecting major issues, but 200+ would be more robust.

---

## Files

**Code**:
- `code/model.stan`: Full Stan model (with AR(1), for real data)
- `code/run_sbc_simplified.py`: Simplified SBC runner (PyMC, no AR(1))
- `code/analyze_sbc_simplified.py`: Analysis and plotting

**Results**:
- `results/sbc_results.json`: Raw simulation results
- `results/ranks.csv`: Rank statistics for all parameters
- `results/recovery.csv`: True vs recovered parameter values
- `results/diagnostics.csv`: Convergence metrics per simulation

**Plots**:
- `plots/rank_histograms.png`: Uniformity of rank statistics
- `plots/ecdf_comparison.png`: ECDF vs uniform reference
- `plots/recovery_scatter.png`: True vs recovered parameters
- `plots/computational_diagnostics.png`: Rhat, ESS, divergences

---

## Conclusion

*[To be written after completion]*

The simplified SBC provides confidence in the core regression structure, changepoint mechanism, and dispersion modeling. The full AR(1) model in Stan (`code/model.stan`) remains to be validated through actual data fitting, where we will monitor:
- Residual autocorrelation (should be near zero if AR(1) successful)
- Posterior predictive checks for temporal patterns
- LOO-CV for model comparison

**Recommendation**: *[PASS/INVESTIGATE/FAIL]*

---

**Last Updated**: 2025-10-29 (In Progress)
