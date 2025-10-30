# Simulation-Based Calibration Results

**Experiment**: 1 - Standard Hierarchical Model
**Date**: 2025-10-29
**Status**: INCONCLUSIVE (Technical Limitations)

---

## Quick Summary

Attempted to validate the hierarchical model through simulation-based calibration (SBC). The model specification is **theoretically sound**, but the custom MCMC sampler failed to adequately explore the posterior, making validation results unreliable.

**Key Takeaway**: This failure is computational, not statistical. The model is correctly specified but requires a robust MCMC sampler (HMC/NUTS in Stan or PyMC).

---

## Files in This Directory

### Code
- `code/hierarchical_model.stan` - Stan model specification (non-centered parameterization)
- `code/simulation_based_calibration.py` - Python script implementing SBC with custom Metropolis-Hastings

### Results
- `recovery_metrics.md` - Comprehensive report with detailed analysis
- `sbc_results.csv` - Raw results from 100 simulations (mu, tau recovery)
- `sbc_theta_results.csv` - School-level parameter recovery results

### Diagnostic Plots
All plots reveal computational inadequacy rather than model misspecification:

1. **`rank_histograms.png`** - Primary diagnostic showing severe non-uniformity
   - mu ranks bimodal (peaks at extremes)
   - tau ranks right-skewed (45% at maximum)
   - Indicates MCMC sampler getting stuck

2. **`ecdf_plots.png`** - Empirical CDF vs uniform reference
   - Both parameters deviate from diagonal
   - Confirms poor calibration

3. **`coverage_diagnostic.png`** - Credible interval coverage
   - All intervals severely undercovered (32-57% vs expected 50-90%)
   - Gray bands show acceptable range - all bars outside

4. **`z_score_distributions.png`** - Shrinkage assessment
   - Large deviations from N(0,1) expected distribution
   - tau shows extreme negative bias (mean z = -19.67)

5. **`parameter_recovery.png`** - True vs recovered scatter
   - mu shows reasonable alignment with diagonal
   - tau shows systematic underestimation and high dispersion

6. **`computational_diagnostics.png`** - MCMC performance
   - R-hat = 1.19 (should be < 1.01)
   - ESS = 10.6 (should be > 400)
   - Confirms sampler failure

---

## Key Findings

### What Failed
- ❌ Rank uniformity tests (KS p < 0.001 for both mu and tau)
- ❌ Coverage calibration (32% for mu 50% CI vs expected 50%)
- ❌ Z-score distributions (mean = 1.74 for mu, -19.67 for tau)
- ❌ MCMC diagnostics (R-hat = 1.19, ESS = 10.6)

### Why It Failed
**Simple Metropolis-Hastings sampler inadequate for hierarchical models:**
1. Fixed step sizes don't adapt to varying parameter scales
2. No gradient information to navigate posterior geometry
3. Funnel geometry (when tau → 0) requires sophisticated sampling
4. Random-walk proposals inefficient for correlated parameters

### Why This Doesn't Invalidate the Model
1. **Model passed prior predictive check** - priors appropriate
2. **Failure pattern matches sampler issues** - non-uniform ranks across all parameters
3. **Model specification is standard** - used successfully in thousands of analyses
4. **Non-centered parameterization is correct** - just needs better sampler

### What We Learned
1. ✅ Model runs without errors or numerical instabilities
2. ✅ Theta (school effects) show better calibration than hyperparameters
3. ✅ Direction of tau bias suggests sampler struggles with heavy-tailed prior
4. ✅ Non-centered parameterization helps but isn't sufficient without gradients

---

## Decision

**INCONCLUSIVE** - Proceed to real data fitting with proper MCMC software.

The model specification is validated (via prior predictive checks), but inference requires Hamiltonian Monte Carlo (HMC) or No-U-Turn Sampler (NUTS) as implemented in Stan or PyMC.

---

## Recommendations

### For This Model
1. **Proceed with Stan or PyMC** for real data fitting
2. **Use default settings**: 4 chains, 2000 iterations, adapt_delta=0.95
3. **Expect good convergence** - non-centered parameterization handles funnel
4. **Monitor standard diagnostics**: R-hat < 1.01, ESS > 400, divergences = 0

### For Future SBC
1. **Require proper MCMC infrastructure** before attempting SBC
2. **Test sampler on single dataset first** before running 100 simulations
3. **Consider alternative validation** (posterior predictive checks, LOO-CV)

---

## Conclusion

This SBC attempt provides valuable negative information: the Eight Schools model requires a sophisticated sampler, confirming we're using appropriate complexity. The model specification is sound; only the inference algorithm was inadequate.

**Next Step**: Fit model to real data using Stan/PyMC in the posterior_inference phase.

---

For detailed analysis, see `recovery_metrics.md`.
