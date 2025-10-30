# Prior Predictive Check - Experiment 1

**Status**: ✓ PASS
**Date**: 2025-10-29
**Model**: Standard Hierarchical Model (mu ~ N(0,50), tau ~ HalfCauchy(0,25))

## Quick Summary

Prior specification is **adequate for model fitting**. All observed data fall within reasonable prior predictive range (46-64 percentiles), with no extreme outliers or prior-data conflicts.

## Key Results

- **Prior predictive simulations**: 2,000 datasets generated
- **Coverage**: All 8 schools within 45-65 percentile range (perfect)
- **Plausibility**: 58.8% of datasets fully reasonable (|y| < 100)
- **Flexibility**: Prior supports both strong pooling (10.9%) and minimal pooling (56.0%)
- **Sensitivity**: Results relatively robust to prior specification choices

## Files

### Code
- `code/prior_predictive_check.py` - Main analysis script (Python)
- `code/prior_predictive.stan` - Stan model for reference

### Plots
1. `plots/parameter_priors.png` - Prior distributions vs observed data
2. `plots/prior_predictive_spaghetti.png` - 100 prior datasets overlaid
3. `plots/prior_predictive_coverage.png` - School-by-school percentiles
4. `plots/prior_predictive_summaries.png` - Four-panel statistical summaries
5. `plots/extreme_value_diagnostic.png` - Extreme value frequency check
6. `plots/prior_sensitivity.png` - Comparison of 5 alternative priors

### Report
- `findings.md` - Comprehensive assessment with detailed evidence

## Decision

**PASS - Proceed with model fitting**

Minor caveat: HalfCauchy heavy tails occasionally generate extreme tau values (mean=609, max=684k), but this is expected and acceptable. The likelihood will constrain these during MCMC.

## Next Step

Proceed to Simulation-Based Calibration (SBC) to verify the model can recover known parameters.
