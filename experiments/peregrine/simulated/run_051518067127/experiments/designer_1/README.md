# Model Designer 1: Count Likelihood Families

## Output Summary

**Designer Focus**: Bayesian models using appropriate count likelihood families (Negative Binomial, hierarchical structures, AR processes)

**Main Deliverable**: `/workspace/experiments/designer_1/proposed_models.md`

## Three Proposed Model Classes

### Model 1: Negative Binomial GLM with Quadratic Trend
- **Hypothesis**: Overdispersion is intrinsic, constant across time
- **Priority**: 1 (Baseline)
- **Implementation**: Stan, ~30 seconds
- **Key falsification**: Residual ACF > 0.5 → temporal structure not captured

### Model 2: Hierarchical NB with Time-Varying Dispersion
- **Hypothesis**: Overdispersion evolves across periods (early: 0.68, middle: 13.11, late: 7.23)
- **Priority**: 3 (Most complex)
- **Implementation**: Stan, ~2-3x Model 1 time
- **Key falsification**: Period phi posteriors overlap >80% → no evidence for variation

### Model 3: NB with AR(1) Latent Process
- **Hypothesis**: Autocorrelation (ACF=0.971) is primary mechanism
- **Priority**: 2 (Address temporal structure)
- **Implementation**: Stan, ~5-10x Model 1 time
- **Key falsification**: Posterior rho < 0.3 → autocorrelation not important

## Critical Features

All models include:
- Full Bayesian specification with priors
- Stan implementation (CmdStanPy)
- `log_lik` in generated quantities for LOO-CV
- Explicit falsification criteria
- Convergence requirements (Rhat < 1.01, ESS > 400)

## Decision Tree

1. Fit all 3 models in parallel
2. Compare LOO-CV ELPD (difference >3 is meaningful)
3. Check posterior predictive diagnostics (variance, ACF, coverage)
4. If all fail → pivot to state-space, GP, or changepoint models

## Emergency Pivots Documented

- State-space with random walk
- Conway-Maxwell-Poisson (flexible mean-variance)
- Gaussian process on latent intensity
- Piecewise regression with changepoints

## Key Files

- `proposed_models.md`: Full model specifications (21k+ words)
- `README.md`: This navigation guide

## Success Criteria

A good model:
- Converges (Rhat < 1.01, no divergences)
- Posterior predictive variance within 20% of data
- Residual ACF < 0.4 (or explicitly modeled)
- Well-calibrated uncertainty (empirical coverage ≈ nominal)

A failed model → we learn what's wrong → pivot intelligently
