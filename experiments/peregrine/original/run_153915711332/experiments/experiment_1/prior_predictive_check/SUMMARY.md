# Prior Predictive Check Summary

**Status:** FAIL
**Date:** 2025-10-29
**Decision:** Priors require tightening before proceeding

---

## Quick Verdict

The current priors are **too diffuse** and generate implausibly extreme data:
- 0.4% of prior predictive counts exceed 10,000 (observed max: 272)
- Extreme maximum of 175,837 observed in prior predictive samples
- Prior predictive mean is 4x higher than observed mean (419 vs 109)

**Root causes:**
1. `sigma_eta ~ Exp(10)` is too wide - allows volatile random walks that compound over 40 time steps
2. `phi ~ Exp(0.1)` is too diffuse - allows both extreme overdispersion and near-Poisson behavior

---

## Recommended Fixes

```
# Change these two priors:
σ_η ~ Exponential(20)    # Was: Exp(10) - Tighter around 0.05
φ ~ Exponential(0.05)    # Was: Exp(0.1) - Tighter around 20

# Keep these:
δ ~ Normal(0.05, 0.02)   # GOOD - appropriate drift
η_1 ~ Normal(log(50), 1) # GOOD - appropriate initial state
```

---

## Key Evidence

### Visual Evidence
All plots in `/workspace/experiments/experiment_1/prior_predictive_check/plots/`:

1. **prior_predictive_trajectories.png** - Prior 95% CI balloons to 7000 by t=40
2. **prior_predictive_coverage.png** - Observed data at extreme left tail of prior distribution
3. **computational_red_flags.png** - Shows clustering of extreme values at high sigma_eta

### Numerical Evidence
From `/workspace/experiments/experiment_1/prior_predictive_check/code/prior_predictive_summary.json`:

```
Prior predictive mean count: 419 (observed: 109)
Prior predictive max (95%):  11,610 (observed: 272)
Growth factor (median):      6.6x (observed: 8.45x)
Extreme counts (>10k):       0.40% of samples
Extreme growth (>100x):      1.6% of samples
```

---

## Why This Matters

In dynamic models, **priors compound over time**:
- Even modest `sigma_eta = 0.1` accumulates over 40 steps
- Cumulative effect: exp(Σ innovations) can explode
- Result: Marginal priors that seem fine create extreme joint distributions

**The lesson:** Prior predictive checks are essential for time series models!

---

## Next Steps

1. Update priors in `/workspace/experiments/experiment_1/metadata.md`
2. Re-run this prior predictive check
3. Verify: <0.1% extreme counts, <0.5% extreme growth
4. Only after PASS: Proceed to simulation-based calibration

---

## Files Generated

**Documentation:**
- `findings.md` - Full detailed analysis (14KB)
- `SUMMARY.md` - This quick reference

**Code:**
- `code/run_prior_predictive_numpy.py` - Sampling script
- `code/visualize_prior_predictive.py` - Plotting script
- `code/prior_predictive_model.stan` - Stan model reference
- `code/prior_samples.npz` - Saved samples
- `code/prior_predictive_summary.json` - Numerical summary

**Plots (6 diagnostic visualizations):**
- `plots/parameter_prior_marginals.png`
- `plots/prior_predictive_trajectories.png`
- `plots/prior_predictive_coverage.png`
- `plots/computational_red_flags.png`
- `plots/latent_state_prior.png`
- `plots/joint_prior_diagnostics.png`
