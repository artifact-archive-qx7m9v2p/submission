# Prior Predictive Check - Experiment 1: Negative Binomial Quadratic

## Summary

**Model:** Negative Binomial with quadratic time trend
**Decision:** ADJUST (priors need tightening)
**Status:** Model structure is sound; requires prior refinement before SBC

---

## Quick Findings

- **Problem Identified:** Priors too vague, generating extreme predictions (max >42,000 vs observed max 272)
- **Root Cause:** Quadratic coefficient β₂~Normal(0.3, 0.2) too wide; allows explosive growth
- **Solution:** Tighten priors, especially β₂ σ from 0.2→0.1
- **Next Step:** Re-run prior predictive check with adjusted priors

---

## Directory Structure

```
/workspace/experiments/experiment_1/prior_predictive_check/
├── code/
│   ├── negbinom_quadratic_prior_pred.stan  # Stan model (not used - CmdStan unavailable)
│   ├── run_prior_check_numpy.py            # Working implementation (NumPy/SciPy)
│   ├── run_prior_check_pymc.py             # Alternative PyMC version (not used)
│   └── run_prior_check.py                  # Original CmdStanPy version (not used)
│
├── plots/                                   # 6 diagnostic visualizations (PNG, 300 DPI)
│   ├── parameter_plausibility.png          # Marginal prior distributions
│   ├── prior_predictive_trajectories.png   # Spaghetti plot with observed data
│   ├── prior_predictive_coverage.png       # Coverage diagnostics (4-panel)
│   ├── expected_value_trajectories.png     # Expected values with credible intervals
│   ├── parameter_space_coverage.png        # Pairwise parameter plots
│   └── growth_pattern_diversity.png        # Curvature classification
│
├── findings.md                              # Comprehensive assessment and recommendations
└── README.md                                # This file
```

---

## Key Statistics

- **Samples Generated:** 1000 parameter sets × 40 observations = 40,000 simulated counts
- **Domain Violations:** 0 (no negative counts)
- **Extreme Values:** 21 counts >10,000 detected (computational risk)
- **Coverage:** 65.8% cover observed min, 97.1% cover observed max
- **Plausibility:** 89.4% of simulations have mean in [10, 500] range

---

## Recommended Prior Adjustments

| Parameter | Current | Proposed | Change |
|-----------|---------|----------|--------|
| β₀ | Normal(4.7, 0.5) | Normal(4.7, 0.3) | Tighter |
| β₁ | Normal(0.8, 0.3) | Normal(0.8, 0.2) | Tighter |
| β₂ | Normal(0.3, 0.2) | Normal(0.3, 0.1) | **Critical: tighter** |
| φ | Gamma(2, 0.5) | Gamma(2, 0.5) | No change |

---

## How to Reproduce

```bash
cd /workspace/experiments/experiment_1/prior_predictive_check/code
python run_prior_check_numpy.py
```

Output: 6 diagnostic plots in `../plots/` and console diagnostics

---

## Visual Highlights

1. **prior_predictive_trajectories.png** - Shows the main problem: most trajectories reasonable, but visible subset explodes to >10,000
2. **expected_value_trajectories.png** - Demonstrates systematic over-prediction at late time points (median 4× observed)
3. **growth_pattern_diversity.png** - 93% of trajectories show upward curvature, contributing to explosive growth

---

## Assessor Notes

This prior predictive check successfully identified a critical issue before model fitting:
- Priors mathematically correct but scientifically too vague
- Quadratic + exponential link + wide priors = explosive predictions
- Adjustment straightforward: reduce prior standard deviations
- Model class worth keeping - problem is prior scale, not structure

See `findings.md` for complete analysis and decision rationale.
