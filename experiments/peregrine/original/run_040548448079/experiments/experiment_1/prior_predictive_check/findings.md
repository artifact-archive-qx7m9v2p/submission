# Prior Predictive Check: Experiment 1
## Fixed Changepoint Negative Binomial Regression

**Date**: 2025-10-29
**Model**: Fixed changepoint NB regression with AR(1) errors
**Simulations**: 1,000 prior predictive datasets

---

## Executive Summary

**VERDICT: REVISE - Autocorrelation Prior Needs Adjustment**

The prior specification is largely sound and generates scientifically plausible data, but one critical issue requires attention:

- **PROBLEM**: The observed ACF(1) = 0.944 falls at the 100th percentile of the prior predictive distribution, indicating the prior on autocorrelation (ρ) is too conservative
- **IMPACT**: The model may struggle to capture the extremely high temporal correlation in the data
- **RECOMMENDATION**: Adjust ρ prior to Beta(12, 1) or Beta(15, 1) to allow for stronger autocorrelation while maintaining some regularization

All other aspects pass validation. After this single adjustment, proceed to model fitting.

---

## Visual Diagnostics Summary

Five diagnostic plots were created to assess prior plausibility and coverage:

1. **parameter_plausibility.png**: Marginal distributions of all 6 prior parameters
   - Purpose: Verify priors are centered appropriately and have reasonable spread

2. **prior_predictive_coverage.png**: Observed data overlaid on prior predictive envelope
   - Purpose: Check if observed trajectory falls within simulated data range

3. **summary_statistics_comparison.png**: Six key statistics (min, max, mean, var/mean, ACF, growth)
   - Purpose: Quantify where observed statistics fall in prior predictive distribution

4. **structural_patterns.png**: Four-panel diagnostic of structural features
   - Purpose: Assess slope changes, autocorrelation, overdispersion, and growth patterns

5. **range_diagnostic.png**: Detailed view of count range coverage
   - Purpose: Verify priors generate counts in plausible ranges

---

## Critical Checks Assessment

### 1. Range Check: PASS ✓
**Criterion**: Do 50% of prior draws include counts in [10, 400]?
**Result**: **99.1% of draws** cover this range

**Evidence** (`range_diagnostic.png`, `summary_statistics_comparison.png`):
- Observed minimum (19) at 88.7th percentile - well within prior predictive range
- Observed maximum (272) at 6.4th percentile - on lower end but reasonable
- Prior generates mostly plausible counts, with some extreme values (13.8% of sims had counts > 10,000)

**Interpretation**: The priors correctly allow for a wide range while centering on plausible values. The occasional extreme counts are not problematic (they occur in <15% of draws) and reflect genuine prior uncertainty.

---

### 2. Growth Pattern: PASS ✓
**Criterion**: Do >30% of draws show some positive growth?
**Result**: **90.6% of draws** show positive growth (ratio > 1)

**Evidence** (`structural_patterns.png` - lower right panel):
- Observed growth factor (8.5) at 40.4th percentile
- Prior distribution skewed toward growth, with long tail toward high growth
- Distribution appropriately captures both modest and extreme growth scenarios

**Interpretation**: The priors encode the expectation of growth while allowing for diverse patterns. The observed 745% total growth is well-represented in the prior predictive distribution.

---

### 3. Autocorrelation: FAIL ✗
**Criterion**: Do >50% of simulations show ACF(1) ∈ [0.6, 0.99]?
**Result**: **24.8% of draws** in this range (FAIL)

**Evidence** (`summary_statistics_comparison.png`, `structural_patterns.png` - upper right):
- **Critical finding**: Observed ACF(1) = 0.944 at **100.0th percentile** (EXTREME)
- Prior E[ρ] = 0.80, but realized ACF(1) distribution much lower (median ~0.6)
- Gap between ρ prior and realized ACF indicates model-prior mismatch

**Root Cause Analysis**:
The AR(1) model transforms ρ into observed ACF(1) through the likelihood. The current Beta(8, 2) prior on ρ (E = 0.8, SD = 0.12) generates realized autocorrelations that are systematically lower than observed. This happens because:
1. AR(1) errors are added to log(μ), not directly to counts
2. Negative Binomial sampling adds additional noise
3. These combine to "wash out" the autocorrelation structure

**Recommendation**:
- **Increase ρ prior concentration**: Use Beta(12, 1) giving E[ρ] ≈ 0.92, or Beta(15, 1) giving E[ρ] ≈ 0.94
- This shifts the prior predictive ACF(1) distribution upward to cover the observed value
- Alternative: Consider Beta(10, 1) as a middle ground (E[ρ] ≈ 0.91)

---

### 4. Structural Break Variety: PASS ✓
**Criterion**: Do >30% of draws show slope increase?
**Result**: **70.8% of draws** show slope increase > 0.1

**Evidence** (`structural_patterns.png` - upper left panel):
- Observed pre-break slope: 0.78 (log scale)
- Observed post-break slope: 0.96 (log scale)
- Prior allows diverse patterns: some with breaks, some without, some with decreases

**Interpretation**: The priors correctly encode uncertainty about whether a structural break exists and its magnitude. The observed pattern (modest slope increase) is well within the prior predictive distribution. Points above the diagonal line indicate slope increases, which dominate but don't overwhelm the prior.

---

### 5. Overdispersion: PASS ✓
**Criterion**: Is variance > mean in >80% of simulations?
**Result**: **99.8% of draws** are overdispersed

**Evidence** (`structural_patterns.png` - lower left panel):
- Observed variance/mean ratio: 66.3 at 8.0th percentile
- Prior generates high overdispersion (most draws have ratio 20-150)
- Gamma(2, 1/3) prior on α (E = 0.67) appropriately induces overdispersion

**Interpretation**: The Negative Binomial specification with informative α prior correctly generates overdispersed counts. The observed ratio is on the lower end, suggesting the data might be slightly less overdispersed than the prior expects, but this is not problematic - it means the prior is weakly informative and will let the data determine the dispersion level.

---

## Observed Data Position in Prior Predictive Distribution

| Statistic | Observed Value | Percentile | Status |
|-----------|---------------|------------|--------|
| Minimum count | 19 | 88.7th | GOOD ✓ |
| Maximum count | 272 | 6.4th | GOOD ✓ |
| Mean count | 109.45 | 21.0th | GOOD ✓ |
| Variance/Mean | 66.3 | 8.0th | GOOD ✓ |
| **ACF(1)** | **0.944** | **100.0th** | **EXTREME ✗** |
| Growth factor | 8.5 | 40.4th | GOOD ✓ |

**Interpretation**:
- "GOOD" = Observed value between 5th and 95th percentile (well-covered by prior)
- "EDGE" = Between 1st and 99th percentile (covered but at boundary)
- "EXTREME" = Outside 99th percentile (prior doesn't adequately cover observation)

The ACF(1) result is the sole concern. All other statistics indicate the priors generate data consistent with observations.

---

## Computational Stability Assessment

### Flags from 1,000 Simulations:

| Flag | Count | Percentage | Assessment |
|------|-------|------------|------------|
| Overflow warnings (log(μ) > 10) | 80 | 8.0% | Minor concern |
| Negative μ | 0 | 0.0% | Excellent ✓ |
| Extreme counts (>10,000) | 138 | 13.8% | Acceptable |
| Zero inflation (>50% zeros) | 6 | 0.6% | Excellent ✓ |
| NaN values | 0 | 0.0% | Excellent ✓ |

**Assessment**: No serious computational pathologies detected.

- **Overflow warnings**: 8% of draws have very large expected counts (exp(10) ≈ 22,000). This reflects genuine prior uncertainty about growth rates. Not problematic for MCMC.
- **Extreme counts**: Related to overflow warnings. The prior allows for extreme growth scenarios. Posterior will constrain these.
- **Zero inflation**: Negligible. Model correctly avoids generating unrealistic zero-heavy data.

The model is computationally stable for MCMC sampling.

---

## Key Visual Evidence

### Most Important Plot: `summary_statistics_comparison.png`
This plot reveals the ACF(1) problem clearly: the observed value (red line) falls completely outside the prior predictive distribution (blue histogram). All other statistics show good coverage.

### Second Most Important: `prior_predictive_coverage.png`
Despite the ACF issue, the prior predictive envelope broadly covers the observed trajectory. However, this plot has a scaling issue that obscures detail - the y-axis extends to 1.6 million due to extreme outlier simulations, making the actual data (max 272) appear as a flat line at the bottom. The observed counts fall well within the 90% interval (which would be visible with better y-axis limits).

### Third Most Important: `structural_patterns.png`
The four-panel view shows:
- Upper left: Slope changes are diverse and include the observed pattern
- Upper right: **ACF(1) problem clearly visible** - observed is far right of distribution
- Lower left: Overdispersion is well-captured
- Lower right: Growth patterns are appropriate

---

## Prior Specification Review

### Current Priors (from specification):
```
β_0 ~ Normal(4.3, 0.5)      # Intercept at year=0
β_1 ~ Normal(0.35, 0.3)     # Pre-break slope
β_2 ~ Normal(0.85, 0.5)     # Post-break slope increase
α ~ Gamma(2, 3)             # Dispersion (E = 0.67)
ρ ~ Beta(8, 2)              # AR(1) coefficient (E = 0.80)  ← NEEDS ADJUSTMENT
σ_ε ~ Exponential(2)        # AR(1) noise (E = 0.5)
```

### Realized Prior Draws (from 1,000 samples):
| Parameter | Mean | Std | Range |
|-----------|------|-----|-------|
| β_0 | 4.31 | 0.49 | [2.68, 6.23] |
| β_1 | 0.37 | 0.30 | [-0.53, 1.31] |
| β_2 | 0.85 | 0.49 | [-0.66, 2.81] |
| α | 0.67 | 0.46 | [0.01, 3.45] |
| ρ | 0.80 | 0.12 | [0.30, 1.00] |
| σ_ε | 0.50 | 0.50 | [0.00, 4.01] |

All parameters sample correctly from their specified distributions.

---

## Recommendations

### Required Action: Adjust Autocorrelation Prior

**Current**: ρ ~ Beta(8, 2)
**Recommended**: ρ ~ Beta(12, 1) or Beta(15, 1)

**Rationale**:
- Beta(12, 1): E[ρ] = 0.923, SD = 0.073 - moderately strong push toward high autocorrelation
- Beta(15, 1): E[ρ] = 0.938, SD = 0.061 - stronger push, closer to observed ACF(1) = 0.944
- Beta(10, 1): E[ρ] = 0.909, SD = 0.087 - conservative middle ground

**Testing the adjustment**:
After changing the prior, re-run prior predictive check and verify:
1. ACF(1) distribution now includes 0.944 within 95% interval
2. Other checks still pass (they should, as this change only affects temporal correlation)
3. At least 50% of draws show ACF(1) ∈ [0.6, 0.99]

### Optional Considerations

1. **Y-axis scaling in coverage plot**: The current plot is hard to read due to extreme outliers. Consider clipping y-axis to [0, 2000] or using log scale for visualization (implementation detail, doesn't affect validation).

2. **σ_ε prior**: The Exponential(2) prior (E = 0.5) generates some very large values (max = 4.01). This is probably fine - it reflects genuine uncertainty about innovation variance. However, if MCMC has divergences, consider Exponential(3) or Exponential(4) for tighter control.

3. **β_2 prior width**: Normal(0.85, 0.5) allows for slope decreases (β_2 < 0 in 5% of draws). If domain knowledge rules out post-break decreases, consider truncating at 0 or narrowing the SD. Not required for passing validation.

---

## Conclusion

The prior specification is **scientifically sound but requires one adjustment** before proceeding to model fitting:

**Passing elements**:
- ✓ Priors generate plausible count ranges (99% coverage of [10, 400])
- ✓ Growth patterns are well-represented (91% positive growth)
- ✓ Structural break variety is appropriate (71% show slope increases)
- ✓ Overdispersion is correctly induced (99.8% of draws)
- ✓ No computational pathologies (0% NaN, 0% negative μ)
- ✓ Observed statistics (except ACF) fall within reasonable percentiles

**Failing element**:
- ✗ Autocorrelation prior is too weak (observed ACF at 100th percentile)

**Next Steps**:
1. **Revise** ρ prior to Beta(12, 1), Beta(15, 1), or Beta(10, 1)
2. **Re-run** prior predictive check (quick - just 1,000 simulations)
3. **Verify** ACF(1) check now passes (>50% of draws in [0.6, 0.99])
4. **Proceed** to Simulation-Based Calibration once revised check passes

**Time estimate**: 5-10 minutes to revise and re-validate

---

## Appendix: Files Generated

### Code
- `/workspace/experiments/experiment_1/prior_predictive_check/code/prior_predictive_simulation.py` - Main simulation script
- `/workspace/experiments/experiment_1/prior_predictive_check/code/visualize_prior_predictive.py` - Visualization script
- `/workspace/experiments/experiment_1/prior_predictive_check/code/prior_samples.csv` - All 1,000 parameter draws
- `/workspace/experiments/experiment_1/prior_predictive_check/code/summary_statistics.csv` - Summary stats for all simulations
- `/workspace/experiments/experiment_1/prior_predictive_check/code/simulated_datasets_full.npy` - Complete simulated data (1000 × 40 array)

### Plots
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/parameter_plausibility.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/prior_predictive_coverage.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/summary_statistics_comparison.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/structural_patterns.png`
- `/workspace/experiments/experiment_1/prior_predictive_check/plots/range_diagnostic.png`

All outputs are reproducible via the provided scripts with seed=42.
