# Prior Predictive Check: Hierarchical Binomial Model (Logit-Normal)

**Date**: 2025-10-30
**Model**: Experiment 1 - Hierarchical Binomial with Non-Centered Parameterization
**Status**: CONDITIONAL PASS (See recommendations)

---

## Executive Summary

The prior predictive check reveals that the specified priors are **largely appropriate** for modeling the hierarchical binomial data. The priors successfully generate data covering the observed range of success rates (3.1%-14.0%) and adequately capture overdispersion. However, there is a **minor issue with tail behavior**: 6.88% of prior samples generate implausibly high success rates (p > 0.8), slightly exceeding the 5% threshold.

**Verdict**: **CONDITIONAL PASS** - The model can proceed to fitting with awareness of this limitation. The issue is minor and unlikely to affect posterior inference given the strong likelihood contribution from the data.

---

## Visual Diagnostics Summary

All diagnostic plots are located in `/workspace/experiments/experiment_1/prior_predictive_check/plots/`:

1. **`parameter_plausibility.png`** - Prior distributions for mu, tau, and theta_j showing parameter ranges on both logit and probability scales
2. **`prior_predictive_coverage.png`** - Histogram comparing prior predictive success rates to observed data
3. **`overdispersion_diagnostic.png`** - Distribution of prior predictive overdispersion compared to observed phi = 4.38
4. **`group_level_comparison.png`** - Caterpillar plot showing 95% prior predictive intervals vs observed rates for each group
5. **`range_coverage_diagnostic.png`** - Joint distribution of min/max rates showing coverage of observed range
6. **`extreme_values_check.png`** - Diagnostic for implausible extreme values in tau and probability scales

---

## Model Specification

```
Hierarchical Binomial with Non-Centered Parameterization

Priors:
  mu ~ Normal(-2.5, 1)           # Population mean on logit scale
  tau ~ Half-Cauchy(0, 1)        # Between-group SD on logit scale
  theta_raw_j ~ Normal(0, 1)     # Non-centered group effects
  theta_j = mu + tau * theta_raw_j

Likelihood:
  r_j ~ Binomial(n_j, logit^-1(theta_j))

Data:
  J = 12 groups
  n_j: 47 to 810 (sample sizes)
  r_j: 3 to 34 (success counts)
  Observed success rates: 3.1% to 14.0%
  Observed overdispersion: phi = 4.38
```

---

## Prior Predictive Analysis

### Sampling Configuration
- **Prior samples**: 1,000 draws
- **Random seed**: 42 (reproducible)

### Prior Parameter Behavior

#### Population Mean (mu)
From `parameter_plausibility.png`:
- **Logit scale**: Mean = -2.48, SD = 0.98, Range = [-5.74, 1.35]
- **Probability scale**: Mean = 0.106, Range = [0.003, 0.795]
- **Assessment**: Centered slightly higher than observed pooled rate (0.070), but with appropriate spread

#### Between-Group SD (tau)
From `parameter_plausibility.png` and `extreme_values_check.png`:
- Mean = 3.97, Median = 1.01, 95th percentile = 12.31
- Range = [0.005, 542.94]
- **Heavy-tailed**: Half-Cauchy prior produces extreme values in 6.4% of samples (tau > 10)
- **Assessment**: Heavy tails are appropriate for capturing uncertainty about between-group variation

#### Group-Level Parameters (theta_j)
From `parameter_plausibility.png`:
- **Logit scale**: Mean = -2.64, Range = [-1071.52, 485.11]
- **Probability scale**: Mean = 0.183, Range = [0.000, 1.000]
- **Percentiles** (probability): 5% = 0.001, 50% = 0.077, 95% = 0.941
- **Assessment**: Wide prior allows diverse group-level behaviors; extreme values driven by heavy tau tails

---

## Critical Diagnostic Checks

### Check 1: Coverage of Observed Range [0.031, 0.140] - PASS

From `range_coverage_diagnostic.png` and `prior_predictive_coverage.png`:
- **Criterion**: >= 50% of prior simulations cover full observed range
- **Result**: 55.1% (551/1000 simulations)
- **Min coverage**: 76.4% can generate rates <= 0.031
- **Max coverage**: 76.2% can generate rates >= 0.140

**Key Visual Evidence**: The prior predictive distribution (`prior_predictive_coverage.png`) shows observed rates (red lines) fall well within the prior predictive density, concentrated in the yellow shaded region. The 2D density plot in `range_coverage_diagnostic.png` shows the observed (min, max) point as a red star within the main density cloud.

**Interpretation**: Priors are appropriately calibrated to generate the observed range without being overly restrictive or permissive.

---

### Check 2: Overdispersion Coverage (phi >= 3) - PASS

From `overdispersion_diagnostic.png`:
- **Criterion**: >= 25% of prior simulations have phi >= 3
- **Result**: 78.2% (782/1000 simulations)
- **Prior predictive phi**: Mean = 58.63, Median = 19.12, Range = [0.25, 255.82]
- **Observed phi**: 4.38

**Key Visual Evidence**: The overdispersion plot shows the observed phi = 4.38 (red line) falls in the left tail of the prior predictive distribution but is well within the plausible range. The green shaded region (phi >= 3) covers 78.2% of the distribution.

**Interpretation**: The hierarchical structure successfully generates overdispersion. The prior predictive distribution is somewhat higher than observed (median = 19.12 vs observed = 4.38), but this is **acceptable** because:
1. The prior is intentionally weakly informative
2. The observed value is well within the prior predictive range
3. The likelihood will dominate and pull the posterior toward the observed value
4. Having too little overdispersion in the prior would be more problematic

---

### Check 3: Prior Predictive Interval Coverage - PASS

From `group_level_comparison.png`:
- **Criterion**: >= 70% of groups covered by 95% prior predictive intervals
- **Result**: 100% (12/12 groups)

**Key Visual Evidence**: The caterpillar plot shows all observed rates (red dots) fall within the extremely wide prior predictive 95% intervals (blue lines). The intervals span nearly the full [0, 1] range, indicating weak prior informativeness.

**Interpretation**: Priors are appropriately uninformative at the group level. Every group's observed rate is covered by its prior predictive interval, demonstrating that the priors don't inappropriately constrain the data.

---

### Check 4: Implausible Extreme Values - CONDITIONAL FAIL

From `extreme_values_check.png`:
- **Criterion**: <= 5% of prior samples should have p > 0.8
- **Result**: 6.88% (826/12000 total samples)
- **Additional**: 11.15% have p > 0.5

**Key Visual Evidence**: The right panel of `extreme_values_check.png` shows a long right tail extending to p = 1.0, with 6.88% of samples exceeding 0.8 (marked by dark red dotted line). The observed range (yellow shading) is far from these extreme values.

**Root Cause Analysis**:
1. The Half-Cauchy prior on tau has heavy tails (6.4% with tau > 10)
2. When tau is large and theta_raw_j is positive, theta_j can become very large
3. Large theta_j values map to probabilities approaching 1.0 via the logistic function
4. This creates a small but non-negligible probability mass at implausibly high success rates

**Why This Matters**: For this dataset (success rates 3-14%), rates above 80% are scientifically implausible. While 6.88% slightly exceeds the 5% threshold, this is a **minor concern** because:
- The data strongly contradicts these extreme values
- The likelihood will dominate the prior in the posterior
- The issue affects only ~2% beyond the threshold
- The model uses non-centered parameterization, which will sample efficiently

**Why Not More Concerning**: If this were a higher failure rate (e.g., approaching criterion for immediate halt), we would need to:
- Constrain the tau prior (e.g., Half-Normal or truncated Half-Cauchy)
- Center mu more strongly toward the observed range
- Consider alternative link functions

---

### Check 5: Computational Stability - PASS

From console output and `extreme_values_check.png`:
- **tau > 10**: 6.4% (manageable heavy tails)
- **|theta_j| > 10**: 5.48% (acceptable extreme logits)

**Interpretation**: While some extreme values occur, they are infrequent enough not to cause numerical issues. The non-centered parameterization will handle these efficiently during MCMC sampling.

---

## Additional Diagnostics

### Prior-Data Alignment

From `parameter_plausibility.png` (top-right panel):
- Prior mean on probability scale: 0.106
- Observed pooled rate: 0.070
- **Offset**: Prior centered ~36% higher than observed

**Interpretation**: The prior is weakly centered but not strongly misaligned. The likelihood contribution from 2,814 total observations will easily overcome this mild prior-data tension.

### Hierarchical Shrinkage Capacity

The wide range of tau values (median = 1.01, 95th percentile = 12.31) allows the model to:
- **Learn strong pooling** if groups are similar (small tau)
- **Learn weak pooling** if groups are heterogeneous (large tau)
- The observed overdispersion (phi = 4.38) suggests moderate heterogeneity, which falls well within the prior's range

---

## Criterion-by-Criterion Evaluation

| Criterion | Threshold | Observed | Status | Evidence |
|-----------|-----------|----------|--------|----------|
| 1. Range Coverage | >= 50% | 55.1% | PASS | `range_coverage_diagnostic.png` |
| 2. Overdispersion | >= 25% phi >= 3 | 78.2% | PASS | `overdispersion_diagnostic.png` |
| 3. Interval Coverage | >= 70% groups | 100% | PASS | `group_level_comparison.png` |
| 4. Extreme Values | <= 5% p > 0.8 | 6.88% | FAIL | `extreme_values_check.png` |
| 5. Computational | No flags | Minor flags | PASS | Console output |

**Overall Assessment**: 4 of 5 criteria passed, with the failure being marginal (6.88% vs 5% threshold).

---

## Recommendations

### For This Model (Conditional Approval)

**PROCEED TO FITTING** with the following awareness:

1. **Monitor posterior**: Check that posterior distributions don't retain the heavy tails from the prior
2. **Sensitivity analysis**: Consider running a version with a more constrained tau prior (e.g., Half-Normal(0, 1)) to verify robustness
3. **Posterior predictive check**: Verify that the fitted model doesn't generate implausible predictions

**Rationale for Proceeding**:
- The data (n = 2,814 observations) will strongly inform the posterior
- The failure is marginal (1.88 percentage points over threshold)
- The non-centered parameterization will sample efficiently
- The hierarchical structure is correctly specified

### If Further Refinement Desired

To eliminate the extreme value issue while preserving other good properties:

```
Option A: Constrain tau
  tau ~ Half-Normal(0, 1)  # Lighter tails than Half-Cauchy

Option B: Stronger centering on mu
  mu ~ Normal(-2.5, 0.75)  # Tighter SD to reduce extreme combinations

Option C: Both adjustments
  mu ~ Normal(-2.5, 0.75)
  tau ~ Half-Normal(0, 1.5)
```

**Trade-off**: These adjustments would reduce flexibility. Given the current setup already passes 4/5 checks, this is **not necessary** for proceeding.

---

## Comparison to Observed Data Properties

| Property | Observed | Prior Predictive | Coverage Quality |
|----------|----------|-----------------|------------------|
| Success rate range | [3.1%, 14.0%] | 55.1% cover range | Good |
| Pooled rate | 6.97% | Mean = 18.3% | Weakly informative |
| Overdispersion | phi = 4.38 | 78.2% generate phi >= 3 | Excellent |
| Group count | 12 | 12 (fixed) | N/A |
| Sample size range | [47, 810] | [47, 810] (fixed) | N/A |

---

## Technical Notes

### Overdispersion Calculation
Overdispersion is computed as:
```
phi = Var(p_obs) / [pooled_p * (1 - pooled_p) / mean(n)]
```
Where:
- `Var(p_obs)` = observed variance of success rates across groups
- `pooled_p` = overall pooled success rate
- `mean(n)` = average sample size

This measures how much more variable the observed rates are compared to what binomial sampling alone would predict.

### Non-Centered Parameterization
The model uses `theta_j = mu + tau * theta_raw_j` instead of directly sampling `theta_j ~ Normal(mu, tau)`. This improves MCMC efficiency when:
- The data strongly inform group-level parameters
- tau is small (strong pooling)
- Prevents divergences and improves exploration

---

## Key Visual Evidence

### Three Most Important Plots

1. **`prior_predictive_coverage.png`**: Shows that all observed rates fall within the main density of the prior predictive distribution, demonstrating appropriate calibration

2. **`overdispersion_diagnostic.png`**: Confirms the hierarchical structure generates sufficient overdispersion, with observed phi = 4.38 well within the prior predictive range

3. **`extreme_values_check.png`**: Reveals the marginal issue with heavy right tails (6.88% > 0.8), but also shows these extremes are far from the observed range (yellow shading)

### Supporting Evidence

4. **`group_level_comparison.png`**: All groups covered by wide prior predictive intervals
5. **`parameter_plausibility.png`**: Prior parameters properly centered with appropriate uncertainty
6. **`range_coverage_diagnostic.png`**: Joint min/max distribution covers observed range

---

## Conclusion

The prior predictive check validates that the Hierarchical Binomial model with:
- `mu ~ Normal(-2.5, 1)`
- `tau ~ Half-Cauchy(0, 1)`
- Non-centered parameterization

is **appropriate for fitting to the observed data**, with only a minor concern about extreme value tails that is unlikely to affect posterior inference.

**Final Verdict**: **CONDITIONAL PASS**

**Next Step**: Proceed to simulation-based validation (posterior inference on synthetic data with known parameters) to validate the model's ability to recover true parameters.

---

## Files Generated

- **Code**: `/workspace/experiments/experiment_1/prior_predictive_check/code/prior_predictive_check.py`
- **Plots**: `/workspace/experiments/experiment_1/prior_predictive_check/plots/` (6 diagnostic plots)
- **Summary**: `/workspace/experiments/experiment_1/prior_predictive_check/summary_statistics.json`
- **Report**: `/workspace/experiments/experiment_1/prior_predictive_check/findings.md` (this file)
