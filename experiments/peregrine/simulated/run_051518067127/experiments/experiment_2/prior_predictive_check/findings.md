# Prior Predictive Check: Experiment 2 - AR(1) Log-Normal with Regime-Switching

**Date**: 2025-10-30
**Model**: AR(1) Log-Normal with Regime-Switching Variance
**Prior Draws**: 1,000
**Decision**: **FAIL - CRITICAL ISSUES IDENTIFIED**

---

## Visual Diagnostics Summary

All diagnostic plots are located in `/workspace/experiments/experiment_2/prior_predictive_check/plots/`:

1. **parameter_plausibility.png** - Prior distributions for all 7 parameters
2. **prior_predictive_coverage.png** - Prior predictive intervals vs observed data (count scale)
3. **prior_trajectories.png** - Sample trajectories showing AR temporal structure
4. **prior_autocorrelation_diagnostic.png** - ACF distribution and phi vs ACF relationship
5. **regime_variance_diagnostic.png** - Multi-panel regime structure diagnostics
6. **log_scale_diagnostic.png** - Count vs log-scale prior predictive comparison

---

## Executive Summary

The prior predictive check reveals **THREE CRITICAL FAILURES** that must be addressed before fitting:

1. **CRITICAL: Autocorrelation Prior-Data Mismatch** - The uniform phi prior U(-0.95, 0.95) generates ACF centered near 0, but observed data has ACF = 0.961. The prior does not adequately favor high autocorrelation.

2. **CRITICAL: Extreme Prediction Variability** - 5.8% of predictions exceed 1,000 (max: 348 million!), indicating heavy right tail from log-normal + wide priors. This creates numerical instability risk.

3. **LOW COVERAGE: Only 2.8% of prior draws produce fully plausible data** ([10, 500] range), suggesting priors may be too diffuse or model structure creates conflicts.

**Recommendation**: FAIL - Adjust priors before proceeding to simulation validation.

---

## Key Visual Evidence

### 1. Autocorrelation Diagnostic (Most Critical)
**Plot**: `prior_autocorrelation_diagnostic.png`

**Left panel** shows prior ACF lag-1 distribution is nearly uniform from -1 to +1, with median at -0.059. **Observed ACF = 0.961 is outside the main prior mass** and in the extreme right tail.

**Right panel** reveals the relationship between phi and implied ACF. While high phi values (>0.8) do produce high ACF, the uniform prior on phi doesn't favor this region enough.

**Implication**: The model CAN represent high autocorrelation, but the prior doesn't encode our domain knowledge that this data is highly autocorrelated.

### 2. Prior Predictive Coverage
**Plot**: `prior_predictive_coverage.png`

Shows massive uncertainty in count scale predictions, with 95% intervals reaching 12,000+ at later time points. The observed data (red dots) sits at the bottom edge of the prior predictive distribution, near the median but dwarfed by the wide intervals.

**Implication**: Log-normal likelihood combined with wide sigma priors creates heavy right tails that dominate predictions.

### 3. Log-Scale vs Count-Scale Comparison
**Plot**: `log_scale_diagnostic.png`

**Critical insight**: On log-scale (right panel), the prior predictions look reasonable and well-centered around observed data. But transformation to count scale (left panel) creates extreme right skew.

**Implication**: The issue is not the trend structure (which is good on log-scale), but the combination of log-normal likelihood + wide variance priors.

---

## Detailed Diagnostic Results

### 1. Domain Violations

```
Negative counts:        0 (0.00%)    ✓ PASS
Extreme high (>1000):   2,322 (5.80%)   ✗ FAIL
Extreme low (<1):       459 (1.15%)     ~ Borderline
NaN/Inf values:         0 (0.00%)    ✓ PASS
```

**Assessment**: No computational red flags (no NaN/Inf), but 5.8% of predictions are implausibly high (>1000 when max observed is 269). This suggests heavy right tail will create sampling challenges.

### 2. Plausibility Ranges

```
Observed range:         [21, 269]
Plausible range:        [10, 500]

Draws fully in observed range:     0.0%
Draws fully in plausible range:    2.8%
```

**Assessment**: FAIL - Only 2.8% of prior draws produce entirely plausible data. This is far below the 50% threshold for a well-specified prior. Most draws include at least some extreme values.

### 3. Autocorrelation Structure (MOST CRITICAL)

```
Observed log(C) ACF lag-1:          0.961
Prior ACF lag-1 (median):           -0.059
Prior ACF lag-1 (90% CI):           [-0.860, 0.772]
Prior covers observed:              FALSE
```

**Assessment**: CRITICAL FAIL - The observed ACF of 0.961 is outside the 90% prior predictive interval. The uniform prior on phi U(-0.95, 0.95) doesn't encode our strong domain knowledge that this process is highly autocorrelated.

**Key insight from visualization**: The relationship phi → ACF is not one-to-one. Regime-switching variances and the AR initialization also affect the implied ACF. But fundamentally, a uniform phi prior is inappropriate when we know the data is strongly autocorrelated.

### 4. Prior Parameter Distributions

**Plot**: `parameter_plausibility.png`

```
alpha:    median = 4.313, 90% CI = [3.537, 5.138]   ✓ Reasonable
beta_1:   median = 0.873, 90% CI = [0.545, 1.199]   ✓ Reasonable
beta_2:   median = 0.000, 90% CI = [-0.470, 0.485]  ✓ Reasonable
phi:      median = -0.031, 90% CI = [-0.854, 0.847] ✗ Too diffuse for ACF=0.96
sigma_1:  median = 0.655, 90% CI = [0.060, 1.945]   ✗ Heavy right tail
sigma_2:  median = 0.645, 90% CI = [0.048, 1.968]   ✗ Heavy right tail
sigma_3:  median = 0.717, 90% CI = [0.068, 2.065]   ✗ Heavy right tail
```

**Assessment**:
- Trend parameters (alpha, beta_1, beta_2) are well-specified
- **phi prior is too diffuse** given strong domain knowledge of high autocorrelation
- **Sigma priors are too wide** (HalfNormal(0,1) allows values >2 which is implausibly large on log-scale)

### 5. Regime Variance Structure

**Plot**: `regime_variance_diagnostic.png`

**Top-left panel**: All three sigma priors are identical (HalfNormal(0,1)) and heavily overlapping. Prior doesn't encode any expectation about which regime is more variable.

**Top-right panel**: Prior is nearly uniform over which regime has largest sigma (Late=36%, Early=32%, Middle=31%). This is appropriate if we have no prior information, but creates identifiability challenges.

**Bottom-left panel**: Prior predictive variance by regime shows one extreme outlier, illustrating the heavy tail problem.

**Bottom-right panel**: Example trajectory shows the AR structure does create temporal smoothness (good!), but with wild excursions (bad!).

**Assessment**: Regime structure is specified reasonably, but sigma priors need tightening to prevent extreme predictions.

---

## Structural Assessment

### What's Working Well

1. **AR(1) structure generates temporally smooth trajectories** (`prior_trajectories.png` shows connected paths, not independent noise)
2. **Log-scale trend is well-specified** (right panel of `log_scale_diagnostic.png` shows good coverage)
3. **No computational red flags** (no NaN/Inf, no negative counts)
4. **Regime boundaries don't create discontinuities** (trajectories are smooth across regime transitions)

### What's Failing

1. **Autocorrelation prior doesn't match domain knowledge**
   - Uniform phi prior generates ACF centered at 0
   - Observed ACF = 0.961 is in extreme prior tail
   - Need to shift prior mass toward high positive phi

2. **Log-normal + wide sigma creates extreme predictions**
   - HalfNormal(0,1) allows sigma >2 on log-scale
   - When exponentiated, creates predictions >100,000
   - Need tighter sigma priors

3. **Prior-data disconnect on autocorrelation**
   - The model CAN represent high autocorrelation
   - But the prior doesn't favor it
   - This is a prior specification issue, not a structural issue

---

## Root Cause Analysis

### Why is ACF Prior So Different from Observed?

The uniform prior phi ~ U(-0.95, 0.95) encodes **no preference** for positive vs negative autocorrelation. This is appropriate when we have no domain knowledge, but **we do have strong domain knowledge**:

- Data ACF lag-1 = 0.971 (from EDA)
- This is VERY high autocorrelation
- Uniform prior on phi ignores this information

### Why Are Predictions So Extreme?

Log-normal likelihood means: C ~ exp(Normal(mu, sigma))

When sigma is large (e.g., sigma=2), the right tail is extremely heavy:
- 95th percentile at exp(mu + 2*1.96) = exp(mu + 3.92)
- If mu=5, this is exp(8.92) = 7,440

The HalfNormal(0,1) prior allows sigma >2 in ~2.5% of draws, creating occasional extreme predictions that dominate the prior predictive distribution.

---

## Recommendations for Prior Adjustment

### 1. CRITICAL: Tighten phi Prior to Favor High Autocorrelation

**Current**: phi ~ Uniform(-0.95, 0.95)
**Proposed**: phi ~ Beta(20, 2) rescaled to (0, 0.95)

**Rationale**:
- Beta(20, 2) has median ≈ 0.91, matching observed ACF
- Still allows moderate flexibility (95% CI: ~[0.75, 0.98])
- Respects stationarity constraint |phi| < 1
- Encodes strong domain knowledge of high positive autocorrelation

**Alternative**: phi ~ Beta(10, 1) rescaled to (0, 0.95) if more diffuse prior desired (median ≈ 0.86)

### 2. CRITICAL: Tighten Sigma Priors

**Current**: sigma_regime[1:3] ~ HalfNormal(0, 1)
**Proposed**: sigma_regime[1:3] ~ HalfNormal(0, 0.5)

**Rationale**:
- On log-scale, residual SD > 1 is implausibly large for this application
- HalfNormal(0, 0.5) concentrates 95% of mass in [0, 0.98]
- This still allows substantial variation but prevents extreme tails
- Log-scale residuals in data appear to be σ ≈ 0.2-0.5 (from EDA)

**Alternative**: sigma_regime[1:3] ~ Exponential(2) also works (median=0.35, 95% at 1.5)

### 3. OPTIONAL: Informative Prior on beta_1

**Current**: beta_1 ~ Normal(0.86, 0.2)
**Status**: Already reasonable, but could tighten slightly

**Proposed**: beta_1 ~ Normal(0.86, 0.15)

**Rationale**: EDA showed very strong linear trend (β=0.862 with tight SE). Current prior allows β in [0.46, 1.26] which is quite wide. Tightening won't hurt and improves prior predictive coverage.

### 4. Keep Other Priors Unchanged

- alpha ~ Normal(4.3, 0.5) - Good
- beta_2 ~ Normal(0, 0.3) - Good (weakly informative for quadratic term)
- Regime structure - Good (no reason to impose ordering without data evidence)

---

## Revised Prior Specification

```
# Trend parameters (keep existing)
alpha ~ Normal(4.3, 0.5)
beta_1 ~ Normal(0.86, 0.15)          # Slightly tighter
beta_2 ~ Normal(0, 0.3)

# AR coefficient (CRITICAL CHANGE)
phi ~ Beta(20, 2) rescaled to (0, 0.95)
# Implementation: phi = 0.95 * Beta(20, 2)

# Regime variances (CRITICAL CHANGE)
sigma_regime[1:3] ~ HalfNormal(0, 0.5)
```

---

## Expected Impact of Prior Adjustments

After implementing the recommended changes:

1. **Autocorrelation coverage**: Prior ACF will be centered at ~0.85-0.90, covering observed 0.96
2. **Extreme predictions reduced**: Max predictions should drop from 348M to ~1,000-2,000
3. **Plausibility coverage**: Should increase from 2.8% to 20-30%
4. **No structural changes**: Model specification remains the same, only priors adjusted

---

## Next Steps

1. **Implement revised priors** in model specification
2. **Re-run prior predictive check** to verify improvements
3. **Proceed to simulation validation** only after prior predictive check passes

**Do NOT proceed to fitting** with current priors - the autocorrelation mismatch and extreme predictions will cause:
- Poor posterior convergence (extreme proposals rejected)
- Biased inference (prior-data conflict)
- Misleading predictive intervals (dominated by prior tail)

---

## Decision: FAIL

**Reason**: Critical prior-data mismatch on autocorrelation structure + extreme prior predictions

**Required Action**: Implement recommended prior adjustments and re-run prior predictive check

**Confidence**: High - The issues are clear and the solutions are straightforward

---

## Technical Notes

### AR(1) Implementation Quality

The sequential generation code correctly implements:
- Stationary initialization: epsilon[1] ~ N(0, sigma / sqrt(1 - phi^2))
- Sequential dependence: mu[t] includes phi * epsilon[t-1]
- Proper error calculation: epsilon[t] = log(C[t]) - trend[t]

No implementation issues detected.

### Computational Considerations

With current priors, expect:
- Slow warmup due to extreme proposals
- High divergences if HMC hits extreme log-normal tail
- Possibly R-hat > 1.05 due to prior-data conflict

After prior adjustment, these issues should resolve.

---

## Appendix: Prior Predictive Statistics

```
Overall median prediction:      76.0
Overall mean prediction:        14,593.6  (heavily right-skewed!)
Min prediction:                 0.0
Max prediction:                 348,453,273.1  (absurd!)

90% quantile (across all):      ~800
95% quantile (across all):      ~2,000
99% quantile (across all):      ~50,000
```

The massive difference between median (76) and mean (14,593) confirms the heavy right tail problem.

---

## Files Generated

All files in `/workspace/experiments/experiment_2/prior_predictive_check/`:

**Code**:
- `code/prior_predictive_check.py` - Full implementation with AR(1) sequential generation

**Visualizations**:
1. `plots/parameter_plausibility.png` - Prior distributions for 7 parameters
2. `plots/prior_predictive_coverage.png` - Coverage plot showing extreme intervals
3. `plots/prior_trajectories.png` - Sample paths demonstrating AR smoothness
4. `plots/prior_autocorrelation_diagnostic.png` - ACF distribution and phi relationship
5. `plots/regime_variance_diagnostic.png` - 4-panel regime structure diagnostics
6. `plots/log_scale_diagnostic.png` - Count vs log-scale comparison

**Documentation**:
- `findings.md` - This document

---

**Prepared by**: Claude (Bayesian Model Validator)
**Review Status**: Ready for Principal Investigator review
**Action Required**: Revise priors per recommendations before proceeding
