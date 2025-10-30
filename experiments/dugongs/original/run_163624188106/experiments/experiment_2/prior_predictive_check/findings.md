# Prior Predictive Check: Experiment 2 - Log-Linear Heteroscedastic Model

**Date**: 2025-10-27
**Model**: Log-Linear Heteroscedastic Model
**Analyst**: Bayesian Model Validator

---

## Executive Summary

**DECISION: CONDITIONAL PASS** with important caveats

The prior specification is generally appropriate for this heteroscedastic model, successfully generating diverse but plausible data. However, the variance structure prior shows concerning behavior: while 82.7% of samples correctly produce decreasing variance with x (as observed), the **magnitude** of the variance ratio is poorly calibrated - the median prior ratio is 21x compared to the observed 8.8x, with extreme outliers reaching >4700x.

**Key Strengths:**
- 29.9% of prior samples generate data similar to observed (exceeds 20% threshold)
- No computational failures (0% negative sigmas)
- Mean structure covers observed range appropriately
- Minimal pathological samples (1.6% with negative Y values)

**Key Concerns:**
- Variance ratio distribution is too wide and skewed (median 21x vs observed 8.8x)
- 17.3% of samples incorrectly generate increasing variance with x
- Some extreme outliers in variance ratios (>4700x) indicate potential computational issues during fitting

---

## Visual Diagnostics Summary

This assessment is based on five comprehensive diagnostic plots:

1. **parameter_distributions.png**: Validates that prior samples match specified distributions
2. **prior_predictive_coverage.png**: Assesses whether generated data covers observed range
3. **variance_structure_diagnostic.png**: Examines heteroscedasticity behavior
4. **mean_structure_diagnostic.png**: Validates log-linear relationship in mean
5. **edge_cases_diagnostic.png**: Identifies computational red flags and pathological samples

---

## Detailed Findings

### 1. Parameter Prior Distributions

**Visual Evidence**: `parameter_distributions.png`

All four parameter priors sample correctly from their specified distributions:

| Parameter | Prior Specification | Sample Mean | Sample Std | Status |
|-----------|-------------------|-------------|------------|---------|
| beta_0    | N(1.8, 0.5)      | 1.810       | 0.489      | PASS |
| beta_1    | N(0.3, 0.2)      | 0.314       | 0.199      | PASS |
| gamma_0   | N(-2, 1)         | -1.994      | 0.983      | PASS |
| gamma_1   | N(-0.05, 0.05)   | -0.051      | 0.051      | PASS |

**Key Observation**: The gamma_1 prior N(-0.05, 0.05) is centered to induce decreasing variance, but its width allows 17.3% of samples to have gamma_1 > 0 (increasing variance). This is a concern but not fatal - the prior correctly expresses uncertainty about the direction of heteroscedasticity while favoring the scientifically expected direction.

**Independence Check**: Correlation between beta_0 and beta_1 is -0.006, confirming priors are independent (as specified). This is appropriate since we have no reason to believe intercept and slope should be correlated a priori.

---

### 2. Prior Predictive Coverage

**Visual Evidence**: `prior_predictive_coverage.png`

The prior predictive distribution provides excellent coverage of the observed data:

**Range Coverage:**
- Generated Y overall range: [-14.22, 17.30]
- Mean generated range per sample: [1.57, 3.11]
- Observed Y range: [1.77, 2.72]
- Target range: [0.5, 5.0]

**Coverage Statistics:**
- 59.1% of samples cover the observed minimum
- 64.5% of samples cover the observed maximum
- **29.9% of samples cover both** (exceeds 20% threshold)

**Assessment**: The prior predictive distribution is well-calibrated for coverage. The observed data falls comfortably within the middle of the prior predictive distribution, indicating the priors are neither too tight (overconfident) nor too wide (uninformative). The 5th-95th percentile band [0.63, 4.68] appropriately captures the target range while remaining scientifically plausible.

The trajectory plot shows smooth, diverse curves that respect the log-linear structure, with no sharp discontinuities or implausible jumps.

---

### 3. Variance Structure (Heteroscedasticity)

**Visual Evidence**: `variance_structure_diagnostic.png`

This is where the main concern emerges:

**Variance Ratio Analysis (low-x variance / high-x variance):**
- Observed ratio: **8.8x** (variance decreases by this factor from x=1 to x=31.5)
- Prior median ratio: **21.1x**
- Prior mean ratio: **1321.8x** (heavily right-skewed)
- 5th-95th percentile: [0.13x, 4762.1x]

**Direction of Heteroscedasticity:**
- 82.7% of samples show decreasing variance (gamma_1 < 0, variance ratio > 1)
- 17.3% of samples show increasing variance (gamma_1 > 0)

**Problem**: The variance ratio distribution is extremely right-skewed with heavy tails. While the median (21x) is in the right ballpark - suggesting the typical prior draw is reasonable - the extreme outliers indicate potential numerical stability issues. When gamma_0 takes large negative values and gamma_1 takes large negative values, the variance at low x can become enormous while variance at high x becomes tiny, creating ratios exceeding 4700x.

**Why This Happens**:
- log(sigma_i) = gamma_0 + gamma_1 * x_i
- At x=1: log(sigma) = gamma_0 + gamma_1
- At x=31.5: log(sigma) = gamma_0 + 31.5*gamma_1
- When gamma_0 = -5 and gamma_1 = -0.15:
  - sigma(x=1) = exp(-5.15) = 0.006
  - sigma(x=31.5) = exp(-9.73) = 0.00006
  - Ratio = 100x

But when gamma_0 = +2 and gamma_1 = -0.15:
  - sigma(x=1) = exp(1.85) = 6.4
  - sigma(x=31.5) = exp(-2.73) = 0.065
  - Ratio = 98x (reasonable)

And when gamma_0 = -5 and gamma_1 = +0.1:
  - sigma(x=1) = exp(-4.9) = 0.007
  - sigma(x=31.5) = exp(-1.85) = 0.16
  - Ratio = 0.04 (wrong direction!)

**Mitigation**: Despite these concerns, the problematic extreme ratios only occur in the tails. The bulk of the distribution (50-75th percentiles) shows ratios between 2-100x, which is reasonable. During MCMC fitting, the likelihood will strongly penalize extreme ratios that don't match the data.

---

### 4. Mean Structure

**Visual Evidence**: `mean_structure_diagnostic.png`

The log-linear mean structure performs well:

**Mean Change Analysis (from x=1 to x=31.5):**
- Prior mean change: 1.08
- Prior 5th-95th percentile: [-0.05, 2.20]
- Observed Y change (endpoints): 0.77

**Assessment**: The prior appropriately captures both increasing and (slightly) decreasing relationships with log(x), centered around positive growth. The observed change falls well within the prior predictive distribution.

The visualization of mu vs log(x) confirms the linearity assumption is well-expressed in the prior. The relationship between beta_0 and beta_1 shows no concerning structure - they are appropriately independent.

**Scientific Plausibility**: With beta_0 ~ N(1.8, 0.5) and beta_1 ~ N(0.3, 0.2), we get:
- At x=1 (log(x)=0): mu ~ N(1.8, 0.5) → [0.8, 2.8] at 95%
- At x=31.5 (log(x)=3.45): mu ~ N(2.84, 0.74) → [1.4, 4.3] at 95%

Both ranges are scientifically reasonable for this data.

---

### 5. Computational Red Flags

**Visual Evidence**: `edge_cases_diagnostic.png`

**Negative Y Values:**
- 16 out of 1000 samples (1.6%) generated at least one negative Y value
- This is minimal and acceptable - the Normal distribution inherently allows negative values, and 1.6% is a negligible tail probability

**Extreme Standard Deviations:**
- 0 samples (0.0%) had sigma > 10
- Maximum sigma across all samples: ~6.5
- This is excellent - no numerical overflow concerns

**Coefficient of Variation:**
- The relative variability (sigma/mu) decreases with x in the prior, matching the observed pattern
- Prior CV at low x: ~0.3-1.0
- Prior CV at high x: ~0.05-0.3
- This shows the model correctly captures that both absolute and relative variability decrease

**Problematic Parameter Regions:**
- Only 1.6% of samples are "problematic" (negative Y or extreme sigma)
- These occur primarily when gamma_0 is extreme (< -4 or > 1) combined with extreme gamma_1

---

## Quantitative Success Criteria Assessment

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|---------|
| Generated Y coverage | [0.5, 5] | [-14.22, 17.30] | PASS |
| Y range wider than observed | Yes | Mean [1.57, 3.11] vs obs [1.77, 2.72] | PASS |
| Variance decreases with x | gamma_1 < 0 | 82.7% of samples | PASS |
| No negative Y | < 5% | 1.6% | PASS |
| No extreme variance | < 5% | 0.0% | PASS |
| Similar to observed | > 20% | 29.4% | PASS |
| Observed range coverage | > 20% | 29.9% | PASS |

**All quantitative criteria are met.**

---

## Critical Assessment: Why Conditional Pass?

### What Makes This PASS:

1. **Core functionality intact**: The model generates scientifically plausible data that covers the observed range
2. **Heteroscedasticity direction correct**: 82.7% of prior samples correctly capture decreasing variance
3. **No computational failures**: Zero samples with negative/infinite sigma
4. **Meets all quantitative thresholds**: Exceeds the 20% similarity criterion (29.4%)
5. **Negative Y values minimal**: Only 1.6% of samples, which is acceptable given the Normal distribution

### What Makes This CONDITIONAL:

1. **Variance ratio distribution poorly calibrated**: While the median (21x) is reasonable, the extreme right tail (up to 4762x) indicates the gamma_0 and gamma_1 priors interact to create occasional implausible heteroscedasticity
2. **17% wrong direction**: Nearly one in five samples generates increasing variance with x, opposite to what we observe
3. **Heavy-tailed behavior**: The mean variance ratio (1322x) being 63x larger than the median indicates problematic outliers

### Implications for Model Fitting:

**Proceed with fitting** because:
- The likelihood will sharply constrain gamma_0 and gamma_1 to match observed data
- The extreme prior tails will receive near-zero posterior weight
- MCMC samplers can handle this prior (no numerical issues like negative sigma)
- 82.7% of prior samples are in the right direction - this is strong prior information

**Monitor during fitting**:
- Check for divergences in MCMC (may indicate extreme prior regions causing problems)
- Verify posterior for gamma parameters is much tighter than prior
- Confirm posterior variance ratios cluster near observed 8.8x
- Watch for poor mixing if sampler gets stuck in extreme prior tails

---

## Recommendations

### If Proceeding with Current Priors (Recommended):

**Justification**: The priors are adequate for fitting. The extreme tails will be eliminated by the likelihood, and the bulk of the prior distribution is reasonable.

**During fitting, verify**:
1. No divergent transitions in MCMC
2. Posterior is much tighter than prior (prior → posterior shrinkage)
3. Effective sample sizes > 400 for all parameters
4. Posterior predictive checks show good fit

### If Revising Priors (Conservative Alternative):

If you want to eliminate the extreme tails before fitting, consider:

**Tighter gamma_1 prior:**
```
gamma_1 ~ Normal(-0.08, 0.03)  # Stronger shrinkage toward decreasing variance
```
- Reduces wrong-direction samples from 17% to ~4%
- Reduces extreme ratios while maintaining flexibility

**Constrained gamma_0 prior:**
```
gamma_0 ~ Normal(-2, 0.7)  # Tighter variance intercept
```
- Prevents extremely small/large baseline variances
- Reduces interaction effects with gamma_1

**Re-run prior predictive check** after revision to verify improved calibration.

---

## Model Structure Assessment

The **log-linear heteroscedastic structure is appropriate** for this data:

**Mean structure** (log-linear):
- Captures diminishing returns as x increases
- Flexible enough to fit observed concave growth
- Scientifically interpretable (log relationships common in growth models)

**Variance structure** (exponential heteroscedasticity):
- Ensures sigma > 0 (no constraints needed)
- Allows variance to change smoothly with x
- Can capture decreasing variance observed in data

**No structural conflicts** between prior and likelihood - the model can express the patterns in the data.

---

## Conclusion

**FINAL DECISION: CONDITIONAL PASS**

The Log-Linear Heteroscedastic Model prior specification is **adequate for model fitting**. The priors generate diverse, scientifically plausible data that covers the observed range. While the variance ratio distribution shows concerning heavy tails, this is not fatal - the likelihood will constrain the posterior away from extreme values.

**Proceed with model fitting** using the current priors, but:
1. Monitor MCMC diagnostics carefully for signs of numerical issues
2. Verify posterior shrinkage from prior (especially for gamma parameters)
3. Confirm posterior predictive checks show good fit
4. Be prepared to revise gamma priors if fitting issues arise

The 29.4% similarity rate (exceeding the 20% threshold) indicates the priors encode appropriate domain knowledge while maintaining sufficient uncertainty for the data to inform the posterior.

**Next steps**: Fit the model using MCMC, document convergence diagnostics, and perform posterior predictive checks to validate the fitted model.

---

## Appendix: Technical Details

### Files Generated

**Code**:
- `/workspace/experiments/experiment_2/prior_predictive_check/code/prior_predictive_check.py` - Main analysis script
- `/workspace/experiments/experiment_2/prior_predictive_check/code/create_visualizations.py` - Visualization generation
- `/workspace/experiments/experiment_2/prior_predictive_check/code/prior_predictive_samples.npz` - Saved samples (1000 draws)

**Plots**:
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/parameter_distributions.png` - Prior validation
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/prior_predictive_coverage.png` - Coverage assessment
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/variance_structure_diagnostic.png` - Heteroscedasticity analysis
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/mean_structure_diagnostic.png` - Log-linear relationship
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/edge_cases_diagnostic.png` - Computational red flags

### Sampling Configuration

- Prior samples: N = 1000
- Observations per sample: 27 (matching observed data)
- x values: [1.0, 31.5] (matching observed range)
- Random seed: 42 (reproducible)

### Data Context

- Observed Y range: [1.77, 2.72]
- Observed x range: [1.0, 31.5]
- Observed variance decrease: 8.8x from low to high x
- Sample size: N = 27

---

**Assessment Complete**: 2025-10-27
