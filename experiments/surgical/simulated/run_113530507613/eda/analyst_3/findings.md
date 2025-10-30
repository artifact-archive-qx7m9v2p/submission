# EDA Findings Report: Model-Relevant Features & Data Quality

**Analyst:** Analyst 3
**Focus:** Data quality, binomial assumptions, pooling justification, prior specification
**Date:** 2025-10-30

---

## Executive Summary

**Data Quality: EXCELLENT** - All binomial constraints satisfied, no missing values, no data integrity issues.

**Key Finding: HIERARCHICAL PARTIAL POOLING STRONGLY RECOMMENDED**
- Chi-square test decisively rejects homogeneity (p < 0.001)
- 42% of variance is between groups (ICC = 0.42)
- Groups vary substantially (4.5x range in success rates)
- Empirical Bayes shrinkage factors average 0.52

**Modeling Implications:**
- Use logit-normal hierarchical model with weakly informative priors
- Prior sensitivity analysis essential (smallest group has n=47)
- Partial pooling will improve estimates for small-sample groups

---

## 1. Data Quality Assessment

### 1.1 Data Integrity: PASSED ALL CHECKS

**Binomial Constraints:**
- ✓ All r_successes ≤ n_trials (0 violations)
- ✓ No negative values (0 found)
- ✓ No zero trials (0 found)
- ✓ Success rates in [0, 1] (all valid)
- ✓ Success rate calculations verified (max error < 1e-10)

**Completeness:**
- ✓ No missing values (0 out of 48 total)
- ✓ No duplicate groups
- ✓ All group IDs sequential 1-12

**Visual Evidence:** See `distributions_overview.png` for data distribution characteristics.

### 1.2 Data Characteristics

**Sample Sizes (n_trials):**
- Range: [47, 810]
- Mean: 234.5, Median: 201.5
- SD: 198.4 (CV = 0.846 - high variability)
- Distribution: Right-skewed (skewness = 2.52)
  - 2 groups < 100 trials
  - 1 group < 50 trials (Group 1: n=47)
  - 1 dominant group (Group 4: n=810, 29% of all data)

**Success Counts (r_successes):**
- Range: [3, 34]
- Mean: 16.3, Median: 14.5
- 1 group with r ≤ 5 (Group 10: r=3)
- No zero-success groups (good for stability)

**Success Rates:**
- Range: [0.031, 0.140] - **4.51x variation**
- Mean: 0.079, Median: 0.071, Pooled: 0.070
- SD: 0.0348 (CV = 0.441 - substantial variation)
- 90% interval: [0.043, 0.128]

**Visual Evidence:** See `sample_size_adequacy.png` and `success_rates_with_ci.png`.

### 1.3 Data Quality Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Binomial validity | ✓ Passed | All constraints satisfied |
| Completeness | ✓ Complete | No missing data |
| Precision | ⚠ Variable | SE ranges from 0.007 to 0.049 |
| Balance | ⚠ Unbalanced | Sample sizes vary 17-fold |
| Sparsity | ⚠ Minor | 1 group with only 3 successes |

**Conclusion:** Data is clean and ready for modeling. Main concern is sample size heterogeneity affecting precision.

---

## 2. Pooling vs No-Pooling Evidence

### 2.1 Statistical Test for Heterogeneity

**Chi-Square Test for Homogeneity of Proportions:**
- Test statistic: χ² = 39.52
- Degrees of freedom: 11
- **P-value: < 0.001** (0.000043)
- **Conclusion: REJECT homogeneity** - groups are significantly different

**Visual Evidence:** See `heterogeneity_test.png` showing observed vs expected successes and standardized residuals.

### 2.2 Between-Group Variability

**Descriptive Statistics:**
- Success rate range: [0.031, 0.140] - **4.5x difference**
- Standard deviation: 0.0348
- Coefficient of variation: 0.441 (44%)
- IQR: 0.030
- Ratio max/min: 4.51

**Groups with Extreme Rates:**
- **Highest:** Group 8 (0.140), Group 2 (0.128), Group 1 (0.128)
- **Lowest:** Group 10 (0.031), Group 4 (0.042), Group 5 (0.057)
- **Near pooled:** Groups 3, 6, 7, 11, 12 (within 0.01 of 0.070)

**Visual Evidence:** See `pooling_comparison.png` showing deviation from pooled rate.

### 2.3 Variance Decomposition

**Estimated Variance Components:**
- Between-group variance (τ²): **0.000383**
- Within-group variance (avg): 0.000528
- Total variance: 0.000910

**Intraclass Correlation (ICC):**
- **ICC = 0.420** (42%)
- Interpretation: **42% of total variance is between groups**
- Threshold for hierarchical modeling: >0.1 (weak), >0.3 (strong)
- **Conclusion: STRONG evidence for group structure**

**Overdispersion:**
- Expected variance (binomial): 0.0648
- Observed variance (between groups): 0.0012
- Variance ratio: 0.019
- No overdispersion detected (as expected for group summaries)

**Visual Evidence:** See variance decomposition panel in `pooling_comparison.png`.

### 2.4 Shrinkage Analysis

**Empirical Bayes Shrinkage Estimates:**
- Mean shrinkage factor (λ): **0.520**
- Interpretation: Optimal estimates are midway between individual MLEs and pooled rate

**Shrinkage by Sample Size:**
| Group | n | Raw Rate | Shrinkage λ | Shrunk Rate | Interpretation |
|-------|---|----------|-------------|-------------|----------------|
| 1 | 47 | 0.128 | 0.217 | 0.082 | Strong shrinkage (small n) |
| 4 | 810 | 0.042 | 0.827 | 0.047 | Minimal shrinkage (large n) |
| 10 | 97 | 0.031 | 0.364 | 0.056 | Moderate shrinkage (sparse) |

**Pattern:** Smaller samples shrink more toward pooled mean - appropriate behavior.

**Visual Evidence:** See `sample_size_precision_shrinkage.png` showing shrinkage effect.

### 2.5 Pooling Decision

**Complete Pooling (Single rate for all groups):**
- Estimate: 0.0697
- 95% CI: [0.060, 0.079]
- **NOT RECOMMENDED:** Rejected by chi-square test (p < 0.001)
- Ignores substantial between-group variation

**No Pooling (Separate rate per group):**
- Group-specific estimates with wide variation
- **NOT RECOMMENDED:**
  - Overfits, especially for small samples
  - Group 1 (n=47): SE = 0.049 (very imprecise)
  - Group 10 (r=3): Unstable estimate
  - Doesn't share information across groups

**Partial Pooling (Hierarchical model):**
- Balances group-specific data with global pattern
- **STRONGLY RECOMMENDED:**
  - ICC = 0.42 > 0.30 (strong group structure)
  - Chi-square test rejects homogeneity
  - Shrinkage analysis shows optimal λ ≈ 0.5
  - Improves estimates for small-sample groups
  - Provides uncertainty on both group rates and population parameters

**Visual Evidence:** See `pooling_comparison.png` for comprehensive comparison.

---

## 3. Plausible Parameter Ranges for Priors

### 3.1 Empirical Summaries

**Success Rate Quantiles:**
| Quantile | Value | Interpretation |
|----------|-------|----------------|
| 1% | 0.032 | Extreme low |
| 5% | 0.037 | Lower tail |
| 25% | 0.060 | Q1 |
| 50% | 0.071 | Median |
| 75% | 0.090 | Q3 |
| 95% | 0.133 | Upper tail |
| 99% | 0.138 | Extreme high |

**Central Estimates:**
- Pooled rate: 0.0697
- Mean rate: 0.0789
- Median rate: 0.0707

**Dispersion:**
- SD: 0.0348
- 90% interval: [0.043, 0.128]
- 95% interval: [0.037, 0.133]

### 3.2 Beta Distribution Priors

**Non-Informative Options:**
1. **Beta(1, 1)** - Uniform on [0, 1]
   - Completely flat, no prior information
   - Effective sample size: 2
   - Use for sensitivity check

2. **Beta(0.5, 0.5)** - Jeffreys prior
   - Favors extremes (0 and 1)
   - Not recommended for rates near 0.07

3. **Beta(2, 2)** - Weakly peaked at 0.5
   - Minimal information
   - Effective sample size: 4

**Weakly Informative (RECOMMENDED for sensitivity):**
- **Beta(2, 26.7)** - Centered on pooled rate
  - Mean: 0.070
  - Effective sample size: ~29
  - Weak enough to be dominated by data
  - Strong enough to stabilize small samples

**Moderately Informative:**
- **Beta(4.65, 54.36)** - Matches observed mean and SD
  - Mean: 0.079
  - SD: 0.035
  - Effective sample size: ~59
  - Based on between-group variation

**Strongly Informative:**
- **Beta(16.3, 217.2)** - Based on pooled with sampling variance
  - Mean: 0.070
  - SD: 0.017
  - Effective sample size: ~233
  - Only if we trust pooling assumption

**Visual Evidence:** See `prior_options.png` for comparison of prior shapes.

### 3.3 Hierarchical Model Hyperpriors (RECOMMENDED)

**Population Mean (μ):**

*Probability scale:*
- Normal(0.070, 0.035)
- Covers 95% interval: [0.001, 0.139]

*Logit scale (more stable):*
- **Normal(-2.6, 1.0)** ← RECOMMENDED
- logit(0.070) = -2.59
- Covers wide range while centering on data

**Between-Group SD (τ):**
- Empirical estimate: τ = 0.0196
- Plausible range: [0.01, 0.04]

*Prior options:*
1. **Half-Normal(0, 0.039)** ← RECOMMENDED
   - Weakly informative
   - 95% mass below 0.064

2. **Exponential(1/0.02 = 50)**
   - Mean = 0.02
   - Alternative to half-normal

3. **Half-Cauchy(0, 0.025)**
   - Heavy tails
   - More robust to misspecification

**Group-Level Parameters (θⱼ):**
- Logit scale: θⱼ ~ Normal(μ, τ)
- Probability scale: pⱼ = logit⁻¹(θⱼ)
- Prior predictive should cover [0.03, 0.14]

**Visual Evidence:** See hierarchical model panel in `prior_options.png`.

### 3.4 Prior Strength Analysis

**Effective Sample Sizes:**
| Prior Type | Effective n | Influence on Small Group (n=47) | Influence on Large Group (n=810) |
|------------|-------------|--------------------------------|----------------------------------|
| Beta(1, 1) | 2 | Minimal | Negligible |
| Beta(2, 27) | 29 | Moderate (~40% weight) | Negligible (~3% weight) |
| Beta(5, 65) | 70 | Strong (~60% weight) | Minimal (~8% weight) |
| Beta(16, 217) | 233 | Very strong (~83% weight) | Moderate (~22% weight) |

**Implications:**
- Even "weak" priors matter for Group 1 (n=47)
- Large groups (n>500) relatively insensitive to priors
- Prior choice critical for sparse groups (Group 10: r=3)

**Visual Evidence:** See `prior_sensitivity.png` comparing posterior distributions under different priors.

---

## 4. Prior Specification Recommendations

### 4.1 Primary Recommendation: Hierarchical Model

**Model Structure:**
```
# Hyperpriors
mu ~ Normal(-2.6, 1.0)         # population mean (logit scale)
tau ~ Half-Normal(0, 0.05)     # between-group SD

# Group-level parameters
for j in 1:J {
  theta[j] ~ Normal(mu, tau)   # logit scale
  p[j] = inv_logit(theta[j])   # probability scale
}

# Likelihood
for j in 1:J {
  y[j] ~ Binomial(n[j], p[j])
}
```

**Why This Model:**
1. Automatically determines optimal pooling strength
2. Borrows strength for small-sample groups
3. Respects heterogeneity (ICC = 0.42)
4. Provides uncertainty on both group and population parameters
5. Easily extends to include covariates later

### 4.2 Alternative: Beta-Binomial (Simpler)

**Model Structure:**
```
# Hyperpriors (specify mean and overdispersion)
mu_p ~ Beta(2, 27)           # population mean success rate
kappa ~ Gamma(2, 0.1)        # concentration parameter

# Implied alpha, beta
alpha = mu_p * kappa
beta = (1 - mu_p) * kappa

# Group-level parameters
for j in 1:J {
  p[j] ~ Beta(alpha, beta)
}

# Likelihood
for j in 1:J {
  y[j] ~ Binomial(n[j], p[j])
}
```

**Advantages:** Conjugate, simpler
**Disadvantages:** Less flexible, harder to add covariates

### 4.3 Required Sensitivity Analyses

**Must run at least 3 prior specifications:**

1. **Weak/Vague:**
   - Beta(1, 1) or tau ~ Half-Cauchy(0, 1)
   - Purpose: Check that data dominates

2. **Weakly Informative (baseline):**
   - Beta(2, 27) or mu ~ N(-2.6, 1), tau ~ Half-N(0, 0.05)
   - Purpose: Primary analysis

3. **Moderately Informative:**
   - Beta(5, 65) or stronger tau prior
   - Purpose: Check robustness

**Compare across priors:**
- Posterior means for group rates
- Posterior SD for group rates
- Estimated μ and τ
- Posterior predictive for new group
- Conclusions about which groups differ

**Expected result:** Conclusions robust if data informative enough.

### 4.4 Prior Predictive Checks

**Before fitting model, check prior predictive:**
1. Simulate p[j] from hierarchical prior
2. Check if simulated rates cover plausible range [0, 0.3]
3. Adjust hyperpriors if prior predictive too wide or too narrow

**Example check:**
```
For 1000 simulations:
  mu ~ Normal(-2.6, 1)
  tau ~ Half-Normal(0, 0.05)
  theta ~ Normal(mu, tau)
  p = inv_logit(theta)

Check: Do 95% of p fall in [0.01, 0.20]?
```

---

## 5. Data Limitations and Concerns

### 5.1 Identified Limitations

**Sample Size Imbalance:**
- Group 4 (n=810) dominates pooled estimate (29% of data)
- Group 1 (n=47) has low precision (SE=0.049)
- 17-fold range in sample sizes
- **Impact:** Unweighted analyses may be misleading; hierarchical model handles this naturally

**Sparse Data:**
- Group 10: Only 3 successes in 97 trials (3.1% rate)
- Posterior may be sensitive to prior for this group
- **Mitigation:** Partial pooling will stabilize estimate

**Unknown Context:**
- Don't know what groups represent (regions? time periods? treatments?)
- Don't know why sample sizes vary
- **Impact:** Limits interpretation; can estimate parameters but not explain causes

**No Covariates:**
- Cannot explain between-group variation
- Model estimates group effects but not their determinants
- **Impact:** Purely descriptive; cannot build predictive model for new groups

**Limited Number of Groups (J=12):**
- Can estimate τ but with moderate uncertainty
- Hyperprior choice matters more than with J>30
- **Impact:** Prior sensitivity analysis essential

### 5.2 Assumptions to Check Post-Modeling

**Binomial Likelihood Appropriate:**
- Assume trials independent within groups ✓
- Assume success probability constant within group ✓
- If violated: Consider beta-binomial to allow overdispersion

**Exchangeability:**
- Assume groups exchangeable (no natural ordering)
- If groups are time periods: Consider trend model
- If groups are nested: Consider multilevel structure

**Logit-Normal Distribution:**
- Assume group effects normally distributed on logit scale
- If violated: Check residuals, consider mixture model

**No Outliers:**
- Check for groups that don't fit hierarchical structure
- Use posterior predictive checks
- Consider robust hierarchical models if outliers present

### 5.3 Recommendations to Address Limitations

**Before Modeling:**
1. Understand group context from data source
2. Check if any groups should be excluded
3. Consider if sample sizes informative (e.g., stopped early)

**During Modeling:**
1. Use hierarchical model to handle size imbalance
2. Check MCMC convergence (especially for sparse groups)
3. Examine posterior for Group 10 (r=3) closely

**After Modeling:**
1. Posterior predictive checks for all groups
2. Check if any groups poorly fit
3. Sensitivity to prior on τ
4. Compare to frequentist shrinkage estimator

---

## 6. Implications for Modeling

### 6.1 Model Class Recommendations

**Tier 1 (Strongly Recommended):**
1. **Logit-Normal Hierarchical Model**
   - Handles heterogeneity (ICC=0.42)
   - Stabilizes small-sample estimates
   - Standard in literature
   - Extensible to covariates

**Tier 2 (Useful for Comparison):**
2. **Beta-Binomial Hierarchical Model**
   - Simpler, conjugate
   - Good for sensitivity check
   - Natural for proportions

**Tier 3 (Advanced/Robustness):**
3. **Mixture Model**
   - If suspect distinct subpopulations
   - More flexible than single hierarchy
   - More complex to fit and interpret

### 6.2 Software Implementation

**Stan (Recommended):**
```stan
data {
  int<lower=1> J;              // number of groups
  int<lower=0> n[J];           // trials per group
  int<lower=0> y[J];           // successes per group
}
parameters {
  real mu;                     // population mean (logit scale)
  real<lower=0> tau;           // between-group SD
  vector[J] theta_raw;         // non-centered parameterization
}
transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}
model {
  mu ~ normal(-2.6, 1);
  tau ~ normal(0, 0.05);
  theta_raw ~ normal(0, 1);
  y ~ binomial_logit(n, theta);
}
generated quantities {
  vector[J] p = inv_logit(theta);
  real p_new = inv_logit(normal_rng(mu, tau));  // prediction for new group
}
```

**PyMC (Alternative):**
```python
import pymc as pm

with pm.Model() as model:
    # Hyperpriors
    mu = pm.Normal('mu', mu=-2.6, sigma=1)
    tau = pm.HalfNormal('tau', sigma=0.05)

    # Group effects
    theta = pm.Normal('theta', mu=mu, sigma=tau, shape=J)
    p = pm.Deterministic('p', pm.math.invlogit(theta))

    # Likelihood
    y = pm.Binomial('y', n=n, p=p, observed=y_obs)

    # Sample
    trace = pm.sample(2000, tune=1000)
```

### 6.3 Model Validation Plan

**Convergence Diagnostics:**
- Rhat < 1.01 for all parameters
- Effective sample size > 400
- No divergent transitions
- Trace plots show good mixing

**Posterior Predictive Checks:**
1. Simulate y_rep from posterior
2. Compare to observed y
3. Check coverage of 95% predictive intervals
4. Look for systematic patterns in residuals

**Sensitivity Checks:**
1. Run with 3 different priors
2. Check if posterior means change > 10%
3. Check if credible intervals overlap substantially
4. Most important for τ and small-sample groups

**Comparison to Alternatives:**
1. Empirical Bayes (frequentist shrinkage)
2. Complete pooling (as baseline)
3. No pooling (as baseline)
4. Should hierarchical Bayes give sensible intermediate estimates

### 6.4 Expected Results

**Population Parameters:**
- μ (logit): Posterior mean ≈ -2.6, SD ≈ 0.2
- μ (probability): Posterior mean ≈ 0.07, 95% CI ≈ [0.05, 0.09]
- τ (logit SD): Posterior mean ≈ 0.02, 95% CI ≈ [0.01, 0.04]

**Group Estimates:**
- Small groups (n<100): Shrunk substantially toward μ
- Large groups (n>500): Close to MLE
- Sparse groups (r≤5): Wide credible intervals

**Prediction for New Group:**
- Mean ≈ 0.07
- 95% interval ≈ [0.03, 0.13]
- Wider than posterior for μ (accounts for between-group variation)

---

## 7. Summary and Action Items

### 7.1 Key Findings Summary

**Data Quality:**
✓ Excellent - no cleaning needed
✓ All binomial constraints satisfied
✓ No missing values or errors

**Pooling Decision:**
✗ Complete pooling rejected (chi-square p < 0.001)
✗ No pooling not recommended (overfitting concern)
✓ **Partial pooling strongly recommended (ICC = 0.42)**

**Prior Specification:**
✓ Weakly informative hierarchical priors appropriate
✓ Must conduct sensitivity analysis (smallest n = 47)
✓ Plausible ranges well-characterized

**Modeling:**
✓ Logit-normal hierarchical model recommended
✓ Beta-binomial alternative for comparison
✓ Expect shrinkage factor ≈ 0.5 on average

### 7.2 Immediate Action Items

1. **Fit hierarchical model** with recommended priors
2. **Check MCMC diagnostics** (convergence, ESS, divergences)
3. **Run sensitivity analysis** with 3 prior specifications
4. **Perform posterior predictive checks** for all groups
5. **Compare to empirical Bayes** shrinkage estimator
6. **Report results** with uncertainty quantification

### 7.3 Critical Questions to Answer

Before finalizing model:
- What do the groups represent contextually?
- Should any groups be excluded or combined?
- Are there covariates that explain group differences?
- Is prediction for existing or new groups the goal?

After fitting model:
- Do posteriors pass convergence diagnostics?
- Are results robust to prior choice?
- Do posterior predictive checks pass?
- Are estimates sensible given domain knowledge?

### 7.4 Files and Code

**Visualizations (all in `/workspace/eda/analyst_3/visualizations/`):**
1. `distributions_overview.png` - Data distributions and characteristics
2. `sample_size_adequacy.png` - Sample sizes vs success counts/rates
3. `success_rates_with_ci.png` - Group rates with confidence intervals
4. `precision_vs_sample_size.png` - How precision varies with n
5. `pooling_comparison.png` - Complete vs no vs partial pooling
6. `heterogeneity_test.png` - Evidence against homogeneity
7. `sample_size_precision_shrinkage.png` - Shrinkage visualization
8. `prior_options.png` - Comparison of prior distributions
9. `prior_sensitivity.png` - Prior influence on posteriors

**Code (all in `/workspace/eda/analyst_3/code/`):**
1. `01_initial_exploration.py` - Data quality checks
2. `02_data_quality_visualizations.py` - Distribution plots
3. `03_pooling_analysis.py` - Statistical tests for heterogeneity
4. `04_pooling_visualizations.py` - Pooling comparison plots
5. `05_prior_specification.py` - Prior parameter estimation
6. `06_prior_visualizations.py` - Prior comparison plots

**Reports:**
- `findings.md` - This report (summary)
- `eda_log.md` - Detailed exploration process

---

## Appendix: Supporting Evidence

### A. Group-Level Details

| Group | n | r | Observed Rate | 95% CI | Shrunk Rate | Residual |
|-------|---|---|---------------|---------|-------------|----------|
| 1 | 47 | 6 | 0.1277 | [0.054, 0.249] | 0.0823 | +1.26 |
| 2 | 148 | 19 | 0.1284 | [0.082, 0.190] | 0.0970 | +2.22 |
| 3 | 119 | 8 | 0.0672 | [0.033, 0.126] | 0.0687 | -0.11 |
| 4 | 810 | 34 | 0.0420 | [0.030, 0.058] | 0.0468 | -3.88 |
| 5 | 211 | 12 | 0.0569 | [0.032, 0.096] | 0.0626 | -0.80 |
| 6 | 196 | 13 | 0.0663 | [0.039, 0.109] | 0.0679 | -0.20 |
| 7 | 148 | 9 | 0.0608 | [0.032, 0.111] | 0.0655 | -0.49 |
| 8 | 215 | 30 | 0.1395 | [0.099, 0.193] | 0.1087 | +3.34 |
| 9 | 207 | 16 | 0.0773 | [0.047, 0.121] | 0.0739 | +0.43 |
| 10 | 97 | 3 | 0.0309 | [0.010, 0.087] | 0.0556 | -2.10 |
| 11 | 256 | 19 | 0.0742 | [0.047, 0.113] | 0.0724 | +0.23 |
| 12 | 360 | 27 | 0.0750 | [0.051, 0.107] | 0.0733 | +0.27 |

**Notes:**
- CI = Wilson score 95% confidence interval
- Shrunk Rate = Empirical Bayes estimate
- Residual = Standardized residual from pooled model

### B. Statistical Test Details

**Chi-Square Goodness of Fit:**
- H₀: All groups have same success rate (pooled = 0.0697)
- H₁: Groups have different success rates
- Test statistic: Σ[(O-E)²/E] = 39.52
- Critical value (α=0.05, df=11): 19.68
- Decision: Reject H₀ (39.52 > 19.68)

### C. Variance Calculation Details

**Method of Moments Estimator:**
1. Weighted mean: μ̂ = Σ(nⱼpⱼ) / Σnⱼ = 0.0697
2. Weighted variance: σ̂² = Σ[nⱼ(pⱼ - μ̂)²] / Σnⱼ = 0.000910
3. Within-group variance: σ̂²_within = mean[pⱼ(1-pⱼ)/nⱼ] = 0.000528
4. Between-group variance: τ̂² = max(0, σ̂² - σ̂²_within) = 0.000383
5. ICC = τ̂² / σ̂² = 0.420

---

**End of Report**

*For detailed exploration process, see `eda_log.md`*
*For code and reproducibility, see `/workspace/eda/analyst_3/code/`*
