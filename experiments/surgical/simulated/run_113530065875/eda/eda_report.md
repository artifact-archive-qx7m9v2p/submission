# Exploratory Data Analysis Report
## Binomial Data - 12 Groups

**Date**: 2024
**Analysis Method**: Parallel independent analysts with synthesis

---

## Executive Summary

This EDA examined binomial data from 12 groups (total N=2,814 trials, 196 successes). **Strong evidence supports a Bayesian hierarchical model** with partial pooling:

- ✅ **Overdispersion confirmed**: Variance 3.6× expected under binomial model (χ²=39.47, p<0.0001)
- ✅ **Hierarchical structure**: ICC=0.56 (56% variance between groups)
- ✅ **Exchangeable groups**: No ordering/covariate effects detected
- ✅ **Three outlier groups identified**: Groups 2, 4, 8 (need careful monitoring)
- ✅ **Priors recommended**: Beta(5,50) for rates, Half-Cauchy(0,1) for between-group SD

---

## Data Structure

| Metric | Value |
|--------|-------|
| Number of groups | 12 |
| Total trials (N) | 2,814 |
| Total successes (r) | 196 |
| Sample size range | 47 - 810 trials per group |
| Success rate range | 3.1% - 14.0% |

**Raw Data:**
```
group,n,r,rate
1,47,6,12.8%
2,148,19,12.8%
3,119,8,6.7%
4,810,34,4.2%    ← Largest group, lowest rate
5,211,12,5.7%
6,196,13,6.6%
7,148,9,6.1%
8,215,30,14.0%   ← Highest rate
9,207,16,7.7%
10,97,3,3.1%     ← Smallest rate
11,256,19,7.4%
12,360,27,7.5%
```

---

## Key Findings

### 1. Strong Overdispersion (Groups Are Heterogeneous)

**Evidence:**
- **Chi-square test**: χ² = 39.47 (df=11), p < 0.0001
- **Dispersion parameter**: φ = 3.59 (observed variance = 3.6 × theoretical)
- **ICC**: 0.56 (56% of total variance is between-group)

**Visual Evidence:**
- `eda/analyst_1/visualizations/funnel_plot.png` - Many groups outside 99.8% confidence bands
- `eda/analyst_2/visualizations/06_deviation_analysis.png` - Systematic deviations from pooled rate

**Interpretation**: Groups have genuinely different success rates, not just sampling noise. A pooled (single-rate) model is **empirically rejected**.

### 2. Success Rate Distribution

| Statistic | Value |
|-----------|-------|
| Pooled rate | 6.97% (95% CI: [6.1%, 8.0%]) |
| Mean group rate | 7.70% |
| Median group rate | 6.70% |
| SD of group rates | 3.39% |
| Range | 3.1% - 14.0% (4.5-fold difference) |

**Visual Evidence:**
- `eda/analyst_1/visualizations/success_rate_by_group.png` - Bar chart showing wide variation
- `eda/analyst_2/visualizations/01_caterpillar_plot.png` - Rates with confidence intervals

**Interpretation**: Substantial heterogeneity with roughly symmetric distribution around pooled rate.

### 3. Extreme Groups (Outliers)

| Group | n | r | Rate | Std. Residual | Status |
|-------|---|---|------|---------------|--------|
| 8 | 215 | 30 | 14.0% | +4.0 | High outlier |
| 2 | 148 | 19 | 12.8% | +2.8 | High outlier |
| 4 | 810 | 34 | 4.2% | -3.1 | Low outlier (dominates 29% of data) |
| 10 | 97 | 3 | 3.1% | -1.6 | Low (small sample) |

**Visual Evidence:**
- `eda/analyst_1/visualizations/success_rate_by_group.png` - Outliers highlighted in red
- `eda/analyst_2/visualizations/09_extreme_groups.png` - Multi-panel extreme group analysis

**Interpretation**:
- Group 4 is particularly influential (29% of all trials) with low rate
- Groups 2 and 8 have significantly higher rates than population
- These will drive shrinkage patterns in hierarchical model

### 4. Sample Size Effects

**Findings:**
- **No systematic bias**: Correlation between n and rate = -0.34 (p=0.28, not significant)
- **Precision follows theory**: SE ∝ 1/√n as expected
- **Range**: 16-fold difference (47 to 810 trials)

**Visual Evidence:**
- `eda/analyst_1/visualizations/success_rate_vs_sample_size.png` - Scatter with trend line
- `eda/analyst_2/visualizations/08_sample_size_precision.png` - Power-law relationship

**Interpretation**: Large and small groups have similar mean rates. Sample size can be used for precision weighting without bias concerns.

### 5. Exchangeability (No Ordering Effects)

**Statistical Tests:**
- **Runs test**: Z = -1.21 (p=0.23) → Random pattern, no autocorrelation
- **Correlation with group ID**: r = -0.21 (p=0.30) → No trend
- **Levene's test**: p = 0.74 → Variance homogeneous across sample sizes

**Interpretation**: Groups appear exchangeable. No evidence for temporal/spatial structure. Standard hierarchical model assumptions are met.

---

## Hierarchical Model Evidence

### Pooling Comparison

| Model | Structure | Goodness-of-Fit | Recommendation |
|-------|-----------|-----------------|----------------|
| **Pooled** | Single rate for all groups | χ² p < 0.0001 (rejected) | ❌ Do not use |
| **Unpooled** | Independent rate per group | Overparameterized | ❌ Wastes information |
| **Hierarchical** | Partial pooling | Expected to fit best | ✅ **Recommended** |

**Visual Evidence:**
- `eda/analyst_2/visualizations/03_pooling_comparison.png` - **KEY PLOT** showing shrinkage arrows
- `eda/analyst_2/visualizations/07_hierarchical_evidence.png` - Variance decomposition

### Expected Shrinkage Patterns

**Between-group standard deviation**: τ ≈ 0.36 on logit scale

| Sample Size Category | Expected Shrinkage | Example Groups |
|---------------------|-------------------|----------------|
| Small (n<100) | 60-72% | Groups 1, 10 |
| Medium (n=100-250) | 30-50% | Groups 2, 3, 5, 6, 7, 8, 9, 11 |
| Large (n>250) | 19-30% | Groups 4, 12 |

**Visual Evidence:**
- `eda/analyst_2/visualizations/04_shrinkage_analysis.png` - Quantified shrinkage by sample size

**Interpretation**: Small-sample groups will borrow substantial strength from population mean. Group 4 (n=810) will shrink least, anchoring population estimate near 4.2%.

---

## Prior Recommendations for Bayesian Model

### Group-Level Success Rates (θᵢ)
**Recommended**: Beta(5, 50) or equivalent logit-normal
- **Mean**: ~0.09 (9%)
- **95% interval**: [0.02, 0.20]
- **Rationale**: Weakly informative, covers observed range (3-14%), centered slightly above pooled rate

### Population Mean (μ on logit scale)
**Recommended**: Normal(-2.5, 1)
- **Back-transforms to**: ~7-8% success rate
- **Rationale**: Weakly informative prior centered at reasonable value

### Between-Group SD (τ on logit scale)
**Recommended**: Half-Cauchy(0, 1)
- **Allows range**: 0 to ~2 (most mass below 1)
- **Rationale**: Standard weakly informative prior for hierarchical SD, regularizes extreme heterogeneity

**Visual Evidence:**
- `eda/analyst_2/visualizations/05_prior_predictive.png` - Evaluates 6 prior options against data

---

## Visualizations Index

### Analyst 1 (Diagnostics Focus)
1. **success_rate_by_group.png** - Bar chart with outliers highlighted
2. **success_rate_vs_sample_size.png** - Scatter showing no confounding
3. **success_rate_distribution.png** - Histogram and box plot
4. **funnel_plot.png** - Precision vs rate with confidence bands (shows overdispersion)
5. **diagnostic_panel.png** - 6-panel comprehensive diagnostics

### Analyst 2 (Pooling Focus)
1. **01_caterpillar_plot.png** - Individual rates vs pooled
2. **02_forest_plot_sample_sizes.png** - Forest plot with CIs
3. **03_pooling_comparison.png** - ⭐ **KEY**: Three-way comparison with shrinkage arrows
4. **04_shrinkage_analysis.png** - Expected shrinkage quantification
5. **05_prior_predictive.png** - Prior evaluation
6. **06_deviation_analysis.png** - Deviation patterns
7. **07_hierarchical_evidence.png** - Variance decomposition
8. **08_sample_size_precision.png** - Precision analysis
9. **09_extreme_groups.png** - Extreme group identification
10. **10_summary_dashboard.png** - Comprehensive overview

---

## Model Specification for Phase 2

### Recommended Model: Bayesian Hierarchical Binomial

**Stan/PyMC Pseudocode:**
```
data {
  int<lower=1> J;              // Number of groups (12)
  int<lower=0> n[J];           // Trials per group
  int<lower=0> r[J];           // Successes per group
}

parameters {
  real mu;                      // Population mean (logit scale)
  real<lower=0> tau;            // Between-group SD (logit scale)
  vector[J] theta_raw;          // Non-centered parameterization
}

transformed parameters {
  vector[J] theta = mu + tau * theta_raw;  // Group-level logit rates
}

model {
  // Priors
  mu ~ normal(-2.5, 1);         // Weakly informative
  tau ~ cauchy(0, 1);           // Half-Cauchy (lower=0 in parameters)
  theta_raw ~ normal(0, 1);     // Non-centered

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  vector[J] p = inv_logit(theta);           // Success probabilities
  vector[J] log_lik;                         // For LOO
  for (j in 1:J) {
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
  }
}
```

### Alternative Models to Consider
1. **Beta-binomial** (simpler, population-level only)
2. **Robust hierarchical** (Student-t hyperprior instead of Normal)
3. **Pooled** (baseline for comparison only)
4. **Unpooled** (baseline for comparison only)

---

## Falsification Criteria

### Prior Predictive Checks
- ✅ Prior should generate rates in observed range (3-14%)
- ✅ Prior should allow for overdispersion (not centered at pooled rate)

### Posterior Predictive Checks
- ✅ Model should capture variance ratio ~3.6
- ✅ Posterior predictions should cover observed data
- ✅ Extreme groups should not be systematically mis-predicted

### Model Comparison
- ✅ Hierarchical should outperform pooled (ΔLOO > 2×SE)
- ✅ LOO should prefer hierarchical over unpooled (parsimony)
- ✅ Pareto k values should be acceptable (<0.7)

### Shrinkage Validation
- ✅ Small-n groups should shrink more than large-n groups
- ✅ Group 4 (n=810) should shrink least (~19%)
- ✅ Groups near pooled rate should shrink minimally

---

## Data Quality

### ✅ Strengths
- No missing values
- No data entry errors (all r ≤ n)
- Reasonable total sample size (N=2,814)
- Clear signal of heterogeneity

### ⚠️ Challenges
- 16-fold range in sample sizes
- Group 4 dominates (29% of data)
- Group 10 has only 3 successes (unstable)
- Three outlier groups may be influential

### ✅ No Issues Detected
- No implausible values
- No collection artifacts
- Consistent binomial structure

---

## Conclusions

1. **Hierarchical Bayesian model is strongly recommended** based on overwhelming evidence for between-group heterogeneity (ICC=0.56, χ²=39.47 p<0.0001)

2. **Groups are exchangeable** - no covariates needed, standard hierarchical assumptions met

3. **Prior recommendations are concrete and weakly informative** - Beta(5,50) for rates, Half-Cauchy(0,1) for τ

4. **Expected challenges identified** - Group 4 dominance, outlier groups 2 and 8, small-sample instability in groups 1 and 10

5. **Falsification criteria defined** - Prior/posterior predictive checks, LOO comparison, shrinkage validation

**Next Step**: Proceed to Phase 2 (Model Design) with parallel designers to propose 2-3 specific model implementations.

---

## References to Detailed Reports
- Full Analyst 1 findings: `eda/analyst_1/findings.md`
- Full Analyst 2 findings: `eda/analyst_2/findings.md`
- Synthesis document: `eda/synthesis.md`
