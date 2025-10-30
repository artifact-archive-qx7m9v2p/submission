# EDA Log - Analyst 1: Distributional Characteristics and Variance Patterns

## Initial Observations

### Dataset Structure
- **Size**: 12 groups (small sample!)
- **Variables**: group_id, n_trials, r_successes, success_rate
- **No missing values** - clean dataset

### Key Initial Findings

1. **Success Rate Distribution**
   - Mean: 0.0789 (7.89%)
   - Median: 0.0707 (7.07%)
   - SD: 0.0348
   - Range: [0.0309, 0.1395] - approximately 4.5x difference between min and max
   - **Positive skewness (0.73)**: Right-skewed distribution suggesting a few groups with higher success rates
   - **Negative kurtosis (-0.51)**: Flatter than normal distribution
   - **High CV (0.44)**: Substantial relative variability

2. **Sample Size (n_trials) Distribution**
   - Mean: 234.5 trials
   - Median: 201.5 trials
   - Range: [47, 810] - 17x difference!
   - **Highly right-skewed (2.52)** with high kurtosis (7.34)
   - Suggests one or more groups with very large sample sizes (potential outlier: group 4 with 810 trials)

3. **Preliminary Questions**
   - Are the success rates truly heterogeneous, or is this variation explained by binomial sampling?
   - Does variance scale appropriately with n_trials?
   - Are there outlier groups that don't fit the pattern?
   - Is the heterogeneity in success rates related to sample size?

---

## Round 1: Distribution Analysis

### Visualization Created: `01_distribution_overview.png`

**Key Findings:**

1. **Success Rate Outliers (IQR Method)**
   - Group 8: success_rate = 0.1395 (215 trials) - OUTLIER
   - Approximately double the pooled rate!

2. **Sample Size Outliers**
   - Group 4: 810 trials (3.6x larger than mean)
   - Group 12: 360 trials
   - These groups have far more data than others

3. **Normality Assessment**
   - Shapiro-Wilk p-value = 0.096 (marginally cannot reject normality)
   - However, Q-Q plot shows some deviation
   - Right skewness confirmed (0.73)
   - INTERPRETATION: Borderline normal, with tendency toward right tail

4. **Extreme Groups**
   - **Lowest rates**: Group 10 (3.09%), Group 4 (4.20%), Group 5 (5.69%)
   - **Highest rates**: Group 8 (13.95%), Group 2 (12.84%), Group 1 (12.77%)
   - Note: Group 1 has smallest sample size (n=47), so high rate could be noise

---

## Round 2: Sample Size Relationship

### Visualization Created: `02_sample_size_relationship.png`

**Key Findings:**

1. **Pooled Success Rate**: 6.97% (196 successes / 2814 total trials)

2. **Groups Outside 95% Confidence Band** (CRITICAL FINDING!)
   - **Group 2**: rate=12.84%, z-score=2.81 (p=0.005)
   - **Group 4**: rate=4.20%, z-score=-3.09 (p=0.002)
   - **Group 8**: rate=13.95%, z-score=4.03 (p<0.001)

3. **Groups Outside 99.7% Band** (EXTREME outliers)
   - Group 4 (z=-3.09): 40% BELOW pooled rate
   - Group 8 (z=4.03): 100% ABOVE pooled rate
   - These are statistically very unlikely under homogeneous model

4. **Correlation with Sample Size**
   - Pearson r = -0.341 (p=0.278) - not significant
   - Spearman rho = -0.018 (p=0.957) - no relationship
   - **INTERPRETATION**: Success rate heterogeneity is NOT explained by sample size
   - This suggests true underlying differences between groups

---

## Round 3: Funnel Plot Analysis

### Visualization Created: `03_funnel_plot.png`

**Key Findings:**

1. **Funnel Plot Interpretation**
   - 25% of groups (3/12) outside 95% CI funnel
   - Expected under homogeneity: ~5%
   - **5x excess of outliers - SUBSTANTIAL evidence of heterogeneity**

2. **High-Precision Outliers** (SMOKING GUN!)
   - 7 groups with n > 150
   - 2 of these are outside 95% CI (Groups 4 and 8)
   - **CRITICAL**: These large-sample outliers cannot be dismissed as noise
   - Strong evidence that groups truly differ in success rates

3. **Asymmetry Test**
   - 6 groups above pooled rate (mean n=205.5)
   - 6 groups below pooled rate (mean n=263.5)
   - Egger's regression intercept: 2.54 (p=0.14) - not significant
   - **INTERPRETATION**: No systematic bias, but true heterogeneity present

---

## Round 4: Variance Analysis (MOST IMPORTANT!)

### Visualization Created: `04_variance_analysis.png`

**Key Findings:**

1. **Overdispersion Test**
   - Empirical variance: 0.00121
   - Expected binomial variance: 0.00044
   - **Variance Ratio: 2.78** (nearly 3x larger than expected!)
   - **CONCLUSION: SUBSTANTIAL OVERDISPERSION**

2. **Chi-Square Test for Homogeneity**
   - Chi-square = 39.52 (df=11)
   - p-value = 0.000043
   - Critical value = 19.68
   - **STRONGLY REJECT homogeneity hypothesis**

3. **Variance Decomposition**
   - Total variance: 0.00121
   - Within-group (binomial) variance: 0.00044
   - **Between-group variance: 0.00077**
   - **64% of variance is between-group differences!**
   - Only 36% from binomial sampling noise

4. **Coefficient of Variation**
   - Empirical CV: 0.441
   - Expected CV: 0.300
   - Ratio: 1.47
   - Much more variability than binomial model predicts

5. **Standardized Residuals**
   - 3 groups with |z| > 1.96
   - 3 groups with |z| > 2.576
   - 2 groups with |z| > 3 (extreme)
   - Mean z-score: 0.29 (slightly positive - suggests mild right skew)
   - SD of z-scores: 1.87 (should be ~1.0 under homogeneity)

---

## Round 5: Group-Specific Analysis

### Visualization Created: `05_group_specific_analysis.png`

**Detailed Outlier Profiles:**

1. **Group 8 (EXTREME HIGH)**
   - n=215, successes=30, rate=13.95%
   - Z-score: 4.03
   - Deviation: +6.99 percentage points (+100.3% relative to pooled)
   - P(|z| >= 4.03) = 0.000057 (1 in 17,500 chance under null!)

2. **Group 4 (EXTREME LOW)**
   - n=810, successes=34, rate=4.20%
   - Z-score: -3.09
   - Deviation: -2.77 percentage points (-39.7% relative to pooled)
   - P(|z| >= 3.09) = 0.0020 (1 in 500 chance)

3. **Group 2 (MODERATE HIGH)**
   - n=148, successes=19, rate=12.84%
   - Z-score: 2.81
   - Within 3-sigma, but outside 95% CI

4. **Typical Groups**
   - 9 groups (75%) fall within 95% CI
   - These show expected binomial variation
   - Mean success rate: 7.07%

**Category Comparison:**
- Extreme outliers: mean n=512, mean rate=9.08%
- Typical groups: mean n=182, mean rate=7.07%
- Outliers tend to have larger samples (but not significant, p=0.12)

---

## Round 6: Hypothesis Testing

**Three competing hypotheses tested:**

### H1: Homogeneous Model (All groups same rate)
- Chi-square test: p = 0.000043 → **REJECTED**
- Variance ratio: 2.78 → **Overdispersion detected**
- AIC: 90.63
- BIC: 91.11
- **VERDICT: Strong evidence AGAINST this model**

### H2: Sample-Size Dependent (Rate varies with n)
- Pearson correlation: r = -0.341 (p=0.278) → Not significant
- Linear regression: slope = -0.00006 (p=0.278)
- T-test (small vs large n): p = 0.899
- **VERDICT: NO SUPPORT for this hypothesis**

### H3: Heterogeneous Model (Groups differ)
- Likelihood ratio test: LR = 36.27, p = 0.000153 → **STRONG SUPPORT**
- AIC: 76.36 (14.3 points better than H1)
- BIC: 82.18 (8.9 points better than H1)
- High-precision outliers: 2 detected
- **VERDICT: STRONG EVIDENCE for this model**

**Overall Conclusion:**
STRONG EVIDENCE for heterogeneity (H3) based on:
1. Chi-square test rejects homogeneity
2. Variance ratio (2.78) indicates overdispersion
3. Likelihood ratio test favors heterogeneous model
4. 2 high-precision outliers detected
5. Lower AIC/BIC for heterogeneous model

---

## Key Insights Summary

1. **Heterogeneity is REAL, not noise**: 64% of variance is between-group differences
2. **Not explained by sample size**: Correlation with n is weak and non-significant
3. **Two extreme outliers** (Groups 4 and 8) drive much of the heterogeneity
4. **Most groups are typical**: 75% fall within expected binomial variation
5. **Distribution is right-skewed**: A few groups have higher success rates

## Implications for Modeling

**STRONG RECOMMENDATION: Use hierarchical/mixed-effects model**

Reasons:
1. Clear evidence of between-group heterogeneity
2. Cannot assume common success rate
3. Need to model both within and between-group variance
4. Should allow for group-specific random effects

**Model Suggestions:**
1. **Beta-binomial model**: Accounts for overdispersion
2. **Hierarchical logistic regression**: Group-level random intercepts
3. **Bayesian hierarchical model**: Can estimate group-specific rates with shrinkage
4. **Consider outlier-robust models**: Groups 4 and 8 may need special treatment

**Do NOT use:**
- Simple pooled binomial model (strongly rejected)
- Fixed-effects only models (ignores heterogeneity structure)
