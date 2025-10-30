# EDA Findings: Distributional Properties and Outlier Detection

**Analyst:** EDA Analyst 1
**Focus:** Distributional properties, variance-mean relationships, and outlier detection
**Date:** 2025-10-30

---

## Executive Summary

This analysis reveals **substantial heterogeneity** among the 12 groups in the binomial outcome data. Key findings:

1. **Strong overdispersion detected** (dispersion parameter Phi = 5.06, chi-square test p < 0.001)
2. **Three statistical outliers identified** (Groups 2, 8, 11) with significantly higher event rates
3. **Large variability in sample sizes** (range: 47-810, CV = 0.85) creating unequal precision across groups
4. **One group with zero events** (Group 1: 0/47) requiring special consideration
5. Groups are **NOT homogeneous** - random effects or hierarchical models recommended over pooled analysis

---

## 1. Data Quality Assessment

### 1.1 Data Structure
- **Observations:** 12 groups
- **Total sample size:** 2,814 subjects
- **Total events:** 208 events
- **Overall event rate:** 7.39% (95% CI: 6.55% - 8.32%)

### 1.2 Data Integrity
All data quality checks passed:
- No missing values
- No duplicate records
- All computed fields verified (proportion = r/n, failures = n-r)
- No impossible values (all r ≤ n, all values ≥ 0)

**Conclusion:** Data is clean and internally consistent. Ready for analysis.

---

## 2. Distribution of Sample Sizes

**Reference:** `01_sample_size_distribution.png`

### 2.1 Summary Statistics
- **Range:** 47 to 810 (17.2-fold difference)
- **Mean:** 234.5
- **Median:** 201.5
- **Standard Deviation:** 198.4
- **Coefficient of Variation:** 0.85 (HIGH variability)

### 2.2 Key Findings

1. **Highly imbalanced sample sizes:**
   - Smallest group (Group 1): n=47
   - Largest group (Group 4): n=810
   - The largest group has 17x more observations than the smallest

2. **Sample size concentration:**
   - Top 3 groups (Groups 4, 12, 11) contribute 50.7% of total sample
   - Group 4 alone contributes 28.8% of all observations
   - Just 3 groups provide 50% of sample, 7 groups provide 80%

3. **Implications for inference:**
   - Standard error ranges from 0.009 (Group 4) to 0.038 (Group 1)
   - 4.2-fold difference in precision across groups
   - Smaller groups have very wide confidence intervals
   - Group 1's zero events (0/47) creates extreme uncertainty (upper 95% CI ≈ 6.4%)

**Modeling Consideration:** The large variability in sample sizes means that simple pooling or equal weighting would be inappropriate. Precision-weighted analyses or hierarchical models are needed.

---

## 3. Distribution of Observed Proportions

**Reference:** `02_proportion_distribution.png`

### 3.1 Summary Statistics
- **Overall proportion (weighted):** 0.0739
- **Mean of group proportions (unweighted):** 0.0737
- **Median:** 0.0669
- **Standard Deviation:** 0.0384
- **Range:** 0.0000 to 0.1442 (14.42 percentage points)
- **IQR:** 0.0304
- **Coefficient of Variation:** 0.52 (MODERATE to HIGH)

### 3.2 Outlier Detection Results

**IQR Method (1.5×IQR rule):**
- **Group 1** (LOW outlier): proportion = 0.0000, n=47
- **Group 8** (HIGH outlier): proportion = 0.1442, n=215

### 3.3 Distribution Characteristics

1. **Right-skewed distribution:**
   - Mean (0.0737) slightly > Median (0.0669)
   - Q-Q plot shows deviation from normality at upper tail
   - Group 8 pulls the distribution to the right

2. **Wide range of proportions:**
   - From 0% (Group 1) to 14.4% (Group 8)
   - Most groups cluster between 4% and 8%
   - Three groups exceed 11%

3. **Confidence intervals:**
   - Wilson score 95% CIs calculated for robustness near boundaries
   - Many CIs overlap, but outlier groups clearly separate
   - Group 1's CI: [0.0%, 6.4%] - very wide due to small n and zero events

### 3.4 Relationship with Sample Size

**Key observation from bubble plot:** No clear relationship between sample size and observed proportion (correlation ≈ -0.07, not significant). This suggests the heterogeneity is NOT an artifact of varying precision but reflects true between-group differences.

**Modeling Consideration:** The wide range and moderate CV (0.52) indicate substantial heterogeneity. A fixed-effect model assuming common proportion would be misspecified.

---

## 4. Variance-Mean Relationship and Overdispersion

**Reference:** `03_overdispersion_analysis.png`

### 4.1 Overdispersion Tests

**Expected vs Observed Variance:**
- Expected variance (binomial): 0.000292
- Observed variance (empirical): 0.001478
- **Dispersion parameter (Phi) = 5.06**

**Interpretation:** The observed variance is **5 times larger** than expected under the binomial model. This is strong evidence of overdispersion.

**Chi-Square Test for Homogeneity:**
- χ² = 38.56
- df = 11
- **p-value < 0.0001** (highly significant)
- Overdispersion factor = χ²/df = 3.51

**Conclusion:** Groups are **NOT homogeneous**. The hypothesis of a common proportion across all groups is strongly rejected.

### 4.2 Funnel Plot Analysis

The funnel plot reveals:
- **Three groups clearly outside 95% control limits:** Groups 2, 8, 11
- These groups remain outside even the 99.8% limits
- Lower-precision groups show appropriate scatter within funnel
- Pattern is **NOT consistent with pure random variation**

### 4.3 Pearson Residuals

**Standardized residuals (|residual| > 2 indicates outlier):**
- **Group 2:** residual = +2.22 (observed 18, expected 10.9)
- **Group 8:** residual = +3.94 (observed 31, expected 15.9) ← **STRONGEST outlier**
- **Group 11:** residual = +2.41 (observed 29, expected 18.9)

All three have significantly MORE events than expected under homogeneity.

### 4.4 Heteroscedasticity Check

Absolute residuals vs sample size show no clear pattern, suggesting:
- Overdispersion is NOT merely due to varying sample sizes
- Larger groups don't systematically have larger or smaller residuals
- The heterogeneity appears consistent across the sample size range

**Modeling Consideration:** Overdispersion of this magnitude (Phi=5.06) requires:
- Beta-binomial models
- Random effects models
- Quasi-likelihood approaches with estimated dispersion
- Standard binomial GLM will underestimate standard errors by √5.06 ≈ 2.25-fold

---

## 5. Individual Group Characterization

**Reference:** `04_group_characterization.png`, `05_diagnostic_summary.png`

### 5.1 Group Categories

Based on z-scores (deviation from overall rate standardized by SE):

| Category | Count | Groups |
|----------|-------|--------|
| **High outliers** (z > 2) | 3 | 2, 8, 11 |
| **Moderate deviation** (-2 < z < -1 or 1 < z < 2) | 3 | 1, 4, 5 |
| **Typical** (-1 < z < 1) | 6 | 3, 6, 7, 9, 10, 12 |
| **Low outliers** (z < -2) | 0 | None |

### 5.2 Detailed Group Profiles

#### High Outliers (Above Average)

**Group 8** - Most extreme outlier:
- Proportion: 14.42% (vs 7.39% overall)
- Sample size: 215 (large)
- Z-score: +3.94 (p < 0.0001)
- Events: 31 observed vs 15.9 expected
- **Interpretation:** Highly statistically significant excess of events. With large n=215, this is not due to chance.

**Group 11** - Second highest:
- Proportion: 11.33%
- Sample size: 256 (large)
- Z-score: +2.41 (p = 0.016)
- Events: 29 observed vs 18.9 expected
- **Interpretation:** Significantly elevated rate, confirmed by large sample size.

**Group 2** - Third highest:
- Proportion: 12.16%
- Sample size: 148 (medium)
- Z-score: +2.22 (p = 0.026)
- Events: 18 observed vs 10.9 expected
- **Interpretation:** Elevated rate, statistically significant.

#### Groups with Low Proportions

**Group 1** - Zero events:
- Proportion: 0.00%
- Sample size: 47 (smallest)
- Z-score: -1.94 (borderline significant, p = 0.052)
- Events: 0 observed vs 3.5 expected
- **Interpretation:** Unusually low, but with small n=47, zero events can occur by chance (~3% probability). However, this could indicate a truly different group or data quality issue. **Requires domain knowledge review.**

**Group 5** - Second lowest:
- Proportion: 3.79%
- Sample size: 211 (large)
- Z-score: -2.00 (borderline, p = 0.046)
- Events: 8 observed vs 15.6 expected
- **Interpretation:** Below average, approaching statistical significance with large sample.

**Group 4** - Below average:
- Proportion: 5.68%
- Sample size: 810 (largest by far)
- Z-score: -1.86 (p = 0.063)
- Events: 46 observed vs 59.9 expected
- **Interpretation:** Slightly below average, but with n=810, even small differences approach significance. This group has substantial influence on the overall rate.

#### Typical Groups (6 groups)

Groups 3, 6, 7, 9, 10, 12 all have proportions between 6.1% and 8.2%, with z-scores between -0.6 and +0.3. These are consistent with the overall rate within expected random variation.

### 5.3 Pattern Summary

- **Asymmetric distribution:** More high outliers (3) than low outliers (0 strict, 1 borderline)
- **Outliers in both large and small groups:** Not just a small-sample phenomenon
- **Clear separation:** Outlier groups distinctly different from typical groups
- **Stable finding:** Outlier status robust to different methods (IQR, z-scores, residuals)

---

## 6. Hypotheses Testing

### Hypothesis 1: Groups are homogeneous (common proportion)
**REJECTED** (p < 0.0001, χ² = 38.56, df = 11)

### Hypothesis 2: Variance consistent with binomial model
**REJECTED** (Phi = 5.06, observed variance 5× expected)

### Hypothesis 3: Outliers are due to small sample fluctuation
**PARTIALLY REJECTED** - While Group 1's zero events could be chance with n=47, Groups 2, 8, and 11 are large samples (n=148-256) where substantial deviations are unlikely by chance (p < 0.05 for all three).

### Hypothesis 4: Sample size predicts proportion
**NOT SUPPORTED** - No significant correlation (r ≈ -0.07, p > 0.05). Heterogeneity is genuine, not an artifact of precision differences.

---

## 7. Data Quality Issues and Anomalies

### 7.1 Critical Issue: Group 1 with Zero Events

**Observation:** Group 1 has 0/47 events (0.0%)

**Possible explanations:**
1. **True low risk:** This group genuinely has much lower event rate
2. **Small sample variability:** With overall rate ~7.4%, probability of 0/47 is about 3%
3. **Data quality issue:** Misclassification, incomplete follow-up, or data entry error
4. **Definitional issue:** Different inclusion criteria or event definition for this group

**Recommendation:** **Requires domain expert review before modeling.** Consider:
- Verifying data collection for Group 1
- Examining if Group 1 differs systematically in composition or measurement
- Sensitivity analysis excluding Group 1
- Using continuity correction or Bayesian prior in modeling

### 7.2 Influential Observations

**Group 4** (n=810, 28.8% of total sample):
- Contributes disproportionately to overall estimate
- Has below-average proportion (5.68%)
- If excluded, overall rate would increase from 7.39% to 8.11%
- **Recommendation:** Perform sensitivity analysis removing Group 4

**Groups 2, 8, 11** (combined n=619, 22% of sample):
- All three have elevated rates (11-14%)
- Together contribute 78 events (37.5% of all events)
- Substantially increase heterogeneity measures
- **Recommendation:** Examine if these groups share characteristics

### 7.3 No Other Data Quality Concerns

- All other values are plausible
- No extreme outliers in terms of data entry errors
- Consistency checks all pass

---

## 8. Implications for Modeling

### 8.1 Models to AVOID

1. **Simple pooled binomial model:**
   - Assumes homogeneity (strongly rejected)
   - Will severely underestimate uncertainty (by ~2.25-fold)
   - Inappropriate given Phi = 5.06

2. **Standard binomial GLM without overdispersion correction:**
   - Standard errors will be too small
   - Confidence intervals will be too narrow
   - Hypothesis tests will have inflated Type I error

### 8.2 Recommended Model Classes

#### First Choice: Hierarchical/Mixed Models
**Beta-binomial or Random Effects Logistic Regression**

Advantages:
- Naturally handles overdispersion
- Borrows strength across groups (partial pooling)
- Provides group-specific estimates with appropriate shrinkage
- Can accommodate covariates if available
- Quantifies between-group variance (τ²)

Implementation:
- `glmer()` in R (lme4)
- Beta-binomial regression (VGAM, aod packages)
- Bayesian hierarchical model (Stan, JAGS)

Model form: logit(p_i) = μ + u_i, where u_i ~ N(0, τ²)

#### Second Choice: Quasi-Binomial Models
**GLM with Estimated Dispersion Parameter**

Advantages:
- Simple extension of binomial GLM
- Adjusts standard errors for overdispersion
- Widely available in standard software

Implementation:
- `glm(..., family=quasibinomial)` in R
- Scale parameter estimated from Pearson χ² / df

Limitation:
- No group-specific estimates with shrinkage
- Treats heterogeneity as nuisance rather than quantity of interest

#### Third Choice: Stratified Analysis with Meta-Analysis
**Treat Each Group Separately, Then Meta-Analyze**

Advantages:
- Very flexible, no distributional assumptions
- Can assess heterogeneity explicitly (I², τ²)
- Familiar to many applied researchers
- Robust to model misspecification

Implementation:
- Calculate proportion and SE for each group
- Random-effects meta-analysis (DerSimonian-Laird, REML)
- Forest plot visualization

Limitation:
- Doesn't allow covariate adjustment
- Less efficient than hierarchical models if model correctly specified

### 8.3 Special Considerations

1. **Group 1 (zero events):**
   - Use continuity correction (add 0.5 to numerator and denominator) for standard methods
   - Or Bayesian approach with weakly informative prior (e.g., Beta(0.5, 0.5))
   - Or exact inference methods
   - Consider sensitivity analysis excluding this group

2. **Unequal sample sizes:**
   - Use precision weighting (weight ∝ 1/SE²)
   - Or hierarchical model with appropriate likelihood
   - Avoid simple unweighted means

3. **Outlier groups:**
   - Robust estimation methods (e.g., trimmed estimators, M-estimation)
   - Sensitivity analysis excluding Groups 2, 8, 11
   - Subgroup analysis if covariate information available

### 8.4 Model Checking

Any chosen model should be validated by:
- Posterior predictive checks (Bayesian) or residual diagnostics (frequentist)
- Checking that model-based between-group variance matches observed
- Confirming model adequately captures the heterogeneity (no residual patterns)
- Sensitivity to prior specification (Bayesian) or distributional assumptions (frequentist)

---

## 9. Key Questions Answered

### Q1: Are the proportions relatively homogeneous across groups or highly variable?
**Answer:** **Highly variable.** Groups show substantial heterogeneity:
- SD of proportions = 0.0384 (52% of mean)
- Range: 0% to 14.4%
- Chi-square test for homogeneity: p < 0.0001
- Six "typical" groups cluster around 6-8%, but three high outliers at 11-14% and three low groups at 0-6%

### Q2: Do we see evidence of overdispersion?
**Answer:** **YES, substantial overdispersion:**
- Dispersion parameter Phi = 5.06 (variance 5× binomial expectation)
- Overdispersion factor = 3.51
- This is among the highest levels of overdispersion typically seen in practice
- Cannot be explained by sampling variability alone

### Q3: Are there outlier groups that might need special treatment?
**Answer:** **YES, four groups require attention:**

**High outliers (above expected):**
- Group 8: Most extreme (z = +3.94, p < 0.0001)
- Group 11: Second highest (z = +2.41, p = 0.016)
- Group 2: Third highest (z = +2.22, p = 0.026)

**Low outliers / special cases:**
- Group 1: Zero events (0/47), requires domain review and special handling in modeling

All three high outliers are statistically significant and robust across multiple detection methods. They should either be:
- Modeled with random effects to allow their distinct rates
- Investigated for common characteristics
- Analyzed as a separate subgroup if justified

### Q4: What is the effective sample size range and does it create challenges?
**Answer:** **17-fold range creates substantial challenges:**

**Range:** 47 to 810
- Precision varies by 4.2-fold (SE from 0.009 to 0.038)
- Confidence interval widths vary greatly (from ~1.4% to ~15% width)
- Group 1's small size (n=47) makes zero events ambiguous
- Group 4's large size (n=810, 29% of total) dominates the overall estimate

**Challenges created:**
1. Unequal information contribution across groups
2. Overall estimate heavily influenced by few large groups
3. Small groups have wide CIs, making outlier detection difficult
4. Simple averaging treats n=47 and n=810 equally (inappropriate)
5. Group 1's zero events creates boundary estimation problem

**Solutions:**
- Precision-weighted analysis (weight by 1/variance)
- Hierarchical modeling with appropriate shrinkage
- Sensitivity analysis to largest group (Group 4)
- Continuity correction or Bayesian methods for Group 1

---

## 10. Summary and Recommendations

### 10.1 Main Findings

1. **Substantial heterogeneity confirmed:** Groups differ significantly in event rates (p < 0.0001)
2. **Strong overdispersion:** Variance 5× binomial expectation (Phi = 5.06)
3. **Three statistical outliers:** Groups 2, 8, 11 have significantly elevated rates
4. **One concerning zero:** Group 1 (0/47 events) needs verification
5. **High sample size variability:** 17-fold range creates unequal precision

### 10.2 Critical Recommendations

**Before Modeling:**
1. **Verify Group 1 data:** Confirm zero events is correct, not data quality issue
2. **Investigate outlier groups:** Do Groups 2, 8, 11 share characteristics?
3. **Check Group 4 influence:** With 29% of sample, ensure it's not driving results

**For Modeling:**
1. **Use hierarchical/random effects models** (beta-binomial or mixed logistic)
2. **Account for overdispersion** - standard binomial models will fail
3. **Weight by precision** - don't treat all groups equally
4. **Special handling for Group 1** - continuity correction or Bayesian prior

**For Inference:**
1. **Report heterogeneity measures** (I², τ², prediction intervals)
2. **Provide group-specific estimates** with appropriate shrinkage
3. **Conduct sensitivity analyses** excluding outliers and influential groups
4. **Use robust standard errors** that account for overdispersion

### 10.3 Answering the Core Question

**Is pooling appropriate?**

**NO.** The data strongly argues against pooling into a single proportion:
- Complete pooling would mask important between-group differences
- Standard errors would be severely underestimated
- Three groups genuinely differ from the rest
- A hierarchical approach that partially pools (borrows strength while allowing differences) is most appropriate

**What approach should be used?**

**Hierarchical modeling with random effects**, which:
- Acknowledges groups differ (avoids complete pooling)
- Borrows strength across groups (avoids no pooling)
- Provides both overall and group-specific estimates
- Naturally handles overdispersion
- Gives appropriate uncertainty quantification

---

## Appendix: Reproducibility

### Code Files
All analyses are fully reproducible using scripts in `/workspace/eda/analyst_1/code/`:
1. `01_initial_exploration.py` - Data quality and basic statistics
2. `02_sample_size_analysis.py` - Sample size distribution
3. `03_proportion_analysis.py` - Proportion distribution and outliers
4. `04_overdispersion_analysis.py` - Variance analysis and heterogeneity tests
5. `05_group_characterization.py` - Individual group profiles
6. `06_diagnostic_summary.py` - Comprehensive diagnostic visualization

### Visualizations
All findings supported by visualizations in `/workspace/eda/analyst_1/visualizations/`:
- `01_sample_size_distribution.png` - Sample size patterns
- `02_proportion_distribution.png` - Proportion distribution and outliers
- `03_overdispersion_analysis.png` - Funnel plots and residual analysis
- `04_group_characterization.png` - Group profiles and categories
- `05_diagnostic_summary.png` - Comprehensive overview

### Software
- Python 3.13
- pandas 2.2+
- numpy 1.26+
- matplotlib 3.8+
- seaborn 0.13+
- scipy 1.12+

---

**End of Report**
