# Exploratory Data Analysis Report: Eight Schools Dataset

**Analysis Date:** 2025-10-28
**Analyst:** EDA Specialist
**Dataset:** Eight Schools Hierarchical Meta-Analysis Problem

---

## Executive Summary

This report presents a comprehensive exploratory data analysis of the classic "Eight Schools" dataset, a hierarchical meta-analysis of treatment effects across eight educational institutions. The dataset consists of 8 observations, each with an observed treatment effect (`y`) and known measurement standard error (`sigma`).

**Key Findings:**
- **No statistical evidence of heterogeneity** across schools (Cochran's Q test: p = 0.696)
- **All observed variation is consistent with sampling error alone** (I² = 0%, tau² = 0)
- **Strong support for complete or near-complete pooling** of information
- **High measurement uncertainty** (mean sigma = 12.5) relative to signal (SD of effects = 10.4)
- **No outliers or influential observations** that would bias pooled estimates
- **Bayesian hierarchical model recommended** but expected to strongly favor low between-school variance

---

## 1. Data Overview

### 1.1 Dataset Structure

| Variable | Description | Type | Range |
|----------|-------------|------|-------|
| `school` | School identifier | Integer | 1-8 |
| `y` | Observed treatment effect | Numeric | [-3, 28] |
| `sigma` | Standard error of measurement | Numeric | [9, 18] |

**Complete Dataset:**

| School | Effect (y) | SE (sigma) | Precision | 95% CI |
|--------|-----------|------------|-----------|---------|
| 1 | 28 | 15 | 0.0044 | [-2, 58] |
| 2 | 8 | 10 | 0.0100 | [-12, 28] |
| 3 | -3 | 16 | 0.0039 | [-35, 29] |
| 4 | 7 | 11 | 0.0083 | [-15, 29] |
| 5 | -1 | 9 | 0.0123 | [-19, 17] |
| 6 | 1 | 11 | 0.0083 | [-21, 23] |
| 7 | 18 | 10 | 0.0100 | [-2, 38] |
| 8 | 12 | 18 | 0.0031 | [-24, 48] |

### 1.2 Data Quality

- **Completeness:** No missing values (100% complete)
- **Duplicates:** None detected
- **Consistency:** All values within plausible ranges
- **Known quantities:** Standard errors (sigma) are *known* measurement uncertainties, not estimates

---

## 2. Distribution Analysis

### 2.1 Observed Effects (y)

**Summary Statistics:**
- Mean: 8.75
- Median: 7.50
- Standard Deviation: 10.44
- Range: 31 (from -3 to 28)
- IQR: 13.0
- Skewness: 0.662 (moderate right skew)
- Kurtosis: -0.578 (slightly platykurtic)

**Distribution Shape:**
- Shapiro-Wilk test: W = 0.937, p = 0.583 (consistent with normality)
- Q-Q plot shows reasonable adherence to normal distribution with minor tail deviations
- Moderate positive skew driven by School 1's high value (28)

**Visualization Reference:** See `distribution_analysis.png` (panel A: histogram)

### 2.2 Standard Errors (sigma)

**Summary Statistics:**
- Mean: 12.50
- Median: 11.00
- Standard Deviation: 3.34
- Range: 9 (from 9 to 18)
- IQR: 5.25
- Coefficient of Variation: 0.267

**Key Observation:** Standard errors are relatively homogeneous across schools, varying by a factor of 2 (9 to 18). The mean standard error (12.5) is **larger than the standard deviation of observed effects (10.4)**, suggesting that measurement uncertainty dominates true heterogeneity.

**Visualization Reference:** See `distribution_analysis.png` (panel B: histogram)

### 2.3 Effect-Uncertainty Relationship

**Correlation Analysis:**
- Pearson correlation: r = 0.213, p = 0.612 (not significant)
- Spearman correlation: rho = 0.108, p = 0.798 (not significant)
- Linear regression R² = 0.045 (explains <5% of variance)

**Interpretation:** No evidence that larger treatment effects are associated with greater or lesser measurement uncertainty. This absence of correlation suggests no systematic bias or "small study effects."

**Visualization Reference:** See `effect_vs_uncertainty.png`

---

## 3. Heterogeneity Assessment

### 3.1 Classical Meta-Analysis Tests

#### Cochran's Q Test
```
Q = 4.71
df = 7
p-value = 0.6957
```

**Interpretation:** The Q statistic tests whether observed variation exceeds what would be expected by sampling error alone. With p = 0.696, we **fail to reject the null hypothesis of homogeneity**. This is strong evidence that all schools share a common true effect.

#### I² Statistic
```
I² = 0.0%
```

**Interpretation:** I² represents the percentage of total variation attributable to between-study heterogeneity rather than sampling error. Categories:
- 0-25%: Low heterogeneity
- 25-50%: Moderate heterogeneity
- 50-75%: Substantial heterogeneity
- 75-100%: Considerable heterogeneity

Our I² = 0% indicates **all observed variation is attributable to sampling error**.

#### Between-Study Variance (tau²)
```
DerSimonian-Laird estimate: tau² = 0.00, tau = 0.00
```

**Interpretation:** The estimated between-school variance is at the boundary of parameter space (cannot be negative). This point estimate suggests no true variation across schools beyond measurement error.

**Visualization Reference:** See `heterogeneity_diagnostics.png` (4-panel diagnostic)

### 3.2 Variance Decomposition

| Source | Variance | Percentage |
|--------|----------|------------|
| Observed total variance | 109.1 | 100% |
| Average sampling variance | 166.0 | 152% |
| Between-study variance (tau²) | 0.0 | 0% |

**Key Finding:** The observed variance (109) is actually **less than** the average sampling variance (166), yielding a variance ratio of 0.66. This is strong evidence that sampling error fully explains the observed variation.

### 3.3 Pooled Estimates

**Weighted Mean (inverse-variance weighting):**
- Estimate: 7.69
- Standard Error: 4.07
- 95% CI: [-0.30, 15.67]

**Unweighted Mean:**
- Estimate: 8.75
- Standard Error: 3.69

The similarity between weighted (7.69) and unweighted (8.75) means, combined with relatively uniform precision weights, suggests no strong precision-effect relationship.

---

## 4. Outlier and Influence Analysis

### 4.1 Standardized Residuals

Using the weighted mean (7.69) as reference:

| School | Effect | Residual | Std. Residual (z) | Flag |
|--------|--------|----------|-------------------|------|
| 1 | 28 | 20.31 | 1.35 | - |
| 2 | 8 | 0.31 | 0.03 | - |
| 3 | -3 | -10.69 | -0.67 | - |
| 4 | 7 | -0.69 | -0.06 | - |
| 5 | -1 | -8.69 | -0.97 | - |
| 6 | 1 | -6.69 | -0.61 | - |
| 7 | 18 | 10.31 | 1.03 | - |
| 8 | 12 | 4.31 | 0.24 | - |

**Findings:**
- No schools exceed the |z| > 2 outlier threshold
- Largest positive residual: School 1 (z = 1.35)
- Largest negative residual: School 5 (z = -0.97)
- All residuals are well within expected range

**Coverage Under Homogeneity:**
All 8 schools (100%) fall within their expected 95% confidence intervals assuming a common true effect of 7.69. This is exceptionally strong evidence for homogeneity.

**Visualization Reference:** See `precision_analysis.png` (panel B: standardized residuals)

### 4.2 Influential Observations

**Leave-One-Out Analysis:**

| School Removed | New Weighted Mean | Change | Influence Level |
|----------------|-------------------|--------|-----------------|
| 5 | 9.92 | +2.24 | Moderate (high precision, low effect) |
| 7 | 5.64 | -2.05 | Moderate (high precision, high effect) |
| 1 | 6.07 | -1.62 | Modest (extreme effect, low precision) |
| 6 | 8.75 | +1.06 | Low |
| 3 | 8.43 | +0.74 | Low |
| 8 | 7.45 | -0.23 | Low |
| 4 | 7.79 | +0.11 | Low |
| 2 | 7.62 | -0.06 | Low |

**Key Insights:**
- **School 5** (effect = -1, sigma = 9) is most influential, pulling mean down due to high precision and negative effect
- **School 7** (effect = 18, sigma = 10) is second most influential, pulling mean up
- **School 1** (effect = 28, sigma = 15), despite extreme value, has modest influence due to low precision (down-weighted)
- Maximum influence is ~2 points on pooled mean of 7.69 (±26%), which is moderate
- No school dominates the pooled estimate

**Visualization Reference:** See `heterogeneity_diagnostics.png` (panel D: leave-one-out)

---

## 5. Forest Plot Analysis

The forest plot (`forest_plot.png`) displays each school's observed effect with 95% confidence intervals (±2 standard errors), sorted by effect size.

**Key Observations:**

1. **Wide, Overlapping Intervals:** All confidence intervals substantially overlap, consistent with homogeneity

2. **School 1 (y=28, sigma=15):**
   - Highest point estimate but widest CI [-2, 58]
   - CI includes pooled mean (7.69) comfortably
   - Not a statistical outlier

3. **School 3 (y=-3, sigma=16):**
   - Only negative point estimate
   - Wide CI [-35, 29] includes pooled mean
   - Consistent with sampling variability

4. **Pooled Estimate (red dashed line at 7.69):**
   - Falls within or near all individual CIs
   - Central tendency clearly visible

5. **Precision Pattern:**
   - Schools 5 and 7 (sigma = 9-10) have narrowest CIs
   - Schools 1, 3, 8 (sigma = 15-18) have widest CIs
   - About 2:1 ratio in precision

**Visual Conclusion:** The forest plot provides no evidence of systematic heterogeneity. All observations are consistent with random sampling from a common underlying effect.

---

## 6. Hypothesis Testing Results

### Hypothesis 1: Complete Pooling (Homogeneity)
**H0:** All schools share the same true treatment effect (theta_1 = theta_2 = ... = theta_8)

**Evidence:**
- Cochran's Q test: p = 0.696 ✓
- I² = 0% ✓
- tau² = 0 ✓
- All observations within expected range ✓
- Variance ratio < 1 ✓

**Conclusion:** **STRONGLY SUPPORTED** - Data are entirely consistent with complete pooling.

### Hypothesis 2: No Pooling (Complete Independence)
**H0:** Each school has a completely independent true effect

**Evidence:**
- Individual CIs extremely wide (36-72 unit width)
- Variance ratio = 0.66 (observed < expected)
- No precision information gained from independence assumption

**Conclusion:** **NOT SUPPORTED** - Data do not require or benefit from complete independence.

### Hypothesis 3: Partial Pooling (Hierarchical Model)
**H0:** Schools share information through hierarchical structure with between-school variance tau² > 0

**Evidence:**
- tau² estimate = 0 (boundary)
- Empirical Bayes estimates: 100% shrinkage to pooled mean
- Mathematically equivalent to complete pooling given data

**Conclusion:** **APPROPRIATE FRAMEWORK** but data favor tau² ≈ 0, reducing to complete pooling.

### Hypothesis 4: Subgroup Structure
**H0:** Schools cluster into distinct subgroups

**Evidence:**
- Median split creates apparent "low" vs "high" groups
- Mann-Whitney U test: p = 0.029 (borderline significant)
- Gap ratio = 2.5 suggests possible bimodality
- **BUT:** Post-hoc analysis, contradicts Q test, likely spurious

**Conclusion:** **TENTATIVE/LIKELY SPURIOUS** - No strong evidence for distinct subgroups.

### Hypothesis 5: Effect-Uncertainty Relationship
**H0:** Larger effects are associated with larger (or smaller) uncertainties

**Evidence:**
- Pearson r = 0.213, p = 0.612 (not significant)
- Spearman rho = 0.108, p = 0.798 (not significant)
- No evidence of "small study effects" or publication bias

**Conclusion:** **NOT SUPPORTED** - No systematic relationship detected.

---

## 7. Modeling Recommendations

### 7.1 Recommended Approach: Bayesian Hierarchical Model

**Model Structure:**
```
y_i ~ Normal(theta_i, sigma_i)    [Data model: sigma_i are known]
theta_i ~ Normal(mu, tau)          [School effects: random effects]
mu ~ Normal(mu_0, sigma_mu)        [Prior on grand mean]
tau ~ Half-Cauchy(0, scale_tau)    [Prior on between-school SD]
```

**Rationale:**
1. **Philosophically appropriate:** Hierarchical structure respects the data generation process (schools are exchangeable units from a population)
2. **Conservative:** Allows for heterogeneity even though data don't require it
3. **Automatic adaptation:** Model will adapt to evidence, shrinking appropriately
4. **Handles boundary:** Bayesian approach naturally handles tau ≈ 0 without boundary issues
5. **Uncertainty quantification:** Provides full posterior distributions for all parameters

### 7.2 Prior Recommendations

#### Prior on tau (between-school SD):
**Recommended:** `Half-Cauchy(0, 5)` or `Half-Cauchy(0, 10)`

**Justification:**
- Gelman (2006) recommendation for hierarchical models
- Allows tau to be near 0 (as data suggest) but doesn't force it
- Heavy tails permit large tau if data support it
- Scale parameter 5-10 reasonable given observed effect scale (~10 units)

**Alternatives:**
- `Half-Normal(0, 10)`: More conservative, concentrates mass near 0
- `Uniform(0, 50)`: Less informative but upper bound needed for computation

**Expected Posterior:** Concentrated near 0 with long right tail capturing uncertainty

#### Prior on mu (grand mean):
**Recommended:** `Normal(0, 20)` or `Normal(10, 10)`

**Justification:**
- Weakly informative: allows wide range of plausible values
- Centered near 0 or small positive value (assuming treatment effects centered)
- SD = 10-20 accommodates observed range [-3, 28]
- Data are informative (pooled SE = 4.07) so prior has modest influence

**Expected Posterior:** Mean ≈ 7-8, SD ≈ 4, similar to frequentist pooled estimate

#### Prior on theta_i (individual school effects):
No explicit prior needed - determined by hierarchical structure `theta_i ~ Normal(mu, tau)`

**Expected Posterior:** Strong shrinkage toward mu (due to tau ≈ 0 and large sigma_i)

### 7.3 Alternative Models Considered

#### Model A: Complete Pooling (Fixed Effect)
```
y_i ~ Normal(mu, sigma_i)
mu ~ Normal(0, 20)
```

**Pros:**
- Simplest model
- Data strongly support this
- Easiest interpretation

**Cons:**
- Ignores hierarchical structure
- Can't estimate between-school variance
- Less conservative

**Recommendation:** Suitable for sensitivity analysis but less flexible than hierarchical model

#### Model B: No Pooling (Independent Effects)
```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ Normal(0, 20)    [Independent priors]
```

**Pros:**
- Maximum flexibility
- No pooling assumptions

**Cons:**
- Ignores exchangeability of schools
- Wide posterior intervals (no information sharing)
- Data don't require this flexibility

**Recommendation:** Not recommended - overly conservative given evidence for homogeneity

#### Model C: Mixture Model (Latent Subgroups)
```
y_i ~ Normal(theta_i, sigma_i)
theta_i ~ w * Normal(mu_1, tau_1) + (1-w) * Normal(mu_2, tau_2)
```

**Pros:**
- Could capture bimodality if present
- Tests subgroup hypothesis explicitly

**Cons:**
- No theoretical justification
- Overparameterized for n=8
- Q test shows no evidence for subgroups
- Risk of overfitting

**Recommendation:** Not recommended - insufficient evidence for subgroup structure

### 7.4 Expected Results

**Posterior Predictions:**

| Parameter | Prior | Expected Posterior | Interpretation |
|-----------|-------|-------------------|----------------|
| mu | N(0,20) | N(7.7, 4.1) | Grand mean effect ≈ 8 |
| tau | HC(0,5) | Concentrated near 0, wide tail | Little between-school variation |
| theta_1 | N(mu,tau) | ≈ N(7.7, 4.5) | Shrunk from 28 toward 7.7 |
| theta_2 | N(mu,tau) | ≈ N(7.7, 4.2) | Minimal shrinkage (already close) |
| theta_5 | N(mu,tau) | ≈ N(7.7, 4.0) | Shrunk from -1 toward 7.7 |

**Shrinkage Pattern:**
- High precision schools (5, 7) will shrink less but still substantially
- Low precision schools (1, 3, 8) will shrink more toward pooled mean
- Expected shrinkage: 70-90% toward pooled mean for most schools

**Model Comparison:**
If comparing models via LOO-CV or WAIC, hierarchical model may not substantially outperform complete pooling model due to tau ≈ 0.

### 7.5 Implementation Notes

**Stan/PyMC Considerations:**
- Use non-centered parameterization for better sampling:
  ```
  theta_i = mu + tau * theta_raw_i
  theta_raw_i ~ Normal(0, 1)
  ```
- Monitor effective sample size for tau (boundary parameter)
- Check for divergent transitions (none expected for this simple model)
- Posterior predictive checks: simulate new schools, check coverage

**Diagnostics:**
- R-hat < 1.01 for all parameters
- Effective sample size > 400 for stable estimates
- Visual posterior checks against prior expectations
- Shrinkage plot: show theta_i posteriors vs observed y_i

---

## 8. Limitations and Caveats

### 8.1 Small Sample Size
- **n = 8 schools** provides limited power to detect heterogeneity
- Cochran's Q test has low power with small n
- tau² estimates are imprecise (wide confidence intervals)
- **Implication:** Failure to reject homogeneity doesn't prove homogeneity, but data provide no evidence against it

### 8.2 Known vs Estimated Standard Errors
- This dataset assumes sigma_i are **known** (not estimated)
- In practice, standard errors are often estimated, adding uncertainty
- With estimated sigma_i, would need to model measurement error uncertainty
- **Implication:** Results are conditional on sigma_i being correct

### 8.3 Selection Bias
- No information on how schools were selected
- Could be convenience sample, selected on outcomes, or random sample
- Selection bias could affect generalizability but not internal validity
- **Implication:** Caution in generalizing beyond these 8 schools

### 8.4 Publication Bias
- No evidence from funnel plot or Egger's test
- But n=8 provides limited power to detect bias
- Classic problem: are there unpublished studies with null results?
- **Implication:** Cannot rule out publication bias, but no positive evidence

### 8.5 Contextual Interpretation
- Analysis is purely statistical, no domain knowledge incorporated
- Don't know: What was the treatment? What do effects represent?
- Effect size of 8 could be large, small, or trivial depending on context
- **Implication:** Statistical significance ≠ practical significance

### 8.6 Boundary Estimate
- tau² = 0 is at parameter boundary (true tau² ≥ 0)
- Point estimate of 0 doesn't mean true tau² is exactly 0
- Could be small positive value that data lack power to detect
- **Implication:** Bayesian credible interval for tau will extend above 0

---

## 9. Practical Significance

### 9.1 Effect Size Interpretation

**Pooled estimate:** 7.69 ± 4.07 (95% CI: [-0.30, 15.67])

**Key Questions:**
1. What is the effect measured in? (test scores, percentage points, standardized units?)
2. What is a minimally important difference in this context?
3. Is the treatment cost-effective at this effect size?

**Without domain context, we can say:**
- Effect is likely positive (95% CI barely includes 0)
- Considerable uncertainty remains (CI width = 16 units)
- Uncertainty is primarily due to measurement error, not between-school variation

### 9.2 Decision-Making Implications

**If this were informing policy:**

1. **Individual school decisions:** With strong pooling, all schools estimated to have similar true effects (~7.7). School 1's high observed value (28) should not be treated as evidence that School 1 is "special."

2. **Future predictions:** Best prediction for a new school's effect is the pooled estimate (7.7 ± 4.1), not any individual school's observed value.

3. **Sample size planning:** If planning new studies, account for large measurement error (sigma ≈ 12). Would need sigma < 5 to distinguish effects differing by 10 units.

4. **Intervention targeting:** No evidence for heterogeneous treatment effects, so no basis for targeting intervention to specific types of schools.

---

## 10. Summary and Conclusions

### 10.1 Key Findings Recap

1. **Homogeneity:** Strong evidence that all schools share a common true effect
   - Cochran's Q: p = 0.696
   - I² = 0%
   - tau² = 0

2. **Measurement uncertainty dominates:** Average sigma (12.5) > SD of effects (10.4)
   - Variance ratio = 0.66
   - All variation attributable to sampling error

3. **Pooled estimate:** 7.69 ± 4.07
   - Robust to influence of any single school
   - All schools consistent with this estimate given their measurement error

4. **No outliers or subgroups:** All observations within expected range
   - No school exceeds |z| > 2
   - No evidence of distinct clusters

5. **Model recommendation:** Bayesian hierarchical model
   - Appropriate structure for exchangeable units
   - Will naturally adapt to favor low tau
   - More conservative than complete pooling

### 10.2 What Makes This Dataset Interesting

The Eight Schools problem is pedagogically valuable because it demonstrates:

- **Apparent heterogeneity can be spurious:** Effects range from -3 to 28 (31 unit span), yet no statistical evidence of true heterogeneity
- **Importance of measurement error:** Without accounting for sigma, data look heterogeneous; with sigma, they look homogeneous
- **Boundary estimation:** tau² at boundary (0) creates inferential challenges
- **Shrinkage estimation:** Individual estimates should be substantially shrunk toward pooled mean
- **Model comparison:** Demonstrates when hierarchical and complete pooling models converge

### 10.3 Final Recommendation

**Proceed with Bayesian hierarchical model:**
- Prior: tau ~ Half-Cauchy(0, 5), mu ~ Normal(0, 20)
- Expect posterior: tau ≈ 0-3, mu ≈ 7-8
- Interpret: Strong shrinkage toward pooled mean
- Report: Full posterior distributions, shrinkage plot, posterior predictive checks

**Alternative for simplicity:** Complete pooling model
- If computation/interpretation simplicity valued over model flexibility
- Sensitivity analysis: compare to hierarchical model results
- Expect: Nearly identical point estimates and intervals

**Do NOT use:** No pooling or mixture models
- No evidence justifies additional complexity
- Risk of overfitting and misleading conclusions

---

## 11. References and Reproducibility

### 11.1 Analysis Scripts

All analysis code is available in `/workspace/eda/code/`:

1. **01_initial_exploration.py**
   - Data loading and validation
   - Summary statistics
   - Heterogeneity tests (Q, I², tau²)
   - Outlier detection
   - Leave-one-out analysis

2. **02_visualizations.py**
   - Forest plot
   - Distribution analysis (4-panel)
   - Effect-uncertainty relationship
   - Precision analysis (funnel plot, residuals)
   - School profiles (bubble plot)
   - Heterogeneity diagnostics (4-panel)

3. **03_hypothesis_testing.py**
   - Complete pooling tests
   - No pooling analysis
   - Partial pooling (empirical Bayes)
   - Subgroup structure tests
   - Effect-uncertainty relationship tests

### 11.2 Visualizations

All plots are saved as high-resolution PNG files (300 DPI) in `/workspace/eda/visualizations/`:

| Filename | Description | Key Insights |
|----------|-------------|--------------|
| `forest_plot.png` | Classic meta-analysis forest plot | All CIs overlap, consistent with homogeneity |
| `distribution_analysis.png` | 4-panel distribution overview | Effects approximately normal, sigma relatively uniform |
| `effect_vs_uncertainty.png` | Scatter plot with correlation | No relationship (r=0.21, p=0.61) |
| `precision_analysis.png` | Funnel plot + standardized residuals | No outliers, symmetric funnel |
| `school_profiles.png` | Bubble plot with precision weighting | School 5 high precision/low effect, School 1 low precision/high effect |
| `heterogeneity_diagnostics.png` | 4-panel diagnostic suite | All diagnostics consistent with homogeneity |

### 11.3 Data Files

- **Input:** `/workspace/data/data.csv` (original dataset)
- **Output:** `/workspace/eda/code/data_with_diagnostics.csv` (augmented with z-scores and precision)

### 11.4 Reproducibility

**Software:**
- Python 3.x
- pandas, numpy, scipy, matplotlib, seaborn
- All random seeds set where applicable
- No proprietary software required

**To reproduce:**
```bash
cd /workspace/eda/code
python 01_initial_exploration.py
python 02_visualizations.py
python 03_hypothesis_testing.py
```

### 11.5 Further Reading

**On the Eight Schools Problem:**
- Rubin, D. B. (1981). Estimation in parallel randomized experiments. Journal of Educational Statistics, 6(4), 377-401.
- Gelman, A., et al. (2013). Bayesian Data Analysis (3rd ed.), Section 5.5.
- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. Bayesian Analysis, 1(3), 515-534.

**On Meta-Analysis:**
- Borenstein, M., et al. (2009). Introduction to Meta-Analysis.
- Higgins, J. P., & Thompson, S. G. (2002). Quantifying heterogeneity in a meta-analysis. Statistics in Medicine, 21(11), 1539-1558.

**On Hierarchical Models:**
- Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and Multilevel/Hierarchical Models.
- McElreath, R. (2020). Statistical Rethinking (2nd ed.), Chapter 13.

---

## Appendix A: Statistical Formulas

### A.1 Cochran's Q Statistic
```
Q = Σ w_i * (y_i - ȳ_w)²
where:
  w_i = 1 / sigma_i²  (precision weights)
  ȳ_w = Σ(w_i * y_i) / Σ(w_i)  (weighted mean)

Under H0: Q ~ χ²(k-1)
```

### A.2 I² Statistic
```
I² = max(0, 100 * (Q - df) / Q)
where df = k - 1
```

### A.3 DerSimonian-Laird Estimator
```
tau² = max(0, (Q - df) / C)
where:
  C = Σw_i - (Σw_i²) / (Σw_i)
```

### A.4 Empirical Bayes Shrinkage
```
θ_i^EB = B_i * y_i + (1 - B_i) * ȳ_w
where:
  B_i = tau² / (tau² + sigma_i²)  (shrinkage factor)
```

---

## Appendix B: Detailed School Profiles

| School | Observed Effect | SE | Precision | Weighted Mean | Residual | Z-score | EB Estimate | Shrinkage | Influence (LOO) |
|--------|----------------|----|-----------|--------------|---------|---------|-----------|-----------|-----------------
| 1 | 28.0 | 15 | 0.0044 | 7.69 | 20.31 | 1.35 | 7.69 | 100% | -1.62 |
| 2 | 8.0 | 10 | 0.0100 | 7.69 | 0.31 | 0.03 | 7.69 | 100% | -0.06 |
| 3 | -3.0 | 16 | 0.0039 | 7.69 | -10.69 | -0.67 | 7.69 | 100% | +0.74 |
| 4 | 7.0 | 11 | 0.0083 | 7.69 | -0.69 | -0.06 | 7.69 | 100% | +0.11 |
| 5 | -1.0 | 9 | 0.0123 | 7.69 | -8.69 | -0.97 | 7.69 | 100% | +2.24 |
| 6 | 1.0 | 11 | 0.0083 | 7.69 | -6.69 | -0.61 | 7.69 | 100% | +1.06 |
| 7 | 18.0 | 10 | 0.0100 | 7.69 | 10.31 | 1.03 | 7.69 | 100% | -2.05 |
| 8 | 12.0 | 18 | 0.0031 | 7.69 | 4.31 | 0.24 | 7.69 | 100% | -0.23 |

**Notes:**
- All schools shrink 100% to pooled mean because tau² = 0
- Z-scores all within ±2, no outliers
- Influence values show moderate impact of removing any single school
- Schools 5 and 7 most influential due to high precision

---

**End of Report**

*For detailed analysis process and intermediate findings, see `/workspace/eda/eda_log.md`*
