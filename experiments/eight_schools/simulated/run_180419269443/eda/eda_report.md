# Exploratory Data Analysis Report
## Meta-Analysis Dataset: Comprehensive Findings and Modeling Recommendations

**Analysis Date:** 2025-10-28
**Dataset:** `/workspace/data/data.csv`
**Analyst:** EDA Specialist Agent
**Context:** Hierarchical meta-analysis with J=8 studies

---

## Executive Summary

This EDA examined a meta-analysis dataset with 8 studies, each reporting an observed effect (y) and known standard error (sigma). The analysis reveals **very low between-study heterogeneity** (I² = 2.9%), indicating that most observed variation is due to sampling error rather than true differences between studies. The **pooled effect estimate is 11.27 (95% CI: 3.29-19.25)**, representing a positive and statistically significant effect. Pooling studies provides substantial benefit due to high within-study uncertainty. However, results show sensitivity to individual studies, particularly Study 4, warranting careful sensitivity analyses.

**Key Recommendation:** Use a hierarchical Bayesian model with partial pooling, emphasizing uncertainty quantification given the small sample size.

---

## 1. Data Characteristics

### 1.1 Dataset Overview

| Characteristic | Value |
|---------------|-------|
| Number of studies | 8 |
| Variables | study (ID), y (effect), sigma (SE) |
| Missing values | 0 |
| Data quality | Excellent - no obvious issues |

### 1.2 Descriptive Statistics

**Observed Effects (y):**
- Mean: 12.50 (unweighted), 11.27 (precision-weighted)
- Median: 11.92
- Range: [-4.88, 26.08]
- SD: 11.15
- IQR: 16.10

**Standard Errors (sigma):**
- Mean: 12.50
- Median: 11.00
- Range: [9.00, 18.00]
- SD: 3.34

**Key Observation:** Standard errors are large relative to effect sizes, indicating substantial within-study uncertainty. This makes pooling particularly valuable.

### 1.3 Study-Level Summary

| Study | Effect (y) | SE (sigma) | Precision | Weight | 95% CI |
|-------|-----------|-----------|-----------|--------|--------|
| 1 | 20.02 | 15.00 | 0.067 | 7.4% | [-9.38, 49.42] |
| 2 | 15.30 | 10.00 | 0.100 | 16.6% | [-4.30, 34.90] |
| 3 | 26.08 | 16.00 | 0.063 | 6.5% | [-5.28, 57.44] |
| 4 | 25.73 | 11.00 | 0.091 | 13.7% | [4.17, 47.29] |
| 5 | -4.88 | 9.00 | 0.111 | 20.5% | [-22.52, 12.76] |
| 6 | 6.08 | 11.00 | 0.091 | 13.7% | [-15.48, 27.64] |
| 7 | 3.17 | 10.00 | 0.100 | 16.6% | [-16.43, 22.77] |
| 8 | 8.55 | 18.00 | 0.056 | 5.1% | [-26.73, 43.83] |

**Note:** Weights are based on inverse-variance weighting (1/sigma²), normalized to sum to 100%.

---

## 2. Visual Evidence

All visualizations are saved in `/workspace/eda/visualizations/`. Below is a guide to interpreting each plot.

### 2.1 Forest Plot (`01_forest_plot.png`)

**What it shows:** Classic meta-analysis forest plot with studies sorted by effect size, showing point estimates and 95% confidence intervals.

**Key insights:**
- All confidence intervals are wide due to large standard errors
- Study 5 is the only study with a negative point estimate
- All CIs overlap with the pooled estimate (red dashed line at 11.27)
- Point sizes are inversely proportional to SE (larger = more precise)
- Studies 2, 5, 6, 7 have smaller SEs and thus more weight

**Implication:** Despite varying point estimates, all studies are compatible with a common underlying effect.

### 2.2 Effect Distribution (`02_effect_distribution.png`)

**What it shows:** Two panels - (1) histogram with KDE overlay showing distribution of observed effects, (2) Q-Q plot testing normality.

**Key insights:**
- Distribution is roughly symmetric with slight negative skew
- Mean (12.50) and median (11.92) are close
- Q-Q plot shows reasonable normality except in tails
- Study 5's negative effect creates left tail deviation
- Wide spread reflects both sampling error and potential heterogeneity

**Implication:** Normal distributional assumptions are reasonable for modeling.

### 2.3 Standard Error Distribution (`03_sigma_distribution.png`)

**What it shows:** Two panels showing distribution and box plot of standard errors.

**Key insights:**
- SEs range from 9-18 with mean of 12.5
- Fairly uniform distribution - no concentration at particular values
- No extreme outliers in precision
- Study 8 has largest SE (18), Study 5 has smallest (9)

**Implication:** Studies have similar levels of precision; no single study has exceptionally high or low quality.

### 2.4 Effect-Precision Relationship (`04_effect_precision_relationship.png`)

**What it shows:** Four panels examining relationships between effect size, standard error, and precision.

**Key insights:**
- **Panel 1 (y vs sigma):** Positive correlation (r = 0.428, p = 0.290) but not significant
- **Panel 2 (y vs precision):** Negative correlation (r = -0.428) - mirror of Panel 1
- **Panel 3 (Funnel plot):** Reasonably symmetric around pooled estimate
- **Panel 4 (Weighted contributions):** Study 5 contributes most (20.5%), despite negative effect

**Implication:** No evidence of publication bias (funnel plot symmetric) or systematic relationship between study precision and effect size.

### 2.5 Heterogeneity Diagnostics (`05_heterogeneity_diagnostics.png`)

**What it shows:** Four panels providing comprehensive heterogeneity assessment.

**Key insights:**
- **Panel 1 (Standardized effects):** All z-scores within ±2, most within ±1.5
- **Panel 2 (SE by study):** SEs vary but not dramatically (9-18 range)
- **Panel 3 (Effects with ±1 SE bands):** All error bands overlap with pooled mean
- **Panel 4 (Galbraith plot):** Points cluster around regression line (slope = pooled effect)

**Implication:** Strong evidence for homogeneity; variation consistent with sampling error alone.

### 2.6 Study-Level Details (`06_study_level_details.png`)

**What it shows:** Detailed view of each study with error bars, weights, and 95% CI of pooled effect.

**Key insights:**
- Individual study CIs are very wide (reflect large SEs)
- Pooled estimate CI (red band) is much narrower
- Weights (w) shown above each study range from 0.05 to 0.21
- Demonstrates benefit of meta-analysis: pooling reduces uncertainty

**Implication:** No single study is precise enough alone; pooling is essential for inference.

### 2.7 Shrinkage Analysis (`07_shrinkage_analysis.png`)

**What it shows:** Four panels illustrating shrinkage from observed to pooled estimates.

**Key insights:**
- **Panel 1:** Arrows show shrinkage direction - all studies pulled strongly toward mean
- **Panel 2:** Three pooling strategies compared - partial pooling intermediate
- **Panel 3:** Shrinkage factors all < 0.05, indicating >95% shrinkage
- **Panel 4:** SE reduction from shrinkage ranges 2.5-2.8%

**Implication:** Strong shrinkage indicates high trust in pooled estimate; within-study variance dominates between-study variance by factor of ~40.

### 2.8 Model Comparison (`08_model_comparison.png`)

**What it shows:** Four panels comparing different modeling approaches.

**Key insights:**
- **Panel 1:** Partial pooling narrows CIs substantially compared to no pooling
- **Panel 2:** Bootstrap distribution of pooled estimate is symmetric and stable
- **Panel 3:** Prediction interval (17.8 wide) much wider than CI (16.0 wide)
- **Panel 4:** Variance decomposition shows 97.5% within-study, 2.5% between-study

**Implication:** Partial pooling provides optimal bias-variance tradeoff; prediction intervals appropriately account for future study uncertainty.

---

## 3. Heterogeneity Assessment

### 3.1 Statistical Tests

| Test | Statistic | Result | Interpretation |
|------|-----------|--------|----------------|
| Cochran's Q | 7.21 (df=7) | p = 0.407 | Cannot reject homogeneity |
| I² statistic | 2.9% | Low | 97.1% of variation is sampling error |
| Tau² (DL estimator) | 4.08 | Small | Between-study variance minimal |
| Tau (between-study SD) | 2.02 | Small | Much smaller than typical SE (12.5) |

### 3.2 Interpretation

**Conclusion:** The data shows **very low heterogeneity**. This is somewhat surprising given the wide range of point estimates (-4.88 to 26.08), but the large standard errors explain most of this variation. Only 2.9% of observed variation is attributed to true between-study differences.

**Practical significance:**
- Studies are measuring similar underlying effects
- Pooling is appropriate and beneficial
- Common effect model is statistically justified
- Random effects model adds little, but provides more conservative inference

---

## 4. Outlier and Influence Analysis

### 4.1 Outlier Detection

**Z-score analysis:** No studies have |z-score| > 2 based on standardized residuals.

**Study 5 investigation:** Despite being the only negative effect, Study 5 is not a statistical outlier:
- Effect: -4.88, SE: 9.00
- Z-score: 1.67 (< 2.0 threshold)
- 95% CI: [-22.52, 12.76] overlaps with pooled estimate
- Standardized effect: -0.54 standard errors from zero

**Conclusion:** Study 5 appears to be sampling variation, not a true outlier.

### 4.2 Influence Analysis (Leave-One-Out)

| Study Removed | Pooled Estimate | Change | % Change |
|---------------|----------------|--------|----------|
| None (full) | 11.27 | - | - |
| Study 1 | 9.23 | -2.04 | -18.1% |
| Study 2 | 8.98 | -2.29 | -20.3% |
| Study 3 | 8.91 | -2.36 | -20.9% |
| **Study 4** | **7.53** | **-3.74** | **-33.2%** |
| Study 5 | 13.86 | +2.59 | +23.0% |
| Study 6 | 10.65 | -0.62 | -5.5% |
| Study 7 | 11.39 | +0.12 | +1.0% |
| Study 8 | 10.10 | -1.17 | -10.3% |

**Key findings:**
- **Study 4 is most influential:** Removing it decreases estimate by 33.2%
- Study 5 is second most influential (23.0% increase when removed)
- Studies 6 and 7 have minimal influence
- No single study dominates, but results are sensitive given small sample

**Implication:** Sensitivity analyses removing Studies 4 and 5 are essential for robustness assessment.

---

## 5. Publication Bias Assessment

### 5.1 Egger's Test

- Regression: Standardized effect = 2.18 - 14.52 × precision
- Intercept: 2.18 (SE: 17.38)
- p-value: 0.435

**Conclusion:** No significant asymmetry detected (p > 0.05). No evidence of publication bias.

### 5.2 Funnel Plot Symmetry

Visual inspection of funnel plot (`04_effect_precision_relationship.png`, Panel 3) shows:
- Studies distributed symmetrically around pooled estimate
- No concentration of small, positive-effect studies
- Study 5 (negative effect) present, suggesting no selective reporting

### 5.3 Effect-Precision Correlation

- Correlation between |effect| and SE: r = 0.354, p = 0.390
- No significant relationship between effect magnitude and precision

**Overall conclusion:** Data show no evidence of publication bias or small-study effects.

---

## 6. Shrinkage and Pooling Analysis

### 6.1 Pooling Strategies Compared

| Study | No Pooling (y) | Partial Pooling | Complete Pooling | Shrinkage Factor |
|-------|---------------|-----------------|------------------|------------------|
| 1 | 20.02 | 11.42 | 11.27 | 0.018 |
| 2 | 15.30 | 11.42 | 11.27 | 0.039 |
| 3 | 26.08 | 11.50 | 11.27 | 0.016 |
| 4 | 25.73 | 11.74 | 11.27 | 0.033 |
| 5 | -4.88 | 10.49 | 11.27 | 0.048 |
| 6 | 6.08 | 11.10 | 11.27 | 0.033 |
| 7 | 3.17 | 10.95 | 11.27 | 0.039 |
| 8 | 8.55 | 11.23 | 11.27 | 0.012 |

**Key insights:**
- Shrinkage factors all < 0.05, indicating >95% weight on pooled mean
- Partial pooling estimates very close to complete pooling
- Average shrinkage amount: 8.99 units
- Maximum shrinkage: 15.38 (Study 5, from -4.88 to 10.49)

**Interpretation:** Within-study variance (sigma²) is so large compared to between-study variance (tau²) that the data strongly favor pooling. Individual study estimates are unreliable in isolation.

### 6.2 Effective Sample Size

- Number of studies: 8
- Effective number: 6.82
- Efficiency: 85.2%

**Interpretation:** Despite varying precisions, the meta-analysis efficiently uses information from most studies. Weight distribution is reasonably balanced (no single study dominates completely).

---

## 7. Model Selection and Comparison

### 7.1 Model Comparison (AIC)

| Model | Log-Likelihood | Parameters | AIC | Rank |
|-------|---------------|------------|-----|------|
| Common effect | -30.93 | 1 | 63.85 | 1 (best) |
| Random effects | -30.91 | 2 | 65.82 | 2 |
| No pooling | -27.32 | 8 | 70.64 | 3 |

**Result:** Common effect model preferred by AIC, but random effects very close (difference < 2). No pooling model clearly worst.

### 7.2 Model Recommendations

**Three viable modeling approaches:**

#### Option 1: Common Effect (Fixed Effect) Model
```
y_i ~ N(mu, sigma_i²)
mu ~ N(0, 50)  [weakly informative prior]
```

**Pros:**
- Most parsimonious (supported by AIC)
- Justified by low heterogeneity (I² = 2.9%)
- Q-test cannot reject homogeneity

**Cons:**
- Assumes all studies measure identical effect
- Overly optimistic CI if heterogeneity present
- Less conservative

**Use when:** You believe studies are estimating same parameter and want narrowest CI.

#### Option 2: Random Effects (Hierarchical) Model ⭐ **RECOMMENDED**
```
y_i ~ N(theta_i, sigma_i²)
theta_i ~ N(mu, tau²)
mu ~ N(0, 50)
tau ~ Half-Normal(0, 10)
```

**Pros:**
- Accounts for potential between-study variation
- More conservative inference (wider CI)
- Appropriate for small sample (J=8)
- Provides prediction intervals for future studies
- Standard in meta-analysis field

**Cons:**
- Tau² difficult to estimate precisely with J=8
- Slightly wider CI than common effect

**Use when:** You want robust inference that accounts for uncertainty in heterogeneity estimation (most cases).

#### Option 3: Bayesian Hierarchical with Sensitivity Analysis
```
y_i ~ N(theta_i, sigma_i²)
theta_i ~ N(mu, tau²)
mu ~ N(0, 50)
tau ~ Half-Cauchy(0, 5)  [allows heavier tails]
```

**Pros:**
- Full uncertainty quantification
- Can incorporate informative priors if available
- Provides posterior distributions, not just point estimates
- Easy to conduct sensitivity analyses (varying priors)

**Cons:**
- Requires MCMC (Stan, PyMC, etc.)
- More complex to implement

**Use when:** You have computational resources and want most complete uncertainty characterization.

### 7.3 Final Recommendation

**Use a Bayesian hierarchical random effects model (Option 3) with:**
- Weakly informative priors: mu ~ N(0, 50), tau ~ Half-Normal(0, 10)
- Report both common effect and random effects estimates for comparison
- Conduct sensitivity analyses removing Studies 4 and 5
- Report prediction intervals alongside confidence intervals
- Emphasize uncertainty given small sample size

---

## 8. Prior Elicitation Guidance

Based on the observed data, suggested priors for Bayesian modeling:

### 8.1 Mean Effect (mu)

**Option 1 (Weakly Informative - RECOMMENDED):**
```
mu ~ Normal(0, 50)
```
- Covers wide range [-100, 100] with 95% probability
- Allows data to dominate
- Appropriate when no strong prior knowledge

**Option 2 (Data-Driven):**
```
mu ~ Normal(11, 22)
```
- Centered on weighted mean
- SD = 2 × observed SD
- Use only for sensitivity analysis, not primary

**Option 3 (Skeptical):**
```
mu ~ Normal(0, 20)
```
- Shrinks toward null hypothesis
- Use for robustness check

### 8.2 Between-Study SD (tau)

**Option 1 (Weakly Informative - RECOMMENDED):**
```
tau ~ Half-Normal(0, 10)
```
- Allows for moderate heterogeneity
- Constrains to reasonable values
- Standard choice in meta-analysis

**Option 2 (Heavy-Tailed):**
```
tau ~ Half-Cauchy(0, 5)
```
- Allows for more extreme heterogeneity
- Robust to misspecification
- Use if concerned about heavy tails

**Option 3 (Uniform):**
```
tau ~ Uniform(0, 20)
```
- Non-informative
- May lead to boundary issues with small J

### 8.3 Justification

Given:
- Estimated tau = 2.02
- Median sigma = 11.0
- Ratio tau/sigma = 0.184

The suggested priors are weakly informative, allowing the data to dominate while preventing unreasonable parameter values.

---

## 9. Data Quality and Limitations

### 9.1 Strengths

1. **Complete data:** No missing values
2. **Known SEs:** Standard errors reported, not estimated
3. **No obvious errors:** All values plausible
4. **Balanced weights:** No single study dominates completely (max weight 20.5%)
5. **No publication bias:** Egger's test and funnel plot show no asymmetry

### 9.2 Limitations

1. **Small sample size (J=8):**
   - Heterogeneity parameters difficult to estimate precisely
   - Results sensitive to individual studies (Study 4 especially)
   - Limited power to detect heterogeneity even if present
   - Wide confidence intervals on pooled estimate

2. **Large within-study uncertainty:**
   - Individual study SEs range from 9-18
   - Most individual CIs very wide
   - Limits conclusions about any single study

3. **Study 5 unexplained:**
   - Only negative effect
   - No covariate data to investigate why
   - Could be sampling variation or unobserved moderator

4. **No covariate information:**
   - Cannot conduct meta-regression
   - Cannot explain heterogeneity if present
   - Cannot identify subgroups

5. **Assumption of known SEs:**
   - Analysis treats sigma as fixed/known
   - If SEs were estimated, additional uncertainty not accounted for

### 9.3 Data Quality Concerns

**None identified.** The data appear clean, complete, and suitable for meta-analysis.

---

## 10. Modeling Recommendations

### 10.1 Primary Analysis

**Recommended Model:** Bayesian hierarchical random effects

**Implementation (pseudocode):**
```
for i in 1:8:
    y[i] ~ Normal(theta[i], sigma[i])  # Likelihood
    theta[i] ~ Normal(mu, tau)          # Study-specific effects

mu ~ Normal(0, 50)                      # Prior on mean
tau ~ HalfNormal(0, 10)                 # Prior on heterogeneity
```

**Reporting:**
- Pooled effect estimate (mu) with 95% credible interval
- Between-study SD (tau) with 95% credible interval
- I² statistic
- 95% prediction interval for a new study
- Forest plot with posterior means and intervals

### 10.2 Sensitivity Analyses

**Required:**

1. **Prior sensitivity:**
   - Vary prior on tau: Half-Cauchy(0, 5), Uniform(0, 20)
   - Vary prior on mu: N(0, 20), N(0, 100)
   - Check if conclusions change

2. **Influence analysis:**
   - Refit model excluding Study 4
   - Refit model excluding Study 5
   - Refit model excluding both Studies 4 and 5
   - Report range of estimates

3. **Model comparison:**
   - Fit common effect model
   - Compare to random effects via WAIC or LOO-CV
   - Report both estimates

**Optional:**

4. **Outlier robustness:**
   - Use t-distributed errors instead of normal: y[i] ~ Student_t(nu, theta[i], sigma[i])
   - Check if results change

5. **Publication bias:**
   - Fit selection models or PET-PEESE regression
   - Though Egger's test shows no bias, good to confirm

### 10.3 Model Classes to Consider

Based on the data characteristics, three model classes are appropriate:

#### 1. Bayesian Hierarchical Models ⭐ **BEST FIT**
- **Why:** Properly handles small sample uncertainty, provides full posterior distributions
- **Tools:** Stan, PyMC3, JAGS, brms (R)
- **Advantages:** Full uncertainty quantification, prediction intervals, flexible priors

#### 2. Meta-Analysis with Inverse-Variance Weighting
- **Why:** Standard frequentist approach, simple interpretation
- **Tools:** metafor (R), meta (R), statsmodels.meta (Python)
- **Advantages:** Well-established, easy to implement, standard reporting

#### 3. Empirical Bayes / REML
- **Why:** Compromise between fixed and random effects
- **Tools:** lme4 (R), nlme (R), statsmodels (Python)
- **Advantages:** Automatic shrinkage, no prior specification needed

**NOT recommended:**
- ❌ No pooling (each study separate): Ignores valuable information sharing
- ❌ Complete pooling with no uncertainty on mu: Overconfident
- ❌ Fixed effects model as only analysis: Too optimistic given J=8

---

## 11. Key Findings Summary

### 11.1 Robust Findings (High Confidence)

1. **Low heterogeneity:** I² = 2.9%, Q-test p = 0.407
2. **Positive pooled effect:** 11.27 (95% CI: 3.29-19.25)
3. **Strong shrinkage:** >95% weight on pooled estimate
4. **No publication bias:** Egger's test p = 0.435, funnel plot symmetric
5. **Pooling is beneficial:** Reduces uncertainty substantially
6. **All CIs overlap:** No strong outliers
7. **High precision efficiency:** 85.2% effective sample size

### 11.2 Tentative Findings (Lower Confidence)

1. **Exact pooled value:** Sensitive to Study 4 (±33% if removed)
2. **Study 5 interpretation:** Unclear if sampling variation or true difference
3. **Tau² estimate:** Difficult to estimate precisely with J=8
4. **Prediction interval width:** Uncertain due to imprecise tau estimate
5. **Common vs random effects:** Both models fit similarly (ΔAIC = 1.97)

### 11.3 Areas Requiring Further Investigation

1. **Study 4 influence:** Why does it have such high leverage? Investigate study quality
2. **Study 5 negativity:** Is there a moderator explaining the negative effect?
3. **Covariates:** Are there study-level characteristics that explain variation?
4. **Future studies:** Prediction interval [2.36, 20.18] is wide - expect high variability
5. **Subgroups:** Might there be hidden subgroups in the data?

---

## 12. Practical Implications

### 12.1 For Decision Makers

1. **Effect is positive:** Pooled estimate of 11.27 is significantly greater than zero
2. **Uncertainty is substantial:** 95% CI is wide [3.29, 19.25]
3. **Future studies unpredictable:** Prediction interval even wider [2.36, 20.18]
4. **More data needed:** With only J=8 studies, precision is limited

### 12.2 For Researchers

1. **Pooling is essential:** Individual studies too imprecise alone
2. **Bayesian approach recommended:** Better handles small-sample uncertainty
3. **Sensitivity analysis critical:** Results depend on specific studies included
4. **Report both estimates:** Common effect and random effects for completeness
5. **Covariates valuable:** Meta-regression could explain Study 5

### 12.3 For Future Meta-Analyses

1. **Need more studies:** J=8 is minimal for reliable heterogeneity estimation
2. **Collect covariates:** Study design, population, setting, etc.
3. **Report raw data:** Allow future IPD meta-analysis
4. **Pre-register:** Specify analysis plan before data collection

---

## 13. Conclusion

This comprehensive EDA of a meta-analysis dataset with 8 studies reveals a **positive pooled effect of 11.27 (95% CI: 3.29-19.25)** with **very low between-study heterogeneity (I² = 2.9%)**. The large within-study standard errors (9-18) make individual studies imprecise, but pooling provides substantial benefit through shrinkage toward the common mean.

**Key takeaways:**
- Pooling is highly beneficial and statistically justified
- Both common effect and random effects models are defensible
- Results are sensitive to individual studies (especially Study 4)
- No evidence of publication bias or data quality issues
- Bayesian hierarchical modeling is recommended for optimal uncertainty quantification

**Recommended approach:** Fit a Bayesian hierarchical random effects model with weakly informative priors, conduct sensitivity analyses removing influential studies, and report both confidence and prediction intervals. Emphasize uncertainty given the small sample size and large within-study variation.

---

## 14. References and Resources

### Code Files
All analysis code is reproducible and located in `/workspace/eda/code/`:
- `01_initial_exploration.py` - Basic statistics and heterogeneity tests
- `02_visualizations.py` - Main visualization suite
- `03_hypothesis_testing.py` - Hypothesis testing framework
- `04_advanced_diagnostics.py` - Shrinkage and model comparison
- `05_shrinkage_visualization.py` - Shrinkage-specific plots

### Visualization Files
All plots saved in `/workspace/eda/visualizations/`:
- `01_forest_plot.png`
- `02_effect_distribution.png`
- `03_sigma_distribution.png`
- `04_effect_precision_relationship.png`
- `05_heterogeneity_diagnostics.png`
- `06_study_level_details.png`
- `07_shrinkage_analysis.png`
- `08_model_comparison.png`

### Detailed Log
Complete exploration process documented in `/workspace/eda/eda_log.md`

---

**Report prepared by:** EDA Specialist Agent
**Date:** 2025-10-28
**Total visualizations:** 8 comprehensive figures
**Total hypotheses tested:** 5 competing models
**Analysis depth:** 3 rounds of iterative exploration
