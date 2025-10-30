# EDA Log: Eight Schools Dataset
## Detailed Exploration Process and Intermediate Findings

**Date:** 2025-10-28
**Analyst:** EDA Specialist
**Dataset:** Eight Schools Hierarchical Meta-Analysis

---

## Round 1: Initial Data Understanding

### Data Structure Discovery
- **8 schools** with paired observations (effect, standard error)
- **Variables:**
  - `school`: Identifier (1-8)
  - `y`: Observed treatment effect
  - `sigma`: Known measurement standard error
- **Data quality:** No missing values, no duplicates, complete dataset

### Initial Observations

**Observed Effects (y):**
- Range: [-3, 28]
- Mean: 8.75
- Median: 7.5
- Std Dev: 10.44
- Strong positive skew (0.662)
- School 1 has notably high effect (28)
- School 3 has negative effect (-3)

**Standard Errors (sigma):**
- Range: [9, 18]
- Mean: 12.5
- Median: 11.0
- Std Dev: 3.34
- Relatively homogeneous across schools
- Coefficient of variation: 0.267 (moderate variability)

**Key Finding:** Large measurement uncertainty (sigma) relative to observed effects. Average sigma (12.5) is larger than the standard deviation of effects (10.44), suggesting sampling error may dominate true heterogeneity.

### Initial Hypothesis
Given the high measurement uncertainty, I hypothesized that:
1. Much of the observed variation might be due to sampling error
2. There may not be strong evidence for heterogeneity
3. Complete or near-complete pooling might be appropriate

---

## Round 2: Distribution Analysis

### Normality Assessment
- **Shapiro-Wilk test for y:** W=0.937, p=0.583 (consistent with normality)
- **Shapiro-Wilk test for sigma:** W=0.866, p=0.138 (consistent with normality)
- **Q-Q plot:** Effects follow approximately normal distribution with slight deviation in tails

**Visualization:** `distribution_analysis.png`
- Four-panel view showing histograms, box plots, and Q-Q plot
- Effects show right skew but not extreme
- Standard errors relatively uniform

### Correlation Analysis
- **Pearson correlation (y, sigma):** r = 0.213, p = 0.612
- **Spearman correlation (y, sigma):** rho = 0.108, p = 0.798
- **Interpretation:** No significant relationship between effect size and uncertainty

**Question raised:** Is School 1 (y=28, sigma=15) an outlier, or just an extreme observation consistent with its measurement error?

**Visualization:** `effect_vs_uncertainty.png`
- Scatter plot shows no clear trend
- School 1 appears high but within plausible range given its large SE

---

## Round 3: Heterogeneity Assessment

### Classical Meta-Analysis Tests

**Cochran's Q Test:**
- Q = 4.71, df = 7, p = 0.696
- **Result:** FAIL to reject homogeneity
- **Interpretation:** No statistical evidence that schools differ in their true effects

**I² Statistic:**
- I² = 0.0%
- **Interpretation:** All observed variation attributable to sampling error
- Categories: <25% = low, 25-50% = moderate, 50-75% = substantial, >75% = considerable
- Our 0% falls in "low heterogeneity" category

**Between-Study Variance (DerSimonian-Laird):**
- tau² = 0.00
- tau = 0.00
- **Interpretation:** Point estimate suggests no between-study variance
- This is a boundary estimate (cannot be negative)

**Visualization:** `heterogeneity_diagnostics.png`
- Four-panel diagnostic plot showing:
  1. Observed vs expected under homogeneity (all points near identity line)
  2. Contribution to Q statistic (all schools contribute minimally)
  3. Precision-effect relationship (no pattern)
  4. Leave-one-out influence (Schools 5 and 7 most influential, but changes modest)

### Key Finding
**All 8 schools fall within their expected 95% confidence intervals** assuming a common true effect of 7.69. This is strong evidence for homogeneity.

---

## Round 4: Outlier and Influence Analysis

### Standardized Residuals
Using weighted mean (7.69) as reference:
- School 1: z = 1.35 (largest positive)
- School 7: z = 1.03
- School 5: z = -0.97 (largest negative)
- School 3: z = -0.67
- **No schools exceed |z| > 2 threshold**

**Visualization:** `precision_analysis.png`
- Funnel plot shows reasonable symmetry
- Standardized residuals bar chart confirms no extreme outliers

### Leave-One-Out Analysis
Changes in weighted mean when removing each school:
- Without School 5: +2.24 (most influential, pulling mean down)
- Without School 7: -2.05 (pulling mean up)
- Without School 1: -1.62 (pulling mean up)
- Other schools: <1.5 change

**Interpretation:**
- Some schools are moderately influential but none are dominating
- School 1, despite large effect, has large SE so gets down-weighted
- School 5 has high precision and negative effect, pulling mean down

**Visualization:** `school_profiles.png`
- Bubble plot where size = precision
- Shows School 5 has high precision (large bubble) at low effect
- School 1 has low precision (small bubble) at high effect

---

## Round 5: Hypothesis Testing

### Tested Hypotheses

#### H1: Complete Pooling (All schools share same true effect)
**Evidence FOR:**
- Cochran's Q test: p = 0.696 (strong failure to reject)
- I² = 0%
- All observations within expected range
- Variance ratio = 0.66 (observed variance < sampling variance)

**Evidence AGAINST:**
- None substantial

**Conclusion:** **DATA STRONGLY SUPPORT complete pooling**

#### H2: No Pooling (All schools completely different)
**Evidence FOR:**
- Individual CIs are wide (reflecting uncertainty)
- Some schools appear different (School 1 vs School 3)

**Evidence AGAINST:**
- Very wide individual CIs provide little information
- Variance ratio < 1 (observed variation less than expected by chance)
- Q test shows variation consistent with sampling error

**Conclusion:** **DATA DO NOT REQUIRE independent estimates**

#### H3: Partial Pooling (Hierarchical model)
**Analysis:**
- Empirical Bayes with tau² = 0 shrinks all estimates 100% to pooled mean
- This is mathematically equivalent to complete pooling
- However, hierarchical model framework is still philosophically appropriate

**Conclusion:** **Hierarchical model COLLAPSES to complete pooling given this data**

#### H4: Subgroup Structure
**Analysis:**
- Median split creates "low" (n=4) and "high" (n=4) groups
- Mann-Whitney U test: p = 0.029 (borderline significant)
- Gap analysis: largest gap = 10.0 (between 18 and 28)
- Gap ratio = 2.50 suggests possible bimodality

**HOWEVER:**
- This is data dredging (post-hoc split)
- No theoretical basis for two groups
- Cochran's Q already tested and failed to find heterogeneity
- Likely spurious finding from small sample

**Conclusion:** **NO STRONG EVIDENCE for distinct subgroups** (tentative finding)

#### H5: Effect-Uncertainty Relationship
**Analysis:**
- Pearson r = 0.213, p = 0.612 (not significant)
- No evidence of "small study effects" or publication bias
- Linear regression R² = 0.045 (explains <5% of variance)

**Conclusion:** **NO RELATIONSHIP between effect size and uncertainty**

---

## Round 6: Model Implications

### Pooling Comparison

| Model Type | Point Estimate | Uncertainty | Implications |
|-----------|---------------|-------------|--------------|
| **Complete Pooling** | 7.69 | SE = 4.07 | Assumes all schools identical |
| **No Pooling** | Individual y's | Large (9-18) | Ignores shared information |
| **Partial Pooling** | EB estimates | Intermediate | With tau²=0, equals complete pooling |

### Bayesian Modeling Considerations

**Prior on tau (between-school SD):**
- Data suggest tau ≈ 0
- But should use weakly informative prior to allow for heterogeneity
- Options:
  - Half-Cauchy(0, 5) or Half-Cauchy(0, 10) [Gelman recommendation]
  - Half-Normal(0, 10)
  - Uniform(0, 50)
- Prior will matter less here since data are informative about tau ≈ 0

**Prior on mu (grand mean):**
- Weakly informative: Normal(0, 20) or Normal(10, 10)
- Data are reasonably informative (pooled SE = 4.07)

**Prior on school effects (theta_i):**
- In hierarchical model: theta_i ~ Normal(mu, tau)
- As tau → 0, all theta_i → mu
- Model will naturally induce strong shrinkage

**Expected Posterior:**
- Posterior for tau will be concentrated near 0
- Posterior for mu will be similar to frequentist pooled estimate
- Posterior for individual theta_i will be strongly shrunk toward mu
- Wide credible intervals for tau (small n, boundary issue)

---

## Unexpected Findings

1. **Variance ratio < 1:** Observed variance (109) is actually LESS than average sampling variance (166). This is unusual and strongly supports homogeneity.

2. **Perfect CI coverage:** All 8 schools fall within their expected 95% CIs under homogeneity. With 8 independent tests, we'd expect ~0-1 to be outside by chance. Having all 8 inside is strong evidence.

3. **Boundary estimate:** tau² = 0 is at the boundary of parameter space. This makes inference tricky - is true tau exactly 0, or just very small? Bayesian approach handles this better than frequentist.

4. **Subgroup finding:** The Mann-Whitney test for median split was significant (p=0.029), but this contradicts the Q test. Likely a false positive from multiple comparisons or small sample variability.

---

## Robust vs Tentative Findings

### ROBUST Findings (High Confidence)
✓ No evidence of heterogeneity (Q test, I², tau²)
✓ Observed variation consistent with sampling error alone
✓ No relationship between effect size and uncertainty
✓ No extreme outliers or influential observations
✓ Data support complete or near-complete pooling

### TENTATIVE Findings (Lower Confidence)
? Possible bimodality (gap analysis) - likely spurious
? Subgroup structure (median split) - post-hoc, contradicts Q test
? School 1 as "special" - within expected range given SE

---

## Questions for Further Investigation

1. **Contextual:** What is the substantive meaning of these effects? Are effects of size ~8 meaningful in practice?

2. **Prior information:** Is there external information about:
   - Typical effect sizes in this domain?
   - Expected heterogeneity across schools?
   - Direction of effects (all positive expected)?

3. **Sampling:** How were schools selected? Is this a representative sample or convenience sample?

4. **Study design:** Were the treatments and measurements identical across schools? Any differences could explain heterogeneity.

5. **Missing data:** Are there other schools that weren't included? Publication bias concerns?

---

## Analysis Workflow Summary

```
Initial Exploration (Script 01)
    ↓
Distribution Analysis (Script 02)
    ↓
Visualizations (Script 02) → 6 multi-panel plots
    ↓
Hypothesis Testing (Script 03) → 5 competing hypotheses
    ↓
Model Recommendations (Final Report)
```

**Total analysis time:** ~3 rounds of iterative exploration
**Key decision point:** After Round 3, evidence pointed strongly to homogeneity. Rounds 4-5 confirmed this and tested alternative explanations.

---

## Files Generated

### Code
- `01_initial_exploration.py` - Data structure and summary statistics
- `02_visualizations.py` - All plots
- `03_hypothesis_testing.py` - Formal hypothesis tests
- `data_with_diagnostics.csv` - Augmented data with z-scores

### Visualizations
- `forest_plot.png` - Classic meta-analysis forest plot
- `distribution_analysis.png` - 4-panel distribution overview
- `effect_vs_uncertainty.png` - Scatter plot with correlation
- `precision_analysis.png` - 2-panel: funnel plot + residuals
- `school_profiles.png` - Bubble plot of school estimates
- `heterogeneity_diagnostics.png` - 4-panel diagnostic plots

---

## Conclusion

This EDA reveals a classic example of **apparent heterogeneity that is actually homogeneity masked by measurement error**. The Eight Schools problem is famous precisely because it challenges our intuitions about pooling - the data *look* heterogeneous (effects range from -3 to 28) but statistical tests consistently fail to find evidence of true heterogeneity.

The key insight is that **high measurement uncertainty (large sigma) can make homogeneous data appear heterogeneous**. Only by properly accounting for the known measurement error can we see that the data are consistent with a common true effect.

This has important implications for Bayesian modeling: while we should use a hierarchical model structure (to be conservative and allow for heterogeneity), we should expect the posterior to strongly favor tau ≈ 0, resulting in substantial shrinkage toward the pooled mean.
