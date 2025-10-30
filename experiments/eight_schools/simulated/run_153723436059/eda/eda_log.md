# Eight Schools Dataset - Exploratory Data Analysis Log

## Analysis Overview
**Date**: 2025-10-29
**Dataset**: Eight Schools (Classic Hierarchical Modeling Dataset)
**Analyst**: EDA Specialist
**Purpose**: Comprehensive exploration to inform hierarchical Bayesian model design

---

## Round 1: Initial Data Exploration

### Data Structure
- **Sample size**: 8 schools
- **Variables**: school (ID), effect (observed treatment effect), sigma (known standard error)
- **Data quality**: No missing values, no duplicates, all sigma values positive
- **Range**: Schools 1-8, effects from -4.88 to 26.08, sigma from 9 to 18

### Key Descriptive Statistics

#### Effects (Treatment Outcomes)
- Mean: 12.50
- Median: 11.92
- SD: 11.15
- Range: 30.96 (from -4.88 to 26.08)
- IQR: 16.10

#### Standard Errors (Uncertainty)
- Mean: 12.50
- Median: 11.00
- SD: 3.34
- Range: 9 (from 9 to 18)

### Initial Observations

1. **Large Variation in Effects**: The observed effects show substantial variation (SD=11.15), ranging from negative to strongly positive values.

2. **School 5 is Negative Outlier**: School 5 shows a negative effect (-4.88), with z-score of -1.56, making it notably different from other schools.

3. **Variable Uncertainty**: Standard errors vary by factor of 2 (9 to 18), suggesting different study precision across schools.

4. **Only One "Significant" School**: School 4 has effect/sigma ratio > 2 (2.34), making it the only school nominally significant at the 95% level if tested individually.

5. **Normality Appears Reasonable**: Shapiro-Wilk test (p=0.675) indicates effects are consistent with normal distribution.

### Variance Components Analysis

**Critical Finding**: The ratio of between-school to within-school variance is 0.749

- Between-school variance (empirical): 124.27
- Mean within-school variance (sigma²): 166.00
- **Interpretation**: Observed variation is actually LESS than expected from sampling uncertainty alone!

This suggests:
- Effects might be more similar than independent random draws
- Complete pooling could be justified
- However, hierarchical model would adaptively determine optimal shrinkage

---

## Round 2: Hypothesis Testing

### Hypothesis 1: Are All Effects Equal? (Complete Pooling Test)

**Chi-square test for homogeneity**:
- Test statistic: 7.12 (df=7)
- p-value: 0.417
- **Decision**: FAIL TO REJECT null hypothesis
- **Interpretation**: No strong evidence against complete pooling

**Heterogeneity assessment (I² statistic)**:
- I² = 1.6% (very low heterogeneity)
- Interpretation: Only 1.6% of variation attributable to true between-school differences
- Classification: "Low heterogeneity"

**Pooled Estimates**:
- Weighted mean (inverse variance): 10.02
- Unweighted mean: 12.50
- Difference: 2.48 (weighted mean pulls toward more precise estimates)

### Hypothesis 2: Are Effects Independent? (No Pooling Test)

**Variance ratio test**:
- Empirical variance / Expected variance = 0.655
- Interpretation: Observed variance is ~35% LESS than expected under independence
- Suggests effects may share common structure

**Runs test for randomness**:
- Observed runs: 2
- Expected runs: 5.0
- Z-statistic: -2.29, p-value: 0.022
- **Interpretation**: Evidence of non-random pattern (too few runs)
- Schools cluster by effect magnitude

### Hypothesis 3: Does Uncertainty Relate to Effect Size?

**Correlation tests**:
- Pearson r = 0.428 (p=0.290) - not significant
- Spearman rho = 0.615 (p=0.105) - marginally non-significant
- **Interpretation**: Weak positive trend but not statistically significant
- Standard hierarchical model assumptions appear reasonable

**Egger's test for publication bias**:
- Intercept: 2.184 (p=0.435)
- **Interpretation**: No evidence of funnel plot asymmetry
- No obvious publication bias or small-study effects

### Hypothesis 4: Are Effects Normally Distributed?

**Multiple normality tests**:
1. Shapiro-Wilk: W=0.946, p=0.675 → Accept normality
2. Anderson-Darling: A²=0.216 → Accept at all significance levels
3. Jarque-Bera: JB=0.514, p=0.774 → Accept normality

**Distributional moments**:
- Skewness: -0.125 (nearly symmetric)
- Excess kurtosis: -1.216 (slightly platykurtic, but within reasonable range)
- **Interpretation**: Data consistent with normal distribution

---

## Round 3: Deep Dive Analyses

### Signal-to-Noise Analysis

Effect/Sigma ratios by school:
- School 1: 1.33
- School 2: 1.53
- School 3: 1.63
- **School 4: 2.34** (only one >2)
- School 5: -0.54 (negative)
- School 6: 0.55
- School 7: 0.32
- School 8: 0.47

**Observations**:
- Most schools have moderate signal-to-noise (0.5-1.6)
- Only School 4 crosses conventional significance threshold
- School 5 has negative but weak signal
- High uncertainty prevents strong individual conclusions

### Precision Analysis

Precision weights (1/sigma²):
- Range: 0.0031 to 0.0123
- Ratio of max to min: 4.00x
- Schools 5, 2, 7 have highest precision (smaller sigma)
- Schools 8, 3, 1 have lowest precision (larger sigma)

**Implication**: Precision varies substantially, justifying weighted pooling approaches

### Pooling Comparison

Three strategies compared:
1. **No Pooling**: Use individual estimates (effect_i ± sigma_i)
2. **Complete Pooling**: Use unweighted mean = 12.50
3. **Weighted Pooling**: Inverse variance weighted = 10.02

**Key Insight**: Weighted mean is pulled down by the more precise negative estimate (School 5) and moderate estimates with high precision.

---

## Unexpected Findings

1. **Very Low Heterogeneity**: I² of only 1.6% is surprisingly low. This is often interpreted as evidence FOR complete pooling in meta-analysis contexts.

2. **Variance Paradox**: Empirical variance (124) is less than average sampling variance (166), suggesting effects are more homogeneous than random independent draws.

3. **Runs Test Result**: Significant non-randomness (p=0.022) suggests systematic ordering pattern, though this may be chance with n=8.

4. **Correlation Mystery**: Moderate Spearman correlation (0.615) between effect and sigma is not statistically significant but shows consistent trend. With more data, this might be meaningful.

---

## Robust vs. Tentative Findings

### ROBUST (High Confidence):
1. Effects are consistent with normal distribution (multiple tests agree)
2. No evidence of complete homogeneity OR strong heterogeneity
3. School 4 has largest effect, School 5 has negative effect
4. No evidence of publication bias
5. Substantial variation in precision across schools

### TENTATIVE (Lower Confidence):
1. Evidence for complete pooling (low I², high p-value on homogeneity test)
   - **However**: Small sample size (n=8) limits power
2. Positive correlation between effect and uncertainty
   - Spearman rho=0.615 suggests pattern but p=0.105
3. Non-random ordering of effects
   - May be spurious with small n

### AMBIGUOUS (Conflicting Evidence):
1. **Pooling decision**: Tests suggest complete pooling acceptable, but variance ratio and practical considerations favor hierarchical approach
2. **Effect-uncertainty relationship**: Correlation tests non-significant, but Spearman rho=0.615 is moderately strong

---

## Data Quality Assessment

### Strengths:
- Complete data (no missing values)
- Known uncertainty (sigma provided, not estimated)
- Clean structure with no duplicates
- Reasonable distributional properties

### Limitations:
- **Small sample size** (n=8): Low power for hypothesis tests
- **High uncertainty**: Most individual effects not distinguishable from zero
- **Possible ordering bias**: Runs test suggests non-randomness, unclear cause
- **Wide range of precision**: 4-fold variation complicates interpretation

### Issues Flagged:
1. **School 5**: Negative effect with z-score -1.56, investigate if this is methodologically different
2. **School 8**: Largest uncertainty (sigma=18), verify data collection quality
3. **School 4**: Only nominally significant result, possible winner's curse if selected for follow-up

---

## Modeling Recommendations

### Primary Recommendation: Hierarchical Model with Partial Pooling

**Model Structure**:
```
y_i ~ N(theta_i, sigma_i²)    [Likelihood]
theta_i ~ N(mu, tau²)          [School-level]
mu ~ N(0, large variance)      [Hyperprior on mean]
tau ~ half-Cauchy(0, scale)    [Hyperprior on between-school SD]
```

**Justification**:
1. Balances between complete pooling (suggested by tests) and no pooling
2. Allows data to determine optimal shrinkage via tau estimation
3. If tau → 0, model converges to complete pooling
4. If tau is large, minimal shrinkage applied
5. Standard approach for this dataset in literature

### Alternative Model 1: Complete Pooling

**Model**: theta_i = mu for all i

**When to consider**:
- If I² < 25% is taken as strong evidence (we have 1.6%)
- If simplicity is prioritized over flexibility
- For comparison/sensitivity analysis

**Limitations**: Ignores possibility of real school differences

### Alternative Model 2: No Pooling

**Model**: theta_i independent, no common structure

**When to consider**:
- If schools are believed fundamentally different
- For comparison to show benefit of pooling

**Limitations**: Ignores information sharing, high uncertainty

### Model Class Recommendations

Based on findings, these model classes are appropriate:

1. **Bayesian Hierarchical Model** (PRIMARY)
   - Normal-normal conjugate hierarchy
   - Accounts for known sigma_i
   - Estimates mu and tau from data
   - Natural framework for partial pooling

2. **Random Effects Meta-Analysis** (EQUIVALENT)
   - Standard frequentist approach
   - DerSimonian-Laird or REML for tau estimation
   - Provides similar results to Bayesian approach with flat priors

3. **Empirical Bayes** (SIMPLER ALTERNATIVE)
   - Estimate hyperparameters from data
   - Plug in estimates for shrinkage
   - Computationally simpler but doesn't propagate uncertainty in tau

### NOT Recommended (based on findings):

- **Robust models with t-distributed errors**: Data consistent with normality
- **Mixture models**: No evidence of subgroups or multimodality
- **Non-parametric approaches**: Normal model adequate, small sample size
- **Models with effect-sigma relationship**: No significant correlation detected

---

## Visualizations Created

All visualizations saved to `/workspace/eda/visualizations/`:

1. **01_forest_plot.png**: Primary visualization showing effects with ±1σ and ±2σ intervals, plus pooled estimates
   - **Key insight**: Wide, overlapping intervals; most effects consistent with common mean

2. **02_effect_distributions.png**: Four-panel view of effect distribution
   - Histogram with KDE
   - Q-Q plot (confirms normality)
   - Box plot with individual points
   - Empirical CDF
   - **Key insight**: Distribution approximately normal, no extreme outliers

3. **03_effect_vs_sigma.png**: Relationship between effect size and uncertainty
   - Scatter plot with regression line (r=0.428, p=0.290)
   - Signal-to-noise ratios by school
   - **Key insight**: No significant correlation, School 4 only one with SNR>2

4. **04_variance_components.png**: Within vs. between school variance
   - Within variance (sigma²) by school
   - Between variance (deviations from mean)
   - Precision weights
   - **Key insight**: Within variance dominates, supporting pooling

5. **05_pooling_comparison.png**: Three pooling strategies compared
   - No pooling (individual estimates)
   - Complete pooling (unweighted mean)
   - Weighted pooling (inverse variance)
   - **Key insight**: Substantial shrinkage toward pooled estimates would occur

6. **06_comprehensive_summary.png**: Multi-panel summary dashboard
   - Integrates all key visualizations
   - Summary statistics panel
   - **Key insight**: One-stop overview of dataset structure

---

## Next Steps for Modeling

1. **Implement hierarchical model** with proper hyperpriors:
   - Use half-Cauchy(0, 25) or half-t(3, 0, 10) for tau
   - Weakly informative prior on mu

2. **Conduct sensitivity analysis**:
   - Test different hyperpriors
   - Compare posterior with/without School 5
   - Assess impact of tau prior choice

3. **Model checking**:
   - Posterior predictive checks
   - Leave-one-out cross-validation
   - Compare to complete pooling baseline

4. **Report shrinkage**:
   - Calculate shrinkage factors for each school
   - Show posterior intervals vs. observed
   - Quantify information borrowing

5. **Investigate School 5**:
   - If methodology differs, consider excluding
   - Or model with mixture allowing outlier component

---

## Questions for Domain Experts

1. Are there known methodological differences in School 5's study that would explain negative effect?
2. Why does School 8 have such high uncertainty (sigma=18)?
3. Is there any reason to expect correlation between effect size and study precision?
4. Should all schools receive equal weight in analysis, or are some more reliable?
5. What is the context for this intervention - education, health, other?

---

## Conclusion

The Eight Schools dataset presents a classic case for hierarchical modeling. While formal tests suggest complete pooling might be adequate (I²=1.6%, homogeneity p=0.42), the hierarchical approach provides the best of both worlds:

- **If tau is small**: Model will naturally shrink toward complete pooling
- **If tau is substantial**: Schools retain individual character
- **Data-driven**: Optimal balance learned from data

The key insight is that observed between-school variance (124) is less than average within-school variance (166), suggesting effects are more similar than independent studies. However, we should be cautious about strong conclusions with n=8 schools and high individual uncertainty.

**Primary Recommendation**: Fit normal-normal hierarchical model with weakly informative hyperpriors and let the data determine the degree of pooling.
