# EDA Log - Analyst 2: Patterns, Structure, and Relationships

## Initial Observations

### Data Structure
- 12 groups with binomial outcome data
- No missing values, data quality is excellent
- Calculations verified: proportion = r/n, failures = n-r

### Key Metrics
- Total observations: 2,814
- Total events: 208
- Overall pooled proportion: 0.0739 (7.39%)
- Sample sizes range: 47 to 810 (CV = 0.846, highly variable)
- Proportions range: 0.0 to 0.144 (CV = 0.521, moderate variability)

### Critical Findings
1. **Group 1 has zero events** (0/47) - this is a rare event problem
2. **Substantial heterogeneity**: proportions range from 0% to 14.4%
3. **Extreme sample size variation**: 17-fold difference between smallest (n=47) and largest (n=810)
4. **Weighted vs unweighted means very similar** (0.0739 vs 0.0737), suggesting sample size not strongly correlated with proportion

### Hypotheses to Test
1. **H1**: There is a sequential/temporal trend in proportions across groups
2. **H2**: Sample size is correlated with observed proportions (selection bias?)
3. **H3**: Groups are exchangeable (supporting hierarchical pooling)
4. **H4**: Heterogeneity is due to random variation vs true group differences

## Round 1: Detailed Exploration

### Visualization 1: Sequential Pattern Analysis (01_sequential_patterns.png)
**Purpose**: Examine if there's any ordering/temporal pattern across groups

**Key Findings**:
- No obvious linear trend across group indices
- Proportions fluctuate between 0% and 14.4%
- Three notable peaks: Groups 2, 8, and 11 (all above 11%)
- Sample sizes highly variable, with Group 4 being exceptionally large (n=810)
- Confidence interval widths inversely related to sample size (as expected)
- Groups 1, 3, 10 have small sample sizes (n < 100) leading to high uncertainty

**Statistical Tests**:
- Pearson correlation (group vs proportion): r = 0.37, p = 0.23 (NOT significant)
- Spearman correlation: rho = 0.40, p = 0.20 (NOT significant)
- Linear regression slope: 0.004, R² = 0.14 (weak, not significant)
- Runs test: 5 runs (expected ~6 for random sequence)

**Conclusion**: **NO significant sequential/temporal pattern detected**. Groups appear randomly ordered.

### Visualization 2: Sample Size vs Proportion Relationships (02_sample_size_relationships.png)
**Purpose**: Test if sample size correlates with observed proportions (selection bias?)

**Key Findings**:
- Near-zero correlation between n and proportion (r = 0.006, p = 0.99)
- No systematic bias: large samples don't systematically show higher/lower rates
- Uncertainty (CI width) follows theoretical 1/sqrt(n) relationship closely
- On log scale, relationship between n and uncertainty is approximately linear
- Group 8 (n=215, p=0.144) is an outlier with high proportion
- Groups 2 and 11 also show elevated proportions but with different sample sizes

**Statistical Tests**:
- Correlation (n vs proportion): r = 0.006, p = 0.99 (NOT significant)
- Mann-Whitney U test (small vs large n): p = 0.82 (NOT significant)
- Small n groups (≤202): mean proportion = 0.066
- Large n groups (>202): mean proportion = 0.081
- Difference not statistically significant

**Conclusion**: **NO selection bias detected**. Sample size is independent of proportion.

### Visualization 3: Uncertainty Quantification (03_uncertainty_quantification.png)
**Purpose**: Quantify precision and evaluate confidence intervals

**Key Findings from Forest Plot**:
- Group 1 (zero events) has wide CI: [0.00, 0.08]
- Most confident: Group 4 (largest n=810): CI = [0.04, 0.07]
- Three groups clearly above pooled estimate: 2, 8, 11
- One group on margin: Group 1 (but includes zero)

**Precision Analysis**:
- Precision varies 20-fold across groups (excluding zero group)
- Group 4 has highest precision due to large sample size
- Groups 1, 3, 10 have lowest precision due to small samples

**Z-Score Distribution**:
- Most groups within ±2 SD of pooled estimate
- Three outliers detected: Groups 2, 8, 11 (z > 2)
- Group 5 borderline (z = -2.00)
- Distribution NOT centered at zero, suggesting heterogeneity

**Observed vs Expected SE**:
- Points cluster around diagonal but with scatter
- Some groups show higher SE than expected under pooled model
- Suggests extra-binomial variation

**Conclusion**: Substantial uncertainty in estimates, especially for small-n groups. Outliers suggest true group differences.

### Visualization 4: Rare Events Analysis (04_rare_events_analysis.png)
**Purpose**: Investigate zero inflation and rare event handling

**Key Findings**:
- **Group 1: Zero events** (0/47)
  - Under pooled model, P(r=0) ≈ 0.025 (2.5%)
  - Not extremely unlikely, but noteworthy
  - Standard residual: z = -1.94 (borderline)

- **Expected vs Observed Events**:
  - Group 8: observed 31, expected 15.9 (z = 3.94) - major outlier
  - Group 11: observed 29, expected 18.9 (z = 2.41) - outlier
  - Group 2: observed 18, expected 10.9 (z = 2.22) - outlier
  - Group 4: observed 46, expected 59.8 (z = -1.86) - slight undershoot
  - Group 5: observed 8, expected 15.6 (z = -2.00) - borderline outlier

- **Standardized Residuals**:
  - 3 groups exceed |z| > 2 threshold
  - Residuals NOT randomly distributed around zero
  - Clear evidence of heterogeneity

**Conclusion**: Group 1's zero count is unusual but not extreme. Three groups (2, 8, 11) show significantly elevated rates.

## Round 2: Hypothesis Testing and Model Selection

### Statistical Test Results

#### H1: Sequential Trend
**Result**: REJECTED (p = 0.20)
- No significant correlation between group index and proportion
- No temporal or spatial ordering effect detected
- Group indices appear arbitrary

#### H2: Sample Size Correlation
**Result**: REJECTED (p = 0.99)
- No correlation between sample size and proportion
- No evidence of selection bias
- Sample size variation appears random

#### H3: Homogeneity Test
**Result**: REJECTED (p < 0.0001)
- Chi-square test: χ² = 38.56, p = 6.3×10⁻⁵
- Likelihood ratio test: LR = 38.53, p = 6.4×10⁻⁵
- **Strong evidence of heterogeneity**
- Groups are NOT exchangeable
- **Overdispersion parameter φ = 3.51** (severe overdispersion)

#### H4: Variance Decomposition
**Result**: CONFIRMED
- **ICC = 0.662** (66% of variance is between-group)
- **I² = 71.5%** (moderate-to-high heterogeneity)
- Between-group variance (0.000569) > Within-group variance (0.000290)
- **Substantial true differences between groups**

### Visualization 5: Pooling Considerations (05_pooling_considerations.png)
**Purpose**: Evaluate complete vs no pooling vs partial pooling strategies

**Key Insights**:
1. **Complete Pooling** (orange): All groups = 0.0739
   - Ignores group differences
   - Underestimates uncertainty for high-rate groups
   - Overestimates for low-rate groups
   - NOT RECOMMENDED given heterogeneity

2. **No Pooling** (blue): Use observed proportions
   - Honors individual estimates
   - High uncertainty for small-n groups
   - Group 1 = 0.0, unstable
   - Overfits to noise

3. **Partial Pooling** (green): Hierarchical/empirical Bayes
   - Shrinks extreme estimates toward grand mean
   - Amount of shrinkage ∝ uncertainty
   - Group 1: shrunk from 0.00 to ~0.04
   - Group 8: shrunk from 0.144 to ~0.12
   - **RECOMMENDED approach**

**Effective Sample Sizes**:
- Partial pooling increases effective n for small groups
- Borrows strength from other groups
- Group 1 benefits most (small n + zero events)

**Variance Comparison**:
- No pooling: highest variance (0.0015)
- Complete pooling: zero variance (all same)
- Partial pooling: intermediate (0.0008)
- Balances bias-variance tradeoff

## Key Findings Summary

### 1. No Sequential Patterns
- Group indices appear arbitrary
- No temporal, spatial, or ordering effects
- Proportions vary randomly across groups

### 2. Sample Size Independent of Proportion
- No selection bias detected
- Large samples don't show systematically different rates
- Sample size variation appears random

### 3. Substantial Heterogeneity
- Groups are NOT homogeneous (p < 0.0001)
- 3.5× overdispersion relative to binomial
- 71.5% of variation is between-group (I²)
- ICC = 0.66

### 4. Three Outlier Groups
- Group 2: 12.2% (z = 2.22)
- Group 8: 14.4% (z = 3.94) **highest**
- Group 11: 11.3% (z = 2.41)
- All significantly above pooled estimate of 7.4%

### 5. One Zero-Event Group
- Group 1: 0/47 events
- Not extremely unlikely under pooled model (P ≈ 2.5%)
- But requires special handling in modeling

### 6. High Uncertainty in Small Groups
- Groups 1, 3, 10 have n < 100
- Confidence intervals 2-3× wider than large groups
- Benefit most from hierarchical pooling

## Modeling Recommendations

### Primary Recommendation: Hierarchical (Partial Pooling) Model
**Rationale**:
- Strong evidence of between-group variation (ICC = 0.66, I² = 71.5%)
- Groups are heterogeneous (p < 0.0001)
- But complete independence would overfit small-n groups
- Partial pooling balances individual and group-level information

**Suggested Models**:
1. **Beta-Binomial Model** (accounts for overdispersion)
   - Naturally handles overdispersion parameter φ = 3.5
   - More flexible than binomial for heterogeneous data

2. **Random Effects Logistic Regression**
   - `logit(p_i) = μ + α_i, α_i ~ N(0, τ²)`
   - Estimate between-group variance τ²
   - Shrinkage automatic via BLUP/EB

3. **Bayesian Hierarchical Model**
   - `r_i ~ Binomial(n_i, p_i)`
   - `logit(p_i) ~ N(μ, τ²)`
   - Prior on μ and τ² (e.g., weakly informative)
   - Naturally handles zero-event group via continuity correction

### Alternative: Two-Component Mixture Model
**Rationale**:
- Three groups (2, 8, 11) clearly elevated
- Remaining groups cluster around 6-7%
- Could represent two subpopulations

**Model**:
- Component 1: Low risk (~7%)
- Component 2: High risk (~12-14%)
- Estimate mixture proportion π

### NOT Recommended: Complete Pooling
- Rejected by homogeneity test (p < 0.0001)
- Ignores 66% of variance
- Severe overdispersion (φ = 3.5)

### NOT Recommended: No Pooling (Saturated Model)
- 12 parameters for 12 groups (overfitting)
- Unstable for small-n groups
- Group 1 estimate = 0 is problematic

## Data Quality Considerations

### Strengths
- No missing data
- Calculations verified correct
- Reasonable sample sizes overall (total n = 2,814)
- Clear documentation

### Concerns
1. **Zero-event group**: Group 1 requires continuity correction or Bayesian prior
2. **Extreme outliers**: Groups 8, 11 may warrant investigation (data quality? different population?)
3. **Sample size imbalance**: 17-fold range (47 to 810) complicates pooling decisions

### Pre-modeling Checks Needed
- Verify Group 1 data (truly zero events or missing data?)
- Investigate Groups 2, 8, 11 (what makes them different?)
- Check if group labels have meaning (geography, time, clinic, etc.)
- Confirm binomial assumption (independence within groups)

## Tentative vs Robust Findings

### Robust (High Confidence)
✓ Groups are heterogeneous (multiple tests, p < 0.0001)
✓ Substantial between-group variation (ICC = 0.66)
✓ No sample size bias (p = 0.99)
✓ No sequential trend (p = 0.20)
✓ Three groups significantly elevated (z > 2)

### Tentative (Requires Validation)
? Zero-event in Group 1 (P = 2.5% under null, unusual but not extreme)
? Mixture model interpretation (needs formal model comparison)
? Outlier groups represent true subpopulation vs measurement error

## Next Steps for Modeling

1. **Fit hierarchical binomial/beta-binomial model**
   - Estimate between-group variance τ²
   - Calculate shrunken estimates
   - Compare to complete/no pooling via AIC/BIC

2. **Sensitivity analysis for Group 1**
   - Include/exclude zero-event group
   - Use continuity correction (add 0.5 to r and n)
   - Compare Bayesian vs frequentist handling

3. **Investigate outliers**
   - External validation (are Groups 2, 8, 11 truly different?)
   - Sensitivity analysis excluding outliers
   - Random effects vs fixed effects for outliers

4. **Model comparison**
   - Binomial vs Beta-Binomial vs Mixture
   - Complete vs No vs Partial pooling
   - Use cross-validation or posterior predictive checks

5. **Posterior predictive checks**
   - Simulate data from fitted model
   - Compare to observed (especially for Group 1 and outliers)
   - Assess model adequacy
