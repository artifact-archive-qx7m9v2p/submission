# Comprehensive Exploratory Data Analysis Report
## Binomial Dataset Analysis

**Date**: 2025-10-30
**Dataset**: `/workspace/data/data.csv`
**Observations**: 12 binomial trials
**Analyst**: EDA Specialist

---

## Executive Summary

This dataset contains 12 independent binomial trials with varying sample sizes (n = 47 to 810) and success counts. The primary finding is **strong evidence of overdispersion**: the variance in success proportions is approximately **3.5 times larger** than expected under a simple binomial model with constant probability (χ² = 38.6, df = 11, p < 0.001).

**Key Recommendations**:
1. Do NOT use a simple binomial model with constant probability
2. Consider Beta-Binomial or mixture models to account for overdispersion
3. The pooled success probability is approximately 0.074, but substantial variation exists
4. No evidence of temporal trends or sample-size-dependent bias

---

## 1. Data Structure and Quality

### 1.1 Dataset Description

| Attribute | Details |
|-----------|---------|
| **Observations** | 12 trials (trial_id 1-12) |
| **Total trials** | 2,814 (sum of n) |
| **Total successes** | 208 (sum of r) |
| **Overall success rate** | 0.0739 (7.39%) |
| **Missing values** | None |
| **Data quality issues** | None detected |

### 1.2 Variable Distributions

**Sample Sizes (n)**:
- Range: [47, 810]
- Mean: 234.5, Median: 201.5
- High variability (CV = 0.846)
- One very large sample: trial 4 (n = 810)

See visualization: `visualizations/sample_size_distribution.png`

**Success Proportions**:
- Range: [0.000, 0.144]
- Mean: 0.074, Median: 0.067
- Std: 0.038
- Shows substantial variation across trials

See visualization: `visualizations/proportion_distribution.png`

### 1.3 Data Validation

All data quality checks passed:
- ✓ No missing values
- ✓ All r ≤ n (successes ≤ trials)
- ✓ Proportions correctly calculated
- ✓ No negative values
- ✓ No data entry errors detected

---

## 2. Overdispersion Analysis

### 2.1 Statistical Evidence

**Chi-Square Goodness of Fit Test**:
```
H0: Data follows Binomial(n_i, p) with constant p = 0.0739
Test statistic: χ² = 38.56
Degrees of freedom: 11
P-value: 0.000063
Result: REJECT H0
```

**Conclusion**: Strong evidence against constant probability model.

**Dispersion Parameter**:
```
φ = χ²/df = 38.56/11 = 3.505
```

The observed variance is **3.5 times larger** than expected under the binomial model.

### 2.2 Standardized Residuals

Under a binomial model with constant p, standardized residuals should follow approximately N(0, 1).

**Observed**:
- Mean: 0.077 (expected: 0)
- Variance: 3.50 (expected: 1.0)
- Range: [-2.0, 3.9]

**Outliers** (|z| > 2):
- Trial 2: z = 2.22, p = 0.122
- Trial 8: z = 3.94, p = 0.144 ← **most extreme**
- Trial 11: z = 2.41, p = 0.113

See visualizations:
- `visualizations/standardized_residuals.png`
- `visualizations/funnel_plot.png`

### 2.3 Interpretation

The overdispersion indicates that:
1. Success probabilities vary across trials
2. A single constant p is inadequate
3. Extra-binomial variation is present
4. Need models that account for probability heterogeneity

---

## 3. Pattern and Structure Analysis

### 3.1 Temporal Patterns

**Question**: Is there a trend across trial_id?

**Analysis**:
- Pearson correlation: r = 0.371, p = 0.235
- Spearman correlation: ρ = 0.399, p = 0.199
- Linear regression: R² = 0.138, p = 0.235

**Conclusion**: **No significant temporal trend** detected. Trial order appears unimportant.

See visualization: `visualizations/proportion_vs_trial.png`

### 3.2 Sample Size Effects

**Question**: Does sample size affect observed proportions?

**Analysis**:
- Pearson correlation: r = 0.006, p = 0.986
- Spearman correlation: ρ = 0.088, p = 0.787

**Conclusion**: **No relationship** between sample size and proportion. This:
- Rules out size-dependent bias
- Supports exchangeability assumption
- Validates pooling across different sample sizes

See visualization: `visualizations/proportion_vs_sample_size.png`

### 3.3 Group Structure

**Question**: Are there distinct probability regimes?

**Median Split Analysis**:
```
Group 0 (Below median, n=6):
  Trials: 1, 4, 5, 6, 7, 12
  Mean proportion: 0.048
  Std: 0.026

Group 1 (Above median, n=6):
  Trials: 2, 3, 8, 9, 10, 11
  Mean proportion: 0.099
  Std: 0.032

T-test: t = -3.08, p = 0.012
```

**Conclusion**: Evidence for **two distinct groups** with different underlying probabilities.

**Tercile Analysis**:
```
Low (n=4):    trials 1, 4, 5, 7     → mean p = 0.039
Medium (n=4): trials 3, 6, 9, 12    → mean p = 0.067
High (n=4):   trials 2, 8, 10, 11   → mean p = 0.115
```

Suggests possible gradient of success probabilities rather than sharp dichotomy.

### 3.4 Outlier Detection

**IQR Method** (1.5×IQR rule):
- Lower bound: 0.014
- Upper bound: 0.136

**Outliers identified**:
1. **Trial 1**: p = 0.000 (0/47)
   - Probability under pooled model: 2.7%
   - Unusual but not impossible

2. **Trial 8**: p = 0.144 (31/215)
   - Highest proportion in dataset
   - Standardized residual: 3.94

**Interpretation**: Both are statistical outliers but could represent:
- Natural binomial variation (unlikely but possible)
- Different underlying processes
- Extreme draws from a heterogeneous distribution

### 3.5 Sequence Randomness

**Runs Test**: Testing if above/below median forms random sequence
- Observed runs: 5
- Expected runs: 7.0
- P-value: 0.226

**Conclusion**: Sequence appears **random**. No evidence of systematic ordering or autocorrelation.

---

## 4. Visual Findings

### 4.1 Key Visualizations

All visualizations are stored in `/workspace/eda/visualizations/`:

1. **`sample_size_distribution.png`**
   - Shows right-skewed distribution of sample sizes
   - Trial 4 (n=810) is clear outlier
   - Most samples cluster around 150-250

2. **`proportion_distribution.png`**
   - Success proportions spread across [0, 0.14]
   - Roughly uniform with possible bimodality
   - Pooled estimate (red line) near center

3. **`proportion_vs_trial.png`**
   - No clear temporal trend (flat pattern)
   - Gray error bars show binomial 95% CI
   - Multiple observations exceed expected range

4. **`proportion_vs_sample_size.png`**
   - No correlation between n and proportion
   - Points scatter randomly
   - Validates size-independence assumption

5. **`standardized_residuals.png`** (2-panel)
   - Left: Residuals vs trial ID (no pattern)
   - Right: Distribution vs N(0,1) (heavier tails)
   - Clear evidence of excess variation

6. **`comprehensive_comparison.png`** (4-panel)
   - Success counts vary widely across trials
   - Sample sizes highly heterogeneous
   - Observed variance consistently exceeds binomial expectation
   - Log-scale panel shows magnitude of discrepancy

7. **`qq_plot.png`**
   - Q-Q plot of standardized residuals vs normal
   - Moderate departure from diagonal
   - Heavier tails than expected

8. **`funnel_plot.png`**
   - Classic funnel plot: proportion vs precision
   - Red dashed lines: 95% CI under binomial model
   - Multiple points outside funnel → overdispersion
   - Trials 1, 2, 8, 11 clearly exceed limits

### 4.2 Visualization Summary

The plots consistently demonstrate:
- Overdispersion is real and substantial
- No systematic biases (temporal, size-based)
- Possible group structure
- Need for models beyond simple binomial

---

## 5. Competing Hypotheses

### Hypothesis 1: Constant Probability Model
```
Model: r_i ~ Binomial(n_i, p) with p constant across all trials
Status: REJECTED
Evidence: χ² = 38.6, p < 0.001
Confidence: Very high
```

### Hypothesis 2: Temporal Trend
```
Model: p_i varies systematically with trial_id
Status: NOT SUPPORTED
Evidence: Correlation p > 0.2
Confidence: Moderate (small sample)
```

### Hypothesis 3: Sample Size Bias
```
Model: p_i depends on n_i
Status: NOT SUPPORTED
Evidence: Correlation p > 0.7
Confidence: High
```

### Hypothesis 4: Two-Group Mixture
```
Model: Trials come from two populations with different p
Status: TENTATIVELY SUPPORTED
Evidence: Median-split t-test p = 0.012
Confidence: Moderate (needs validation)
```

### Hypothesis 5: Continuous Variation (Beta-Binomial)
```
Model: p_i ~ Beta(α, β), r_i ~ Binomial(n_i, p_i)
Status: PLAUSIBLE
Evidence: Dispersion parameter φ = 3.5
Confidence: Cannot distinguish from mixture yet
```

---

## 6. Modeling Recommendations

### 6.1 Do NOT Use

**Simple Binomial Model**: `r_i ~ Binomial(n_i, p)`
- Strongly rejected by data
- Will severely underestimate uncertainty
- Confidence intervals will be too narrow
- Predictions will be overconfident

### 6.2 Recommended Model Classes

#### Option 1: Beta-Binomial Model (RECOMMENDED)
```
θ_i ~ Beta(α, β)
r_i | θ_i ~ Binomial(n_i, θ_i)
```

**Advantages**:
- Naturally accounts for overdispersion
- Two parameters (α, β) describe variation in success probability
- Mathematically tractable
- Standard implementation in most software

**Prior Recommendations**:
- E[θ] = α/(α+β) should be near 0.074
- Var[θ] should allow dispersion factor of 3-4
- Example: α ~ Gamma(2, 20), β ~ Gamma(25, 250)
- Or use φ parameterization with φ ~ Gamma(2, 0.5)

#### Option 2: Two-Component Mixture Model
```
Component 1: p_low ≈ 0.05, weight w_1
Component 2: p_high ≈ 0.11, weight w_2 = 1 - w_1
```

**Advantages**:
- Captures possible discrete probability regimes
- Interpretable components
- Can identify which trials belong to which group

**Prior Recommendations**:
- p_low ~ Beta(2, 38) [centers near 0.05]
- p_high ~ Beta(4, 32) [centers near 0.11]
- w ~ Dirichlet(1, 1) [uniform mixing]

**Considerations**:
- More complex than Beta-Binomial
- May overfit with only 12 observations
- Use for sensitivity analysis

#### Option 3: Hierarchical Binomial Model
```
θ_i ~ Beta(α, β) for each trial i
r_i | θ_i ~ Binomial(n_i, θ_i)
```

**Advantages**:
- Most flexible
- Each trial has own probability
- Can examine trial-specific posteriors
- Borrows strength across observations

**Prior Recommendations**:
- Hyperpriors on α, β
- Example: α ~ Gamma(2, 2), β ~ Gamma(20, 20)
- Centers mass near 0.074 with substantial spread

### 6.3 Prior Specification Details

**For pooled probability**:
```
Based on data: pooled p̂ = 0.0739
Reasonable range: [0, 0.2]
Suggested prior: Beta(2, 25)
  - E[p] = 2/27 ≈ 0.074
  - Mode = 1/25 = 0.04
  - 95% CI: [0.005, 0.20]
```

**For overdispersion parameter** (Beta-Binomial):
```
Based on data: φ̂ = 3.5
Need: Prior that allows substantial overdispersion
Suggested: φ ~ Gamma(2, 0.5)
  - E[φ] = 4
  - Allows range [0.5, 10]
```

**For mixture proportions**:
```
No strong evidence for unequal mixing
Suggested: Dirichlet(1, 1) or Dirichlet(2, 2)
```

### 6.4 Model Selection Strategy

1. **Fit all three recommended models**
2. **Compare using**:
   - WAIC (Widely Applicable Information Criterion)
   - LOO-CV (Leave-One-Out Cross-Validation)
   - Posterior predictive checks
3. **Assess**:
   - Predictive accuracy
   - Uncertainty quantification
   - Interpretability
4. **Sensitivity analysis**:
   - Fit with/without trial 1
   - Vary prior specifications
   - Check robustness of conclusions

---

## 7. Specific Findings for Modeling

### 7.1 Parameter Estimates

**Pooled Proportion**:
- Point estimate: 0.0739
- 95% confidence interval (binomial): [0.064, 0.084]
- Note: This CI is too narrow (doesn't account for overdispersion)
- Adjusted CI (accounting for φ=3.5): approximately [0.04, 0.11]

**Dispersion**:
- Variance multiplier: 3.5
- Suggests substantial heterogeneity in underlying probabilities
- Beta-Binomial φ parameter should be in range [2, 5]

### 7.2 Covariate Considerations

**Trial ID**: Not predictive (p > 0.2) → Do not include as covariate

**Sample Size**: No relationship with proportion (p > 0.7) → Exchangeability holds

**Other covariates**: Not available in dataset. If domain knowledge suggests relevant covariates (e.g., experimental conditions, time periods, locations), consider hierarchical model with covariates.

### 7.3 Special Considerations

**Trial 1 (zero successes)**:
- Probability: 2.7% under pooled model
- Decision: Include in main analysis but conduct sensitivity analysis
- If excluded, pooled p increases slightly to ~0.079

**Trial 4 (n=810)**:
- Dominates total sample size
- Has proportion (0.057) close to pooled estimate
- Provides strong anchor but shouldn't drive all conclusions
- Weighting in hierarchical model is appropriate

---

## 8. Assumptions for Modeling

### 8.1 Validated Assumptions

✓ **Independence**: No evidence of temporal dependence (runs test p = 0.23)

✓ **Exchangeability**: No sample size effect (p > 0.7)

✓ **Data Quality**: No missing data, errors, or anomalies

✓ **Binomial Likelihood**: Valid given r ≤ n for all observations

### 8.2 Violated Assumptions

✗ **Constant Probability**: Strongly rejected (p < 0.001)

✗ **Binomial Variance**: Variance is 3.5x expected

### 8.3 Uncertain Assumptions

? **Homogeneity**: Possible 2-3 groups vs continuous variation (needs modeling)

? **Trial 1**: True outlier vs extreme draw (sensitivity analysis needed)

---

## 9. Data Generation Process Considerations

### 9.1 What Could Cause Overdispersion?

**Possible mechanisms**:

1. **Heterogeneous populations**: Trials sample from different populations with different success rates

2. **Unmeasured covariates**: Success probability depends on factors not recorded

3. **Batch effects**: Trials conducted under different conditions

4. **Natural variation**: True probability varies over time/space

5. **Measurement heterogeneity**: Definition of "success" varies slightly

### 9.2 Domain Context Questions

Understanding the data generation process would help model selection:

- Are these trials from the same experiment or different studies?
- Is there temporal structure (even if trial_id doesn't capture it)?
- Are there known experimental conditions that vary?
- What is being measured? (e.g., clinical outcomes, manufacturing defects, survey responses)

### 9.3 Practical Significance

**Effect size**:
- Range of proportions: [0, 0.14]
- IQR: [0.06, 0.09]
- Practical importance depends on domain

**Uncertainty**:
- Standard binomial model: SE ≈ 0.005
- Accounting for overdispersion: SE ≈ 0.009 (1.9x larger)
- Substantially affects decision-making under uncertainty

---

## 10. Validation and Next Steps

### 10.1 Recommended Validation Steps

1. **Posterior Predictive Checks**:
   - Simulate new datasets from fitted models
   - Compare to observed data
   - Check: proportion distribution, overdispersion, outliers

2. **Cross-Validation**:
   - LOO-CV for each observation
   - Assess predictive accuracy
   - Identify influential observations

3. **Sensitivity Analysis**:
   - Vary priors (weak to strong)
   - Exclude trial 1
   - Exclude trial 4
   - Check robustness of conclusions

4. **Model Comparison**:
   - Fit all recommended models
   - Compare WAIC/LOO
   - Examine posterior predictives
   - Check calibration

### 10.2 Additional Analyses (if needed)

1. **Bayesian Model Averaging**: If models similar performance, average predictions

2. **Robust Methods**: If concerned about outliers, consider Student-t distribution

3. **Zero-Inflation**: Trial 1 suggests possible zero-inflation (unlikely but check)

4. **Covariate Search**: If domain knowledge suggests relevant variables

---

## 11. Summary and Conclusions

### 11.1 Key Findings (High Confidence)

1. **Strong overdispersion present** (φ = 3.5, p < 0.001)
   - Most robust finding
   - Multiple tests confirm
   - Visual evidence compelling

2. **Pooled success rate ≈ 0.074** (7.4%)
   - Based on 208/2814 successes
   - But substantial variation exists

3. **No temporal trend** (p > 0.2)
   - Trial order uninformative
   - Sequence appears random

4. **No sample size bias** (p > 0.7)
   - Exchangeability assumption valid
   - Can pool across different n

5. **Simple binomial model inadequate**
   - Strongly rejected
   - Would severely underestimate uncertainty

### 11.2 Tentative Findings (Lower Confidence)

1. **Possible two-group structure** (p = 0.012)
   - Needs model-based validation
   - Small sample limits certainty

2. **Trial 1 is statistical outlier**
   - But only 2.7% probability
   - Could be natural variation

3. **Trials 2, 8, 11 form high-probability group**
   - Visual clustering
   - Needs formal testing

### 11.3 Modeling Strategy

**Primary recommendation**: **Beta-Binomial model**
- Best balance of flexibility and simplicity
- Naturally handles overdispersion
- Standard implementation

**Secondary**: **Two-component mixture**
- For sensitivity analysis
- If group structure is of interest

**For comparison**: **Hierarchical binomial**
- Most flexible
- Trial-specific inference

### 11.4 Critical Considerations

1. **Do not use constant-p binomial** → Will give wrong answers
2. **Account for overdispersion** → Essential for valid inference
3. **Check sensitivity to outliers** → Especially trial 1
4. **Use informative priors carefully** → Data somewhat sparse (n=12)
5. **Validate with posterior predictives** → Check model adequacy

---

## 12. Reproducibility

All analyses are fully reproducible using:

**Code**: `/workspace/eda/code/`
- `01_initial_exploration.py` - Data quality and descriptives
- `02_overdispersion_analysis.py` - Statistical tests
- `03_visualization.py` - All plots
- `04_pattern_analysis.py` - Pattern detection

**Visualizations**: `/workspace/eda/visualizations/`
- 8 publication-quality plots
- All referenced in this report

**Documentation**:
- `/workspace/eda/eda_report.md` (this file)
- `/workspace/eda/eda_log.md` (detailed log)

**Data**: `/workspace/data/data.csv`

All code uses:
- Python 3
- pandas, numpy, scipy, matplotlib, seaborn
- No proprietary software required
- Random seeds set where applicable

---

## Appendix: Quick Reference

### Model Comparison Table

| Model | Pros | Cons | Complexity | Recommended |
|-------|------|------|------------|-------------|
| Simple Binomial | Simple, fast | **REJECTED** | Low | **NO** |
| Beta-Binomial | Handles overdispersion, standard | Less flexible | Medium | **YES** |
| 2-Component Mixture | Interpretable groups | May overfit | Medium-High | For sensitivity |
| Hierarchical | Most flexible, trial-specific | Complex | High | For deep analysis |

### Key Statistics Reference

```
Sample size:         n = 12 trials
Total observations:  N = 2,814
Total successes:     r = 208
Pooled proportion:   p̂ = 0.0739
Overdispersion:      φ = 3.5
Chi-square:          χ² = 38.6, p < 0.001
Proportion range:    [0.000, 0.144]
Proportion variance: 0.00148 (vs 0.00046 expected)
```

### Files Reference

```
/workspace/data/data.csv                          # Original data
/workspace/eda/code/01_initial_exploration.py     # Initial EDA
/workspace/eda/code/02_overdispersion_analysis.py # Dispersion tests
/workspace/eda/code/03_visualization.py           # All plots
/workspace/eda/code/04_pattern_analysis.py        # Pattern detection
/workspace/eda/visualizations/*.png               # 8 plots
/workspace/eda/eda_report.md                      # This report
/workspace/eda/eda_log.md                         # Detailed log
```

---

**Report End**

For questions or additional analyses, refer to the detailed exploration log at `/workspace/eda/eda_log.md`.
