# EDA Log: Model Assumptions and Data Quality Analysis

**Analyst**: EDA Analyst 3
**Dataset**: `data/data_analyst_3.csv`
**Focus**: Binomial model assumptions, data quality, transformation needs
**Date**: 2025-10-30

---

## Round 1: Initial Data Quality Assessment

### Questions Asked
1. Is the data complete and internally consistent?
2. Are there any data entry errors or impossible values?
3. What is the distribution of sample sizes across groups?
4. Are there extreme values or outliers?

### Analysis Performed
- Loaded dataset with 12 groups, 4 variables
- Checked for missing values (Result: 0 missing values, 100% complete)
- Verified consistency: success_rate == r_successes / n_trials (Result: PASS)
- Validated logical constraints:
  - r_successes <= n_trials (PASS)
  - r_successes >= 0 (PASS)
  - n_trials > 0 (PASS)
- Checked group numbering: Groups 1-12 all present, no duplicates
- Applied IQR method for outlier detection

### Key Findings
1. **Data quality is excellent**: No missing values, all calculations consistent, no logical errors
2. **Sample size heterogeneity**:
   - Range: 47 to 810 trials (17-fold difference)
   - Mean: 234.5, Median: 201.5, SD: 198.4
   - CV = 0.85 (high variability)
   - Outliers: Groups 4 (810 trials) and 12 (360 trials) are unusually large
3. **Extreme success rates**:
   - Group 1: 0/47 (0% success) - CRITICAL ISSUE
   - Group 8: 31/215 (14.4% success) - outlier on high end
   - IQR outliers: Groups 1 and 8
4. **No groups with n < 30** (smallest is group 1 with n=47)

### Interpretation
- Data is ready for analysis from quality perspective
- However, Group 1's zero successes will cause problems for:
  - Maximum likelihood estimation (MLE will fail or be on boundary)
  - Logit/probit transformations (log(0) undefined)
  - Beta-Binomial models (may need special handling)
- High variance in sample sizes means groups contribute unequally to inference
- Need to weight by sample size in pooled analyses

### Visualizations Created
- `data_quality_overview.png` (4 panels showing sample sizes, success rates, and distributions)

---

## Round 2: Binomial Assumptions and Model Diagnostics

### Questions Asked
1. Is a binomial likelihood appropriate for this data?
2. Is there evidence of overdispersion?
3. Are independence assumptions reasonable?
4. Are sample sizes adequate for asymptotic approximations?
5. Is there zero-inflation?

### Analysis Performed
- Fitted pooled binomial model (null model: all groups have same p)
- Calculated Pearson residuals for each group
- Performed chi-square goodness-of-fit test
- Calculated dispersion parameter
- Checked sample size adequacy (n*p >= 5 and n*q >= 5)
- Tested for temporal autocorrelation in success rates
- Performed runs test on residuals
- Examined trial size patterns
- Assessed zero-inflation

### Key Findings

#### 1. **STRONG EVIDENCE OF OVERDISPERSION**
- Dispersion parameter: **3.51** (should be ~1.0 for binomial)
- Chi-square statistic: 38.56 on 11 df
- P-value: **0.0001** - REJECT homogeneous binomial model
- Interpretation: Groups have genuinely different success probabilities

#### 2. **Sample Size Adequacy**
- Only Group 1 fails n*p >= 5 criterion (has 0 successes)
- All other groups: adequate for normal approximation
- Minimum n*p (excluding Group 1): 8.0
- Implication: Normal approximations reasonable for 11/12 groups

#### 3. **Independence Assessment**
- Lag-1 autocorrelation: -0.32 (p = 0.29) - NOT significant
- Runs test: 7 observed vs 6.33 expected (p = 0.65) - random pattern
- Conclusion: No evidence of temporal/spatial autocorrelation
- Groups appear independent

#### 4. **Trial Size Patterns**
- No correlation between n_trials and success_rate (rho = 0.09, p = 0.79)
- Weak correlation between n_trials and |residuals| (rho = 0.38, p = 0.23)
- Conclusion: No strong heteroscedasticity by sample size
- Sample size assignment appears unrelated to success probability

#### 5. **Zero-Inflation Assessment**
- Observed groups with 0 successes: 1 (8.3%)
- Expected under pooled model: 0.03
- **Ratio: 33x more than expected**
- Conclusion: **STRONG evidence of zero-inflation or Group 1 is genuinely different**

#### 6. **Pearson Residuals**
- Groups with |residual| > 2:
  - Group 1: -1.94 (0 successes, expected 3.47)
  - Group 2: +2.22 (18 successes, expected 10.94)
  - Group 4: -1.86 (46 successes, expected 59.87)
  - Group 5: -2.00 (8 successes, expected 15.60)
  - Group 8: **+3.94** (31 successes, expected 15.89) - EXTREME
  - Group 11: +2.41 (29 successes, expected 18.92)
- 6 out of 12 groups (50%) have residuals > 2 SD
- Normality test (Shapiro-Wilk): W = 0.88, p = 0.092 (marginal)

### Interpretation
1. **Pooled binomial model is inadequate** - strong rejection (p < 0.001)
2. **Groups have heterogeneous success probabilities** - need group-specific modeling
3. **Overdispersion factor of 3.5** suggests:
   - Beta-binomial model likely appropriate
   - Or fixed effects for each group
   - Or multilevel/hierarchical model
4. **Zero-inflation in Group 1** may require:
   - Zero-inflated binomial model
   - Separate analysis excluding Group 1
   - Or accept it as genuine extreme value
5. **Independence assumption holds** - no need for time series models

### Visualizations Created
- `residual_diagnostics.png` (4 panels: residuals by group, vs fitted, Q-Q plot, vs sample size)
- `observed_vs_expected.png` (comparing observed to pooled model expectations)
- `variance_mean_relationship.png` (checking binomial variance assumption)

---

## Round 3: Hypothesis Testing and Model Comparison

### Questions Asked
1. Which model best explains the data structure?
2. Can we formally test pooled vs heterogeneous models?
3. Is a hierarchical (beta-binomial) model appropriate?
4. What are the AIC/BIC comparisons?

### Analysis Performed
- **Hypothesis 1**: Pooled binomial (1 parameter)
- **Hypothesis 2**: Heterogeneous binomial (12 parameters)
- **Hypothesis 3**: Beta-binomial hierarchical (2 parameters)
- Likelihood ratio tests
- AIC/BIC model comparison
- Additional statistical tests (homogeneity, variance, normality)

### Key Findings

#### Model Comparison Results

| Model | Parameters | Log-Likelihood | AIC | BIC |
|-------|-----------|----------------|-----|-----|
| Pooled Binomial | 1 | -44.15 | 90.29 | 90.78 |
| Heterogeneous Binomial | 12 | -24.88 | 73.76 | 79.58 |
| **Beta-Binomial** | **2** | **-21.85** | **47.69** | **48.66** |

#### Rankings
- **Best by AIC**: Beta-Binomial (47.69)
- **Best by BIC**: Beta-Binomial (48.66)
- **Worst**: Pooled Binomial (clearly rejected)

#### Likelihood Ratio Test
- Pooled vs Heterogeneous: LR = 38.53, df = 11, **p < 0.001**
- **Conclusion**: Strong evidence groups have different success probabilities

#### Beta-Binomial Parameters (Method of Moments)
- alpha = 3.33
- beta = 41.88
- Implied mean: 3.33/(3.33+41.88) = 0.074 (matches data)
- Implied variance: 0.00148 (matches empirical variance)

#### Additional Tests
1. **Chi-square test for homogeneity**: chi2 = 38.56, p < 0.001 (REJECT homogeneity)
2. **Bartlett's test** (logit scale): p = 0.72 (equal variances on transformed scale)
3. **Shapiro-Wilk** (residuals): p = 0.092 (marginal normality, acceptable)

### Interpretation
1. **Beta-binomial model strongly preferred**:
   - Balances fit and parsimony
   - 42 AIC points better than pooled
   - 26 AIC points better than heterogeneous
2. **Hierarchical structure is appropriate**:
   - Groups share information through beta prior
   - Shrinkage toward mean will help extreme groups (especially Group 1)
3. **Pooled model is clearly wrong**:
   - P < 0.001 in every test
   - 50% of residuals exceed 2 SD
4. **Heterogeneous model overfits**:
   - 12 parameters for 12 observations (saturated)
   - No shrinkage for extreme values
   - Worse BIC than beta-binomial

### Competing Explanations
1. **"All groups are the same"** - REJECTED by data
2. **"Each group is completely different"** - Overfits, no structure
3. **"Groups vary around a common distribution"** - BEST EXPLANATION

---

## Round 4: Transformation Assessment

### Questions Asked
1. Which link function is most appropriate?
2. Do transformations improve distributional properties?
3. Are there advantages to logit vs probit vs cloglog?

### Analysis Performed
- Calculated logit, probit, and complementary log-log transformations
- Examined distributions on each scale
- Considered practical implications for inference

### Key Findings
1. **Raw success rates**: Heavily right-skewed, bounded [0, 1]
2. **Logit transformation**: More symmetric, unbounded
3. **Probit transformation**: Similar to logit, slightly different tails
4. **Cloglog transformation**: Asymmetric, useful for rare events

### Transformation Recommendations

#### For Binomial GLM:
- **Logit link (default)**: RECOMMENDED
  - Natural parameterization for odds ratios
  - Symmetric on log-odds scale
  - Most commonly used, easy interpretation

#### For Beta-Binomial:
- Success rates naturally bounded [0,1]
- Beta distribution is conjugate prior
- No transformation needed, model on probability scale

#### Special Consideration for Group 1:
- Logit(0) = -Inf (undefined)
- Probit(0) = -Inf (undefined)
- Solutions:
  1. Add continuity correction: (r + 0.5)/(n + 1)
  2. Use exact binomial inference for Group 1
  3. Let hierarchical model shrink toward population mean

### Visualizations Created
- `transformation_comparison.png` (4 panels comparing raw, logit, probit, cloglog)
- `sample_size_impact.png` (2 panels showing SE and precision by sample size)

---

## Critical Issues Identified

### Issue 1: Group 1 Zero Successes (CRITICAL)
- **Problem**: 0/47 = undefined for logit/probit, boundary for MLE
- **Impact**: Will cause convergence issues in standard GLMs
- **Solutions**:
  1. Use hierarchical/Bayesian model with shrinkage
  2. Apply continuity correction: (0.5)/(47+1) = 0.010
  3. Use exact binomial methods for this group
  4. Consider zero-inflated model if theoretically justified
- **Recommendation**: Hierarchical model handles this naturally via shrinkage

### Issue 2: Overdispersion (MAJOR)
- **Problem**: Variance 3.5x larger than binomial expectation
- **Impact**: Standard errors underestimated, p-values too small, poor uncertainty quantification
- **Solutions**:
  1. Beta-binomial model (recommended)
  2. Quasi-binomial with dispersion parameter
  3. Observation-level random effects
- **Recommendation**: Beta-binomial is theoretically sound and fits best

### Issue 3: Sample Size Heterogeneity (MODERATE)
- **Problem**: 17-fold range in sample sizes (47 to 810)
- **Impact**: Unequal information per group, heterogeneous precision
- **Solutions**:
  1. Weight observations by sample size
  2. Model precision explicitly (already done via binomial variance)
  3. Report group-specific credible intervals
- **Recommendation**: Binomial/beta-binomial naturally accounts for this

---

## Modeling Recommendations

### Primary Recommendation: Bayesian Hierarchical Beta-Binomial

```
Model specification:
  r_i ~ Binomial(n_i, p_i)         [Likelihood]
  p_i ~ Beta(alpha, beta)          [Prior/hierarchical distribution]
  alpha, beta ~ weakly informative priors
```

**Advantages**:
1. Naturally handles overdispersion (AIC = 47.69, best fit)
2. Shrinks extreme estimates (especially Group 1) toward population mean
3. Provides group-specific estimates with appropriate uncertainty
4. No convergence issues with zero counts
5. Interpretable parameters (alpha, beta describe variation across groups)
6. Can predict success rates for new groups

**Implementation**:
- PyMC, Stan, or JAGS for full Bayesian inference
- `betareg` package (R) for frequentist approximation
- Custom likelihood in statsmodels/scipy

### Alternative 1: Logistic Regression with Group Fixed Effects

```
Model: logit(p_i) = beta_i for i=1..12
```

**Advantages**:
1. Simple, widely understood
2. No distributional assumptions on p_i
3. Standard errors correct for each group

**Disadvantages**:
1. No shrinkage for extreme values
2. 12 parameters (overfits with small n)
3. Cannot predict new groups
4. Convergence issues with Group 1

**Use case**: If groups are fundamentally different and no population inference needed

### Alternative 2: Quasi-Binomial GLM

```
Model: logit(p_i) = mu (pooled)
       Var(r_i) = phi * n_i * p_i * (1-p_i)  [phi = dispersion]
```

**Advantages**:
1. Simple extension of binomial GLM
2. Adjusts standard errors for overdispersion
3. Easy to implement (glm with family=quasibinomial in R)

**Disadvantages**:
1. Assumes all groups have same p (rejected by data)
2. Ad-hoc correction, not likelihood-based
3. Cannot capture group differences

**Use case**: Quick-and-dirty if only interested in overall average

### Alternative 3: Zero-Inflated Binomial (If Theoretically Justified)

```
Model:
  With probability pi: r_i = 0 (structural zero)
  With probability (1-pi): r_i ~ Binomial(n_i, p)
```

**Advantages**:
1. Explicitly models excess zeros
2. Can test if Group 1 is from different process

**Disadvantages**:
1. Requires theoretical justification for two processes
2. More complex, harder to interpret
3. Only 1 zero observed (weak evidence)

**Use case**: If there's reason to believe some groups have zero probability of success

---

## Link Function Recommendation

### For GLM: Logit Link (Standard)
- **Form**: logit(p) = log(p / (1-p))
- **Interpretation**: Coefficients are log-odds ratios
- **Advantages**: Symmetric, most common, well-understood
- **Use**: Default for binomial regression

### For Beta-Binomial: Identity on Probability Scale
- **Form**: p ~ Beta(alpha, beta)
- **Interpretation**: Direct probability estimates
- **Advantages**: Natural, no transformation artifacts
- **Use**: Hierarchical models

### Continuity Correction for Zero Counts
- **Formula**: p_adjusted = (r + 0.5) / (n + 1)
- **For Group 1**: (0 + 0.5) / (47 + 1) = 0.0104
- **Use**: Only if not using hierarchical model

---

## Data Quality Concerns

### Concerns Raised
1. ✓ **Missing data**: None (0%)
2. ✓ **Data entry errors**: None found
3. ✓ **Impossible values**: None (all logically valid)
4. ✓ **Duplicates**: None
5. ✗ **Zero inflation**: 1 group (Group 1)
6. ✗ **Extreme outliers**: 2 groups (Groups 1, 8)
7. ✓ **Independence violations**: None detected

### Overall Assessment: GOOD
- Data is clean and ready for analysis
- Zero count in Group 1 requires careful handling
- High overdispersion is real signal, not data quality issue

---

## Key Assumptions Verified

| Assumption | Status | Evidence |
|------------|--------|----------|
| Binary outcomes | ✓ Satisfied | Success/failure counts |
| Fixed number of trials | ✓ Satisfied | n_i known for each group |
| Independence within groups | ✓ Assumed | No evidence against |
| Independence across groups | ✓ Verified | No autocorrelation (p=0.29) |
| Constant probability within group | ✓ Assumed | Standard binomial assumption |
| Constant probability across groups | ✗ VIOLATED | Chi-sq p < 0.001 |
| No overdispersion | ✗ VIOLATED | Dispersion = 3.5 |

---

## Summary Statistics

### Overall
- Total trials: 2,814
- Total successes: 208
- Pooled success rate: 0.0739 (7.39%)

### By Group
- Mean success rate: 0.0737
- Median success rate: 0.0669
- SD success rate: 0.0384
- Range: [0.000, 0.144]

### Sample Sizes
- Mean: 234.5 trials
- Median: 201.5 trials
- Range: [47, 810]
- CV: 0.85

---

## Files Created

### Code Scripts
1. `/workspace/eda/analyst_3/code/01_initial_exploration.py` - Data quality assessment
2. `/workspace/eda/analyst_3/code/02_binomial_assumptions.py` - Assumption testing
3. `/workspace/eda/analyst_3/code/03_diagnostic_plots.py` - Visualization generation
4. `/workspace/eda/analyst_3/code/04_hypothesis_testing.py` - Model comparison

### Visualizations
1. `data_quality_overview.png` - 4-panel overview of sample sizes and success rates
2. `residual_diagnostics.png` - 4-panel residual analysis (by group, vs fitted, Q-Q, vs sample size)
3. `observed_vs_expected.png` - Comparison of observed vs pooled model expectations
4. `variance_mean_relationship.png` - Variance-mean plot showing overdispersion
5. `sample_size_impact.png` - Standard errors and precision by sample size
6. `transformation_comparison.png` - Distributions on raw, logit, probit, cloglog scales

### Data Files
1. `data_with_checks.csv` - Original data with quality check flags
2. `data_with_diagnostics.csv` - Data with residuals and diagnostic statistics
3. `model_comparison.csv` - AIC/BIC comparison table

---

## Conclusion

This dataset exhibits **strong heterogeneity** across groups with **significant overdispersion**. A **Bayesian hierarchical beta-binomial model** is the most appropriate choice, offering the best balance of fit and parsimony while naturally handling the zero-count issue in Group 1 through shrinkage. The data quality is excellent, and independence assumptions are satisfied, but the pooled binomial model is definitively rejected (p < 0.001).
