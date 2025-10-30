# EDA Findings Report - Analyst 2: Patterns, Structure, and Relationships

**Dataset**: Binomial outcome data with 12 groups
**Total Observations**: 2,814
**Total Events**: 208
**Overall Rate**: 7.39% (95% CI: [6.48%, 8.42%])

---

## Executive Summary

This EDA focused on identifying patterns, quantifying relationships, and evaluating pooling strategies for 12 groups with binomial outcomes. Key findings:

1. **Strong evidence of heterogeneity**: Groups differ significantly (p < 0.0001, φ = 3.5)
2. **No systematic patterns**: No sequential trends or sample size biases detected
3. **Three outlier groups** identified with elevated rates (2, 8, 11)
4. **One zero-event group** requiring special handling (Group 1)
5. **Hierarchical pooling strongly recommended** based on variance decomposition (ICC = 0.66)

---

## 1. Sequential Pattern Analysis

**Question**: Is there any ordering, temporal, or spatial pattern in the proportions?

**Visualization**: `01_sequential_patterns.png` (3-panel figure)

### Key Findings

**No significant sequential trend detected** (Spearman ρ = 0.40, p = 0.20)
- Proportions fluctuate randomly across group indices (range: 0% to 14.4%)
- Linear regression shows weak relationship (R² = 0.14, p = 0.23)
- Runs test consistent with random ordering (5 runs observed, ~6 expected)

**Three distinct peaks identified**:
- Group 2: 12.2% (18/148 events)
- Group 8: 14.4% (31/215 events) - **highest rate**
- Group 11: 11.3% (29/256 events)

**Sample size variation**:
- Highly variable (CV = 0.85): ranges from 47 (Group 1) to 810 (Group 4)
- Group 4 dominates dataset (29% of total observations)
- Three groups have n < 100: Groups 1, 3, 10

**Uncertainty patterns**:
- Confidence interval widths inversely proportional to √n (as expected)
- Widest CI: Group 1 [0.00, 0.08] - zero events with small sample
- Narrowest CI: Group 4 [0.04, 0.07] - large sample provides precision

### Interpretation
Group ordering appears arbitrary with no temporal/spatial structure. This suggests:
- Groups can be treated as exchangeable for modeling purposes
- No need to model sequential dependencies
- But substantial heterogeneity exists (see Section 3)

---

## 2. Sample Size vs Proportion Relationship

**Question**: Do larger samples show systematically different rates? (Selection bias?)

**Visualization**: `02_sample_size_relationships.png` (4-panel figure)

### Key Findings

**No correlation detected** (Pearson r = 0.006, p = 0.99)
- Near-zero relationship between sample size and proportion
- Small-n groups (≤202): mean proportion = 6.6%
- Large-n groups (>202): mean proportion = 8.1%
- Mann-Whitney U test: p = 0.82 (NOT significant)

**Uncertainty follows theoretical relationship**:
- Observed CI widths closely match 1/√n expectation
- On log scale, nearly perfect linear relationship
- No evidence of extra-binomial variation in uncertainty structure

**Group 8 identified as outlier**:
- Proportion = 14.4% (highest)
- Sample size = 215 (moderate, not extreme)
- Standardized residual z = 3.94 (highly significant)

### Interpretation
No selection bias detected. Sample size variation appears random and unrelated to outcome rates. This is positive for modeling - it means:
- No need to adjust for sample size as predictor
- Heterogeneity is not driven by sampling strategy
- But does not rule out true group differences (see Section 3)

---

## 3. Heterogeneity and Variance Decomposition

**Question**: Are groups exchangeable or do they have true differences?

**Visualizations**:
- `03_uncertainty_quantification.png` (forest plot and variance analysis)
- `05_pooling_considerations.png` (pooling strategies comparison)

### Statistical Tests

**Chi-square test for homogeneity**: REJECTED (χ² = 38.56, df = 11, p < 0.0001)
- Strong evidence groups are NOT homogeneous
- Likelihood ratio test confirms: LR = 38.53, p < 0.0001

**Overdispersion analysis**:
- Overdispersion parameter: **φ = 3.51** (severe)
- Expected φ = 1.0 under binomial assumption
- 3.5× more variation than expected from sampling alone

**Variance decomposition** (excluding zero-event group):
- Within-group variance: 0.000290 (expected under binomial)
- Total observed variance: 0.000859
- Between-group variance: 0.000569
- **Intraclass correlation (ICC) = 0.662**
  - **66% of variation is between groups**
  - 34% is within-group sampling variation

**I² statistic**: **71.5%** (moderate-to-high heterogeneity)
- I² < 25%: low heterogeneity
- I² 25-75%: moderate heterogeneity
- I² > 75%: high heterogeneity
- Our value of 71.5% indicates substantial true differences

### Outlier Detection

**Three groups exceed |z| > 2 threshold**:
1. **Group 8**: z = 3.94, p = 0.0001 (observed 31, expected 15.9)
2. **Group 11**: z = 2.41, p = 0.016 (observed 29, expected 18.9)
3. **Group 2**: z = 2.22, p = 0.026 (observed 18, expected 10.9)

**Borderline outliers**:
- Group 5: z = -2.00, p = 0.045 (observed 8, expected 15.6) - below average
- Group 1: z = -1.94, p = 0.052 (zero events)

### Interpretation
Groups exhibit substantial true differences, not just sampling variation. This has major implications:
- **Complete pooling inappropriate** (would ignore 66% of variance)
- **No pooling risky** (overfits small groups, unstable for Group 1)
- **Partial pooling optimal** (balances individual and group information)

---

## 4. Rare Events and Zero Inflation

**Question**: How should we handle groups with zero or very low event counts?

**Visualization**: `04_rare_events_analysis.png` (4-panel analysis)

### Group 1: Zero Events (0/47)

**Statistical assessment**:
- Under pooled model (p = 0.0739), P(r = 0 | n = 47) ≈ 0.025 (2.5%)
- Unusual but not extremely unlikely (1 in 40 chance)
- Standardized residual: z = -1.94 (borderline, p = 0.052)

**Implications for modeling**:
- Maximum likelihood estimate = 0.0 (degenerate)
- Requires continuity correction or Bayesian prior
- Wide uncertainty: 95% CI [0.00, 0.08]
- Benefits most from hierarchical pooling

### Other Rare Event Groups

Only **1 group with r ≤ 5** (Group 1 with r = 0)
- All other groups have r ≥ 8
- Sufficient events for stable estimation
- No systematic zero-inflation problem

### Expected vs Observed Events

Under pooled model (p = 0.0739):
- **Large deviations**:
  - Group 8: +15.1 events (95% excess)
  - Group 11: +10.1 events (53% excess)
  - Group 4: -13.8 events (23% deficit)

- **Moderate deviations**:
  - Group 2: +7.1 events (65% excess)
  - Group 5: -7.6 events (49% deficit)

### Interpretation
Zero-event in Group 1 is noteworthy but not alarming (P = 2.5%). More concerning are the large positive deviations in Groups 2, 8, 11, suggesting these groups may represent different subpopulation or have different risk factors.

---

## 5. Pooling Strategy Evaluation

**Question**: When is complete vs no pooling vs partial pooling appropriate?

**Visualization**: `05_pooling_considerations.png` (comparison of three strategies)

### Complete Pooling (All groups = 7.39%)

**NOT RECOMMENDED**
- Ignores all between-group variation (ICC = 66%)
- Rejected by homogeneity test (p < 0.0001)
- Severely underestimates uncertainty for outlier groups
- Overdispersion φ = 3.5 indicates poor fit

**When appropriate**: Only if groups truly identical (our data: NO)

### No Pooling (Saturated Model - 12 parameters)

**NOT RECOMMENDED**
- Overfits to small-sample groups
- Group 1 estimate = 0.0 (unstable, degenerate)
- High variance in estimates (Var = 0.0015)
- 12 parameters for 12 groups (no degrees of freedom for testing)
- Does not borrow strength across groups

**When appropriate**: Very large samples per group OR known group differences (our data: NO)

### Partial Pooling (Hierarchical Model)

**STRONGLY RECOMMENDED**
- Shrinks extreme estimates toward group mean
- Shrinkage proportional to uncertainty
- Stabilizes small-sample estimates
- Variance reduced to 0.0008 (vs 0.0015 for no pooling)
- Effective sample sizes increase for small groups

**Shrinkage examples** (empirical Bayes):
- Group 1: 0.00 → 0.04 (large shrinkage, zero events)
- Group 8: 0.144 → 0.12 (moderate shrinkage, large sample)
- Group 4: 0.057 → 0.068 (toward mean, very large sample provides resistance)

**When appropriate**: Heterogeneous groups with variable sample sizes (our data: YES)

### Effective Sample Sizes

Partial pooling increases effective n by borrowing strength:
- Group 1: n = 47 → eff_n ≈ 150 (3× increase)
- Group 3: n = 119 → eff_n ≈ 250 (2× increase)
- Group 4: n = 810 → eff_n ≈ 900 (minimal, already large)

---

## 6. Confidence Interval Analysis

**Visualization**: `03_uncertainty_quantification.png` (Panel 1: Forest plot)

### Wilson Score Confidence Intervals (95%)

**Zero-event group**:
- Group 1: [0.00, 0.08] - widest interval, includes zero

**Small-sample groups** (n < 150):
- Group 3 (n=119): [0.03, 0.13] - width = 0.10
- Group 7 (n=148): [0.03, 0.11] - width = 0.08
- Group 10 (n=97): [0.04, 0.15] - width = 0.11

**Large-sample groups** (n > 200):
- Group 4 (n=810): [0.04, 0.07] - width = 0.03 (most precise)
- Group 8 (n=215): [0.10, 0.19] - width = 0.09
- Group 11 (n=256): [0.08, 0.15] - width = 0.07

### Precision Analysis

**Highest precision** (inverse variance):
1. Group 4: 1/SE² ≈ 13,000 (n=810)
2. Group 11: 1/SE² ≈ 2,500 (n=256)
3. Group 8: 1/SE² ≈ 1,800 (n=215)

**Lowest precision**:
1. Group 10: 1/SE² ≈ 650 (n=97, small sample)
2. Group 3: 1/SE² ≈ 900 (n=119)
3. Group 1: Cannot compute (zero events)

### Interpretation
Precision varies 20-fold across groups. Small-sample groups (1, 3, 10) have 2-3× wider confidence intervals than large-sample groups. This justifies partial pooling to improve precision for uncertain estimates.

---

## Modeling Recommendations

### Primary Recommendation: Hierarchical Bayesian or Random Effects Model

**Model 1: Beta-Binomial (accounts for overdispersion)**
```
r_i ~ BetaBinomial(n_i, α, β)
α, β parametrize the beta distribution
Overdispersion naturally accommodated
```

**Advantages**:
- Explicitly models overdispersion (φ = 3.5)
- More flexible than binomial
- Closed-form conjugate updates

**Model 2: Random Effects Logistic Regression**
```
r_i ~ Binomial(n_i, p_i)
logit(p_i) = μ + α_i
α_i ~ Normal(0, τ²)
```

**Advantages**:
- Estimates between-group variance τ²
- Automatic shrinkage via BLUP/Empirical Bayes
- Standard software available (lme4, nlme)

**Model 3: Bayesian Hierarchical Model**
```
r_i ~ Binomial(n_i, p_i)
logit(p_i) ~ Normal(μ, τ²)
μ ~ Normal(0, 10)  [weakly informative]
τ ~ HalfCauchy(0, 2.5)  [weakly informative]
```

**Advantages**:
- Full uncertainty quantification
- Naturally handles zero-event group via prior
- Posterior predictive checks for model assessment
- Can incorporate covariates easily

### Alternative: Two-Component Finite Mixture Model

**Rationale**: Three groups (2, 8, 11) clearly separated from others

**Model**:
```
Component 1 (low risk): p_1 ≈ 0.06-0.07 (9 groups)
Component 2 (high risk): p_2 ≈ 0.12-0.14 (3 groups)
Mixture weight: π ≈ 0.25
```

**When to consider**:
- If groups represent distinct subpopulations
- If covariates can predict component membership
- If prediction for new groups needed (classify as high/low risk)

### Model Selection Criteria

**Compare via**:
1. AIC/BIC (penalized likelihood)
2. Cross-validation (predictive accuracy)
3. Posterior predictive checks (model adequacy)
4. DIC/WAIC (Bayesian model comparison)

**Expected ranking**:
1. Hierarchical model (best balance)
2. Mixture model (if truly bimodal)
3. No pooling (overfits)
4. Complete pooling (underfits)

---

## Data Quality Assessment

### Strengths
- No missing values
- Clean, well-structured data
- Calculations verified (proportion = r/n, failures = n-r)
- Reasonable total sample size (n = 2,814)
- Good range of sample sizes per group (47 to 810)

### Concerns and Recommendations

**1. Zero-event group (Group 1)**
- Recommendation: Verify data entry (truly zero or missing?)
- If valid: Use Bayesian prior or continuity correction
- Consider sensitivity analysis including/excluding

**2. Outlier groups (2, 8, 11)**
- Recommendation: Investigate what differentiates these groups
- Possible explanations:
  - Different geographic regions?
  - Different time periods?
  - Different populations (age, risk factors)?
  - Measurement differences?
- Consider collecting covariates to explain heterogeneity

**3. Sample size imbalance**
- 17-fold range (47 to 810)
- Group 4 represents 29% of data
- Recommendation: Check if intentional or convenience sampling

**4. Binomial assumption**
- Assumes independence within groups
- Recommendation: Verify no clustering (e.g., repeated measures, families)
- If clustering present, need multilevel model

### Pre-modeling Checklist

- [ ] Verify Group 1 truly has zero events
- [ ] Investigate what Groups 2, 8, 11 have in common
- [ ] Confirm group labels have substantive meaning
- [ ] Check for within-group clustering/dependencies
- [ ] Consider collecting group-level covariates
- [ ] Validate data entry for outlier groups

---

## Summary: Robust vs Tentative Findings

### Robust Findings (High Confidence)

**Groups are heterogeneous** ✓
- Multiple tests confirm (Chi-square, LR, variance decomposition)
- p < 0.0001 across all tests
- Overdispersion φ = 3.5
- ICC = 0.66, I² = 71.5%

**No sample size bias** ✓
- Correlation r = 0.006, p = 0.99
- Mann-Whitney U test p = 0.82
- Uncertainty follows theoretical relationship

**No sequential trend** ✓
- Spearman ρ = 0.40, p = 0.20
- Runs test consistent with randomness
- No evidence of temporal/spatial structure

**Three groups significantly elevated** ✓
- Groups 2, 8, 11 all have z > 2.0
- Robustly detected across multiple methods
- Not artifacts of sample size

**Partial pooling recommended** ✓
- Strongly supported by variance decomposition
- Between-group variance > within-group variance
- Multiple model options available

### Tentative Findings (Require Further Investigation)

**Group 1 zero-event**
- P = 2.5% under null (unusual but not extreme)
- Could be sampling variability OR true low risk
- Needs verification and sensitivity analysis

**Mixture model interpretation**
- Visually plausible (bimodal pattern)
- But formal model comparison needed
- May or may not improve fit over hierarchical model

**Outlier groups represent subpopulation**
- Groups 2, 8, 11 clearly different
- But mechanism unknown (biology? measurement? population?)
- External validation needed

---

## Files Generated

### Code (`/workspace/eda/analyst_2/code/`)
- `01_initial_exploration.py` - Data structure and quality checks
- `02_comprehensive_visualizations.py` - All visualizations
- `03_statistical_tests.py` - Hypothesis testing

### Visualizations (`/workspace/eda/analyst_2/visualizations/`)
- `01_sequential_patterns.png` - 3-panel: proportions, sample sizes, CI widths over groups
- `02_sample_size_relationships.png` - 4-panel: n vs p relationships, uncertainty analysis
- `03_uncertainty_quantification.png` - 4-panel: forest plot, precision, observed vs expected
- `04_rare_events_analysis.png` - 4-panel: rare events, zero inflation, residuals
- `05_pooling_considerations.png` - 4-panel: pooling strategy comparison

### Reports
- `findings.md` - This comprehensive report
- `eda_log.md` - Detailed exploration process and intermediate findings

---

## Conclusion

This dataset exhibits **strong heterogeneity** (ICC = 66%, φ = 3.5, p < 0.0001) but **no systematic biases** (no sequential trends, no sample size effects). Three groups (2, 8, 11) show significantly elevated rates, while one group (1) has zero events requiring special handling.

**Primary recommendation**: Fit a **hierarchical Bayesian or random effects model** to appropriately balance individual group estimates with shared information. This approach will:
- Stabilize estimates for small-sample groups (especially Group 1)
- Appropriately shrink outlier estimates (Groups 2, 8, 11)
- Provide honest uncertainty quantification
- Allow for future predictions on new groups

**Alternative**: If groups represent distinct subpopulations, consider a **two-component mixture model** to formally test for bimodality.

**Do NOT use**: Complete pooling (rejected by tests) or no pooling (overfits small groups).

Further investigation needed to understand what differentiates high-rate groups (2, 8, 11) from others - this could lead to covariate inclusion and better predictive models.
