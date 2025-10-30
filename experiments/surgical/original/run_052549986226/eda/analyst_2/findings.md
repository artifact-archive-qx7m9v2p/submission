# EDA Findings: Hierarchical Structure and Group-Level Patterns

**Analyst 2 - Focus: Hierarchical modeling justification and group heterogeneity**

---

## Executive Summary

This analysis provides **overwhelming evidence for hierarchical modeling** of the group-level success rate data. Key findings:

- **5x overdispersion** relative to binomial model (p < 0.001)
- **ICC = 72.7%**: Most variance is between-group, not within-group sampling
- **No evidence for discrete clusters**: Groups follow continuous distribution
- **Strong shrinkage potential**: Average 85.6% toward pooled mean
- **Recommendation**: Use hierarchical/multilevel binomial model with partial pooling

---

## 1. Data Structure

### Sample Composition
- **12 groups** with highly variable sample sizes
- **Total observations**: 2,814 trials, 208 successes
- **Sample size range**: 47 to 810 trials (17.2:1 ratio)
- **Pooled success rate**: 7.39%

### Group-Level Variation
- **Mean success rate**: 7.37% (SD = 3.84%)
- **Coefficient of variation**: 0.52
- **Range**: 0% (Group 1) to 14.4% (Group 8)
- **Median**: 6.69%

**Visualization**: `caterpillar_plot_sorted.png` shows success rates with 95% confidence intervals, sorted by rate. Point sizes reflect sample size.

---

## 2. Evidence for Hierarchical Structure

### 2.1 Overdispersion Test

**Question**: Is between-group variance greater than expected from binomial sampling alone?

**Method**: Compare observed variance in success rates to expected variance under binomial model.

**Results**:
- Observed variance: 0.00148
- Expected variance (binomial): 0.00029
- **Variance ratio: 5.06**

**Interpretation**: The observed variance is **5x larger** than we would expect if all groups shared the same underlying success probability. This is strong evidence for genuine heterogeneity.

### 2.2 Chi-Square Test for Homogeneity

**Null hypothesis**: All groups have identical success probability

**Results**:
- Chi-square statistic: 38.56
- Degrees of freedom: 11
- **p-value: 0.000063**

**Conclusion**: REJECT null hypothesis at any reasonable significance level. Groups do NOT share the same underlying rate.

### 2.3 Variance Decomposition

**Method**: DerSimonian-Laird estimator for between-group variance component.

**Results**:
- Between-group variance (tau-squared): 0.000778
- Between-group SD (tau): 0.0279
- This represents ~2.8 percentage points of SD in true group rates

**Visualization**: `pooling_comparison.png` (left panel) shows three modeling strategies:
- **No pooling**: Uses only group-specific data (blue circles)
- **Complete pooling**: Uses only overall rate (red squares)
- **Partial pooling**: Hierarchical compromise (green triangles)

### 2.4 Intraclass Correlation Coefficient (ICC)

**Definition**: Proportion of total variance attributable to between-group differences.

**Result**: **ICC = 0.727 (72.7%)**

**Interpretation**:
- 72.7% of variance is due to genuine group differences
- 27.3% is due to within-group sampling variation
- **ICC > 0.1 threshold**: Hierarchical modeling highly beneficial

**Practical meaning**: If we randomly sample two observations from the same group, they are much more similar than two observations from different groups.

---

## 3. Shrinkage Analysis

### 3.1 Shrinkage Factors by Group

**Method**: Calculate optimal shrinkage factor λ = n/(n + 1/τ²) for each group.

**Results** (selected groups):

| Group | n_trials | Observed Rate | Shrinkage Factor | Shrinkage % | Partial Pooled Rate |
|-------|----------|---------------|------------------|-------------|---------------------|
| 1     | 47       | 0.000         | 0.035            | 96.5%       | 0.071               |
| 4     | 810      | 0.057         | 0.386            | 61.4%       | 0.067               |
| 8     | 215      | 0.144         | 0.143            | 85.7%       | 0.084               |

**Average shrinkage**: 85.6% toward pooled mean

**Interpretation**:
- **Group 1** (0/47): Extreme observed rate (0%) shrinks heavily to 7.1%
- **Group 4** (46/810): Large sample shrinks less (from 5.7% to 6.7%)
- **Group 8** (31/215): High observed rate (14.4%) shrinks moderately to 8.4%

**Visualization**: `shrinkage_visualization.png` shows arrows from observed to pooled rates, and `pooling_comparison.png` (right panel) shows shrinkage factor increasing with sample size.

### 3.2 Benefits of Shrinkage

1. **Stabilizes small-sample estimates**: Group 1's 0% becomes more plausible 7.1%
2. **Reduces overfitting**: Extreme rates are pulled toward population mean
3. **Improves prediction**: Partial pooling typically outperforms both extremes
4. **Handles imbalance**: Automatically weights groups by precision

---

## 4. Clustering vs Continuous Variation

### 4.1 Hypothesis Testing

**H1**: Groups cluster into distinct subpopulations (e.g., high/low performers)
**H2**: Groups drawn from continuous distribution
**H3**: Groups are homogeneous (all from same population)

**Already rejected H3** (chi-square test, ICC analysis). Now testing H1 vs H2.

### 4.2 Evidence for Continuous Distribution

**Normality test**:
- Shapiro-Wilk statistic: 0.94, p = 0.496
- Cannot reject normality
- Q-Q plot shows good fit (see `distribution_analysis.png`, bottom-left panel)

**Gap analysis**:
- Mean gap between consecutive rates: 0.0131
- Only 1 large gap detected: Between Group 1 (0%) and others
- CV of gaps: 0.948 (high variability, but no clear bimodality)

**CI overlap**:
- 84.8% of group pairs have overlapping confidence intervals
- Only 15.2% are non-overlapping

**Dendrogram** (see `clustering_analysis.png`, right panel):
- Shows gradual, hierarchical merging
- No clear "step" indicating discrete clusters

**Conclusion**: **H2 supported** - groups appear drawn from continuous distribution, likely normal.

### 4.3 Median Split Analysis

For illustration, we split at median (6.69%):
- **Lower cluster** (6 groups): Mean = 4.8%, SD = 2.6%
- **Upper cluster** (6 groups): Mean = 9.9%, SD = 3.2%

While this creates two groups, there's **no natural gap** in the data to justify this split. It's an artificial dichotomization.

**Visualization**: `clustering_analysis.png` (left panel) shows the split, but substantial overlap remains.

---

## 5. Regression to the Mean

### 5.1 Sample Size Effect

**Question**: Do smaller samples show more extreme rates?

**Method**: Spearman correlation between sample size and |rate - pooled rate|.

**Result**:
- Correlation: -0.042
- p-value: 0.897
- **Not significant**

**Interpretation**: This is **unexpected**. Typically, smaller samples show more extreme rates purely by chance. The lack of correlation suggests:
1. Between-group heterogeneity is genuine (not just sampling noise)
2. Large and small groups are similarly variable
3. Further supports hierarchical modeling

**Visualization**: `regression_to_mean.png` shows nearly flat trend line.

---

## 6. Data Quality Considerations

### 6.1 Edge Cases

**Group 1: Zero successes**
- 0/47 trials (0% success rate)
- 95% CI: [0%, 7.6%]
- Z-score: > 2 (outlier)
- **Recommendation**: Hierarchical model will naturally shrink this toward ~7%, which is more defensible than reporting 0%

**Group 4: Largest sample**
- 46/810 trials (5.7% success rate)
- Very narrow 95% CI: [4.3%, 7.5%]
- Will experience least shrinkage (only 61.4%)
- Most influential for estimating population mean

### 6.2 Confidence Interval Quality

- Wilson score intervals used (appropriate for binomial proportions)
- Widths range from 3.2% (Group 4, n=810) to 11.2% (Group 10, n=97)
- 67% of groups have CIs including pooled rate
- 3 groups significantly above pooled: Groups 2, 8, 11
- 1 group significantly below pooled: Group 5

**Visualization**: All three caterpillar plots show CI width inversely related to sample size (as expected).

---

## 7. Modeling Recommendations

### 7.1 Strongly Recommended: Hierarchical Binomial Model

**Specification**:
```
y_i ~ Binomial(n_i, p_i)           # Likelihood for group i
logit(p_i) = α + θ_i               # Group-specific effect
θ_i ~ Normal(0, τ)                 # Hierarchical prior
α ~ Normal(logit(0.074), 1)        # Population mean (weakly informative)
τ ~ HalfNormal(1)                  # Between-group SD (weakly informative)
```

**Justification**:
1. ICC = 72.7% → substantial between-group variance
2. Normality test passed → normal prior appropriate
3. 5x overdispersion → need to model group effects
4. Unbalanced design → partial pooling handles this naturally
5. Small samples → shrinkage improves estimates

**Implementation**: Stan, PyMC, brms (R), or rstanarm

### 7.2 Alternative: Beta-Binomial Model

**When to consider**: If overdispersion is the primary concern and you don't care about estimating group-specific effects.

**Specification**:
```
y_i ~ BetaBinomial(n_i, α, β)
```

**Trade-off**: Simpler but doesn't provide group-specific estimates (no partial pooling).

### 7.3 NOT Recommended: Fixed Effects

**Why not**:
1. Ignores shared structure across groups
2. Unstable for small samples (especially Group 1)
3. No shrinkage → prone to overfitting
4. 12 parameters instead of 2-3 (less parsimonious)

**Exception**: If groups represent fundamentally different populations with no shared structure (not supported by data).

---

## 8. Sensitivity Analyses to Conduct

### 8.1 Prior Sensitivity

- **Prior on τ**: Try HalfNormal(0.5), HalfNormal(2) to assess sensitivity
- **Prior on α**: Should be robust, but check weakly vs moderately informative

### 8.2 Outlier Impact

- **Fit with/without Group 1** to assess influence
- **Fit with/without Group 4** (largest sample) to check robustness

### 8.3 Model Comparison

Compare via:
- **LOO-CV** (leave-one-out cross-validation)
- **WAIC** (Widely Applicable Information Criterion)
- **Posterior predictive checks**

Models to compare:
1. Hierarchical binomial (recommended)
2. Beta-binomial
3. Pooled binomial (null model)
4. Fixed effects (for reference)

---

## 9. Key Visualizations Summary

All visualizations saved to `/workspace/eda/analyst_2/visualizations/`:

1. **caterpillar_plot_sorted.png**: Core visualization showing group rates with CIs, sorted by rate
2. **caterpillar_plot_by_id.png**: Same but in original group order (no spatial pattern)
3. **shrinkage_visualization.png**: Arrows showing shrinkage from observed to pooled
4. **pooling_comparison.png**: Three strategies compared; shrinkage vs sample size
5. **distribution_analysis.png**: 4-panel distribution diagnostics (histogram, KDE, Q-Q, bars)
6. **clustering_analysis.png**: Clustering attempts (median split, dendrogram)
7. **regression_to_mean.png**: Sample size vs deviation from pooled (no correlation)

---

## 10. Quantitative Summary Table

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Between-group variance (τ²)** | 0.000778 | Moderate heterogeneity |
| **Between-group SD (τ)** | 0.0279 | ~2.8 percentage points |
| **ICC** | 0.727 | 72.7% variance is between-group |
| **Variance ratio** | 5.06 | 5x overdispersion |
| **Chi-square p-value** | 0.000063 | Strong heterogeneity |
| **Shapiro-Wilk p-value** | 0.496 | Consistent with normality |
| **CI overlap %** | 84.8% | High overlap (continuous) |
| **Average shrinkage** | 85.6% | Substantial information pooling |
| **Sample size CV** | 0.92 | Highly unbalanced |

---

## 11. Conclusions

### Main Findings

1. **Hierarchical structure is strongly justified**: ICC = 72.7%, 5x overdispersion, p < 0.001
2. **Groups follow continuous distribution**: Not discrete clusters, normal distribution plausible
3. **Substantial shrinkage recommended**: Average 85.6% toward pooled mean
4. **No pooling would overfit**: Especially for small groups (Group 1: 0/47)
5. **Complete pooling would underfit**: Ignores genuine heterogeneity (chi-square p < 0.001)

### Modeling Implications

- **Use hierarchical/multilevel binomial model** with partial pooling
- **Normal prior for group effects** supported by data
- **Expect substantial shrinkage** for small-sample groups
- **Large-sample groups** (e.g., Group 4) will retain more of observed rate
- **Model naturally handles** unbalanced design

### Data Quality Issues

- **Group 1** (0/47 successes) is an outlier but will benefit from shrinkage
- **No missing data** or other quality concerns
- **Confidence intervals** appropriately wide for small samples

### Next Steps for Modeling

1. Fit hierarchical binomial model (e.g., in Stan/PyMC)
2. Check prior sensitivity (especially on τ)
3. Perform posterior predictive checks
4. Compare via LOO-CV to simpler models
5. Report both observed and partial-pooled estimates
6. Visualize posterior distributions of group effects

---

## Appendix: Reproducible Code

All analysis code available in `/workspace/eda/analyst_2/code/`:

- `01_initial_exploration.py`: Basic statistics, CI calculations
- `02_caterpillar_plots.py`: Caterpillar plots, CI overlap analysis
- `03_variance_decomposition.py`: ICC, variance components, shrinkage
- `04_clustering_analysis.py`: Distribution tests, clustering, regression to mean

Intermediate data files:
- `group_data_with_ci.csv`: Original data + confidence intervals
- `hierarchical_analysis.csv`: + shrinkage factors and partial pooled rates
- `clustering_analysis.csv`: + clustering assignments and distances
