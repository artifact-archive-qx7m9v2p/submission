# Exploratory Data Analysis Log

## Dataset Overview
- **File**: `/workspace/data/data.csv`
- **Structure**: 12 observations of binomial trials
- **Columns**: trial_id, n (sample size), r (successes), proportion (r/n)

---

## Round 1: Initial Data Exploration

### Data Quality Assessment
**Script**: `01_initial_exploration.py`

**Findings**:
1. **Data Integrity**: All checks passed
   - No missing values
   - No invalid entries (r <= n for all observations)
   - Proportions correctly calculated (max error: 8.33e-17)
   - No negative values

2. **Sample Size Characteristics**:
   - Range: [47, 810]
   - Mean: 234.5
   - Variance: 39,359.18
   - CV: 0.846 (high variability in sample sizes)
   - Total trials: 2,814
   - One very large sample (trial 4, n=810) dominates

3. **Proportion Characteristics**:
   - Range: [0.0000, 0.1442]
   - Mean: 0.0737
   - Median: 0.0669
   - Variance: 0.00148
   - Std: 0.0384
   - Pooled estimate: 0.0739

4. **Notable Observations**:
   - Trial 1: Zero successes (0/47) - unusual but not impossible
   - Trial 8: Highest proportion (0.1442, 31/215)
   - Trial 4: Largest sample (810 trials)

**Questions Raised**:
- Is the variance in proportions larger than expected under binomial model?
- Are trials 1 and 8 true outliers or natural variation?
- Does sample size affect observed proportions?

---

## Round 2: Overdispersion Analysis

### Testing Binomial Model Assumptions
**Script**: `02_overdispersion_analysis.py`

**Key Findings**:

1. **Chi-Square Goodness of Fit Test**:
   - Test statistic: 38.56
   - Degrees of freedom: 11
   - P-value: 0.000063
   - **Result**: REJECT null hypothesis of constant probability

2. **Dispersion Parameter**:
   - Calculated: 3.51
   - Interpretation: Variance is 3.5x larger than expected under binomial model
   - **Strong evidence of overdispersion**

3. **Standardized Residuals**:
   - Expected variance: 1.0
   - Observed variance: 3.50
   - Mean: 0.077 (close to expected 0)
   - Distribution shows more extreme values than expected

4. **Outliers (|z| > 2)**:
   - Trial 2: z = 2.22 (p = 0.1216)
   - Trial 8: z = 3.94 (p = 0.1442) - **most extreme**
   - Trial 11: z = 2.41 (p = 0.1133)

5. **Interpretation**:
   - Simple binomial model with constant p is inadequate
   - Beta-binomial or mixture model likely needed
   - Suggests underlying probability varies across trials

**Questions for Next Round**:
- Are there distinct probability regimes?
- Is there a temporal or systematic pattern?
- Can we identify subgroups?

---

## Round 3: Pattern Detection

### Systematic Pattern Analysis
**Script**: `04_pattern_analysis.py`

**Hypothesis Tests Conducted**:

1. **Temporal Trend** (H0: No trend across trial_id):
   - Pearson r = 0.371, p = 0.235
   - Spearman rho = 0.399, p = 0.199
   - **Result**: No significant temporal trend
   - Conclusion: Trial order doesn't explain variation

2. **Sample Size Effect** (H0: No relationship between n and proportion):
   - Pearson r = 0.006, p = 0.986
   - Spearman rho = 0.088, p = 0.787
   - **Result**: No significant relationship
   - Conclusion: Sample size bias not present; supports homogeneity across samples

3. **Group Structure**:
   - Median split creates significantly different groups (p = 0.012)
   - Below median: mean p = 0.048 (trials: 1, 4, 5, 6, 7, 12)
   - Above median: mean p = 0.099 (trials: 2, 3, 8, 9, 10, 11)
   - Tercile split shows gradual progression:
     - Low: 0.039 (trials 1, 4, 5, 7)
     - Medium: 0.067 (trials 3, 6, 9, 12)
     - High: 0.115 (trials 2, 8, 10, 11)

4. **Gap Analysis**:
   - Largest gap: 0.038 between 0.000 and 0.038
   - Second largest: 0.031 between 0.082 and 0.113
   - Suggests possible separation points but not definitive

5. **Outliers**:
   - Trial 1: 0.000 (unusual - only 2.7% probability under pooled model)
   - Trial 8: 0.144 (high but within realm of possibility)

6. **Randomness Test**:
   - Runs test: p = 0.226
   - **Result**: Sequence appears random
   - No evidence of systematic ordering

7. **Variance Homogeneity**:
   - Large vs small samples have similar variance
   - Levene test: p = 0.853
   - **Result**: Variance is homogeneous across sample sizes

**Key Insights**:
- No temporal or sample-size-based patterns
- Evidence for distinct probability groups
- Variance is consistent across subgroups
- Overdispersion is real, not artifact of heterogeneous variance

---

## Round 4: Visualization Analysis

### Visual Exploration
**Script**: `03_visualization.py`

**Plots Created**:

1. **`sample_size_distribution.png`**:
   - Right-skewed distribution
   - Most samples between 100-250
   - One outlier at 810 (trial 4)

2. **`proportion_distribution.png`**:
   - Roughly uniform/slightly bimodal
   - Spans 0 to 0.14
   - Pooled estimate (0.074) near center

3. **`proportion_vs_trial.png`**:
   - No clear temporal trend (confirms statistical test)
   - High variability throughout
   - Several points exceed binomial 95% CI

4. **`proportion_vs_sample_size.png`**:
   - No relationship between n and p (confirms test)
   - Scatter appears random
   - Validates exchangeability assumption

5. **`standardized_residuals.png`**:
   - Shows three observations beyond ±2 SD
   - Distribution wider than expected N(0,1)
   - Visual confirmation of overdispersion

6. **`comprehensive_comparison.png`** (4-panel):
   - Success counts vary widely
   - Sample sizes highly variable
   - Observed variance exceeds binomial expectations consistently

7. **`qq_plot.png`**:
   - Moderate departure from normality
   - Tails heavier than expected
   - Consistent with overdispersion

8. **`funnel_plot.png`**:
   - Points outside funnel limits confirm overdispersion
   - Trials 1, 2, 8, 11 most extreme
   - Classic signature of extra-binomial variation

**Visual Insights**:
- Multiple perspectives confirm overdispersion
- No systematic biases detected
- Variation appears genuine (not artifact)

---

## Alternative Hypotheses Tested

### Hypothesis 1: Constant Probability Model
**Test**: Chi-square goodness of fit
**Result**: REJECTED (p < 0.001)
**Conclusion**: Simple Binomial(n, p) inadequate

### Hypothesis 2: Temporal Trend Exists
**Test**: Correlation analysis
**Result**: NOT SUPPORTED (p > 0.2)
**Conclusion**: Trial order not informative

### Hypothesis 3: Sample Size Bias
**Test**: Correlation between n and proportion
**Result**: NOT SUPPORTED (p > 0.7)
**Conclusion**: No size-dependent bias

### Hypothesis 4: Two Distinct Groups
**Test**: Median split with t-test
**Result**: SUPPORTED (p = 0.012)
**Conclusion**: Evidence for heterogeneity

### Hypothesis 5: Multiple Probability Regimes
**Test**: Gap analysis and tercile splits
**Result**: TENTATIVELY SUPPORTED
**Conclusion**: Possible 2-3 probability levels

---

## Data Quality Issues

1. **Trial 1 (zero successes)**:
   - Probability under pooled model: 2.7%
   - Unusual but not impossible
   - Consider sensitivity analysis with/without this observation

2. **Sample Size Heterogeneity**:
   - Not a quality issue per se
   - Complicates variance estimation
   - Requires weighted approaches in modeling

3. **No True Quality Issues**:
   - Data appears clean and valid
   - All values within expected ranges
   - No evidence of data entry errors

---

## Robust vs Tentative Findings

### Robust (High Confidence):
1. Overdispersion is present (multiple tests, p < 0.001)
2. No temporal trend (p > 0.2)
3. No sample size bias (p > 0.7)
4. Pooled proportion approximately 0.074
5. Variance is 3.5x binomial expectation

### Tentative (Lower Confidence):
1. Two distinct probability groups (p = 0.012, but small sample)
2. Trial 1 as true outlier vs natural variation
3. Exact number of probability regimes (2 vs 3 vs continuous)
4. Whether trials 2, 8, 11 represent same underlying process

---

## Implications for Modeling

### Model Selection Recommendations:

1. **Do NOT use**: Simple Binomial(n, p) with constant p
   - Strongly rejected by data
   - Will underestimate uncertainty

2. **Consider**: Beta-Binomial Model
   - Accounts for overdispersion naturally
   - Parameters: α, β define variation in success probability
   - Appropriate for continuous variation in p

3. **Consider**: Mixture Model
   - 2-3 component mixture of binomials
   - Captures potential discrete probability regimes
   - More complex but may fit data structure better

4. **Consider**: Hierarchical Model
   - Each trial has own probability θ_i
   - Hyperprior on θ distribution
   - Most flexible, captures full uncertainty

### Prior Specification Guidance:

1. **For pooled probability**:
   - Central tendency near 0.074
   - Reasonable range: [0, 0.2]
   - Beta(2, 25) centers near 0.074 with appropriate spread

2. **For overdispersion parameter** (Beta-Binomial):
   - Dispersion factor approximately 3.5
   - Need priors that allow substantial overdispersion
   - Avoid overly restrictive priors

3. **For mixture models**:
   - Two components likely sufficient
   - Consider components near 0.05 and 0.11
   - Mixing proportions possibly equal (no strong evidence otherwise)

---

## Recommendations for Further Analysis

1. **Sensitivity Analysis**:
   - Fit models with/without trial 1
   - Assess impact of extreme observations

2. **Model Comparison**:
   - Fit Beta-Binomial, 2-component mixture, hierarchical
   - Use WAIC/LOO for comparison
   - Check posterior predictive distributions

3. **Domain Context**:
   - Seek explanation for overdispersion
   - Are trials from different populations/conditions?
   - Could inform model structure

4. **Validation**:
   - Posterior predictive checks
   - Simulate new data from fitted models
   - Compare to observed patterns

---

## Summary

This dataset exhibits strong overdispersion that cannot be explained by a simple binomial model. The variation appears genuine (not artifactual), with no systematic temporal or sample-size patterns. Evidence suggests possible grouping into 2-3 probability regimes, though this is less certain than the overdispersion finding itself. Beta-binomial or mixture models are recommended for proper inference, with careful prior specification around the observed pooled proportion of 0.074.
