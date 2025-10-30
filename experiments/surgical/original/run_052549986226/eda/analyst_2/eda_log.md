# EDA Log - Analyst 2: Hierarchical Structure Analysis

## Dataset Overview
- **Data file**: `/workspace/data/data_analyst_2.csv`
- **Structure**: 12 groups with varying trial counts and success rates
- **Variables**: group, n_trials, r_successes, success_rate

## Initial Observations (Data Read)
- 12 distinct groups (labeled 1-12)
- Highly imbalanced trial counts: range from 47 (group 1) to 810 (group 4)
- Success rates range from 0.0 (group 1) to 0.144 (group 8)
- Group 1 has 0 successes out of 47 trials - potential edge case

## Analysis Plan
1. **Round 1: Descriptive Analysis**
   - Calculate basic statistics per group and overall
   - Visualize group-level variation with confidence intervals
   - Compare within-group vs between-group variability

2. **Round 2: Hierarchical Structure Testing**
   - Test for overdispersion relative to binomial model
   - Estimate intraclass correlation
   - Calculate shrinkage potential
   - Compare pooled vs unpooled estimates

3. **Round 3: Clustering and Heterogeneity**
   - Test competing hypotheses about group structure
   - Evaluate evidence for natural clustering
   - Sensitivity analysis for small sample groups

---

## Round 1: Descriptive Analysis

### Basic Statistics (from 01_initial_exploration.py)

**Group-level summary:**
- 12 groups, highly variable sample sizes (47 to 810 trials, ratio 17.2:1)
- Success rates: Mean=0.074, Median=0.067, SD=0.038, Range=[0.0, 0.144]
- Coefficient of variation: 0.52 (moderate variability)
- Pooled success rate: 0.0739 (208/2814 total)

**Key findings:**
- Group 1 is an outlier (0% success rate, z-score > 2)
- Confidence interval widths vary substantially by sample size (0.032 to 0.112)
- 8 of 12 groups (67%) have CIs that include the pooled rate
- 3 groups have CIs entirely above pooled (Groups 2, 8, 11)
- 1 group has CI entirely below pooled (Group 5)

### Visualizations Created:
1. **caterpillar_plot_sorted.png**: Groups sorted by success rate with 95% CIs
   - Clear visualization of uncertainty by sample size
   - Most CIs overlap substantially with pooled rate

2. **caterpillar_plot_by_id.png**: Groups in original order
   - No obvious spatial pattern by group ID

3. **shrinkage_visualization.png**: Arrows showing potential shrinkage toward pooled mean
   - Groups 1, 2, 8, 11 show largest deviations from pooled rate
   - These would experience most shrinkage under hierarchical model

### CI Overlap Analysis:
- Total possible pairs: 66
- Overlapping pairs: 56 (84.8%)
- Non-overlapping pairs: 10 (15.2%)
- **Interpretation**: High overlap suggests continuous variation rather than discrete clusters

---

## Round 2: Hierarchical Structure Testing

### Variance Decomposition (from 03_variance_decomposition.py)

**1. Overdispersion Test:**
- Observed variance: 0.00148
- Expected variance (binomial): 0.00029
- **Variance ratio: 5.06x**
- **Conclusion**: STRONG evidence for overdispersion - variance is 5x what we'd expect from sampling alone

**2. Chi-square Test for Homogeneity:**
- Chi-square statistic: 38.56
- df: 11
- **p-value: 0.000063**
- **Conclusion**: REJECT null hypothesis - significant heterogeneity across groups

**3. Between-group Variance Component:**
- DerSimonian-Laird tau-squared: 0.000778
- DerSimonian-Laird tau (SD): 0.0279
- This represents the between-group standard deviation in success rates

**4. Intraclass Correlation (ICC):**
- **ICC = 0.727 (72.7%)**
- 72.7% of total variance is between-group
- 27.3% of variance is within-group (sampling)
- **Conclusion**: SUBSTANTIAL between-group variation → hierarchical modeling highly beneficial

**5. Shrinkage Analysis:**
- Average shrinkage factor: 0.144 (85.6% shrinkage toward pooled mean)
- Range: 0.035 (Group 1, smallest) to 0.387 (Group 4, largest)
- Small samples get heavy shrinkage, large samples less so
- This is appropriate given high between-group variance

### Visualization:
- **pooling_comparison.png**: Shows three approaches side-by-side
  - No pooling (observed rates)
  - Complete pooling (same rate for all)
  - Partial pooling (hierarchical - middle ground)
  - Right panel: Shrinkage increases with sample size (as expected)

---

## Round 3: Clustering and Heterogeneity

### Distribution Analysis (from 04_clustering_analysis.py)

**1. Normality Testing:**
- Shapiro-Wilk test: p = 0.496
- **Cannot reject normality** - success rates plausibly from normal distribution
- Q-Q plot shows reasonable fit to normal
- **Supports hierarchical model with normal prior for group effects**

**2. Clustering Analysis:**
- Median split: 0.0669
  - Lower cluster (6 groups): Mean=0.048, SD=0.026, Range=[0.000, 0.067]
  - Upper cluster (6 groups): Mean=0.099, SD=0.032, Range=[0.067, 0.144]

**3. Gap Analysis:**
- Mean gap between consecutive rates: 0.0131
- CV of gaps: 0.948 (high variability)
- One large gap detected: Between 0.000 and 0.038 (Group 1 vs others)
- **Interpretation**: One outlier (Group 1), but otherwise continuous distribution

**4. Sample Size Effect:**
- Spearman correlation (n_trials vs |rate - pooled|): -0.042, p=0.897
- **No significant correlation**
- This is UNEXPECTED - typically smaller samples show more extreme rates
- Suggests genuine between-group heterogeneity (not just sampling noise)

### Visualizations:
- **distribution_analysis.png**: 4-panel view
  - Histogram: Unimodal distribution
  - KDE: Smooth, continuous distribution (except Group 1)
  - Q-Q plot: Good fit to normal
  - Bar chart: No obvious pattern by group ID

- **clustering_analysis.png**:
  - Left: 2-cluster split shows some separation but substantial overlap
  - Right: Dendrogram shows gradual merging (continuous structure)

- **regression_to_mean.png**:
  - No clear trend (flat regression line)
  - Group 1 is clear outlier (0% success despite n=47)

---

## Summary of Key Findings

### Evidence FOR Hierarchical Modeling:
1. **Strong overdispersion** (5x expected variance)
2. **High ICC** (72.7% between-group variance)
3. **Significant chi-square test** (p < 0.001)
4. **Continuous distribution** of success rates (supports normal prior)
5. **Substantial shrinkage potential** (avg 85.6%)
6. **High CI overlap** (84.8% of pairs) - not discrete clusters

### Evidence AGAINST Fixed Effects:
1. Would treat each group independently, ignoring shared structure
2. Unstable estimates for small groups (e.g., Group 1 with n=47)
3. Doesn't leverage information across groups

### Edge Cases to Note:
- **Group 1**: 0/47 successes - will benefit most from shrinkage
- **Group 4**: Largest sample (n=810) - will shrink least
- **Groups 2, 8, 11**: High performers (CIs above pooled)
- **Group 5**: Low performer (CI below pooled)

### Recommended Modeling Approach:
**Hierarchical/Multilevel Binomial Model** with:
- Group-level random effects
- Normal prior for group effects (justified by normality test)
- Partial pooling to balance group-specific vs overall information
- Should naturally handle unbalanced sample sizes

### Alternative Models to Consider:
1. **Beta-Binomial**: If overdispersion is primary concern
2. **Bayesian Hierarchical Logistic**: If interested in covariate effects
3. **Mixture Model**: Only if we had evidence for discrete clusters (we don't)
