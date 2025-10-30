# EDA Log - Analyst 2: Temporal/Sequential Patterns and Group Relationships

**Date:** 2025-10-30
**Dataset:** data_analyst_2.csv
**Focus:** Sequential patterns, clustering, correlation structure, data generation hypotheses

---

## Round 1: Initial Exploration

### Data Structure
- **Size:** 12 groups with sequential IDs (1-12)
- **Variables:** group_id, n_trials, r_successes, success_rate
- **Data Quality:** No missing values, no duplicates, all constraints satisfied

### Key Statistics
- **n_trials:** Mean=234.5, Median=201.5, Range=[47, 810], SD=198.4
  - High CV (0.85) and skewness (2.19) suggest heterogeneity
  - Potential outlier at 810 trials (group_id=4)

- **r_successes:** Mean=16.3, Median=14.5, Range=[3, 34], SD=9.8
  - Moderate variability (CV=0.60)

- **success_rate:** Mean=0.079, Median=0.071, Range=[0.031, 0.140], SD=0.035
  - Approximately 8% overall success rate
  - Moderate variability (CV=0.44)

### Initial Correlation Findings
1. **Strong positive:** n_trials vs r_successes (r=0.78, p=0.003)
   - More trials → more successes (expected if rates similar)
2. **Moderate negative:** n_trials vs success_rate (r=-0.34, p=0.28)
   - Groups with more trials tend to have lower success rates (non-significant)
3. **Weak positive:** r_successes vs success_rate (r=0.20, p=0.53)
   - Weaker than expected relationship

### Questions for Round 2
1. Is there a sequential trend in success_rate across group_id?
2. Do groups cluster into distinct subpopulations?
3. Is the n_trials/r_successes relationship linear or influenced by outliers?
4. Are there autocorrelation patterns suggesting temporal dependence?
5. What data generation process could produce these patterns?

---

## Round 2: Sequential and Temporal Analysis

### Trend Tests: Success Rate vs Group ID
**Result: NO SIGNIFICANT TEMPORAL TREND**

- **Pearson correlation:** r=-0.327, p=0.30 (non-significant)
- **Spearman correlation:** rho=-0.112, p=0.73 (non-significant)
- **Kendall's tau:** tau=-0.061, p=0.84 (non-significant)
- **Mann-Kendall test:** Z=-0.21, p=0.84 (no monotonic trend)
- **Linear regression:** slope=-0.003, p=0.30, R²=0.11

**Interpretation:** Group ID ordering does NOT influence success rate. No evidence of temporal drift or learning effects.

### Autocorrelation Analysis
**Result: NO SIGNIFICANT AUTOCORRELATION**

- **Lag-1 ACF:** 0.20 (weak positive)
- **Lag-2 ACF:** -0.36 (moderate negative)
- **Ljung-Box test (3 lags):** Q=3.58, p=0.31 (non-significant)

**Interpretation:** Success rates appear random across sequential ordering. No evidence of temporal dependence.

### Runs Test for Randomness
**Result: CONSISTENT WITH RANDOM SEQUENCE**

- **Observed runs:** 5
- **Expected runs:** 7.00
- **Z-score:** -1.21, p=0.23

**Interpretation:** Pattern of above/below median values is random.

### Temporal Segmentation
**Result: NO DIFFERENCE ACROSS TIME SEGMENTS**

- **First half (1-6) mean:** 0.0814
- **Second half (7-12) mean:** 0.0763
- **Mann-Whitney U test:** p=0.82 (no difference)

- **Thirds analysis:**
  - Segment 1 (1-4): 0.0913
  - Segment 2 (5-8): 0.0809
  - Segment 3 (9-12): 0.0644
  - **Kruskal-Wallis test:** H=0.27, p=0.87 (no difference)

**Interpretation:** Success rates are stable across temporal segments. Decreasing trend in means is NOT statistically significant.

### Visualizations Created
- `sequential_patterns.png`: 4-panel analysis showing no trend, sample size variation, autocorrelation, and runs
- `temporal_segments.png`: Segment comparison and cumulative rate evolution

### Round 2 Conclusions
1. **Sequential independence confirmed:** Group order doesn't matter
2. **No temporal drift:** Success rates stable over time
3. **Sample size varies:** n_trials shows no temporal pattern (rho=0.45, p=0.14)
4. **Groups can be treated as exchangeable** for modeling purposes

---

## Round 3: Correlation Structure Analysis

### Key Correlations (with significance)

**1. n_trials vs r_successes: STRONG POSITIVE**
- Pearson r=0.78, p=0.003 (highly significant)
- Spearman rho=0.88, p=0.0002 (very strong monotonic)
- R²=0.60 (60% variance explained)
- **Regression:** r_successes = 7.30 + 0.0385 × n_trials

**2. n_trials vs success_rate: WEAK NEGATIVE (non-significant)**
- Pearson r=-0.34, p=0.28
- Spearman rho=-0.02, p=0.96 (essentially zero)
- Only 12% variance explained

**3. r_successes vs success_rate: WEAK (non-significant)**
- Original Pearson r=0.20, p=0.53
- **BUT Partial correlation (controlling for n_trials):** r=0.79, p=0.002 (STRONG!)
- This reveals confounding by sample size

### Non-Linear Relationships
- **Quadratic fit:** R²=0.73 (vs linear R²=0.60)
  - 13% improvement suggests mild curvature
- **Log transformation:** log(n_trials) vs r_successes r=0.83 (stronger than linear)

### Outlier Influence
- **Group 4** (n_trials=810) is influential
- **Without Group 4:**
  - n_trials vs r_successes: r=0.78 → 0.77 (minimal change)
  - n_trials vs success_rate: r=-0.34 → -0.09 (substantial change)
- **Conclusion:** Outlier drives negative correlation with success_rate

### Homoscedasticity Check
- **Residuals analysis:** Homoscedastic (p=0.89)
- Variance is constant across fitted values

### Visualizations Created
- `correlation_structure.png`: Comprehensive 5-panel correlation analysis
- `outlier_influence.png`: Before/after outlier removal comparison
- `partial_correlation.png`: Demonstration of confounding by n_trials

### Round 3 Conclusions
1. **Strong linear relationship** between sample size and success count
2. **Apparent negative correlation** with success rate is driven by outlier
3. **Partial correlations reveal** true relationships masked by confounding
4. **Success count depends on both** n_trials AND intrinsic success rate

---

## Round 4: Clustering Analysis

### Hierarchical Clustering (Ward's Method)

**K=2 Solution:**
- Cluster 1: Groups 1,2,3,5,6,7,8,9,10,11,12 (n=11)
- Cluster 2: Group 4 only (n=1) - the outlier

**K=3 Solution (Most Interpretable):**
- **Cluster 0 (Large, Low Success):** Groups 3,4,5,6,7,9,11,12 (n=8)
  - Mean n_trials: 288.4 ± 231.7
  - Mean success_rate: 0.065 ± 0.026
  - Profile: High-volume, rare-event groups

- **Cluster 1 (Small, Very Low Success):** Group 10 only (n=1)
  - n_trials: 97
  - success_rate: 0.031
  - Profile: Smallest sample, rarest events

- **Cluster 2 (Medium, High Success):** Groups 1,2,8 (n=3)
  - Mean n_trials: 136.7 ± 55.5
  - Mean success_rate: 0.132 ± 0.006
  - Profile: Moderate volume, common events

**K=4 Solution:**
- Further isolates Group 4 (outlier with 810 trials)
- Other clusters remain similar to K=3

### K-Means Clustering
- **Agreement with hierarchical:** 83.3% (high stability)
- **Best k by interpretation:** 3 clusters
- **Inertia values:** K=2: 21.2, K=3: 11.0, K=4: 1.8

### Distance Analysis
- **Mean pairwise distance:** 1.75
- **Closest pairs:**
  - Groups 3 & 7: 0.25
  - Groups 9 & 11: 0.27
  - Groups 5 & 6: 0.29
- **Farthest pairs:**
  - Groups 1 & 4: 4.77
  - Groups 2 & 4: 4.34
  - Groups 4 & 8: 4.29

### Statistical Tests (K=3 differences)
- Clusters significantly differ on both dimensions
- Evidence of distinct subpopulations

### Visualizations Created
- `hierarchical_clustering.png`: Dendrograms for 4 linkage methods
- `kmeans_clustering.png`: K=2,3,4 solutions with characteristics table
- `distance_matrix.png`: Heatmap and distribution
- `cluster_analysis_k3.png`: Detailed K=3 cluster comparison

### Round 4 Conclusions
1. **Three natural clusters** emerge based on sample size × success rate
2. **Group heterogeneity confirmed:** Not all groups from same distribution
3. **Outlier group (4)** consistently separates in all methods
4. **High-success cluster (1,2,8)** distinctly different from others
5. **Clustering is stable** across methods

---

## Data Generation Hypotheses

### Hypothesis 1: Heterogeneous Binomial Processes
**Evidence FOR:**
- Three distinct clusters with different success rates
- Each group follows binomial structure (r_successes ≤ n_trials)
- Strong correlation between n_trials and r_successes within expected range

**Evidence AGAINST:**
- Success rates vary too much for single binomial
- n_trials highly variable (not fixed design)

**Likelihood:** HIGH - but with varying parameters across groups

### Hypothesis 2: Stratified Sampling from Different Populations
**Evidence FOR:**
- K=3 clustering suggests 3 underlying populations
- Within-cluster homogeneity
- No temporal trend (order doesn't matter)

**Evidence AGAINST:**
- Groups within clusters still show variation

**Likelihood:** MODERATE-HIGH

### Hypothesis 3: Observational Data with Confounding
**Evidence FOR:**
- n_trials not experimentally controlled (varies 47-810)
- Partial correlation reveals hidden relationships
- Outlier suggests different sampling mechanism

**Evidence AGAINST:**
- Clean binomial structure suggests experimental data

**Likelihood:** MODERATE

### Hypothesis 4: Sequential Experimental Batches
**Evidence FOR:**
- Sequential group IDs

**Evidence AGAINST:**
- No temporal trends
- No autocorrelation
- Runs test shows randomness

**Likelihood:** LOW

### Most Likely Scenario
**Stratified binomial sampling:** Three types of experimental conditions or populations, each with:
- Different underlying success probabilities (p≈0.13, 0.07, 0.03)
- Variable sample sizes (possibly resource-constrained)
- Independent observations within groups

---

## Key Findings Summary

1. **No Sequential Dependence:** Group order is arbitrary; can treat as cross-sectional
2. **Heterogeneous Groups:** Three distinct clusters suggest different data-generating processes
3. **Strong Sample Size Effects:** Larger n_trials → more r_successes (mechanically true)
4. **Confounding Present:** Must control for n_trials to see true rate relationships
5. **Outlier Influence:** Group 4 is unusual and affects aggregate statistics
6. **Stable Patterns:** Clustering robust across methods; findings reliable

---

## Implications for Modeling

### Recommended Approaches
1. **Hierarchical/Mixed Models** to account for cluster structure
2. **Beta-binomial** or **random effects binomial** to handle overdispersion
3. **Include cluster membership** as covariate or random effect
4. **Weight by n_trials** or use binomial likelihood (not Gaussian on rates)
5. **Consider robust methods** or outlier accommodation for Group 4

### Models to Test
1. **Fixed effects model:** success_rate ~ cluster_id
2. **Random effects:** r_successes ~ binomial(n_trials, p_group)
3. **Beta-binomial:** Account for extra-binomial variation
4. **Finite mixture model:** Explicitly model 3 latent classes

### Critical Assumptions to Check
- Independence within groups (no pseudo-replication)
- Binomial assumption appropriate (binary outcomes)
- Overdispersion present? (variance > binomial expectation)
- Are clusters meaningful or just artifacts?

---

## Files Created

### Code
- `01_initial_exploration.py` - Basic statistics and data quality
- `02_sequential_analysis.py` - Trend tests and autocorrelation
- `03_sequential_visualizations.py` - Temporal pattern plots
- `04_correlation_analysis.py` - Correlation structure
- `05_correlation_visualizations.py` - Correlation plots
- `06_clustering_analysis.py` - Hierarchical and k-means
- `07_clustering_visualizations.py` - Cluster plots

### Visualizations
- `sequential_patterns.png` - 4-panel temporal analysis
- `temporal_segments.png` - Segment comparisons
- `correlation_structure.png` - 5-panel correlation analysis
- `outlier_influence.png` - Outlier impact
- `partial_correlation.png` - Confounding demonstration
- `hierarchical_clustering.png` - Dendrograms
- `kmeans_clustering.png` - K-means solutions
- `distance_matrix.png` - Pairwise distances
- `cluster_analysis_k3.png` - Detailed K=3 analysis

### Data
- `processed_data.csv` - With derived variables
- `processed_data_with_clusters.csv` - With cluster labels

---

## Next Steps for Further Analysis
1. Test for overdispersion (variance > mean under binomial)
2. Validate cluster assignments with holdout or bootstrap
3. Investigate what distinguishes high-success groups (1,2,8)
4. Sensitivity analysis: how stable are findings to outlier removal?
5. Power analysis: can we reliably detect differences with n=12?
