# Prior Predictive Check: Experiment 2 - Hierarchical Partial Pooling Model

**Date**: 2025-10-28
**Model**: Hierarchical Partial Pooling with Known Measurement Error
**Status**: PASS
**Decision**: Proceed to Simulation-Based Calibration (SBC)

---

## Executive Summary

The hierarchical prior specification for the Partial Pooling Model has been validated through comprehensive prior predictive checks. The hyperpriors `mu ~ Normal(10, 20)` and `tau ~ Half-Normal(0, 10)` generate scientifically plausible parameter values that appropriately account for both population-level uncertainty and between-group heterogeneity.

**Key Finding**: All 8 observations fall within reasonable percentile ranks (26th-74th percentile) of their respective prior predictive distributions, indicating excellent prior-data compatibility. The tau prior is appropriately regularizing (median = 6.8) while still allowing sufficient flexibility for the data to inform the degree of pooling. No computational issues were detected.

**Comparison to Model 1**: This hierarchical model has wider prior predictive distributions than the complete pooling model due to the additional between-group variance component (tau), providing greater flexibility to capture potential heterogeneity across groups.

---

## Visual Diagnostics Summary

The following visualizations were created to assess hierarchical prior plausibility:

1. **hyperprior_distributions.png** - Validates that hyperpriors (mu and tau) generate reasonable parameter values and shows their joint distribution
2. **group_level_parameters.png** - Examines the distribution of group-level means (theta_i) and their between-group variation
3. **prior_predictive_coverage.png** - Demonstrates that prior predictions cover observed data for all 8 groups
4. **hierarchical_structure.png** - Analyzes how tau affects the degree of pooling and shows prior predictive trajectories
5. **prior_data_compatibility.png** - Assesses percentile ranks, standardized residuals, and overall compatibility

---

## Model Specification

### Hyperprior Distributions
```
mu ~ Normal(10, 20)           # Population mean
tau ~ Half-Normal(0, 10)       # Between-group SD (regularizing prior)
```

**Justification**:
- **mu**: Same as Model 1, centered at 10 (near EDA weighted mean = 10.02), SD = 20 allows range [-30, 50]
- **tau**: Half-Normal(0, 10) is regularizing with median ≈ 6.8, preventing overfitting while allowing moderate heterogeneity
  - Mean: 8.0
  - Median: 6.8
  - 95% quantile: 19.6

### Group-Level Model
```
theta_i ~ Normal(mu, tau)      for i = 1, ..., 8
```

**Non-Centered Parameterization** (computational efficiency):
```
theta_raw_i ~ Normal(0, 1)
theta_i = mu + tau * theta_raw_i
```

### Likelihood
```
y_i ~ Normal(theta_i, sigma_i)    for i = 1, ..., 8
```

where `sigma_i` are known measurement errors from the data (range: 9-18).

---

## Hyperprior Analysis

### mu (Population Mean) - (`hyperprior_distributions.png`, Panel A)

**Prior samples (n=5000)**:
- Mean: 10.11 (matches specified prior mean of 10)
- Standard deviation: 19.93 (matches specified prior SD of 20)
- 95% credible interval: [-29.4, 48.9]
- Range: [-54.8, 88.5]

**Assessment**:
- The prior generates parameter values centered near the observed mean (12.5)
- The 95% interval width (78.3 units) is appropriately wide for population-level uncertainty
- The observed mean falls at the 60th percentile of the prior, indicating good compatibility
- No extreme values detected (0% of samples exceed |100|)

**Visual Evidence**: The left panel of `hyperprior_distributions.png` shows the prior distribution (blue histogram with red theoretical density) aligns well with the observed mean (green dashed line at 12.5). The distribution is weakly informative, allowing data to dominate.

### tau (Between-Group SD) - (`hyperprior_distributions.png`, Panel B)

**Prior samples (n=5000)**:
- Mean: 8.04
- Median: 6.84 (close to theoretical median of 6.77)
- Standard deviation: 6.11
- 95% credible interval: [0.30, 22.42]
- Range: [0.01, 39.22]

**Assessment**:
- The Half-Normal(0, 10) prior is **regularizing**: median tau ≈ 6.8 suggests moderate between-group variation
- 95% of prior mass falls below tau = 22, preventing excessive heterogeneity
- Prior allows both complete pooling (tau near 0) and substantial heterogeneity (tau up to ~20)
- The regularization is appropriate: encourages similarity across groups while letting data override if needed

**Visual Evidence**: Panel B of `hyperprior_distributions.png` shows the Half-Normal distribution (coral histogram with red theoretical density). The median (purple dashed line at 6.8) indicates the prior gently pushes toward moderate pooling. The 95% quantile (orange line at 19.6) shows the prior still allows substantial variation if warranted.

**Comparison to Alternative Priors**:
- **Half-Cauchy(0, 5)** (Gelman's recommendation): Heavier tails, less regularizing
- **Half-Normal(0, 10)** (chosen): Moderate regularization, balances parsimony and flexibility
- **Exponential(1)**: Stronger regularization toward zero, may be too restrictive

### Joint Distribution (mu vs tau) - (`hyperprior_distributions.png`, Panel C)

**Key Insight**: The hexbin plot shows mu and tau are sampled independently (no correlation), as expected from the prior specification. This allows the data to determine both the population mean and the degree of heterogeneity without prior constraints on their relationship.

**Visual Evidence**: The hexbin plot shows uniform density across the (mu, tau) space with no visible correlation pattern. The concentration is centered around mu=10 and tau=7 (median values), with symmetric spread.

---

## Group-Level Parameter Analysis

### Distribution of theta_i (`group_level_parameters.png`)

**Theta (group mean) samples (5000 x 8 = 40,000 total)**:
- Mean: 10.13 (matches population prior mean)
- Standard deviation: 22.41
- 95% credible interval: [-34.1, 53.8]
- Range: [-90.4, 118.9]

**Assessment**:
- The group-level means are more dispersed than the population mean (SD = 22.4 vs 20.0) due to added between-group variation from tau
- The 95% interval is wide (87.9 units), reflecting substantial prior uncertainty about individual group means
- This is appropriate: each group can differ from the population mean, but the degree of difference is informed by tau

**Visual Evidence**:
- **Panel A** (`group_level_parameters.png`): Shows the distribution of all theta samples (green histogram). The observed y values (thin red vertical lines) fall within the bulk of this distribution, with the observed mean (thick red dashed line) near the center.
- **Panel C**: Box plots for each group show wide prior distributions, with observed values (red X's) falling comfortably within the boxes for most groups. Groups differ in their prior distributions due to sampling variability, but all share the same hierarchical structure.

### Between-Group Variation (`group_level_parameters.png`, Panel B)

**Within-sample standard deviation (SD across 8 groups for each prior draw)**:
- Mean SD: 7.27
- Median SD: 5.78
- 95% interval: [0.25, 22.68]

**Comparison to tau prior**:
The within-sample SD distribution (coral) closely matches the tau prior distribution (blue), confirming correct implementation of the hierarchical structure. The slight difference in mean (7.27 vs 8.04) is due to sampling variability with only 8 groups.

**Assessment**: The prior allows for a wide range of between-group variation:
- **Low variation** (SD < 2): Complete pooling regime, groups essentially identical
- **Moderate variation** (SD = 5-10): Partial pooling regime, groups differ but share information
- **High variation** (SD > 15): Weak pooling regime, groups nearly independent

**Visual Evidence**: Panel B shows overlapping distributions of within-sample SD (coral) and tau (blue), with mean within-SD (red dashed line at 7.27) near the tau median. This confirms that the hierarchical structure is working as intended.

### Prior Sample Trajectories (`group_level_parameters.png`, Panel D)

**Visual Evidence**: Panel D shows 200 random prior draws of group means (theta) as blue lines, with the observed data as a red line. The observed data falls well within the 50% interval (dark green) for most groups and easily within the 95% interval (light green) for all groups.

**Key Insight**: The prior trajectories show diverse patterns of between-group variation:
- Some draws are nearly flat (low tau, complete pooling)
- Some show substantial variation (high tau, heterogeneous groups)
- Most show moderate variation (median tau ≈ 6.8)

This demonstrates that the prior does not force a particular pooling structure but allows the data to determine it.

---

## Prior Predictive Coverage

### Individual Observation Coverage (`prior_predictive_coverage.png`)

For each of the 8 observations, we generated 5000 prior predictive samples and computed the percentile rank of the observed value:

| Obs | y_obs  | sigma | Percentile Rank | Status |
|-----|--------|-------|-----------------|--------|
| 0   | 20.02  | 15    | 64.8%          | OK     |
| 1   | 15.30  | 10    | 57.7%          | OK     |
| 2   | 26.08  | 16    | 71.3%          | OK     |
| 3   | 25.73  | 11    | 73.7%          | OK     |
| 4   | -4.88  | 9     | 26.0%          | OK     |
| 5   | 6.08   | 11    | 42.8%          | OK     |
| 6   | 3.17   | 10    | 38.1%          | OK     |
| 7   | 8.55   | 18    | 47.9%          | OK     |

**Assessment**:
- All observations fall within the 10th-90th percentile range
- No observations in extreme tails (< 5% or > 95%)
- Percentile ranks span a reasonable range (26.0% - 73.7%), showing good diversity
- The prior predictive distributions appropriately account for varying measurement errors
- Ranks are similar to Model 1, suggesting the added hierarchical structure doesn't drastically change prior predictive behavior

**Visual Evidence**: The 8-panel plot shows each observation's prior predictive distribution (blue histogram). Unlike Model 1, there is no simple theoretical density to overlay (the distribution is a mixture over mu and tau), so we show only the empirical distribution. The observed value (red dashed line) falls comfortably within the bulk of each distribution, with vertical green dashed lines marking the 95% interval.

**Comparison to Model 1**:
The percentile ranks are very similar between models (within ±2% for most observations), indicating that both priors are compatible with the data. The hierarchical model has slightly wider predictive distributions due to the tau variance component.

---

## Hierarchical Structure Check

### Effect of tau on Pooling (`hierarchical_structure.png`, Panel A)

To illustrate how tau controls the degree of pooling, we generated group means for fixed mu=10 and varying tau values:

- **tau = 0** (red): Complete pooling - all groups identical at mu=10
- **tau = 5** (green): Moderate pooling - groups differ moderately
- **tau = 10** (blue): Substantial pooling - groups differ substantially
- **tau = 20** (purple): Weak pooling - groups nearly independent

**Visual Evidence**: Panel A shows four different pooling scenarios. The observed data (yellow circles with black edges) exhibits moderate variation around the grand mean, suggesting tau is likely in the moderate range (5-10), consistent with the prior median of 6.8.

**Key Insight**: The prior allows the full spectrum of pooling behavior, from complete (tau→0) to none (tau→∞). The data will determine where on this spectrum the model lands. The EDA suggested complete pooling (tau²=0), so we expect the posterior to favor low tau values.

### Prior Predictive Trajectories (`hierarchical_structure.png`, Panel B)

**Visual Evidence**: Panel B shows 100 random prior predictive datasets (blue lines with transparency). The observed data (red line) falls well within both the 50% interval (dark green) and 95% interval (light green). The median trajectory (black line) is flat near y=10, reflecting the prior mean.

**Assessment**: The prior predictive trajectories show enormous diversity, from flat lines (complete pooling) to highly variable patterns (no pooling). This flexibility is appropriate - the model should not force a particular pattern but let the data determine it.

### Variance Decomposition (`hierarchical_structure.png`, Panel C)

The prior predictive variance for each observation has three components:
1. **Hyperprior mu variance** (blue): σ²_mu = 400
2. **Between-group tau variance** (coral): E[τ²] ≈ 64
3. **Measurement error variance** (green): σ²_i (varies by observation)

**Variance contribution percentages**:

For all groups (approximately equal):
- **Mu uncertainty**: ~63% (largest component)
- **Between-group variation**: ~10% (moderate component)
- **Measurement error**: ~27% (varies by group, 9² to 18²)

**Assessment**:
- The mu prior dominates the total variance, indicating substantial population-level uncertainty
- The tau component (10%) is modest, reflecting the regularizing prior
- Measurement error contributes less than in Model 1 (where it was 35-65%) because hierarchical variance components add to total uncertainty

**Visual Evidence**: The stacked bar chart shows three colored regions for each group. The blue region (mu variance) dominates, the coral region (tau variance) is moderate, and the green region (measurement error) varies by group depending on sigma_i.

**Key Difference from Model 1**: In Model 1, variance decomposition was just mu + sigma². Here, we add tau², increasing total predictive uncertainty. This is appropriate for a more flexible model.

### Percentile Ranks per Group (`hierarchical_structure.png`, Panel D)

**Visual Evidence**: All 8 groups show green bars (within 5th-95th percentile), with lengths ranging from 26% to 74%. No red bars (extreme tails) are present. The bars are well-distributed across the 0-100% range, avoiding clustering at extremes.

**Assessment**: Excellent prior-data compatibility across all groups.

---

## Prior-Data Compatibility

### Rank Distribution (`prior_data_compatibility.png`, Panel A)

**Visual Evidence**: The histogram of percentile ranks (blue bars) shows all 8 observations fall between 20-80%, avoiding both extreme tails (red shaded regions). With only 8 observations, we cannot expect perfect uniformity, but the distribution shows no concerning patterns.

**Assessment**: The ranks are well-spread and avoid extreme tails, indicating the prior is neither too informative nor too vague. The lack of observations below 5% or above 95% confirms no prior-data conflict.

### Observed vs Prior Predictive Means (`prior_data_compatibility.png`, Panel B)

**Visual Evidence**: Blue error bars show the prior predictive mean ± 1.96 SD for each group. All means are centered near 10 (reflecting the mu prior), with wide 95% confidence intervals (spanning ~100 units). Red X marks show observed values falling well within these intervals.

**Assessment**: The prior predictive means are all similar (centered at mu=10) because the prior doesn't favor any particular group. The wide intervals reflect large prior uncertainty from both mu and tau. Observed values fall within these intervals, showing compatibility.

### Q-Q Plot: Standardized Residuals (`prior_data_compatibility.png`, Panel C)

**Standardized residuals**: For each observation, z = (y_obs - mean(y_pred)) / sd(y_pred)

**Visual Evidence**: The Q-Q plot shows standardized residuals (blue dots) falling reasonably close to the diagonal line (red), indicating the prior predictive distributions are approximately normal with appropriate spread. Some deviation at the tails is expected with only 8 observations.

**Assessment**: No systematic deviations from normality, confirming the hierarchical normal model is structurally appropriate for these data.

### Boxplots (`prior_data_compatibility.png`, Panel D)

**Visual Evidence**: Box plots (light blue) show the full prior predictive distribution for each group. Observed values (red X's) fall within the interquartile range (boxes) for 5 out of 8 observations, and within the whiskers for all 8.

**Assessment**: This confirms good prior-data compatibility - observed values are "typical" under the prior predictive distribution.

---

## Computational Diagnostics

### Numerical Stability

**Checks performed**:
- NaN values: 0 (PASS)
- Inf values: 0 (PASS)
- Max absolute value: 147.8 (PASS - well below threshold of 1000)
- Extreme hyperparameter values: 0 (PASS)
  - |mu| > 100: 0 samples (0.00%)
  - tau > 50: 0 samples (0.00%)

**Assessment**: No computational issues detected. The hierarchical prior generates values in a reasonable range that will not cause numerical instability during MCMC sampling. The max value of 147.8 is acceptable (within ±150), though larger than Model 1 (max = 112.7) due to additional variance from tau.

**Implication for MCMC**: The non-centered parameterization should prevent funnel geometry issues. The prior values are in a range that will not cause overflow, underflow, or gradient problems. We expect good NUTS performance with target_accept = 0.95.

---

## Decision Criteria Evaluation

### PASS Criteria (All Met)

1. **Hyperpriors are weakly informative**:
   - mu: 95% interval width = 78.3 units (same as Model 1)
   - tau: 95% interval = [0.30, 22.42], median = 6.8
   - Not too narrow (would be overly informative)
   - Not too wide (would cause computational issues)
   - **Status**: PASS

2. **tau prior is appropriately regularizing**:
   - Median tau = 6.8 (moderate between-group variation)
   - 95% quantile = 22.4 (allows substantial variation if needed)
   - Allows both complete pooling (tau→0) and heterogeneity (tau up to ~20)
   - **Status**: PASS

3. **Observed data within reasonable prior predictive range**:
   - All observations fall between 26th-74th percentile
   - No observations in extreme tails (< 5% or > 95%)
   - **Status**: PASS

4. **No prior-data conflict**:
   - Observed mean (12.5) is near prior mean (10)
   - All individual observations compatible with prior predictions
   - Q-Q plot shows no systematic deviations
   - **Status**: PASS

5. **Hierarchical structure allows appropriate pooling**:
   - Prior supports full range from complete to no pooling
   - tau prior median (6.8) is reasonable for the data scale
   - Between-group variation is plausible
   - **Status**: PASS

6. **No computational issues**:
   - Zero NaN or Inf values
   - All values in reasonable range (max = 147.8)
   - No extreme hyperparameter values
   - **Status**: PASS

---

## Key Visual Evidence

The three most important diagnostic plots:

1. **hyperprior_distributions.png** (Panel B - tau prior): Demonstrates that the Half-Normal(0, 10) prior is appropriately regularizing with median ≈ 6.8, encouraging moderate pooling while allowing the data to override if heterogeneity is present. This is the key difference from Model 1.

2. **hierarchical_structure.png** (Panel A - effect of tau): Illustrates how different tau values lead to different degrees of pooling, from complete (tau=0) to none (tau=20). The observed data's moderate variation suggests tau will be in the 5-10 range, consistent with the prior median.

3. **prior_predictive_coverage.png**: All 8 panels show observed values (red dashed lines) falling comfortably within the bulk of their respective prior predictive distributions (blue histograms), with percentile ranks ranging from 26% to 74%. This confirms excellent prior-data compatibility across all groups.

---

## Scientific Interpretation

### Domain Plausibility

The hierarchical prior specification encodes reasonable domain knowledge:

1. **Population mean (mu)**: Same as Model 1, centered at 10 (near observed mean), allowing range [-30, 50]

2. **Between-group variation (tau)**:
   - Median ≈ 6.8 suggests groups can differ by roughly ±7 units from the population mean
   - 95% quantile ≈ 22 allows for substantial heterogeneity if present
   - This is appropriate for the observed data scale (y range: -5 to 26)

3. **Group means (theta_i)**:
   - Each group can deviate from mu, with the amount controlled by tau
   - Prior allows diverse patterns: some groups above mu, some below
   - The degree of deviation is informed by data, not forced by prior

4. **Data-prior balance**:
   - Mu variance (400) is largest component, giving data strong influence on population mean
   - Tau variance (median²=46) is moderate, allowing flexibility without overfitting
   - Measurement error variance (81-324) varies by group as observed

### No Prior-Data Conflict

**Assessment**: No evidence of prior-data conflict. The hierarchical prior and likelihood are compatible:
- Observed data in mid-range of prior predictive (not in extreme tails)
- Q-Q plot shows no systematic misfit
- Variance decomposition is sensible
- All decision criteria passed

**Comparison to Model 1**: Both models show excellent prior-data compatibility. The hierarchical model has slightly wider predictive distributions but similar percentile ranks, indicating the additional flexibility from tau does not create conflict.

### Expected Posterior Behavior

Based on the prior predictive check, we expect:

1. **mu posterior**: Centered near observed mean (12.5), narrower than prior due to data
2. **tau posterior**: Likely near 0-5 (EDA suggests tau²=0), as groups show limited heterogeneity beyond measurement error
3. **theta_i posterior**: Pulled toward mu (shrinkage), with amount depending on tau and sigma_i
4. **Potential issues**:
   - Funnel geometry if tau→0 (mitigated by non-centered parameterization)
   - Divergences < 5% acceptable if tau is near boundary
   - tau may be poorly identified with only 8 groups (wide posterior expected)

---

## Comparison to Model 1 (Complete Pooling)

| Aspect | Model 1 | Model 2 (Hierarchical) |
|--------|---------|----------------------|
| **Parameters** | 1 (mu) | 10 (mu, tau, theta[1:8]) |
| **Prior predictive mean** | 10 | 10 (same) |
| **Prior predictive SD** | 21-27 | 23-29 (wider) |
| **Percentile ranks** | 25-75% | 26-74% (very similar) |
| **Variance components** | mu + sigma² | mu + tau² + sigma² |
| **Prior-data compatibility** | Excellent | Excellent |
| **Max absolute value** | 112.7 | 147.8 (larger) |
| **Computational concerns** | None | Potential funnel if tau→0 |

**Key Differences**:
1. **Flexibility**: Model 2 adds tau, allowing between-group variation. Model 1 forces all groups to share the same mean.
2. **Variance**: Model 2 has wider prior predictive distributions due to tau² component.
3. **Complexity**: Model 2 has 10 parameters vs 1, requiring more computational resources.
4. **Prior-data fit**: Both models show excellent compatibility, with nearly identical percentile ranks.

**Implications**:
- Both priors are appropriate for the data
- Model 2 is more flexible but more complex
- The data will determine if the added complexity is justified (via LOO-CV comparison)
- Based on EDA (tau²=0), we expect Model 1 to be preferred (parsimony), but formal Bayesian analysis may reveal subtle heterogeneity

---

## Potential Issues Detected

**None**. The hierarchical prior specification passed all checks:
- Hyperpriors generate plausible values
- tau prior is appropriately regularizing
- Prior predictive distributions cover observed data
- No observations in extreme tails (< 5% or > 95%)
- No computational issues (NaN, Inf, extreme values)
- Hierarchical structure allows appropriate range of pooling

---

## Recommendations

### Immediate Next Steps

1. **Proceed to Simulation-Based Calibration (SBC)**:
   - The prior predictive check validates that the hierarchical prior is appropriate
   - SBC will validate that the computational implementation (MCMC with non-centered parameterization) can recover known parameters
   - Expected outcome: SBC should pass if non-centered parameterization is implemented correctly
   - **Special attention**: Check that tau can be recovered when it's near 0 (complete pooling) and when it's moderate (5-10)

2. **Maintain current prior specification**:
   - Do not adjust the hyperpriors - they are well-calibrated
   - mu ~ Normal(10, 20) and tau ~ Half-Normal(0, 10) should be used in all subsequent analyses
   - The non-centered parameterization should be used to avoid funnel geometry

### SBC-Specific Recommendations

Given the hierarchical structure, pay special attention in SBC to:

1. **Funnel geometry**: Even with non-centered parameterization, check for poor recovery when tau is near 0
2. **tau identifiability**: With only 8 groups, tau may be poorly identified. Check if SBC recovers tau with wide uncertainty.
3. **Shrinkage**: Verify that theta_i values are appropriately shrunk toward mu, with shrinkage amount depending on tau and sigma_i
4. **Computational efficiency**: Check ESS and divergences across different tau regimes

### Long-term Validation Pipeline

After SBC:
1. **Posterior inference**: Fit model to observed data
   - Use target_accept=0.95 (higher than Model 1's 0.90) due to hierarchical structure
   - Use 2000 tuning iterations (more than Model 1's 1000) for better adaptation
   - Monitor for divergences (< 5% acceptable if tau near 0)
2. **Posterior predictive check**: Validate model fit
   - Check if model reproduces observed patterns
   - Compare to Model 1 posterior predictive
3. **Model critique**: Compare to Model 1 via LOO-CV
   - Expected outcome: Model 1 preferred (simpler, EDA supports complete pooling)
   - If tau posterior is clearly > 0, Model 2 may be preferred

---

## Technical Notes

### Prior Predictive Distribution (Hierarchical Model)

Unlike Model 1, the hierarchical prior predictive does not have a simple closed-form distribution. It is a mixture:

```
For observation j:
  mu ~ Normal(10, 20)
  tau ~ Half-Normal(0, 10)
  theta_j ~ Normal(mu, tau)
  y_j ~ Normal(theta_j, sigma_j)
```

Marginalizing over theta_j:
```
y_j | mu, tau ~ Normal(mu, sqrt(tau² + sigma_j²))
```

Marginalizing over mu and tau requires numerical integration (we use Monte Carlo):
```
y_j ~ ∫∫ Normal(mu, sqrt(tau² + sigma_j²)) * Normal(mu|10, 20) * HalfNormal(tau|0, 10) dmu dtau
```

This is why we cannot overlay a simple theoretical density in the prior predictive coverage plots.

### Variance Decomposition

Total prior predictive variance for observation j:
```
Var(y_j) = Var_mu(E[y_j | mu]) + E_mu(Var[y_j | mu])
         = Var(mu) + E[tau² + sigma_j²]
         = 400 + E[tau²] + sigma_j²
         ≈ 400 + 64 + sigma_j²  (since E[tau²] ≈ 64 for Half-Normal(0, 10))
```

For example:
- Obs 4 (sigma=9): Var(y_j) ≈ 400 + 64 + 81 = 545, SD ≈ 23.3
- Obs 7 (sigma=18): Var(y_j) ≈ 400 + 64 + 324 = 788, SD ≈ 28.1

### Half-Normal Properties

For tau ~ Half-Normal(0, σ):
- Mean: σ * sqrt(2/π) ≈ 0.798 * σ
- Median: σ * sqrt(2 * ln(2)) ≈ 1.177 * σ
- Variance: σ² * (1 - 2/π) ≈ 0.363 * σ²

For σ = 10:
- Mean: 7.98
- Median: 11.77 (note: our empirical median was 6.84, likely sampling variability)
- Variance: 36.3

**Correction**: The theoretical median of Half-Normal(0, 10) is actually 6.77 (not 11.77), computed as:
```
median = scale * sqrt(2 * log(2)) = 10 * sqrt(2 * log(2)) ≈ 6.77
```

This matches our empirical median of 6.84, confirming correct implementation.

### Reproducibility

- Random seed: 42
- Prior samples: 5000
- All code available in: `/workspace/experiments/experiment_2/prior_predictive_check/code/prior_predictive_check.py`
- Summary statistics saved in: `/workspace/experiments/experiment_2/prior_predictive_check/diagnostics/summary_stats.json`

---

## Conclusion

**DECISION: PASS**

The hierarchical prior specification (mu ~ Normal(10, 20), tau ~ Half-Normal(0, 10)) is appropriate for the Partial Pooling Model. It generates scientifically plausible parameter values, provides appropriate prior predictive coverage of observed data, allows the full range of pooling behavior, and exhibits no computational issues.

**Key Strengths**:
1. **Weakly informative hyperpriors**: Allow data to dominate while preventing extreme values
2. **Regularizing tau prior**: Encourages parsimony (moderate pooling) while allowing flexibility
3. **Excellent prior-data compatibility**: All observations in mid-range of prior predictive (26-74%)
4. **Hierarchical structure validated**: Prior allows full spectrum from complete to no pooling
5. **Computational stability**: No numerical issues, ready for MCMC

**Recommendation**: Proceed to Simulation-Based Calibration (SBC) to validate the computational implementation (non-centered parameterization) before fitting to observed data.

**Expected SBC Outcome**: Should pass if non-centered parameterization is correctly implemented. Pay special attention to tau recovery, particularly when tau is near 0 (funnel regime) and when tau is moderate (typical of prior).

---

## Appendix: Summary Statistics

```json
{
  "decision": "PASS",
  "n_observations": 8,
  "n_prior_samples": 5000,
  "hyperpriors": {
    "mu_mean": 10,
    "mu_sd": 20,
    "tau_sd": 10,
    "mu_95_interval": [-29.40, 48.91],
    "tau_median": 6.84,
    "tau_95_interval": [0.30, 22.42]
  },
  "observed_data": {
    "mean": 12.50,
    "range": [-4.88, 26.08]
  },
  "group_parameters": {
    "theta_mean": 10.13,
    "theta_95_interval": [-34.10, 53.77],
    "within_sample_sd_mean": 7.27,
    "within_sample_sd_median": 5.78
  },
  "prior_predictive": {
    "percentile_ranks": [64.8, 57.7, 71.3, 73.7, 26.0, 42.8, 38.1, 47.9],
    "n_extreme_low": 0,
    "n_extreme_high": 0,
    "mean": 10.19,
    "std": 25.95
  },
  "computational": {
    "n_nan": 0,
    "n_inf": 0,
    "max_abs_value": 147.81
  },
  "issues": []
}
```

---

**Analysis completed**: 2025-10-28
**Analyst**: Claude (Bayesian Model Validator)
**Next step**: Simulation-Based Calibration (SBC)
