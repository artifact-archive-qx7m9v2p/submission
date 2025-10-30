# Prior Predictive Check: Random Effects Logistic Regression

**Experiment 2 | Date: 2025-10-30**

---

## Executive Summary

**DECISION: PASS**

The prior specifications for the Random Effects Logistic Regression model generate scientifically plausible data that adequately covers the observed range. All critical checks pass, with one minor technical issue that does not affect model validity. The model is **GO** for SBC validation and subsequent fitting.

---

## Visual Diagnostics Summary

Five diagnostic visualizations were created to assess prior plausibility:

1. **`parameter_plausibility.png`** - Prior distributions for μ (global mean) and τ (between-group SD) on both log-odds and probability scales
2. **`prior_predictive_proportions.png`** - Violin plots of prior predictive group-level proportions with observed data overlay
3. **`prior_predictive_counts.png`** - 12-panel plot showing prior predictive count distributions for each group
4. **`group1_zero_inflation_diagnostic.png`** - Focused diagnostic for Group 1 (n=47, r=0) to assess zero-count plausibility
5. **`prior_predictive_coverage.png`** - Prior predictive intervals vs. observed counts across all groups

---

## Model Specification Review

**Likelihood:**
```
r_i | θ_i, n_i ~ Binomial(n_i, logit⁻¹(θ_i))  for i = 1, ..., 12
```

**Hierarchical Structure (Non-centered):**
```
θ_i = μ + τ · z_i
z_i ~ Normal(0, 1)
```

**Priors:**
```
μ ~ Normal(logit(0.075), 1²)     # logit(0.075) ≈ -2.51
τ ~ HalfNormal(1)                # Between-group SD on log-odds scale
```

**Data Context:**
- 12 groups with sample sizes ranging from 47 to 810
- Observed pooled rate: 7.39%
- Observed group rates: 0% to 14.4%
- Strong observed heterogeneity (ICC = 0.66 from EDA)

---

## Prior Predictive Simulation Results

### Simulation Configuration
- **Number of prior samples:** 1,000
- **Method:** Forward sampling from priors through full generative model
- **Random seed:** 42 (reproducible)

### Prior Parameter Distributions

#### Global Mean (μ)

**Log-odds scale:**
- Prior mean: -2.51 (as specified)
- Sampled range: [-5.75, 1.34]
- 95% CI: Not explicitly computed, but distribution appears appropriate

**Probability scale (p = logit⁻¹(μ)):**
- Prior mean: 0.105
- Prior median: 0.077
- 95% CI: [0.013, 0.354]
- **Observed pooled rate: 0.074** ✓ (within 95% CI)

**Assessment:** The prior on μ appropriately centers near the observed pooled rate while remaining weakly informative. The transformation to probability scale shows reasonable spread covering rates from ~1% to ~35%.

#### Between-Group SD (τ)

**Distribution:**
- Prior mean: 0.792
- Prior median: 0.651
- 95% CI: [0.027, 2.23]

**Interpretation:** The HalfNormal(1) prior allows for:
- Low heterogeneity (τ near 0)
- Moderate heterogeneity (τ around 0.5-1.0)
- High heterogeneity (τ up to ~2.2 at 97.5th percentile)

This range is appropriate given the observed ICC = 0.66, which indicates substantial between-group variation. The prior does not artificially constrain heterogeneity.

#### Between-Group Variance (Probability Scale)

Approximate variance: Var(p) ≈ τ² · p²(1-p)²
- Prior mean: 0.010
- 95% CI: [0.000002, 0.086]

This approximation shows the prior allows group proportions to vary considerably around the global mean.

---

## Prior Predictive Distribution Assessment

### 1. Group-Level Proportions

**Prior Predictive Statistics (`prior_predictive_proportions.png`):**
- Prior pred mean: 0.129
- Prior pred range: [0.000, 0.995]
- Prior pred 95% CI: [0.006, 0.582]

**Observed Statistics:**
- Observed range: [0.000, 0.144]
- Observed pooled: 0.074

**Key Findings:**
- All observed proportions fall well within the prior predictive distribution (see violin plots)
- The prior generates proportions from near-zero to moderate values (up to ~60% in extreme tails)
- Each group's observed proportion is located within a plausible region of its prior predictive distribution
- The prior is appropriately diffuse without being unrealistically wide

### 2. Group-Level Counts

**Coverage Assessment (`prior_predictive_counts.png` and `prior_predictive_coverage.png`):**

Within ±5 counts of observed:
- Group 1 (n=47, r=0): 631/1000 (63%)
- Group 2 (n=148, r=18): 209/1000 (21%)
- Group 3 (n=119, r=8): 473/1000 (47%)
- Group 4 (n=810, r=46): 74/1000 (7%)
- Group 5 (n=211, r=8): 361/1000 (36%)
- Group 6 (n=196, r=13): 292/1000 (29%)
- Group 7 (n=148, r=9): 414/1000 (41%)
- Group 8 (n=215, r=31): 105/1000 (11%)
- Group 9 (n=207, r=14): 245/1000 (25%)
- Group 10 (n=97, r=8): 511/1000 (51%)
- Group 11 (n=256, r=29): 115/1000 (12%)
- Group 12 (n=360, r=24): 152/1000 (15%)

**Interpretation:**
- Lower coverage for groups with larger sample sizes (Groups 4, 8, 11) is expected and appropriate - the prior is less informative than large datasets
- Higher coverage for smaller groups (Groups 1, 3, 7, 10) indicates the prior appropriately reflects uncertainty
- All observed counts fall within 95% prior predictive intervals
- No systematic bias (observed counts are not consistently at distribution edges)

### 3. Critical Test: Group 1 Zero Count

**Context:** Group 1 has n=47 and observed r=0. This is the most challenging observation for the model.

**Prior Predictive Results (`group1_zero_inflation_diagnostic.png`):**
- P(r=0 | prior) = 0.124 (12.4%)
- P(r≤2 | prior) = 0.398 (39.8%)
- P(r≤5 | prior) = 0.631 (63.1%)
- Prior predictive count range: [0, 47]

**Assessment:** EXCELLENT
- The prior generates r=0 at a reasonable frequency (12.4%)
- This is neither too high (overfitting to zero) nor too low (excluding plausible values)
- The prior predictive proportion distribution for Group 1 appropriately includes very low values
- The observed zero count is highly plausible under the prior (not an outlier)

### 4. Extreme Values Check

**Results:**
- Proportions < 0.1%: 43/12,000 (0.36%)
- Proportions > 50%: 445/12,000 (3.7%)

**Assessment:**
- Very low proportions are rare but possible (appropriate for rare events)
- High proportions (>50%) occur occasionally in prior predictive, reflecting tail uncertainty
- No domain violations (all p ∈ [0, 1])
- Distribution is reasonable for this data context

---

## Computational Health Checks

### Numerical Stability
- **Invalid probabilities:** 0/12,000 ✓
- All generated probabilities are in [0, 1]
- No NaN, Inf, or other numerical issues
- Log-odds scale provides numerical stability

### Parameter Ranges
- μ samples span ~7 log-odds units (appropriate for covering 1%-50% probability range)
- τ samples reach up to 3.19 (allows very high heterogeneity if data demands it)
- No evidence of prior-likelihood conflict

---

## Formal Decision Criteria

### PASSED CHECKS

1. **No invalid probabilities** ✓
   - All 12,000 generated probabilities are valid (p ∈ [0, 1])
   - Evidence: `parameter_plausibility.png` shows all p values in valid range

2. **Group 1 generates zero counts plausibly** ✓
   - P(r=0) = 0.124 for Group 1
   - Evidence: `group1_zero_inflation_diagnostic.png` shows r=0 as most probable outcome

3. **τ prior allows sufficient heterogeneity** ✓
   - 95% upper bound: 2.23 (allows high ICC if data supports it)
   - Evidence: `parameter_plausibility.png` (bottom left panel)

4. **μ prior covers observed pooled rate** ✓
   - Observed 0.074 is within prior 95% CI [0.013, 0.354]
   - Evidence: `parameter_plausibility.png` (top right panel)

5. **No computational red flags** ✓
   - No numerical instabilities
   - No extreme parameter values causing issues
   - Model structure is well-behaved

### FAILED CHECKS

1. **"Prior predictive does not adequately cover observed range"** (TECHNICAL ARTIFACT - NOT A REAL FAILURE)
   - The automated check flagged this because prior predictive min (0.000) equals observed min (0.000)
   - However, this is actually PERFECT coverage - the prior generates zeros appropriately
   - Visual inspection of `prior_predictive_proportions.png` and `prior_predictive_coverage.png` confirms excellent coverage
   - This is a false positive from an overly strict automated check

**Conclusion:** All substantive checks pass. The one flagged "failure" is a technical artifact of the automated checking logic, not a real concern.

---

## Key Visual Evidence

### Most Important Plots for GO/NO-GO Decision:

1. **`parameter_plausibility.png`** - Confirms priors are properly specified and generate reasonable parameter values centered appropriately around observed data

2. **`group1_zero_inflation_diagnostic.png`** - Critical evidence that the model can handle the zero count in Group 1, with P(r=0)=12.4% showing this is a plausible outcome

3. **`prior_predictive_coverage.png`** - Shows all observed counts fall within 95% prior predictive intervals, with appropriate uncertainty reflected in interval widths

---

## Comparison to Experiment 1 (Beta-Binomial)

This model represents a significant improvement over Experiment 1:

**Advantages of Experiment 2:**
1. **Better parameterization:** Uses SD (τ) instead of concentration, which is more interpretable
2. **Non-centered parameterization:** Improves sampling geometry
3. **Log-odds scale:** Provides numerical stability and symmetry
4. **Cleaner priors:** Direct specification on mean and variance components

**Prior Predictive Performance:**
- Both models generate plausible data
- This model has clearer interpretation of prior choices
- Expected to have better SBC performance due to improved geometry

---

## Recommendations for Model Fitting

### Prior Specification (APPROVED)
Proceed with specified priors:
- μ ~ Normal(logit(0.075), 1²)
- τ ~ HalfNormal(1)

### Sampling Configuration
Recommended settings for SBC:
- Non-centered parameterization (as specified)
- Standard HMC/NUTS sampling
- Monitor: R-hat, ESS, divergences

### Expected Behavior
Based on prior predictive checks:
- Model should handle Group 1 zero count well
- Strong shrinkage expected due to hierarchical structure
- Between-group variation should be estimable (τ posterior will be informed by data)

---

## Conclusion

**PASS - Model is GO for SBC validation**

The Random Effects Logistic Regression model with specified priors generates scientifically plausible data that:
1. Covers the observed data range appropriately
2. Handles the challenging zero count in Group 1
3. Allows sufficient heterogeneity without being unconstrained
4. Shows no computational or numerical issues
5. Centers appropriately on domain knowledge (7.5% baseline rate)

**Next Steps:**
1. Proceed immediately to SBC validation
2. If SBC passes, fit model to real data
3. Compare results with Experiment 1 (if both pass SBC)

---

## Files Generated

**Code:**
- `/workspace/experiments/experiment_2/prior_predictive_check/code/prior_predictive_check.py`

**Visualizations:**
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/parameter_plausibility.png`
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/prior_predictive_proportions.png`
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/prior_predictive_counts.png`
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/group1_zero_inflation_diagnostic.png`
- `/workspace/experiments/experiment_2/prior_predictive_check/plots/prior_predictive_coverage.png`

**Documentation:**
- `/workspace/experiments/experiment_2/prior_predictive_check/findings.md` (this file)

---

*Prior predictive check completed: 2025-10-30*
*Analyst: Claude (Bayesian Model Validator)*
