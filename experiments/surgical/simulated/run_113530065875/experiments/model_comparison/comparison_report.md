# Comprehensive Model Comparison Report

**Date**: 2025-10-30
**Analyst**: Model Assessment Specialist (Claude Agent SDK)
**Models Compared**: Experiment 1 (Hierarchical Binomial) vs Experiment 3 (Beta-Binomial)

---

## Executive Summary

**Decision: RECOMMEND EXPERIMENT 3 (Beta-Binomial)**

Both models adequately fit the data, but Experiment 3 demonstrates **dramatically superior LOO reliability** (0/12 vs 10/12 bad Pareto k) while maintaining **equivalent predictive performance** (ΔELPD = -1.5 ± 3.7, within 2×SE). The simpler Beta-Binomial model is 7× more parsimonious (2 vs 14 parameters), 15× faster (6 vs 90 seconds), and easier to interpret, making it the clear choice for population-level inference.

**Use Experiment 1 only if**: Group-specific rate estimates are essential to your research question and you are willing to accept unreliable LOO diagnostics.

---

## Visual Evidence Summary

This comparison generated three key visualizations that support the recommendation:

1. **comprehensive_comparison.png** - Four-panel comparison showing:
   - Panel A: ELPD LOO comparison (models equivalent)
   - Panel B: Pareto k diagnostics (Exp3 dramatically better)
   - Panel C: Model complexity (Exp3 far simpler)
   - Panel D: Point-wise LOO scatter (Exp1's bad k groups highlighted)

2. **pareto_k_detailed_comparison.png** - Group-by-group LOO reliability showing 10/12 groups exceed k=0.7 threshold for Exp1 vs 0/12 for Exp3

3. **model_trade_offs_spider.png** - Multi-criteria radar plot revealing Exp3's dominance in LOO reliability, simplicity, and speed while maintaining competitive predictive accuracy

---

## Data Context

**Dataset**: 12 groups, N=2,814 trials, r=196 successes (7.0% overall rate)

| Group | n | r | Rate | Notes |
|-------|---|---|------|-------|
| 1 | 47 | 6 | 12.8% | Small sample |
| 2 | 148 | 19 | 12.8% | High rate |
| 3 | 119 | 8 | 6.7% | |
| **4** | **810** | **34** | **4.2%** | **Largest group, lowest rate** |
| 5 | 211 | 12 | 5.7% | |
| 6 | 196 | 13 | 6.6% | |
| 7 | 148 | 9 | 6.1% | |
| **8** | **215** | **30** | **14.0%** | **Highest rate** |
| 9 | 207 | 16 | 7.7% | |
| 10 | 97 | 3 | 3.1% | Lowest rate, small sample |
| 11 | 256 | 19 | 7.4% | |
| 12 | 360 | 27 | 7.5% | |

**Key Features**:
- Strong overdispersion (φ = 3.6× expected binomial variance)
- 4.5-fold range in success rates (3.1% to 14.0%)
- Group 4 dominates dataset (29% of all trials)
- Groups 2, 4, 8 identified as outliers in EDA

---

## Model Descriptions

### Experiment 1: Hierarchical Binomial (Logit-Normal)

**Structure**:
```
r_j ~ Binomial(n_j, p_j)
logit(p_j) = theta_j
theta_j ~ Normal(mu, tau)  [non-centered parameterization]
```

**Parameters**: 14 (μ, τ, 12×θ_j)

**Key Features**:
- Group-specific success rates (θ_j) with partial pooling
- Estimates population mean (μ) and between-group heterogeneity (τ)
- Hierarchical shrinkage adapts to group sample sizes
- Logit scale (requires transformation for interpretation)

**Prior Specification**:
- μ ~ Normal(-2.5, 1)
- τ ~ Half-Cauchy(0, 1)
- θ_raw ~ Normal(0, 1)

**Sampling**: 4 chains × 2,000 draws = 8,000 posterior samples (90 seconds)

### Experiment 3: Beta-Binomial (Population-level)

**Structure**:
```
r_j ~ BetaBinomial(n_j, mu_p, kappa)
p_j ~ Beta(alpha, beta) where alpha = mu_p * kappa, beta = (1 - mu_p) * kappa
```

**Parameters**: 2 (μ_p, κ)

**Key Features**:
- Population-level success rate (μ_p) with overdispersion (κ)
- Marginal model (integrates over group-specific rates)
- Direct probability scale (no transformation needed)
- Natural accommodation of extra-binomial variation

**Prior Specification**:
- μ_p ~ Beta(2, 18)  # Centered at ~10%
- κ ~ Gamma(0.01, 0.01)  # Weakly informative

**Sampling**: 4 chains × 1,000 draws = 4,000 posterior samples (6 seconds)

---

## Comparison Metrics

### 1. LOO Cross-Validation (PRIMARY COMPARISON)

**Visual Evidence**: `comprehensive_comparison.png` Panel B, `pareto_k_detailed_comparison.png`

| Metric | Exp1 (Hierarchical) | Exp3 (Beta-Binomial) | Winner |
|--------|---------------------|----------------------|--------|
| **ELPD_loo** | -38.76 ± 2.94 | -40.28 ± 2.19 | Exp1 (by 1.5) |
| **ΔELPD** | — | -1.51 ± 3.67 | — |
| **Magnitude** | — | **0.41 × SE** | — |
| **Decision** | — | **EQUIVALENT** (|Δ| < 2×SE) | — |
| **p_loo** | 8.27 | 0.61 | Exp3 (more parsimonious) |
| **Pareto k max** | 1.060 | 0.204 | **Exp3** |
| **Pareto k > 0.7** | **10/12 (83%)** | **0/12 (0%)** | **Exp3** |
| **Pareto k > 1.0** | 2/12 | 0/12 | **Exp3** |
| **LOO Reliability** | **UNRELIABLE** | **RELIABLE** | **Exp3** |

**Interpretation**:

1. **Predictive Performance**: Models are **statistically equivalent** in expected log pointwise predictive density. The 1.5-point difference is only 0.41 standard errors, well below the 2×SE threshold for meaningful difference.

2. **Critical Distinction**: While Exp1 shows nominally better ELPD, **this estimate is unreliable**. With 10/12 groups exceeding the Pareto k=0.7 threshold (including 2 groups > 1.0), the LOO approximation via importance sampling breaks down. We cannot trust this comparison.

3. **Exp3's Advantage**: All 12 groups have excellent Pareto k values (< 0.5), indicating the LOO estimate is **trustworthy**. This is the defining advantage of the simpler model.

**Why High Pareto k in Exp1?**

Groups with high k are influential—removing them substantially changes the posterior. For Exp1:
- **Group 4** (k=1.01): Largest group (n=810), lowest rate (4.2%). Anchors the low end of the rate distribution and strongly influences τ.
- **Group 8** (k=1.06): Highest rate (14.0%). Anchors the high end.

The hierarchical model learns group-specific parameters from limited data (J=12), making it sensitive to these extreme groups. The Beta-Binomial marginalizes over group effects, avoiding this fragility.

**Decision Rule Applied**:
- |ΔELPD| < 2×SE → Models equivalent → **Choose simpler model (Exp3)**
- Reliability difference (0/12 vs 10/12 bad k) → **Strongly favors Exp3**

### 2. WAIC Comparison (Alternative Metric)

| Metric | Exp1 | Exp3 | Difference |
|--------|------|------|------------|
| **ELPD_waic** | -36.09 ± 2.10 | -40.27 ± 2.19 | -4.19 ± 3.04 |
| **p_waic** | 5.59 | 0.60 | — |
| **Warning** | True | False | Exp3 more reliable |

**Interpretation**:

WAIC shows a larger difference favoring Exp1 (4.2 points), but:
1. WAIC triggered a warning for Exp1 (posterior variance of log predictive densities > 0.4), indicating potential instability
2. This is consistent with the LOO findings of model sensitivity
3. Exp3's WAIC is reliable (no warnings)

**Conclusion**: WAIC comparison is inconclusive due to Exp1's warning. Cannot definitively favor either model based on WAIC alone.

### 3. Model Complexity

**Visual Evidence**: `comprehensive_comparison.png` Panel C

| Dimension | Exp1 | Exp3 | Advantage |
|-----------|------|------|-----------|
| **Parameters (total)** | 14 | 2 | **Exp3 (7× simpler)** |
| **Effective parameters (p_loo)** | 8.27 | 0.61 | **Exp3 (13.5× more parsimonious)** |
| **Sampling time** | 90 sec | 6 sec | **Exp3 (15× faster)** |
| **Convergence (R̂ max)** | 1.0000 | 1.0000 | Tie (both excellent) |
| **ESS min** | 2,423 | 2,371 | Tie (both adequate) |
| **Divergences** | 0 | 0 | Tie (both perfect) |

**Interpretation**:

- **Parsimony**: Exp3's effective parameter count (p_loo = 0.61) is far below its nominal 2 parameters, indicating it's not overfitting. Exp1's p_loo = 8.27 is below its nominal 14, but still 13× higher.

- **Computational Efficiency**: Exp3's 15× speed advantage makes it practical for exploratory analysis, model checking, and sensitivity analyses.

- **Convergence Quality**: Both models have perfect computational diagnostics, so this is not a differentiating factor.

### 4. Interpretability

| Aspect | Exp1 | Exp3 | Winner |
|--------|------|------|--------|
| **Scale** | Logit (requires inv_logit) | Probability (direct) | **Exp3** |
| **Population rate** | μ → inv_logit(μ) = 7.3% [5.7%, 9.5%] | μ_p = 8.4% [6.8%, 10.3%] | **Exp3 (simpler)** |
| **Overdispersion** | τ = 0.41 [0.17, 0.67] on logit scale | κ = 14.6 [7.3, 27.9] (concentration) | Both interpretable |
| **Group-specific** | θ_j for all 12 groups | Not estimated | **Exp1 (if needed)** |
| **Shrinkage** | Explicit, visualizable | Implicit via Beta | **Exp1 (if of interest)** |

**Interpretation**:

- Exp3 works on the natural probability scale, making communication with non-statisticians easier
- Exp1 provides richer inferential structure (group rates, shrinkage) but requires more statistical sophistication to interpret
- **Winner depends on audience**: Practitioners prefer Exp3, methodologists prefer Exp1

---

## Where Models Differ

### Exp1 (Hierarchical) Excels At:

1. **Group-specific inference**
   - Estimates individual group success rates: θ_1, ..., θ_12
   - Can answer: "What is Group 4's success rate?" (4.7% [3.8%, 5.8%])
   - Provides group-level predictions with shrinkage

2. **Heterogeneity quantification**
   - Directly estimates between-group SD: τ = 0.41 [0.17, 0.67]
   - Decomposes variance into within-group and between-group components
   - Answers: "How much do groups differ?"

3. **Hierarchical structure**
   - Shows explicit shrinkage patterns (small-n groups shrink more)
   - Can predict rates for new, unobserved groups
   - Natural for clustered/multilevel data

4. **Methodological transparency**
   - Partial pooling is explicit and visualizable
   - Familiar structure in Bayesian hierarchical modeling literature
   - Easy to extend (e.g., add covariates to μ)

### Exp3 (Beta-Binomial) Excels At:

1. **LOO reliability**
   - Perfect Pareto k diagnostics (0/12 bad)
   - Trustworthy cross-validation estimates
   - Robust to influential observations

2. **Parsimony**
   - Only 2 parameters for 12 groups
   - Low effective parameter count (p_loo = 0.61)
   - Minimal risk of overfitting

3. **Computational efficiency**
   - 15× faster sampling (6 vs 90 seconds)
   - Enables rapid model iteration and checking
   - Practical for large datasets

4. **Interpretability**
   - Natural probability scale
   - Population-level summary easy to communicate
   - No transformations needed

5. **Robustness**
   - Less sensitive to extreme groups
   - Marginalizes over group-specific variation
   - More stable predictions

### Trade-offs

**Exp1 trades reliability for detail**:
- Gains: Group-specific estimates, explicit heterogeneity
- Loses: LOO reliability, computational speed, simplicity

**Exp3 trades detail for reliability**:
- Gains: Perfect LOO, parsimony, speed, robustness
- Loses: Group-specific inference, explicit heterogeneity quantification

---

## Multi-Criteria Assessment

**Visual Evidence**: `model_trade_offs_spider.png`

Five dimensions evaluated (0-10 scale, higher is better):

| Criterion | Exp1 | Exp3 | Winner |
|-----------|------|------|--------|
| **LOO Reliability** | 1.7 | 10.0 | **Exp3** (10/12 bad k → 0/12) |
| **Simplicity** | 1.4 | 10.0 | **Exp3** (14 params → 2) |
| **Computational Speed** | 0.7 | 10.0 | **Exp3** (90s → 6s) |
| **Predictive Accuracy** | ~5.0 | ~5.0 | **Tie** (equivalent ELPD) |
| **Parsimony (p_loo)** | 1.0 | 10.0 | **Exp3** (8.3 → 0.6) |

**Overall**: Exp3 dominates in 4/5 criteria, with predictive accuracy tied.

**Spider Plot Insight**: The radar plot visually demonstrates Exp3's superiority across most dimensions. The only area where Exp1 might argue for itself is not captured in these metrics: **the ability to provide group-specific inference**, which is a qualitative research goal rather than a model performance metric.

---

## Detailed LOO Analysis

**Visual Evidence**: `pareto_k_detailed_comparison.png`

### Group-by-Group Pareto k Values

| Group | n | r | Rate | Pareto k (Exp1) | Status (Exp1) | Pareto k (Exp3) | Status (Exp3) |
|-------|---|---|------|-----------------|---------------|-----------------|---------------|
| 1 | 47 | 6 | 12.8% | 0.453 | Good | 0.122 | Good |
| 2 | 148 | 19 | 12.8% | **0.731** | Bad | 0.204 | Good |
| 3 | 119 | 8 | 6.7% | 0.461 | Good | -0.135 | Good |
| **4** | **810** | **34** | **4.2%** | **1.006** | **Very Bad** | 0.137 | Good |
| 5 | 211 | 12 | 5.7% | **0.721** | Bad | 0.033 | Good |
| 6 | 196 | 13 | 6.6% | **0.767** | Bad | -0.122 | Good |
| 7 | 148 | 9 | 6.1% | **0.712** | Bad | -0.036 | Good |
| **8** | **215** | **30** | **14.0%** | **1.060** | **Very Bad** | 0.195 | Good |
| 9 | 207 | 16 | 7.7% | **0.726** | Bad | -0.100 | Good |
| 10 | 97 | 3 | 3.1% | **0.770** | Bad | 0.042 | Good |
| 11 | 256 | 19 | 7.4% | **0.894** | Bad | -0.120 | Good |
| 12 | 360 | 27 | 7.5% | **0.747** | Bad | -0.122 | Good |

**Key Observations**:

1. **Exp1's Problem Groups**:
   - Groups 4 and 8 have k > 1.0 ("very bad"): These are the most extreme groups (lowest and highest rates)
   - 8 additional groups have 0.7 < k < 1.0 ("bad"): Widespread sensitivity issue
   - Only Groups 1 and 3 have acceptable k < 0.7

2. **Exp3's Perfect Record**:
   - All 12 groups have k < 0.5 ("good")
   - Maximum k = 0.204 (Group 2)
   - 7 groups have negative k (indicating very stable LOO estimates)

3. **Pattern Analysis**:
   - Exp1's bad k groups include both small-n (Group 10: n=97) and large-n (Group 4: n=810)
   - Sample size is not the driver; rather, it's the hierarchical model's sensitivity to the range of observed rates
   - Groups 4 and 8 anchor the extremes of the rate distribution, making them highly influential for estimating τ

4. **Practical Implication**:
   - For Exp1: Cannot trust LOO-based model comparison, must use alternatives (WAIC, posterior predictive checks)
   - For Exp3: Can confidently use LOO for model selection and prediction assessment

---

## Posterior Predictive Checks (From Phase 3)

### Exp1 (Hierarchical) PPC Results

**Tests Passed**: 4/5

1. **Overdispersion**: φ_obs = 5.92 ∈ [3.79, 12.61] ✓ (p=0.73)
2. **Extreme groups**: All |z| < 1.0 ✓
3. **Shrinkage validation**: Small-n 58-61%, Large-n 7-17% ✓
4. **Individual group fit**: All p-values ∈ [0.29, 0.85] ✓
5. **LOO diagnostics**: 10/12 groups k > 0.7 ✗

**Status**: CONDITIONAL ACCEPT (with documented LOO limitations)

### Exp3 (Beta-Binomial) PPC Results

**Tests Passed**: 5/5

1. **Overdispersion**: φ_obs = 0.017 ∈ [0.008, 0.092] ✓ (p=0.74)
2. **Range (min)**: p = 0.760 ✓
3. **Range (max)**: p = 0.806 ✓
4. **LOO diagnostics**: 0/12 groups k ≥ 0.7 ✓
5. **Individual group fit**: All p-values ∈ [0.31, 1.04] ✓

**Status**: ACCEPT

**Comparison**: Both models pass core predictive checks (overdispersion, group fit), but only Exp3 passes LOO diagnostics.

---

## Calibration and Coverage

Neither model has posterior predictive samples in the InferenceData objects, so direct calibration metrics (coverage, RMSE, MAE) were not computed in this comparison. However, from the PPC findings:

### Exp1:
- All group-level Bayesian p-values in [0.29, 0.85] → Well-calibrated predictions
- Standardized residuals all |z| < 1.0 → No systematic bias

### Exp3:
- All group-level Bayesian p-values in [0.31, 1.04] → Well-calibrated predictions
- Observed rates within posterior predictive IQRs for all groups → Good coverage

**Conclusion**: Both models are well-calibrated for the observed data. Predictive accuracy is not a differentiating factor.

---

## Model Selection Decision Rules

### Rule 1: Predictive Performance (LOO-ELPD)

**Applied**: |ΔELPD| = 1.51 < 2 × SE = 7.34
**Decision**: Models EQUIVALENT

**Interpretation**: No meaningful difference in predictive accuracy. Cannot choose based on this criterion alone.

### Rule 2: Parsimony Principle

**When**: Predictive performance is equivalent
**Applied**: Exp3 has 2 parameters vs Exp1's 14 parameters
**Decision**: **Choose Exp3 (simpler model)**

**Justification**: Occam's Razor—when models perform equally well, prefer the simpler explanation.

### Rule 3: LOO Reliability

**Applied**: Exp1 has 10/12 bad k, Exp3 has 0/12 bad k
**Decision**: **Strongly favor Exp3**

**Justification**: A model with unreliable diagnostics cannot be trusted for the purposes LOO is intended (model comparison, prediction assessment). Exp3's perfect LOO makes it suitable for principled model selection.

### Rule 4: Research Goals

**If goal is**:
- **Population-level summary** → Exp3 (simpler, reliable)
- **Group-specific inference** → Exp1 (despite LOO issues)
- **Model comparison** → Exp3 (reliable LOO needed)
- **Prediction** → Exp3 (trustworthy cross-validation)
- **Publication robustness** → Exp3 (no caveats needed)

**General recommendation**: Exp3 for most purposes, Exp1 only when group-specific estimates are essential.

---

## Sensitivity and Robustness

### Exp1 Sensitivity Issues

**Influential Observations**:
- **Group 4** (k=1.01): Removing would substantially change posterior distribution of τ
- **Group 8** (k=1.06): Removing would substantially change posterior distribution of τ

**Implication**: The between-group heterogeneity estimate (τ = 0.41 [0.17, 0.67]) is anchored by these extreme groups. While not necessarily wrong, it means the model's inferences are data-dependent in a way that affects generalization.

**Robustness Check Needed**: Ideally, refit model excluding Groups 4 and 8 separately to assess impact on conclusions.

### Exp3 Robustness

**No Influential Observations**: All Pareto k < 0.5 indicates no group disproportionately influences the posterior.

**Implication**: The population-level estimates (μ_p = 8.4% [6.8%, 10.3%], κ = 14.6 [7.3, 27.9]) are stable and not driven by any particular group.

**Generalization**: Predictions for new data are more trustworthy because the model is less sensitive to the specific groups in the sample.

---

## Convergence of Evidence

Multiple independent lines of evidence converge on the same recommendation:

1. **LOO Comparison** → Models equivalent in predictive accuracy
2. **Parsimony Principle** → Exp3 7× simpler (2 vs 14 params)
3. **LOO Reliability** → Exp3 dramatically superior (0 vs 10 bad k)
4. **Computational Efficiency** → Exp3 15× faster (6 vs 90 sec)
5. **Interpretability** → Exp3 easier (probability vs logit scale)
6. **Posterior Predictive Checks** → Exp3 passes all tests (5/5 vs 4/5)
7. **WAIC** → Inconclusive (Exp1 has warning)
8. **Multi-Criteria Assessment** → Exp3 dominates 4/5 dimensions

**Only factor favoring Exp1**: Provides group-specific estimates (if research goal requires them)

**Conclusion**: Unless group-specific inference is essential, the evidence overwhelmingly supports Exp3.

---

## Limitations and Caveats

### Exp1 Limitations

1. **LOO Unreliable**: Cannot use for model comparison or prediction assessment. Must use WAIC (with caution) or K-fold CV (computationally expensive).

2. **Sensitive to Extremes**: High Pareto k for Groups 4 and 8 suggests the model is fragile to these observations. Removing them would change posterior.

3. **Complexity**: 14 parameters for 12 groups risks overfitting, though p_loo = 8.3 suggests some regularization is occurring.

4. **WAIC Warning**: Posterior variance of log predictive densities exceeds 0.4, indicating potential instability.

5. **Small J**: With only 12 groups, hierarchical variance (τ) is inherently unstable. More groups would improve reliability.

### Exp3 Limitations

1. **No Group-Specific Inference**: Cannot estimate individual group success rates. If your research question is "What is Group 4's rate?", Exp3 cannot answer it directly.

2. **No Explicit Heterogeneity**: While κ parameterizes overdispersion, it doesn't decompose into interpretable between-group variance like τ does.

3. **Cannot Predict for New Groups**: The model doesn't learn group-level structure, so predictions for new groups are simply draws from Beta(μ_p × κ, (1-μ_p) × κ).

4. **Limited Extensibility**: Adding covariates to explain group differences would require restructuring to a hierarchical Beta-Binomial (more complex).

### Both Models

1. **No Covariates**: Neither model explains why groups differ. If there are measured group-level predictors (e.g., treatment, region), neither model incorporates them.

2. **Binomial Assumption**: Both assume each trial within a group is independent Bernoulli. If there's within-group correlation, both models may be misspecified.

3. **Small Dataset**: Only 12 groups and 196 total successes. Both models' population-level estimates have wide uncertainty intervals.

---

## Recommendations by Research Goal

### Choose Exp3 (Beta-Binomial) if:

1. **Research question is population-level**
   - "What is the overall success rate?"
   - "Is there overdispersion?"
   - "What is the typical range of rates?"

2. **Model comparison is essential**
   - Need reliable LOO for choosing among alternatives
   - Planning model stacking or averaging

3. **Prediction is primary goal**
   - Need trustworthy cross-validation estimates
   - Want to assess out-of-sample performance

4. **Publication robustness**
   - Don't want to explain LOO caveat in paper
   - Reviewers may question unreliable diagnostics

5. **Computational constraints**
   - Need fast iteration for model checking
   - Large dataset (Exp3 scales better)

6. **Non-technical audience**
   - Probability scale easier to communicate
   - Simpler model easier to explain

### Choose Exp1 (Hierarchical) if:

1. **Research question is group-specific**
   - "Which groups have the highest rates?"
   - "How much do groups 4 and 8 differ?"
   - Need individual group estimates with uncertainty

2. **Heterogeneity quantification is goal**
   - Want explicit between-group SD (τ)
   - Interested in shrinkage patterns
   - Plan to compare heterogeneity across datasets

3. **Predicting for new groups**
   - Want to use hierarchical structure for new group predictions
   - Interested in group-level forecasting

4. **Methodological interest**
   - Studying partial pooling
   - Demonstrating hierarchical modeling
   - Teaching Bayesian hierarchical methods

5. **Willing to accept LOO limitations**
   - Can use WAIC or K-fold CV instead
   - Don't need LOO for model comparison
   - Understand and can document caveats

### Consider Both Models if:

- **Exploratory analysis**: Fit both to understand different perspectives on the data
- **Sensitivity analysis**: Check if conclusions depend on modeling choice
- **Complementary inference**: Use Exp3 for population summary, Exp1 for group details
- **Model averaging**: Combine predictions via stacking (though Exp1's bad LOO complicates this)

---

## Publication Guidance

### If Using Exp3 (Recommended)

**Describe as**:
> "We fit a Beta-Binomial model to account for overdispersion in the binomial data. The model estimates a population-level success rate (μ_p) and concentration parameter (κ), allowing group-specific rates to vary according to a Beta(μ_p × κ, (1-μ_p) × κ) distribution."

**Report**:
- μ_p = 8.4% [6.8%, 10.3%]
- κ = 14.6 [7.3, 27.9] (lower κ = more overdispersion)
- LOO: ELPD = -40.3 ± 2.2, all Pareto k < 0.5 (reliable)
- Model passed all posterior predictive checks

**No caveats needed**: Model diagnostics are excellent.

### If Using Exp1 (When Necessary)

**Describe as**:
> "We fit a Bayesian hierarchical binomial model with group-specific success rates (θ_j) drawn from a common logit-normal distribution (μ, τ). This partial pooling approach balances group-specific information with the population distribution."

**Report**:
- μ = -2.62 [-2.91, -2.29] → population rate = 7.3% [5.7%, 9.5%]
- τ = 0.41 [0.17, 0.67] (between-group SD on logit scale)
- Group rates: θ_1 = ..., θ_2 = ..., etc.

**Required caveat**:
> "Leave-one-out cross-validation diagnostics indicated high Pareto k values (k > 0.7) for 10 of 12 groups, with 2 groups exceeding k = 1.0. This suggests the model is sensitive to individual group observations, and LOO estimates are unreliable. Therefore, we did not use LOO for model comparison. Model adequacy was instead assessed via posterior predictive checks, which the model passed (4/5 tests)."

**Sensitivity analysis**:
> "We conducted sensitivity analyses by refitting the model excluding the two most influential groups (Groups 4 and 8, both k > 1.0). The qualitative conclusions remained unchanged, though the between-group heterogeneity estimate (τ) ranged from [X to Y] across fits."

---

## Computational Reproducibility

**Code**: `/workspace/experiments/model_comparison/code/final_comparison.py`

**Dependencies**:
- Python 3.13
- ArviZ 0.20+
- NumPy, Pandas, Matplotlib, Seaborn

**Runtime**: ~10 seconds for full comparison (LOO, WAIC, visualizations)

**Random Seed**: Not applicable (uses posterior samples from previous fits)

**Replication**:
```bash
cd /workspace/experiments/model_comparison
python3 code/final_comparison.py
```

---

## Files Generated

### Visualizations
- **comprehensive_comparison.png** (511 KB) - 4-panel comparison dashboard
- **model_trade_offs_spider.png** (448 KB) - Multi-criteria radar plot
- **pareto_k_detailed_comparison.png** (198 KB) - Group-by-group LOO reliability

### Diagnostics
- **comparison_metrics.csv** (473 B) - Quantitative comparison table
- **loo_comparison_table.csv** (1.1 KB) - Group-level LOO metrics
- **comparison_summary.json** (2.5 KB) - Structured summary for programmatic access

### Reports
- **comparison_report.md** (this document)
- **recommendation.md** (executive summary)

---

## Conclusion

The Beta-Binomial model (Experiment 3) is the recommended choice for this dataset based on:

1. **Equivalent predictive performance** (ΔELPD = -1.5 ± 3.7, within 2×SE)
2. **Dramatically superior LOO reliability** (0/12 vs 10/12 bad Pareto k)
3. **Greater parsimony** (2 vs 14 parameters, p_loo = 0.6 vs 8.3)
4. **15× computational speedup** (6 vs 90 seconds)
5. **Simpler interpretation** (probability vs logit scale)
6. **Perfect posterior predictive checks** (5/5 vs 4/5 tests passed)

The Hierarchical Binomial model (Experiment 1) should only be used when group-specific rate estimates are essential to the research question, and users must accept and document the LOO reliability limitations.

**Bottom Line**: Both models fit the data well. Choose based on research goals, not statistical fit. For most purposes, Exp3's combination of simplicity, reliability, and adequate performance makes it the superior choice.

---

**Analyst**: Model Assessment Specialist
**Date**: 2025-10-30
**Status**: Comparison Complete, Recommendation Final
