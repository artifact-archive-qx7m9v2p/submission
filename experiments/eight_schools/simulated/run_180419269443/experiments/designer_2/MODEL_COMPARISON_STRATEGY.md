# Model Comparison Strategy: A Visual Guide

## Overview of Three Model Classes

This document provides a visual/conceptual guide to understanding when and why each model class succeeds or fails.

---

## Model Space Landscape

```
Simplicity                                           Complexity
←─────────────────────────────────────────────────────────────→

Common Effect    Hierarchical     t-Hierarchical    Mixture      Dirichlet
     │           Normal (D1)      (Model 1)        (Model 2)    Process (M3)
     │               │                │                │             │
  No hetero      Low hetero      Heavy tails      Subgroups    Unknown K
  tau = 0        tau ~ N(0,σ)    + nu param      K=2 fixed    K learned
```

**Designer 1's domain:** Left side (classical models)
**Designer 2's domain:** Right side (robust alternatives)

**Key question:** How far right do we need to go?

---

## Decision Flowchart with Thresholds

```
START: Fit Model 1 (t-distribution with nu)
│
├─ Check nu posterior
│  │
│  ├─ nu > 50 (posterior mean)
│  │  ├─ Interpretation: Data prefer normal tails
│  │  ├─ Action: Use Designer 1's hierarchical normal
│  │  └─ Justification: Simpler model adequate
│  │
│  ├─ 20 ≤ nu ≤ 50
│  │  ├─ Interpretation: Slightly heavy tails
│  │  ├─ Action: Report Model 1 as primary
│  │  └─ Justification: Robustness warranted
│  │
│  └─ nu < 20
│     ├─ Interpretation: Strong heavy tails
│     ├─ Action: Investigate data quality + fit Model 2
│     └─ Justification: May indicate subpopulations
│
├─ If nu < 20 OR other evidence of subgroups:
│  │
│  FIT Model 2 (Mixture with pi)
│  │
│  ├─ Check pi posterior
│  │  │
│  │  ├─ pi < 0.1 OR pi > 0.9
│  │  │  ├─ Interpretation: Single cluster (collapsed)
│  │  │  ├─ Action: Return to Model 1
│  │  │  └─ Justification: Data don't support mixture
│  │  │
│  │  ├─ 0.1 ≤ pi ≤ 0.9 AND |mu_1 - mu_2| > 5
│  │  │  ├─ Interpretation: Genuine two-component mixture
│  │  │  ├─ Action: Report Model 2 as primary
│  │  │  └─ Justification: Hidden subpopulations exist
│  │  │
│  │  └─ 0.1 ≤ pi ≤ 0.9 BUT |mu_1 - mu_2| < 5
│  │     ├─ Interpretation: Weak separation
│  │     ├─ Action: Report range, emphasize uncertainty
│  │     └─ Justification: Borderline case
│  │
│  └─ If evidence suggests >2 clusters:
│     │
│     FIT Model 3 (Dirichlet Process)
│     │
│     ├─ Check K_eff posterior
│     │  │
│     │  ├─ K_eff < 1.5
│     │  │  ├─ Interpretation: Single cluster
│     │  │  ├─ Action: Use hierarchical normal
│     │  │  └─ Justification: Complexity not justified
│     │  │
│     │  ├─ 1.5 ≤ K_eff < 2.5
│     │  │  ├─ Interpretation: Two clusters
│     │  │  ├─ Action: Cross-validate with Model 2
│     │  │  └─ Justification: Confirm with simpler mixture
│     │  │
│     │  └─ K_eff ≥ 2.5
│     │     ├─ Interpretation: Complex heterogeneity
│     │     ├─ Action: Question pooling assumption entirely
│     │     └─ Justification: May need study-level estimates
│     │
│     └─ Final check: Compare LOO-CV
│        ├─ If DP not better than simpler models
│        │  └─ Use simpler model (parsimony)
│        └─ If DP substantially better (ΔELPD > 4)
│           └─ Report DP, but validate findings
```

---

## Quantitative Thresholds Summary

| Model | Parameter | Threshold | Interpretation | Action |
|-------|-----------|-----------|----------------|--------|
| **Model 1** | nu > 50 | Normal adequate | Use hierarchical normal |
| | 20 ≤ nu ≤ 50 | Mild heavy tails | Report Model 1 |
| | nu < 20 | Strong heavy tails | Investigate + try Model 2 |
| **Model 2** | pi < 0.1 | Cluster 1 dominates | Single cluster model |
| | pi > 0.9 | Cluster 2 dominates | Single cluster model |
| | 0.1 ≤ pi ≤ 0.9 | Potential mixture | Check mu separation |
| | \|mu_1 - mu_2\| > 5 | Well-separated | Report mixture |
| | \|mu_1 - mu_2\| < 5 | Weak separation | Uncertain |
| **Model 3** | K_eff < 1.5 | Single cluster | Use hierarchical normal |
| | 1.5 ≤ K_eff < 2.5 | Two clusters | Validate with Model 2 |
| | K_eff ≥ 2.5 | Complex | Question pooling |

---

## LOO-CV Comparison Framework

### Pairwise Comparisons

```
Compare all fitted models:

Model A vs Model B:
ΔELPD = ELPD_A - ELPD_B
SE_diff = sqrt(SE_A² + SE_B²)

Decision Rule:
├─ If ΔELPD > 2 × SE_diff  → Model A significantly better
├─ If ΔELPD < -2 × SE_diff → Model B significantly better
└─ If |ΔELPD| < 2 × SE_diff → Models equivalent, prefer simpler
```

### Expected Comparisons

Assuming I²=2.9% holds:

| Comparison | Expected Result | Implication |
|------------|----------------|-------------|
| Model 1 vs Normal | ΔELPD ≈ 0 | nu > 50, both equivalent |
| Model 2 vs Model 1 | Model 1 better | Mixture collapses |
| Model 3 vs Model 1 | Model 1 better | K_eff ≈ 1, too complex |
| D1 Normal vs D2 Model 1 | Equivalent | Robust to t-tails |

**Surprising outcome:**
- If Model 2 or 3 substantially better → Complex heterogeneity not visible in I²

---

## Posterior Predictive Check Strategy

### Test Statistics to Compute

For each model, generate y_rep and check:

```python
test_statistics = {
    'mean': np.mean(y_rep, axis=1),
    'sd': np.std(y_rep, axis=1),
    'min': np.min(y_rep, axis=1),
    'max': np.max(y_rep, axis=1),
    'range': np.ptp(y_rep, axis=1),
    'q25': np.percentile(y_rep, 25, axis=1),
    'q75': np.percentile(y_rep, 75, axis=1)
}

for stat_name, stat_values in test_statistics.items():
    p_value = np.mean(stat_values >= observed_stat[stat_name])
    print(f"{stat_name}: p = {p_value:.3f}")
    if p_value < 0.05 or p_value > 0.95:
        print(f"  WARNING: Poor calibration for {stat_name}")
```

### Interpretation Guidelines

| p-value | Interpretation | Action |
|---------|----------------|--------|
| < 0.01 | Systematic underestimation | Model generates too small values |
| 0.01 - 0.05 | Marginal underestimation | Watch carefully |
| 0.05 - 0.95 | Good calibration | Model fits well |
| 0.95 - 0.99 | Marginal overestimation | Watch carefully |
| > 0.99 | Systematic overestimation | Model generates too large values |

**Red flag:** If multiple test statistics have p < 0.05 or p > 0.95
**Action:** Model misspecified, try alternative

---

## Convergence Diagnostic Checklist

For each model, verify all of these:

### 1. R-hat (Gelman-Rubin statistic)
```
Target: R-hat < 1.01 for all parameters

Check:
- If R-hat > 1.05 → Chains haven't converged, run longer
- If R-hat > 1.1 → Serious mixing issues, reparameterize
- If R-hat > 1.2 → Model fundamentally problematic
```

### 2. Effective Sample Size (ESS)
```
Target: ESS > 400 for all parameters (bulk and tail)

Check:
- If ESS < 100 → Very inefficient, need much longer run
- If ESS < 400 → Marginal, run longer or reparameterize
- If ESS > 1000 → Excellent
```

### 3. Divergent Transitions (HMC/NUTS)
```
Target: 0 divergences (or < 1% if unavoidable)

Check:
- If divergences > 1% → Increase adapt_delta (0.95 → 0.99)
- If persist despite high adapt_delta → Reparameterize
- Common fix: Non-centered parameterization for hierarchical models
```

### 4. Energy Diagnostic (HMC)
```
Check E-BFMI (Energy Bayesian Fraction of Missing Information)

Target: E-BFMI > 0.3 for all chains

If E-BFMI < 0.2:
- Model geometry problematic
- Try reparameterization or different priors
```

---

## Sensitivity Analysis Protocol

### 1. Prior Sensitivity

For each model, vary priors systematically:

```
Baseline:
  mu ~ Normal(0, 50)
  tau ~ Half-Normal(0, 10)
  nu ~ Gamma(2, 0.1)     [Model 1]
  pi ~ Beta(2, 2)         [Model 2]
  alpha ~ Gamma(1, 1)     [Model 3]

Sensitivity variants:
  mu: Normal(0, 25), Normal(0, 100)
  tau: Half-Normal(0, 5), Half-Normal(0, 20)
  nu: Gamma(1, 0.1), Gamma(5, 0.2)
  pi: Beta(1, 1), Beta(5, 5)
  alpha: Gamma(2, 1), Gamma(0.5, 0.5)

Compare:
- Posterior means across variants
- If change > 10% → Data weakly informative
- If change < 5% → Robust to prior choice
```

### 2. Influence Analysis

Remove studies systematically:

```
Full data (J=8):
  Fit model, record mu_posterior

Remove Study 4 (most influential from EDA):
  Fit model, record mu_posterior_minus4
  Change = |mu_posterior - mu_posterior_minus4|

Remove Study 5 (only negative effect):
  Fit model, record mu_posterior_minus5
  Change = |mu_posterior - mu_posterior_minus5|

Remove both:
  Fit model, record mu_posterior_minus45

Interpretation:
- If any change > 20% → Results fragile
- If all changes < 10% → Results robust
```

### 3. Computational Validation

```
Run model with different configurations:

Config 1: 4 chains × 2000 samples, seed=123
Config 2: 4 chains × 2000 samples, seed=456
Config 3: 8 chains × 1000 samples, seed=789

Compare:
- Posterior means (should agree within 1 SE)
- Posterior SDs (should agree within 10%)
- LOO estimates (should agree within 2)

If not consistent → Model unidentified or unstable
```

---

## Expected Results Matrix

Given EDA findings (I²=2.9%, no outliers, Study 4 influential):

| Model | Expected nu/pi/K_eff | Expected mu | Expected tau | LOO rank |
|-------|---------------------|-------------|--------------|----------|
| **Hierarchical Normal** (D1) | - | 11.0 - 11.5 | 1.5 - 2.5 | 1 or 2 |
| **Model 1 (t)** | nu = 35-45 | 11.0 - 11.5 | 1.5 - 2.5 | 1 or 2 |
| **Model 2 (Mixture)** | pi < 0.1 or > 0.9 | 11.0 - 11.5 | - | 3 |
| **Model 3 (DP)** | K_eff = 1.0-1.5 | 11.0 - 11.5 | - | 4 |

**Interpretation:** Expect simple models to win given low I²

**What would surprise me:**

| Finding | Interpretation | Action |
|---------|----------------|--------|
| nu < 15 | Unexpectedly heavy tails | Investigate data quality |
| 0.2 < pi < 0.8 | Hidden subpopulations | Meta-regression to explain |
| K_eff > 3 | High complexity | Question pooling entirely |
| mu outside [9, 13] | Large shift from EDA | Check model specification |
| tau > 5 | High heterogeneity | Contradicts I²=2.9%, investigate |

---

## Synthesis Across Designers

### How Designer 1 and 2 Results Connect

```
Designer 1 Models:
├─ Common effect (tau = 0)
├─ Hierarchical normal
└─ Possibly random effects meta-analysis

Designer 2 Models:
├─ Model 1: Hierarchical t-distribution
├─ Model 2: Mixture
└─ Model 3: Dirichlet Process

Comparison:
1. Estimate mu from all models
   → Should agree if data clear
   → If disagree, quantify uncertainty range

2. Compare LOO-CV across all models
   → Identify best-performing model(s)
   → Report ΔELPD with SE

3. Assess robustness
   → Designer 2 models test assumptions
   → If D2 models agree with D1 → Robust
   → If D2 reveals issues → Important finding

4. Final recommendation
   → Consensus model if clear
   → Range of estimates if uncertain
   → Sensitivity to assumptions documented
```

---

## Reporting Template

### If All Models Agree (Expected)

```markdown
## Results

All models (classical normal, t-distribution, mixture, DP) converge on similar
estimates for the pooled effect:

| Model | mu (95% CI) | tau | LOO ELPD |
|-------|-------------|-----|----------|
| Hierarchical Normal | 11.2 (3.1, 19.3) | 2.0 | -30.9 ± 2.1 |
| t-Distribution (nu=42) | 11.3 (3.2, 19.4) | 2.1 | -30.8 ± 2.1 |
| Mixture (collapsed) | 11.1 (3.0, 19.2) | - | -31.5 ± 2.2 |

**Conclusion:** Data clearly favor simple hierarchical normal structure.
The t-distribution model (nu ≈ 42) confirms that normal tails are adequate.
Mixture model collapses to single cluster, validating low heterogeneity.

**Recommendation:** Use classical hierarchical normal model (simplest adequate model).
```

### If Models Disagree (Surprising)

```markdown
## Results

Models give divergent estimates, indicating fundamental uncertainty:

| Model | mu (95% CI) | Key Parameter | LOO ELPD |
|-------|-------------|---------------|----------|
| Hierarchical Normal | 11.5 (4.0, 19.0) | tau = 2.1 | -31.2 ± 2.0 |
| t-Distribution | 9.8 (2.5, 17.1) | nu = 12 | -29.5 ± 2.1 |
| Mixture | 14.2 (7.1, 21.3) | pi = 0.35 | -28.8 ± 2.3 |

**Conclusion:** Data do not strongly favor any single model structure.
Heavy tails (nu=12) suggest robustness concerns. Mixture model (pi=0.35)
indicates potential subpopulations.

**Recommendation:** Report range of estimates [9.8, 14.2] and emphasize
uncertainty. Additional data or covariates needed to resolve ambiguity.
```

---

## Key Takeaways

### 1. Start Simple, Increase Complexity Only If Needed
- Begin with Model 1 (t-distribution)
- Only fit Model 2/3 if evidence demands

### 2. Use Multiple Criteria
- LOO-CV (predictive performance)
- Posterior predictive checks (calibration)
- Convergence diagnostics (computational)
- Scientific interpretability

### 3. Be Ready to Stop Early
- If nu > 50 → Normal adequate, stop
- If pi → 0 or 1 → Single cluster, stop
- If K_eff ≈ 1 → Simple model, stop

### 4. Plan for Disagreement
- Designer 1 and 2 may reach different conclusions
- Document why they differ
- Synthesize into honest assessment

### 5. Falsification Is Success
- Discovering assumptions are wrong = good
- Pivoting to better model = progress
- Reporting uncertainty = honest science

---

## Implementation Order

```
Week 1:
  Day 1-2: Fit Model 1 (t-distribution)
  Day 3-4: Convergence diagnostics, LOO-CV
  Day 5: Decision point (stop or continue?)

Week 2 (conditional):
  Day 1-2: Fit Model 2 (mixture) if warranted
  Day 3-4: Cluster analysis, comparison to Model 1
  Day 5: Decision point (sufficient or try Model 3?)

Week 3 (advanced, optional):
  Day 1-3: Fit Model 3 (DP) if needed
  Day 4-5: Comprehensive model comparison

Week 4:
  Day 1-2: Compare with Designer 1's results
  Day 3-4: Synthesis and final recommendation
  Day 5: Write report and visualizations
```

---

**File:** `/workspace/experiments/designer_2/MODEL_COMPARISON_STRATEGY.md`
**Purpose:** Quantitative guide for model selection decisions
**Status:** Ready for use during implementation phase
