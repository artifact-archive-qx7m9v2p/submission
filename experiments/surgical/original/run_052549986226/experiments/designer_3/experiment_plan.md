# Experiment Plan: Robust and Alternative Models
## Model Designer 3

**Date:** 2025-10-30
**Focus:** Robustness to outliers, alternative modeling perspectives
**Dataset:** 12 groups, 5 outliers (42% outlier rate), severe overdispersion

---

## Problem Formulation

### The Core Challenge

The EDA reveals an **outlier-dominated dataset**:
- 5 of 12 groups are statistical outliers (42%)
- Group 8 is extreme outlier (z=3.94, rate=14.4% vs mean 7.6%)
- Group 1 has zero successes (0/47)
- Overdispersion parameter φ ≈ 3.5-5.1 (should be ≈1.0)

**Critical question:** Are these outliers:
1. **Tail events** of a continuous heavy-tailed distribution? (Student-t)
2. **Sparse effects** where most groups are identical? (Horseshoe)
3. **Distinct subpopulations** with different underlying rates? (Mixture)

### Competing Hypotheses

**Hypothesis 1: Heavy-tailed continuous variation**
- All groups drawn from single distribution with heavy tails
- Outliers are plausible extreme values, not errors
- Student-t random effects appropriate

**Hypothesis 2: Sparse heterogeneity**
- Most groups (8-9 of 12) are essentially identical
- Few groups (3-4) genuinely deviate
- Horseshoe prior induces automatic selection

**Hypothesis 3: Latent subgroups**
- Two distinct populations (normal vs outlier)
- Unmeasured binary covariate separates groups
- Finite mixture model captures discrete structure

---

## Model Classes to Explore

### Model 1: Student-t Hierarchical Model

**Mathematical specification:**
```
r_i ~ Binomial(n_i, p_i)
logit(p_i) = μ + α_i
α_i ~ Student-t(ν, 0, σ)

Priors:
  μ ~ Normal(-2.5, 1)
  σ ~ Half-Cauchy(0, 1)
  ν ~ Gamma(2, 0.1)
```

**Why this might work:**
- Heavy tails (ν < 30) accommodate outliers without contaminating μ
- Continuous distribution (no assumption of discrete clusters)
- Well-studied, computationally stable

**I will abandon this if:**
1. Posterior ν > 50 (data doesn't need heavy tails)
2. Posterior ν < 2 with divergences (need discrete mixture)
3. Posterior predictive fails (can't reproduce outlier frequency)
4. LOO-CV significantly worse than alternatives

**Evidence that would support this:**
- Posterior ν ∈ [5, 30]
- Group 8 has wider posterior than under Normal prior
- Population μ more precise than under Normal (less contamination)

**Implementation:** Stan via CmdStanPy, 4 chains × 2000 iterations

---

### Model 2: Horseshoe Prior Model

**Mathematical specification:**
```
r_i ~ Binomial(n_i, p_i)
logit(p_i) = μ + α_i
α_i ~ Normal(0, τ · λ_i)

Local-global shrinkage:
  λ_i ~ Cauchy^+(0, 1)
  τ ~ Cauchy^+(0, τ_0)

τ_0 = (3 / 9) * (1 / sqrt(12)) ≈ 0.096
```

**Why this might work:**
- If most groups truly interchangeable, enforces sparsity
- Automatic outlier detection (large λ_i)
- Better prediction via aggressive shrinkage of non-outliers

**I will abandon this if:**
1. All λ_i ≈ 0.5-1.0 (no sparsity detected)
2. Posterior τ >> τ_0 (too many non-zero effects)
3. Computational issues (slow mixing, divergences)
4. Interpretation unclear (sparsity assumption wrong)

**Evidence that would support this:**
- Clear bimodal distribution of λ_i (small vs large)
- n_active ≈ 3-5 groups (matches expected outliers)
- Better LOO-CV than continuous models
- Groups 2, 8, 11 have λ > 0.5; others λ < 0.2

**Implementation:** Stan via CmdStanPy, 4 chains × 3000 iterations (longer for Cauchy priors)

---

### Model 3: Finite Mixture Model

**Mathematical specification:**
```
r_i ~ Binomial(n_i, p_i)
logit(p_i) = μ_z[i] + α_i

α_i | z_i ~ Normal(0, σ_z[i])
z_i ~ Categorical(π)

Priors:
  μ_1 ~ Normal(-3.0, 0.5)  [Low-rate cluster]
  μ_2 ~ Normal(-2.0, 0.5)  [High-rate cluster]
  σ_1 ~ Half-Normal(0, 0.3)
  σ_2 ~ Half-Normal(0, 0.5)
  π ~ Uniform(0, 1)

Constraint: μ_1 < μ_2
```

**Why this might work:**
- If truly two distinct populations in data
- Explains bimodality observed in EDA
- Identifies which groups are "different"

**I will abandon this if:**
1. Posterior π → 0 or π → 1 (no mixture)
2. Posterior μ_1 ≈ μ_2 (clusters not separated)
3. Label switching (identifiability issues)
4. Cluster assignments ambiguous (all P ≈ 0.5)
5. N=12 too small for reliable clustering

**Evidence that would support this:**
- Posterior π ∈ [0.2, 0.8]
- Clear separation: μ_2 - μ_1 > 0.5
- Groups 2, 8, 11 assigned to cluster 2 with P > 0.8
- Better posterior predictive fit than single-component models

**Implementation:** Stan via CmdStanPy, 4 chains × 4000 iterations, initialized with μ = [-3, -2]

---

## Red Flags: When to Change Model Classes

### Triggers for Complete Reconsideration

**Red Flag 1: All models fail posterior predictive checks**
- Can't reproduce overdispersion (φ)
- Can't reproduce outlier frequency
- Systematic misfit in residuals

**Action:** Reconsider data generation process entirely
- Negative binomial (different overdispersion mechanism)
- Beta regression (continuous outcome)
- Data quality investigation

**Red Flag 2: High Pareto k values (> 0.7) for multiple groups**
- LOO-CV unreliable
- Models too sensitive to individual observations

**Action:** Investigate influential points
- Check Group 8 for data errors
- Consider leaving Group 8 out entirely
- Collect more groups to reduce influence

**Red Flag 3: Prior-posterior conflict**
- Posterior concentrates at edge of prior support
- KL divergence minimal (not learning from data)

**Action:** Revise priors or model structure
- Expand prior support
- Check for model misspecification

**Red Flag 4: Computational pathologies**
- Divergences despite high adapt_delta
- Non-convergence (Rhat > 1.05)
- Extreme parameter values

**Action:** Often indicates model misspecification
- Student-t: If ν < 3, switch to mixture
- Horseshoe: If λ_i → ∞, switch to Student-t
- Mixture: If label switching, insufficient data

---

## Decision Points for Major Pivots

### Checkpoint 1: After Student-t Model (Primary)

**Decision:** Continue to alternatives?

- **If ν > 50**: Stop here, use Normal hierarchical (robustness unnecessary)
- **If ν ∈ [5, 30]**: Student-t is appropriate, fit alternatives for comparison
- **If ν < 5**: Skip to mixture model (very heavy tails suggest discrete clusters)

### Checkpoint 2: After Horseshoe Model (Sparse Alternative)

**Decision:** Is sparsity present?

- **If 3-5 groups active (λ > 0.5)**: Sparsity detected, compare with Student-t
- **If all λ similar**: No sparsity, Student-t is simpler and better
- **If computational issues**: Abandon, not worth the complexity

### Checkpoint 3: After Mixture Model (Discrete Alternative)

**Decision:** Are clusters real?

- **If π ∈ [0.2, 0.8] AND μ_2 - μ_1 > 0.5**: Mixture justified
- **If π → 0 or 1**: No mixture, revert to Student-t
- **If label switching**: N=12 too small, mixture unreliable

### Checkpoint 4: Model Comparison (LOO-CV)

**Decision:** Which model to use?

- **ΔLOO < 2**: Models equivalent, choose simplest (Student-t)
- **ΔLOO ∈ [2, 10]**: Some evidence, but check interpretability
- **ΔLOO > 10**: Clear winner, use best model
- **All Pareto k > 0.7**: All models problematic, investigate data

---

## Alternative Approaches (If Models Fail)

### Backup Plan A: Negative Binomial Model

**If:** Beta-binomial alternatives all fail

**Why:** Different overdispersion mechanism
- Beta-binomial: overdispersion from varying p_i
- Negative binomial: overdispersion from varying "trials"

**Model:**
```
r_i ~ NegativeBinomial(μ_i, φ)
log(μ_i) = log(n_i) + α_i
α_i ~ Normal(0, σ)
```

### Backup Plan B: Zero-Inflated Model

**If:** Group 1 (zero count) systematically different

**Why:** Two processes
- Bernoulli (zero vs non-zero)
- Binomial (if non-zero, how many)

**Model:**
```
r_i ~ ZeroInflated(π_i, Binomial(n_i, p_i))
```

### Backup Plan C: Data Quality Investigation

**If:** All models fail AND computational issues

**Action:**
1. Verify Group 8 data (z=3.94 is extreme)
2. Check for data entry errors
3. Check temporal stability (if time data available)
4. Exclude outliers and refit

---

## Stress Tests and Validation

### Stress Test 1: Exclude Group 8

**Purpose:** Test robustness to extreme outlier

**Method:**
1. Remove Group 8 from dataset
2. Refit all models
3. Compare population parameters (μ, σ)

**Pass criterion:**
- Robust models: μ should change < 10%
- Non-robust models: μ should shift significantly

### Stress Test 2: Simulation-Based Calibration

**Purpose:** Can models recover known parameters?

**Method:**
1. Generate 100 datasets from fitted model posteriors
2. Refit model to each fake dataset
3. Check if true parameters within 95% CI

**Pass criterion:** 90-95% coverage for all parameters

### Stress Test 3: Prior Sensitivity

**Purpose:** Are conclusions robust to prior choice?

**Method:**
1. Refit with wider priors (double SD)
2. Refit with narrower priors (half SD)
3. Compare posterior conclusions

**Pass criterion:** Key conclusions unchanged

### Stress Test 4: Cross-Validation

**Purpose:** Out-of-sample prediction

**Method:**
1. Leave each group out
2. Predict r_i from remaining 11 groups
3. Check prediction intervals

**Pass criterion:** 90% of groups within 50% PI

---

## Success Criteria

### This modeling effort succeeds if:

1. **At least one model validates**
   - Passes posterior predictive checks
   - LOO Pareto k < 0.7 for all groups
   - Rhat < 1.01, ESS > 400

2. **Outlier uncertainty quantified**
   - Can identify which groups are outliers with high confidence
   - OR acknowledge that data insufficient to distinguish

3. **Population mean robustly estimated**
   - μ posterior not overly influenced by Group 8
   - Credible intervals reflect genuine uncertainty

4. **Falsification criteria documented**
   - Clear statements of what would reject each model
   - Honest assessment of model limitations

5. **Interpretable results**
   - Can explain findings to domain experts
   - Recommendations are actionable

### This modeling effort fails if:

1. **All models fail validation**
   - Posterior predictive checks systematically fail
   - High Pareto k for multiple groups
   - Computational pathologies unresolved

2. **Models too complex to interpret**
   - Can't explain why one model preferred
   - Conclusions change dramatically between models

3. **Data quality issues unresolved**
   - Group 8 data error suspected but not verified
   - Measurement process unclear

4. **Overconfident conclusions**
   - Claim strong evidence with N=12
   - Ignore model uncertainty

---

## Expected Timeline

**Phase 1: Model Fitting (1-2 hours)**
- Fit Student-t model (~5 min)
- Fit Horseshoe model (~10 min)
- Fit Mixture model (~20 min)
- Troubleshoot any sampling issues

**Phase 2: Validation (1-2 hours)**
- Posterior predictive checks
- LOO-CV computation
- Diagnostic plots
- Check falsification criteria

**Phase 3: Comparison (30 min)**
- Compare LOO-CV
- Interpret model differences
- Select best model or declare equivalent

**Phase 4: Stress Testing (1-2 hours)**
- Exclude Group 8 sensitivity
- Simulate from fitted models
- Prior sensitivity analysis

**Phase 5: Reporting (1 hour)**
- Summarize findings
- Document limitations
- Provide recommendations

**Total estimated time: 4-7 hours**

---

## Key Outputs

### Files to Generate

1. **Model fits:** `results/*/samples/` (Stan CSV files)
2. **Group estimates:** `results/*_group_estimates.csv`
3. **Model comparison:** `results/model_comparison.csv`
4. **Diagnostic plots:** `results/*/diagnostics/`
5. **Summary report:** `results/summary_report.md`

### Deliverables

1. **Best model recommendation** with justification
2. **Posterior estimates** for all groups with uncertainty
3. **Outlier classifications** with probabilities
4. **Model limitations** and assumptions
5. **Sensitivity analyses** demonstrating robustness

---

## Critical Reflection

### What Could Go Wrong?

**Scenario 1: All models agree**
- **Signal:** Posteriors nearly identical
- **Interpretation:** Robustness features unnecessary
- **Action:** Report that standard Normal hierarchical is sufficient
- **Is this failure?** NO - parsimony is good

**Scenario 2: All models disagree**
- **Signal:** Different μ, different outlier classifications
- **Interpretation:** Data insufficient to distinguish hypotheses
- **Action:** Report fundamental uncertainty, need more data
- **Is this failure?** NO - honesty about uncertainty is success

**Scenario 3: Best model is most complex**
- **Signal:** Mixture wins LOO but π ≈ 0.5 (ambiguous)
- **Interpretation:** Overfitting vs interpretability trade-off
- **Action:** Use simpler model, document trade-off
- **Is this failure?** NO - interpretability matters

**Scenario 4: Group 8 dominates all inference**
- **Signal:** Remove Group 8, conclusions change dramatically
- **Interpretation:** Single point has too much influence
- **Action:** Investigate data quality, report sensitivity
- **Is this failure?** MAYBE - depends on if Group 8 is error

### Philosophical Questions

**Are these outliers or errors?**
- Group 8 (z=3.94) is suspicious but not impossible
- Robust models reduce impact, but can't fix bad data
- Always report sensitivity to outliers

**Is N=12 enough?**
- Mixture models prefer N > 20
- Horseshoe benefits from larger N
- With N=12, be cautious about strong claims

**Are we overthinking this?**
- Maybe beta-binomial from EDA is sufficient
- Occam's razor: simplest model that fits
- Robust models add complexity for modest gains
- But: if outliers are real, robustness protects inference

---

## Final Recommendations

### Primary Model: Student-t Hierarchical

**Rationale:**
- Most robust to outliers
- Simplest of three alternatives
- Well-studied, interpretable
- Good balance of flexibility and parsimony

**When to use:**
- Outliers plausible but concerning
- Want robust population inference
- No strong prior belief about sparsity or clusters

### Sensitivity: Horseshoe Prior

**Rationale:**
- Tests sparsity hypothesis
- Better prediction if sparsity real
- Identifies which groups truly differ

**When to use:**
- Student-t validation marginal
- Care about prediction > inference
- Computational budget allows

### Exploratory: Mixture Model

**Rationale:**
- N=12 is small for clustering
- High risk of overinterpretation
- Computational challenges

**When to use:**
- Student-t and Horseshoe both fail
- Strong theoretical reason for discrete groups
- Only if π and μ posteriors clearly separated

---

## Remember: Success ≠ Task Completion

**Finding that robust models are unnecessary is a success.**

If posterior ν → ∞ (Student-t reduces to Normal), that's valuable information. It means:
- Data cleaner than suspected
- Standard hierarchical model sufficient
- Complexity not warranted

**The goal is truth, not complexity.**

If all three models fail, that's important to know. It suggests:
- Data generation process not captured
- Need different model class
- Data quality issues

**Failure modes teach us about the data.**

---

**End of Experiment Plan**

All model implementations: `/workspace/experiments/designer_3/`
Detailed specifications: `/workspace/experiments/designer_3/proposed_models.md`
