# Experiment Plan: Bayesian Models for Overdispersed Binomial Data

## Executive Summary

This document synthesizes proposals from two independent model designers into a prioritized, systematic experiment plan. Based on EDA findings (strong heterogeneity ICC=0.66, substantial overdispersion φ=3.5-5.1) and expert design recommendations, we will implement and evaluate 4 distinct model classes in priority order.

**Minimum Attempt Policy**: We will attempt at least the first 2 experiments unless Experiment 1 fails pre-fit validation (prior or simulation checks).

---

## Model Class Inventory (De-duplicated)

From parallel designers, we have 4 unique model classes (Student-t proposed by both):

| Model Class | Source | Priority | Rationale |
|-------------|--------|----------|-----------|
| Beta-Binomial Hierarchical | Designer 1 | **1 (HIGHEST)** | Canonical for overdispersed binomial, conjugate, EDA-aligned |
| Random Effects Logistic | Designer 1 | **2 (HIGH)** | Standard GLMM, well-understood, computationally efficient |
| Student-t Robust | Both | **3 (MODERATE)** | Heavy tails for outliers, consensus from both designers |
| Finite Mixture (K=2) | Designer 2 | **4 (EXPLORATORY)** | Tests bimodality hypothesis, structural alternative |
| Dirichlet Process | Designer 2 | **5 (BACKUP)** | Nonparametric, high complexity, only if 1-4 inadequate |

**Note**: We will NOT implement Dirichlet Process (Model 5) unless all other approaches fail, due to high computational cost and complexity.

---

## Prioritization Rationale

### Why Beta-Binomial First?
1. **Direct overdispersion modeling**: β parameter explicitly controls φ via φ = 1 + 1/κ
2. **Theoretical alignment**: Canonical model for overdispersed binomial data
3. **EDA support**: Observed φ=3.5-5.1 directly translates to κ ≈ 0.2-0.4
4. **Conjugate structure**: Mathematically elegant, computationally stable
5. **Both EDA analysts** and Designer 1 recommended this as primary

### Why Random Effects Logistic Second?
1. **Standard practice**: Most widely used for grouped binomial data
2. **Familiar parameterization**: Log-odds scale interpretable
3. **Computational robustness**: Non-centered parameterization samples well
4. **Easy to extend**: Can add covariates if needed
5. **Strong secondary recommendation** from Designer 1

### Why Student-t Third?
1. **Convergent design**: ONLY model proposed by BOTH designers independently
2. **Outlier robustness**: Heavy tails accommodate Groups 2, 8, 11
3. **Diagnostic value**: Posterior ν tells us if heavy tails needed
4. **Fallback role**: Use if Models 1-2 struggle with outliers

### Why Mixture Fourth?
1. **Alternative structure**: Tests different hypothesis (discrete subpopulations vs continuous)
2. **EDA hint**: Possible bimodal distribution (typical ~7%, elevated ~12%)
3. **Scientific interest**: Subgroup identification may be valuable
4. **Exploratory**: Lower priority, attempt only if time/resources allow

---

## Experiment Specifications

### **Experiment 1: Beta-Binomial Hierarchical Model**

**Scientific Hypothesis**: Group-level proportions vary continuously according to a Beta distribution, with overdispersion directly modeled through the concentration parameter.

**Model Specification**:
```
Likelihood:   r_i | p_i, n_i ~ Binomial(n_i, p_i)
Group level:  p_i | μ, κ ~ Beta(μκ, (1-μ)κ)
Priors:       μ ~ Beta(2, 18)           # E[μ] ≈ 0.1, centered on pooled 7.4%
              κ ~ Gamma(2, 0.1)         # E[κ] = 20, but flexible
```

**Parameters of Interest**:
- `μ`: Population mean proportion (expect ~0.074)
- `κ`: Concentration (expect ~0.2-0.4 given ICC=0.66)
- `p_i`: Group-specific proportions (12 values)
- Derived: `φ = 1 + 1/κ` (overdispersion factor, expect ~3.5-5)

**Prior Justification**:
- `μ ~ Beta(2, 18)`: Weakly informative, centers on ~10%, allows 1-30% range
- `κ ~ Gamma(2, 0.1)`: Broad prior allowing moderate to low concentration

**Falsification Criteria** (will REJECT if):
1. **Boundary behavior**: κ → 0 (infinite overdispersion) or κ → ∞ (no heterogeneity)
2. **Poor coverage**: Posterior predictive coverage < 70% of observed groups
3. **φ mismatch**: Posterior φ doesn't overlap observed range [3.5, 5.1]
4. **Computational failure**: Persistent divergences (>2%), Rhat > 1.01, ESS < 400
5. **Implausible posteriors**: Any p_i > 0.3 or μ > 0.2 (scientifically unreasonable)

**Expected Outcomes**:
- κ posterior: 0.2-0.5 (median ~0.35)
- φ posterior: 3.0-6.0 (median ~4.0)
- Group 1 posterior: p₁ ~ 3-5% (shrunk from 0% toward μ)
- Groups 2,8,11 posteriors: Moderate shrinkage toward μ, still elevated
- Runtime: 2-5 minutes (1000 samples × 4 chains)

**Implementation**: `/workspace/experiments/designer_1/proposed_models.md` (Model 1)

---

### **Experiment 2: Random Effects Logistic Regression**

**Scientific Hypothesis**: Group-level log-odds vary according to a normal distribution (continuous, symmetric heterogeneity on logit scale).

**Model Specification**:
```
Likelihood:   r_i | θ_i, n_i ~ Binomial(n_i, logit⁻¹(θ_i))
Group level:  θ_i = μ + τ · z_i        # Non-centered parameterization
              z_i ~ Normal(0, 1)
Priors:       μ ~ Normal(logit(0.075), 1²)  # E[p] ≈ 0.075
              τ ~ HalfNormal(1)              # Moderate between-group SD
```

**Parameters of Interest**:
- `μ`: Population mean log-odds (expect ~-2.5)
- `τ`: Between-group SD on logit scale (expect ~0.7-1.0)
- `θ_i`: Group-specific log-odds (12 values)
- Derived: `p_i = logit⁻¹(θ_i)` (group proportions)

**Prior Justification**:
- `μ ~ Normal(logit(0.075), 1²)`: Weakly informative on logit scale
- `τ ~ HalfNormal(1)`: Moderate heterogeneity prior

**Falsification Criteria** (will REJECT if):
1. **Extreme heterogeneity**: τ > 2.0 (suggests discrete subpopulations, try mixture)
2. **Outlier misfit**: Groups 2, 8, 11 consistently outside 95% posterior intervals
3. **Poor coverage**: Posterior predictive coverage < 70%
4. **Computational failure**: Divergences >1%, Rhat > 1.01, ESS < 400
5. **Implausible posteriors**: Any p_i > 0.3 or μ implies p > 0.2

**Expected Outcomes**:
- τ posterior: 0.6-1.2 (median ~0.9)
- μ posterior: -2.7 to -2.3 (corresponds to p ~ 6-9%)
- Similar substantive conclusions to Experiment 1
- Runtime: 2-5 minutes

**Implementation**: `/workspace/experiments/designer_1/proposed_models.md` (Model 2)

---

### **Experiment 3: Robust Student-t Random Effects**

**Scientific Hypothesis**: Group-level heterogeneity has heavy tails (outliers are legitimate, not modeling failures), requiring Student-t distribution on logit scale.

**Model Specification**:
```
Likelihood:   r_i | θ_i, n_i ~ Binomial(n_i, logit⁻¹(θ_i))
Group level:  θ_i = μ + τ · w_i        # Non-centered
              w_i ~ StudentT(ν, 0, 1)
Priors:       μ ~ Normal(logit(0.075), 1²)
              τ ~ HalfNormal(1)
              ν ~ Gamma(2, 0.1) [constrained > 2]
```

**Parameters of Interest**:
- `μ`, `τ`: Same as Experiment 2
- `ν`: Degrees of freedom (ν=∞ → normal, ν<10 → heavy tails)
- `θ_i`: Group-specific log-odds
- Diagnostic: If ν > 30, heavy tails unnecessary (revert to Model 2)

**Prior Justification**:
- Same as Model 2 for μ, τ
- `ν ~ Gamma(2, 0.1)`: Allows data to determine tail heaviness

**Falsification Criteria** (will REJECT if):
1. **Heavy tails unnecessary**: ν > 30 consistently (use Experiment 2 instead)
2. **Extreme tails**: ν < 3 consistently (too extreme, data quality issue)
3. **No improvement**: LOO similar to Experiment 2 but runtime 2× longer
4. **Computational failure**: Divergences >2%, Rhat > 1.01
5. **Poor fit**: Coverage < 70% despite flexibility

**Expected Outcomes**:
- **If heavy tails needed**: ν ~ 5-15, better outlier accommodation than Model 2
- **If heavy tails NOT needed**: ν > 30, conclude Model 2 is adequate
- Runtime: 5-10 minutes (slower due to complexity)

**Decision Rule**:
- ν posterior > 30: **REJECT** this model, accept Experiment 2 as adequate
- ν posterior 10-30: Marginal improvement, compare via LOO
- ν posterior < 10: Heavy tails confirmed, model useful

**Implementation**:
- Designer 1: `/workspace/experiments/designer_1/proposed_models.md` (Model 3)
- Designer 2: `/workspace/experiments/designer_2/implementation_templates.py` (Model 2)

---

### **Experiment 4: Finite Mixture (K=2 Components)**

**Scientific Hypothesis**: Groups come from two discrete subpopulations (low-risk ~6%, high-risk ~12%), not a continuous distribution.

**Model Specification**:
```
Likelihood:     r_i | c_i, p_{c_i}, n_i ~ Binomial(n_i, p_{c_i})
Cluster:        c_i | w ~ Categorical(w)
Component dist: p_k ~ Beta(a_k, b_k)
Priors:         w ~ Dirichlet(α=[5, 5])    # Equal weight prior
                a_k ~ Gamma(2, 0.1)
                b_k ~ Gamma(2, 0.1)
```

**Parameters of Interest**:
- `w`: Mixture weight (proportion in low-risk vs high-risk)
- `p_1, p_2`: Component means (expect ~0.06 and ~0.12)
- `c_i`: Cluster assignments (which groups are low/high risk)

**Prior Justification**:
- `w ~ Dirichlet([5,5])`: Symmetric prior, slight preference for balanced mixture
- Gamma priors on Beta parameters: Flexible component shapes

**Falsification Criteria** (will REJECT if):
1. **Degenerate mixture**: w < 0.1 or w > 0.9 (essentially one component)
2. **Components too close**: |p_2 - p_1| < 0.03 (no meaningful separation)
3. **Worse than hierarchical**: LOO ΔELPD > 4 favoring Experiments 1 or 2
4. **Computational failure**: Label switching, non-convergence
5. **Poor coverage**: < 70% despite two components

**Expected Outcomes**:
- **If mixture supported**: w ~ 0.5-0.7, p_1 ~ 0.06, p_2 ~ 0.12, clear separation
- **If mixture NOT supported**: w → 0 or 1, or p_1 ≈ p_2 → REJECT
- Runtime: 10-15 minutes (more complex)

**Decision Rule**:
- Mixture clearly better (ΔLOO > 4): Accept, report subgroup structure
- Mixture similar (|ΔLOO| < 4): Report both perspectives
- Mixture worse (ΔLOO < -4): Reject, use continuous model

**Implementation**: `/workspace/experiments/designer_2/implementation_templates.py` (Model 1)

---

## Implementation Strategy

### Phase A: Primary Models (Parallel Execution)
**Execute Experiments 1 and 2 in parallel** (both should succeed):
- Both are well-established, computationally efficient
- Both should converge easily
- Both should provide adequate fit

**Timeline**: ~10-15 minutes total (5 min each, parallel)

### Phase B: Model Comparison
**If both Models 1 and 2 are ACCEPTED**:
1. Compare via LOO cross-validation (`az.compare`)
2. Check ΔELPD ± SE
3. If |ΔELPD| < 2×SE: Models are equivalent, prefer Model 1 (more natural)
4. If ΔELPD > 2×SE: Prefer better model
5. Report selected model as primary, other as sensitivity check

**Timeline**: 5 minutes

### Phase C: Robust Extension (Conditional)
**Fit Experiment 3 (Student-t) IF**:
- Both Models 1-2 show outlier misfit (Groups 2,8,11 outside 95% intervals), OR
- Scientific interest in heavy-tail hypothesis

**Otherwise**: Skip to Phase D

**Timeline**: 10 minutes (if executed)

### Phase D: Alternative Structure (Conditional)
**Fit Experiment 4 (Mixture) IF**:
- Time/resources permit, AND
- Scientific interest in subgroup identification, OR
- Models 1-2 show bimodal residuals

**Otherwise**: Skip, conclude with Phase A-B

**Timeline**: 15 minutes (if executed)

---

## Success Criteria & Stopping Rules

### Minimum Success (Phase A required):
- ✓ At least 1 of {Experiment 1, Experiment 2} passes all validation
- ✓ Posterior predictive coverage > 85%
- ✓ Computational diagnostics clean (Rhat < 1.01, ESS > 400)
- ✓ Scientifically plausible posteriors

### Ideal Success:
- ✓ Both Experiments 1 and 2 pass
- ✓ Similar substantive conclusions (robust finding)
- ✓ Clear preference via LOO or parsimony
- ✓ One model selected as final with strong justification

### Stopping Rule - Adequate Model Found:
**STOP and proceed to final reporting IF**:
- Experiments 1 or 2 passes all checks, AND
- Coverage > 90%, AND
- LOO diagnostics clean (Pareto k < 0.7), AND
- Posteriors scientifically interpretable

**No need to fit Experiments 3-4** unless scientific interest persists.

### Stopping Rule - Iterate:
**Refine models IF**:
- Both Experiments 1-2 REJECTED (coverage < 70%), OR
- Severe outlier misfit despite flexibility, OR
- Computational failures despite tuning

**Then**: Launch model-refiner agent with diagnostic outputs

---

## Falsification Summary Table

| Experiment | Reject if... | Fallback |
|------------|-------------|----------|
| 1: Beta-Binomial | κ → boundary, φ mismatch, coverage < 70% | Try Exp 2 |
| 2: RE Logistic | τ > 2, outlier misfit, coverage < 70% | Try Exp 3 |
| 3: Robust Student-t | ν > 30 (use Exp 2 instead), ν < 3, coverage < 70% | Try Exp 4 |
| 4: Mixture | w extreme, components close, LOO worse | Refine Exp 1-2 |

---

## Expected Timeline

| Phase | Activity | Duration | Status |
|-------|----------|----------|--------|
| A | Fit Experiments 1-2 (parallel) | 10-15 min | **Required** |
| B | Model comparison (LOO) | 5 min | **Required** |
| C | Fit Experiment 3 (conditional) | 10 min | Optional |
| D | Fit Experiment 4 (conditional) | 15 min | Optional |
| - | **Total (minimum)** | **15-20 min** | Phases A-B |
| - | **Total (comprehensive)** | **40-45 min** | All phases |

---

## File References

### Designer 1 Specifications:
- Main document: `/workspace/experiments/designer_1/proposed_models.md`
- Implementation guide: `/workspace/experiments/designer_1/implementation_roadmap.md`
- Quick reference: `/workspace/experiments/designer_1/model_specifications_summary.md`

### Designer 2 Specifications:
- Main document: `/workspace/experiments/designer_2/proposed_models.md`
- Working code: `/workspace/experiments/designer_2/implementation_templates.py`
- Decision tree: `/workspace/experiments/designer_2/decision_tree.md`
- Quick reference: `/workspace/experiments/designer_2/quick_reference.md`

---

## Risk Assessment

### Low Risk (Experiments 1-2):
- Well-established methods
- EDA strongly supports these models
- Computational efficiency high
- **Expected outcome**: Both succeed, one selected

### Moderate Risk (Experiment 3):
- May find ν > 30 → heavy tails not needed
- Computational cost higher
- **Mitigation**: Only fit if warranted by Exp 1-2 results

### Higher Risk (Experiment 4):
- More complex inference (label switching possible)
- May not be identified (w → 0 or 1)
- Computational cost highest
- **Mitigation**: Only fit if time allows and scientific interest strong

---

## Deviation from Plan

**Document any deviations** in `/workspace/experiments/iteration_log.md`:
- Which experiments were skipped and why
- Which experiments were added and why
- Order changes and rationale
- Any failures and pivots

---

## Conclusion

This experiment plan synthesizes parallel designer proposals into a systematic, prioritized workflow:

1. **Start with proven methods** (Beta-binomial, RE logistic) - high success probability
2. **Test consensus hypothesis** (Student-t) if warranted
3. **Explore alternative structure** (Mixture) if time/interest permits
4. **Clear falsification criteria** for each experiment
5. **Flexible stopping rules** - stop when adequate model found

**Expected outcome** (80% confidence): Experiments 1 and 2 both adequate, Model 1 (Beta-binomial) selected as final model due to more natural parameterization for overdispersed binomial data.

**Ready to proceed to Phase 3: Model Development Loop.**
