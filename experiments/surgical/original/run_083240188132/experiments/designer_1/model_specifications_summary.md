# Quick Reference: Model Specifications Summary
## Designer 1 - Robust Hierarchical Models

**Date**: 2025-10-30
**Context**: 12 groups, φ=3.5-5.1, ICC=0.66, zero-event group, three outliers

---

## Model 1: Beta-Binomial Hierarchical (PRIMARY RECOMMENDATION)

### Mathematical Specification
```
r_i | p_i, n_i ~ Binomial(n_i, p_i)
p_i | μ, κ ~ Beta(μκ, (1-μ)κ)
μ ~ Beta(2, 18)                    # mean rate ≈ 0.1
κ ~ Gamma(2, 0.1)                  # concentration
```

### Why This Model?
- **Direct overdispersion modeling** via beta-distributed success probabilities
- **Conjugate structure** provides computational stability
- **Natural for binomial data** - canonical model for overdispersed counts
- ICC = 1/(κ+1) directly matches observed ICC=0.66 → expect κ ≈ 0.5

### Key Parameters
- **μ**: Population mean success rate (expect ~0.07)
- **κ**: Concentration parameter; controls between-group variance
  - Low κ → high variance → strong overdispersion
  - ICC = 1/(κ+1), so ICC=0.66 implies κ ≈ 0.5
- **φ = 1 + 1/κ**: Overdispersion factor (expect ~3.5-5.1)

### Rejection Criteria
- κ → 0 or κ → ∞ (boundary behavior)
- <70% posterior predictive coverage
- Posterior φ far from observed (3.5-5.1)
- Persistent computational failures (divergences, Rhat > 1.05)

### Expected Runtime
2-5 minutes (fast, stable)

---

## Model 2: Random Effects Logistic Regression (ALTERNATIVE PRIMARY)

### Mathematical Specification
```
r_i | θ_i, n_i ~ Binomial(n_i, logit^(-1)(θ_i))
θ_i = μ + τ * z_i              # non-centered parameterization
z_i ~ Normal(0, 1)
μ ~ Normal(logit(0.075), 1²)   # population mean log-odds
τ ~ HalfNormal(1)               # between-group SD
```

### Why This Model?
- **Standard approach** - widely understood and used
- **Excellent computational properties** with non-centered parameterization
- **Easy to extend** with group-level covariates
- **Familiar log-odds scale** for many researchers

### Key Parameters
- **μ**: Population mean log-odds (expect ~-2.6 for 7% rate)
- **τ**: Between-group standard deviation on log-odds scale
  - Expect τ ≈ 0.7-1.0 to achieve φ = 3.5-5.1
- **p_i = logit^(-1)(θ_i)**: Group-level probabilities

### Rejection Criteria
- τ > 2.0 (extreme between-group variation, suggests mixture)
- Poor fit to outliers (Groups 2, 8, 11 outside 95% intervals)
- <70% posterior predictive coverage
- Computational failures despite non-centered approach

### Expected Runtime
2-5 minutes (very efficient with non-centered parameterization)

---

## Model 3: Robust Logistic with Student-t (BACKUP ONLY)

### Mathematical Specification
```
r_i | θ_i, n_i ~ Binomial(n_i, logit^(-1)(θ_i))
θ_i ~ StudentT(ν, μ, τ²)       # heavy-tailed random effects
μ ~ Normal(logit(0.075), 1²)
τ ~ HalfNormal(1)
ν ~ Gamma(2, 0.1)               # degrees of freedom (constrained > 2)
```

### Why This Model?
- **Robust to outliers** - heavy tails accommodate extreme groups
- **Adaptive shrinkage** - outliers shrink less than typical groups
- **Use only if Models 1-2 fail** - particularly for outlier accommodation

### Key Parameters
- **μ, τ**: Same as Model 2
- **ν**: Degrees of freedom, controls tail weight
  - ν > 30: essentially Gaussian (Model 2 sufficient)
  - ν = 5-10: moderately heavy tails
  - ν = 2-4: very heavy tails
  - If posterior ν > 30 → revert to Model 2

### Rejection Criteria
- ν > 30 with high probability (use Model 2 instead)
- ν → 2 (hitting constraint boundary, too extreme)
- Still poor fit despite heavy tails (try mixture model)
- Persistent computational failures (>2% divergences)

### Expected Runtime
5-10 minutes (slower due to Student-t complexity)

### When to Use
**ONLY IF**:
- Model 2 systematically underfits Groups 2, 8, 11
- Posterior predictive checks show outlier accommodation failure
- Not as default choice

---

## Comparative Quick View

| Aspect | Beta-Binomial | Logistic GLMM | Robust Logistic |
|--------|--------------|---------------|-----------------|
| **Scale** | Probability | Log-odds | Log-odds |
| **Overdispersion** | Direct (Beta) | Indirect (RE) | Indirect (heavy-tail RE) |
| **Outlier handling** | Moderate | Moderate | Excellent |
| **Computation** | Fast | Fast | Moderate |
| **Priority** | **1st** | **2nd** | 3rd (conditional) |
| **Use when** | Default | Alternative | Models 1-2 fail |

---

## Implementation Priority

### Phase 1: Fit in Parallel
1. **Model 1**: Beta-binomial (primary)
2. **Model 2**: Logistic GLMM (comparison)

### Phase 2: Assess Both Models
- Prior predictive checks
- MCMC convergence diagnostics
- Posterior predictive checks
- Apply falsification criteria

### Phase 3: Decision
**If both adequate:**
- Compare LOO-CV
- Prefer Model 1 (more natural for binomial overdispersion)
- Report both if results similar

**If only one adequate:**
- Use the adequate model

**If both fail outlier fit:**
- Implement Model 3

### Phase 4: Model 3 (Conditional)
- Only if Models 1-2 show poor outlier accommodation
- Check posterior ν
- If ν > 30: revert to Model 2
- If ν < 30: use Model 3

---

## Key Design Principles Applied

1. **Falsification mindset**: Explicit rejection criteria for each model
2. **Competing hypotheses**: Three different variance structures (Beta, Gaussian, Student-t)
3. **Escape routes**: Clear conditions for switching model classes
4. **Computational realism**: All models feasible with MCMC, runtime <10 minutes
5. **Scientific plausibility**: All models appropriate for overdispersed binomial data
6. **Stress tests**: Posterior predictive checks designed to reveal failures

---

## Expected Outcomes

**Most Likely** (90% confidence):
- Models 1 and 2 both adequate
- Similar substantive conclusions
- Model 1 selected (more direct for overdispersion)
- Model 3 unnecessary

**Less Likely** (10% confidence):
- One model clearly better than other
- Model 3 needed for outlier accommodation
- All models fail (would trigger mixture model consideration)

**Predicted final model**: Model 1 (Beta-binomial hierarchical)

---

## Critical Success Factors

### For Model 1:
- κ posterior reasonable (0.2 to 2.0)
- Posterior φ matches observed (3.5-5.1)
- >85% posterior predictive coverage
- Zero-event and outliers well-handled

### For Model 2:
- τ posterior reasonable (0.5 to 1.5)
- Non-centered parameterization samples well
- >85% posterior predictive coverage
- No systematic outlier underfit

### For Model 3 (if needed):
- ν posterior indicates heavy tails needed (5-20)
- Better outlier fit than Model 2
- Not excessive computational cost
- Adaptive shrinkage working appropriately

---

## Files and Locations

**Main specification**: `/workspace/experiments/designer_1/proposed_models.md` (10,000+ words)
**This summary**: `/workspace/experiments/designer_1/model_specifications_summary.md`
**Data**: `/workspace/data/data.csv`
**EDA Report**: `/workspace/eda/eda_report.md`

---

**Status**: Ready for implementation
**Next Steps**: Prior predictive checks → Model fitting → Posterior assessment
**Designer**: Model Designer 1 (Robust Hierarchical Approaches)
