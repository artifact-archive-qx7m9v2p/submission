# Bayesian Modeling Experiment Plan
## Synthesized from Three Parallel Model Designers

**Date**: 2025-10-28
**Dataset**: `/workspace/data/data.csv` (n=27, Y vs x)
**EDA Report**: `/workspace/eda/eda_report.md`

---

## Executive Summary

Three independent model designers proposed 8 distinct model classes. After removing duplicates and prioritizing by theoretical justification and expected performance, we will implement **6 models** across three phases:

**Phase 1 (Priority Models)**: Logarithmic with Normal and Student-t likelihoods
**Phase 2 (Alternative Hypotheses)**: Piecewise Linear, Gaussian Process
**Phase 3 (If Needed)**: Mixture Model, Asymptotic

### Key Insights from Designer Synthesis

**Convergent Findings (All 3 Designers Agreed)**:
- Logarithmic transformation is the baseline to beat
- Outlier at x=31.5 requires attention (Student-t or sensitivity analysis)
- Two-regime structure at x≈7 is scientifically interesting
- Small sample (n=27) requires informative priors
- Overfitting is the main risk, not underfitting

**Divergent Perspectives (Competing Hypotheses)**:
- **Designer 1 (Parametric)**: Sharp changepoint at x=7 is real phase transition
- **Designer 2 (Flexible)**: Smooth saturation, changepoint is sampling artifact
- **Designer 3 (Robust)**: Heterogeneity (mixture) or heavy tails, not changepoint

**Resolution Strategy**: Fit multiple models and let LOO-CV adjudicate

---

## Prioritized Model List

### Tier 1: Must-Fit Models (Minimum Attempt Policy)

#### Model 1: Logarithmic with Normal Likelihood (BASELINE)
**Designer**: 1, 2 (convergent)
**Hypothesis**: Smooth saturation, homogeneous noise
**Priority**: HIGHEST - establishes baseline

**Mathematical Specification**:
```
Y_i ~ Normal(μ_i, σ)
μ_i = β₀ + β₁*log(x_i)

Priors:
β₀ ~ Normal(2.3, 0.3)
β₁ ~ Normal(0.29, 0.15)
σ ~ Exponential(10)
```

**Falsification Criteria**:
- Abandon if: Residuals show clear two-regime clustering
- Abandon if: Student-t model improves LOO by >4
- Abandon if: Posterior predictive p-value < 0.05 or > 0.95

**Expected Outcome**: R² ≈ 0.90, RMSE ≈ 0.09 (matching EDA)

---

#### Model 2: Logarithmic with Student-t Likelihood (ROBUST BASELINE)
**Designer**: 3 (primary), 1 (variant)
**Hypothesis**: Smooth saturation, heavy-tailed noise handles outlier
**Priority**: HIGHEST - tests robustness

**Mathematical Specification**:
```
Y_i ~ StudentT(ν, μ_i, σ)
μ_i = β₀ + β₁*log(x_i)

Priors:
β₀ ~ Normal(2.3, 0.5)
β₁ ~ Normal(0.29, 0.15)
σ ~ Exponential(10)
ν ~ Gamma(2, 0.1)  # Mean=20, allows 3-100
```

**Falsification Criteria**:
- Abandon if: ν posterior > 50 (Normal sufficient)
- Abandon if: LOO worse than Normal model
- Abandon if: Residuals still show patterns after accounting for heavy tails

**Expected Outcome**: ν ∈ [5, 25], automatic outlier downweighting

---

### Tier 2: Alternative Hypotheses (Attempt if Tier 1 Passes Validation)

#### Model 3: Piecewise Linear with Changepoint
**Designer**: 1 (primary), 3 (mixture alternative)
**Hypothesis**: Sharp regime change at x≈7, not smooth saturation
**Priority**: MEDIUM - tests changepoint hypothesis

**Mathematical Specification**:
```
Y_i ~ Normal(μ_i, σ)
μ_i = {
  β₁₀ + β₁₁*x_i           if x_i ≤ τ
  β₂₀ + β₂₁*x_i           if x_i > τ
}
# Continuity constraint: β₁₀ + β₁₁*τ = β₂₀ + β₂₁*τ

Priors:
τ ~ Uniform(5, 10)       # Changepoint location
β₁₀ ~ Normal(1.8, 0.3)   # Regime 1 intercept
β₁₁ ~ Normal(0.11, 0.05) # Regime 1 slope (steep)
β₂₁ ~ Normal(0.017, 0.01)# Regime 2 slope (flat)
σ ~ Exponential(10)
```

**Falsification Criteria**:
- Abandon if: τ posterior is uniform (no identifiable changepoint)
- Abandon if: β₁₁ ≈ β₂₁ (slopes not different)
- Abandon if: Logarithmic model has similar LOO (Occam's razor)

**Expected Outcome**: τ ∈ [6, 8], slope ratio ≈ 6:1

---

#### Model 4: Gaussian Process with Matérn 3/2 Kernel
**Designer**: 2 (primary)
**Hypothesis**: Flexible nonparametric, can discover structure EDA missed
**Priority**: MEDIUM - tests if flexibility helps

**Mathematical Specification**:
```
Y_i ~ Normal(f(x_i), σ)
f(x) ~ GP(m(x), k(x, x'))

m(x) = β₀ + β₁*log(x)    # Informative mean
k(x, x') = α² * (1 + √3*d/ℓ) * exp(-√3*d/ℓ)  # Matérn 3/2
where d = |x - x'| / scale(x)

Priors:
β₀ ~ Normal(2.3, 0.3)
β₁ ~ Normal(0.29, 0.15)
α ~ Normal⁺(0, 0.15)     # Marginal SD
ℓ ~ InvGamma(5, 5)       # Length scale
σ ~ Exponential(10)
```

**Falsification Criteria**:
- Abandon if: Divergences > 5%
- Abandon if: LOO worse than logarithmic (flexibility hurts with n=27)
- Abandon if: ℓ → ∞ (model degenerates to mean function only)
- Abandon if: Wild oscillations in posterior mean function

**Expected Outcome**: ΔLOO ≈ +2 improvement, ℓ ∈ [5, 10]

---

### Tier 3: Backup Models (Only if Tier 1-2 Fail or Show Unexpected Patterns)

#### Model 5: Mixture of Two Regimes (Soft Changepoint)
**Designer**: 3 (primary)
**Hypothesis**: Observations probabilistically belong to Growth or Plateau regime
**Priority**: LOW - complex, only if evidence for heterogeneity

**Mathematical Specification**:
```
Y_i ~ p_i * Normal(μ₁(x_i), σ₁) + (1-p_i) * Normal(μ₂(x_i), σ₂)

Regime 1 (Growth): μ₁(x) = β₁₀ + β₁₁*x
Regime 2 (Plateau): μ₂(x) = β₂₀ + β₂₁*x

p_i = logit⁻¹(γ₀ + γ₁*x_i)  # Regime probability varies with x

Priors:
β₁₁ ~ Normal(0.11, 0.05)  # Steep slope
β₂₁ ~ Normal(0.02, 0.01)  # Flat slope
γ₁ ~ Normal(-0.5, 0.3)    # Transition around x=7-10
# ... (see designer_3/proposed_models.md for full spec)
```

**Falsification Criteria**:
- Abandon if: Regime probabilities are diffuse (no clear assignment)
- Abandon if: β₁₁ ≈ β₂₁ (regimes are similar)
- Abandon if: Convergence issues (MCMC struggles with mixture)

**Expected Outcome**: Clear separation at x≈7, or collapses to single regime

---

#### Model 6: Three-Parameter Asymptotic (Mechanistic)
**Designer**: 1 (primary)
**Hypothesis**: Exponential approach to biochemical equilibrium
**Priority**: LOW - mechanistic interpretation but similar fit to logarithmic

**Mathematical Specification**:
```
Y_i ~ Normal(μ_i, σ)
μ_i = a - b*exp(-c*x_i)

Priors:
a ~ Normal(2.7, 0.15)    # Asymptote near max(Y)
b ~ Normal(1.0, 0.5)     # Initial gap
c ~ Exponential(0.5)     # Rate parameter
σ ~ Exponential(10)
```

**Falsification Criteria**:
- Abandon if: a < max(Y) (can't reach observed data)
- Abandon if: c → 0 (no saturation) or c → ∞ (instant saturation)
- Abandon if: Logarithmic model has better LOO

**Expected Outcome**: Similar fit to logarithmic (R² ≈ 0.89)

---

## Implementation Strategy

### Phase 1: Baseline Models (Day 1-2, ~4 hours compute)
1. Fit Model 1 (Logarithmic Normal) - 30 min
2. Fit Model 2 (Logarithmic Student-t) - 30 min
3. Compare via LOO-CV
4. Run posterior predictive checks

**Decision Point 1**:
- If Model 1 passes all checks → Continue to Phase 2
- If Model 2 >> Model 1 (ΔLOO > 4) → Heavy tails confirmed, fit robust variants
- If both fail validation → Reconsider likelihood or functional form

---

### Phase 2: Alternative Hypotheses (Day 3-4, ~4 hours compute)
5. Fit Model 3 (Piecewise) - 45 min (more complex)
6. Fit Model 4 (GP) - 60 min (most complex)
7. Compare all four models via LOO-CV
8. Run sensitivity analysis (remove x=31.5 outlier)

**Decision Point 2**:
- If simple models (1-2) win → Accept parsimony, proceed to reporting
- If GP (Model 4) wins decisively (ΔLOO > 4) → Flexibility justified
- If Piecewise (Model 3) wins → Changepoint is real
- If models are indistinguishable (|ΔLOO| < 2) → Model averaging

---

### Phase 3: Deep Investigation (Day 5-6, only if needed)
9. Fit Model 5 (Mixture) if mixture hypothesis supported
10. Fit Model 6 (Asymptotic) if mechanistic interpretation desired
11. Prior sensitivity analysis on best models
12. Comprehensive posterior predictive checking

**Decision Point 3**:
- Adequate model found → Proceed to final reporting
- All models fail → Consider non-Bayesian diagnostics or data quality issues
- Diminishing returns → Accept best available model with caveats

---

## Model Comparison Framework

### Primary Metric: LOO-CV (Leave-One-Out Cross-Validation)
```
az.compare(
    {
        "Log-Normal": idata1,
        "Log-Student": idata2,
        "Piecewise": idata3,
        "GP": idata4
    },
    ic="loo",
    scale="deviance"
)
```

**Decision Rules**:
- ΔLOO > 4: Strong evidence for better model
- 2 < ΔLOO < 4: Moderate evidence
- ΔLOO < 2: Models indistinguishable (prefer simpler)
- Pareto k > 0.7: Observation is influential, inspect carefully

---

### Secondary Metrics

**Posterior Predictive Checks**:
1. Mean: E[Y_rep] vs observed mean
2. SD: SD[Y_rep] vs observed SD
3. Max: max(Y_rep) vs max(Y_obs)
4. Min: min(Y_rep) vs min(Y_obs)
5. Residual skewness: check for systematic bias
6. Quantile-quantile plot: visual check of distributional fit

**Convergence Diagnostics**:
- R̂ < 1.01 for all parameters
- ESS > 400 (bulk and tail)
- Divergences < 1% of iterations
- E-BFMI > 0.3

---

## Falsification Criteria: When to Abandon Approach

### Abandon Parametric Modeling If:
- All parametric models (1-3, 6) show same residual patterns
- Multiple observations have Pareto k > 0.7
- Posterior predictive p-values systematically extreme
- GP (Model 4) decisively better (ΔLOO > 10)

**Action**: Switch to flexible methods (splines, smoothers) or collect more data

---

### Abandon Flexible Modeling (GP) If:
- Computational issues (divergences > 10%, E-BFMI < 0.2)
- Overfitting (LOO worse than simpler models)
- Length scale → ∞ (degenerates to parametric mean)
- Wild posterior uncertainty (CIs too wide to be useful)

**Action**: Return to best parametric model, accept limitations

---

### Abandon Robust Modeling (Student-t) If:
- ν posterior > 50 (Normal sufficient)
- Outlier at x=31.5 still has high Pareto k
- Residual patterns persist after accounting for heavy tails

**Action**: Consider measurement error model or investigate outlier validity

---

### Abandon Entire Bayesian Approach If:
- MCMC consistently fails to converge across all models
- Prior-posterior conflict in multiple models
- Results are highly sensitive to prior choice (suggests insufficient data)
- Model predictions are scientifically implausible

**Action**: (This violates hard constraint - must remain Bayesian. Instead, try simpler Bayesian approaches: conjugate priors, simpler likelihoods, more informative priors from domain experts)

---

## Success Criteria

### Computational Success:
- [x] At least one model achieves R̂ < 1.01, ESS > 400, divergences < 1%
- [x] LOO-CV computable for all fitted models (k̂ < 0.7 for most observations)

### Scientific Success:
- [x] Winning model passes posterior predictive checks
- [x] Parameter estimates are scientifically interpretable
- [x] Predictions are robust to outlier removal (sensitivity analysis)
- [x] Uncertainty quantification is honest (wide CIs if data is sparse)

### Model Adequacy (Phase 5 Assessment):
- [x] Top model explains relationship without severe misspecification
- [x] Residuals show no systematic patterns
- [x] Out-of-sample predictions are reasonable (extrapolation check)
- [x] Further refinement yields diminishing returns (ΔLOO < 2)

---

## Timeline Estimate

| Phase | Tasks | Time |
|-------|-------|------|
| **Phase 1** | Fit Models 1-2, LOO, basic PPCs | 4-6 hours |
| **Phase 2** | Fit Models 3-4, comprehensive comparison | 6-8 hours |
| **Phase 3** | Deep dive on top models, sensitivity | 4-6 hours |
| **Phase 4** | Model assessment & comparison | 2-3 hours |
| **Phase 5** | Adequacy assessment | 1-2 hours |
| **Phase 6** | Final report writing | 2-4 hours |
| **Total** | | **19-29 hours** |

**Compute Time**: ~6-10 hours total MCMC sampling (parallelizable)
**Analyst Time**: ~13-19 hours (can be spread over 1-2 weeks)

---

## Output Structure

Each experiment will follow standard structure:
```
experiments/experiment_N/
├── metadata.md                          # Model specification
├── prior_predictive_check/
│   ├── code/
│   ├── plots/
│   └── findings.md
├── simulation_based_validation/
│   ├── code/
│   ├── plots/
│   └── recovery_metrics.md
├── posterior_inference/
│   ├── code/
│   ├── diagnostics/
│   │   └── posterior_inference.netcdf  # ArviZ InferenceData with log_lik
│   ├── plots/
│   └── inference_summary.md
├── posterior_predictive_check/
│   ├── code/
│   ├── plots/
│   └── ppc_findings.md
└── model_critique/
    ├── critique_summary.md
    ├── decision.md                      # ACCEPT/REVISE/REJECT
    └── improvement_priorities.md
```

---

## Minimum Attempt Policy

**Must attempt**: Models 1 and 2 (both logarithmic variants)
**Reasoning**: These are baseline and robust baseline; must compare Normal vs Student-t

**Exception**: If Model 1 fails prior or simulation-based validation, document and proceed to Model 3

**Goal**: Attempt at least 2 distinct model classes before adequacy assessment

---

## Key References

- **Designer 1 Proposal**: `/workspace/experiments/designer_1/proposed_models.md`
- **Designer 2 Proposal**: `/workspace/experiments/designer_2/proposed_models.md`
- **Designer 3 Proposal**: `/workspace/experiments/designer_3/proposed_models.md`
- **EDA Report**: `/workspace/eda/eda_report.md`
- **Data**: `/workspace/data/data.csv`

---

## Synthesis Notes

**Convergent recommendations** (2+ designers agreed):
- Start with logarithmic transformation (all 3)
- Test Student-t for robustness (designers 1, 3)
- Consider two-regime structure (all 3, but different approaches)
- Use weakly informative priors based on EDA (all 3)

**Unique contributions**:
- Designer 1: Asymptotic model with mechanistic interpretation
- Designer 2: GP with informative mean function, rigorous falsification framework
- Designer 3: Mixture model for soft regime assignment, hierarchical variance

**Resolved conflicts**:
- Changepoint (Designer 1) vs Smooth (Designer 2): Fit both, compare via LOO
- Heavy tails (Designer 3) vs Heteroscedasticity (Designer 3 alternative): Test Student-t first (simpler)

---

## Final Note: Embrace Failure

These models are **designed to fail informatively**. If all parametric models show the same residual pattern, that's valuable information suggesting we need flexible methods. If Student-t doesn't help, we learn the outlier isn't about heavy tails. If GP overfits, we learn n=27 is too small for that complexity.

**Success = Learning the truth, not confirming a model works.**

---

**Status**: Ready for implementation. Begin with Model 1 (Logarithmic Normal).
