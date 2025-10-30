# Experiment 2: Logarithmic Model with Student-t Likelihood

**Date Created**: 2025-10-28
**Status**: In Progress
**Model Class**: Parametric (Logarithmic Transformation) with Robust Likelihood
**Likelihood**: Student-t (heavy-tailed)

---

## Model Specification

### Likelihood
```
Y_i ~ StudentT(ν, μ_i, σ)
μ_i = β₀ + β₁*log(x_i)
```

### Priors
```
β₀ ~ Normal(2.3, 0.5)       # Intercept: centered at observed mean Y
β₁ ~ Normal(0.29, 0.15)     # Slope: centered at EDA estimate
σ ~ Exponential(10)         # Scale: mean = 0.1
ν ~ Gamma(2, 0.1)           # Degrees of freedom: mean=20, allows 3-100 range
```

---

## Theoretical Justification

**Why Student-t Likelihood?**
- Robust to outliers through heavy tails
- Automatically downweights extreme observations
- Model 1 showed minor Q-Q tail deviations
- One potential outlier at x=31.5 flagged in EDA
- When ν→∞, converges to Normal (tests if robustness needed)

**Why Same Functional Form as Model 1?**
- Model 1 passed all validation checks
- Functional form (logarithmic) is well-justified
- This is a likelihood comparison, not functional form test
- Isolates the effect of tail robustness

**Degrees of Freedom Interpretation**:
- ν < 5: Very heavy tails, strong outlier protection
- ν ∈ [5, 30]: Moderate robustness
- ν > 30: Nearly Normal
- If posterior ν > 50, Normal likelihood is adequate (Model 1 preferred)

---

## Comparison to Model 1

**Model 1** (Logarithmic Normal):
- R²=0.889, RMSE=0.087, LOO-ELPD=24.89±2.82
- All Pareto k < 0.5
- PPC: 10/10 test statistics OK
- Decision: ACCEPTED

**Model 2** (this model):
- Tests if heavy tails improve fit
- Expected: Similar fit if ν > 30
- Expected: Better LOO if ν < 20 (robustness helps)

---

## Expected Outcomes

**Parameters**:
- β₀ ∈ [1.7, 2.6] (similar to Model 1)
- β₁ ∈ [0.15, 0.45] (similar to Model 1)
- σ ∈ [0.07, 0.12] (similar to Model 1)
- ν ∈ [10, 40] (posterior will inform if robustness needed)

**Model Fit**:
- R² ≈ 0.88-0.91 (similar to Model 1)
- LOO-ELPD ≈ 25-27 (slightly better if heavy tails justified)
- Possibly wider credible intervals (more uncertainty from heavy tails)

---

## Falsification Criteria

**I will REJECT this model if:**
1. ν posterior > 50 (Normal sufficient, Model 1 preferred by parsimony)
2. LOO worse than Model 1 (ΔLOO < -2, complexity penalty not justified)
3. Residual patterns persist (Student-t didn't solve issues)
4. Convergence failures (R̂ > 1.01, ESS < 400, divergences > 5%)

**I will ACCEPT this model if:**
1. ν posterior ∈ [5, 30] (robustness justified)
2. LOO improves over Model 1 (ΔLOO > 2)
3. All convergence diagnostics pass
4. PPC shows no systematic failures

**I will prefer Model 1 (Normal) if:**
1. ν posterior > 30 (tails not heavy enough to justify complexity)
2. ΔLOO < 2 (no substantial improvement)
3. Parameters nearly identical (robustness didn't change inference)

---

## Model Strengths

- **Automatic robustness**: Downweights outliers without manual intervention
- **Diagnostic value**: ν posterior tells us if heavy tails are needed
- **Generalizes Model 1**: Includes Normal as special case (ν→∞)
- **Well-studied**: Student-t regression is standard robust approach

---

## Model Weaknesses

- **One extra parameter**: More complex than Model 1
- **Slower MCMC**: Student-t likelihood is computationally heavier
- **May not help**: If ν > 30, complexity not justified
- **Same functional form**: Still assumes logarithmic saturation

---

## Decision Rules for Model Comparison

**Choose Model 2 (Student-t) if**:
- ν < 20 AND ΔLOO > 2 (robustness justified and improves fit)

**Choose Model 1 (Normal) if**:
- ν > 30 OR ΔLOO < 2 (parsimony preferred)

**Equivalence threshold**: |ΔLOO| < 2 → Models indistinguishable, prefer simpler (Model 1)

---

## References

- **Model 1**: `/workspace/experiments/experiment_1/`
- **EDA Report**: `/workspace/eda/eda_report.md`
- **Experiment Plan**: `/workspace/experiments/experiment_plan.md`
- **Designer 3 Proposal**: `/workspace/experiments/designer_3/proposed_models.md` (Student-t primary)
- **Designer 1 Proposal**: `/workspace/experiments/designer_1/proposed_models.md` (Student-t variant)
