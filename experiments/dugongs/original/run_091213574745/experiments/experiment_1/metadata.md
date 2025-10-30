# Experiment 1: Logarithmic Model with Normal Likelihood

**Date Created**: 2025-10-28
**Status**: In Progress
**Model Class**: Parametric (Logarithmic Transformation)
**Likelihood**: Normal (Gaussian)

---

## Model Specification

### Likelihood
```
Y_i ~ Normal(μ_i, σ)
μ_i = β₀ + β₁*log(x_i)
```

### Priors
```
β₀ ~ Normal(2.3, 0.3)      # Intercept: centered at observed mean Y
β₁ ~ Normal(0.29, 0.15)    # Slope: centered at EDA estimate, weakly informative
σ ~ Exponential(10)         # Scale: mean = 0.1, matching observed RMSE ≈ 0.087
```

---

## Theoretical Justification

**Why Logarithmic?**
- EDA shows strong fit: R² = 0.897, RMSE = 0.087
- Represents smooth saturation process (Weber-Fechner law)
- Each doubling of x produces same additive gain in Y
- Common in dose-response, learning curves, diminishing returns

**Why Normal Likelihood?**
- Baseline assumption for continuous responses
- EDA shows no strong evidence of heteroscedasticity
- Will be compared against Student-t for robustness testing

---

## Expected Outcomes

**Parameters**:
- β₀ ∈ [1.8, 2.6] (centered near 2.3)
- β₁ ∈ [0.15, 0.45] (centered near 0.29)
- σ ∈ [0.07, 0.12] (near EDA RMSE of 0.087)

**Model Fit**:
- R² ≈ 0.90
- LOO-ELPD ≈ 10-15 (baseline)
- Most Pareto k < 0.5, possibly k > 0.7 for x=31.5 outlier

---

## Falsification Criteria

**I will REJECT this model if:**
1. Residuals show clear two-regime clustering
2. Posterior predictive p-values < 0.05 or > 0.95
3. Student-t model (Experiment 2) improves LOO by ΔLOO > 4
4. Multiple observations have Pareto k > 0.7
5. Parameter estimates are scientifically implausible

**I will ACCEPT this model if:**
1. All convergence diagnostics pass (R̂ < 1.01, ESS > 400, div < 1%)
2. Posterior predictive checks pass
3. Residuals show no systematic patterns
4. LOO-CV is competitive with alternatives (ΔLOO < 4)

**I will REVISE this model if:**
1. Minor misspecification detected (e.g., slight heteroscedasticity)
2. Outlier handling needed but overall structure is correct
3. Prior sensitivity suggests need for adjustment

---

## Model Strengths

- **Simplicity**: Only 3 parameters (β₀, β₁, σ)
- **Interpretability**: Clear mechanistic interpretation (diminishing returns)
- **Established baseline**: Matches EDA results
- **Computational efficiency**: Fast MCMC sampling

---

## Model Weaknesses

- **Assumes homoscedasticity**: Variance constant across x
- **Smooth saturation only**: Cannot capture sharp regime changes
- **Outlier sensitivity**: Normal likelihood downweights outliers weakly
- **Assumes monotonic**: Cannot detect non-monotonic patterns

---

## Validation Pipeline

1. **Prior Predictive Check**: Verify priors generate reasonable data
2. **Simulation-Based Validation**: Check parameter recovery with known truth
3. **Posterior Inference**: Fit model to real data with MCMC
4. **Posterior Predictive Check**: Validate fit to observed data
5. **Model Critique**: Comprehensive assessment and decision

---

## Comparison Context

**Compared to**:
- Model 2 (Student-t): Tests robustness to outlier
- Model 3 (Piecewise): Tests sharp changepoint hypothesis
- Model 4 (GP): Tests flexible nonparametric alternative

**Decision Rule**:
- If this model wins (lowest LOO): Accept parsimony
- If Student-t wins: Heavy tails matter
- If Piecewise/GP win: Need more complexity

---

## References

- **EDA Report**: `/workspace/eda/eda_report.md`
- **Experiment Plan**: `/workspace/experiments/experiment_plan.md`
- **Designer Proposals**:
  - `/workspace/experiments/designer_1/proposed_models.md` (primary)
  - `/workspace/experiments/designer_2/proposed_models.md` (GP mean function)
  - `/workspace/experiments/designer_3/proposed_models.md` (baseline for Student-t)
