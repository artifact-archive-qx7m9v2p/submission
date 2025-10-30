# Experiment 3: Log-Log Power Law Model

## Model Specification

**Hypothesis**: The relationship follows a power law with constant elasticity, which becomes linear in log-log space.

### Functional Form
```
Transformed scale:
log(Y_i) ~ Normal(μ_i, σ)
μ_i = α + β * log(x_i)

Original scale (equivalent):
Y_i = exp(α) * x_i^β * ε_i,  where log(ε_i) ~ Normal(0, σ)
```

### Parameters
- `α`: Intercept on log-log scale (relates to scaling constant)
- `β`: Power law exponent (elasticity of Y with respect to x)
- `σ`: Residual standard deviation on log scale

### Priors

```
α ~ Normal(0.6, 0.3)       # Centered at log(1.8) ≈ 0.59
β ~ Normal(0.12, 0.05)     # Centered at OLS estimate, tightened per PPC
σ ~ Half-Cauchy(0, 0.05)   # Log-scale residual variation, tightened per PPC
```

**Prior Justification**:
- α: log(Y) ranges [0.54, 0.97], center prior at 0.6 with SD covering plausible range
- β: OLS log-log fit from EDA gives β ≈ 0.121, prior allows [0.02, 0.22] at 95%. **Revised** from SD=0.1 to 0.05 to reduce negative draws (11.8% → 0.9%)
- σ: On log scale, tight variation expected; half-Cauchy provides robustness. **Revised** scale from 0.1 to 0.05 to reduce heavy tail extremes (5.7% > 1.0 → 0.5% > 1.0)

**Revision History**:
- v1: Original priors from designer proposals
- v2: Tightened β and σ based on prior predictive check (trajectory pass rate: 62.8% → ~85%)

## Theoretical Justification

**Why this model?**
1. **Empirical**: EDA shows log-log transformation achieves r=0.92 (strongest linearity observed)
2. **Theoretical**: Power laws ubiquitous in natural phenomena (allometry, scaling laws)
3. **Mathematical**: Transforms nonlinear problem to simple linear regression
4. **Computational**: Fast inference, no nonlinear optimization needed
5. **Robust**: Multiplicative errors often more realistic than additive

**Power Law Interpretation**:
- Y = exp(α) * x^β
- β is the elasticity: 1% increase in x → β% increase in Y
- Diminishing returns naturally captured by 0 < β < 1

## Falsification Criteria

**Abandon this model if**:
1. R² < 0.75 (on original scale after back-transformation)
2. Residuals on log-log scale show systematic curvature
3. Back-transformed predictions systematically deviate from observed Y
4. β posterior includes zero (no relationship)
5. σ on log scale exceeds 0.3 (implies transformation not helpful)

## Expected Performance

- **R²**: ~0.81 (based on EDA OLS log-log fit)
- **Convergence**: Excellent (linear model, no convergence issues expected)
- **Speed**: Very fast (<10 seconds)
- **Interpretability**: Good (power law exponent has clear meaning)

## Implementation

- **PPL**: Stan (CmdStanPy)
- **Likelihood**: Gaussian on log scale
- **Data transformation**: log(x) and log(Y) computed beforehand
- **Post-processing**: Back-transform predictions to original scale for validation
- **LOO**: Save log_likelihood for model comparison

## Validation Focus

- **Prior predictive**: Check that exp(α + β*log(x)) stays in [1.0, 3.5]
- **Posterior predictive**: Compare predictions on original Y scale (not log scale)
- **Residuals**: Check for homoscedasticity and normality on log scale
- **Replicates**: Verify good fit at 6 replicated x-values
- **Extrapolation**: Check behavior for x > 31.5 (should continue power law trend)

## Status

- Created: 2025-10-27
- Status: Ready for implementation
