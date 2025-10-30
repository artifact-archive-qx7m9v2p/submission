# Experiment 1: Log-Linear Negative Binomial (Baseline)

## Model Specification

**Likelihood**:
```
C[i] ~ NegativeBinomial(μ[i], φ)    for i = 1,...,40
log(μ[i]) = β₀ + β₁ × year[i]
```

**Priors**:
```
β₀ ~ Normal(4.3, 1.0)     # Intercept: log(mean count at year=0)
β₁ ~ Normal(0.85, 0.5)    # Slope: expected log-growth rate
φ  ~ Exponential(0.667)   # Dispersion: E[φ] = 1.5
```

## Theoretical Justification

1. **Simplicity**: Only 3 parameters, minimal complexity
2. **Interpretability**: β₁ = instantaneous growth rate, exp(β₁) = multiplicative factor per year
3. **EDA Support**: Linear fit has R² = 0.92
4. **Exponential growth**: Common in population dynamics, technology adoption

## Expected Parameters (from EDA)

- β₀: 4.3 ± 0.15 (log of mean count at standardized year = 0)
- β₁: 0.85 ± 0.10 (implies 134% annual growth)
- φ: 1.5 ± 0.5 (overdispersion parameter)

## Falsification Criteria

**I will REJECT this model if**:

1. **LOO-CV**: ΔELPD > 4 compared to quadratic model (strong evidence against)
2. **Systematic curvature**: Posterior predictive residuals show clear U-shape or inverted-U
3. **Late period failure**: Mean absolute error in last 10 observations is >2× early 10 observations
4. **Variance mismatch**: 95% of posterior predictive Var/Mean ratios fall outside [50, 90]
5. **Poor calibration**: <80% of observations fall within 90% prediction intervals

## Status

- Created: 2025-10-29
- Status: Starting validation pipeline
