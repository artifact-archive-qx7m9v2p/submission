# Experiment 2: Quadratic Negative Binomial

## Model Specification

**Likelihood**:
```
C[i] ~ NegativeBinomial(μ[i], φ)    for i = 1,...,40
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]
```

**Priors**:
```
β₀ ~ Normal(4.3, 1.0)     # Intercept: log(mean count at year=0)
β₁ ~ Normal(0.85, 0.5)    # Linear term: expected log-growth rate
β₂ ~ Normal(0, 0.5)       # Quadratic term: acceleration (centered at 0)
φ  ~ Exponential(0.667)   # Dispersion: E[φ] = 1.5
```

## Theoretical Justification

1. **Addresses Model 1 deficiency**: Captures accelerating growth (quadratic coefficient = -5.22 in Model 1 residuals)
2. **EDA support**: Quadratic fit R² = 0.96 vs. 0.92 for linear
3. **Growth acceleration**: 9.6× increase in growth rate from early to late period
4. **Moderate complexity**: 4 parameters (nested within Model 1)

## Expected Parameters (from EDA + Model 1 diagnostics)

- β₀: 4.3 ± 0.15 (log of mean count at standardized year = 0)
- β₁: 0.85 ± 0.10 (linear growth term)
- β₂: 0.3 ± 0.15 (positive acceleration expected based on EDA curvature)
- φ: 1.5-15 (dispersion - Model 1 suggested ~14 after accounting for trend)

## Falsification Criteria

**I will REJECT this model if**:

1. **β₂ not significant**: 95% credible interval includes 0 AND |β₂| < 0.1
2. **No improvement over Model 1**: LOO-CV ΔELPD < 2 (models equivalent)
3. **Residual curvature persists**: |quadratic coefficient in residuals| > 1.0
4. **Late period still poor**: MAE ratio (late/early) > 2.0
5. **Var/Mean still off**: 95% of posterior predictive samples outside [50, 90]

**I will ACCEPT this model if**:

1. **β₂ significantly positive**: 95% CI excludes 0, β₂ > 0.1
2. **Strong improvement**: LOO-CV ΔELPD > 4 vs. Model 1
3. **Residuals improved**: |curvature coefficient| < 1.0
4. **Better late period fit**: MAE ratio < 2.0
5. **Good calibration**: 80-95% coverage, Var/Mean recovery

## Status

- Created: Based on Model 1 critique recommendations
- Motivation: Model 1 showed clear inverted-U residual pattern
- Expected: Substantial improvement in all diagnostics
