# Experiment 1: Robust Logarithmic Regression

**Model Class:** Parametric regression with robust likelihood
**Status:** In Progress
**Date Started:** 2024-10-27

## Model Specification

### Likelihood
```
Y_i ~ StudentT(ν, μ_i, σ)
```

### Mean Function
```
μ_i = α + β·log(x_i + c)
```

### Parameters
- **α** (intercept): Location of curve
- **β** (slope): Rate of increase per log-unit of x
- **c** (shift): Translation constant for log
- **ν** (degrees of freedom): Tail heaviness (robustness parameter)
- **σ** (scale): Residual standard deviation

### Prior Distributions

**REVISED (after prior predictive check failure):**
```stan
alpha ~ normal(2.0, 0.5);         // Y centered ~2.3
beta ~ normal(0.3, 0.2);          // Positive slope, EDA ~0.27-0.30 [TIGHTENED]
c ~ gamma(2, 2);                  // Mean=1, standard log(x+1)
nu ~ gamma(2, 0.1);               // Mean=20, allows heavy tails
sigma ~ normal(0, 0.15);          // Half-normal, lighter tails [CHANGED]
  // with lower=0 constraint in Stan
```

**Changes from original:**
- **beta**: SD 0.3 → 0.2 (reduce negative slope probability)
- **sigma**: half_cauchy(0, 0.2) → half_normal(0, 0.15) (eliminate compound heavy-tail problem)

**Original (failed prior predictive check):**
```stan
sigma ~ half_cauchy(0, 0.2);      // TOO HEAVY TAILS
beta ~ normal(0.3, 0.3);          // TOO DIFFUSE
```

## Justification

### Why This Model?
1. **EDA support**: R²=0.888, best simple functional form
2. **Designer consensus**: All 3 independent designers ranked this #1
3. **Parsimony**: Only 5 parameters with n=27
4. **Robustness**: Student-t handles outlier at x=31.5
5. **Interpretability**: Clear diminishing returns interpretation

### Theoretical Basis
- Logarithmic relationship common in learning curves, dose-response
- Diminishing returns pattern evident in data
- No unbounded growth (log increases slowly at large x)

## Falsification Criteria

Will abandon this model if:

1. **Posterior ν < 5**: Extreme heavy tails suggest multiple outliers or misspecification
2. **Systematic residual patterns**: Runs test p < 0.05 or clear visual trend
3. **Change-point model wins**: ΔWAIC(Model 2 - Model 1) > 6
4. **Log shift at boundary**: Posterior c > 4 or c < 0.2
5. **Replicate prediction failure**: Coverage < 60% on replicated x values

## Validation Pipeline Status

- [ ] Prior predictive check
- [ ] Simulation-based validation
- [ ] Model fitting
- [ ] Posterior predictive check
- [ ] Model critique

## Notes

This is the PRIMARY model with highest expected success (80%). If this model passes all validation, it may be sufficient without fitting additional models.
