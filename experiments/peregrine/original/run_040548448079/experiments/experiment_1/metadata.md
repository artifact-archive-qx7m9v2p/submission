# Experiment 1: Fixed Changepoint Negative Binomial Regression

## Model Class
Bayesian Negative Binomial regression with:
- Fixed structural break at observation 17
- AR(1) autocorrelation structure
- Log link function

## Mathematical Specification

### Observation Model
```
C_t ~ NegativeBinomial(μ_t, α)
log(μ_t) = β_0 + β_1 × year_t + β_2 × I(t > 17) × (year_t - year_17) + ε_t
```

### Autocorrelation Structure
```
ε_t ~ Normal(ρ × ε_{t-1}, σ_ε) for t > 1
ε_1 ~ Normal(0, σ_ε / √(1 - ρ²))  # stationary initialization
```

### Parameters
- **β_0**: Intercept (log-rate at year=0)
- **β_1**: Pre-break slope
- **β_2**: Additional slope post-break (β_1 + β_2 = total post-break slope)
- **α**: Dispersion parameter (Negative Binomial parameterization: variance = μ + α×μ²)
- **ρ**: AR(1) coefficient
- **σ_ε**: Innovation standard deviation

## Priors (REVISED VERSION)

**Original priors** (prior predictive check identified ρ issue):
```
β_0 ~ Normal(4.3, 0.5)
β_1 ~ Normal(0.35, 0.3)
β_2 ~ Normal(0.85, 0.5)
α ~ Gamma(2, 3)
ρ ~ Beta(8, 2)              # TOO CONSERVATIVE
σ_ε ~ Exponential(2)
```

**Revised priors** (current):
```
β_0 ~ Normal(4.3, 0.5)      # log(median(C)) ≈ 4.31 at year=0
β_1 ~ Normal(0.35, 0.3)     # pre-break slope from EDA
β_2 ~ Normal(0.85, 0.5)     # post-break increase
α ~ Gamma(2, 3)             # E[α] ≈ 0.67, EDA α ≈ 0.61
ρ ~ Beta(12, 1)             # E[ρ] ≈ 0.923, allows stronger autocorrelation
σ_ε ~ Exponential(2)        # E[σ_ε] = 0.5
```

**Justification** (from EDA):
- β_0: log(74.5) ≈ 4.31 (median count at year≈0)
- β_1: Pre-break slope ≈ 0.35 (14.87/exp(4.31) on original scale)
- β_2: Post-break increase ≈ 0.85 (post slope 1.2 minus pre slope 0.35)
- α: EDA estimated α ≈ 0.61 for NB
- ρ: EDA ACF(1) = 0.944; Beta(12,1) gives E[ρ] ≈ 0.923, 90% CI [0.79, 0.99]
- σ_ε: Weakly informative, allows data to determine innovation variance

## Constants
- **τ**: Changepoint at observation 17 (fixed, from EDA)
- **year_17**: Standardized year at observation 17 ≈ -0.213

## Data
- **Source**: `/workspace/data/data.csv`
- **Observations**: 40
- **Variables**: year (standardized), C (counts)

## EDA Summary
- Structural break at t=17: 730% growth rate increase
- Strong autocorrelation: ACF(1) = 0.944
- Overdispersion: variance/mean = 67.99
- Range: C ∈ [19, 272]
- Growth: 745% total increase over 40 observations

## Falsification Criteria

This model should be **REJECTED** if:

1. **No regime change**: β_2 posterior 95% CI includes 0
2. **Autocorrelation not captured**: Residual ACF(1) > 0.5
3. **LOO failure**: Pareto k > 0.7 for >10% of observations
4. **Convergence failure**: Rhat > 1.01 or ESS_bulk < 400 for any parameter
5. **Systematic misfit**: Posterior predictive checks show clear pattern around t=17
6. **Parameter nonsense**: Posteriors at extreme values or prior-posterior conflict

## Implementation

### Tool
- **Primary**: Stan (CmdStanPy)
- **Fallback**: PyMC if Stan has numerical issues

### Parameterization
- Use **non-centered** for ε_t: ε_t = ρ×ε_{t-1} + σ_ε×z_t where z_t ~ Normal(0,1)

### Sampling Configuration
- **Chains**: 4
- **Iterations**: 2000 (1000 warmup, 1000 sampling)
- **adapt_delta**: 0.95 (increase if divergences)
- **max_treedepth**: 12

### Log-Likelihood
Compute `vector[N] log_lik` in generated quantities block for LOO:
```stan
generated quantities {
  vector[N] log_lik;
  for (t in 1:N) {
    log_lik[t] = neg_binomial_2_lpmf(C[t] | mu[t], 1/alpha);
  }
}
```

## Expected Runtime
- Prior predictive check: 2-3 minutes
- Simulation-based calibration: 30-60 minutes
- Posterior inference: 5-10 minutes
- Posterior predictive checks: 5 minutes
- **Total**: 45-80 minutes

## Validation Status

- [x] Prior predictive check: COMPLETED (revised priors)
- [ ] Simulation-based calibration: PENDING
- [ ] Posterior inference: PENDING
- [ ] Posterior predictive check: PENDING
- [ ] Model critique: PENDING

## Version History

- v1.0: Initial specification with Beta(8,2) for ρ
- v1.1: **REVISED** ρ ~ Beta(12,1) based on prior predictive check results (ACF issue)

---

**Status**: Ready for simulation-based calibration with revised priors
**Last updated**: Current session
