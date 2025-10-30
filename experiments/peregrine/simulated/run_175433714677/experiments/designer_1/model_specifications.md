# Model Specifications: Quick Reference

## Model 1: Log-Linear Negative Binomial (RECOMMENDED BASELINE)

### Mathematical Form
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i]
```

### Priors
```
β₀ ~ Normal(4.3, 1.0)
β₁ ~ Normal(0.85, 0.5)
φ  ~ Exponential(0.667)    # E[φ] = 1.5
```

### Expected Posteriors
- β₀: 4.3 ± 0.15 (log-scale intercept)
- β₁: 0.85 ± 0.10 (growth rate = 134% per year)
- φ: 1.5 ± 0.5 (overdispersion)

### Stan Implementation
```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> y;
  vector[N] year;
}

parameters {
  real beta_0;
  real beta_1;
  real<lower=0> phi;
}

model {
  beta_0 ~ normal(4.3, 1.0);
  beta_1 ~ normal(0.85, 0.5);
  phi ~ exponential(0.667);

  y ~ neg_binomial_2_log(beta_0 + beta_1 * year, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;

  for (i in 1:N) {
    real mu_i = beta_0 + beta_1 * year[i];
    log_lik[i] = neg_binomial_2_log_lpmf(y[i] | mu_i, phi);
    y_rep[i] = neg_binomial_2_log_rng(mu_i, phi);
  }
}
```

---

## Model 2: Quadratic Negative Binomial (IF MODEL 1 SHOWS CURVATURE)

### Mathematical Form
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]
```

### Priors
```
β₀ ~ Normal(4.3, 1.0)
β₁ ~ Normal(0.85, 0.5)
β₂ ~ Normal(0, 0.3)
φ  ~ Exponential(0.667)
```

### Expected Posteriors
- β₀: 4.3 ± 0.15
- β₁: 0.85 ± 0.10
- β₂: 0.3 ± 0.15 (acceleration term)
- φ: 1.5 ± 0.5

### Stan Implementation
```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> y;
  vector[N] year;
  vector[N] year_sq;  // Precompute year^2
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real<lower=0> phi;
}

model {
  beta_0 ~ normal(4.3, 1.0);
  beta_1 ~ normal(0.85, 0.5);
  beta_2 ~ normal(0, 0.3);
  phi ~ exponential(0.667);

  y ~ neg_binomial_2_log(beta_0 + beta_1 * year + beta_2 * year_sq, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;

  for (i in 1:N) {
    real mu_i = beta_0 + beta_1 * year[i] + beta_2 * year_sq[i];
    log_lik[i] = neg_binomial_2_log_lpmf(y[i] | mu_i, phi);
    y_rep[i] = neg_binomial_2_log_rng(mu_i, phi);
  }
}
```

---

## Sampling Configuration

### Initial Settings
```python
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.8,
    max_treedepth=10,
    seed=12345
)
```

### If Divergences Occur
```python
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=2000,
    iter_sampling=1000,
    adapt_delta=0.99,  # More conservative
    max_treedepth=12,
    seed=12345
)
```

---

## Model Comparison

### LOO-CV Decision Rules
- **ΔELPD < 2**: Choose simpler model (Model 1)
- **2 < ΔELPD < 4**: Weak evidence, consider interpretability
- **ΔELPD > 4**: Strong evidence for complex model (Model 2)

### Convergence Checklist
- [ ] R̂ < 1.01 for all parameters
- [ ] ESS_bulk > 400
- [ ] ESS_tail > 400
- [ ] Divergences < 0.5%
- [ ] No hitting max treedepth
- [ ] BFMI > 0.3

### Posterior Predictive Checks
- [ ] Variance-to-mean ratio ≈ 70 ± 20
- [ ] 90% prediction intervals contain ~90% of observations
- [ ] Can generate values up to observed max (269)
- [ ] Residuals show no systematic patterns

---

## Python Analysis Template

```python
import cmdstanpy
import arviz as az
import numpy as np
import pandas as pd

# Load data
data = pd.read_json('/workspace/data.json')
stan_data = {
    'N': len(data),
    'y': data['C'].values,
    'year': data['year'].values
}

# Fit Model 1
model1 = cmdstanpy.CmdStanModel(stan_file='model1_linear.stan')
fit1 = model1.sample(data=stan_data, chains=4, iter_warmup=1000,
                      iter_sampling=1000, adapt_delta=0.8)

# Convert to ArviZ
idata1 = az.from_cmdstanpy(fit1)

# Diagnostics
print(az.summary(idata1, var_names=['beta_0', 'beta_1', 'phi']))
print(f"Divergences: {fit1.num_divergences()}")

# LOO-CV
loo1 = az.loo(idata1, pointwise=True)
print(f"LOO-ELPD: {loo1.elpd_loo:.2f} ± {loo1.se:.2f}")

# Posterior predictive check
y_rep = idata1.posterior_predictive['y_rep'].values
ppc_var_mean = np.var(y_rep, axis=(0,1)) / np.mean(y_rep, axis=(0,1))
print(f"Posterior predictive Var/Mean: {np.mean(ppc_var_mean):.2f}")
print(f"Observed Var/Mean: {np.var(data['C']) / np.mean(data['C']):.2f}")

# Compare models (if Model 2 fitted)
# comp = az.compare({'Linear': idata1, 'Quadratic': idata2})
# print(comp)
```

---

## Key Parameters Interpretation

### β₀ (Intercept)
- **Scale**: Log-scale
- **Interpretation**: log(expected count) when year = 0 (middle of time series)
- **Exponentiated**: exp(4.3) ≈ 73.7 counts at year = 0

### β₁ (Linear slope)
- **Scale**: Log-scale per unit year
- **Interpretation**: Growth rate on log-scale
- **Exponentiated**: exp(0.85) ≈ 2.34 = 134% growth per year

### β₂ (Quadratic term) - Model 2 only
- **Scale**: Log-scale per unit year²
- **Interpretation**: Acceleration/deceleration in growth rate
- **Sign**: Positive = accelerating, Negative = decelerating

### φ (Dispersion)
- **Scale**: Positive real number
- **Interpretation**: Overdispersion parameter
- **Relationship**: Variance = μ + μ²/φ (larger φ = less overdispersion)
- **EDA estimate**: φ ≈ 1.5 (severe overdispersion)

---

## Falsification Thresholds

### Abandon Model 1 if:
1. LOO-CV: ΔELPD > 4 vs. Model 2
2. Posterior predictive Var/Mean < 50 or > 90
3. >20% of observations outside 90% prediction intervals
4. Clear U-shaped residual pattern vs. fitted values

### Abandon Model 2 if:
1. β₂ 95% CI includes 0 AND |β₂| < 0.1
2. LOO-CV: ΔELPD < 2 vs. Model 1
3. Pareto-k > 0.7 for >10% of observations
4. >0.5% divergent transitions

---

## Computational Expectations

| Model | Parameters | Warmup Time | Sampling Time | Total Time |
|-------|-----------|-------------|---------------|------------|
| Model 1 | 3 | ~5s | ~5s | ~10s |
| Model 2 | 4 | ~10s | ~5s | ~15s |

*Times estimated for 4 chains × 1000 iterations on standard CPU*

---

## Next Steps After Fitting

1. **Check convergence** (R̂, ESS, divergences)
2. **Posterior predictive checks** (Var/Mean, calibration)
3. **LOO-CV** (model comparison)
4. **Parameter interpretation** (compare to EDA)
5. **Sensitivity analysis** (prior robustness)
6. **Final model selection** (parsimony vs. fit)

---

## Contact & Collaboration

This design prioritizes **parsimony**. If other designers propose more complex models (e.g., GP, changepoint, hierarchical), use this as a **baseline** for comparison.

**Key question**: Is the added complexity worth it? Compare via ΔELPD and interpretability.
