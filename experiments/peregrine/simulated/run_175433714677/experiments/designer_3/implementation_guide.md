# Implementation Guide: Alternative Bayesian Models

**Purpose**: Detailed implementation instructions for the three proposed alternative models.

---

## Quick Reference

| Model | Stan File | Complexity | Expected Runtime | Key Challenge |
|-------|-----------|------------|------------------|---------------|
| Model 1: Hierarchical Gamma-Poisson | `model_1_hierarchical.stan` | Medium | ~2-5 min | May need marginalization |
| Model 2: Student-t Regression | `model_2_studentt.stan` | Low | ~1-2 min | Back-transformation to counts |
| Model 3: COM-Poisson | `model_3_compoisson.stan` | High | ~10-30 min | Custom likelihood, normalizing constant |

---

## Model 1: Hierarchical Gamma-Poisson

### Complete Stan Code

```stan
// Hierarchical Gamma-Poisson Model for Overdispersed Counts
// Equivalent to NegBin but with explicit random effects

data {
  int<lower=0> N;                    // number of observations
  array[N] int<lower=0> C;          // count data
  vector[N] year;                    // predictor (standardized)
}

parameters {
  real beta_0;                       // intercept on log-scale
  real beta_1;                       // slope (growth rate)
  real<lower=0> phi;                 // dispersion parameter (inverse of overdispersion)
  vector<lower=0>[N] lambda;         // latent random effects (intensities)
}

transformed parameters {
  vector[N] mu;                      // mean structure
  mu = exp(beta_0 + beta_1 * year);
}

model {
  // Priors
  beta_0 ~ normal(4.3, 1.0);         // based on EDA: log(mean) ≈ 4.3
  beta_1 ~ normal(0.85, 0.3);        // based on EDA: growth rate ≈ 0.85
  phi ~ gamma(2, 0.1);               // weakly informative, allows phi ~ 1-20

  // Random effects (hierarchical layer)
  // lambda[i] ~ Gamma(alpha, beta) where alpha = mu[i]*phi, beta = phi
  // This gives E[lambda[i]] = mu[i] and Var[lambda[i]] = mu[i]/phi
  lambda ~ gamma(mu * phi, phi);

  // Likelihood
  C ~ poisson(lambda);
}

generated quantities {
  vector[N] log_lik;                 // for LOO-CV
  array[N] int C_rep;                // posterior predictive samples
  real var_mean_ratio;               // check overdispersion recovery

  // Compute log-likelihood for each observation
  for (i in 1:N) {
    log_lik[i] = poisson_lpmf(C[i] | lambda[i]);
    C_rep[i] = poisson_rng(lambda[i]);
  }

  // Compute Var/Mean ratio of replicated data
  {
    real mean_C_rep = mean(to_vector(C_rep));
    real var_C_rep = variance(to_vector(C_rep));
    var_mean_ratio = var_C_rep / mean_C_rep;
  }
}
```

### Python Fitting Code

```python
import cmdstanpy
import numpy as np
import pandas as pd
import json
import arviz as az

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Prepare data for Stan
stan_data = {
    'N': len(data['C']),
    'C': data['C'],
    'year': data['year']
}

# Compile model
model = cmdstanpy.CmdStanModel(stan_file='model_1_hierarchical.stan')

# Fit model
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,
    max_treedepth=12,
    show_progress=True,
    seed=12345
)

# Check convergence
summary = fit.summary()
print("Convergence diagnostics:")
print(summary[summary['R_hat'] > 1.01])

# Check divergences
print(f"\nDivergent transitions: {fit.num_divergences()}")

# Convert to ArviZ InferenceData for analysis
idata = az.from_cmdstanpy(fit)

# Compute LOO-CV
loo = az.loo(idata, pointwise=True)
print(f"\nLOO-CV ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")

# Posterior predictive checks
C_rep = fit.stan_variable('C_rep')
var_mean_ratio = fit.stan_variable('var_mean_ratio')

print(f"\nObserved Var/Mean: {np.var(data['C']) / np.mean(data['C']):.2f}")
print(f"Predicted Var/Mean: {np.mean(var_mean_ratio):.2f} (95% CI: [{np.percentile(var_mean_ratio, 2.5):.2f}, {np.percentile(var_mean_ratio, 97.5):.2f}])")

# Save results
fit.save_csvfiles(dir='experiments/designer_3/fits/model_1/')
```

### Diagnostics to Check

1. **Convergence**:
   - Rhat < 1.01 for beta_0, beta_1, phi
   - ESS > 400 for all parameters
   - No divergent transitions

2. **Random effects structure**:
   ```python
   lambda_post = fit.stan_variable('lambda')
   lambda_mean = lambda_post.mean(axis=0)

   # Check if lambda correlates with year (heteroscedasticity signal)
   from scipy.stats import pearsonr
   corr, pval = pearsonr(data['year'], lambda_mean)
   print(f"Correlation(lambda, year): r={corr:.3f}, p={pval:.3f}")

   # If |corr| > 0.3, suggests time-varying dispersion
   ```

3. **Posterior predictive**:
   - Compare distribution of C_rep to C (KS test, QQ plot)
   - Var/Mean ratio should be ~70

### When to Abandon

- If phi posterior is extremely wide (CV > 1): Data doesn't constrain dispersion
- If lambda[i] shows no interpretable pattern: Hierarchy adds complexity without insight
- If sampling is slow (>10 min) or has divergences: Consider marginalized NegBin instead

---

## Model 2: Student-t Regression on Log-Counts

### Complete Stan Code

```stan
// Robust Student-t Regression on Log-Transformed Counts
// Alternative to count-based GLM

data {
  int<lower=0> N;                    // number of observations
  vector[N] y;                       // log(C + 0.5)
  vector[N] year;                    // predictor (standardized)
}

parameters {
  real beta_0;                       // intercept on log-scale
  real beta_1;                       // slope (growth rate)
  real<lower=0> sigma;               // residual standard deviation
  real<lower=1> nu;                  // degrees of freedom (tail heaviness)
}

transformed parameters {
  vector[N] mu;                      // mean structure
  mu = beta_0 + beta_1 * year;
}

model {
  // Priors
  beta_0 ~ normal(4.3, 1.0);         // log(mean count) ≈ 4.3
  beta_1 ~ normal(0.85, 0.3);        // growth rate ≈ 0.85
  sigma ~ exponential(1);            // weakly informative on residual scale
  nu ~ gamma(2, 0.1);                // allows nu from ~1 (Cauchy) to ~50 (Normal)

  // Likelihood (robust to outliers)
  y ~ student_t(nu, mu, sigma);
}

generated quantities {
  vector[N] log_lik;                 // for LOO-CV
  vector[N] y_rep;                   // replicated log-counts
  array[N] int C_rep;                // back-transformed to count scale
  real var_mean_ratio;               // check overdispersion on count scale

  // Simulate from posterior predictive
  for (i in 1:N) {
    log_lik[i] = student_t_lpdf(y[i] | nu, mu[i], sigma);
    y_rep[i] = student_t_rng(nu, mu[i], sigma);

    // Back-transform to count scale
    // Use Poisson with rate = exp(y_rep) to generate integer counts
    C_rep[i] = poisson_rng(exp(y_rep[i]));
  }

  // Compute Var/Mean ratio on count scale
  {
    real mean_C_rep = mean(to_vector(C_rep));
    real var_C_rep = variance(to_vector(C_rep));
    var_mean_ratio = var_C_rep / mean_C_rep;
  }
}
```

### Python Fitting Code

```python
import cmdstanpy
import numpy as np
import pandas as pd
import json
import arviz as az

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Transform counts to log-scale
y = np.log(np.array(data['C']) + 0.5)

# Prepare data for Stan
stan_data = {
    'N': len(data['C']),
    'y': y.tolist(),
    'year': data['year']
}

# Compile and fit
model = cmdstanpy.CmdStanModel(stan_file='model_2_studentt.stan')

fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.95,
    seed=12345
)

# Convergence diagnostics
summary = fit.summary()
print(summary.loc[['beta_0', 'beta_1', 'sigma', 'nu']])

# Key parameter: nu (degrees of freedom)
nu_post = fit.stan_variable('nu')
print(f"\nPosterior nu: {nu_post.mean():.2f} (95% CI: [{np.percentile(nu_post, 2.5):.2f}, {np.percentile(nu_post, 97.5):.2f}])")

if nu_post.mean() > 30:
    print("WARNING: nu > 30 suggests Normal-like data, Student-t may be unnecessary")

# Posterior predictive check on count scale
C_rep = fit.stan_variable('C_rep')
var_mean_ratio = fit.stan_variable('var_mean_ratio')

print(f"\nObserved Var/Mean: {np.var(data['C']) / np.mean(data['C']):.2f}")
print(f"Predicted Var/Mean: {np.mean(var_mean_ratio):.2f}")

# Check residuals on log-scale
y_rep = fit.stan_variable('y_rep')
residuals = y[:, None] - y_rep.T  # (N, n_samples)
residual_mean = residuals.mean(axis=1)

# Heteroscedasticity check
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(data['year'], residual_mean)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Year')
plt.ylabel('Residuals (log-scale)')
plt.title('Residual vs. Year')

plt.subplot(1, 2, 2)
plt.scatter(np.exp(y), np.abs(residual_mean))
plt.xlabel('Observed Count')
plt.ylabel('|Residuals|')
plt.title('Residual Magnitude vs. Count')

plt.tight_layout()
plt.savefig('experiments/designer_3/model_2_residuals.png')
plt.close()

# LOO-CV
idata = az.from_cmdstanpy(fit)
loo = az.loo(idata, pointwise=True)
print(f"\nLOO-CV ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")

# Save results
fit.save_csvfiles(dir='experiments/designer_3/fits/model_2/')
```

### Critical Validation

**Must check back-transformation**:

```python
# Compare empirical distribution of C_rep to observed C
from scipy.stats import ks_2samp

C_rep_flat = C_rep.flatten()
ks_stat, ks_pval = ks_2samp(data['C'], C_rep_flat)
print(f"KS test (C_rep vs C): D={ks_stat:.3f}, p={ks_pval:.3f}")

if ks_pval < 0.05:
    print("WARNING: Back-transformed predictions don't match observed distribution")
```

### When to Abandon

- If nu > 30 with high certainty: Just use Normal, no need for Student-t
- If back-transformed C_rep has wrong distribution (KS test fails)
- If residuals show strong patterns (heteroscedasticity not captured by constant σ)

---

## Model 3: COM-Poisson Regression

### Complete Stan Code

**Warning**: This is computationally expensive!

```stan
// Conway-Maxwell-Poisson Regression
// Flexible dispersion model for count data

functions {
  // Approximate COM-Poisson log PMF using truncation
  real com_poisson_lpmf(int y, real lambda, real nu, int max_count) {
    vector[max_count + 1] log_terms;
    real log_Z;  // log normalizing constant
    real log_prob;

    // Compute log normalizing constant
    for (k in 0:max_count) {
      log_terms[k + 1] = k * log(lambda) - nu * lgamma(k + 1);
    }
    log_Z = log_sum_exp(log_terms);

    // Compute log probability
    log_prob = y * log(lambda) - nu * lgamma(y + 1) - log_Z;

    return log_prob;
  }
}

data {
  int<lower=0> N;                    // number of observations
  array[N] int<lower=0> C;          // count data
  vector[N] year;                    // predictor (standardized)
  int<lower=1> max_count;           // truncation for normalizing constant (e.g., 500)
}

parameters {
  real beta_0;                       // intercept on log-scale
  real beta_1;                       // slope (growth rate)
  real<lower=0.01> nu;               // dispersion parameter
}

transformed parameters {
  vector[N] lambda;                  // rate parameter (NOT the mean!)
  lambda = exp(beta_0 + beta_1 * year);
}

model {
  // Priors
  beta_0 ~ normal(4.3, 1.0);
  beta_1 ~ normal(0.85, 0.3);
  nu ~ gamma(2, 2);                  // centered at 1 (Poisson), allows exploration

  // Likelihood
  for (i in 1:N) {
    target += com_poisson_lpmf(C[i] | lambda[i], nu, max_count);
  }
}

generated quantities {
  vector[N] log_lik;

  for (i in 1:N) {
    log_lik[i] = com_poisson_lpmf(C[i] | lambda[i], nu, max_count);
  }

  // Note: C_rep would require custom RNG, omitted for simplicity
}
```

### Python Fitting Code

```python
import cmdstanpy
import numpy as np
import json
import arviz as az

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Prepare data
# Set max_count to safely exceed observed maximum
max_count = int(max(data['C']) * 2)  # 2x safety margin

stan_data = {
    'N': len(data['C']),
    'C': data['C'],
    'year': data['year'],
    'max_count': max_count
}

# Compile (may take longer due to custom function)
print("Compiling COM-Poisson model (this may take a while)...")
model = cmdstanpy.CmdStanModel(stan_file='model_3_compoisson.stan')

# Fit with generous timeout
fit = model.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1500,  # longer warmup for difficult geometry
    iter_sampling=1000,
    adapt_delta=0.98,  # higher adapt_delta for stability
    max_treedepth=12,
    show_progress=True,
    seed=12345
)

# Check convergence
summary = fit.summary()
print(summary.loc[['beta_0', 'beta_1', 'nu']])

# Key parameter: nu
nu_post = fit.stan_variable('nu')
print(f"\nPosterior nu: {nu_post.mean():.3f} (95% CI: [{np.percentile(nu_post, 2.5):.3f}, {np.percentile(nu_post, 97.5):.3f}])")

if np.abs(nu_post.mean() - 1.0) < 0.1:
    print("WARNING: nu ≈ 1 suggests Poisson is adequate (contradicts EDA)")

# LOO-CV
idata = az.from_cmdstanpy(fit)
loo = az.loo(idata, pointwise=True)
print(f"\nLOO-CV ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")

# Save results
fit.save_csvfiles(dir='experiments/designer_3/fits/model_3/')
```

### Computational Notes

1. **Normalizing constant approximation**: Truncating at `max_count` introduces bias if true mass extends beyond. Check:
   ```python
   # Verify truncation is adequate
   lambda_max = fit.stan_variable('lambda').max()
   # Rule of thumb: max_count should be >> lambda_max
   print(f"Max lambda: {lambda_max:.1f}, Truncation: {max_count}")
   ```

2. **Expected runtime**: 10-30 minutes (vs. 1-2 min for standard models)

3. **Potential failures**:
   - Divergences if nu explores near 0
   - Slow mixing if posterior is multimodal
   - Numerical issues if normalizing constant underflows

### When to Abandon

- If fitting takes >1 hour or has excessive divergences
- If nu ≈ 1 (just use Poisson)
- If LOO-CV is similar to NegBin (complexity not justified)
- If computational cost outweighs scientific benefit

---

## Model Comparison Workflow

### Step 1: Fit All Models

```python
import os
os.makedirs('experiments/designer_3/fits', exist_ok=True)
os.makedirs('experiments/designer_3/fits/model_1', exist_ok=True)
os.makedirs('experiments/designer_3/fits/model_2', exist_ok=True)
os.makedirs('experiments/designer_3/fits/model_3', exist_ok=True)

# Fit models in priority order
models = {}
loos = {}

# Model 2 (fastest)
print("=" * 60)
print("Fitting Model 2: Student-t Regression")
print("=" * 60)
# ... fit Model 2 ...
models['Student-t'] = fit_2
loos['Student-t'] = loo_2

# Model 1 (moderate)
print("\n" + "=" * 60)
print("Fitting Model 1: Hierarchical Gamma-Poisson")
print("=" * 60)
# ... fit Model 1 ...
models['Hierarchical'] = fit_1
loos['Hierarchical'] = loo_1

# Model 3 (slowest, optional)
print("\n" + "=" * 60)
print("Fitting Model 3: COM-Poisson (may take 10-30 min)")
print("=" * 60)
try:
    # ... fit Model 3 ...
    models['COM-Poisson'] = fit_3
    loos['COM-Poisson'] = loo_3
except Exception as e:
    print(f"Model 3 failed: {e}")
    print("Continuing with Models 1-2 only")
```

### Step 2: Compare LOO-CV

```python
import pandas as pd

# Create comparison table
loo_comparison = pd.DataFrame({
    'Model': list(loos.keys()),
    'ELPD': [loo.elpd_loo for loo in loos.values()],
    'SE': [loo.se for loo in loos.values()],
})

loo_comparison = loo_comparison.sort_values('ELPD', ascending=False)
loo_comparison['Delta_ELPD'] = loo_comparison['ELPD'] - loo_comparison['ELPD'].max()

print("\nLOO-CV Comparison:")
print(loo_comparison)

# Decision rule
best_model = loo_comparison.iloc[0]['Model']
delta_elpd = abs(loo_comparison.iloc[1]['Delta_ELPD'])
se = loo_comparison.iloc[1]['SE']

if delta_elpd < 2 * se:
    print(f"\nModels are statistically tied (ΔELPD < 2×SE)")
    print("Prefer simpler model or examine substantive differences")
else:
    print(f"\nClear winner: {best_model} (ΔELPD = {delta_elpd:.2f}, SE = {se:.2f})")

# Save comparison
loo_comparison.to_csv('experiments/designer_3/model_comparison.csv', index=False)
```

### Step 3: Posterior Predictive Checks

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (name, fit) in enumerate(models.items()):
    # Get C_rep
    C_rep = fit.stan_variable('C_rep')

    # Distribution comparison
    ax = axes[0, idx]
    ax.hist(data['C'], bins=20, alpha=0.5, label='Observed', density=True)
    ax.hist(C_rep.flatten(), bins=20, alpha=0.5, label='Predicted', density=True)
    ax.set_xlabel('Count')
    ax.set_ylabel('Density')
    ax.set_title(f'{name}: Distribution')
    ax.legend()

    # Var/Mean ratio
    ax = axes[1, idx]
    var_mean_ratio = fit.stan_variable('var_mean_ratio')
    ax.hist(var_mean_ratio, bins=30, alpha=0.7)
    ax.axvline(np.var(data['C']) / np.mean(data['C']), color='red',
               linestyle='--', linewidth=2, label=f'Observed: {np.var(data["C"]) / np.mean(data["C"]):.1f}')
    ax.set_xlabel('Var/Mean Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name}: Overdispersion Recovery')
    ax.legend()

plt.tight_layout()
plt.savefig('experiments/designer_3/ppc_comparison.png', dpi=150)
plt.close()

print("\nPosterior Predictive Check saved to: experiments/designer_3/ppc_comparison.png")
```

---

## Troubleshooting Guide

### Issue 1: Divergent Transitions

**Symptom**: Model reports divergences after warmup

**Solutions**:
1. Increase `adapt_delta` to 0.98 or 0.99
2. Reparameterize (e.g., non-centered for hierarchical models)
3. Check for pathological posteriors (plotting pairs plots)

### Issue 2: Low ESS

**Symptom**: Effective sample size < 400

**Solutions**:
1. Increase `max_treedepth` to 12 or 15
2. Run longer chains (iter_sampling = 2000)
3. Check for high posterior correlation (reparameterize)

### Issue 3: Rhat > 1.01

**Symptom**: Chains haven't converged

**Solutions**:
1. Increase warmup iterations to 2000
2. Check trace plots for mixing issues
3. Try different initial values
4. Consider model misspecification

### Issue 4: COM-Poisson Too Slow

**Symptom**: Fitting takes >1 hour

**Solutions**:
1. Reduce `max_count` (check truncation is still valid)
2. Use PyMC instead (may have optimized COM-Poisson)
3. Skip Model 3 and focus on Models 1-2

---

## Expected Timeline

| Task | Time | Cumulative |
|------|------|------------|
| Compile all models | 5 min | 5 min |
| Fit Model 2 | 2 min | 7 min |
| Fit Model 1 | 5 min | 12 min |
| Fit Model 3 (optional) | 20 min | 32 min |
| Compute LOO-CV | 2 min | 34 min |
| Posterior predictive checks | 5 min | 39 min |
| Generate visualizations | 5 min | 44 min |
| Write summary report | 15 min | 59 min |

**Total**: ~1 hour (or ~40 min if skipping Model 3)

---

## Success Metrics

**Minimum Viable Success**:
- At least 2 models converge (Rhat < 1.01)
- LOO-CV comparison is interpretable (SE < |ΔELPD| or SE ≈ |ΔELPD|)
- Winner recovers Var/Mean ≈ 70 in PPC

**Full Success**:
- All 3 models converge
- Clear LOO-CV winner (ΔELPD > 2×SE)
- Winner passes all posterior predictive checks
- Results are robust to prior sensitivity

**Failure Modes** (still informative):
- All models fail to recover overdispersion → Need time-varying dispersion
- All models give similar LOO-CV → Distributional choice doesn't matter
- Computational failures → Indicates model/data pathologies

---

## Files Checklist

After implementation, verify these files exist:

- [ ] `experiments/designer_3/model_1_hierarchical.stan`
- [ ] `experiments/designer_3/model_2_studentt.stan`
- [ ] `experiments/designer_3/model_3_compoisson.stan` (optional)
- [ ] `experiments/designer_3/fits/model_1/*.csv`
- [ ] `experiments/designer_3/fits/model_2/*.csv`
- [ ] `experiments/designer_3/model_comparison.csv`
- [ ] `experiments/designer_3/ppc_comparison.png`
- [ ] `experiments/designer_3/convergence_diagnostics.txt`
- [ ] `experiments/designer_3/summary_report.md`

---

**Ready to implement**: All code snippets are copy-paste ready. Start with Model 2 (fastest) to establish baseline.
