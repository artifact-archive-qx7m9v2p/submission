# Stan Implementation Templates: Designer 1 Models

## Model 1: Complete Pooling (Common Effect)

### Stan Code
```stan
data {
  int<lower=1> J;              // Number of studies
  vector[J] y;                 // Observed effects
  vector<lower=0>[J] sigma;    // Known standard errors
}

parameters {
  real mu;                     // Pooled effect
}

model {
  // Prior
  mu ~ normal(0, 50);          // Weakly informative

  // Likelihood
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[J] y_rep;             // Posterior predictive
  vector[J] log_lik;           // For LOO-CV
  vector[J] resid;             // Residuals
  vector[J] resid_std;         // Standardized residuals

  for (j in 1:J) {
    y_rep[j] = normal_rng(mu, sigma[j]);
    log_lik[j] = normal_lpdf(y[j] | mu, sigma[j]);
    resid[j] = y[j] - mu;
    resid_std[j] = (y[j] - mu) / sigma[j];
  }
}
```

### Python Interface (PyStan/CmdStanPy)
```python
import cmdstanpy
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('/workspace/data/data.csv')
stan_data = {
    'J': len(data),
    'y': data['y'].values,
    'sigma': data['sigma'].values
}

# Compile and fit
model = cmdstanpy.CmdStanModel(stan_file='model1_complete_pooling.stan')
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    seed=12345,
    show_progress=True
)

# Diagnostics
print(fit.diagnose())
print(fit.summary())

# Extract posterior
mu_post = fit.stan_variable('mu')
print(f"Posterior mean mu: {mu_post.mean():.2f}")
print(f"95% CI: [{np.percentile(mu_post, 2.5):.2f}, {np.percentile(mu_post, 97.5):.2f}]")
```

---

## Model 2: Partial Pooling (Hierarchical Random Effects)

### Stan Code (Non-Centered Parameterization - RECOMMENDED)
```stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}

parameters {
  real mu;                     // Mean effect
  real<lower=0> tau;           // Between-study SD
  vector[J] theta_raw;         // Non-centered study effects
}

transformed parameters {
  vector[J] theta;             // Actual study effects
  theta = mu + tau * theta_raw;
}

model {
  // Priors
  mu ~ normal(0, 50);
  tau ~ normal(0, 10);         // Half-normal via constraint
  theta_raw ~ normal(0, 1);    // Standard normal

  // Likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  vector[J] y_rep;
  vector[J] log_lik;
  real tau_sq = tau^2;
  vector[J] shrinkage;         // Shrinkage factors
  real I_squared;              // I² statistic

  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma[j]);
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
    shrinkage[j] = tau^2 / (tau^2 + sigma[j]^2);
  }

  // I² = tau² / (tau² + average(sigma²))
  I_squared = tau^2 / (tau^2 + mean(sigma .* sigma));
}
```

### Stan Code (Centered Parameterization - USE ONLY IF NON-CENTERED FAILS)
```stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}

parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta;             // Directly parameterized
}

model {
  mu ~ normal(0, 50);
  tau ~ normal(0, 10);
  theta ~ normal(mu, tau);     // Centered parameterization
  y ~ normal(theta, sigma);
}

generated quantities {
  // Same as non-centered version
  vector[J] y_rep;
  vector[J] log_lik;
  real tau_sq = tau^2;
  vector[J] shrinkage;

  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma[j]);
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
    shrinkage[j] = tau^2 / (tau^2 + sigma[j]^2);
  }
}
```

### Python Interface with Diagnostics
```python
import cmdstanpy
import numpy as np
import pandas as pd
import arviz as az

# Load data
data = pd.read_csv('/workspace/data/data.csv')
stan_data = {
    'J': len(data),
    'y': data['y'].values,
    'sigma': data['sigma'].values
}

# Compile and fit
model = cmdstanpy.CmdStanModel(stan_file='model2_partial_pooling_nc.stan')
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    adapt_delta=0.95,  # Increase if divergences occur
    seed=12345,
    show_progress=True
)

# Comprehensive diagnostics
print("=== Convergence Diagnostics ===")
print(fit.diagnose())

summary = fit.summary()
print("\n=== Parameter Summary ===")
print(summary[['Mean', 'StdDev', 'R_hat', 'ESS_bulk', 'ESS_tail']])

# Check for divergences
divergences = fit.method_variables()['divergent__'].sum()
print(f"\nDivergent transitions: {divergences}")
if divergences > 0:
    print("WARNING: Divergences detected. Try increasing adapt_delta.")

# Extract posteriors
mu_post = fit.stan_variable('mu')
tau_post = fit.stan_variable('tau')
theta_post = fit.stan_variable('theta')
shrinkage_post = fit.stan_variable('shrinkage')

print(f"\n=== Posterior Estimates ===")
print(f"mu: {mu_post.mean():.2f} [{np.percentile(mu_post, 2.5):.2f}, {np.percentile(mu_post, 97.5):.2f}]")
print(f"tau: {tau_post.mean():.2f} [{np.percentile(tau_post, 2.5):.2f}, {np.percentile(tau_post, 97.5):.2f}]")
print(f"Mean shrinkage: {shrinkage_post.mean():.3f}")

# Convert to ArviZ for advanced diagnostics
idata = az.from_cmdstanpy(fit)

# LOO cross-validation
loo = az.loo(idata, pointwise=True)
print(f"\n=== LOO-CV ===")
print(f"LOO-ELPD: {loo.loo:.2f} ± {loo.loo_se:.2f}")
print(f"Pareto k diagnostics:")
print(loo.pareto_k)
if (loo.pareto_k > 0.7).any():
    print("WARNING: Some studies have high Pareto k (potential outliers/influential points)")
```

---

## Model 3: Skeptical Prior (Hierarchical with Tighter Priors)

### Stan Code (Non-Centered)
```stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}

parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_raw;
}

transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}

model {
  // Skeptical priors (tighter than Model 2)
  mu ~ normal(0, 15);          // 95% CI ≈ [-30, 30]
  tau ~ normal(0, 5);          // Expect low heterogeneity
  theta_raw ~ normal(0, 1);

  y ~ normal(theta, sigma);
}

generated quantities {
  vector[J] y_rep;
  vector[J] log_lik;
  real tau_sq = tau^2;
  vector[J] shrinkage;

  // Sample from priors for comparison
  real prior_mu_sample = normal_rng(0, 15);
  real prior_tau_sample = fabs(normal_rng(0, 5));

  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma[j]);
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
    shrinkage[j] = tau^2 / (tau^2 + sigma[j]^2);
  }
}
```

### Python Interface with Prior-Posterior Comparison
```python
import cmdstanpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fit model (same as Model 2)
data = pd.read_csv('/workspace/data/data.csv')
stan_data = {
    'J': len(data),
    'y': data['y'].values,
    'sigma': data['sigma'].values
}

model = cmdstanpy.CmdStanModel(stan_file='model3_skeptical_prior.stan')
fit = model.sample(data=stan_data, chains=4, iter_sampling=2000, adapt_delta=0.95)

# Extract posteriors and prior samples
mu_post = fit.stan_variable('mu')
tau_post = fit.stan_variable('tau')
prior_mu = fit.stan_variable('prior_mu_sample')
prior_tau = fit.stan_variable('prior_tau_sample')

# Prior-posterior comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# mu comparison
axes[0].hist(prior_mu, bins=50, alpha=0.5, label='Prior', density=True)
axes[0].hist(mu_post, bins=50, alpha=0.5, label='Posterior', density=True)
axes[0].axvline(0, color='red', linestyle='--', label='Prior mean')
axes[0].axvline(mu_post.mean(), color='blue', linestyle='--', label='Posterior mean')
axes[0].set_xlabel('mu')
axes[0].set_title('Prior vs Posterior: mu')
axes[0].legend()

# tau comparison
axes[1].hist(prior_tau, bins=50, alpha=0.5, label='Prior', density=True)
axes[1].hist(tau_post, bins=50, alpha=0.5, label='Posterior', density=True)
axes[1].axvline(tau_post.mean(), color='blue', linestyle='--', label='Posterior mean')
axes[1].set_xlabel('tau')
axes[1].set_title('Prior vs Posterior: tau')
axes[1].legend()

plt.tight_layout()
plt.savefig('prior_posterior_comparison.png', dpi=150)

# Compute prior-posterior overlap
from scipy.stats import gaussian_kde
mu_prior_kde = gaussian_kde(prior_mu)
mu_post_kde = gaussian_kde(mu_post)
x = np.linspace(-50, 50, 1000)
overlap = np.minimum(mu_prior_kde(x), mu_post_kde(x)).sum() * (x[1] - x[0])
print(f"Prior-posterior overlap for mu: {overlap:.3f}")
if overlap > 0.5:
    print("WARNING: Prior may be dominating posterior")
```

---

## Complete Analysis Pipeline

### Step 1: Fit All Models
```python
import cmdstanpy
import pandas as pd
import numpy as np
import arviz as az

# Load data
data = pd.read_csv('/workspace/data/data.csv')
stan_data = {
    'J': len(data),
    'y': data['y'].values,
    'sigma': data['sigma'].values
}

# Fit Model 1
model1 = cmdstanpy.CmdStanModel(stan_file='model1_complete_pooling.stan')
fit1 = model1.sample(data=stan_data, chains=4, iter_sampling=2000)

# Fit Model 2
model2 = cmdstanpy.CmdStanModel(stan_file='model2_partial_pooling_nc.stan')
fit2 = model2.sample(data=stan_data, chains=4, iter_sampling=2000, adapt_delta=0.95)

# Fit Model 3
model3 = cmdstanpy.CmdStanModel(stan_file='model3_skeptical_prior.stan')
fit3 = model3.sample(data=stan_data, chains=4, iter_sampling=2000, adapt_delta=0.95)

# Convert to ArviZ
idata1 = az.from_cmdstanpy(fit1)
idata2 = az.from_cmdstanpy(fit2)
idata3 = az.from_cmdstanpy(fit3)
```

### Step 2: Compare Models with LOO
```python
# Compute LOO for each model
loo1 = az.loo(idata1)
loo2 = az.loo(idata2)
loo3 = az.loo(idata3)

# Compare
loo_compare = az.compare({'Model 1 (Complete)': idata1,
                          'Model 2 (Partial)': idata2,
                          'Model 3 (Skeptical)': idata3})
print(loo_compare)

# Interpretation
best_model = loo_compare.index[0]
delta_loo = loo_compare.loc[loo_compare.index[1], 'elpd_diff']
print(f"\nBest model: {best_model}")
if abs(delta_loo) < 4:
    print("Models are effectively equivalent (ΔLOO < 4)")
else:
    print(f"Meaningful difference: ΔLOO = {delta_loo:.1f}")
```

### Step 3: Posterior Predictive Checks
```python
# For Model 2 (can repeat for others)
y_obs = stan_data['y']
y_rep = fit2.stan_variable('y_rep')  # Shape: (n_samples, J)

# Test statistics
def test_stats(y):
    return {
        'mean': np.mean(y),
        'sd': np.std(y),
        'min': np.min(y),
        'max': np.max(y),
        'range': np.max(y) - np.min(y)
    }

# Observed
obs_stats = test_stats(y_obs)

# Replicated
rep_stats = {k: [] for k in obs_stats.keys()}
for i in range(y_rep.shape[0]):
    stats = test_stats(y_rep[i, :])
    for k, v in stats.items():
        rep_stats[k].append(v)

# Bayesian p-values
for stat_name, obs_val in obs_stats.items():
    rep_vals = np.array(rep_stats[stat_name])
    p_val = np.mean(rep_vals >= obs_val)
    print(f"{stat_name}: observed = {obs_val:.2f}, p = {p_val:.3f}")
    if p_val < 0.05 or p_val > 0.95:
        print(f"  WARNING: Extreme p-value for {stat_name}")
```

### Step 4: Influence Analysis (Study 4 Removal)
```python
# Refit without Study 4
study_4_idx = 3  # 0-indexed
mask = np.ones(len(data), dtype=bool)
mask[study_4_idx] = False

stan_data_no4 = {
    'J': len(data) - 1,
    'y': data['y'].values[mask],
    'sigma': data['sigma'].values[mask]
}

# Refit Model 2 without Study 4
fit2_no4 = model2.sample(data=stan_data_no4, chains=4, iter_sampling=2000, adapt_delta=0.95)

# Compare
mu_full = fit2.stan_variable('mu')
mu_no4 = fit2_no4.stan_variable('mu')

print(f"Full data: mu = {mu_full.mean():.2f} [{np.percentile(mu_full, 2.5):.2f}, {np.percentile(mu_full, 97.5):.2f}]")
print(f"Without Study 4: mu = {mu_no4.mean():.2f} [{np.percentile(mu_no4, 2.5):.2f}, {np.percentile(mu_no4, 97.5):.2f}]")
print(f"Change: {mu_no4.mean() - mu_full.mean():.2f} ({100*(mu_no4.mean() - mu_full.mean())/mu_full.mean():.1f}%)")
```

### Step 5: Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Posterior distributions
axes[0, 0].hist(fit1.stan_variable('mu'), bins=50, alpha=0.5, label='Model 1')
axes[0, 0].hist(fit2.stan_variable('mu'), bins=50, alpha=0.5, label='Model 2')
axes[0, 0].hist(fit3.stan_variable('mu'), bins=50, alpha=0.5, label='Model 3')
axes[0, 0].set_xlabel('mu')
axes[0, 0].set_title('Posterior Distributions: mu')
axes[0, 0].legend()

# Tau posterior (Model 2 and 3)
axes[0, 1].hist(fit2.stan_variable('tau'), bins=50, alpha=0.5, label='Model 2')
axes[0, 1].hist(fit3.stan_variable('tau'), bins=50, alpha=0.5, label='Model 3')
axes[0, 1].axvline(2.02, color='red', linestyle='--', label='DL estimate')
axes[0, 1].set_xlabel('tau')
axes[0, 1].set_title('Posterior Distributions: tau')
axes[0, 1].legend()

# Shrinkage (Model 2)
shrinkage = fit2.stan_variable('shrinkage').mean(axis=0)
axes[1, 0].bar(range(1, len(shrinkage)+1), shrinkage)
axes[1, 0].set_xlabel('Study')
axes[1, 0].set_ylabel('Shrinkage Factor')
axes[1, 0].set_title('Mean Shrinkage by Study (Model 2)')
axes[1, 0].axhline(0.95, color='red', linestyle='--', label='95% threshold')
axes[1, 0].legend()

# Forest plot
theta_mean = fit2.stan_variable('theta').mean(axis=0)
theta_lower = np.percentile(fit2.stan_variable('theta'), 2.5, axis=0)
theta_upper = np.percentile(fit2.stan_variable('theta'), 97.5, axis=0)
mu_mean = fit2.stan_variable('mu').mean()

axes[1, 1].errorbar(range(1, len(theta_mean)+1), theta_mean,
                    yerr=[theta_mean - theta_lower, theta_upper - theta_mean],
                    fmt='o', label='Posterior theta')
axes[1, 1].scatter(range(1, len(data)+1), data['y'], marker='x', s=100,
                   color='red', label='Observed y')
axes[1, 1].axhline(mu_mean, color='blue', linestyle='--', label='Posterior mu')
axes[1, 1].set_xlabel('Study')
axes[1, 1].set_ylabel('Effect')
axes[1, 1].set_title('Observed vs Posterior Effects (Model 2)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('model_comparison_all.png', dpi=150)
```

---

## Diagnostic Checklist

### For Each Model:

1. **Convergence**
   - [ ] Rhat < 1.01 for all parameters
   - [ ] ESS_bulk > 400 for key parameters (mu, tau)
   - [ ] ESS_tail > 400 for key parameters
   - [ ] No divergent transitions (or <1% if unavoidable)

2. **Computational**
   - [ ] Trace plots show good mixing
   - [ ] No excessive autocorrelation
   - [ ] Pairs plots show no concerning correlations
   - [ ] Funnel diagnostic OK (for hierarchical models)

3. **Model Fit**
   - [ ] Posterior predictive checks pass (p-values in [0.05, 0.95])
   - [ ] Residuals randomly scattered
   - [ ] No systematic patterns in residuals
   - [ ] All Pareto-k < 0.7 in LOO

4. **Scientific**
   - [ ] Parameter estimates plausible
   - [ ] Credible intervals cover scientifically reasonable values
   - [ ] Results consistent with EDA (or explanation for differences)
   - [ ] Robust to removing influential studies

---

## Troubleshooting Guide

### Problem: Divergent Transitions
**Cause:** Difficult posterior geometry (funnel, multimodality)
**Solutions:**
1. Increase adapt_delta to 0.99
2. Use non-centered parameterization
3. Reparameterize tau (e.g., log-scale)
4. Increase warmup iterations

### Problem: Low ESS for tau
**Cause:** Posterior difficult to sample (common with small J)
**Solutions:**
1. This is often OK if ESS > 100
2. Increase sampling iterations
3. Check if tau posterior makes sense despite low ESS
4. Non-centered parameterization may help

### Problem: Rhat > 1.01
**Cause:** Chains haven't converged
**Solutions:**
1. Increase warmup iterations
2. Check trace plots for stuck chains
3. Try different initial values
4. May indicate model misspecification

### Problem: Pareto-k > 0.7
**Cause:** Influential observations or outliers
**Solutions:**
1. Investigate which studies are flagged
2. Consider robust likelihoods (Student-t)
3. Conduct sensitivity analysis removing those studies
4. May need different model class

### Problem: Posterior Predictive Checks Fail
**Cause:** Model doesn't capture data features
**Solutions:**
1. Check which test statistics fail
2. Consider more flexible models (mixture, robust)
3. May need covariates (meta-regression)
4. Likelihood may be misspecified

---

**Implementation Note:** All code assumes CmdStanPy interface. For PyStan3 or PyMC, syntax will differ but logic is identical.

**File Location:** `/workspace/experiments/designer_1/stan_templates.md`
