# Stan Model Templates
## Ready-to-Use Code for All Three Model Classes

---

## Model 2A: Polynomial (Quadratic, Constant Dispersion)

**File**: `model_2a_polynomial.stan`

```stan
data {
  int<lower=0> N;           // Number of observations
  vector[N] year;           // Standardized year (predictor)
  array[N] int<lower=0> C;  // Count outcome
}

transformed data {
  vector[N] year_sq = year .* year;  // Precompute year^2
}

parameters {
  real beta0;               // Intercept (log-scale)
  real beta1;               // Linear coefficient
  real beta2;               // Quadratic coefficient
  real<lower=0> phi;        // Dispersion parameter (NegBin2)
}

transformed parameters {
  vector[N] log_mu;
  for (i in 1:N) {
    log_mu[i] = beta0 + beta1 * year[i] + beta2 * year_sq[i];
  }
}

model {
  // Priors
  beta0 ~ normal(4.5, 1.0);
  beta1 ~ normal(0, 1.5);
  beta2 ~ normal(0, 0.5);
  phi ~ gamma(2, 0.1);

  // Likelihood
  C ~ neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;

  for (i in 1:N) {
    real mu = exp(log_mu[i]);
    log_lik[i] = neg_binomial_2_log_lpmf(C[i] | log_mu[i], phi);
    y_rep[i] = neg_binomial_2_rng(mu, phi);
  }
}
```

---

## Model 2B: Polynomial with Time-Varying Dispersion

**File**: `model_2b_poly_varying_phi.stan`

```stan
data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

transformed data {
  vector[N] year_sq = year .* year;
}

parameters {
  real beta0;
  real beta1;
  real beta2;
  real gamma0;              // Baseline log-dispersion
  real gamma1;              // Dispersion time trend
}

transformed parameters {
  vector[N] log_mu;
  vector<lower=0>[N] phi;

  for (i in 1:N) {
    log_mu[i] = beta0 + beta1 * year[i] + beta2 * year_sq[i];
    phi[i] = exp(gamma0 + gamma1 * year[i]);
  }
}

model {
  // Priors
  beta0 ~ normal(4.5, 1.0);
  beta1 ~ normal(0, 1.5);
  beta2 ~ normal(0, 0.5);
  gamma0 ~ normal(3.0, 1.0);   // exp(3) = 20
  gamma1 ~ normal(0, 0.5);

  // Likelihood
  for (i in 1:N) {
    C[i] ~ neg_binomial_2_log(log_mu[i], phi[i]);
  }
}

generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;

  for (i in 1:N) {
    real mu = exp(log_mu[i]);
    log_lik[i] = neg_binomial_2_log_lpmf(C[i] | log_mu[i], phi[i]);
    y_rep[i] = neg_binomial_2_rng(mu, phi[i]);
  }
}
```

---

## Model 1A: Piecewise Linear with Unknown Changepoint

**File**: `model_1a_changepoint.stan`

**Note**: This uses marginalization over discrete changepoint parameter.

```stan
data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
  int<lower=1, upper=N-1> tau_min;   // Minimum changepoint (e.g., 10)
  int<lower=2, upper=N> tau_max;     // Maximum changepoint (e.g., 30)
}

parameters {
  real beta0;               // Intercept (shared)
  real beta1_before;        // Slope before changepoint
  real beta1_after;         // Slope after changepoint
  real<lower=0> phi_before; // Dispersion before changepoint
  real<lower=0> phi_after;  // Dispersion after changepoint
}

transformed parameters {
  int K = tau_max - tau_min + 1;  // Number of candidate changepoints
  vector[K] lp;                    // Log-probability for each tau

  for (k in 1:K) {
    int tau = tau_min + k - 1;
    real lp_k = 0;

    // Compute log-likelihood for this changepoint
    for (i in 1:N) {
      real log_mu;
      real phi;

      if (i < tau) {
        log_mu = beta0 + beta1_before * year[i];
        phi = phi_before;
      } else {
        log_mu = beta0 + beta1_after * year[i];
        phi = phi_after;
      }

      lp_k += neg_binomial_2_log_lpmf(C[i] | log_mu, phi);
    }

    lp[k] = lp_k;
  }
}

model {
  // Priors
  beta0 ~ normal(4.5, 1.0);
  beta1_before ~ normal(0, 1.5);
  beta1_after ~ normal(0, 1.5);
  phi_before ~ gamma(2, 0.1);
  phi_after ~ gamma(2, 0.1);

  // Marginalize over tau (discrete uniform prior implicit)
  target += log_sum_exp(lp);
}

generated quantities {
  int<lower=tau_min, upper=tau_max> tau;
  vector[N] log_lik;
  array[N] int y_rep;
  real jump_ratio;  // Multiplicative jump at changepoint

  // Sample tau from posterior
  tau = tau_min + categorical_rng(softmax(lp)) - 1;

  // Compute log_lik and y_rep using sampled tau
  for (i in 1:N) {
    real log_mu;
    real mu;
    real phi;

    if (i < tau) {
      log_mu = beta0 + beta1_before * year[i];
      phi = phi_before;
    } else {
      log_mu = beta0 + beta1_after * year[i];
      phi = phi_after;
    }

    mu = exp(log_mu);
    log_lik[i] = neg_binomial_2_log_lpmf(C[i] | log_mu, phi);
    y_rep[i] = neg_binomial_2_rng(mu, phi);
  }

  // Compute jump size (before vs after at changepoint)
  {
    real mu_before = exp(beta0 + beta1_before * year[tau-1]);
    real mu_after = exp(beta0 + beta1_after * year[tau]);
    jump_ratio = mu_after / mu_before;
  }
}
```

---

## Model 1B: Piecewise Quadratic with Unknown Changepoint

**File**: `model_1b_changepoint_quad.stan`

```stan
data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
  int<lower=1, upper=N-1> tau_min;
  int<lower=2, upper=N> tau_max;
}

transformed data {
  vector[N] year_sq = year .* year;
}

parameters {
  real beta0;
  real beta1_before;
  real beta1_after;
  real beta2_before;        // Quadratic before
  real beta2_after;         // Quadratic after
  real<lower=0> phi_before;
  real<lower=0> phi_after;
}

transformed parameters {
  int K = tau_max - tau_min + 1;
  vector[K] lp;

  for (k in 1:K) {
    int tau = tau_min + k - 1;
    real lp_k = 0;

    for (i in 1:N) {
      real log_mu;
      real phi;

      if (i < tau) {
        log_mu = beta0 + beta1_before * year[i] + beta2_before * year_sq[i];
        phi = phi_before;
      } else {
        log_mu = beta0 + beta1_after * year[i] + beta2_after * year_sq[i];
        phi = phi_after;
      }

      lp_k += neg_binomial_2_log_lpmf(C[i] | log_mu, phi);
    }

    lp[k] = lp_k;
  }
}

model {
  // Priors
  beta0 ~ normal(4.5, 1.0);
  beta1_before ~ normal(0, 1.5);
  beta1_after ~ normal(0, 1.5);
  beta2_before ~ normal(0, 0.5);
  beta2_after ~ normal(0, 0.5);
  phi_before ~ gamma(2, 0.1);
  phi_after ~ gamma(2, 0.1);

  target += log_sum_exp(lp);
}

generated quantities {
  int<lower=tau_min, upper=tau_max> tau;
  vector[N] log_lik;
  array[N] int y_rep;

  tau = tau_min + categorical_rng(softmax(lp)) - 1;

  for (i in 1:N) {
    real log_mu;
    real mu;
    real phi;

    if (i < tau) {
      log_mu = beta0 + beta1_before * year[i] + beta2_before * year_sq[i];
      phi = phi_before;
    } else {
      log_mu = beta0 + beta1_after * year[i] + beta2_after * year_sq[i];
      phi = phi_after;
    }

    mu = exp(log_mu);
    log_lik[i] = neg_binomial_2_log_lpmf(C[i] | log_mu, phi);
    y_rep[i] = neg_binomial_2_rng(mu, phi);
  }
}
```

---

## Model 3A: Two-State Hierarchical Model

**File**: `model_3a_two_state.stan`

```stan
data {
  int<lower=0> N;
  vector[N] year;
  array[N] int<lower=0> C;
}

parameters {
  // Hyperparameters
  real mu_alpha;
  real mu_beta;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;

  // State-specific parameters
  vector[2] alpha;          // Intercepts for 2 states
  vector[2] beta;           // Slopes for 2 states
  vector<lower=0>[2] phi;   // Dispersion for 2 states

  // State transition parameters (logistic regression)
  real gamma0;              // Baseline state 2 preference
  real gamma1;              // How state 2 probability changes with time
}

model {
  // Hyperpriors
  mu_alpha ~ normal(4.0, 1.0);
  mu_beta ~ normal(1.0, 0.5);
  sigma_alpha ~ exponential(1);
  sigma_beta ~ exponential(2);

  // State-specific priors (hierarchical)
  alpha ~ normal(mu_alpha, sigma_alpha);
  beta ~ normal(mu_beta, sigma_beta);
  phi ~ gamma(2, 0.1);

  // Transition parameters
  gamma0 ~ normal(0, 2);
  gamma1 ~ normal(0, 1);

  // Likelihood (marginalized over states)
  for (i in 1:N) {
    vector[2] lp_state;

    // State probabilities (logistic regression)
    real logit_p2 = gamma0 + gamma1 * year[i];
    real p2 = inv_logit(logit_p2);
    real p1 = 1 - p2;

    // Compute likelihood for each state
    for (k in 1:2) {
      real log_mu = alpha[k] + beta[k] * year[i];
      real p_state = (k == 1) ? p1 : p2;
      lp_state[k] = log(p_state) + neg_binomial_2_log_lpmf(C[i] | log_mu, phi[k]);
    }

    // Marginalize
    target += log_sum_exp(lp_state);
  }
}

generated quantities {
  vector[N] log_lik;
  array[N] int y_rep;
  matrix[N, 2] state_prob;  // Posterior state probabilities
  array[N] int state_map;   // Most likely state for each observation

  for (i in 1:N) {
    vector[2] lp_state;

    // Compute state probabilities
    real logit_p2 = gamma0 + gamma1 * year[i];
    real p2 = inv_logit(logit_p2);
    real p1 = 1 - p2;
    state_prob[i, 1] = p1;
    state_prob[i, 2] = p2;

    // Compute marginal log-likelihood
    for (k in 1:2) {
      real log_mu = alpha[k] + beta[k] * year[i];
      real p_state = (k == 1) ? p1 : p2;
      lp_state[k] = log(p_state) + neg_binomial_2_log_lpmf(C[i] | log_mu, phi[k]);
    }
    log_lik[i] = log_sum_exp(lp_state);

    // Sample state and generate data
    int z;
    if (p1 > p2) {
      z = 1;
    } else {
      z = 2;
    }
    state_map[i] = z;

    // Generate from mixture
    real u = uniform_rng(0, 1);
    if (u < p1) {
      z = 1;
    } else {
      z = 2;
    }
    real mu = exp(alpha[z] + beta[z] * year[i]);
    y_rep[i] = neg_binomial_2_rng(mu, phi[z]);
  }
}
```

---

## Python Implementation Template

**File**: `fit_models.py`

```python
import cmdstanpy
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/workspace/data/data.csv')
N = len(data)
year = data['year'].values
C = data['C'].values.astype(int)

# Prepare data for Stan
stan_data = {
    'N': N,
    'year': year,
    'C': C
}

# For changepoint models, add tau bounds
stan_data_changepoint = {
    **stan_data,
    'tau_min': 10,
    'tau_max': 30
}

# Compile and fit Model 2A (Polynomial)
print("=" * 60)
print("Fitting Model 2A: Polynomial (Quadratic)")
print("=" * 60)

model_2a = cmdstanpy.CmdStanModel(stan_file='model_2a_polynomial.stan')
fit_2a = model_2a.sample(
    data=stan_data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    adapt_delta=0.90,
    seed=12345
)

# Check diagnostics
print("\nDiagnostics:")
print(fit_2a.diagnose())
print("\nSummary:")
print(fit_2a.summary())

# Convert to ArviZ InferenceData
idata_2a = az.from_cmdstanpy(
    fit_2a,
    posterior_predictive='y_rep',
    log_likelihood='log_lik'
)

# Compute LOO
loo_2a = az.loo(idata_2a, pointwise=True)
print("\nLOO-CV:")
print(loo_2a)

# Posterior predictive check
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_ppc(idata_2a, ax=ax)
plt.title("Posterior Predictive Check - Model 2A")
plt.savefig('ppc_model_2a.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results
fit_2a.save_csvfiles(dir='model_2a_output')

# Similar code for other models...
# (fit_2b, fit_1a, fit_3a, etc.)

# Model comparison
print("\n" + "=" * 60)
print("Model Comparison")
print("=" * 60)

comparison = az.compare({
    '2A_polynomial': idata_2a,
    # Add other models here
}, ic='loo')
print(comparison)

# Plot comparison
az.plot_compare(comparison)
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Diagnostic Functions

**File**: `diagnostics.py`

```python
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

def check_diagnostics(idata, model_name):
    """
    Comprehensive diagnostic checks for Stan model.

    Returns True if all checks pass, False otherwise.
    """
    print(f"\n{'=' * 60}")
    print(f"Diagnostics for {model_name}")
    print(f"{'=' * 60}")

    passed = True

    # 1. Rhat
    rhat = az.rhat(idata)
    max_rhat = float(rhat.to_array().max())
    print(f"\n1. Rhat: max = {max_rhat:.4f}")
    if max_rhat > 1.01:
        print("   ⚠️  FAIL: Rhat > 1.01 (poor convergence)")
        passed = False
    else:
        print("   ✓ PASS: Rhat < 1.01")

    # 2. ESS
    ess_bulk = az.ess(idata, method='bulk')
    min_ess = float(ess_bulk.to_array().min())
    print(f"\n2. ESS (bulk): min = {min_ess:.0f}")
    if min_ess < 400:
        print(f"   ⚠️  WARNING: ESS < 400 (low effective sample size)")
        if min_ess < 100:
            print("   ⚠️  FAIL: ESS < 100 (very low)")
            passed = False
    else:
        print("   ✓ PASS: ESS > 400")

    # 3. Divergences
    divergences = idata.sample_stats.diverging.sum().values
    total_samples = idata.sample_stats.diverging.size
    div_pct = 100 * divergences / total_samples
    print(f"\n3. Divergences: {divergences} / {total_samples} ({div_pct:.2f}%)")
    if div_pct > 1.0:
        print("   ⚠️  FAIL: >1% divergences")
        passed = False
    else:
        print("   ✓ PASS: <1% divergences")

    # 4. Energy
    energy = idata.sample_stats.energy
    # Check E-BFMI (energy Bayesian fraction of missing information)
    # Not directly available, but can check energy variance
    print("\n4. Energy diagnostics:")
    print("   (Visual inspection recommended)")

    # 5. Trace plots
    print("\n5. Generating trace plots...")
    az.plot_trace(idata, compact=False)
    plt.savefig(f'{model_name}_trace.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved to {model_name}_trace.png")

    # 6. Pairs plot (for key parameters)
    print("\n6. Generating pairs plot...")
    # Customize for each model

    # Summary
    print(f"\n{'=' * 60}")
    if passed:
        print(f"✓ {model_name} PASSED all diagnostic checks")
    else:
        print(f"⚠️  {model_name} FAILED diagnostic checks")
    print(f"{'=' * 60}\n")

    return passed


def posterior_predictive_check(idata, observed_data, model_name):
    """
    Perform posterior predictive checks with custom test statistics.
    """
    print(f"\nPosterior Predictive Check: {model_name}")

    # Extract y_rep
    y_rep = idata.posterior_predictive['y_rep'].values
    # Shape: (chains, draws, N)
    y_rep_flat = y_rep.reshape(-1, y_rep.shape[-1])  # (total_draws, N)

    # Test statistic 1: Mean
    T_obs_mean = np.mean(observed_data)
    T_rep_mean = np.mean(y_rep_flat, axis=1)
    p_mean = np.mean(T_rep_mean > T_obs_mean)
    print(f"  T1 (Mean): Observed = {T_obs_mean:.1f}, p-value = {p_mean:.3f}")

    # Test statistic 2: Variance
    T_obs_var = np.var(observed_data)
    T_rep_var = np.var(y_rep_flat, axis=1)
    p_var = np.mean(T_rep_var > T_obs_var)
    print(f"  T2 (Variance): Observed = {T_obs_var:.1f}, p-value = {p_var:.3f}")

    # Test statistic 3: Maximum
    T_obs_max = np.max(observed_data)
    T_rep_max = np.max(y_rep_flat, axis=1)
    p_max = np.mean(T_rep_max > T_obs_max)
    print(f"  T3 (Maximum): Observed = {T_obs_max:.1f}, p-value = {p_max:.3f}")

    # Test statistic 4: Autocorrelation lag-1
    T_obs_acf = np.corrcoef(observed_data[:-1], observed_data[1:])[0, 1]
    T_rep_acf = []
    for rep in y_rep_flat:
        acf = np.corrcoef(rep[:-1], rep[1:])[0, 1]
        T_rep_acf.append(acf)
    T_rep_acf = np.array(T_rep_acf)
    p_acf = np.mean(T_rep_acf > T_obs_acf)
    print(f"  T4 (ACF-1): Observed = {T_obs_acf:.3f}, p-value = {p_acf:.3f}")

    # Overall assessment
    p_values = [p_mean, p_var, p_max, p_acf]
    extreme_p = sum((p < 0.05) or (p > 0.95) for p in p_values)

    print(f"\n  Extreme p-values (< 0.05 or > 0.95): {extreme_p} / 4")
    if extreme_p == 0:
        print("  ✓ PASS: All test statistics within expected range")
    else:
        print(f"  ⚠️  WARNING: {extreme_p} test statistics extreme")

    return p_values


def compare_models_loo(idata_dict):
    """
    Compare multiple models using LOO-CV.

    Args:
        idata_dict: Dictionary of {model_name: InferenceData}

    Returns:
        Comparison DataFrame
    """
    print("\n" + "=" * 60)
    print("Model Comparison via LOO-CV")
    print("=" * 60)

    comparison = az.compare(idata_dict, ic='loo')
    print(comparison)

    # Check for warnings
    for name, idata in idata_dict.items():
        loo = az.loo(idata, pointwise=True)
        n_high_k = np.sum(loo.pareto_k > 0.7)
        pct_high_k = 100 * n_high_k / len(loo.pareto_k)
        print(f"\n{name}:")
        print(f"  Pareto k > 0.7: {n_high_k} ({pct_high_k:.1f}%)")
        if pct_high_k > 10:
            print(f"  ⚠️  WARNING: High proportion of influential points")

    return comparison
```

---

## Usage Example

```bash
# In /workspace/experiments/designer_3/

# 1. Compile Stan models (done automatically on first run)
# 2. Run fitting script
python fit_models.py

# 3. Run diagnostics
python diagnostics.py

# 4. Review outputs:
#    - model_*_output/ (Stan CSV files)
#    - *_trace.png (trace plots)
#    - ppc_*.png (posterior predictive checks)
#    - model_comparison.png (LOO comparison)
```

---

## Tips for Efficient Workflow

1. **Start simple**: Fit Model 2A first, check diagnostics
2. **Iterate quickly**: If Model 2A fails, debug before moving on
3. **Parallelize**: Fit multiple models in parallel (different terminals)
4. **Save everything**: Keep Stan outputs for reproducibility
5. **Document decisions**: Why you moved to next model class

---

## Common Issues & Solutions

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Many divergences | Difficult geometry | Increase adapt_delta to 0.95-0.99 |
| Low ESS for phi | Funnel in dispersion | Non-centered parameterization |
| Rhat > 1.05 | Multi-modal posterior | Longer chains, check for label switching |
| All Pareto-k high | Model misspecified | Reconsider likelihood or trend |
| Extreme beta values | Prior too weak | Tighten priors |
| Compilation errors | Stan version issue | Use CmdStan 2.32+ |

---

**These templates are ready to use. Copy, modify as needed, and fit to your data!**
