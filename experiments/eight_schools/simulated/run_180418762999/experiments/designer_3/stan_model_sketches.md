# Stan Model Code Sketches
## Designer 3: Adversarial Models

These are implementation-ready Stan code sketches for the three adversarial models.

---

# Model 1: Measurement Error Inflation Factor

```stan
// Model 1: Test if reported measurement errors are systematically wrong
// Key innovation: Estimate inflation factor lambda
// If lambda ≈ 1.0, measurement errors are accurate
// If lambda > 1.5, errors are underestimated (hides true heterogeneity)

data {
  int<lower=1> N;              // Number of observations (8)
  vector[N] y;                 // Observed values
  vector<lower=0>[N] sigma;    // Reported measurement errors
}

parameters {
  real mu;                     // Population mean
  real<lower=0> tau;           // Between-group SD
  real<lower=0.5, upper=3.0> lambda;  // Error inflation factor
  vector[N] theta_raw;         // Non-centered group means
}

transformed parameters {
  vector[N] theta;
  vector<lower=0>[N] sigma_effective;

  // Non-centered parameterization for numerical stability
  theta = mu + tau * theta_raw;

  // Effective measurement error (inflated)
  sigma_effective = sigma * lambda;
}

model {
  // Priors
  mu ~ normal(0, 30);          // Vague prior on mean
  tau ~ cauchy(0, 10);         // Allow substantial variation
  lambda ~ uniform(0.5, 3.0);  // TEST if errors are wrong
  theta_raw ~ std_normal();    // Standard normal for non-centered

  // Likelihood with inflated errors
  y ~ normal(theta, sigma_effective);
}

generated quantities {
  // Posterior predictive checks
  vector[N] y_rep;
  vector[N] log_lik;
  real lambda_is_one;  // Indicator: is lambda ≈ 1?

  for (n in 1:N) {
    y_rep[n] = normal_rng(theta[n], sigma_effective[n]);
    log_lik[n] = normal_lpdf(y[n] | theta[n], sigma_effective[n]);
  }

  // Diagnostic: What proportion of posterior believes lambda ≈ 1?
  lambda_is_one = (lambda > 0.9 && lambda < 1.1) ? 1.0 : 0.0;
}
```

**Key diagnostics to check:**
- `lambda` posterior: Is 95% CrI within [0.9, 1.1]?
- Correlation between `lambda` and `tau`: Is it >0.8? (identifiability issue)
- `y_rep` vs `y`: Does model reproduce observed variance?

---

# Model 2: Latent Mixture (K=2 clusters)

```stan
// Model 2: Test for hidden subgroup structure
// Key innovation: Allow 2 latent clusters with different means/variances
// If clusters collapse (mu[1] ≈ mu[2]), no subgroup structure exists

data {
  int<lower=1> N;              // Number of observations (8)
  vector[N] y;                 // Observed values
  vector<lower=0>[N] sigma;    // Known measurement errors
  int<lower=2> K;              // Number of clusters (2)
}

parameters {
  ordered[K] mu;               // Cluster means (ordered to prevent label switching)
  vector<lower=0>[K] tau;      // Cluster-specific SDs
  simplex[K] pi;               // Mixing proportions
  vector[N] theta;             // Group means
}

model {
  // Priors
  mu[1] ~ normal(0, 10);       // Lower cluster
  mu[2] ~ normal(20, 10);      // Upper cluster
  tau ~ cauchy(0, 5);
  pi ~ dirichlet(rep_vector(1.0, K));  // Uniform mixing

  // Mixture likelihood for group means
  for (n in 1:N) {
    vector[K] log_pi_theta;
    for (k in 1:K) {
      log_pi_theta[k] = log(pi[k]) + normal_lpdf(theta[n] | mu[k], tau[k]);
    }
    target += log_sum_exp(log_pi_theta);
  }

  // Observation likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  // Cluster assignments (posterior mode)
  int<lower=1, upper=K> z[N];
  vector[N] y_rep;
  vector[N] log_lik;
  real cluster_separation;  // Diagnostic: how separated are clusters?

  // Assign each observation to most likely cluster
  for (n in 1:N) {
    vector[K] log_prob;
    for (k in 1:K) {
      log_prob[k] = log(pi[k]) + normal_lpdf(theta[n] | mu[k], tau[k]);
    }
    z[n] = categorical_rng(softmax(log_prob));

    // Posterior predictive
    y_rep[n] = normal_rng(theta[n], sigma[n]);
    log_lik[n] = normal_lpdf(y[n] | theta[n], sigma[n]);
  }

  // Measure cluster separation (0 = collapsed, large = separated)
  cluster_separation = abs(mu[2] - mu[1]) / sqrt(tau[1]^2 + tau[2]^2);
}
```

**Alternative: Finite Mixture with Unknown Assignment**

```stan
// Alternative implementation using mixture directly

transformed parameters {
  vector[N] theta;

  for (n in 1:N) {
    // Expected theta given mixing proportions
    theta[n] = pi[1] * mu[1] + pi[2] * mu[2];
  }
}

model {
  // Marginalize over cluster assignments
  for (n in 1:N) {
    vector[K] lps;
    for (k in 1:K) {
      lps[k] = log(pi[k]) +
               normal_lpdf(y[n] | mu[k], sqrt(tau[k]^2 + sigma[n]^2));
    }
    target += log_sum_exp(lps);
  }
}
```

**Key diagnostics:**
- `cluster_separation`: Is it >2? (clusters are distinct)
- `z` assignments: Are they stable across MCMC samples?
- `pi`: Is mixing proportion close to [0.5, 0.5] with low uncertainty?

---

# Model 3: Functional Measurement Error

```stan
// Model 3: Test if measurement error scales with true value
// Key innovation: Exponential error model sigma_i * exp(alpha * theta_i)
// If alpha ≈ 0, error is independent of true value (EDA is right)
// If alpha > 0, larger true values have larger errors

data {
  int<lower=1> N;              // Number of observations (8)
  vector[N] y;                 // Observed values
  vector<lower=0>[N] sigma;    // Base measurement errors
}

parameters {
  real mu;                     // Population mean
  real<lower=0> tau;           // Between-group SD
  real<lower=-0.5, upper=0.5> alpha;  // Error scaling parameter
  vector[N] theta_raw;         // Non-centered group means
}

transformed parameters {
  vector[N] theta;
  vector<lower=0>[N] sigma_effective;

  // Non-centered parameterization
  theta = mu + tau * theta_raw;

  // Functional relationship: error scales with true value
  for (n in 1:N) {
    sigma_effective[n] = sigma[n] * exp(alpha * theta[n]);
  }
}

model {
  // Priors
  mu ~ normal(10, 20);         // Weakly informative
  tau ~ cauchy(0, 10);
  alpha ~ normal(0, 0.1);      // Small effect size, centered at independence
  theta_raw ~ std_normal();

  // Likelihood with functional error
  y ~ normal(theta, sigma_effective);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  real alpha_is_zero;  // Indicator: is alpha ≈ 0?
  real max_error_ratio;  // Diagnostic: range of error inflation

  for (n in 1:N) {
    y_rep[n] = normal_rng(theta[n], sigma_effective[n]);
    log_lik[n] = normal_lpdf(y[n] | theta[n], sigma_effective[n]);
  }

  // Diagnostic: Does posterior support independence?
  alpha_is_zero = (alpha > -0.05 && alpha < 0.05) ? 1.0 : 0.0;

  // What's the range of error inflation?
  max_error_ratio = max(sigma_effective) / min(sigma_effective);
}
```

**Alternative: Additive Heteroscedasticity**

```stan
// Alternative: Linear scaling instead of exponential

transformed parameters {
  vector[N] theta = mu + tau * theta_raw;
  vector<lower=0>[N] sigma_effective;

  for (n in 1:N) {
    // Combined error: base error + proportional error
    sigma_effective[n] = sqrt(sigma[n]^2 + (beta * theta[n])^2);
  }
}

parameters {
  real<lower=-0.5, upper=0.5> beta;  // Proportional error coefficient
  // ... other parameters same as above
}
```

**Key diagnostics:**
- `alpha` posterior: Is 95% CrI within [-0.05, 0.05]?
- `max_error_ratio`: Is it close to 1.0? (no variation)
- Compare WAIC with constant-error model

---

# Baseline: Complete Pooling (for comparison)

```stan
// Simplest model: All groups share same mean
// Known measurement errors, no between-group variation

data {
  int<lower=1> N;
  vector[N] y;
  vector<lower=0>[N] sigma;
}

parameters {
  real mu;  // Single population mean
}

model {
  mu ~ normal(10, 20);  // Weakly informative prior
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for (n in 1:N) {
    y_rep[n] = normal_rng(mu, sigma[n]);
    log_lik[n] = normal_lpdf(y[n] | mu, sigma[n]);
  }
}
```

**This is the EDA's recommended model.** All adversarial models must beat this.

---

# Model Comparison Code (Python)

```python
import cmdstanpy
import arviz as az
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('/workspace/data/data.csv')
stan_data = {
    'N': len(data),
    'y': data['y'].values,
    'sigma': data['sigma'].values
}

# Fit baseline (complete pooling)
baseline_model = cmdstanpy.CmdStanModel(stan_file='complete_pooling.stan')
baseline_fit = baseline_model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    seed=42
)

# Fit Model 1 (Inflation)
model1 = cmdstanpy.CmdStanModel(stan_file='model1_inflation.stan')
model1_fit = model1.sample(
    data=stan_data,
    chains=4,
    iter_warmup=2000,
    iter_sampling=2000,
    adapt_delta=0.95,
    seed=42
)

# Fit Model 2 (Mixture)
stan_data_mixture = {**stan_data, 'K': 2}
model2 = cmdstanpy.CmdStanModel(stan_file='model2_mixture.stan')
model2_fit = model2.sample(
    data=stan_data_mixture,
    chains=8,  # More chains for label switching detection
    iter_warmup=3000,
    iter_sampling=2000,
    adapt_delta=0.90,
    seed=42
)

# Fit Model 3 (Functional error)
model3 = cmdstanpy.CmdStanModel(stan_file='model3_functional.stan')
model3_fit = model3.sample(
    data=stan_data,
    chains=4,
    iter_warmup=1500,
    iter_sampling=2000,
    adapt_delta=0.95,
    seed=42
)

# Convergence diagnostics
print("=== CONVERGENCE DIAGNOSTICS ===")
for name, fit in [('Baseline', baseline_fit),
                   ('Model 1', model1_fit),
                   ('Model 2', model2_fit),
                   ('Model 3', model3_fit)]:
    print(f"\n{name}:")
    summary = fit.summary()
    print(f"  Max R-hat: {summary['R_hat'].max():.4f}")
    print(f"  Min ESS_bulk: {summary['ess_bulk'].min():.0f}")
    print(f"  Divergences: {fit.num_divergences()}")

# Model comparison using LOO
print("\n=== MODEL COMPARISON (LOO) ===")
baseline_idata = az.from_cmdstanpy(baseline_fit)
model1_idata = az.from_cmdstanpy(model1_fit)
model2_idata = az.from_cmdstanpy(model2_fit)
model3_idata = az.from_cmdstanpy(model3_fit)

baseline_loo = az.loo(baseline_idata, pointwise=True)
model1_loo = az.loo(model1_idata, pointwise=True)
model2_loo = az.loo(model2_idata, pointwise=True)
model3_loo = az.loo(model3_idata, pointwise=True)

# Compare models
comparison = az.compare({
    'Complete Pooling': baseline_idata,
    'Inflation Factor': model1_idata,
    'Mixture': model2_idata,
    'Functional Error': model3_idata
})
print(comparison)

# Posterior summaries for key parameters
print("\n=== KEY PARAMETER POSTERIORS ===")

print("\nModel 1 - Inflation Factor:")
lambda_posterior = model1_fit.stan_variable('lambda')
print(f"  lambda: {lambda_posterior.mean():.3f} [{np.percentile(lambda_posterior, 2.5):.3f}, {np.percentile(lambda_posterior, 97.5):.3f}]")
print(f"  Pr(lambda ∈ [0.9, 1.1]): {((lambda_posterior > 0.9) & (lambda_posterior < 1.1)).mean():.3f}")

print("\nModel 2 - Mixture:")
mu1 = model2_fit.stan_variable('mu')[:, 0]
mu2 = model2_fit.stan_variable('mu')[:, 1]
pi = model2_fit.stan_variable('pi')
print(f"  mu[1]: {mu1.mean():.3f} [{np.percentile(mu1, 2.5):.3f}, {np.percentile(mu1, 97.5):.3f}]")
print(f"  mu[2]: {mu2.mean():.3f} [{np.percentile(mu2, 2.5):.3f}, {np.percentile(mu2, 97.5):.3f}]")
print(f"  Separation: {(mu2 - mu1).mean():.3f} units")
print(f"  pi: [{pi[:, 0].mean():.3f}, {pi[:, 1].mean():.3f}]")

print("\nModel 3 - Functional Error:")
alpha_posterior = model3_fit.stan_variable('alpha')
print(f"  alpha: {alpha_posterior.mean():.4f} [{np.percentile(alpha_posterior, 2.5):.4f}, {np.percentile(alpha_posterior, 97.5):.4f}]")
print(f"  Pr(alpha ∈ [-0.05, 0.05]): {((alpha_posterior > -0.05) & (alpha_posterior < 0.05)).mean():.3f}")

# Falsification decisions
print("\n=== FALSIFICATION DECISIONS ===")

# Model 1
lambda_falsified = ((lambda_posterior > 0.9) & (lambda_posterior < 1.1)).mean() > 0.8
print(f"Model 1 (Inflation): {'FALSIFIED' if lambda_falsified else 'NOT FALSIFIED'}")
print(f"  Evidence: {100 * ((lambda_posterior > 0.9) & (lambda_posterior < 1.1)).mean():.1f}% of posterior in [0.9, 1.1]")

# Model 2
elpd_diff = comparison.loc['Mixture', 'elpd_diff']
se_diff = comparison.loc['Mixture', 'se']
mixture_falsified = (elpd_diff < -4) or (elpd_diff - 2*se_diff < 0)
print(f"Model 2 (Mixture): {'FALSIFIED' if mixture_falsified else 'NOT FALSIFIED'}")
print(f"  Evidence: ELPD diff = {elpd_diff:.2f} ± {se_diff:.2f}")

# Model 3
alpha_falsified = ((alpha_posterior > -0.05) & (alpha_posterior < 0.05)).mean() > 0.8
print(f"Model 3 (Functional): {'FALSIFIED' if alpha_falsified else 'NOT FALSIFIED'}")
print(f"  Evidence: {100 * ((alpha_posterior > -0.05) & (alpha_posterior < 0.05)).mean():.1f}% of posterior in [-0.05, 0.05]")

# Final recommendation
print("\n=== FINAL RECOMMENDATION ===")
best_model = comparison.index[0]
print(f"Best model by LOO: {best_model}")

if all([lambda_falsified, mixture_falsified, alpha_falsified]):
    print("\nCONCLUSION: All adversarial models were falsified.")
    print("The EDA conclusion (complete pooling) is STRONGLY supported.")
    print("Measurement errors are accurate, groups are homogeneous, no functional relationship.")
else:
    print("\nCONCLUSION: At least one adversarial model found evidence against complete pooling.")
    print("The EDA may have missed important structure in the data.")
    print(f"Recommend: {best_model}")
```

---

# Posterior Predictive Check Code

```python
import matplotlib.pyplot as plt
import seaborn as sns

def posterior_predictive_check(fit, data, model_name):
    """
    Generate posterior predictive checks for a fitted model.
    """
    y_obs = data['y']
    y_rep = fit.stan_variable('y_rep')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Observed vs Replicated (histogram overlay)
    axes[0, 0].hist(y_obs, bins=10, alpha=0.5, label='Observed', density=True)
    for i in range(min(100, y_rep.shape[0])):
        axes[0, 0].hist(y_rep[i, :], bins=10, alpha=0.01, color='blue', density=True)
    axes[0, 0].set_xlabel('y')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Observed vs Posterior Predictive')
    axes[0, 0].legend()

    # 2. Test statistic: variance
    var_obs = np.var(y_obs)
    var_rep = np.var(y_rep, axis=1)
    axes[0, 1].hist(var_rep, bins=30, alpha=0.7, label='Replicated')
    axes[0, 1].axvline(var_obs, color='red', linestyle='--', label='Observed')
    axes[0, 1].set_xlabel('Variance')
    axes[0, 1].set_title('Variance: Can model reproduce it?')
    axes[0, 1].legend()
    p_value_var = (var_rep > var_obs).mean()
    axes[0, 1].text(0.05, 0.95, f'p-value: {p_value_var:.3f}',
                     transform=axes[0, 1].transAxes, va='top')

    # 3. Test statistic: minimum value
    min_obs = np.min(y_obs)
    min_rep = np.min(y_rep, axis=1)
    axes[1, 0].hist(min_rep, bins=30, alpha=0.7, label='Replicated')
    axes[1, 0].axvline(min_obs, color='red', linestyle='--', label='Observed')
    axes[1, 0].set_xlabel('Minimum value')
    axes[1, 0].set_title('Extreme values: Coverage check')
    axes[1, 0].legend()
    p_value_min = (min_rep < min_obs).mean()
    axes[1, 0].text(0.05, 0.95, f'p-value: {p_value_min:.3f}',
                     transform=axes[1, 0].transAxes, va='top')

    # 4. Observation-level coverage
    coverage = np.zeros(len(y_obs))
    for i in range(len(y_obs)):
        y_rep_i = y_rep[:, i]
        lower = np.percentile(y_rep_i, 2.5)
        upper = np.percentile(y_rep_i, 97.5)
        coverage[i] = 1 if lower <= y_obs[i] <= upper else 0

    axes[1, 1].bar(range(len(coverage)), coverage)
    axes[1, 1].axhline(0.95, color='red', linestyle='--', label='Expected 95%')
    axes[1, 1].set_xlabel('Observation')
    axes[1, 1].set_ylabel('In 95% CrI?')
    axes[1, 1].set_title(f'Coverage: {coverage.sum()}/{len(coverage)} obs')
    axes[1, 1].legend()

    plt.suptitle(f'Posterior Predictive Checks: {model_name}')
    plt.tight_layout()
    plt.savefig(f'/workspace/experiments/designer_3/ppc_{model_name.lower().replace(" ", "_")}.png',
                dpi=150)
    plt.close()

    print(f"\n{model_name} - Posterior Predictive Checks:")
    print(f"  Variance p-value: {p_value_var:.3f} (expect ~0.5)")
    print(f"  Minimum p-value: {p_value_min:.3f} (expect ~0.5)")
    print(f"  Coverage: {coverage.sum()}/{len(coverage)} (expect ~95%)")

# Run for all models
posterior_predictive_check(baseline_fit, stan_data, "Complete Pooling")
posterior_predictive_check(model1_fit, stan_data, "Inflation Factor")
posterior_predictive_check(model2_fit, stan_data, "Mixture")
posterior_predictive_check(model3_fit, stan_data, "Functional Error")
```

---

# Stress Test: Synthetic Data Validation

```python
def stress_test_model(stan_model, true_params, n_reps=50):
    """
    Generate synthetic data with known parameters.
    Fit model to synthetic data.
    Check if model recovers true parameters.
    """
    results = []

    for rep in range(n_reps):
        # Generate synthetic data
        N = 8
        sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

        if 'lambda' in true_params:
            # Model 1: Known inflation
            mu_true = true_params['mu']
            tau_true = true_params['tau']
            lambda_true = true_params['lambda']
            theta_true = np.random.normal(mu_true, tau_true, N)
            y_synth = np.random.normal(theta_true, sigma * lambda_true)

        elif 'alpha' in true_params:
            # Model 3: Known functional error
            mu_true = true_params['mu']
            tau_true = true_params['tau']
            alpha_true = true_params['alpha']
            theta_true = np.random.normal(mu_true, tau_true, N)
            sigma_eff = sigma * np.exp(alpha_true * theta_true)
            y_synth = np.random.normal(theta_true, sigma_eff)

        else:
            # Complete pooling
            mu_true = true_params['mu']
            y_synth = np.random.normal(mu_true, sigma)

        # Fit model
        synth_data = {'N': N, 'y': y_synth, 'sigma': sigma}
        fit = stan_model.sample(
            data=synth_data,
            chains=2,
            iter_warmup=500,
            iter_sampling=1000,
            show_console=False
        )

        # Extract posterior mean
        mu_est = fit.stan_variable('mu').mean()
        results.append({
            'mu_true': mu_true,
            'mu_est': mu_est,
            'error': mu_est - mu_true
        })

        if 'lambda' in true_params:
            lambda_est = fit.stan_variable('lambda').mean()
            results[-1]['lambda_true'] = lambda_true
            results[-1]['lambda_est'] = lambda_est

        if 'alpha' in true_params:
            alpha_est = fit.stan_variable('alpha').mean()
            results[-1]['alpha_true'] = alpha_true
            results[-1]['alpha_est'] = alpha_est

    results_df = pd.DataFrame(results)

    print(f"\nStress Test Results ({n_reps} replications):")
    print(f"  Mean bias in mu: {results_df['error'].mean():.3f}")
    print(f"  RMSE in mu: {np.sqrt((results_df['error']**2).mean()):.3f}")

    if 'lambda_est' in results_df:
        bias_lambda = (results_df['lambda_est'] - results_df['lambda_true']).mean()
        print(f"  Mean bias in lambda: {bias_lambda:.3f}")

    if 'alpha_est' in results_df:
        bias_alpha = (results_df['alpha_est'] - results_df['alpha_true']).mean()
        print(f"  Mean bias in alpha: {bias_alpha:.4f}")

    return results_df

# Stress test Model 1 with known lambda = 1.5
print("=== STRESS TEST: Model 1 (lambda = 1.5) ===")
results_m1 = stress_test_model(
    model1,
    true_params={'mu': 10, 'tau': 5, 'lambda': 1.5},
    n_reps=20
)

# Stress test Model 3 with known alpha = 0.1
print("\n=== STRESS TEST: Model 3 (alpha = 0.1) ===")
results_m3 = stress_test_model(
    model3,
    true_params={'mu': 10, 'tau': 3, 'alpha': 0.1},
    n_reps=20
)
```

---

# Summary: Implementation Checklist

## Before Running Models
- [ ] Check Stan version (>= 2.30 recommended)
- [ ] Verify data is loaded correctly
- [ ] Run prior predictive checks (do priors make sense?)

## For Each Model
- [ ] Compile Stan model (check for syntax errors)
- [ ] Run with small iterations first (100 warmup, 100 sampling)
- [ ] Check convergence diagnostics (R-hat, ESS, divergences)
- [ ] If divergences: increase `adapt_delta` to 0.95 or 0.99
- [ ] If poor ESS: increase iterations or use non-centered parameterization
- [ ] Run posterior predictive checks
- [ ] Compute LOO (check Pareto k < 0.7)

## Model Comparison
- [ ] Compare LOO across all models
- [ ] Check ELPD difference > 4 for meaningful differences
- [ ] Evaluate falsification criteria for each model
- [ ] Make decision: Which model(s) to abandon?

## Final Decision
- [ ] If all adversarial models falsified → EDA is correct
- [ ] If any adversarial model succeeds → EDA missed something important
- [ ] Document decision and reasoning
- [ ] Report best model with full diagnostics

---

**End of Stan Implementation Guide**
