# Implementation Guide: Robust Bayesian Models
## Model Designer #3 - Practical Implementation Notes

**Audience**: Modeler/analyst who will implement these designs
**Purpose**: Step-by-step guidance for coding, fitting, and evaluating models
**Time budget**: 6-10 hours for all models + sensitivities

---

## Overview: Three Models to Implement

1. **Student-t Robust Meta-Analysis** (RSTMA) - Priority 1
2. **Uncertainty-Inflated Meta-Analysis** (UIMA) - Priority 2
3. **Finite Mixture Meta-Analysis** (TMMA) - Priority 3

Plus: Standard Normal hierarchical (benchmark)

---

## Setup and Data Preparation

### Data structure expected:
```python
# Input data
J = 8                          # Number of studies
y = [28, 8, -3, 7, -1, 1, 18, 12]  # Effect estimates (from EDA)
sigma = [15, 10, 16, 11, 9, 11, 10, 18]  # Standard errors (from EDA)

# Create dictionary for Stan
data_dict = {
    'J': J,
    'y': y,
    'sigma': sigma
}
```

### Package requirements:
```python
# Core
import numpy as np
import pandas as pd
import cmdstanpy  # or pymc as pm
import arviz as az

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
from scipy import stats
import warnings
```

### File organization:
```
/workspace/experiments/designer_3/
├── models/
│   ├── model_0_standard.stan
│   ├── model_1_student_t.stan
│   ├── model_2_mixture.stan
│   └── model_3_inflation.stan
├── fits/
│   ├── fit_0_standard.pkl
│   ├── fit_1_student_t.pkl
│   ├── fit_2_mixture.pkl
│   └── fit_3_inflation.pkl
├── diagnostics/
│   ├── convergence_checks.html
│   ├── ppc_plots.png
│   └── loo_comparison.csv
└── results/
    └── model_comparison_report.md
```

---

## Model 0: Standard Hierarchical (Benchmark)

### Stan code:
```stan
// model_0_standard.stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}

parameters {
  real mu;
  real<lower=0> tau;
  vector[J] theta_raw;  // Non-centered
}

transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}

model {
  // Priors
  mu ~ normal(0, 25);
  tau ~ cauchy(0, 5);
  theta_raw ~ std_normal();

  // Likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  vector[J] y_rep;
  vector[J] log_lik;

  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma[j]);
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma[j]);
  }
}
```

### Python fitting code:
```python
import cmdstanpy as csp

# Compile
model_0 = csp.CmdStanModel(stan_file='models/model_0_standard.stan')

# Fit
fit_0 = model_0.sample(
    data=data_dict,
    chains=4,
    parallel_chains=4,
    iter_warmup=2000,
    iter_sampling=2000,
    adapt_delta=0.90,
    seed=12345,
    show_progress=True
)

# Save
fit_0.save_csvfiles(dir='fits/')

# Diagnostics
print(fit_0.diagnose())
print(fit_0.summary(['mu', 'tau']))
```

### Expected runtime: ~2-3 minutes

### Expected results:
- mu: posterior around 7-10 (based on EDA pooled estimate)
- tau: posterior concentrated near 0 (based on I²=0%)
- Should converge easily (simple model)

---

## Model 1: Student-t Robust Meta-Analysis (PRIORITY 1)

### Stan code:
```stan
// model_1_student_t.stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}

parameters {
  real mu;
  real<lower=0> tau;
  real<lower=1> nu;  // Degrees of freedom
  vector[J] theta_raw;
}

transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
}

model {
  // Priors
  mu ~ normal(0, 25);
  tau ~ cauchy(0, 5);
  nu ~ gamma(2, 0.1);  // Mean=20, allows 1 to 100+
  theta_raw ~ std_normal();

  // Likelihood (heavy-tailed)
  y ~ student_t(nu, theta, sigma);
}

generated quantities {
  vector[J] y_rep;
  vector[J] log_lik;
  real inflation_equiv = nu < 30 ? 1.0 : 0.0;  // Indicator if heavy tails matter

  for (j in 1:J) {
    y_rep[j] = student_t_rng(nu, theta[j], sigma[j]);
    log_lik[j] = student_t_lpdf(y[j] | nu, theta[j], sigma[j]);
  }
}
```

### Python fitting code:
```python
model_1 = csp.CmdStanModel(stan_file='models/model_1_student_t.stan')

fit_1 = model_1.sample(
    data=data_dict,
    chains=4,
    parallel_chains=4,
    iter_warmup=2000,
    iter_sampling=2000,
    adapt_delta=0.95,  # Higher for Student-t
    max_treedepth=12,
    seed=12345,
    show_progress=True
)

fit_1.save_csvfiles(dir='fits/')
```

### Expected runtime: ~5-10 minutes

### Key checks:
```python
# Check convergence
summary = fit_1.summary(['mu', 'tau', 'nu'])
print(summary)

# Check if heavy tails matter
nu_samples = fit_1.stan_variable('nu')
print(f"Posterior median(nu): {np.median(nu_samples):.1f}")
print(f"P(nu < 30): {np.mean(nu_samples < 30):.3f}")

# If P(nu < 30) < 0.2, heavy tails not needed → use Model 0
# If P(nu < 30) > 0.8, heavy tails important → use Model 1
```

### Falsification check:
```python
def check_student_t_falsification(fit):
    nu_samples = fit.stan_variable('nu')

    # Check 1: nu > 50 (converged to Normal)
    if np.mean(nu_samples > 50) > 0.8:
        print("ABANDON: nu > 50, Student-t unnecessary")
        return False

    # Check 2: nu < 1.5 (extreme tails)
    if np.median(nu_samples) < 1.5 and np.percentile(nu_samples, 95) < 2:
        print("ABANDON: nu < 1.5, need contamination model")
        return False

    # Check 3: Poor PPC
    y_rep = fit.stan_variable('y_rep')
    ppc_pass = np.mean([np.sum((y >= np.percentile(y_rep[:, j], 2.5)) &
                                (y <= np.percentile(y_rep[:, j], 97.5)))
                        for j in range(J)]) / J
    if ppc_pass < 0.8:
        print(f"ABANDON: PPC fail rate {1-ppc_pass:.1%}")
        return False

    print("Student-t model passes falsification checks")
    return True

check_student_t_falsification(fit_1)
```

---

## Model 2: Finite Mixture Meta-Analysis (PRIORITY 3)

### Stan code:
```stan
// model_2_mixture.stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}

parameters {
  ordered[2] mu;  // Ensures mu[1] < mu[2]
  real<lower=0> tau_1;
  real<lower=0> tau_2;
  real<lower=0,upper=1> pi;  // Mixing proportion
  vector[J] theta_raw_1;
  vector[J] theta_raw_2;
}

transformed parameters {
  vector[J] theta_1 = mu[1] + tau_1 * theta_raw_1;
  vector[J] theta_2 = mu[2] + tau_2 * theta_raw_2;
}

model {
  // Priors
  mu[1] ~ normal(0, 20);
  mu[2] ~ normal(10, 20);
  tau_1 ~ cauchy(0, 5);
  tau_2 ~ cauchy(0, 5);
  pi ~ beta(2, 2);
  theta_raw_1 ~ std_normal();
  theta_raw_2 ~ std_normal();

  // Marginalized mixture likelihood
  for (j in 1:J) {
    target += log_mix(pi,
                      normal_lpdf(y[j] | theta_2[j], sigma[j]),
                      normal_lpdf(y[j] | theta_1[j], sigma[j]));
  }
}

generated quantities {
  vector[J] z_prob;  // P(study j in group 2)
  vector[J] log_lik;
  vector[J] y_rep;
  real mu_diff = mu[2] - mu[1];

  for (j in 1:J) {
    // Posterior group assignment
    real lp1 = log1m(pi) + normal_lpdf(y[j] | theta_1[j], sigma[j]);
    real lp2 = log(pi) + normal_lpdf(y[j] | theta_2[j], sigma[j]);
    z_prob[j] = exp(lp2 - log_sum_exp(lp1, lp2));

    // Generate from mixture
    real u = uniform_rng(0, 1);
    if (u < pi) {
      y_rep[j] = normal_rng(theta_2[j], sigma[j]);
    } else {
      y_rep[j] = normal_rng(theta_1[j], sigma[j]);
    }

    // Log-likelihood
    log_lik[j] = log_mix(pi,
                         normal_lpdf(y[j] | theta_2[j], sigma[j]),
                         normal_lpdf(y[j] | theta_1[j], sigma[j]));
  }
}
```

### Python fitting code:
```python
model_2 = csp.CmdStanModel(stan_file='models/model_2_mixture.stan')

# Initialize near K-means solution
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42).fit(np.array(y).reshape(-1, 1))
init_mu = sorted([kmeans.cluster_centers_[0, 0], kmeans.cluster_centers_[1, 0]])

fit_2 = model_2.sample(
    data=data_dict,
    chains=4,
    parallel_chains=4,
    iter_warmup=3000,  # Longer for mixture
    iter_sampling=2000,
    adapt_delta=0.95,
    max_treedepth=12,
    inits=[{'mu': init_mu} for _ in range(4)],  # Smart initialization
    seed=12345,
    show_progress=True
)

fit_2.save_csvfiles(dir='fits/')
```

### Expected runtime: ~10-20 minutes (most complex model)

### Key checks:
```python
# Check group separation
mu_samples = fit_2.stan_variable('mu')
mu_diff = fit_2.stan_variable('mu_diff')
pi_samples = fit_2.stan_variable('pi')

print(f"mu[1] posterior: {np.median(mu_samples[:, 0]):.1f}")
print(f"mu[2] posterior: {np.median(mu_samples[:, 1]):.1f}")
print(f"mu difference: {np.median(mu_diff):.1f}")
print(f"pi posterior: {np.median(pi_samples):.2f}")

# Check group assignments
z_prob = fit_2.stan_variable('z_prob')
z_prob_median = np.median(z_prob, axis=0)
print("\nPosterior group 2 probabilities:")
for j in range(J):
    print(f"Study {j+1}: {z_prob_median[j]:.2f}")
```

### Falsification check:
```python
def check_mixture_falsification(fit):
    mu_samples = fit.stan_variable('mu')
    mu_diff = fit.stan_variable('mu_diff')
    pi_samples = fit.stan_variable('pi')
    z_prob = fit.stan_variable('z_prob')

    # Check 1: Degenerate groups (pi extreme)
    if np.mean((pi_samples < 0.1) | (pi_samples > 0.9)) > 0.8:
        print("ABANDON: Degenerate groups (pi extreme)")
        return False

    # Check 2: Groups not separated
    if np.mean(mu_diff < 5) > 0.7:
        print("ABANDON: Groups not separated (mu_diff < 5)")
        return False

    # Check 3: Uncertain assignments
    z_prob_median = np.median(z_prob, axis=0)
    uncertain = np.sum((z_prob_median > 0.3) & (z_prob_median < 0.7))
    if uncertain > 4:  # >50% uncertain
        print(f"ABANDON: {uncertain}/8 studies have uncertain assignments")
        return False

    print("Mixture model passes falsification checks")
    return True

check_mixture_falsification(fit_2)
```

---

## Model 3: Uncertainty-Inflated Meta-Analysis (PRIORITY 2)

### Stan code:
```stan
// model_3_inflation.stan
data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}

parameters {
  real mu;
  real<lower=0> tau;
  real<lower=0> lambda;  // SE inflation factor
  vector[J] theta_raw;
}

transformed parameters {
  vector[J] theta = mu + tau * theta_raw;
  vector[J] sigma_inflated = sigma * lambda;
}

model {
  // Priors
  mu ~ normal(0, 25);
  tau ~ cauchy(0, 5);
  lambda ~ lognormal(0, 0.5);  // Median=1
  theta_raw ~ std_normal();

  // Likelihood with inflated SEs
  y ~ normal(theta, sigma_inflated);
}

generated quantities {
  vector[J] y_rep;
  vector[J] log_lik;
  real inflation_percent = (lambda - 1) * 100;

  for (j in 1:J) {
    y_rep[j] = normal_rng(theta[j], sigma_inflated[j]);
    log_lik[j] = normal_lpdf(y[j] | theta[j], sigma_inflated[j]);
  }
}
```

### Python fitting code:
```python
model_3 = csp.CmdStanModel(stan_file='models/model_3_inflation.stan')

fit_3 = model_3.sample(
    data=data_dict,
    chains=4,
    parallel_chains=4,
    iter_warmup=2000,
    iter_sampling=2000,
    adapt_delta=0.90,
    seed=12345,
    show_progress=True
)

fit_3.save_csvfiles(dir='fits/')
```

### Expected runtime: ~2-5 minutes

### Key checks:
```python
lambda_samples = fit_3.stan_variable('lambda')
inflation_pct = fit_3.stan_variable('inflation_percent')

print(f"lambda posterior median: {np.median(lambda_samples):.2f}")
print(f"Inflation: {np.median(inflation_pct):.1f}%")
print(f"P(lambda > 1.2): {np.mean(lambda_samples > 1.2):.3f}")
```

### Falsification check:
```python
def check_inflation_falsification(fit):
    lambda_samples = fit.stan_variable('lambda')

    # Check 1: lambda ≈ 1 (no inflation needed)
    if np.mean((lambda_samples > 0.95) & (lambda_samples < 1.05)) > 0.7:
        print("ABANDON: lambda ≈ 1, no inflation needed")
        return False

    # Check 2: Extreme inflation
    if np.median(lambda_samples) > 2.5:
        print("ABANDON: Extreme inflation (lambda > 2.5)")
        return False

    # Check 3: Check correlation with tau
    tau_samples = fit.stan_variable('tau')
    corr = np.corrcoef(lambda_samples, tau_samples)[0, 1]
    if abs(corr) > 0.8:
        print(f"WARNING: High lambda-tau correlation ({corr:.2f})")

    print("Inflation model passes falsification checks")
    return True

check_inflation_falsification(fit_3)
```

---

## Model Comparison via LOO-CV

### Compute LOO for all models:
```python
import arviz as az

# Convert to InferenceData objects
idata_0 = az.from_cmdstanpy(fit_0)
idata_1 = az.from_cmdstanpy(fit_1)
idata_2 = az.from_cmdstanpy(fit_2)
idata_3 = az.from_cmdstanpy(fit_3)

# Compute LOO
loo_0 = az.loo(idata_0, pointwise=True)
loo_1 = az.loo(idata_1, pointwise=True)
loo_2 = az.loo(idata_2, pointwise=True)
loo_3 = az.loo(idata_3, pointwise=True)

# Print LOO summaries
print("Model 0 (Standard):", loo_0)
print("Model 1 (Student-t):", loo_1)
print("Model 2 (Mixture):", loo_2)
print("Model 3 (Inflation):", loo_3)

# Compare models
comparison = az.compare({
    'Standard': idata_0,
    'Student-t': idata_1,
    'Mixture': idata_2,
    'Inflation': idata_3
})

print("\nModel Comparison:")
print(comparison)

# Visualize
az.plot_compare(comparison, insample_dev=False)
plt.tight_layout()
plt.savefig('diagnostics/loo_comparison.png', dpi=300)
```

### Decision rules:
```python
def interpret_loo_comparison(comparison):
    best_model = comparison.index[0]
    elpd_diff = comparison.loc[comparison.index[1], 'elpd_diff']
    se_diff = comparison.loc[comparison.index[1], 'dse']

    print(f"\nBest model by LOO: {best_model}")

    if abs(elpd_diff) < 2 * se_diff:
        print(f"Models equivalent (|elpd_diff| < 2*SE)")
        print("→ Use simplest model")
    elif elpd_diff > 2 * se_diff:
        print(f"Best model clearly better (elpd_diff = {elpd_diff:.1f} ± {se_diff:.1f})")
        print(f"→ Use {best_model}")

    # Check Pareto k diagnostics
    for model_name in comparison.index:
        if model_name == 'Standard':
            loo = loo_0
        elif model_name == 'Student-t':
            loo = loo_1
        elif model_name == 'Mixture':
            loo = loo_2
        else:
            loo = loo_3

        pareto_k = loo.pareto_k
        n_bad = np.sum(pareto_k > 0.7)
        if n_bad > 0:
            print(f"WARNING: {model_name} has {n_bad} studies with Pareto k > 0.7")

interpret_loo_comparison(comparison)
```

---

## Posterior Predictive Checks

### Comprehensive PPC function:
```python
def posterior_predictive_check(fit, y_obs, model_name):
    """Comprehensive posterior predictive checks"""
    y_rep = fit.stan_variable('y_rep')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Overlay plot
    ax = axes[0, 0]
    for j in range(J):
        ax.hist(y_rep[:, j], bins=30, alpha=0.3, density=True)
    ax.axvline(y_obs[j], color='red', linestyle='--', linewidth=2, label='Observed')
    ax.set_title('Posterior Predictive Distribution vs Observed')
    ax.set_xlabel('Effect size')
    ax.legend()

    # 2. Calibration plot
    ax = axes[0, 1]
    coverage = []
    for j in range(J):
        lower = np.percentile(y_rep[:, j], 2.5)
        upper = np.percentile(y_rep[:, j], 97.5)
        coverage.append(1 if lower <= y_obs[j] <= upper else 0)
    ax.bar(range(1, J+1), coverage, color=['green' if c else 'red' for c in coverage])
    ax.axhline(0.95, color='blue', linestyle='--', label='Expected 95%')
    ax.set_title(f'Coverage: {np.mean(coverage)*100:.0f}% in 95% intervals')
    ax.set_xlabel('Study')
    ax.set_ylabel('Covered (1) or Not (0)')
    ax.set_ylim([-0.1, 1.1])
    ax.legend()

    # 3. Test statistics
    ax = axes[1, 0]
    # Mean test
    y_rep_mean = np.mean(y_rep, axis=1)
    y_obs_mean = np.mean(y_obs)
    p_value_mean = np.mean(y_rep_mean >= y_obs_mean)
    ax.hist(y_rep_mean, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(y_obs_mean, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'Mean: p-value = {p_value_mean:.3f}')
    ax.set_xlabel('Mean effect size')

    # 4. SD test
    ax = axes[1, 1]
    y_rep_sd = np.std(y_rep, axis=1)
    y_obs_sd = np.std(y_obs)
    p_value_sd = np.mean(y_rep_sd >= y_obs_sd)
    ax.hist(y_rep_sd, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(y_obs_sd, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'SD: p-value = {p_value_sd:.3f}')
    ax.set_xlabel('SD of effect sizes')

    plt.suptitle(f'Posterior Predictive Checks: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'diagnostics/ppc_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()

    # Summary
    print(f"\n{model_name} PPC Summary:")
    print(f"  Coverage: {np.mean(coverage)*100:.0f}% of studies in 95% intervals")
    print(f"  Mean p-value: {p_value_mean:.3f} (should be ~0.5)")
    print(f"  SD p-value: {p_value_sd:.3f} (should be ~0.5)")

    return np.mean(coverage) >= 0.8  # Pass if >80% coverage

# Run for all models
y_obs = np.array(y)
ppc_pass = {
    'Standard': posterior_predictive_check(fit_0, y_obs, 'Standard'),
    'Student-t': posterior_predictive_check(fit_1, y_obs, 'Student-t'),
    'Mixture': posterior_predictive_check(fit_2, y_obs, 'Mixture'),
    'Inflation': posterior_predictive_check(fit_3, y_obs, 'Inflation')
}

print("\nPPC Pass/Fail:")
for model, passed in ppc_pass.items():
    print(f"  {model}: {'PASS' if passed else 'FAIL'}")
```

---

## Convergence Diagnostics

### Automated diagnostic function:
```python
def check_convergence(fit, model_name, params_to_check=['mu', 'tau']):
    """Check convergence for a fitted model"""
    print(f"\n{'='*60}")
    print(f"Convergence Diagnostics: {model_name}")
    print(f"{'='*60}")

    # Get summary
    summary = fit.summary()

    # Check R-hat
    rhat_max = summary['R_hat'].max()
    rhat_fail = summary[summary['R_hat'] > 1.01]
    print(f"\n1. R-hat:")
    print(f"   Max R-hat: {rhat_max:.4f}")
    if len(rhat_fail) > 0:
        print(f"   FAIL: {len(rhat_fail)} parameters with R-hat > 1.01")
        print(rhat_fail[['R_hat']])
    else:
        print(f"   PASS: All R-hat < 1.01")

    # Check ESS
    ess_bulk_min = summary['ess_bulk'].min()
    ess_tail_min = summary['ess_tail'].min()
    ess_fail_bulk = summary[summary['ess_bulk'] < 400]
    ess_fail_tail = summary[summary['ess_tail'] < 400]
    print(f"\n2. Effective Sample Size:")
    print(f"   Min ESS (bulk): {ess_bulk_min:.0f}")
    print(f"   Min ESS (tail): {ess_tail_min:.0f}")
    if len(ess_fail_bulk) > 0 or len(ess_fail_tail) > 0:
        print(f"   WARNING: Some parameters have ESS < 400")
        if len(ess_fail_bulk) > 0:
            print(f"   Bulk: {len(ess_fail_bulk)} parameters")
        if len(ess_fail_tail) > 0:
            print(f"   Tail: {len(ess_fail_tail)} parameters")
    else:
        print(f"   PASS: All ESS > 400")

    # Check divergences
    try:
        divergences = fit.divergences
        n_div = np.sum(divergences)
        print(f"\n3. Divergent Transitions:")
        print(f"   Count: {n_div}")
        if n_div > 10:
            print(f"   FAIL: >10 divergences")
        elif n_div > 0:
            print(f"   WARNING: Some divergences present")
        else:
            print(f"   PASS: No divergences")
    except:
        print(f"\n3. Divergent Transitions: Could not check")

    # Overall verdict
    print(f"\n4. Overall Verdict:")
    if rhat_max < 1.01 and ess_bulk_min > 400 and n_div < 10:
        print(f"   ✓ CONVERGED - Model ready for inference")
        return True
    else:
        print(f"   ✗ ISSUES DETECTED - Review before inference")
        return False

# Check all models
conv_results = {
    'Standard': check_convergence(fit_0, 'Standard'),
    'Student-t': check_convergence(fit_1, 'Student-t', ['mu', 'tau', 'nu']),
    'Mixture': check_convergence(fit_2, 'Mixture', ['mu', 'tau_1', 'tau_2', 'pi']),
    'Inflation': check_convergence(fit_3, 'Inflation', ['mu', 'tau', 'lambda'])
}
```

---

## Sensitivity Analyses

### Leave-one-out sensitivity:
```python
def leave_one_out_sensitivity(model, data_dict, model_name):
    """Fit model J times, each time leaving out one study"""
    print(f"\nLeave-One-Out Sensitivity: {model_name}")

    results = []
    for j in range(J):
        # Remove study j
        data_loo = {
            'J': J - 1,
            'y': [y[i] for i in range(J) if i != j],
            'sigma': [sigma[i] for i in range(J) if i != j]
        }

        # Fit
        print(f"  Fitting without Study {j+1}...", end='')
        fit_loo = model.sample(
            data=data_loo,
            chains=2,  # Fewer chains for speed
            iter_warmup=1000,
            iter_sampling=1000,
            show_progress=False,
            show_console=False
        )

        # Extract mu
        mu_samples = fit_loo.stan_variable('mu')
        mu_median = np.median(mu_samples)
        mu_ci = np.percentile(mu_samples, [2.5, 97.5])

        results.append({
            'excluded': j+1,
            'mu_median': mu_median,
            'mu_lower': mu_ci[0],
            'mu_upper': mu_ci[1]
        })
        print(f" mu = {mu_median:.2f} [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}]")

    # Plot
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.errorbar(results_df['excluded'], results_df['mu_median'],
                 yerr=[results_df['mu_median'] - results_df['mu_lower'],
                       results_df['mu_upper'] - results_df['mu_median']],
                 fmt='o', capsize=5, capthick=2, markersize=8)
    plt.xlabel('Study Excluded')
    plt.ylabel('Pooled Effect Estimate (mu)')
    plt.title(f'Leave-One-Out Sensitivity: {model_name}')
    plt.axhline(np.median(results_df['mu_median']), color='red', linestyle='--',
                label='Average')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'diagnostics/loo_sensitivity_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()

    # Check Study 1 influence (most influential from EDA)
    study_1_impact = results_df.loc[results_df['excluded'] == 1, 'mu_median'].values[0]
    full_data_mu = np.median(fit.stan_variable('mu'))  # Assuming 'fit' is full data fit
    impact_pct = abs(study_1_impact - full_data_mu) / abs(full_data_mu) * 100
    print(f"\nStudy 1 impact: {impact_pct:.1f}% change in mu when excluded")

    return results_df

# Run for best model(s)
# loo_results_1 = leave_one_out_sensitivity(model_1, data_dict, 'Student-t')
```

### Prior sensitivity:
```python
def prior_sensitivity_tau(model_code_template, data_dict, scales=[2.5, 5.0, 10.0]):
    """Test sensitivity to tau prior scale"""
    print("\nPrior Sensitivity Analysis: tau ~ Half-Cauchy(0, scale)")

    results = []
    for scale in scales:
        # Modify Stan code (assuming template with SCALE placeholder)
        modified_code = model_code_template.replace('cauchy(0, 5)', f'cauchy(0, {scale})')

        # Write temporary file
        with open(f'models/temp_scale_{scale}.stan', 'w') as f:
            f.write(modified_code)

        # Compile and fit
        model_temp = csp.CmdStanModel(stan_file=f'models/temp_scale_{scale}.stan')
        fit_temp = model_temp.sample(data=data_dict, chains=2,
                                     iter_warmup=1000, iter_sampling=1000,
                                     show_progress=False)

        # Extract results
        mu_samples = fit_temp.stan_variable('mu')
        tau_samples = fit_temp.stan_variable('tau')

        results.append({
            'scale': scale,
            'mu_median': np.median(mu_samples),
            'mu_ci': np.percentile(mu_samples, [2.5, 97.5]),
            'tau_median': np.median(tau_samples),
            'tau_ci': np.percentile(tau_samples, [2.5, 97.5])
        })

        print(f"  Scale={scale}: mu={np.median(mu_samples):.2f}, tau={np.median(tau_samples):.2f}")

    # Check if conclusions change
    mu_range = max([r['mu_median'] for r in results]) - min([r['mu_median'] for r in results])
    print(f"\nmu range across priors: {mu_range:.2f}")
    if mu_range < 2:
        print("→ Robust to prior choice")
    else:
        print("→ WARNING: Sensitive to prior choice")

    return results

# Usage (if needed):
# prior_sens_results = prior_sensitivity_tau(model_code, data_dict)
```

---

## Final Report Generation

### Automated report:
```python
def generate_final_report(fit_dict, loo_comparison, ppc_results, conv_results):
    """Generate final markdown report"""

    report = []
    report.append("# Robust Bayesian Meta-Analysis Results")
    report.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    report.append(f"**Dataset**: J={J} studies\n")

    report.append("## Model Comparison Summary\n")
    report.append(loo_comparison.to_markdown())

    report.append("\n## Best Model\n")
    best_model = loo_comparison.index[0]
    report.append(f"**Selected**: {best_model}\n")

    # Extract posterior for best model
    fit_best = fit_dict[best_model]
    mu_samples = fit_best.stan_variable('mu')
    tau_samples = fit_best.stan_variable('tau')

    report.append("### Posterior Estimates\n")
    report.append(f"- **Overall effect (mu)**: {np.median(mu_samples):.2f} ")
    report.append(f"(95% CI: [{np.percentile(mu_samples, 2.5):.2f}, {np.percentile(mu_samples, 97.5):.2f}])")
    report.append(f"\n- **Heterogeneity (tau)**: {np.median(tau_samples):.2f} ")
    report.append(f"(95% CI: [{np.percentile(tau_samples, 2.5):.2f}, {np.percentile(tau_samples, 97.5):.2f}])")
    report.append(f"\n- **P(mu > 0 | data)**: {np.mean(mu_samples > 0):.3f}")
    report.append(f"\n- **P(tau > 0 | data)**: {np.mean(tau_samples > 0):.3f}\n")

    # Model-specific parameters
    if best_model == 'Student-t':
        nu_samples = fit_best.stan_variable('nu')
        report.append(f"\n- **Degrees of freedom (nu)**: {np.median(nu_samples):.1f} ")
        report.append(f"(95% CI: [{np.percentile(nu_samples, 2.5):.1f}, {np.percentile(nu_samples, 97.5):.1f}])")
    elif best_model == 'Inflation':
        lambda_samples = fit_best.stan_variable('lambda')
        report.append(f"\n- **SE inflation (lambda)**: {np.median(lambda_samples):.2f} ")
        report.append(f"({(np.median(lambda_samples)-1)*100:.1f}% inflation)")
    elif best_model == 'Mixture':
        mu_samples_mix = fit_best.stan_variable('mu')
        pi_samples = fit_best.stan_variable('pi')
        report.append(f"\n- **Group 1 mean**: {np.median(mu_samples_mix[:, 0]):.2f}")
        report.append(f"\n- **Group 2 mean**: {np.median(mu_samples_mix[:, 1]):.2f}")
        report.append(f"\n- **Group 2 proportion**: {np.median(pi_samples):.2f}")

    report.append("\n## Model Diagnostics\n")
    report.append("- Convergence: " + ("✓ PASS" if conv_results[best_model] else "✗ ISSUES"))
    report.append("\n- Posterior Predictive Check: " + ("✓ PASS" if ppc_results[best_model] else "✗ FAIL"))

    report.append("\n## Conclusions\n")
    if np.mean(mu_samples > 0) > 0.975:
        report.append("- Strong evidence for positive effect (P > 0.975)")
    elif np.mean(mu_samples > 0) > 0.95:
        report.append("- Moderate evidence for positive effect (0.95 < P < 0.975)")
    elif np.mean(mu_samples > 0) > 0.5:
        report.append("- Weak evidence for positive effect (0.5 < P < 0.95)")
    else:
        report.append("- No clear evidence for positive effect")

    if np.median(tau_samples) < 1:
        report.append("\n- Low heterogeneity (consistent effects across studies)")
    else:
        report.append("\n- Moderate-to-high heterogeneity (effects vary across studies)")

    # Write to file
    with open('results/model_comparison_report.md', 'w') as f:
        f.write('\n'.join(report))

    print("\nFinal report written to: results/model_comparison_report.md")

# Generate report
fit_dict = {
    'Standard': fit_0,
    'Student-t': fit_1,
    'Mixture': fit_2,
    'Inflation': fit_3
}

generate_final_report(fit_dict, comparison, ppc_pass, conv_results)
```

---

## Troubleshooting Guide

### Problem: Divergent transitions
**Solution**:
1. Increase `adapt_delta` to 0.95 or 0.99
2. Use non-centered parameterization (already in code)
3. Increase `max_treedepth` to 12 or 13
4. Check for multicollinearity (e.g., nu and tau in Student-t)

### Problem: Low ESS
**Solution**:
1. Run longer chains (iter_sampling=4000)
2. Thin chains (thin=2)
3. Check for high correlation between parameters
4. Consider reparameterization

### Problem: R-hat > 1.01
**Solution**:
1. Run longer warmup (iter_warmup=3000)
2. Run more chains (chains=6 or 8)
3. Check for multimodal posteriors (mixture models)
4. Try different initializations

### Problem: Model won't compile
**Solution**:
1. Check Stan version compatibility
2. Ensure all brackets/parentheses matched
3. Check that all variables declared before use
4. Look for typos in distribution names

### Problem: Results don't make sense
**Solution**:
1. Run prior predictive check (before seeing data)
2. Check data input (are y and sigma correct?)
3. Verify priors are reasonable
4. Compare to EDA findings (should be roughly consistent)
5. Check falsification criteria

---

## Time Budget Summary

| Task | Expected Time | Priority |
|------|---------------|----------|
| Setup + Data prep | 30 min | HIGH |
| Model 0 (Standard) | 30 min | HIGH |
| Model 1 (Student-t) | 1-2 hours | HIGH |
| Model 3 (Inflation) | 1 hour | MEDIUM |
| Model 2 (Mixture) | 2-3 hours | MEDIUM |
| LOO comparison | 30 min | HIGH |
| PPC for all models | 1 hour | HIGH |
| Leave-one-out sensitivity | 2 hours | MEDIUM |
| Prior sensitivity | 1 hour | LOW |
| Report generation | 1 hour | HIGH |
| **TOTAL** | **6-10 hours** | |

**Minimum viable analysis** (if time limited): Models 0, 1, 3 + LOO + PPC = ~4 hours

---

**Good luck! Remember**: It's okay to abandon models that fail falsification criteria. Success = finding truth, not completing all analyses.
