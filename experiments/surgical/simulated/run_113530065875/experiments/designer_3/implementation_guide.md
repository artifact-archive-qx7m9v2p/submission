# Implementation Guide: From Theory to Practice
## Designer #3: Working Code Examples

**Purpose**: Copy-paste ready code for all three models in both Stan (R/Python) and PyMC

---

## Setup

### R Dependencies
```r
install.packages(c("rstan", "loo", "bayesplot", "tidyverse"))

library(rstan)
library(loo)
library(bayesplot)
library(tidyverse)

# Stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
```

### Python Dependencies
```python
pip install pymc arviz numpy pandas matplotlib seaborn

import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

## Data Loading

### R
```r
# Assuming data is in CSV format
data <- read_csv("data.csv")  # columns: group, n, r

# Prepare for Stan
stan_data <- list(
  J = nrow(data),
  n = data$n,
  r = data$r
)

# Quick check
cat(sprintf("Groups: %d, Total trials: %d, Total successes: %d\n",
            stan_data$J, sum(stan_data$n), sum(stan_data$r)))
```

### Python
```python
# Load data
data = pd.read_csv("data.csv")  # columns: group, n, r

# Quick check
print(f"Groups: {len(data)}, Total trials: {data['n'].sum()}, "
      f"Total successes: {data['r'].sum()}")

# Extract arrays
J = len(data)
n = data['n'].values
r = data['r'].values
```

---

## Model 1: Hierarchical Binomial (Non-Centered)

### Stan Code (Save as `hierarchical.stan`)
```stan
data {
  int<lower=1> J;              // Number of groups
  array[J] int<lower=0> n;     // Trials per group
  array[J] int<lower=0> r;     // Successes per group
}

parameters {
  real mu;                      // Population mean (logit scale)
  real<lower=0> tau;            // Between-group SD (logit scale)
  vector[J] eta;                // Non-centered group effects
}

transformed parameters {
  vector[J] theta = mu + tau * eta;  // Group-level logit rates
  vector[J] p = inv_logit(theta);    // Success probabilities
}

model {
  // Priors
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, 1);           // Half-Cauchy via lower=0 constraint
  eta ~ std_normal();           // Standard normal for non-centered

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  // Posterior predictive checks
  array[J] int r_rep;
  for (j in 1:J) {
    r_rep[j] = binomial_rng(n[j], p[j]);
  }

  // LOO pointwise log-likelihood
  vector[J] log_lik;
  for (j in 1:J) {
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
  }

  // Derived quantities
  real pooled_p = inv_logit(mu);      // Population mean rate
  real tau_out = tau;                  // For monitoring

  // Shrinkage for each group
  vector[J] raw_p = to_vector(r) ./ to_vector(n);  // Raw proportions
  vector[J] shrinkage = abs(p - raw_p) ./ abs(raw_p - pooled_p);
}
```

### R: Fit and Diagnose
```r
# Compile and fit
fit1 <- stan(
  file = "hierarchical.stan",
  data = stan_data,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 42
)

# Print summary
print(fit1, pars = c("mu", "tau", "p", "pooled_p"))

# Convergence diagnostics
check_hmc_diagnostics(fit1)
summary(fit1)$summary[, "Rhat"]  # All should be <1.01

# Extract samples
samples1 <- extract(fit1)

# Posterior summaries
posterior_summary <- data.frame(
  group = 1:stan_data$J,
  raw_rate = stan_data$r / stan_data$n,
  posterior_mean = colMeans(samples1$p),
  posterior_sd = apply(samples1$p, 2, sd),
  q025 = apply(samples1$p, 2, quantile, 0.025),
  q975 = apply(samples1$p, 2, quantile, 0.975)
)

print(posterior_summary)

# Visual diagnostics
mcmc_trace(fit1, pars = c("mu", "tau"))
mcmc_pairs(fit1, pars = c("mu", "tau", "p[1]", "p[4]", "p[8]"))

# LOO cross-validation
log_lik1 <- extract_log_lik(fit1, merge_chains = FALSE)
loo1 <- loo(log_lik1, r_eff = relative_eff(log_lik1))
print(loo1)

# Check Pareto k values
plot(loo1)
pareto_k <- loo1$diagnostics$pareto_k
cat("Pareto k > 0.7:", sum(pareto_k > 0.7), "\n")

# Posterior predictive check
r_rep <- samples1$r_rep
ppc_dens_overlay(stan_data$r, r_rep[1:100, ])
ppc_stat(stan_data$r, r_rep, stat = "var")  # Check variance

# Shrinkage plot
shrinkage_data <- data.frame(
  group = 1:stan_data$J,
  raw = stan_data$r / stan_data$n,
  posterior = colMeans(samples1$p),
  pooled = mean(samples1$pooled_p),
  n = stan_data$n
)

ggplot(shrinkage_data, aes(x = group)) +
  geom_point(aes(y = raw, size = n), color = "blue", alpha = 0.6) +
  geom_point(aes(y = posterior), color = "red", size = 3) +
  geom_segment(aes(x = group, xend = group, y = raw, yend = posterior),
               arrow = arrow(length = unit(0.2, "cm"))) +
  geom_hline(yintercept = mean(shrinkage_data$pooled),
             linetype = "dashed", color = "gray50") +
  labs(title = "Shrinkage: Raw (blue) → Posterior (red)",
       y = "Success Rate", x = "Group") +
  theme_minimal()
```

### Python: Fit and Diagnose
```python
# Using PyStan
import stan

with open("hierarchical.stan", "r") as f:
    stan_code = f.read()

# Build model
posterior1 = stan.build(stan_code, data={"J": J, "n": n, "r": r})

# Sample
fit1 = posterior1.sample(num_chains=4, num_samples=1000, num_warmup=1000)

# Check diagnostics
df1 = fit1.to_frame()
print(az.summary(fit1, var_names=["mu", "tau", "p"]))

# Rhats
rhats = az.rhat(fit1)
print(f"Max Rhat: {rhats.max().values}")

# LOO
loo1 = az.loo(fit1, pointwise=True)
print(loo1)

# Pareto k
pareto_k = loo1.pareto_k.values
print(f"Pareto k > 0.7: {(pareto_k > 0.7).sum()}")

# Posterior predictive check
az.plot_ppc(fit1, num_pp_samples=100)
plt.show()

# Shrinkage plot
p_post = fit1["p"].mean(axis=(0, 1))
p_raw = r / n
pooled = fit1["pooled_p"].mean()

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(J):
    ax.scatter(i+1, p_raw[i], s=n[i]/5, color="blue", alpha=0.6)
    ax.scatter(i+1, p_post[i], s=100, color="red")
    ax.plot([i+1, i+1], [p_raw[i], p_post[i]], 'k-', alpha=0.3)
ax.axhline(pooled, color="gray", linestyle="--")
ax.set_xlabel("Group")
ax.set_ylabel("Success Rate")
ax.set_title("Shrinkage: Raw (blue) → Posterior (red)")
plt.show()
```

### PyMC Implementation
```python
with pm.Model() as model1:
    # Hyperpriors
    mu = pm.Normal('mu', mu=-2.5, sigma=1)
    tau = pm.HalfCauchy('tau', beta=1)

    # Non-centered parameterization
    eta = pm.Normal('eta', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * eta)
    p = pm.Deterministic('p', pm.math.invlogit(theta))

    # Likelihood
    r_obs = pm.Binomial('r_obs', n=n, p=p, observed=r)

    # Posterior predictive
    r_pred = pm.Binomial('r_pred', n=n, p=p)

    # Sample
    trace1 = pm.sample(1000, tune=1000, chains=4, cores=4,
                       target_accept=0.8, return_inferencedata=True)

# Diagnostics
print(az.summary(trace1, var_names=["mu", "tau", "p"]))
az.plot_trace(trace1, var_names=["mu", "tau"])
plt.show()

# LOO
loo1_pymc = az.loo(trace1)
print(loo1_pymc)

# Posterior predictive check
az.plot_ppc(trace1, num_pp_samples=100)
plt.show()
```

---

## Model 2: Beta-Binomial

### Stan Code (Save as `beta_binomial.stan`)
```stan
data {
  int<lower=1> J;
  array[J] int<lower=0> n;
  array[J] int<lower=0> r;
}

parameters {
  real<lower=0, upper=1> mu;      // Mean success rate
  real<lower=0> phi;               // Concentration parameter
}

transformed parameters {
  real alpha = mu * phi;
  real beta = (1 - mu) * phi;
}

model {
  // Priors
  mu ~ beta(5, 50);                // Weakly informative
  phi ~ gamma(1, 0.1);             // Allows overdispersion

  // Likelihood
  for (j in 1:J) {
    r[j] ~ beta_binomial(n[j], alpha, beta);
  }
}

generated quantities {
  array[J] int r_rep;
  vector[J] log_lik;

  for (j in 1:J) {
    r_rep[j] = beta_binomial_rng(n[j], alpha, beta);
    log_lik[j] = beta_binomial_lpmf(r[j] | n[j], alpha, beta);
  }

  // Implied overdispersion
  real var_ratio = 1.0 / (1.0 + phi);
}
```

### R: Fit
```r
fit2 <- stan(
  file = "beta_binomial.stan",
  data = stan_data,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4
)

print(fit2, pars = c("mu", "phi", "var_ratio"))

# LOO
log_lik2 <- extract_log_lik(fit2, merge_chains = FALSE)
loo2 <- loo(log_lik2, r_eff = relative_eff(log_lik2))

# Compare to Model 1
loo_compare(loo1, loo2)
```

### PyMC Implementation
```python
with pm.Model() as model2:
    # Priors
    mu = pm.Beta('mu', alpha=5, beta=50)
    phi = pm.Gamma('phi', alpha=1, beta=0.1)

    # Beta-binomial parameters
    alpha = pm.Deterministic('alpha', mu * phi)
    beta = pm.Deterministic('beta', (1 - mu) * phi)

    # Likelihood
    r_obs = pm.BetaBinomial('r_obs', n=n, alpha=alpha, beta=beta, observed=r)

    # Sample
    trace2 = pm.sample(1000, tune=1000, chains=4, cores=4,
                       return_inferencedata=True)

# Compare models
compare = az.compare({"hierarchical": trace1, "beta_binomial": trace2})
print(compare)
```

---

## Model 3: Robust Hierarchical (Student-t)

### Stan Code (Save as `robust_hierarchical.stan`)
```stan
data {
  int<lower=1> J;
  array[J] int<lower=0> n;
  array[J] int<lower=0> r;
}

parameters {
  real mu;
  real<lower=0> tau;
  real<lower=0> nu;               // Degrees of freedom
  vector[J] eta;
}

transformed parameters {
  vector[J] theta = mu + tau * eta;
  vector[J] p = inv_logit(theta);
}

model {
  // Priors
  mu ~ normal(-2.5, 1);
  tau ~ cauchy(0, 1);
  nu ~ gamma(2, 0.1);             // Learns tail heaviness

  // Heavy-tailed group effects
  eta ~ student_t(nu, 0, 1);      // KEY DIFFERENCE

  // Likelihood
  r ~ binomial_logit(n, theta);
}

generated quantities {
  array[J] int r_rep;
  vector[J] log_lik;

  for (j in 1:J) {
    r_rep[j] = binomial_rng(n[j], p[j]);
    log_lik[j] = binomial_logit_lpmf(r[j] | n[j], theta[j]);
  }

  real nu_out = nu;  // Monitor this!
}
```

### R: Fit
```r
fit3 <- stan(
  file = "robust_hierarchical.stan",
  data = stan_data,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  control = list(adapt_delta = 0.95)  # Important!
)

print(fit3, pars = c("mu", "tau", "nu", "p"))

# Check nu posterior
samples3 <- extract(fit3)
hist(samples3$nu, main = "Posterior nu (degrees of freedom)",
     xlab = "nu", breaks = 30)
abline(v = 5, col = "red", lty = 2)  # Heavy tails if nu < 5
cat("P(nu < 5):", mean(samples3$nu < 5), "\n")

# LOO comparison
log_lik3 <- extract_log_lik(fit3, merge_chains = FALSE)
loo3 <- loo(log_lik3, r_eff = relative_eff(log_lik3))

loo_compare(loo1, loo2, loo3)
```

---

## Complete Workflow Script

### R Version
```r
# Complete analysis script
library(rstan)
library(loo)
library(bayesplot)
library(tidyverse)

# Setup
options(mc.cores = 4)
rstan_options(auto_write = TRUE)

# Load data
data <- read_csv("data.csv")
stan_data <- list(J = nrow(data), n = data$n, r = data$r)

# Fit all models
fit1 <- stan("hierarchical.stan", data = stan_data,
             chains = 4, iter = 2000, cores = 4)
fit2 <- stan("beta_binomial.stan", data = stan_data,
             chains = 4, iter = 2000, cores = 4)
fit3 <- stan("robust_hierarchical.stan", data = stan_data,
             chains = 4, iter = 2000, cores = 4,
             control = list(adapt_delta = 0.95))

# Diagnostics
check_hmc_diagnostics(fit1)
check_hmc_diagnostics(fit3)

# LOO comparison
loo1 <- loo(fit1)
loo2 <- loo(fit2)
loo3 <- loo(fit3)

comparison <- loo_compare(loo1, loo2, loo3)
print(comparison)

# Choose best model
best_model <- rownames(comparison)[1]
cat("\nBest model:", best_model, "\n")

# Detailed results for best model
if (best_model == "fit1") {
  print(fit1, pars = c("mu", "tau", "p"))
  mcmc_trace(fit1, pars = c("mu", "tau"))

  # Shrinkage plot
  samples <- extract(fit1)
  shrinkage_df <- data.frame(
    group = 1:stan_data$J,
    raw = stan_data$r / stan_data$n,
    posterior = colMeans(samples$p),
    n = stan_data$n
  )

  ggplot(shrinkage_df, aes(x = group)) +
    geom_segment(aes(xend = group, y = raw, yend = posterior),
                 arrow = arrow(length = unit(0.2, "cm"))) +
    geom_point(aes(y = raw, size = n), color = "blue", alpha = 0.6) +
    geom_point(aes(y = posterior), color = "red", size = 3) +
    labs(title = "Hierarchical Model: Shrinkage Effect",
         x = "Group", y = "Success Rate") +
    theme_minimal()

  ggsave("shrinkage_plot.png", width = 10, height = 6)
}

# Save results
saveRDS(list(fit1 = fit1, fit2 = fit2, fit3 = fit3,
             loo1 = loo1, loo2 = loo2, loo3 = loo3),
        "model_results.rds")
```

### Python Version
```python
# Complete analysis script
import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")
J = len(data)
n = data['n'].values
r = data['r'].values

# Model 1: Hierarchical
with pm.Model() as model1:
    mu = pm.Normal('mu', mu=-2.5, sigma=1)
    tau = pm.HalfCauchy('tau', beta=1)
    eta = pm.Normal('eta', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * eta)
    p = pm.Deterministic('p', pm.math.invlogit(theta))
    r_obs = pm.Binomial('r_obs', n=n, p=p, observed=r)
    trace1 = pm.sample(1000, tune=1000, chains=4, cores=4,
                       target_accept=0.8, return_inferencedata=True)

# Model 2: Beta-binomial
with pm.Model() as model2:
    mu = pm.Beta('mu', alpha=5, beta=50)
    phi = pm.Gamma('phi', alpha=1, beta=0.1)
    alpha = pm.Deterministic('alpha', mu * phi)
    beta = pm.Deterministic('beta', (1 - mu) * phi)
    r_obs = pm.BetaBinomial('r_obs', n=n, alpha=alpha, beta=beta, observed=r)
    trace2 = pm.sample(1000, tune=1000, chains=4, cores=4,
                       return_inferencedata=True)

# Model 3: Robust
with pm.Model() as model3:
    mu = pm.Normal('mu', mu=-2.5, sigma=1)
    tau = pm.HalfCauchy('tau', beta=1)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)
    eta = pm.StudentT('eta', nu=nu, mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * eta)
    p = pm.Deterministic('p', pm.math.invlogit(theta))
    r_obs = pm.Binomial('r_obs', n=n, p=p, observed=r)
    trace3 = pm.sample(1000, tune=1000, chains=4, cores=4,
                       target_accept=0.95, return_inferencedata=True)

# Compare models
compare = az.compare({"hierarchical": trace1,
                      "beta_binomial": trace2,
                      "robust": trace3})
print(compare)

# Best model
best_model = compare.index[0]
print(f"\nBest model: {best_model}")

# Visualize best model (assuming hierarchical)
az.plot_trace(trace1, var_names=["mu", "tau"])
plt.tight_layout()
plt.savefig("trace_plot.png")

# Shrinkage plot
p_post = trace1.posterior["p"].mean(dim=["chain", "draw"]).values
p_raw = r / n

fig, ax = plt.subplots(figsize=(12, 6))
for i in range(J):
    ax.plot([i+1, i+1], [p_raw[i], p_post[i]], 'k-', alpha=0.3)
    ax.scatter(i+1, p_raw[i], s=n[i]/5, color="blue", alpha=0.6,
               label="Raw" if i == 0 else "")
    ax.scatter(i+1, p_post[i], s=100, color="red",
               label="Posterior" if i == 0 else "")
ax.set_xlabel("Group")
ax.set_ylabel("Success Rate")
ax.set_title("Shrinkage: Raw → Posterior")
ax.legend()
plt.tight_layout()
plt.savefig("shrinkage_plot.png")

# Save results
trace1.to_netcdf("model1_trace.nc")
trace2.to_netcdf("model2_trace.nc")
trace3.to_netcdf("model3_trace.nc")
```

---

## Quick Diagnostic Checks

### R Functions
```r
# Quick diagnostic function
quick_check <- function(fit, model_name) {
  cat("\n=== Diagnostics for", model_name, "===\n")

  # Convergence
  rhats <- summary(fit)$summary[, "Rhat"]
  cat("Max Rhat:", max(rhats), "\n")
  cat("Any Rhat > 1.01:", any(rhats > 1.01), "\n")

  # Effective sample size
  ess_bulk <- summary(fit)$summary[, "n_eff"]
  cat("Min ESS:", min(ess_bulk), "\n")

  # Divergences
  sampler_params <- get_sampler_params(fit, inc_warmup = FALSE)
  divergences <- sum(sapply(sampler_params, function(x) sum(x[, "divergent__"])))
  cat("Divergent transitions:", divergences, "\n")

  # Overall assessment
  if (max(rhats) < 1.01 && min(ess_bulk) > 400 && divergences < 5) {
    cat("✓ ALL DIAGNOSTICS PASS\n")
  } else {
    cat("✗ DIAGNOSTICS FAILED - INVESTIGATE\n")
  }
}

# Usage
quick_check(fit1, "Hierarchical")
quick_check(fit2, "Beta-binomial")
quick_check(fit3, "Robust")
```

### Python Functions
```python
def quick_check(trace, model_name):
    """Quick diagnostic check"""
    print(f"\n=== Diagnostics for {model_name} ===")

    # Convergence
    rhats = az.rhat(trace)
    max_rhat = float(rhats.max().values)
    print(f"Max Rhat: {max_rhat:.4f}")
    print(f"Any Rhat > 1.01: {max_rhat > 1.01}")

    # Effective sample size
    ess = az.ess(trace)
    min_ess = float(ess.min().values)
    print(f"Min ESS: {min_ess:.0f}")

    # Divergences
    divergences = trace.sample_stats.diverging.sum().values
    print(f"Divergent transitions: {divergences}")

    # Overall
    if max_rhat < 1.01 and min_ess > 400 and divergences < 5:
        print("✓ ALL DIAGNOSTICS PASS")
    else:
        print("✗ DIAGNOSTICS FAILED - INVESTIGATE")

# Usage
quick_check(trace1, "Hierarchical")
quick_check(trace2, "Beta-binomial")
quick_check(trace3, "Robust")
```

---

## Troubleshooting Guide

### Issue: Divergent Transitions

**R Solution**:
```r
# Increase adapt_delta
fit1_tuned <- stan(
  file = "hierarchical.stan",
  data = stan_data,
  chains = 4,
  iter = 2000,
  cores = 4,
  control = list(adapt_delta = 0.95)  # Increase from 0.8
)
```

**Python Solution**:
```python
with model1:
    trace1_tuned = pm.sample(
        1000, tune=1000, chains=4, cores=4,
        target_accept=0.95  # Increase from 0.8
    )
```

### Issue: Low ESS for tau

**Cause**: Tau is near zero (groups very similar)
**Solution**: Use tighter prior or accept lower ESS if Rhat is fine

```r
# Tighter prior
# In Stan file, change:
# tau ~ cauchy(0, 1);
# to:
# tau ~ cauchy(0, 0.5);
```

### Issue: Pareto k > 0.7

**Solution**: Use robust model (Model 3) or check those specific groups

```r
# Identify problematic groups
pareto_k <- loo1$diagnostics$pareto_k
problem_groups <- which(pareto_k > 0.7)
cat("Problematic groups:", problem_groups, "\n")

# Check their data
data[problem_groups, ]
```

---

## Final Checklist

Before reporting results:

- [ ] All Rhat < 1.01
- [ ] ESS > 400 for key parameters (mu, tau)
- [ ] Divergences < 1% of post-warmup iterations
- [ ] Trace plots show good mixing
- [ ] Pairs plots show no pathological correlations
- [ ] LOO Pareto k < 0.7 for all groups
- [ ] Posterior predictive checks pass
- [ ] Parameter estimates scientifically plausible
- [ ] Shrinkage follows expected pattern
- [ ] Results stable across different random seeds

---

## Output Files to Save

1. `model_fits.rds` or `traces.nc` - Posterior samples
2. `loo_comparison.csv` - Model comparison results
3. `parameter_estimates.csv` - Point estimates and intervals
4. `shrinkage_plot.png` - Visualization of partial pooling
5. `trace_plots.png` - MCMC diagnostics
6. `ppc_plots.png` - Posterior predictive checks
7. `summary_report.md` - Written interpretation

---

**This implementation guide provides everything needed to go from data to results in <2 hours.**
