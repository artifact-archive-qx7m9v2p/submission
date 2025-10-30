# Stan Model Templates for Proposed Models

This document provides concrete Stan implementations for each proposed model.

---

## Model 1: Quadratic Negative Binomial with Time-Varying Dispersion

```stan
data {
  int<lower=0> N;              // Number of observations
  array[N] int<lower=0> C;     // Count outcomes
  vector[N] year;              // Standardized time variable
}

transformed data {
  vector[N] year_sq = year .* year;  // Precompute year²
}

parameters {
  // Mean function parameters
  real beta_0;                 // Intercept
  real beta_1;                 // Linear term
  real beta_2;                 // Quadratic term

  // Dispersion function parameters
  real gamma_0;                // Baseline log-dispersion
  real gamma_1;                // Time-varying log-dispersion
}

transformed parameters {
  vector[N] log_mu;
  vector[N] mu;
  vector[N] log_phi;
  vector[N] phi;

  // Mean function: log(μ) = β₀ + β₁×year + β₂×year²
  log_mu = beta_0 + beta_1 * year + beta_2 * year_sq;
  mu = exp(log_mu);

  // Dispersion function: log(φ) = γ₀ + γ₁×year
  log_phi = gamma_0 + gamma_1 * year;
  phi = exp(log_phi);
}

model {
  // Priors (informed by EDA)
  beta_0 ~ normal(4.3, 1.0);
  beta_1 ~ normal(0.85, 0.5);
  beta_2 ~ normal(0.3, 0.3);    // Positive acceleration expected

  gamma_0 ~ normal(0.4, 0.5);   // log(1.5) ≈ 0.4
  gamma_1 ~ normal(0, 0.3);

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  real var_mean_ratio;
  vector[N] residuals;

  // Log-likelihood for LOO-CV
  for (i in 1:N) {
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi[i]);
    C_rep[i] = neg_binomial_2_rng(mu[i], phi[i]);
    residuals[i] = C[i] - mu[i];
  }

  // Overdispersion check
  var_mean_ratio = variance(to_vector(C_rep)) / mean(to_vector(C_rep));
}
```

---

## Model 2: Piecewise Negative Binomial (Regime Shift)

```stan
data {
  int<lower=0> N;
  array[N] int<lower=0> C;
  vector[N] year;
  real tau;                    // Changepoint (fixed at -0.21 from EDA)
}

transformed data {
  vector[N] regime;            // Indicator: 1 if year > tau, 0 otherwise
  vector[N] year_post;         // (year - tau) × I(year > tau)

  for (i in 1:N) {
    regime[i] = year[i] > tau ? 1.0 : 0.0;
    year_post[i] = year[i] > tau ? year[i] - tau : 0.0;
  }
}

parameters {
  // Piecewise mean function
  real beta_0;                 // Pre-regime intercept
  real beta_1;                 // Pre-regime slope
  real beta_2;                 // Level shift at changepoint
  real beta_3;                 // Additional slope post-regime

  // Regime-specific dispersion
  real gamma_0;                // Pre-regime dispersion
  real gamma_1;                // Post-regime dispersion change
}

transformed parameters {
  vector[N] log_mu;
  vector[N] mu;
  vector[N] log_phi;
  vector[N] phi;

  // Piecewise mean: log(μ) = β₀ + β₁×year + β₂×I(year>τ) + β₃×(year-τ)×I(year>τ)
  log_mu = beta_0 + beta_1 * year + beta_2 * regime + beta_3 * year_post;
  mu = exp(log_mu);

  // Regime-specific dispersion
  log_phi = gamma_0 + gamma_1 * regime;
  phi = exp(log_phi);
}

model {
  // Priors
  beta_0 ~ normal(4.0, 1.0);
  beta_1 ~ normal(0.3, 0.3);
  beta_2 ~ normal(0, 0.5);
  beta_3 ~ normal(0.5, 0.5);    // Expect positive slope increase

  gamma_0 ~ normal(0.4, 0.5);
  gamma_1 ~ normal(0, 0.5);

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  real var_mean_ratio;
  real growth_rate_pre;        // Growth rate before regime
  real growth_rate_post;       // Growth rate after regime
  real acceleration_factor;    // How much growth accelerates

  for (i in 1:N) {
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi[i]);
    C_rep[i] = neg_binomial_2_rng(mu[i], phi[i]);
  }

  var_mean_ratio = variance(to_vector(C_rep)) / mean(to_vector(C_rep));

  // Interpretable growth rates
  growth_rate_pre = exp(beta_1) - 1;
  growth_rate_post = exp(beta_1 + beta_3) - 1;
  acceleration_factor = (beta_1 + beta_3) / beta_1;
}
```

---

## Model 3: Hierarchical B-Spline with Time-Varying Dispersion

```stan
functions {
  // B-spline basis function (degree 3, cubic)
  vector bspline_basis(real x, vector knots, int degree) {
    int K = num_elements(knots) - degree - 1;
    vector[K] basis;
    // Implementation of Cox-de Boor recursion
    // (Stan has built-in b-spline functions in newer versions)
    // For now, use simplified version or Stan's built-in
    return basis;
  }
}

data {
  int<lower=0> N;
  array[N] int<lower=0> C;
  vector[N] year;

  // B-spline configuration
  int<lower=1> K_mu;           // Number of mean spline basis functions (6)
  int<lower=1> K_phi;          // Number of dispersion spline basis (4)
  matrix[N, K_mu] B_mu;        // Pre-computed mean basis matrix
  matrix[N, K_phi] B_phi;      // Pre-computed dispersion basis matrix
}

parameters {
  // Spline coefficients with hierarchical shrinkage
  real alpha;                  // Overall mean intercept
  vector[K_mu] beta;           // Mean spline coefficients
  real<lower=0> sigma_beta;    // Adaptive shrinkage for mean

  real delta;                  // Overall dispersion level
  vector[K_phi] gamma;         // Dispersion spline coefficients
  real<lower=0> sigma_gamma;   // Adaptive shrinkage for dispersion
}

transformed parameters {
  vector[N] log_mu;
  vector[N] mu;
  vector[N] log_phi;
  vector[N] phi;

  // Spline-based mean function
  log_mu = alpha + B_mu * beta;
  mu = exp(log_mu);

  // Spline-based dispersion function
  log_phi = delta + B_phi * gamma;
  phi = exp(log_phi);
}

model {
  // Hierarchical priors (regularization)
  alpha ~ normal(4.3, 1.0);
  beta ~ normal(0, sigma_beta);
  sigma_beta ~ exponential(1);    // Adaptive shrinkage

  delta ~ normal(0.4, 0.5);
  gamma ~ normal(0, sigma_gamma);
  sigma_gamma ~ exponential(2);   // Stronger shrinkage for dispersion

  // Likelihood
  C ~ neg_binomial_2(mu, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  real var_mean_ratio;
  real effective_params_mean;     // Effective degrees of freedom
  real effective_params_phi;

  for (i in 1:N) {
    log_lik[i] = neg_binomial_2_lpmf(C[i] | mu[i], phi[i]);
    C_rep[i] = neg_binomial_2_rng(mu[i], phi[i]);
  }

  var_mean_ratio = variance(to_vector(C_rep)) / mean(to_vector(C_rep));

  // Estimate effective degrees of freedom (for complexity assessment)
  effective_params_mean = sum(square(beta) / (square(beta) + square(sigma_beta)));
  effective_params_phi = sum(square(gamma) / (square(gamma) + square(sigma_gamma)));
}
```

---

## Python Wrapper for CmdStanPy

```python
import cmdstanpy
import numpy as np
import arviz as az
from scipy.interpolate import BSpline

def fit_model_1(data_dict):
    """Fit quadratic model with time-varying dispersion."""
    model = cmdstanpy.CmdStanModel(stan_file='model_1_quadratic.stan')

    fit = model.sample(
        data=data_dict,
        chains=4,
        iter_warmup=2000,
        iter_sampling=2000,
        adapt_delta=0.95,
        max_treedepth=12,
        seed=42
    )

    return az.from_cmdstanpy(fit)

def fit_model_2(data_dict):
    """Fit piecewise model with regime shift."""
    data_dict['tau'] = -0.21  # Fixed changepoint from EDA

    model = cmdstanpy.CmdStanModel(stan_file='model_2_piecewise.stan')

    fit = model.sample(
        data=data_dict,
        chains=4,
        iter_warmup=2000,
        iter_sampling=2000,
        adapt_delta=0.95,
        max_treedepth=12,
        seed=42
    )

    return az.from_cmdstanpy(fit)

def fit_model_3(data_dict):
    """Fit hierarchical B-spline model."""
    # Construct B-spline basis matrices
    year = data_dict['year']

    # Mean function: 6 knots at quantiles
    knots_mu = np.quantile(year, [0.05, 0.25, 0.40, 0.60, 0.75, 0.95])
    B_mu = construct_bspline_basis(year, knots_mu, degree=3)

    # Dispersion function: 4 knots
    knots_phi = np.quantile(year, [0.25, 0.50, 0.75])
    B_phi = construct_bspline_basis(year, knots_phi, degree=3)

    data_dict['K_mu'] = B_mu.shape[1]
    data_dict['K_phi'] = B_phi.shape[1]
    data_dict['B_mu'] = B_mu
    data_dict['B_phi'] = B_phi

    model = cmdstanpy.CmdStanModel(stan_file='model_3_spline.stan')

    fit = model.sample(
        data=data_dict,
        chains=4,
        iter_warmup=2000,
        iter_sampling=2000,
        adapt_delta=0.95,
        max_treedepth=12,
        seed=42
    )

    return az.from_cmdstanpy(fit)

def construct_bspline_basis(x, knots, degree=3):
    """Construct B-spline basis matrix."""
    from scipy.interpolate import BSpline

    # Extend knots for boundary conditions
    knots_extended = np.concatenate([
        np.repeat(knots[0], degree),
        knots,
        np.repeat(knots[-1], degree)
    ])

    K = len(knots) + degree - 1
    basis = np.zeros((len(x), K))

    for i in range(K):
        coefs = np.zeros(K)
        coefs[i] = 1
        spline = BSpline(knots_extended, coefs, degree)
        basis[:, i] = spline(x)

    return basis

def compare_models(idata_dict):
    """Compare models using LOO-CV."""
    # Compute LOO for each model
    loo_dict = {name: az.loo(idata, pointwise=True)
                for name, idata in idata_dict.items()}

    # Compare
    comparison = az.compare(loo_dict)

    return comparison

# Example usage
if __name__ == "__main__":
    import json

    # Load data
    with open('data.json', 'r') as f:
        data = json.load(f)

    data_dict = {
        'N': len(data['C']),
        'C': data['C'],
        'year': data['year']
    }

    # Fit all models
    print("Fitting Model 1 (Quadratic)...")
    idata1 = fit_model_1(data_dict)

    print("Fitting Model 2 (Piecewise)...")
    idata2 = fit_model_2(data_dict)

    print("Fitting Model 3 (Spline)...")
    idata3 = fit_model_3(data_dict)

    # Compare
    print("\nModel Comparison:")
    comparison = compare_models({
        'Quadratic': idata1,
        'Piecewise': idata2,
        'Spline': idata3
    })
    print(comparison)
```

---

## Diagnostic Checklist

After fitting each model, run these checks:

```python
def run_diagnostics(idata, data, model_name):
    """Comprehensive diagnostic pipeline."""
    print(f"\n{'='*60}")
    print(f"Diagnostics for {model_name}")
    print('='*60)

    # 1. Convergence
    print("\n1. CONVERGENCE DIAGNOSTICS")
    rhat = az.rhat(idata)
    print(f"   Max R-hat: {rhat.max().values:.4f} (want < 1.01)")

    ess_bulk = az.ess(idata, method='bulk')
    ess_tail = az.ess(idata, method='tail')
    print(f"   Min ESS (bulk): {ess_bulk.min().values:.0f} (want > 400)")
    print(f"   Min ESS (tail): {ess_tail.min().values:.0f} (want > 400)")

    # 2. Posterior Predictive Checks
    print("\n2. POSTERIOR PREDICTIVE CHECKS")
    C_rep = idata.posterior_predictive['C_rep'].values
    C_obs = data['C']

    # Overdispersion
    var_mean_ratio_obs = np.var(C_obs) / np.mean(C_obs)
    var_mean_ratio_rep = np.mean(idata.posterior['var_mean_ratio'].values)
    print(f"   Var/Mean (observed): {var_mean_ratio_obs:.2f}")
    print(f"   Var/Mean (replicated): {var_mean_ratio_rep:.2f}")

    # Coverage
    lower = np.percentile(C_rep, 5, axis=(0,1))
    upper = np.percentile(C_rep, 95, axis=(0,1))
    coverage = np.mean((C_obs >= lower) & (C_obs <= upper))
    print(f"   90% interval coverage: {coverage:.2%} (want ~90%)")

    # 3. LOO-CV
    print("\n3. LOO-CV DIAGNOSTICS")
    loo = az.loo(idata, pointwise=True)
    print(f"   ELPD_loo: {loo.elpd_loo:.2f} ± {loo.se:.2f}")

    pareto_k = loo.pareto_k
    n_high_k = np.sum(pareto_k > 0.7)
    print(f"   High Pareto-k (>0.7): {n_high_k}/{len(pareto_k)} (want 0)")

    # 4. Parameter Summary
    print("\n4. PARAMETER ESTIMATES")
    print(az.summary(idata, var_names=['beta_0', 'beta_1', 'beta_2',
                                        'gamma_0', 'gamma_1']))

    return {
        'rhat_max': rhat.max().values,
        'ess_min': min(ess_bulk.min().values, ess_tail.min().values),
        'var_mean_ratio': var_mean_ratio_rep,
        'coverage': coverage,
        'elpd_loo': loo.elpd_loo,
        'n_high_pareto_k': n_high_k
    }
```

---

## Files

- Model templates: `/workspace/experiments/designer_2/stan_model_templates.md`
- Main proposal: `/workspace/experiments/designer_2/proposed_models.md`
