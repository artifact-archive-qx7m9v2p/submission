# Implementation Guide: Smooth Nonlinear Models

## Prerequisites

```python
# Required packages
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
from patsy import dmatrix

# OR for Stan
import cmdstanpy
import stan
```

---

## Workflow

### Step 1: Load Data and EDA Summary

```python
# Load data
data = pd.read_csv("/workspace/data/processed_data.csv")
year = data["year"].values
C = data["C"].values.astype(int)
N = len(C)

# EDA summary
print(f"Observations: {N}")
print(f"Count range: [{C.min()}, {C.max()}]")
print(f"Mean: {C.mean():.2f}, Std: {C.std():.2f}")
print(f"Variance/Mean: {C.var() / C.mean():.2f}")
print(f"Suspected break at observation: 17")
```

---

## Model 1: Polynomial Regression (Baseline)

### Implementation (PyMC)

```python
import pymc as pm
import arviz as az

with pm.Model() as poly_model:
    # Polynomial coefficients
    beta_0 = pm.Normal("beta_0", mu=4.3, sigma=0.5)
    beta_1 = pm.Normal("beta_1", mu=0.5, sigma=0.5)
    beta_2 = pm.Normal("beta_2", mu=0, sigma=0.5)
    beta_3 = pm.Normal("beta_3", mu=0, sigma=0.3)

    # AR(1) parameters
    rho = pm.Beta("rho", alpha=8, beta=2)
    sigma_nu = pm.HalfNormal("sigma_nu", sigma=0.2)

    # AR(1) process (innovation form)
    innovation = pm.Normal("innovation", mu=0, sigma=sigma_nu, shape=N)
    epsilon = pm.Deterministic("epsilon",
                               pm.math.concatenate([
                                   [innovation[0] / pm.math.sqrt(1 - rho**2)],
                                   [rho * epsilon[t-1] + innovation[t] for t in range(1, N)]
                               ]))

    # Polynomial trend
    mu_log = beta_0 + beta_1 * year + beta_2 * year**2 + beta_3 * year**3 + epsilon

    # Dispersion
    phi = pm.Gamma("phi", alpha=2, beta=1)

    # Likelihood
    C_obs = pm.NegativeBinomial("C_obs",
                                 mu=pm.math.exp(mu_log),
                                 alpha=phi,
                                 observed=C)

# Fit
with poly_model:
    trace_poly = pm.sample(2000, tune=1000, chains=4,
                           target_accept=0.95,
                           return_inferencedata=True)

# Diagnostics
print(az.summary(trace_poly, var_names=["beta_0", "beta_1", "beta_2", "beta_3", "rho", "phi"]))
print(f"Divergences: {trace_poly.sample_stats.diverging.sum().values}")

# LOO-CV
loo_poly = az.loo(trace_poly, pointwise=True)
print(f"LOO-ELPD: {loo_poly.loo:.2f} ± {loo_poly.loo_se:.2f}")
print(f"High Pareto-k (>0.7): {(loo_poly.pareto_k > 0.7).sum()}")
```

### Alternative: Stan Implementation

Save as `polynomial_negbin.stan`:

```stan
data {
  int<lower=1> N;
  array[N] int<lower=0> C;
  vector[N] year;
}

parameters {
  real beta_0;
  real beta_1;
  real beta_2;
  real beta_3;
  real<lower=0> phi;
  real<lower=0, upper=1> rho;
  real<lower=0> sigma_nu;
  vector[N] epsilon_raw;
}

transformed parameters {
  vector[N] epsilon;
  vector[N] mu_log;

  // AR(1) process (non-centered)
  epsilon[1] = epsilon_raw[1] * sigma_nu / sqrt(1 - rho^2);
  for (t in 2:N) {
    epsilon[t] = rho * epsilon[t-1] + epsilon_raw[t] * sigma_nu;
  }

  // Polynomial trend
  mu_log = beta_0 + beta_1 * year + beta_2 * square(year) +
           beta_3 * (year .* square(year)) + epsilon;
}

model {
  // Priors
  beta_0 ~ normal(4.3, 0.5);
  beta_1 ~ normal(0.5, 0.5);
  beta_2 ~ normal(0, 0.5);
  beta_3 ~ normal(0, 0.3);
  phi ~ gamma(2, 1);
  rho ~ beta(8, 2);
  sigma_nu ~ normal(0, 0.2);
  epsilon_raw ~ std_normal();

  // Likelihood
  C ~ neg_binomial_2_log(mu_log, phi);
}

generated quantities {
  vector[N] log_lik;
  array[N] int C_rep;
  vector[N] mu = exp(mu_log);

  for (n in 1:N) {
    log_lik[n] = neg_binomial_2_log_lpmf(C[n] | mu_log[n], phi);
    C_rep[n] = neg_binomial_2_log_rng(mu_log[n], phi);
  }
}
```

Run in Python:
```python
import cmdstanpy

# Compile
model = cmdstanpy.CmdStanModel(stan_file="polynomial_negbin.stan")

# Prepare data
stan_data = {"N": N, "C": C, "year": year}

# Fit
fit = model.sample(data=stan_data, chains=4, iter_sampling=2000,
                   iter_warmup=1000, adapt_delta=0.95)

# Diagnostics
print(fit.diagnose())
print(fit.summary())

# Convert to ArviZ for LOO
import arviz as az
trace_poly = az.from_cmdstanpy(fit)
loo_poly = az.loo(trace_poly)
```

---

## Model 2: Gaussian Process Regression

### Implementation (PyMC)

```python
import pymc as pm
import arviz as az

with pm.Model() as gp_model:
    # Mean function parameters
    beta_0 = pm.Normal("beta_0", mu=4.3, sigma=0.5)
    beta_1 = pm.Normal("beta_1", mu=0.5, sigma=0.5)

    # GP hyperparameters
    length_scale = pm.InverseGamma("length_scale", alpha=5, beta=5)
    sigma_f = pm.HalfNormal("sigma_f", sigma=0.5)

    # Mean function
    mean_func = beta_0 + beta_1 * year

    # GP covariance function
    cov_func = sigma_f**2 * pm.gp.cov.ExpQuad(1, ls=length_scale)

    # Latent GP
    gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
    f = gp.prior("f", X=year[:, None])

    # Dispersion
    phi = pm.Gamma("phi", alpha=2, beta=1)

    # Likelihood
    C_obs = pm.NegativeBinomial("C_obs",
                                 mu=pm.math.exp(f),
                                 alpha=phi,
                                 observed=C)

# Fit (may need more iterations)
with gp_model:
    trace_gp = pm.sample(3000, tune=2000, chains=4,
                         target_accept=0.99,
                         return_inferencedata=True)

# Diagnostics
print(az.summary(trace_gp, var_names=["beta_0", "beta_1", "length_scale", "sigma_f", "phi"]))
print(f"Divergences: {trace_gp.sample_stats.diverging.sum().values}")

# LOO-CV
loo_gp = az.loo(trace_gp, pointwise=True)
print(f"LOO-ELPD: {loo_gp.loo:.2f} ± {loo_gp.loo_se:.2f}")

# Check lengthscale
ls_samples = trace_gp.posterior["length_scale"].values.flatten()
print(f"Lengthscale: {ls_samples.mean():.3f} ± {ls_samples.std():.3f}")
if ls_samples.mean() < 0.2:
    print("WARNING: Lengthscale very small - may indicate discrete break")
```

### GP with AR(1) Observation Noise (Advanced)

```python
# If GP alone insufficient, add AR(1) term
with pm.Model() as gp_ar_model:
    # Mean function
    beta_0 = pm.Normal("beta_0", mu=4.3, sigma=0.5)
    beta_1 = pm.Normal("beta_1", mu=0.5, sigma=0.5)
    mean_func = beta_0 + beta_1 * year

    # GP
    length_scale = pm.InverseGamma("length_scale", alpha=5, beta=5)
    sigma_f = pm.HalfNormal("sigma_f", sigma=0.5)
    cov_func = sigma_f**2 * pm.gp.cov.ExpQuad(1, ls=length_scale)
    gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
    f_gp = gp.prior("f_gp", X=year[:, None])

    # AR(1) noise
    rho = pm.Beta("rho", alpha=8, beta=2)
    sigma_nu = pm.HalfNormal("sigma_nu", sigma=0.2)
    innovation = pm.Normal("innovation", mu=0, sigma=sigma_nu, shape=N)
    epsilon = pm.Deterministic("epsilon",
                               pm.math.concatenate([
                                   [innovation[0] / pm.math.sqrt(1 - rho**2)],
                                   [rho * epsilon[t-1] + innovation[t] for t in range(1, N)]
                               ]))

    # Combined
    f = f_gp + epsilon

    # Likelihood
    phi = pm.Gamma("phi", alpha=2, beta=1)
    C_obs = pm.NegativeBinomial("C_obs", mu=pm.math.exp(f), alpha=phi, observed=C)
```

---

## Model 3: Penalized B-Spline Regression

### Implementation (PyMC)

```python
from patsy import dmatrix

# Build B-spline basis
K = 10  # number of basis functions
degree = 3  # cubic splines

# Create interior knots
n_interior_knots = K - degree - 1
knots = np.quantile(year, np.linspace(0, 1, n_interior_knots + 2)[1:-1])

# Build basis matrix
B = dmatrix(
    f"bs(year, knots={list(knots)}, degree={degree}, include_intercept=True) - 1",
    {"year": year},
    return_type="dataframe"
).values

print(f"B-spline basis shape: {B.shape}")

# Build second-difference penalty matrix
D = np.diff(np.eye(K), n=2, axis=0)
D_product = D.T @ D

with pm.Model() as spline_model:
    # Smoothing parameter
    tau = pm.HalfNormal("tau", sigma=1)  # precision parameter

    # Spline coefficients with penalty
    # Penalized prior: exp(-0.5 * tau * beta' D'D beta)
    beta = pm.MvNormal("beta", mu=np.zeros(K), tau=tau * D_product, shape=K)

    # AR(1) noise
    rho = pm.Beta("rho", alpha=8, beta=2)
    sigma_nu = pm.HalfNormal("sigma_nu", sigma=0.2)
    innovation = pm.Normal("innovation", mu=0, sigma=sigma_nu, shape=N)
    epsilon = pm.Deterministic("epsilon",
                               pm.math.concatenate([
                                   [innovation[0] / pm.math.sqrt(1 - rho**2)],
                                   [rho * epsilon[t-1] + innovation[t] for t in range(1, N)]
                               ]))

    # Spline function
    f = pm.math.dot(B, beta) + epsilon

    # Dispersion
    phi = pm.Gamma("phi", alpha=2, beta=1)

    # Likelihood
    C_obs = pm.NegativeBinomial("C_obs", mu=pm.math.exp(f), alpha=phi, observed=C)

# Fit
with spline_model:
    trace_spline = pm.sample(2000, tune=1000, chains=4,
                             target_accept=0.95,
                             return_inferencedata=True)

# Diagnostics
print(az.summary(trace_spline, var_names=["tau", "rho", "phi"]))
print(f"Divergences: {trace_spline.sample_stats.diverging.sum().values}")

# LOO-CV
loo_spline = az.loo(trace_spline, pointwise=True)
print(f"LOO-ELPD: {loo_spline.loo:.2f} ± {loo_spline.loo_se:.2f}")
```

### Sensitivity Analysis: Different Number of Knots

```python
for K_test in [8, 10, 12]:
    print(f"\n=== Testing K={K_test} ===")

    # Rebuild basis
    n_interior_knots = K_test - degree - 1
    knots = np.quantile(year, np.linspace(0, 1, n_interior_knots + 2)[1:-1])
    B_test = dmatrix(
        f"bs(year, knots={list(knots)}, degree={degree}, include_intercept=True) - 1",
        {"year": year}
    ).values

    # Refit model (code above with B_test, K_test)
    # ... fit ...

    # Compare LOO
    loo_test = az.loo(trace_test)
    print(f"LOO-ELPD: {loo_test.loo:.2f}")
```

---

## Diagnostics and Comparison

### Convergence Diagnostics

```python
import arviz as az

def check_convergence(trace, model_name):
    """Check convergence diagnostics"""
    print(f"\n=== {model_name} Convergence ===")

    # R-hat
    rhat = az.rhat(trace)
    max_rhat = rhat.max()
    print(f"Max R-hat: {max_rhat:.4f}")
    if max_rhat > 1.05:
        print("WARNING: R-hat > 1.05, chains not converged")

    # Effective sample size
    ess = az.ess(trace)
    min_ess = ess.min()
    print(f"Min ESS: {min_ess:.0f}")
    if min_ess < 400:
        print("WARNING: ESS < 400, need more samples")

    # Divergences
    if hasattr(trace.sample_stats, 'diverging'):
        n_diverge = trace.sample_stats.diverging.sum().values
        print(f"Divergences: {n_diverge}")
        if n_diverge > 0:
            print("WARNING: Divergences detected, increase target_accept")

    # Energy diagnostic
    if hasattr(trace.sample_stats, 'energy'):
        energy = az.bfmi(trace)
        print(f"BFMI: {energy:.4f}")
        if energy < 0.2:
            print("WARNING: BFMI < 0.2, poor energy transitions")

# Check all models
check_convergence(trace_poly, "Polynomial")
check_convergence(trace_gp, "Gaussian Process")
check_convergence(trace_spline, "B-Spline")
```

### LOO Comparison

```python
# Compare all models
comparison = az.compare({
    "Polynomial": trace_poly,
    "GP": trace_gp,
    "Spline": trace_spline
}, ic="loo")

print("\n=== Model Comparison (LOO) ===")
print(comparison)

# Interpretation
best_model = comparison.index[0]
loo_diff = comparison.loc[comparison.index[1], "loo"]
loo_se = comparison.loc[comparison.index[1], "se"]

print(f"\nBest model: {best_model}")
print(f"Difference to 2nd: {-loo_diff:.2f} ± {loo_se:.2f}")

if -loo_diff > 20:
    print("STRONG evidence for best model")
elif -loo_diff > 10:
    print("Moderate evidence for best model")
else:
    print("Weak evidence, models similar")

# Check Pareto-k diagnostics
for name, trace in [("Polynomial", trace_poly), ("GP", trace_gp), ("Spline", trace_spline)]:
    loo_result = az.loo(trace, pointwise=True)
    high_pk = (loo_result.pareto_k > 0.7).sum()
    print(f"{name}: {high_pk} observations with Pareto-k > 0.7")
```

### Residual Analysis

```python
def residual_diagnostics(trace, year, C, model_name):
    """Check residuals for autocorrelation and patterns"""
    print(f"\n=== {model_name} Residuals ===")

    # Extract fitted values
    if "mu_log" in trace.posterior:
        mu_log_samples = trace.posterior["mu_log"].values
    elif "f" in trace.posterior:
        mu_log_samples = trace.posterior["f"].values
    else:
        raise ValueError("Cannot find log-mean parameter")

    mu_samples = np.exp(mu_log_samples)
    mu_mean = mu_samples.mean(axis=(0, 1))

    # Pearson residuals
    phi_mean = trace.posterior["phi"].values.mean()
    var_pred = mu_mean + mu_mean**2 / phi_mean
    residuals = (C - mu_mean) / np.sqrt(var_pred)

    # ACF
    from statsmodels.tsa.stattools import acf
    acf_vals = acf(residuals, nlags=10, fft=False)
    print(f"ACF(1): {acf_vals[1]:.3f}")
    print(f"ACF(2): {acf_vals[2]:.3f}")

    if abs(acf_vals[1]) > 0.3:
        print("WARNING: Significant autocorrelation at lag 1")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Residuals vs time
    axes[0].scatter(year, residuals)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Pearson Residuals")
    axes[0].set_title(f"{model_name}: Residuals vs Time")

    # ACF plot
    axes[1].bar(range(len(acf_vals)), acf_vals)
    axes[1].axhline(1.96/np.sqrt(len(C)), color='red', linestyle='--')
    axes[1].axhline(-1.96/np.sqrt(len(C)), color='red', linestyle='--')
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("ACF")
    axes[1].set_title("Autocorrelation Function")

    # QQ plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot")

    plt.tight_layout()
    plt.savefig(f"/workspace/experiments/designer_2/{model_name}_residuals.png", dpi=150)
    plt.close()

    return residuals, acf_vals

# Run for all models
res_poly, acf_poly = residual_diagnostics(trace_poly, year, C, "Polynomial")
res_gp, acf_gp = residual_diagnostics(trace_gp, year, C, "GP")
res_spline, acf_spline = residual_diagnostics(trace_spline, year, C, "Spline")
```

### First Derivative Test (Discontinuity Check)

```python
def check_derivative_continuity(trace, year, model_name):
    """Check if first derivative is continuous (smooth) or discontinuous (break)"""
    print(f"\n=== {model_name} Derivative Analysis ===")

    # Extract log-mean function
    if "mu_log" in trace.posterior:
        f_samples = trace.posterior["mu_log"].values
    elif "f" in trace.posterior:
        f_samples = trace.posterior["f"].values
    else:
        raise ValueError("Cannot find log-mean")

    # Compute first derivative (finite differences)
    df_dt = np.diff(f_samples, axis=-1) / np.diff(year)

    # Summary statistics
    df_mean = df_dt.mean(axis=(0, 1))
    df_lower = np.percentile(df_dt, 2.5, axis=(0, 1))
    df_upper = np.percentile(df_dt, 97.5, axis=(0, 1))

    # Check for discontinuity at observation 17
    year_deriv = (year[:-1] + year[1:]) / 2
    idx_break = 16  # between obs 17 and 18

    jump = abs(df_mean[idx_break] - df_mean[idx_break-1])
    print(f"Derivative change at suspected break: {jump:.3f}")

    if jump > 1.0:
        print("WARNING: Large derivative change (>1.0) suggests discontinuity")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(year_deriv, df_mean, label='Mean', linewidth=2)
    plt.fill_between(year_deriv, df_lower, df_upper, alpha=0.3, label='95% CI')
    plt.axvline(year[17], color='red', linestyle='--', label='Suspected break')
    plt.xlabel("Year")
    plt.ylabel("d(log μ)/dt")
    plt.title(f"{model_name}: First Derivative (Growth Rate)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"/workspace/experiments/designer_2/{model_name}_derivative.png", dpi=150)
    plt.close()

    return df_mean, jump

# Run for all models
deriv_poly, jump_poly = check_derivative_continuity(trace_poly, year, "Polynomial")
deriv_gp, jump_gp = check_derivative_continuity(trace_gp, year, "GP")
deriv_spline, jump_spline = check_derivative_continuity(trace_spline, year, "Spline")

# Summary
print("\n=== Derivative Jump Summary ===")
print(f"Polynomial: {jump_poly:.3f}")
print(f"GP: {jump_gp:.3f}")
print(f"Spline: {jump_spline:.3f}")
```

### Leave-Future-Out Cross-Validation

```python
def leave_future_out_cv(model_func, year, C, n_holdout=5):
    """Hold out last n observations and test prediction"""
    n_train = len(C) - n_holdout

    year_train = year[:n_train]
    C_train = C[:n_train]
    year_test = year[n_train:]
    C_test = C[n_train:]

    # Fit on training data
    trace_train = model_func(year_train, C_train)

    # Predict on test data
    # (model-specific prediction code needed)

    # Compute RMSE on log scale
    # ...

    return rmse_log

# Example for polynomial
def fit_polynomial_model(year_data, C_data):
    with pm.Model() as model:
        # ... same as above but with year_data, C_data ...
        trace = pm.sample(...)
    return trace

rmse_poly_lfo = leave_future_out_cv(fit_polynomial_model, year, C)
print(f"Polynomial LFO-CV RMSE (log): {rmse_poly_lfo:.3f}")
```

---

## Final Decision Framework

```python
def make_final_decision(loo_poly, loo_gp, loo_spline,
                       loo_changepoint,  # from Designer 1
                       jump_poly, jump_gp, jump_spline,
                       acf_poly, acf_gp, acf_spline):
    """Systematic decision based on all evidence"""

    print("\n" + "="*60)
    print("FINAL DECISION")
    print("="*60)

    # Best smooth model
    smooth_loos = {"Polynomial": loo_poly.loo, "GP": loo_gp.loo, "Spline": loo_spline.loo}
    best_smooth = max(smooth_loos, key=smooth_loos.get)
    best_smooth_loo = smooth_loos[best_smooth]

    print(f"\nBest smooth model: {best_smooth}")
    print(f"LOO-ELPD: {best_smooth_loo:.2f}")

    # Compare to changepoint
    delta_loo = best_smooth_loo - loo_changepoint.loo
    delta_se = np.sqrt(loo_poly.loo_se**2 + loo_changepoint.loo_se**2)

    print(f"\nComparison to changepoint model:")
    print(f"ΔLOO: {delta_loo:.2f} ± {delta_se:.2f}")

    # Decision logic
    if delta_loo < -20:
        decision = "REJECT smooth models - discrete break is real"
        print(f"\n{decision}")
        print("Evidence: LOO strongly favors changepoint model")
        recommend = "Use changepoint model from Designer 1"

    elif delta_loo < -10:
        # Check other evidence
        if max(jump_poly, jump_gp, jump_spline) > 1.0:
            decision = "REJECT smooth models - derivative discontinuity"
            recommend = "Use changepoint model"
        else:
            decision = "BORDERLINE - weak evidence for changepoint"
            recommend = "Use changepoint but acknowledge uncertainty"

    else:
        # Smooth models competitive
        if max(acf_poly[1], acf_gp[1], acf_spline[1]) > 0.3:
            decision = "ACCEPT smooth models but autocorrelation concerns"
            recommend = f"Use {best_smooth} but improve AR structure"
        else:
            decision = "ACCEPT smooth models - smooth acceleration sufficient"
            recommend = f"Use {best_smooth}"

    print(f"\nRecommendation: {recommend}")

    return decision, recommend

# Run decision
decision, recommendation = make_final_decision(
    loo_poly, loo_gp, loo_spline, loo_changepoint,
    jump_poly, jump_gp, jump_spline,
    acf_poly, acf_gp, acf_spline
)
```

---

## Expected Timeline

| Task | Time | Cumulative |
|------|------|------------|
| Load data, setup | 0.5h | 0.5h |
| Fit Polynomial | 1h | 1.5h |
| Diagnostics Polynomial | 0.5h | 2h |
| Fit GP | 2h | 4h |
| Diagnostics GP | 1h | 5h |
| Fit Spline | 1.5h | 6.5h |
| Diagnostics Spline | 1h | 7.5h |
| Model comparison | 1h | 8.5h |
| Derivative/residual analysis | 1.5h | 10h |
| LFO-CV | 1h | 11h |
| Final decision | 1h | 12h |
| Write report | 2h | 14h |

**Total: ~14 hours**

---

## Output Files

Save all results to `/workspace/experiments/designer_2/`:

1. `polynomial_trace.nc` - ArviZ InferenceData
2. `gp_trace.nc` - ArviZ InferenceData
3. `spline_trace.nc` - ArviZ InferenceData
4. `model_comparison.csv` - LOO comparison table
5. `Polynomial_residuals.png` - Diagnostic plots
6. `GP_residuals.png` - Diagnostic plots
7. `Spline_residuals.png` - Diagnostic plots
8. `Polynomial_derivative.png` - First derivative
9. `GP_derivative.png` - First derivative
10. `Spline_derivative.png` - First derivative
11. `final_decision.txt` - Decision summary

---

**File**: `/workspace/experiments/designer_2/implementation_guide.md`
