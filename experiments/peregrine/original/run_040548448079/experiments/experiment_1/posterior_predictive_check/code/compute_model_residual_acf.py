"""
Compute residual ACF for the fitted model.
"""

import numpy as np
import pandas as pd
import arviz as az

def acf_lag1(x):
    """Compute autocorrelation at lag 1."""
    x_centered = x - x.mean()
    c0 = np.dot(x_centered, x_centered) / len(x)
    c1 = np.dot(x_centered[:-1], x_centered[1:]) / len(x)
    return c1 / c0 if c0 > 0 else 0

# Load data
data = pd.read_csv('/workspace/data/data.csv')
year = data['year'].values
C = data['C'].values
N = len(C)
tau = 17
year_tau = year[tau-1]

# Load trace
trace = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

# Create post-break indicator
post_break = (np.arange(N) >= tau).astype(float)
year_post = post_break * (year - year_tau)

# Extract posterior samples
beta_0_samples = trace.posterior['beta_0'].values.flatten()
beta_1_samples = trace.posterior['beta_1'].values.flatten()
beta_2_samples = trace.posterior['beta_2'].values.flatten()
alpha_samples = trace.posterior['alpha'].values.flatten()

# Compute posterior mean mu
n_samples = len(beta_0_samples)
mu_samples = np.zeros((n_samples, N))

for i in range(n_samples):
    log_mu = beta_0_samples[i] + beta_1_samples[i] * year + beta_2_samples[i] * year_post
    mu_samples[i, :] = np.exp(log_mu)

mu_mean = mu_samples.mean(axis=0)
alpha_mean = alpha_samples.mean()

# Compute Pearson residuals
residuals_pearson = (C - mu_mean) / np.sqrt(mu_mean * (1 + mu_mean / alpha_mean))

# Compute ACF(1)
res_acf1 = acf_lag1(residuals_pearson)

print("="*60)
print("RESIDUAL AUTOCORRELATION ANALYSIS")
print("="*60)
print(f"\nRaw data ACF(1): 0.944 (from EDA)")
print(f"Residual ACF(1): {res_acf1:.4f}")
print(f"Reduction: {100*(1 - res_acf1/0.944):.1f}%")
print("")

if res_acf1 < 0.3:
    print("Status: EXCELLENT - Minimal autocorrelation")
elif res_acf1 < 0.5:
    print("Status: ACCEPTABLE - Some autocorrelation but manageable")
else:
    print("Status: CONCERNING - Strong autocorrelation remains")

print("\nInterpretation:")
print("  Model captures structural break (regime change)")
print("  BUT does not fully capture temporal dependencies")
print("  AR(1) terms would be needed for complete temporal structure")
print("="*60)

# Save for report
with open('/workspace/experiments/experiment_1/posterior_predictive_check/code/residual_acf.txt', 'w') as f:
    f.write(f"Residual ACF(1): {res_acf1:.4f}\n")
