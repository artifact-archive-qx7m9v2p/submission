"""
Compute residual autocorrelation for model assessment.
"""

import numpy as np
import pandas as pd
import arviz as az
from statsmodels.tsa.stattools import acf

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

# Compute ACF
acf_vals = acf(residuals_pearson, nlags=10, fft=False)

print("Residual Autocorrelation Analysis")
print("="*60)
print(f"ACF(1): {acf_vals[1]:.4f}")
print(f"ACF(2): {acf_vals[2]:.4f}")
print(f"ACF(3): {acf_vals[3]:.4f}")
print("")

if acf_vals[1] < 0.3:
    print("Status: EXCELLENT - Minimal autocorrelation")
elif acf_vals[1] < 0.5:
    print("Status: ACCEPTABLE - Some autocorrelation but manageable")
else:
    print("Status: CONCERNING - Strong autocorrelation, AR(1) needed")

print("\nInterpretation:")
print("  - EDA showed raw data ACF(1) = 0.944")
print(f"  - Model residuals ACF(1) = {acf_vals[1]:.4f}")
print(f"  - Reduction: {100*(1 - acf_vals[1]/0.944):.1f}%")
