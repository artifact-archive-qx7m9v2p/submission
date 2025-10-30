"""
Complete the remaining plots for Experiment 1 posterior inference
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
OUTPUT_DIR = "/workspace/experiments/experiment_1/posterior_inference"
DIAGNOSTICS_DIR = f"{OUTPUT_DIR}/diagnostics"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"
DATA_PATH = "/workspace/data/data.csv"

print("Loading inference data and creating remaining plots...")

# Load inference data
trace = az.from_netcdf(f"{DIAGNOSTICS_DIR}/posterior_inference.netcdf")

# Load data
data = pd.read_csv(DATA_PATH)
year = data['year'].values
C = data['C'].values

# Extract posterior samples
posterior = trace.posterior
beta_0_samples = posterior['beta_0'].values.flatten()
beta_1_samples = posterior['beta_1'].values.flatten()
beta_2_samples = posterior['beta_2'].values.flatten()
phi_samples = posterior['phi'].values.flatten()

# 8.2 Posterior distributions (fixed)
print("   - Posterior distributions...")
fig = az.plot_posterior(trace, var_names=['beta_0', 'beta_1', 'beta_2', 'phi'],
                        hdi_prob=0.95, figsize=(12, 10))
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/posterior_distributions.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"     Saved: {PLOTS_DIR}/posterior_distributions.png")

# 8.3 Rank plots for convergence
print("   - Rank plots...")
az.plot_rank(trace, var_names=['beta_0', 'beta_1', 'beta_2', 'phi'])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/rank_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"     Saved: {PLOTS_DIR}/rank_plots.png")

# 8.4 Fitted trend
print("   - Fitted trend with credible intervals...")

# Generate posterior predictive samples for visualization
n_samples = 500
sample_indices = np.random.choice(len(beta_0_samples), size=n_samples, replace=False)

# Create year grid for smooth curve
year_grid = np.linspace(year.min(), year.max(), 200)
mu_samples = np.zeros((n_samples, len(year_grid)))

for i, idx in enumerate(sample_indices):
    log_mu_i = beta_0_samples[idx] + beta_1_samples[idx] * year_grid + beta_2_samples[idx] * year_grid**2
    mu_samples[i, :] = np.exp(log_mu_i)

# Compute credible intervals
mu_median = np.median(mu_samples, axis=0)
mu_50_lower = np.percentile(mu_samples, 25, axis=0)
mu_50_upper = np.percentile(mu_samples, 75, axis=0)
mu_95_lower = np.percentile(mu_samples, 2.5, axis=0)
mu_95_upper = np.percentile(mu_samples, 97.5, axis=0)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(year, C, alpha=0.6, s=50, label='Observed data', zorder=3)
ax.plot(year_grid, mu_median, 'r-', linewidth=2, label='Posterior median', zorder=4)
ax.fill_between(year_grid, mu_50_lower, mu_50_upper, alpha=0.3, color='red',
                label='50% CI', zorder=1)
ax.fill_between(year_grid, mu_95_lower, mu_95_upper, alpha=0.2, color='red',
                label='95% CI', zorder=0)
ax.set_xlabel('Year (standardized)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Fitted Trend with Posterior Credible Intervals', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/fitted_trend.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"     Saved: {PLOTS_DIR}/fitted_trend.png")

# 8.5 Residual diagnostics
print("   - Residual diagnostics...")

# Compute residuals
beta_0_mean = beta_0_samples.mean()
beta_1_mean = beta_1_samples.mean()
beta_2_mean = beta_2_samples.mean()
log_mu_pred = beta_0_mean + beta_1_mean * year + beta_2_mean * year**2
mu_pred = np.exp(log_mu_pred)
residuals = C - mu_pred
residual_acf_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals over time
axes[0, 0].scatter(year, residuals, alpha=0.6)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Year (standardized)')
axes[0, 0].set_ylabel('Residual (Observed - Predicted)')
axes[0, 0].set_title('Residuals over Time')
axes[0, 0].grid(True, alpha=0.3)

# Residuals vs fitted
axes[0, 1].scatter(mu_pred, residuals, alpha=0.6)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Fitted values')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].set_title('Residuals vs Fitted')
axes[0, 1].grid(True, alpha=0.3)

# ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=20, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title(f'Residual ACF (lag-1 = {residual_acf_lag1:.3f})')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelation')

# QQ plot
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot of Residuals')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/residual_diagnostics.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"     Saved: {PLOTS_DIR}/residual_diagnostics.png")

print("\nAll plots completed successfully!")
