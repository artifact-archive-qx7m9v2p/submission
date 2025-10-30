"""
LOO-PIT Analysis for Fixed-Effect Normal Model
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import xarray as xr

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Paths
DATA_PATH = "/workspace/data/data.csv"
IDATA_PATH = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
PLOTS_DIR = Path("/workspace/experiments/experiment_1/posterior_predictive_check/plots")

# Load data
print("Loading data and posterior samples...")
data = pd.read_csv(DATA_PATH)
y_obs = data['y'].values
sigma = data['sigma'].values
n_obs = len(y_obs)

# Load InferenceData
idata = az.from_netcdf(IDATA_PATH)
print(f"Observed data variables: {list(idata.observed_data.data_vars)}")

# Extract posterior samples of theta
theta_samples = idata.posterior['theta'].values.flatten()
n_samples = len(theta_samples)

# Generate posterior predictive samples
print("Generating posterior predictive samples...")
np.random.seed(42)
y_rep = np.zeros((n_samples, n_obs))

for s in range(n_samples):
    for i in range(n_obs):
        y_rep[s, i] = np.random.normal(theta_samples[s], sigma[i])

# Reshape for ArviZ format
n_chains = len(idata.posterior.coords['chain'])
n_draws = len(idata.posterior.coords['draw'])
y_rep_reshaped = y_rep[:n_chains * n_draws, :].reshape(n_chains, n_draws, n_obs)

# Create posterior_predictive with matching name 'y_obs'
print("Creating InferenceData with posterior_predictive...")
posterior_predictive = xr.Dataset(
    {
        'y_obs': xr.DataArray(
            y_rep_reshaped,
            dims=['chain', 'draw', 'y_obs_dim_0'],
            coords={
                'chain': idata.posterior.coords['chain'],
                'draw': idata.posterior.coords['draw'],
                'y_obs_dim_0': np.arange(n_obs)
            }
        )
    }
)

# Create new idata with all needed groups
idata_ppc = az.InferenceData(
    posterior=idata.posterior,
    posterior_predictive=posterior_predictive,
    observed_data=idata.observed_data,
    log_likelihood=idata.log_likelihood,
    sample_stats=idata.sample_stats
)

# Compute LOO-PIT
print("\nComputing LOO-PIT...")
loo_pit = az.loo_pit(idata=idata_ppc, y='y_obs')

# Extract values correctly
if isinstance(loo_pit, np.ndarray):
    pit_values = loo_pit.flatten()
else:
    pit_values = loo_pit.values.flatten()

print(f"LOO-PIT values: {pit_values}")

# Plot LOO-PIT using ArviZ
print("\nPlotting LOO-PIT...")
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_loo_pit(idata=idata_ppc, y='y_obs', legend=True, ax=ax)
ax.set_title('LOO-PIT Uniformity Check\n(Should be uniform on [0,1] if well-calibrated)',
             fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "loo_pit.png", dpi=300, bbox_inches='tight')
print(f"Saved: loo_pit.png")
plt.close()

# Perform uniformity tests
ks_stat, ks_pval = stats.kstest(pit_values, 'uniform')

print(f"\nLOO-PIT Uniformity Test (Kolmogorov-Smirnov):")
print(f"  KS statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_pval:.4f}")
print(f"  Assessment: {'GOOD (uniform)' if ks_pval > 0.05 else 'POOR (non-uniform)'}")

# Additional LOO-PIT diagnostic plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(pit_values, bins=10, density=True, alpha=0.6, color='steelblue',
        edgecolor='black', linewidth=1.5, label='LOO-PIT')
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform')
ax.set_xlabel('LOO-PIT Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('LOO-PIT Histogram\n(Should be flat at 1.0)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Q-Q plot against uniform
ax = axes[1]
stats.probplot(pit_values, dist=stats.uniform, plot=ax)
ax.set_title('Q-Q Plot: LOO-PIT vs Uniform(0,1)', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "loo_pit_detailed.png", dpi=300, bbox_inches='tight')
print(f"Saved: loo_pit_detailed.png")
plt.close()

# Check for shape patterns
print("\nLOO-PIT Pattern Analysis:")
print(f"  Mean: {pit_values.mean():.3f} (should be ~0.5)")
print(f"  Std: {pit_values.std():.3f} (should be ~0.289 for uniform)")

# Count extreme values
n_extreme_low = (pit_values < 0.1).sum()
n_extreme_high = (pit_values > 0.9).sum()
print(f"  Extreme low (<0.1): {n_extreme_low}/{n_obs}")
print(f"  Extreme high (>0.9): {n_extreme_high}/{n_obs}")

# Save results
results = {
    'pit_values': pit_values,
    'ks_stat': ks_stat,
    'ks_pval': ks_pval
}
np.save(PLOTS_DIR.parent / "code" / "loo_pit_results.npy", results, allow_pickle=True)

print("\nLOO-PIT Analysis Complete!")
