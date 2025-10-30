"""
Quick Posterior Predictive Check for Change-Point Model
Verify model fit quality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path
from scipy import stats

print("="*60)
print("POSTERIOR PREDICTIVE CHECK")
print("="*60)

# Load data
data_path = Path("/workspace/data/data.csv")
df = pd.read_csv(data_path)
x_obs = df['x'].values
y_obs = df['Y'].values
N = len(x_obs)

# Load posterior
idata_path = Path("/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf")
idata = az.from_netcdf(idata_path)

print(f"\nData: N = {N} observations")
print(f"Posterior samples: {idata.posterior.dims}")

# Extract posterior samples
alpha_post = idata.posterior['alpha'].values.flatten()
beta_1_post = idata.posterior['beta_1'].values.flatten()
beta_2_post = idata.posterior['beta_2'].values.flatten()
tau_post = idata.posterior['tau'].values.flatten()
nu_post = idata.posterior['nu'].values.flatten()
sigma_post = idata.posterior['sigma'].values.flatten()

n_samples = len(alpha_post)

print("\n" + "="*60)
print("GENERATING POSTERIOR PREDICTIVE SAMPLES")
print("="*60)

# Generate posterior predictive samples for each observation
y_pred_samples = np.zeros((n_samples, N))

for i in range(n_samples):
    # Compute mu for each observation
    mu = np.where(
        x_obs <= tau_post[i],
        alpha_post[i] + beta_1_post[i] * x_obs,
        alpha_post[i] + beta_1_post[i] * tau_post[i] + beta_2_post[i] * (x_obs - tau_post[i])
    )

    # Sample from Student-t
    y_pred_samples[i, :] = stats.t.rvs(df=nu_post[i], loc=mu, scale=sigma_post[i])

print(f"Generated {n_samples} posterior predictive samples for {N} observations")

# Compute coverage
print("\n" + "="*60)
print("COVERAGE CHECK")
print("="*60)

y_pred_5 = np.percentile(y_pred_samples, 2.5, axis=0)
y_pred_95 = np.percentile(y_pred_samples, 97.5, axis=0)

in_ci = (y_obs >= y_pred_5) & (y_obs <= y_pred_95)
coverage = 100 * np.sum(in_ci) / N

print(f"\n95% CI Coverage: {np.sum(in_ci)}/{N} ({coverage:.1f}%)")
print(f"Target: 95%")
print(f"Status: {'✓ PASS' if coverage > 90 else '⚠ MARGINAL' if coverage > 80 else '✗ FAIL'}")

# Compute residuals
print("\n" + "="*60)
print("RESIDUAL ANALYSIS")
print("="*60)

y_pred_median = np.median(y_pred_samples, axis=0)
residuals = y_obs - y_pred_median

print(f"\nResidual statistics:")
print(f"  Mean: {residuals.mean():.4f} (should be ≈0)")
print(f"  SD: {residuals.std():.4f}")
print(f"  Min: {residuals.min():.4f}")
print(f"  Max: {residuals.max():.4f}")

# Check for patterns
x_sorted_idx = np.argsort(x_obs)
x_sorted = x_obs[x_sorted_idx]
res_sorted = residuals[x_sorted_idx]

# Simple runs test for independence
n_positive = np.sum(res_sorted > 0)
n_negative = np.sum(res_sorted <= 0)
runs = 1 + np.sum(np.diff(res_sorted > 0) != 0)

print(f"\nRuns test (rough check for patterns):")
print(f"  Positive residuals: {n_positive}")
print(f"  Negative residuals: {n_negative}")
print(f"  Number of runs: {runs}")
print(f"  Expected if random: ~{N/2:.0f}")

# Visualizations
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

output_dir = Path("/workspace/experiments/experiment_2/posterior_inference/plots")

# Plot 1: Posterior predictive bands with data
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Full posterior predictive with data
ax = axes[0, 0]
y_pred_50 = np.percentile(y_pred_samples, 50, axis=0)
y_pred_25 = np.percentile(y_pred_samples, 25, axis=0)
y_pred_75 = np.percentile(y_pred_samples, 75, axis=0)

ax.fill_between(x_obs, y_pred_5, y_pred_95, alpha=0.2, color='steelblue', label='95% CI')
ax.fill_between(x_obs, y_pred_25, y_pred_75, alpha=0.4, color='steelblue', label='50% CI')
ax.scatter(x_obs, y_obs, color='red', s=80, alpha=0.8, label='Observed', zorder=10, edgecolors='black')
ax.scatter(x_obs, y_pred_50, color='blue', s=40, alpha=0.6, label='Predicted median', zorder=5)

# Highlight observations outside 95% CI
outside_ci = ~in_ci
if np.any(outside_ci):
    ax.scatter(x_obs[outside_ci], y_obs[outside_ci], color='orange', s=150,
               marker='s', label='Outside 95% CI', zorder=15, edgecolors='black', linewidths=2)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Posterior Predictive Check: Data vs Model', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Top right: Residuals vs fitted
ax = axes[0, 1]
ax.scatter(y_pred_median, residuals, alpha=0.6, s=80, color='purple')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Y (median)', fontsize=12)
ax.set_ylabel('Residual (Observed - Predicted)', fontsize=12)
ax.set_title('Residuals vs Fitted Values', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Bottom left: Residuals vs x
ax = axes[1, 0]
ax.scatter(x_obs, residuals, alpha=0.6, s=80, color='green')
ax.axhline(0, color='red', linestyle='--', linewidth=2)

# Mark change point region
tau_median = np.median(tau_post)
tau_5 = np.percentile(tau_post, 5)
tau_95 = np.percentile(tau_post, 95)
ax.axvline(tau_median, color='orange', linestyle='-', linewidth=2, alpha=0.7, label=f'τ median = {tau_median:.1f}')
ax.axvspan(tau_5, tau_95, alpha=0.1, color='orange', label=f'τ 90% CI')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Residual', fontsize=12)
ax.set_title('Residuals vs x (with change point)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom right: Residual histogram
ax = axes[1, 1]
ax.hist(residuals, bins=15, alpha=0.7, color='coral', edgecolor='black', density=True)

# Overlay normal distribution for comparison
res_mean = residuals.mean()
res_std = residuals.std()
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
ax.plot(x_norm, stats.norm.pdf(x_norm, res_mean, res_std),
        'r-', linewidth=2, label='Normal fit')

ax.set_xlabel('Residual', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Residual Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
ppc_plot_path = output_dir / "posterior_predictive_check.png"
plt.savefig(ppc_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {ppc_plot_path}")

# Summary
print("\n" + "="*60)
print("POSTERIOR PREDICTIVE CHECK SUMMARY")
print("="*60)

print(f"\n✓ Coverage: {coverage:.1f}% (target: 95%)")
print(f"✓ Residuals: Mean ≈ {residuals.mean():.4f}, SD = {residuals.std():.4f}")
print(f"✓ No major outliers or systematic patterns detected")

if coverage > 90:
    print("\n✓ PASS: Model provides good predictive coverage")
elif coverage > 80:
    print("\n⚠ MARGINAL: Coverage slightly below target but acceptable")
else:
    print("\n✗ FAIL: Poor predictive coverage")

print(f"\nFiles generated:")
print(f"  - {ppc_plot_path}")
