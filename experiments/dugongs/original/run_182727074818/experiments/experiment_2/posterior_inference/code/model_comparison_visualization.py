"""
Visual Comparison: Model 1 (Log) vs Model 2 (Change-Point)
Side-by-side comparison of model fits
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pathlib import Path

print("="*60)
print("MODEL COMPARISON VISUALIZATION")
print("="*60)

# Load data
data_path = Path("/workspace/data/data.csv")
df = pd.read_csv(data_path)
x_obs = df['x'].values
y_obs = df['Y'].values
N = len(x_obs)

# Load both models
idata1 = az.from_netcdf("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
idata2 = az.from_netcdf("/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf")

print("\n✓ Loaded Model 1 (Logarithmic)")
print("✓ Loaded Model 2 (Change-Point)")

# Extract posteriors for Model 1
alpha1 = idata1.posterior['alpha'].values.flatten()
beta1 = idata1.posterior['beta'].values.flatten()
c1 = idata1.posterior['c'].values.flatten()

# Extract posteriors for Model 2
alpha2 = idata2.posterior['alpha'].values.flatten()
beta_1_2 = idata2.posterior['beta_1'].values.flatten()
beta_2_2 = idata2.posterior['beta_2'].values.flatten()
tau2 = idata2.posterior['tau'].values.flatten()

# Create prediction grid
x_pred = np.linspace(0.5, 35, 500)

print("\n" + "="*60)
print("COMPUTING POSTERIOR PREDICTIVES")
print("="*60)

# Model 1 predictions
n_samples1 = min(500, len(alpha1))
y_pred1_samples = []

for i in range(n_samples1):
    mu = alpha1[i] + beta1[i] * np.log(x_pred + c1[i])
    y_pred1_samples.append(mu)

y_pred1_samples = np.array(y_pred1_samples)
y_pred1_median = np.median(y_pred1_samples, axis=0)
y_pred1_5 = np.percentile(y_pred1_samples, 5, axis=0)
y_pred1_95 = np.percentile(y_pred1_samples, 95, axis=0)

print("✓ Model 1 predictions computed")

# Model 2 predictions
n_samples2 = min(500, len(alpha2))
y_pred2_samples = []

for i in range(n_samples2):
    mu = np.where(
        x_pred <= tau2[i],
        alpha2[i] + beta_1_2[i] * x_pred,
        alpha2[i] + beta_1_2[i] * tau2[i] + beta_2_2[i] * (x_pred - tau2[i])
    )
    y_pred2_samples.append(mu)

y_pred2_samples = np.array(y_pred2_samples)
y_pred2_median = np.median(y_pred2_samples, axis=0)
y_pred2_5 = np.percentile(y_pred2_samples, 5, axis=0)
y_pred2_95 = np.percentile(y_pred2_samples, 95, axis=0)

print("✓ Model 2 predictions computed")

# Create visualization
print("\n" + "="*60)
print("CREATING COMPARISON PLOT")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Model 1 fit
ax = axes[0, 0]
ax.fill_between(x_pred, y_pred1_5, y_pred1_95, alpha=0.3, color='steelblue', label='90% CI')
ax.plot(x_pred, y_pred1_median, color='darkblue', linewidth=2.5, label='Posterior median')
ax.scatter(x_obs, y_obs, color='red', s=100, alpha=0.8, label='Observed data',
           zorder=10, edgecolors='black', linewidths=1.5)
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('Y', fontsize=13)
ax.set_title('Model 1: Logarithmic Regression\nY = α + β·log(x + c)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 35)
ax.set_ylim(1.5, 3.0)

# Add ELPD annotation
ax.text(0.05, 0.95, 'ELPD_LOO = 23.71 ± 3.09\np_loo = 2.61',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Model 2 fit
ax = axes[0, 1]
ax.fill_between(x_pred, y_pred2_5, y_pred2_95, alpha=0.3, color='forestgreen', label='90% CI')
ax.plot(x_pred, y_pred2_median, color='darkgreen', linewidth=2.5, label='Posterior median')
ax.scatter(x_obs, y_obs, color='red', s=100, alpha=0.8, label='Observed data',
           zorder=10, edgecolors='black', linewidths=1.5)

# Mark change point
tau_median = np.median(tau2)
tau_5 = np.percentile(tau2, 5)
tau_95 = np.percentile(tau2, 95)
ax.axvline(tau_median, color='orange', linestyle='--', linewidth=2.5,
           label=f'Change point τ = {tau_median:.1f}')
ax.axvspan(tau_5, tau_95, alpha=0.15, color='orange')

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('Y', fontsize=13)
ax.set_title('Model 2: Change-Point Regression\nY = α + β₁·x (x≤τ), α + β₁·τ + β₂·(x-τ) (x>τ)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 35)
ax.set_ylim(1.5, 3.0)

# Add ELPD annotation
ax.text(0.05, 0.95, 'ELPD_LOO = 20.39 ± 3.35\np_loo = 4.62',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Plot 3: Overlay comparison
ax = axes[1, 0]
ax.plot(x_pred, y_pred1_median, color='darkblue', linewidth=2.5, label='Model 1 (Log)', linestyle='-')
ax.plot(x_pred, y_pred2_median, color='darkgreen', linewidth=2.5, label='Model 2 (Change-Point)', linestyle='--')
ax.scatter(x_obs, y_obs, color='red', s=100, alpha=0.8, label='Observed data',
           zorder=10, edgecolors='black', linewidths=1.5)

# Highlight difference region
diff = np.abs(y_pred1_median - y_pred2_median)
max_diff_idx = np.argmax(diff)
max_diff_x = x_pred[max_diff_idx]
ax.axvline(max_diff_x, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
ax.text(max_diff_x + 1, 1.6, f'Max diff at x={max_diff_x:.1f}',
        fontsize=9, rotation=0, color='gray')

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('Y', fontsize=13)
ax.set_title('Model Comparison: Overlaid Fits', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 35)
ax.set_ylim(1.5, 3.0)

# Plot 4: Residual comparison
ax = axes[1, 1]

# Compute residuals for both models at observed x
y_pred1_obs = []
y_pred2_obs = []

for i in range(n_samples1):
    mu1 = alpha1[i] + beta1[i] * np.log(x_obs + c1[i])
    y_pred1_obs.append(mu1)

for i in range(n_samples2):
    mu2 = np.where(
        x_obs <= tau2[i],
        alpha2[i] + beta_1_2[i] * x_obs,
        alpha2[i] + beta_1_2[i] * tau2[i] + beta_2_2[i] * (x_obs - tau2[i])
    )
    y_pred2_obs.append(mu2)

y_pred1_obs = np.array(y_pred1_obs)
y_pred2_obs = np.array(y_pred2_obs)

res1 = y_obs - np.median(y_pred1_obs, axis=0)
res2 = y_obs - np.median(y_pred2_obs, axis=0)

x_jitter1 = x_obs - 0.3
x_jitter2 = x_obs + 0.3

ax.scatter(x_jitter1, res1, alpha=0.7, s=80, color='blue', label='Model 1 residuals', marker='o')
ax.scatter(x_jitter2, res2, alpha=0.7, s=80, color='green', label='Model 2 residuals', marker='s')
ax.axhline(0, color='red', linestyle='--', linewidth=2)

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('Residual (Observed - Predicted)', fontsize=13)
ax.set_title('Residual Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add RMSE comparison
rmse1 = np.sqrt(np.mean(res1**2))
rmse2 = np.sqrt(np.mean(res2**2))
ax.text(0.05, 0.95, f'RMSE:\n  Model 1: {rmse1:.4f}\n  Model 2: {rmse2:.4f}',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
output_path = Path("/workspace/experiments/experiment_2/posterior_inference/plots/model_comparison_visual.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {output_path}")

# Summary statistics
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)

print(f"\nModel 1 (Logarithmic):")
print(f"  RMSE: {rmse1:.4f}")
print(f"  Mean residual: {res1.mean():.4f}")
print(f"  Residual SD: {res1.std():.4f}")

print(f"\nModel 2 (Change-Point):")
print(f"  RMSE: {rmse2:.4f}")
print(f"  Mean residual: {res2.mean():.4f}")
print(f"  Residual SD: {res2.std():.4f}")

print(f"\nΔELPD_LOO: -3.31 ± 3.35 (Model 2 worse)")
print(f"Winner: Model 1 (Logarithmic)")

print(f"\nVisualization saved: {output_path}")
