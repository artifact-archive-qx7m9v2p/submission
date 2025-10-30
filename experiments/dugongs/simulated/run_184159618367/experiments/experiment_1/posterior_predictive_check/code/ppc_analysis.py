"""
Posterior Predictive Check for Experiment 1: Asymptotic Exponential Model
Model: Y ~ Normal(α - β*exp(-γ*x), σ)

This script performs comprehensive PPC including:
- Visual diagnostics (overlay plots, residuals, Q-Q plots)
- Quantitative metrics (coverage, R², test statistics)
- Model adequacy assessment
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# File paths
DATA_PATH = '/workspace/data/data.csv'
POSTERIOR_PATH = '/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf'
PLOTS_DIR = '/workspace/experiments/experiment_1/posterior_predictive_check/plots/'

print("="*80)
print("POSTERIOR PREDICTIVE CHECK - ASYMPTOTIC EXPONENTIAL MODEL")
print("="*80)

# Load data
print("\n1. Loading observed data...")
data = pd.read_csv(DATA_PATH)
x_obs = data['x'].values
y_obs = data['Y'].values
n_obs = len(y_obs)
print(f"   Observations: {n_obs}")
print(f"   x range: [{x_obs.min():.2f}, {x_obs.max():.2f}]")
print(f"   y range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")

# Load posterior samples
print("\n2. Loading posterior inference data...")
idata = az.from_netcdf(POSTERIOR_PATH)
print(f"   Posterior draws: {idata.posterior.dims['draw']}")
print(f"   Chains: {idata.posterior.dims['chain']}")

# Extract parameter samples
alpha_samples = idata.posterior['alpha'].values.flatten()
beta_samples = idata.posterior['beta'].values.flatten()
gamma_samples = idata.posterior['gamma'].values.flatten()
sigma_samples = idata.posterior['sigma'].values.flatten()
n_samples = len(alpha_samples)
print(f"   Total samples: {n_samples}")

# Parameter summaries
print("\n3. Posterior parameter summaries:")
print(f"   alpha: {alpha_samples.mean():.3f} ± {alpha_samples.std():.3f}")
print(f"   beta:  {beta_samples.mean():.3f} ± {beta_samples.std():.3f}")
print(f"   gamma: {gamma_samples.mean():.3f} ± {gamma_samples.std():.3f}")
print(f"   sigma: {sigma_samples.mean():.3f} ± {sigma_samples.std():.3f}")

# Generate posterior predictive samples
print("\n4. Generating posterior predictive samples...")
n_ppc_samples = 1000
sample_indices = np.random.choice(n_samples, size=n_ppc_samples, replace=False)

# Storage for predictions
y_pred_samples = np.zeros((n_ppc_samples, n_obs))
y_rep_samples = np.zeros((n_ppc_samples, n_obs))

# Model function
def exp_model(x, alpha, beta, gamma):
    """Asymptotic exponential model: α - β*exp(-γ*x)"""
    return alpha - beta * np.exp(-gamma * x)

# Generate predictions
for i, idx in enumerate(sample_indices):
    alpha = alpha_samples[idx]
    beta = beta_samples[idx]
    gamma = gamma_samples[idx]
    sigma = sigma_samples[idx]

    # Mean prediction
    mu = exp_model(x_obs, alpha, beta, gamma)
    y_pred_samples[i, :] = mu

    # Replicated data (with observation noise)
    y_rep_samples[i, :] = np.random.normal(mu, sigma)

print(f"   Generated {n_ppc_samples} posterior predictive samples")

# Compute predictive intervals
y_pred_mean = y_pred_samples.mean(axis=0)
y_pred_median = np.median(y_pred_samples, axis=0)
y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)

# Replicated data intervals
y_rep_lower = np.percentile(y_rep_samples, 2.5, axis=0)
y_rep_upper = np.percentile(y_rep_samples, 97.5, axis=0)

print("\n5. Computing model fit metrics...")

# Coverage: % of observations in 95% predictive interval
in_pi = (y_obs >= y_rep_lower) & (y_obs <= y_rep_upper)
coverage = 100 * in_pi.mean()
print(f"   Coverage (95% PI): {coverage:.1f}% ({in_pi.sum()}/{n_obs} observations)")

# R-squared (using posterior mean)
ss_res = np.sum((y_obs - y_pred_mean)**2)
ss_tot = np.sum((y_obs - y_obs.mean())**2)
r_squared = 1 - ss_res / ss_tot
print(f"   R²: {r_squared:.4f}")

# RMSE
rmse = np.sqrt(np.mean((y_obs - y_pred_mean)**2))
print(f"   RMSE: {rmse:.4f}")

# MAE
mae = np.mean(np.abs(y_obs - y_pred_mean))
print(f"   MAE: {mae:.4f}")

# Residuals
residuals = y_obs - y_pred_mean
print(f"   Mean residual: {residuals.mean():.4f}")
print(f"   Std residual: {residuals.std():.4f}")

print("\n6. Computing test statistics...")

# Test statistics for PPC
def compute_test_stats(y):
    """Compute various test statistics"""
    return {
        'mean': np.mean(y),
        'std': np.std(y),
        'min': np.min(y),
        'max': np.max(y),
        'q25': np.percentile(y, 25),
        'q75': np.percentile(y, 75),
        'skew': stats.skew(y),
        'kurt': stats.kurtosis(y)
    }

# Observed test statistics
obs_stats = compute_test_stats(y_obs)

# Replicated test statistics
rep_stats = {key: [] for key in obs_stats.keys()}
for i in range(n_ppc_samples):
    stats_i = compute_test_stats(y_rep_samples[i, :])
    for key in obs_stats.keys():
        rep_stats[key].append(stats_i[key])

# Bayesian p-values
print("\n   Test statistic p-values (should be near 0.5):")
for key in obs_stats.keys():
    p_value = np.mean([obs_stats[key] <= r for r in rep_stats[key]])
    print(f"   {key:8s}: obs={obs_stats[key]:7.3f}, p-value={p_value:.3f}")

print("\n7. Creating diagnostic visualizations...")

# Plot 1: Posterior Predictive Overlay
print("   - Plot 1: Posterior predictive overlay")
fig, ax = plt.subplots(figsize=(12, 7))

# Plot sample of replicated datasets
n_rep_show = 100
for i in range(n_rep_show):
    idx = np.random.randint(n_ppc_samples)
    ax.scatter(x_obs, y_rep_samples[idx, :], alpha=0.02, color='lightblue', s=20)

# Plot posterior mean and credible interval
x_sorted_idx = np.argsort(x_obs)
ax.fill_between(x_obs[x_sorted_idx], y_pred_lower[x_sorted_idx],
                y_pred_upper[x_sorted_idx], alpha=0.3, color='blue',
                label='95% Credible Interval (mean function)')
ax.plot(x_obs[x_sorted_idx], y_pred_mean[x_sorted_idx], 'b-', linewidth=2,
        label='Posterior Mean')

# Plot observed data
ax.scatter(x_obs, y_obs, color='red', s=80, alpha=0.7,
          edgecolor='darkred', linewidth=1.5, label='Observed Data', zorder=5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Posterior Predictive Check: Observed vs Predicted Data', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}ppc_overlay.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Predictive Intervals with Coverage
print("   - Plot 2: Predictive intervals with coverage")
fig, ax = plt.subplots(figsize=(12, 7))

# Sort by x for better visualization
x_sorted_idx = np.argsort(x_obs)

# Plot 95% predictive interval (includes observation noise)
ax.fill_between(x_obs[x_sorted_idx], y_rep_lower[x_sorted_idx],
                y_rep_upper[x_sorted_idx], alpha=0.2, color='gray',
                label='95% Predictive Interval')

# Plot posterior mean
ax.plot(x_obs[x_sorted_idx], y_pred_mean[x_sorted_idx], 'b-',
        linewidth=2, label='Posterior Mean')

# Plot observed data (color by coverage)
in_interval = in_pi
ax.scatter(x_obs[in_interval], y_obs[in_interval], color='green', s=100,
          alpha=0.7, edgecolor='darkgreen', linewidth=1.5,
          label=f'In PI ({in_pi.sum()} obs)', zorder=5)
if not in_interval.all():
    ax.scatter(x_obs[~in_interval], y_obs[~in_interval], color='red', s=100,
              alpha=0.7, edgecolor='darkred', linewidth=1.5, marker='X',
              label=f'Outside PI ({(~in_pi).sum()} obs)', zorder=5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title(f'Predictive Intervals - Coverage: {coverage:.1f}%',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}predictive_intervals.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Residual Diagnostics
print("   - Plot 3: Residual diagnostics")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals vs fitted
axes[0, 0].scatter(y_pred_mean, residuals, alpha=0.6, s=60)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted Values', fontsize=11)
axes[0, 0].set_ylabel('Residuals', fontsize=11)
axes[0, 0].set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Residuals vs x
axes[0, 1].scatter(x_obs, residuals, alpha=0.6, s=60)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('x', fontsize=11)
axes[0, 1].set_ylabel('Residuals', fontsize=11)
axes[0, 1].set_title('Residuals vs Predictor', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Histogram of residuals
axes[1, 1].hist(residuals, bins=15, density=True, alpha=0.7, edgecolor='black')
x_range = np.linspace(residuals.min(), residuals.max(), 100)
axes[1, 1].plot(x_range, stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
               'r-', linewidth=2, label='Normal fit')
axes[1, 1].set_xlabel('Residuals', fontsize=11)
axes[1, 1].set_ylabel('Density', fontsize=11)
axes[1, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}residual_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Test Statistics Distributions
print("   - Plot 4: Test statistics distributions")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

stat_names = list(obs_stats.keys())
for i, stat_name in enumerate(stat_names):
    axes[i].hist(rep_stats[stat_name], bins=30, density=True, alpha=0.6,
                edgecolor='black', label='Replicated')
    axes[i].axvline(obs_stats[stat_name], color='red', linewidth=2.5,
                   label='Observed', linestyle='--')
    p_val = np.mean([obs_stats[stat_name] <= r for r in rep_stats[stat_name]])
    axes[i].set_title(f'{stat_name}\np-value: {p_val:.3f}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Value', fontsize=10)
    axes[i].set_ylabel('Density', fontsize=10)
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Test Statistics: Observed vs Posterior Predictive',
            fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}test_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Observed vs Predicted Scatter
print("   - Plot 5: Observed vs predicted scatter")
fig, ax = plt.subplots(figsize=(9, 9))

ax.scatter(y_obs, y_pred_mean, s=100, alpha=0.7, edgecolor='black', linewidth=1)
ax.errorbar(y_obs, y_pred_mean,
           yerr=[y_pred_mean - y_pred_lower, y_pred_upper - y_pred_mean],
           fmt='none', ecolor='blue', alpha=0.3, linewidth=1)

# Perfect prediction line
min_val = min(y_obs.min(), y_pred_mean.min())
max_val = max(y_obs.max(), y_pred_mean.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
       label='Perfect Prediction')

ax.set_xlabel('Observed Y', fontsize=12)
ax.set_ylabel('Predicted Y (Posterior Mean)', fontsize=12)
ax.set_title(f'Observed vs Predicted\nR² = {r_squared:.4f}, RMSE = {rmse:.4f}',
            fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}observed_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Distribution Comparison
print("   - Plot 6: Distribution comparison")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Overlaid histograms
axes[0].hist(y_obs, bins=15, density=True, alpha=0.6, label='Observed',
            edgecolor='black', color='red')
# Sample some replicated datasets
y_rep_flat = y_rep_samples.flatten()
axes[0].hist(y_rep_flat, bins=30, density=True, alpha=0.4, label='Replicated (pooled)',
            edgecolor='black', color='blue')
axes[0].set_xlabel('Y', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Distribution Comparison', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Empirical CDFs
axes[1].hist(y_obs, bins=20, cumulative=True, density=True, alpha=0.6,
            label='Observed', edgecolor='black', color='red', histtype='step',
            linewidth=2)
# Plot several replicated CDFs
for i in range(100):
    idx = np.random.randint(n_ppc_samples)
    axes[1].hist(y_rep_samples[idx, :], bins=20, cumulative=True, density=True,
                alpha=0.02, color='blue', histtype='step')
axes[1].set_xlabel('Y', fontsize=12)
axes[1].set_ylabel('Cumulative Probability', fontsize=12)
axes[1].set_title('Empirical CDF Comparison', fontsize=13, fontweight='bold')
axes[1].legend(['Observed', 'Replicated (samples)'], fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 7: ArviZ PPC plot
print("   - Plot 7: ArviZ PPC plot")
try:
    # Add posterior predictive samples to idata
    from xarray import DataArray
    y_rep_xr = DataArray(
        y_rep_samples.reshape(n_ppc_samples//4, 4, n_obs),
        dims=['chain', 'draw', 'obs_id'],
        coords={'chain': range(4), 'draw': range(n_ppc_samples//4), 'obs_id': range(n_obs)}
    )
    y_obs_xr = DataArray(y_obs, dims=['obs_id'], coords={'obs_id': range(n_obs)})

    idata.add_groups({'posterior_predictive': {'Y': y_rep_xr}})
    idata.add_groups({'observed_data': {'Y': y_obs_xr}})

    fig, ax = plt.subplots(figsize=(12, 6))
    az.plot_ppc(idata, num_pp_samples=100, ax=ax)
    ax.set_title('ArviZ Posterior Predictive Check', fontsize=14, fontweight='bold')
    ax.set_xlabel('Y', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}arviz_ppc.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("      ArviZ PPC plot created successfully")
except Exception as e:
    print(f"      Warning: Could not create ArviZ PPC plot: {e}")

# Plot 8: LOO-PIT diagnostic
print("   - Plot 8: LOO-PIT diagnostic")
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_loo_pit(idata, y='Y', ecdf=True, ax=ax)
    ax.set_title('LOO Probability Integral Transform', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}loo_pit.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("      LOO-PIT plot created successfully")
except Exception as e:
    print(f"      Warning: Could not create LOO-PIT plot: {e}")

print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)

print(f"\n1. COVERAGE ASSESSMENT:")
print(f"   - 95% Predictive Interval Coverage: {coverage:.1f}%")
if 90 <= coverage <= 98:
    print(f"   - Status: EXCELLENT (within expected range)")
elif 85 <= coverage < 90 or 98 < coverage <= 99:
    print(f"   - Status: ACCEPTABLE (slightly outside expected range)")
else:
    print(f"   - Status: PROBLEMATIC (outside expected range)")

print(f"\n2. FIT QUALITY:")
print(f"   - R²: {r_squared:.4f}")
if r_squared >= 0.85:
    print(f"   - Status: EXCELLENT (exceeds threshold)")
elif r_squared >= 0.75:
    print(f"   - Status: ACCEPTABLE")
else:
    print(f"   - Status: POOR (below threshold)")

print(f"\n3. RESIDUAL DIAGNOSTICS:")
print(f"   - Mean residual: {residuals.mean():.4f} (should be ~0)")
print(f"   - RMSE: {rmse:.4f}")
print(f"   - MAE: {mae:.4f}")

# Shapiro-Wilk test for normality of residuals
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"   - Normality test (Shapiro-Wilk): p={shapiro_p:.4f}")
if shapiro_p > 0.05:
    print(f"     Status: Residuals appear normally distributed")
else:
    print(f"     Status: Residuals may not be normally distributed")

print(f"\n4. TEST STATISTICS:")
extreme_p_vals = []
for key in obs_stats.keys():
    p_val = np.mean([obs_stats[key] <= r for r in rep_stats[key]])
    if p_val < 0.05 or p_val > 0.95:
        extreme_p_vals.append((key, p_val))

if not extreme_p_vals:
    print(f"   - All test statistics have reasonable p-values (0.05 < p < 0.95)")
else:
    print(f"   - WARNING: Some test statistics show extreme p-values:")
    for key, pval in extreme_p_vals:
        print(f"     {key}: p={pval:.3f}")

print(f"\n5. OVERALL MODEL ADEQUACY:")
adequate = True
reasons = []

if not (90 <= coverage <= 98):
    adequate = False
    reasons.append(f"Coverage ({coverage:.1f}%) outside 90-98% range")

if r_squared < 0.85:
    adequate = False
    reasons.append(f"R² ({r_squared:.4f}) below 0.85 threshold")

if len(extreme_p_vals) > 2:
    adequate = False
    reasons.append(f"{len(extreme_p_vals)} test statistics with extreme p-values")

if adequate:
    print("   *** VERDICT: MODEL IS ADEQUATE ***")
    print("   - Coverage is within expected range")
    print("   - R² exceeds threshold")
    print("   - No systematic discrepancies detected")
else:
    print("   *** VERDICT: MODEL NEEDS IMPROVEMENT ***")
    print("   Issues identified:")
    for reason in reasons:
        print(f"   - {reason}")

print("\n" + "="*80)
print("Analysis complete! Check the plots directory for visualizations.")
print("="*80)

# Save summary statistics to file
summary = {
    'coverage_pct': coverage,
    'r_squared': r_squared,
    'rmse': rmse,
    'mae': mae,
    'mean_residual': residuals.mean(),
    'std_residual': residuals.std(),
    'shapiro_p_value': shapiro_p,
    'n_observations': n_obs,
    'n_ppc_samples': n_ppc_samples,
    'adequate': adequate
}

import json
with open(f'{PLOTS_DIR}../ppc_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary statistics saved to: {PLOTS_DIR}../ppc_summary.json")
