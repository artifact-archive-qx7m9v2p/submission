"""
Posterior Predictive Check for Experiment 3: Log-Log Power Law Model

This script performs comprehensive posterior predictive checks to validate
whether the fitted model can reproduce key features of the observed data.

Model: log(Y) ~ Normal(α + β*log(x), σ)
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.ndimage import uniform_filter1d

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_3/posterior_predictive_check")
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path("/workspace/data/data.csv")
POSTERIOR_PATH = Path("/workspace/experiments/experiment_3/posterior_inference/diagnostics/posterior_inference.netcdf")

print("="*80)
print("POSTERIOR PREDICTIVE CHECK: LOG-LOG POWER LAW MODEL")
print("="*80)

# =============================================================================
# 1. LOAD DATA AND POSTERIOR
# =============================================================================

print("\n1. Loading observed data and posterior samples...")

# Load observed data
data = pd.read_csv(DATA_PATH)
x_obs = data['x'].values
Y_obs = data['Y'].values
n_obs = len(Y_obs)

print(f"   - Observations: {n_obs}")
print(f"   - x range: [{x_obs.min():.1f}, {x_obs.max():.1f}]")
print(f"   - Y range: [{Y_obs.min():.3f}, {Y_obs.max():.3f}]")

# Load posterior inference data
idata = az.from_netcdf(POSTERIOR_PATH)

print(f"   - Posterior chains: {idata.posterior.sizes['chain']}")
print(f"   - Posterior draws per chain: {idata.posterior.sizes['draw']}")
print(f"   - Total posterior samples: {idata.posterior.sizes['chain'] * idata.posterior.sizes['draw']}")

# Extract posterior samples
alpha_samples = idata.posterior['alpha'].values.flatten()
beta_samples = idata.posterior['beta'].values.flatten()
sigma_samples = idata.posterior['sigma'].values.flatten()

print(f"\n   Parameter estimates (posterior mean ± SD):")
print(f"   - α: {alpha_samples.mean():.4f} ± {alpha_samples.std():.4f}")
print(f"   - β: {beta_samples.mean():.4f} ± {beta_samples.std():.4f}")
print(f"   - σ: {sigma_samples.mean():.4f} ± {sigma_samples.std():.4f}")

# =============================================================================
# 2. GENERATE POSTERIOR PREDICTIVE SAMPLES
# =============================================================================

print("\n2. Generating posterior predictive samples...")

n_ppc_samples = 1000  # Number of posterior predictive replications
n_posterior = len(alpha_samples)

# Randomly select posterior samples for PPC
ppc_indices = np.random.choice(n_posterior, size=n_ppc_samples, replace=True)

# Generate predictions on log scale, then back-transform
log_x_obs = np.log(x_obs)
Y_ppc = np.zeros((n_ppc_samples, n_obs))

for i, idx in enumerate(ppc_indices):
    alpha = alpha_samples[idx]
    beta = beta_samples[idx]
    sigma = sigma_samples[idx]

    # Predict on log scale
    log_Y_pred = alpha + beta * log_x_obs

    # Add noise on log scale
    log_Y_sim = log_Y_pred + np.random.normal(0, sigma, n_obs)

    # Back-transform to original scale
    Y_ppc[i, :] = np.exp(log_Y_sim)

print(f"   - Generated {n_ppc_samples} posterior predictive replications")
print(f"   - Shape: {Y_ppc.shape}")

# =============================================================================
# 3. COVERAGE STATISTICS
# =============================================================================

print("\n3. Computing coverage statistics...")

# Compute 50%, 80%, 95% posterior predictive intervals
ppc_quantiles = {
    '95%': (2.5, 97.5),
    '80%': (10, 90),
    '50%': (25, 75)
}

coverage_results = {}
for level, (lower_q, upper_q) in ppc_quantiles.items():
    lower = np.percentile(Y_ppc, lower_q, axis=0)
    upper = np.percentile(Y_ppc, upper_q, axis=0)

    in_interval = (Y_obs >= lower) & (Y_obs <= upper)
    coverage = 100 * in_interval.mean()

    coverage_results[level] = {
        'coverage': coverage,
        'expected': float(level.strip('%')),
        'n_in': int(in_interval.sum()),
        'n_total': n_obs
    }

    print(f"   - {level} PI: {coverage:.1f}% coverage ({in_interval.sum()}/{n_obs} observations)")
    print(f"              Expected: {float(level.strip('%')):.0f}%")

# =============================================================================
# 4. SUMMARY STATISTICS COMPARISON
# =============================================================================

print("\n4. Comparing observed vs replicated summary statistics...")

# Compute statistics for observed data
obs_stats = {
    'mean': Y_obs.mean(),
    'sd': Y_obs.std(ddof=1),
    'min': Y_obs.min(),
    'max': Y_obs.max(),
    'median': np.median(Y_obs),
    'q25': np.percentile(Y_obs, 25),
    'q75': np.percentile(Y_obs, 75)
}

# Compute same statistics for each PPC replication
ppc_stats = {
    'mean': Y_ppc.mean(axis=1),
    'sd': Y_ppc.std(axis=1, ddof=1),
    'min': Y_ppc.min(axis=1),
    'max': Y_ppc.max(axis=1),
    'median': np.median(Y_ppc, axis=1),
    'q25': np.percentile(Y_ppc, 25, axis=1),
    'q75': np.percentile(Y_ppc, 75, axis=1)
}

# Compute p-values (Bayesian p-value: proportion of replications more extreme than observed)
print("\n   Statistic    | Observed | PPC Mean | PPC SD  | p-value")
print("   " + "-"*60)

for stat_name in obs_stats.keys():
    obs_val = obs_stats[stat_name]
    ppc_vals = ppc_stats[stat_name]
    ppc_mean = ppc_vals.mean()
    ppc_sd = ppc_vals.std()

    # Two-sided p-value
    p_value = 2 * min(
        (ppc_vals >= obs_val).mean(),
        (ppc_vals <= obs_val).mean()
    )

    print(f"   {stat_name:12s} | {obs_val:8.3f} | {ppc_mean:8.3f} | {ppc_sd:7.3f} | {p_value:7.3f}")

# =============================================================================
# 5. RESIDUAL ANALYSIS
# =============================================================================

print("\n5. Computing residuals on log scale...")

# Compute posterior mean predictions on log scale
log_Y_obs = np.log(Y_obs)
log_Y_pred_mean = alpha_samples.mean() + beta_samples.mean() * log_x_obs

# Residuals on log scale
residuals_log = log_Y_obs - log_Y_pred_mean

print(f"   - Mean residual: {residuals_log.mean():.6f} (should be ~0)")
print(f"   - SD residual: {residuals_log.std():.4f}")
print(f"   - Min residual: {residuals_log.min():.4f}")
print(f"   - Max residual: {residuals_log.max():.4f}")

# Test for normality
_, p_shapiro = stats.shapiro(residuals_log)
print(f"   - Shapiro-Wilk test: p = {p_shapiro:.4f}")

# Test for homoscedasticity (Breusch-Pagan test approximation)
# Regress squared residuals on log(x)
res_sq = residuals_log**2
corr_res_x = np.corrcoef(log_x_obs, res_sq)[0, 1]
print(f"   - Correlation(log(x), residual²): {corr_res_x:.4f} (should be ~0)")

# =============================================================================
# 6. MODEL PERFORMANCE METRICS
# =============================================================================

print("\n6. Computing model performance metrics...")

# Posterior mean predictions on original scale
Y_pred_mean = np.exp(log_Y_pred_mean + sigma_samples.mean()**2 / 2)  # Bias correction

# R-squared on original scale
ss_total = np.sum((Y_obs - Y_obs.mean())**2)
ss_residual = np.sum((Y_obs - Y_pred_mean)**2)
r_squared = 1 - ss_residual / ss_total

# RMSE
rmse = np.sqrt(np.mean((Y_obs - Y_pred_mean)**2))

# MAE
mae = np.mean(np.abs(Y_obs - Y_pred_mean))

print(f"   - R²: {r_squared:.4f}")
print(f"   - RMSE: {rmse:.4f}")
print(f"   - MAE: {mae:.4f}")

# Posterior predictive R² (compute R² for each PPC sample)
r_squared_ppc = []
for i in range(n_ppc_samples):
    ss_res = np.sum((Y_obs - Y_ppc[i])**2)
    r_sq = 1 - ss_res / ss_total
    r_squared_ppc.append(r_sq)

r_squared_ppc = np.array(r_squared_ppc)
print(f"   - Posterior predictive R²: {r_squared_ppc.mean():.4f} ± {r_squared_ppc.std():.4f}")

# =============================================================================
# 7. CHECK PERFORMANCE AT REPLICATED X-VALUES
# =============================================================================

print("\n7. Analyzing fit at replicated x-values...")

# Identify replicated x-values
x_unique, x_counts = np.unique(x_obs, return_counts=True)
x_replicated = x_unique[x_counts > 1]

print(f"   - Found {len(x_replicated)} x-values with replicates:")

replicate_analysis = []
for x_val in x_replicated:
    mask = x_obs == x_val
    n_reps = mask.sum()
    Y_reps = Y_obs[mask]

    # Observed statistics
    obs_mean = Y_reps.mean()
    obs_sd = Y_reps.std(ddof=1) if n_reps > 1 else 0.0
    obs_range = Y_reps.max() - Y_reps.min()

    # Predicted statistics (from posterior predictive at this x)
    Y_ppc_at_x = Y_ppc[:, mask]  # Shape: (n_ppc_samples, n_reps)

    # Average across replicates for each PPC sample
    Y_ppc_mean_at_x = Y_ppc_at_x.mean(axis=1)
    pred_mean = Y_ppc_mean_at_x.mean()
    pred_sd_within = Y_ppc_at_x.std(axis=1).mean()  # Average within-replicate SD

    # Does observed mean fall in predictive distribution?
    ppc_lower = np.percentile(Y_ppc_mean_at_x, 2.5)
    ppc_upper = np.percentile(Y_ppc_mean_at_x, 97.5)
    in_pi = (obs_mean >= ppc_lower) and (obs_mean <= ppc_upper)

    replicate_analysis.append({
        'x': x_val,
        'n': int(n_reps),
        'obs_mean': float(obs_mean),
        'obs_sd': float(obs_sd),
        'pred_mean': float(pred_mean),
        'pred_sd': float(pred_sd_within),
        'in_95_PI': bool(in_pi)
    })

    print(f"     x = {x_val:4.1f} (n={n_reps}): obs_mean = {obs_mean:.3f}, pred_mean = {pred_mean:.3f}, in_PI = {in_pi}")

replicate_df = pd.DataFrame(replicate_analysis)

# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================

print("\n8. Creating diagnostic visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# -------------------------------------------------------------------------
# Plot 1: Posterior Predictive Check - Overlay
# -------------------------------------------------------------------------
print("   - Creating overlay_ppc.png...")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot 100 random posterior predictive samples
for i in np.random.choice(n_ppc_samples, size=100, replace=False):
    ax.plot(x_obs, Y_ppc[i], 'o', alpha=0.05, color='steelblue', markersize=4)

# Plot posterior predictive intervals
ppc_median = np.median(Y_ppc, axis=0)
ppc_lower_95 = np.percentile(Y_ppc, 2.5, axis=0)
ppc_upper_95 = np.percentile(Y_ppc, 97.5, axis=0)
ppc_lower_50 = np.percentile(Y_ppc, 25, axis=0)
ppc_upper_50 = np.percentile(Y_ppc, 75, axis=0)

# Sort by x for plotting intervals
sort_idx = np.argsort(x_obs)
x_sorted = x_obs[sort_idx]
y_sorted = Y_obs[sort_idx]
ppc_median_sorted = ppc_median[sort_idx]
ppc_lower_95_sorted = ppc_lower_95[sort_idx]
ppc_upper_95_sorted = ppc_upper_95[sort_idx]
ppc_lower_50_sorted = ppc_lower_50[sort_idx]
ppc_upper_50_sorted = ppc_upper_50[sort_idx]

ax.fill_between(x_sorted, ppc_lower_95_sorted, ppc_upper_95_sorted,
                alpha=0.3, color='lightblue', label='95% PI')
ax.fill_between(x_sorted, ppc_lower_50_sorted, ppc_upper_50_sorted,
                alpha=0.5, color='steelblue', label='50% PI')
ax.plot(x_sorted, ppc_median_sorted, 'b-', linewidth=2, label='Posterior Median')

# Plot observed data
ax.plot(x_obs, Y_obs, 'ro', markersize=8, label='Observed Data', zorder=10, alpha=0.8)

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_title('Posterior Predictive Check: Observed vs Replicated Data',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "overlay_ppc.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 2: Residual Plot on Log Scale
# -------------------------------------------------------------------------
print("   - Creating residual_plot_log_scale.png...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Residuals vs log(x)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
ax1.plot(log_x_obs, residuals_log, 'o', markersize=8, alpha=0.7)

ax1.set_xlabel('log(x)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Residual [log(Y) - log(Ŷ)]', fontsize=12, fontweight='bold')
ax1.set_title('Residuals vs log(x)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: Residuals vs fitted values
fitted_log = log_Y_pred_mean
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
ax2.plot(fitted_log, residuals_log, 'o', markersize=8, alpha=0.7)

ax2.set_xlabel('Fitted log(Y)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residual [log(Y) - log(Ŷ)]', fontsize=12, fontweight='bold')
ax2.set_title('Residuals vs Fitted Values', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "residual_plot_log_scale.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 3: Q-Q Plot for Normality
# -------------------------------------------------------------------------
print("   - Creating qq_plot_residuals.png...")

fig, ax = plt.subplots(figsize=(8, 8))

stats.probplot(residuals_log, dist="norm", plot=ax)
ax.set_title(f'Q-Q Plot: Residuals on Log Scale\n(Shapiro-Wilk p = {p_shapiro:.4f})',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "qq_plot_residuals.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 4: Coverage Diagnostic
# -------------------------------------------------------------------------
print("   - Creating coverage_diagnostic.png...")

fig, ax = plt.subplots(figsize=(12, 7))

# Sort by x for cleaner visualization
sort_idx = np.argsort(x_obs)
x_sorted = x_obs[sort_idx]
y_sorted = Y_obs[sort_idx]

# Plot intervals
ax.fill_between(x_sorted, ppc_lower_95_sorted, ppc_upper_95_sorted,
                alpha=0.2, color='lightblue', label='95% PI')
ax.plot(x_sorted, ppc_lower_95_sorted, 'b--', linewidth=1, alpha=0.5)
ax.plot(x_sorted, ppc_upper_95_sorted, 'b--', linewidth=1, alpha=0.5)

# Identify observations outside 95% PI
in_pi = (Y_obs >= ppc_lower_95) & (Y_obs <= ppc_upper_95)
out_pi = ~in_pi

# Plot observations
if in_pi.any():
    ax.plot(x_obs[in_pi], Y_obs[in_pi], 'go', markersize=10,
            label=f'Inside 95% PI (n={in_pi.sum()})', alpha=0.8, zorder=10)
if out_pi.any():
    ax.plot(x_obs[out_pi], Y_obs[out_pi], 'ro', markersize=10,
            label=f'Outside 95% PI (n={out_pi.sum()})', alpha=0.8, zorder=10)

ax.set_xlabel('x', fontsize=12, fontweight='bold')
ax.set_ylabel('Y', fontsize=12, fontweight='bold')
ax.set_title(f'Coverage Diagnostic: {coverage_results["95%"]["coverage"]:.1f}% in 95% PI',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_diagnostic.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 5: Performance at Replicated X-Values
# -------------------------------------------------------------------------
print("   - Creating replicate_performance.png...")

if len(x_replicated) > 0:
    fig, ax = plt.subplots(figsize=(12, 7))

    for x_val in x_replicated:
        mask = x_obs == x_val
        Y_reps = Y_obs[mask]

        # Get PPC distribution at this x
        Y_ppc_at_x = Y_ppc[:, mask]

        # Plot individual observations
        ax.plot([x_val] * len(Y_reps), Y_reps, 'ro', markersize=10, alpha=0.7, zorder=10)

        # Plot PPC violin
        positions = [x_val]
        parts = ax.violinplot([Y_ppc_at_x.flatten()], positions=positions,
                              widths=0.5, showmeans=True, showextrema=True)

        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r',
                   markersize=10, label='Observed replicates'),
        Patch(facecolor='steelblue', alpha=0.3, label='PPC distribution')
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_title('Performance at Replicated X-Values', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "replicate_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------------------
# Plot 6: Summary Statistics Comparison
# -------------------------------------------------------------------------
print("   - Creating summary_statistics_comparison.png...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

stat_names = ['mean', 'sd', 'min', 'max', 'median', 'q75']
stat_labels = ['Mean', 'SD', 'Minimum', 'Maximum', 'Median', '75th Percentile']

for i, (stat_name, stat_label) in enumerate(zip(stat_names, stat_labels)):
    ax = axes[i]

    # Plot PPC distribution
    ax.hist(ppc_stats[stat_name], bins=30, alpha=0.6, color='steelblue',
            edgecolor='black', label='PPC distribution')

    # Plot observed value
    obs_val = obs_stats[stat_name]
    ax.axvline(obs_val, color='red', linewidth=3, label='Observed', linestyle='--')

    # Compute p-value
    ppc_vals = ppc_stats[stat_name]
    p_value = 2 * min((ppc_vals >= obs_val).mean(), (ppc_vals <= obs_val).mean())

    ax.set_xlabel(stat_label, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'{stat_label}\n(p = {p_value:.3f})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "summary_statistics_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 7: Original Scale vs Log Scale Comparison
# -------------------------------------------------------------------------
print("   - Creating scale_comparison.png...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Original scale
ax1.plot(x_obs, Y_obs, 'ro', markersize=8, label='Observed', alpha=0.8, zorder=10)
ax1.fill_between(x_sorted, ppc_lower_95_sorted, ppc_upper_95_sorted,
                 alpha=0.3, color='lightblue', label='95% PI')
ax1.plot(x_sorted, ppc_median_sorted, 'b-', linewidth=2, label='Posterior Median')
ax1.set_xlabel('x', fontsize=12, fontweight='bold')
ax1.set_ylabel('Y', fontsize=12, fontweight='bold')
ax1.set_title('Original Scale', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right: Log-log scale
log_Y_obs_sorted = log_Y_obs[sort_idx]
log_ppc_median = np.log(ppc_median_sorted + 1e-10)  # Avoid log(0)
log_ppc_lower_95 = np.log(ppc_lower_95_sorted + 1e-10)
log_ppc_upper_95 = np.log(ppc_upper_95_sorted + 1e-10)

ax2.plot(log_x_obs, log_Y_obs, 'ro', markersize=8, label='Observed', alpha=0.8, zorder=10)
ax2.fill_between(log_x_obs[sort_idx], log_ppc_lower_95, log_ppc_upper_95,
                 alpha=0.3, color='lightblue', label='95% PI')
ax2.plot(log_x_obs[sort_idx], log_ppc_median, 'b-', linewidth=2, label='Posterior Median')
ax2.set_xlabel('log(x)', fontsize=12, fontweight='bold')
ax2.set_ylabel('log(Y)', fontsize=12, fontweight='bold')
ax2.set_title('Log-Log Scale', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "scale_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 8: Posterior Predictive R² Distribution
# -------------------------------------------------------------------------
print("   - Creating r_squared_distribution.png...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(r_squared_ppc, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(r_squared, color='red', linewidth=3, linestyle='--',
           label=f'Point estimate R² = {r_squared:.3f}')
ax.axvline(0.75, color='orange', linewidth=2, linestyle=':',
           label='Threshold (0.75)')

ax.set_xlabel('Posterior Predictive R²', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title(f'Posterior Predictive R² Distribution\nMean = {r_squared_ppc.mean():.3f} ± {r_squared_ppc.std():.3f}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "r_squared_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECK COMPLETE")
print("="*80)
print(f"\nAll plots saved to: {PLOTS_DIR}")
print("\nKey findings:")
print(f"  - Coverage (95% PI): {coverage_results['95%']['coverage']:.1f}%")
print(f"  - R²: {r_squared:.4f}")
print(f"  - RMSE: {rmse:.4f}")
print(f"  - Mean residual (log scale): {residuals_log.mean():.6f}")
print(f"  - Shapiro-Wilk p-value: {p_shapiro:.4f}")

# Save numerical results
results_summary = {
    'coverage': coverage_results,
    'r_squared': float(r_squared),
    'r_squared_ppc_mean': float(r_squared_ppc.mean()),
    'r_squared_ppc_sd': float(r_squared_ppc.std()),
    'rmse': float(rmse),
    'mae': float(mae),
    'residual_mean': float(residuals_log.mean()),
    'residual_sd': float(residuals_log.std()),
    'shapiro_wilk_p': float(p_shapiro),
    'obs_stats': {k: float(v) for k, v in obs_stats.items()},
    'ppc_stats_mean': {k: float(v.mean()) for k, v in ppc_stats.items()},
    'replicate_performance': replicate_df.to_dict('records')
}

import json
with open(BASE_DIR / "ppc_results.json", 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nNumerical results saved to: {BASE_DIR / 'ppc_results.json'}")
