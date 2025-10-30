"""
Create comprehensive posterior predictive check visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import arviz as az

# Load results
data_file = np.load('/workspace/experiments/experiment_1/posterior_predictive_check/code/ppc_results.npz')
x_obs = data_file['x_obs']
y_obs = data_file['y_obs']
y_rep = data_file['y_rep']
y_pred_mean = data_file['y_pred_mean']
y_pred_median = data_file['y_pred_median']
y_pred_std = data_file['y_pred_std']
y_pred_50 = data_file['y_pred_50']
y_pred_90 = data_file['y_pred_90']
y_pred_95 = data_file['y_pred_95']
residuals = data_file['residuals']
standardized_residuals = data_file['standardized_residuals']

# Load idata for ArviZ plots
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

# ============================================================================
# PLOT 1: PPC OVERVIEW (Multi-panel summary)
# ============================================================================

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel A: Observed vs Predicted with credible intervals
ax1 = fig.add_subplot(gs[0, :2])

# Sort by x for cleaner plotting
sort_idx = np.argsort(x_obs)
x_sorted = x_obs[sort_idx]
y_sorted = y_obs[sort_idx]
y_pred_mean_sorted = y_pred_mean[sort_idx]
y_pred_50_sorted = y_pred_50[:, sort_idx]
y_pred_90_sorted = y_pred_90[:, sort_idx]
y_pred_95_sorted = y_pred_95[:, sort_idx]

# Plot credible intervals
ax1.fill_between(x_sorted, y_pred_95_sorted[0], y_pred_95_sorted[1],
                 alpha=0.2, color='C0', label='95% CI')
ax1.fill_between(x_sorted, y_pred_90_sorted[0], y_pred_90_sorted[1],
                 alpha=0.3, color='C0', label='90% CI')
ax1.fill_between(x_sorted, y_pred_50_sorted[0], y_pred_50_sorted[1],
                 alpha=0.4, color='C0', label='50% CI')
ax1.plot(x_sorted, y_pred_mean_sorted, 'C0-', linewidth=2, label='Posterior mean')

# Plot observed data
ax1.scatter(x_obs, y_obs, color='black', s=50, alpha=0.7, zorder=5, label='Observed')

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_title('A. Observed vs Predicted with Credible Intervals', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel B: ArviZ PPC overlay
ax2 = fig.add_subplot(gs[0, 2])
# Plot subset of replications for clarity
n_rep_plot = 100
rep_idx = np.random.choice(y_rep.shape[0], n_rep_plot, replace=False)
for i in rep_idx[:50]:
    ax2.plot(x_obs, y_rep[i, :], 'C0-', alpha=0.05)
ax2.scatter(x_obs, y_obs, color='black', s=30, alpha=0.7, zorder=5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('Y', fontsize=12)
ax2.set_title('B. Posterior Predictive\nReplicates (n=50)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Panel C: Residuals vs Fitted
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y_pred_mean, residuals, color='C0', s=50, alpha=0.7)
ax3.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax3.axhline(2*residuals.std(), color='red', linestyle=':', linewidth=1, alpha=0.5)
ax3.axhline(-2*residuals.std(), color='red', linestyle=':', linewidth=1, alpha=0.5)
ax3.set_xlabel('Fitted values', fontsize=12)
ax3.set_ylabel('Residuals', fontsize=12)
ax3.set_title('C. Residuals vs Fitted', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Panel D: Residuals vs x
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(x_obs, residuals, color='C0', s=50, alpha=0.7)
ax4.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax4.axhline(2*residuals.std(), color='red', linestyle=':', linewidth=1, alpha=0.5)
ax4.axhline(-2*residuals.std(), color='red', linestyle=':', linewidth=1, alpha=0.5)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('Residuals', fontsize=12)
ax4.set_title('D. Residuals vs x', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Panel E: Q-Q plot
ax5 = fig.add_subplot(gs[1, 2])
stats.probplot(standardized_residuals, dist="norm", plot=ax5)
ax5.set_title('E. Q-Q Plot\n(Standardized Residuals)', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Panel F: Histogram of observed vs replicated
ax6 = fig.add_subplot(gs[2, 0])
# Plot replicated histograms
for i in np.random.choice(y_rep.shape[0], 50, replace=False):
    ax6.hist(y_rep[i, :], bins=15, alpha=0.02, color='C0', edgecolor='none')
# Plot observed
ax6.hist(y_obs, bins=15, alpha=0.7, color='red', edgecolor='black', linewidth=1.5, label='Observed')
ax6.set_xlabel('Y', fontsize=12)
ax6.set_ylabel('Frequency', fontsize=12)
ax6.set_title('F. Distribution:\nObserved vs Replicated', fontsize=13, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# Panel G: Residuals vs Observation Index
ax7 = fig.add_subplot(gs[2, 1])
ax7.scatter(range(len(residuals)), residuals, color='C0', s=50, alpha=0.7)
ax7.plot(range(len(residuals)), residuals, 'C0-', alpha=0.3)
ax7.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax7.axhline(2*residuals.std(), color='red', linestyle=':', linewidth=1, alpha=0.5)
ax7.axhline(-2*residuals.std(), color='red', linestyle=':', linewidth=1, alpha=0.5)
ax7.set_xlabel('Observation index', fontsize=12)
ax7.set_ylabel('Residuals', fontsize=12)
ax7.set_title('G. Residuals vs Index\n(Check for autocorrelation)', fontsize=13, fontweight='bold')
ax7.grid(True, alpha=0.3)

# Panel H: Scale-location plot
ax8 = fig.add_subplot(gs[2, 2])
ax8.scatter(y_pred_mean, np.abs(standardized_residuals), color='C0', s=50, alpha=0.7)
ax8.set_xlabel('Fitted values', fontsize=12)
ax8.set_ylabel('|Standardized residuals|', fontsize=12)
ax8.set_title('H. Scale-Location Plot\n(Check homoscedasticity)', fontsize=13, fontweight='bold')
ax8.grid(True, alpha=0.3)

plt.suptitle('Posterior Predictive Check: Comprehensive Overview',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/ppc_overview.png',
            dpi=300, bbox_inches='tight')
print("Saved: ppc_overview.png")
plt.close()

# ============================================================================
# PLOT 2: TEST STATISTICS WITH POSTERIOR PREDICTIVE P-VALUES
# ============================================================================

# Compute test statistics
test_stats_data = {
    'min': {'obs': y_obs.min(), 'rep': y_rep.min(axis=1)},
    'max': {'obs': y_obs.max(), 'rep': y_rep.max(axis=1)},
    'mean': {'obs': y_obs.mean(), 'rep': y_rep.mean(axis=1)},
    'std': {'obs': y_obs.std(), 'rep': y_rep.std(axis=1)},
    'skewness': {'obs': stats.skew(y_obs),
                 'rep': np.array([stats.skew(y_rep[i, :]) for i in range(y_rep.shape[0])])},
    'range': {'obs': y_obs.max() - y_obs.min(),
              'rep': y_rep.max(axis=1) - y_rep.min(axis=1)},
    'IQR': {'obs': np.percentile(y_obs, 75) - np.percentile(y_obs, 25),
            'rep': np.percentile(y_rep, 75, axis=1) - np.percentile(y_rep, 25, axis=1)}
}

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

stat_names = list(test_stats_data.keys())
for i, stat_name in enumerate(stat_names):
    ax = axes[i]
    obs_val = test_stats_data[stat_name]['obs']
    rep_vals = test_stats_data[stat_name]['rep']

    # Histogram of replicated statistics
    ax.hist(rep_vals, bins=50, alpha=0.7, color='C0', edgecolor='black', linewidth=0.5)

    # Mark observed value
    ax.axvline(obs_val, color='red', linewidth=2.5, label='Observed', zorder=5)

    # Compute p-value
    if stat_name in ['mean', 'std', 'skewness', 'range', 'IQR']:
        p_upper = np.mean(rep_vals >= obs_val)
        p_lower = np.mean(rep_vals <= obs_val)
        p_val = min(p_upper, p_lower) * 2
    else:
        p_val = np.mean(rep_vals >= obs_val)

    # Color based on p-value
    if 0.05 <= p_val <= 0.95:
        status = 'GOOD'
        color = 'green'
    elif 0.01 <= p_val <= 0.99:
        status = 'WARNING'
        color = 'orange'
    else:
        status = 'FAIL'
        color = 'red'

    ax.set_xlabel(f'{stat_name}', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{stat_name}: p = {p_val:.3f} [{status}]',
                fontsize=12, fontweight='bold', color=color)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

# Remove extra subplots
for i in range(len(stat_names), len(axes)):
    fig.delaxes(axes[i])

plt.suptitle('Test Statistics: Observed vs Posterior Predictive Distribution',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/test_statistics.png',
            dpi=300, bbox_inches='tight')
print("Saved: test_statistics.png")
plt.close()

# ============================================================================
# PLOT 3: REPLICATE COVERAGE ANALYSIS
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Identify replicated x values
unique_x = np.unique(x_obs)
replicated_x = unique_x[np.array([sum(x_obs == ux) for ux in unique_x]) > 1]

for idx, ux in enumerate(replicated_x):
    ax = axes.flatten()[idx]

    # Get observations at this x value
    obs_idx = x_obs == ux
    y_subset = y_obs[obs_idx]
    y_rep_subset = y_rep[:, obs_idx]

    # Plot replicated distributions
    positions = np.arange(1, sum(obs_idx) + 1)

    for i, pos in enumerate(positions):
        # Violin plot of posterior predictive
        parts = ax.violinplot([y_rep_subset[:, i]], positions=[pos],
                              widths=0.5, showmeans=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('C0')
            pc.set_alpha(0.3)

        # Add credible intervals
        q25, q50, q75 = np.percentile(y_rep_subset[:, i], [25, 50, 75])
        q5, q95 = np.percentile(y_rep_subset[:, i], [5, 95])

        ax.plot([pos, pos], [q5, q95], 'C0-', linewidth=2, alpha=0.5)
        ax.plot([pos, pos], [q25, q75], 'C0-', linewidth=4)
        ax.plot(pos, q50, 'C0o', markersize=6)

        # Plot observed value
        ax.plot(pos, y_subset[i], 'ro', markersize=10, label='Observed' if i == 0 else '', zorder=5)

    ax.set_xlabel('Replicate', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_title(f'x = {ux}', fontsize=12, fontweight='bold')
    ax.set_xticks(positions)
    if idx == 0:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Replicate Coverage: Observed vs Posterior Predictive at Repeated x Values',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/replicate_coverage.png',
            dpi=300, bbox_inches='tight')
print("Saved: replicate_coverage.png")
plt.close()

# ============================================================================
# PLOT 4: RESIDUAL DIAGNOSTICS (Detailed)
# ============================================================================

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Panel A: Residuals vs Fitted (with LOWESS smooth)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_pred_mean, residuals, color='C0', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
ax1.axhline(0, color='red', linestyle='--', linewidth=2)
ax1.axhline(2*residuals.std(), color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='+/- 2 SD')
ax1.axhline(-2*residuals.std(), color='red', linestyle=':', linewidth=1.5, alpha=0.5)

# Add LOWESS smooth
from scipy.signal import savgol_filter
sort_idx_fitted = np.argsort(y_pred_mean)
if len(y_pred_mean) > 5:
    try:
        smooth = savgol_filter(residuals[sort_idx_fitted],
                              window_length=min(11, len(y_pred_mean)//2*2+1),
                              polyorder=3)
        ax1.plot(y_pred_mean[sort_idx_fitted], smooth, 'g-', linewidth=2.5,
                label='LOWESS smooth', alpha=0.7)
    except:
        pass

ax1.set_xlabel('Fitted values', fontsize=12, fontweight='bold')
ax1.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax1.set_title('A. Residuals vs Fitted Values', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel B: Residuals vs x
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(x_obs, residuals, color='C0', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
ax2.axhline(0, color='red', linestyle='--', linewidth=2)
ax2.axhline(2*residuals.std(), color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.axhline(-2*residuals.std(), color='red', linestyle=':', linewidth=1.5, alpha=0.5)

# Add smooth
sort_idx_x = np.argsort(x_obs)
if len(x_obs) > 5:
    try:
        smooth = savgol_filter(residuals[sort_idx_x],
                              window_length=min(11, len(x_obs)//2*2+1),
                              polyorder=3)
        ax2.plot(x_obs[sort_idx_x], smooth, 'g-', linewidth=2.5, alpha=0.7)
    except:
        pass

ax2.set_xlabel('x', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax2.set_title('B. Residuals vs Predictor', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Panel C: Scale-location
ax3 = fig.add_subplot(gs[0, 2])
sqrt_std_resid = np.sqrt(np.abs(standardized_residuals))
ax3.scatter(y_pred_mean, sqrt_std_resid, color='C0', s=60, alpha=0.7,
           edgecolors='black', linewidth=0.5)

# Add smooth
if len(y_pred_mean) > 5:
    try:
        smooth = savgol_filter(sqrt_std_resid[sort_idx_fitted],
                              window_length=min(11, len(y_pred_mean)//2*2+1),
                              polyorder=3)
        ax3.plot(y_pred_mean[sort_idx_fitted], smooth, 'g-', linewidth=2.5, alpha=0.7)
    except:
        pass

ax3.set_xlabel('Fitted values', fontsize=12, fontweight='bold')
ax3.set_ylabel('√|Standardized residuals|', fontsize=12, fontweight='bold')
ax3.set_title('C. Scale-Location Plot', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Panel D: Normal Q-Q
ax4 = fig.add_subplot(gs[1, 0])
stats.probplot(standardized_residuals, dist="norm", plot=ax4)
ax4.get_lines()[0].set_markerfacecolor('C0')
ax4.get_lines()[0].set_markeredgecolor('black')
ax4.get_lines()[0].set_markersize(8)
ax4.get_lines()[0].set_alpha(0.7)
ax4.get_lines()[1].set_color('red')
ax4.get_lines()[1].set_linewidth(2)
ax4.set_title('D. Normal Q-Q Plot', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Panel E: Histogram of standardized residuals
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(standardized_residuals, bins=15, alpha=0.7, color='C0',
        edgecolor='black', linewidth=1, density=True)
# Overlay normal distribution
x_norm = np.linspace(standardized_residuals.min(), standardized_residuals.max(), 100)
ax5.plot(x_norm, stats.norm.pdf(x_norm, 0, 1), 'r-', linewidth=2.5,
        label='N(0,1)', alpha=0.7)
ax5.set_xlabel('Standardized residuals', fontsize=12, fontweight='bold')
ax5.set_ylabel('Density', fontsize=12, fontweight='bold')
ax5.set_title('E. Residual Distribution', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Panel F: Residuals vs Observation Order
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(range(len(residuals)), residuals, color='C0', s=60, alpha=0.7,
           edgecolors='black', linewidth=0.5)
ax6.plot(range(len(residuals)), residuals, 'C0-', alpha=0.2, linewidth=1)
ax6.axhline(0, color='red', linestyle='--', linewidth=2)
ax6.axhline(2*residuals.std(), color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax6.axhline(-2*residuals.std(), color='red', linestyle=':', linewidth=1.5, alpha=0.5)
ax6.set_xlabel('Observation order', fontsize=12, fontweight='bold')
ax6.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax6.set_title('F. Residuals vs Order', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.suptitle('Residual Diagnostics: Detailed Analysis',
             fontsize=16, fontweight='bold')
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/residual_diagnostics.png',
            dpi=300, bbox_inches='tight')
print("Saved: residual_diagnostics.png")
plt.close()

# ============================================================================
# PLOT 5: DISTRIBUTION COMPARISON (Using ArviZ)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Plot many replicated datasets
n_plot = 100
for i in np.random.choice(y_rep.shape[0], n_plot, replace=False):
    ax.hist(y_rep[i, :], bins=20, alpha=0.02, color='C0', edgecolor='none', density=True)

# Plot observed data more prominently
ax.hist(y_obs, bins=20, alpha=0.8, color='red', edgecolor='black',
       linewidth=2, density=True, label='Observed data')

# Add KDE for observed
from scipy.stats import gaussian_kde
kde = gaussian_kde(y_obs)
x_kde = np.linspace(y_obs.min()-0.2, y_obs.max()+0.2, 200)
ax.plot(x_kde, kde(x_kde), 'r-', linewidth=3, label='Observed KDE', alpha=0.7)

ax.set_xlabel('Y', fontsize=13, fontweight='bold')
ax.set_ylabel('Density', fontsize=13, fontweight='bold')
ax.set_title('Distribution Comparison: Observed vs Posterior Predictive Replicates',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/distribution_comparison.png',
            dpi=300, bbox_inches='tight')
print("Saved: distribution_comparison.png")
plt.close()

print("\nAll plots generated successfully!")
