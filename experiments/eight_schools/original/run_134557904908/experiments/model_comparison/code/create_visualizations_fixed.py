"""
Create Comprehensive Comparison Visualizations
==============================================
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Paths
DATA_PATH = '/workspace/data/data.csv'
MODEL1_PATH = '/workspace/experiments/model_comparison/idata1_with_predictions.netcdf'
MODEL2_PATH = '/workspace/experiments/model_comparison/idata2_with_predictions.netcdf'
PLOTS_DIR = '/workspace/experiments/model_comparison/plots'

print("Loading data and models...")

# Load data
data = pd.read_csv(DATA_PATH)
y_obs = data['y'].values
sigma = data['sigma'].values
n_studies = len(y_obs)

# Load InferenceData
idata1 = az.from_netcdf(MODEL1_PATH)
idata2 = az.from_netcdf(MODEL2_PATH)

# Compute LOO
loo1 = az.loo(idata1, pointwise=True)
loo2 = az.loo(idata2, pointwise=True)

# Get predictions
y_pred1 = idata1.posterior_predictive['y_obs'].values.reshape(-1, n_studies)
y_pred2 = idata2.posterior_predictive['y_obs'].values.reshape(-1, n_studies)
y_pred1_mean = y_pred1.mean(axis=0)
y_pred2_mean = y_pred2.mean(axis=0)

# Metrics
rmse1 = np.sqrt(np.mean((y_obs - y_pred1_mean)**2))
mae1 = np.mean(np.abs(y_obs - y_pred1_mean))
rmse2 = np.sqrt(np.mean((y_obs - y_pred2_mean)**2))
mae2 = np.mean(np.abs(y_obs - y_pred2_mean))

print(f"Creating visualizations in: {PLOTS_DIR}")
print()

# ============================================================================
# PLOT 1: LOO COMPARISON (Primary Decision Visual)
# ============================================================================

print("1. Creating LOO comparison plot...")

fig, ax = plt.subplots(figsize=(10, 6))

comparison = az.compare({'Model 1 (Fixed)': idata1, 'Model 2 (Random)': idata2})

# Extract data for plotting
models = ['Model 1\n(Fixed)', 'Model 2\n(Random)']
elpd_vals = [loo1.elpd_loo, loo2.elpd_loo]
se_vals = [loo1.se, loo2.se]

# Create bar plot with error bars
x_pos = np.arange(len(models))
bars = ax.bar(x_pos, elpd_vals, yerr=se_vals, capsize=10,
               alpha=0.7, color=['#2E86AB', '#A23B72'])

# Add value labels on bars
for i, (bar, val, se) in enumerate(zip(bars, elpd_vals, se_vals)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}\n± {se:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add delta annotation
delta_elpd = elpd_vals[1] - elpd_vals[0]
delta_se = comparison.iloc[1]['dse']
ax.annotate(f'ΔELPD = {delta_elpd:.2f} ± {delta_se:.2f}\n|ΔELPD/SE| = {abs(delta_elpd/delta_se):.2f}',
            xy=(0.5, max(elpd_vals) + max(se_vals)),
            xytext=(0.5, max(elpd_vals) + 1.5*max(se_vals)),
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('ELPD LOO (Expected Log Pointwise Predictive Density)', fontsize=12)
ax.set_title('LOO-CV Model Comparison: Higher is Better', fontsize=14, fontweight='bold')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.grid(axis='y', alpha=0.3)

# Add interpretation text
if abs(delta_elpd/delta_se) < 2:
    interpretation = "No substantial difference (|ΔELPD/SE| < 2)\n→ Prefer simpler model by parsimony"
    color = 'green'
else:
    interpretation = "Models are distinguishable (|ΔELPD/SE| > 2)"
    color = 'red'

ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/1_loo_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 2: PREDICTIVE PERFORMANCE COMPARISON
# ============================================================================

print("2. Creating predictive performance comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Model 1: Observed vs Predicted
axes[0].scatter(y_obs, y_pred1_mean, s=100, alpha=0.6, color='#2E86AB',
                edgecolors='black', linewidths=1.5)
axes[0].plot([y_obs.min(), y_obs.max()], [y_obs.min(), y_obs.max()],
             'r--', lw=2, label='Perfect prediction')

# Add error bars (1 SD)
y_pred1_std = y_pred1.std(axis=0)
axes[0].errorbar(y_obs, y_pred1_mean, yerr=y_pred1_std, fmt='none',
                 ecolor='gray', alpha=0.3, capsize=3)

# Add study labels
for i in range(n_studies):
    axes[0].annotate(f'{i+1}', (y_obs[i], y_pred1_mean[i]),
                     fontsize=8, ha='center', va='center', fontweight='bold')

axes[0].text(0.05, 0.95, f'RMSE = {rmse1:.2f}\nMAE = {mae1:.2f}',
             transform=axes[0].transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[0].set_xlabel('Observed Effect Size', fontsize=12)
axes[0].set_ylabel('Predicted Effect Size', fontsize=12)
axes[0].set_title('Model 1 (Fixed-Effect)\nPredictive Performance',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Model 2: Observed vs Predicted
axes[1].scatter(y_obs, y_pred2_mean, s=100, alpha=0.6, color='#A23B72',
                edgecolors='black', linewidths=1.5)
axes[1].plot([y_obs.min(), y_obs.max()], [y_obs.min(), y_obs.max()],
             'r--', lw=2, label='Perfect prediction')

# Add error bars (1 SD)
y_pred2_std = y_pred2.std(axis=0)
axes[1].errorbar(y_obs, y_pred2_mean, yerr=y_pred2_std, fmt='none',
                 ecolor='gray', alpha=0.3, capsize=3)

# Add study labels
for i in range(n_studies):
    axes[1].annotate(f'{i+1}', (y_obs[i], y_pred2_mean[i]),
                     fontsize=8, ha='center', va='center', fontweight='bold')

axes[1].text(0.05, 0.95, f'RMSE = {rmse2:.2f}\nMAE = {mae2:.2f}',
             transform=axes[1].transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[1].set_xlabel('Observed Effect Size', fontsize=12)
axes[1].set_ylabel('Predicted Effect Size', fontsize=12)
axes[1].set_title('Model 2 (Random-Effects)\nPredictive Performance',
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/2_predictive_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 3: PARETO K DIAGNOSTICS
# ============================================================================

print("3. Creating Pareto k diagnostics plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Model 1 Pareto k
az.plot_khat(loo1, ax=axes[0], show_bins=True)
axes[0].set_title('Model 1 (Fixed-Effect)\nPareto k Diagnostics',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Data Point', fontsize=11)
axes[0].set_ylabel('Pareto k', fontsize=11)
axes[0].axhline(0.7, color='red', linestyle='--', alpha=0.5,
                label='k = 0.7 (problematic)')
axes[0].legend(fontsize=9)

# Model 2 Pareto k
az.plot_khat(loo2, ax=axes[1], show_bins=True)
axes[1].set_title('Model 2 (Random-Effects)\nPareto k Diagnostics',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Data Point', fontsize=11)
axes[1].set_ylabel('Pareto k', fontsize=11)
axes[1].axhline(0.7, color='red', linestyle='--', alpha=0.5,
                label='k = 0.7 (problematic)')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/3_pareto_k_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 4: PARAMETER COMPARISON (FOREST PLOT)
# ============================================================================

print("4. Creating parameter comparison forest plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Extract parameters
theta1 = idata1.posterior['theta'].values.flatten()
mu2 = idata2.posterior['mu'].values.flatten()

# Compute summaries
theta1_mean = theta1.mean()
theta1_hdi = az.hdi(idata1, var_names=['theta'], hdi_prob=0.95)['theta'].values

mu2_mean = mu2.mean()
mu2_hdi = az.hdi(idata2, var_names=['mu'], hdi_prob=0.95)['mu'].values

# Plot HDIs
y_pos = [0, 1]
params = ['Model 1: θ', 'Model 2: μ']
means = [theta1_mean, mu2_mean]
hdis = [theta1_hdi, mu2_hdi]
colors = ['#2E86AB', '#A23B72']

for i, (param, mean, hdi, color) in enumerate(zip(params, means, hdis, colors)):
    # HDI line
    ax.plot([hdi[0], hdi[1]], [y_pos[i], y_pos[i]],
            linewidth=3, color=color, alpha=0.7)
    # Point estimate
    ax.plot(mean, y_pos[i], 'o', markersize=12, color=color,
            markeredgecolor='black', markeredgewidth=1.5)
    # Value label
    ax.text(mean, y_pos[i] + 0.15, f'{mean:.2f}',
            ha='center', fontsize=11, fontweight='bold')
    # HDI label
    ax.text(hdi[1] + 0.5, y_pos[i], f'[{hdi[0]:.2f}, {hdi[1]:.2f}]',
            va='center', fontsize=10, color=color)

ax.set_yticks(y_pos)
ax.set_yticklabels(params, fontsize=12)
ax.set_xlabel('Effect Size (Overall Effect)', fontsize=12)
ax.set_title('Parameter Comparison: Overall Effect Estimates\n95% HDI',
             fontsize=14, fontweight='bold')
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='x', alpha=0.3)
ax.set_ylim(-0.5, 1.5)

# Add difference annotation
diff = mu2_mean - theta1_mean
ax.text(0.98, 0.98, f'Difference: {diff:.2f} ({(diff/theta1_mean)*100:.1f}%)',
        transform=ax.transAxes, fontsize=11, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/4_parameter_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 5: SHRINKAGE PLOT (MODEL 2 ONLY)
# ============================================================================

print("5. Creating shrinkage plot (Model 2)...")

fig, ax = plt.subplots(figsize=(10, 7))

# Get study-specific estimates
theta_i = idata2.posterior['theta'].values
theta_i_mean = theta_i.mean(axis=(0, 1))
theta_i_hdi = az.hdi(idata2, var_names=['theta'], hdi_prob=0.95)['theta'].values

# Plot observed data
ax.errorbar(range(1, n_studies+1), y_obs, yerr=sigma,
            fmt='o', markersize=10, capsize=5, linewidth=2,
            color='black', alpha=0.7, label='Observed data ± SE')

# Plot study-specific estimates
for i in range(n_studies):
    ax.plot([i+1, i+1], [theta_i_hdi[i, 0], theta_i_hdi[i, 1]],
            linewidth=2, color='#A23B72', alpha=0.6)
ax.plot(range(1, n_studies+1), theta_i_mean, 's', markersize=8,
        color='#A23B72', label='Model 2: θᵢ (95% HDI)',
        markeredgecolor='black', markeredgewidth=1)

# Plot overall mean
mu2_mean = mu2.mean()
mu2_hdi = az.hdi(idata2, var_names=['mu'], hdi_prob=0.95)['mu'].values
ax.axhline(mu2_mean, color='red', linestyle='--', linewidth=2,
           label=f'Model 2: μ = {mu2_mean:.2f}')
ax.fill_between(range(1, n_studies+1), mu2_hdi[0], mu2_hdi[1],
                color='red', alpha=0.1)

# Add shrinkage arrows
for i in range(n_studies):
    if abs(y_obs[i] - theta_i_mean[i]) > 0.5:  # Only show significant shrinkage
        ax.annotate('', xy=(i+1, theta_i_mean[i]), xytext=(i+1, y_obs[i]),
                   arrowprops=dict(arrowstyle='->', color='green',
                                  lw=1.5, alpha=0.5))

ax.set_xlabel('Study', fontsize=12)
ax.set_ylabel('Effect Size', fontsize=12)
ax.set_title('Model 2: Shrinkage of Study-Specific Estimates\n(Partial Pooling toward Overall Mean)',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(1, n_studies+1))
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3)
ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/5_shrinkage_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 6: RESIDUAL COMPARISON
# ============================================================================

print("6. Creating residual comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

resid1 = y_obs - y_pred1_mean
resid2 = y_obs - y_pred2_mean

# Standardized residuals
std_resid1 = resid1 / sigma
std_resid2 = resid2 / sigma

# Plot 1: Residuals vs Study (Model 1)
axes[0, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].scatter(range(1, n_studies+1), resid1, s=100,
                   color='#2E86AB', alpha=0.6, edgecolors='black', linewidths=1.5)
axes[0, 0].errorbar(range(1, n_studies+1), resid1, yerr=sigma,
                    fmt='none', ecolor='gray', alpha=0.3, capsize=3)
for i in range(n_studies):
    axes[0, 0].annotate(f'{i+1}', (i+1, resid1[i]),
                        fontsize=8, ha='center', va='center', fontweight='bold')
axes[0, 0].set_xlabel('Study', fontsize=11)
axes[0, 0].set_ylabel('Residual', fontsize=11)
axes[0, 0].set_title('Model 1: Residuals', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xticks(range(1, n_studies+1))

# Plot 2: Residuals vs Study (Model 2)
axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0, 1].scatter(range(1, n_studies+1), resid2, s=100,
                   color='#A23B72', alpha=0.6, edgecolors='black', linewidths=1.5)
axes[0, 1].errorbar(range(1, n_studies+1), resid2, yerr=sigma,
                    fmt='none', ecolor='gray', alpha=0.3, capsize=3)
for i in range(n_studies):
    axes[0, 1].annotate(f'{i+1}', (i+1, resid2[i]),
                        fontsize=8, ha='center', va='center', fontweight='bold')
axes[0, 1].set_xlabel('Study', fontsize=11)
axes[0, 1].set_ylabel('Residual', fontsize=11)
axes[0, 1].set_title('Model 2: Residuals', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xticks(range(1, n_studies+1))

# Plot 3: Standardized Residuals (Model 1)
axes[1, 0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].axhline(-2, color='red', linestyle=':', alpha=0.3, label='±2 SD')
axes[1, 0].axhline(2, color='red', linestyle=':', alpha=0.3)
axes[1, 0].scatter(range(1, n_studies+1), std_resid1, s=100,
                   color='#2E86AB', alpha=0.6, edgecolors='black', linewidths=1.5)
for i in range(n_studies):
    axes[1, 0].annotate(f'{i+1}', (i+1, std_resid1[i]),
                        fontsize=8, ha='center', va='center', fontweight='bold')
axes[1, 0].set_xlabel('Study', fontsize=11)
axes[1, 0].set_ylabel('Standardized Residual', fontsize=11)
axes[1, 0].set_title('Model 1: Standardized Residuals', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].legend(fontsize=9)
axes[1, 0].set_xticks(range(1, n_studies+1))

# Plot 4: Standardized Residuals (Model 2)
axes[1, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].axhline(-2, color='red', linestyle=':', alpha=0.3, label='±2 SD')
axes[1, 1].axhline(2, color='red', linestyle=':', alpha=0.3)
axes[1, 1].scatter(range(1, n_studies+1), std_resid2, s=100,
                   color='#A23B72', alpha=0.6, edgecolors='black', linewidths=1.5)
for i in range(n_studies):
    axes[1, 1].annotate(f'{i+1}', (i+1, std_resid2[i]),
                        fontsize=8, ha='center', va='center', fontweight='bold')
axes[1, 1].set_xlabel('Study', fontsize=11)
axes[1, 1].set_ylabel('Standardized Residual', fontsize=11)
axes[1, 1].set_title('Model 2: Standardized Residuals', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].legend(fontsize=9)
axes[1, 1].set_xticks(range(1, n_studies+1))

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/6_residual_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PLOT 7: INTEGRATED COMPARISON DASHBOARD
# ============================================================================

print("7. Creating integrated comparison dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Panel 1: LOO Comparison
ax1 = fig.add_subplot(gs[0, :2])
models = ['Model 1\n(Fixed)', 'Model 2\n(Random)']
elpd_vals = [loo1.elpd_loo, loo2.elpd_loo]
se_vals = [loo1.se, loo2.se]
x_pos = np.arange(len(models))
bars = ax1.bar(x_pos, elpd_vals, yerr=se_vals, capsize=10,
               alpha=0.7, color=['#2E86AB', '#A23B72'])
for i, (bar, val, se) in enumerate(zip(bars, elpd_vals, se_vals)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}\n± {se:.2f}',
             ha='center', va='bottom', fontweight='bold', fontsize=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, fontsize=11)
ax1.set_ylabel('ELPD LOO', fontsize=11)
ax1.set_title('A. LOO-CV Comparison', fontsize=12, fontweight='bold', loc='left')
ax1.grid(axis='y', alpha=0.3)

# Panel 2: Complexity
ax2 = fig.add_subplot(gs[0, 2])
complexity_data = pd.DataFrame({
    'Model': ['M1', 'M2'],
    'Actual': [1, 10],
    'Effective': [loo1.p_loo, loo2.p_loo]
})
x = np.arange(len(complexity_data))
width = 0.35
ax2.bar(x - width/2, complexity_data['Actual'], width, label='Actual', alpha=0.7)
ax2.bar(x + width/2, complexity_data['Effective'], width, label='Effective (p_LOO)', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(complexity_data['Model'])
ax2.set_ylabel('Parameters', fontsize=11)
ax2.set_title('B. Complexity', fontsize=12, fontweight='bold', loc='left')
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Panel 3: Predictive Performance (Model 1)
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y_obs, y_pred1_mean, s=80, alpha=0.6, color='#2E86AB',
            edgecolors='black', linewidths=1)
ax3.plot([y_obs.min(), y_obs.max()], [y_obs.min(), y_obs.max()],
         'r--', lw=2, alpha=0.5)
ax3.text(0.05, 0.95, f'RMSE: {rmse1:.2f}',
         transform=ax3.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax3.set_xlabel('Observed', fontsize=10)
ax3.set_ylabel('Predicted', fontsize=10)
ax3.set_title('C. Model 1 Predictions', fontsize=12, fontweight='bold', loc='left')
ax3.grid(alpha=0.3)

# Panel 4: Predictive Performance (Model 2)
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(y_obs, y_pred2_mean, s=80, alpha=0.6, color='#A23B72',
            edgecolors='black', linewidths=1)
ax4.plot([y_obs.min(), y_obs.max()], [y_obs.min(), y_obs.max()],
         'r--', lw=2, alpha=0.5)
ax4.text(0.05, 0.95, f'RMSE: {rmse2:.2f}',
         transform=ax4.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax4.set_xlabel('Observed', fontsize=10)
ax4.set_ylabel('Predicted', fontsize=10)
ax4.set_title('D. Model 2 Predictions', fontsize=12, fontweight='bold', loc='left')
ax4.grid(alpha=0.3)

# Panel 5: Error Metrics
ax5 = fig.add_subplot(gs[1, 2])
metrics = ['RMSE', 'MAE']
m1_vals = [rmse1, mae1]
m2_vals = [rmse2, mae2]
x = np.arange(len(metrics))
width = 0.35
ax5.bar(x - width/2, m1_vals, width, label='Model 1', alpha=0.7, color='#2E86AB')
ax5.bar(x + width/2, m2_vals, width, label='Model 2', alpha=0.7, color='#A23B72')
ax5.set_xticks(x)
ax5.set_xticklabels(metrics)
ax5.set_ylabel('Error', fontsize=11)
ax5.set_title('E. Error Metrics', fontsize=12, fontweight='bold', loc='left')
ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.3)

# Panel 6: Pareto k (Model 1)
ax6 = fig.add_subplot(gs[2, 0])
pareto_k1 = loo1.pareto_k.values if hasattr(loo1.pareto_k, 'values') else loo1.pareto_k
ax6.scatter(range(1, n_studies+1), pareto_k1, s=80,
            color='#2E86AB', alpha=0.6, edgecolors='black', linewidths=1)
ax6.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='k=0.5')
ax6.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='k=0.7')
ax6.set_xlabel('Study', fontsize=10)
ax6.set_ylabel('Pareto k', fontsize=10)
ax6.set_title('F. Model 1 Pareto k', fontsize=12, fontweight='bold', loc='left')
ax6.set_xticks(range(1, n_studies+1))
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

# Panel 7: Pareto k (Model 2)
ax7 = fig.add_subplot(gs[2, 1])
pareto_k2 = loo2.pareto_k.values if hasattr(loo2.pareto_k, 'values') else loo2.pareto_k
ax7.scatter(range(1, n_studies+1), pareto_k2, s=80,
            color='#A23B72', alpha=0.6, edgecolors='black', linewidths=1)
ax7.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='k=0.5')
ax7.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='k=0.7')
ax7.set_xlabel('Study', fontsize=10)
ax7.set_ylabel('Pareto k', fontsize=10)
ax7.set_title('G. Model 2 Pareto k', fontsize=12, fontweight='bold', loc='left')
ax7.set_xticks(range(1, n_studies+1))
ax7.legend(fontsize=8)
ax7.grid(alpha=0.3)

# Panel 8: Parameter Comparison
ax8 = fig.add_subplot(gs[2, 2])
theta1_mean = idata1.posterior['theta'].values.mean()
mu2_mean = idata2.posterior['mu'].values.mean()
theta1_std = idata1.posterior['theta'].values.std()
mu2_std = idata2.posterior['mu'].values.std()
params = ['θ (M1)', 'μ (M2)']
means = [theta1_mean, mu2_mean]
stds = [theta1_std, mu2_std]
x = np.arange(len(params))
bars = ax8.bar(x, means, yerr=stds, capsize=10, alpha=0.7,
               color=['#2E86AB', '#A23B72'])
for bar, mean, std in zip(bars, means, stds):
    ax8.text(bar.get_x() + bar.get_width()/2., mean,
             f'{mean:.2f}\n±{std:.2f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(params)
ax8.set_ylabel('Effect Size', fontsize=11)
ax8.set_title('H. Overall Effect', fontsize=12, fontweight='bold', loc='left')
ax8.grid(axis='y', alpha=0.3)
ax8.axhline(0, color='gray', linestyle=':', alpha=0.5)

fig.suptitle('Comprehensive Model Comparison Dashboard',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(f'{PLOTS_DIR}/7_comparison_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

print()
print("="*80)
print("ALL VISUALIZATIONS CREATED")
print("="*80)
print(f"\nPlots saved to: {PLOTS_DIR}")
print("\nGenerated plots:")
print("  1. 1_loo_comparison.png - Primary decision visual")
print("  2. 2_predictive_performance.png - Prediction quality")
print("  3. 3_pareto_k_diagnostics.png - LOO reliability")
print("  4. 4_parameter_comparison.png - Effect size estimates")
print("  5. 5_shrinkage_plot.png - Model 2 partial pooling")
print("  6. 6_residual_comparison.png - Error analysis")
print("  7. 7_comparison_dashboard.png - Integrated overview")
