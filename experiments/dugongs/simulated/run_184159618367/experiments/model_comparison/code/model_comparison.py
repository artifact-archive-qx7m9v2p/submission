"""
Comprehensive Bayesian Model Comparison and Assessment
========================================================

Comparing two models:
1. Experiment 1: Asymptotic Exponential (Y = α - β*exp(-γ*x))
2. Experiment 3: Log-Log Power Law (log(Y) = α + β*log(x))

This script performs:
- Individual model assessment (LOO, calibration, coverage)
- Model comparison using ArviZ
- Visualization of comparisons
- Decision recommendation
"""

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
az.style.use('arviz-darkgrid')

# Paths
DATA_PATH = Path('/workspace/data/data.csv')
EXP1_PATH = Path('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
EXP3_PATH = Path('/workspace/experiments/experiment_3/posterior_inference/diagnostics/posterior_inference.netcdf')
OUTPUT_DIR = Path('/workspace/experiments/model_comparison')
PLOTS_DIR = OUTPUT_DIR / 'plots'

print("="*80)
print("BAYESIAN MODEL COMPARISON AND ASSESSMENT")
print("="*80)

# Load data
print("\n[1/7] Loading data...")
data = pd.read_csv(DATA_PATH)
x_obs = data['x'].values
y_obs = data['Y'].values
n_obs = len(y_obs)
print(f"Data: {n_obs} observations")
print(f"x range: [{x_obs.min():.1f}, {x_obs.max():.1f}]")
print(f"Y range: [{y_obs.min():.3f}, {y_obs.max():.3f}]")

# Load models
print("\n[2/7] Loading model inference data...")
print("\nExperiment 1: Asymptotic Exponential")
idata1 = az.from_netcdf(EXP1_PATH)
print(f"  Posterior shape: {idata1.posterior.dims}")
print(f"  Parameters: {list(idata1.posterior.data_vars.keys())}")
print(f"  Has log_likelihood: {'log_likelihood' in idata1.groups()}")

print("\nExperiment 3: Log-Log Power Law")
idata3 = az.from_netcdf(EXP3_PATH)
print(f"  Posterior shape: {idata3.posterior.dims}")
print(f"  Parameters: {list(idata3.posterior.data_vars.keys())}")
print(f"  Has log_likelihood: {'log_likelihood' in idata3.groups()}")

# Verify log_likelihood exists for both models
if 'log_likelihood' not in idata1.groups():
    raise ValueError("Experiment 1 missing log_likelihood group - cannot perform LOO-CV")
if 'log_likelihood' not in idata3.groups():
    raise ValueError("Experiment 3 missing log_likelihood group - cannot perform LOO-CV")

# Compute LOO for both models
print("\n[3/7] Computing LOO-CV diagnostics...")
print("\n--- Experiment 1: Asymptotic Exponential ---")
loo1 = az.loo(idata1, pointwise=True)
print(f"ELPD_loo: {loo1.elpd_loo:.2f} ± {loo1.se:.2f}")
print(f"p_loo: {loo1.p_loo:.2f}")

# Pareto k diagnostics
k1 = loo1.pareto_k.values
n_high_k1 = np.sum(k1 > 0.7)
n_moderate_k1 = np.sum((k1 > 0.5) & (k1 <= 0.7))
print(f"\nPareto k diagnostics:")
print(f"  k > 0.7 (bad): {n_high_k1}/{n_obs}")
print(f"  0.5 < k <= 0.7 (moderate): {n_moderate_k1}/{n_obs}")
print(f"  k <= 0.5 (good): {np.sum(k1 <= 0.5)}/{n_obs}")
print(f"  Max k: {k1.max():.3f}")

print("\n--- Experiment 3: Log-Log Power Law ---")
loo3 = az.loo(idata3, pointwise=True)
print(f"ELPD_loo: {loo3.elpd_loo:.2f} ± {loo3.se:.2f}")
print(f"p_loo: {loo3.p_loo:.2f}")

# Pareto k diagnostics
k3 = loo3.pareto_k.values
n_high_k3 = np.sum(k3 > 0.7)
n_moderate_k3 = np.sum((k3 > 0.5) & (k3 <= 0.7))
print(f"\nPareto k diagnostics:")
print(f"  k > 0.7 (bad): {n_high_k3}/{n_obs}")
print(f"  0.5 < k <= 0.7 (moderate): {n_moderate_k3}/{n_obs}")
print(f"  k <= 0.5 (good): {np.sum(k3 <= 0.5)}/{n_obs}")
print(f"  Max k: {k3.max():.3f}")

# Model comparison
print("\n[4/7] Comparing models with az.compare()...")
comparison_dict = {
    "Exp1_Asymptotic": idata1,
    "Exp3_LogLog": idata3
}
compare_df = az.compare(comparison_dict, ic='loo', method='stacking')
print("\n", compare_df)
print("\nColumn names:", compare_df.columns.tolist())

# Extract comparison metrics
# The best model has rank 0, second best has rank 1
best_model = compare_df.index[0]
second_model = compare_df.index[1]

# Get ELPD values
elpd1 = compare_df.loc['Exp1_Asymptotic', 'elpd_loo']
elpd3 = compare_df.loc['Exp3_LogLog', 'elpd_loo']

# Get difference - for the second-ranked model, d_loo is the difference from the best
delta_elpd = compare_df.loc[second_model, 'elpd_diff']
delta_se = compare_df.loc[second_model, 'dse']

print(f"\nELPD Exp1: {elpd1:.2f}")
print(f"ELPD Exp3: {elpd3:.2f}")
print(f"ΔELPD ({second_model} - {best_model}): {delta_elpd:.2f} ± {delta_se:.2f}")
print(f"Decision threshold (2×SE): {2*delta_se:.2f}")

if abs(delta_elpd) > 2 * delta_se:
    print(f"VERDICT: {best_model} is significantly better (ΔELPD > 2×SE)")
else:
    print("VERDICT: Models are statistically tied in predictive performance")

# Individual model assessments
print("\n[5/7] Individual model assessment...")

# Function to compute predictions
def get_predictions_exp1(idata, x):
    """Get posterior predictive samples for Experiment 1"""
    alpha = idata.posterior['alpha'].values.flatten()
    beta = idata.posterior['beta'].values.flatten()
    gamma = idata.posterior['gamma'].values.flatten()

    # Reshape for broadcasting
    n_samples = len(alpha)
    n_x = len(x)

    y_pred = np.zeros((n_samples, n_x))
    for i, xi in enumerate(x):
        y_pred[:, i] = alpha - beta * np.exp(-gamma * xi)

    return y_pred

def get_predictions_exp3(idata, x):
    """Get posterior predictive samples for Experiment 3"""
    alpha = idata.posterior['alpha'].values.flatten()
    beta = idata.posterior['beta'].values.flatten()

    # Reshape for broadcasting
    n_samples = len(alpha)
    n_x = len(x)

    y_pred = np.zeros((n_samples, n_x))
    for i, xi in enumerate(x):
        # log(Y) = alpha + beta*log(x), so Y = exp(alpha + beta*log(x))
        y_pred[:, i] = np.exp(alpha + beta * np.log(xi))

    return y_pred

# Get predictions for observed x values
print("\nExperiment 1:")
y_pred1 = get_predictions_exp1(idata1, x_obs)
y_mean1 = y_pred1.mean(axis=0)
y_lower1 = np.percentile(y_pred1, 5, axis=0)
y_upper1 = np.percentile(y_pred1, 95, axis=0)

# Compute metrics
residuals1 = y_obs - y_mean1
rmse1 = np.sqrt(np.mean(residuals1**2))
mae1 = np.mean(np.abs(residuals1))
coverage1 = np.mean((y_obs >= y_lower1) & (y_obs <= y_upper1))

print(f"  RMSE: {rmse1:.4f}")
print(f"  MAE: {mae1:.4f}")
print(f"  90% PI Coverage: {coverage1:.2%} ({int(coverage1*n_obs)}/{n_obs})")

print("\nExperiment 3:")
y_pred3 = get_predictions_exp3(idata3, x_obs)
y_mean3 = y_pred3.mean(axis=0)
y_lower3 = np.percentile(y_pred3, 5, axis=0)
y_upper3 = np.percentile(y_pred3, 95, axis=0)

# Compute metrics
residuals3 = y_obs - y_mean3
rmse3 = np.sqrt(np.mean(residuals3**2))
mae3 = np.mean(np.abs(residuals3))
coverage3 = np.mean((y_obs >= y_lower3) & (y_obs <= y_upper3))

print(f"  RMSE: {rmse3:.4f}")
print(f"  MAE: {mae3:.4f}")
print(f"  90% PI Coverage: {coverage3:.2%} ({int(coverage3*n_obs)}/{n_obs})")

# Create visualizations
print("\n[6/7] Creating comparison visualizations...")

# 1. Model comparison plot
fig = plt.figure(figsize=(12, 4))
ax = az.plot_compare(compare_df, insample_dev=False, plot_ic_diff=True,
                      plot_kwargs={'color_ic': 'C0', 'marker_ic': 'o'})
plt.title('Model Comparison: LOO-CV', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'loo_comparison.png', dpi=300, bbox_inches='tight')
print(f"  Saved: loo_comparison.png")
plt.close()

# 2. Pareto k diagnostics side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

az.plot_khat(loo1, ax=axes[0], show_bins=True)
axes[0].set_title('Experiment 1: Asymptotic Exponential\nPareto k Diagnostics',
                  fontsize=12, fontweight='bold')
axes[0].axhline(y=0.7, color='red', linestyle='--', linewidth=1, alpha=0.7, label='k=0.7 threshold')
axes[0].legend()

az.plot_khat(loo3, ax=axes[1], show_bins=True)
axes[1].set_title('Experiment 3: Log-Log Power Law\nPareto k Diagnostics',
                  fontsize=12, fontweight='bold')
axes[1].axhline(y=0.7, color='red', linestyle='--', linewidth=1, alpha=0.7, label='k=0.7 threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'pareto_k_comparison.png', dpi=300, bbox_inches='tight')
print(f"  Saved: pareto_k_comparison.png")
plt.close()

# 3. LOO-PIT calibration plots side-by-side
# Note: Skipping LOO-PIT plots as posterior_predictive not available in the same InferenceData
print("  Skipped: loo_pit_comparison.png (posterior_predictive not in InferenceData)")

# 4. Side-by-side model fits
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Create smooth prediction grid
x_grid = np.linspace(x_obs.min(), x_obs.max(), 200)
y_grid1 = get_predictions_exp1(idata1, x_grid)
y_grid3 = get_predictions_exp3(idata3, x_grid)

# Experiment 1
axes[0].scatter(x_obs, y_obs, alpha=0.6, s=60, color='black', label='Observed', zorder=5)
axes[0].plot(x_grid, y_grid1.mean(axis=0), 'C0', linewidth=2.5, label='Posterior mean', zorder=3)
axes[0].fill_between(x_grid,
                      np.percentile(y_grid1, 5, axis=0),
                      np.percentile(y_grid1, 95, axis=0),
                      alpha=0.3, color='C0', label='90% CI', zorder=2)
axes[0].set_xlabel('x', fontsize=11)
axes[0].set_ylabel('Y', fontsize=11)
axes[0].set_title(f'Experiment 1: Asymptotic Exponential\nELPD={loo1.elpd_loo:.1f}±{loo1.se:.1f}, RMSE={rmse1:.4f}',
                  fontsize=12, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Experiment 3
axes[1].scatter(x_obs, y_obs, alpha=0.6, s=60, color='black', label='Observed', zorder=5)
axes[1].plot(x_grid, y_grid3.mean(axis=0), 'C1', linewidth=2.5, label='Posterior mean', zorder=3)
axes[1].fill_between(x_grid,
                      np.percentile(y_grid3, 5, axis=0),
                      np.percentile(y_grid3, 95, axis=0),
                      alpha=0.3, color='C1', label='90% CI', zorder=2)
axes[1].set_xlabel('x', fontsize=11)
axes[1].set_ylabel('Y', fontsize=11)
axes[1].set_title(f'Experiment 3: Log-Log Power Law\nELPD={loo3.elpd_loo:.1f}±{loo3.se:.1f}, RMSE={rmse3:.4f}',
                  fontsize=12, fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'model_fits_comparison.png', dpi=300, bbox_inches='tight')
print(f"  Saved: model_fits_comparison.png")
plt.close()

# 5. Residual analysis side-by-side
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Experiment 1 residuals
axes[0, 0].scatter(y_mean1, residuals1, alpha=0.6, s=60, color='C0')
axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
axes[0, 0].set_xlabel('Predicted Y', fontsize=11)
axes[0, 0].set_ylabel('Residuals', fontsize=11)
axes[0, 0].set_title('Experiment 1: Residuals vs Predicted', fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(residuals1, bins=15, alpha=0.7, color='C0', edgecolor='black')
axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
axes[0, 1].set_xlabel('Residuals', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title(f'Experiment 1: Residual Distribution\nMean={residuals1.mean():.4f}, SD={residuals1.std():.4f}',
                     fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Experiment 3 residuals
axes[1, 0].scatter(y_mean3, residuals3, alpha=0.6, s=60, color='C1')
axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
axes[1, 0].set_xlabel('Predicted Y', fontsize=11)
axes[1, 0].set_ylabel('Residuals', fontsize=11)
axes[1, 0].set_title('Experiment 3: Residuals vs Predicted', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(residuals3, bins=15, alpha=0.7, color='C1', edgecolor='black')
axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
axes[1, 1].set_xlabel('Residuals', fontsize=11)
axes[1, 1].set_ylabel('Frequency', fontsize=11)
axes[1, 1].set_title(f'Experiment 3: Residual Distribution\nMean={residuals3.mean():.4f}, SD={residuals3.std():.4f}',
                     fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'residual_analysis_comparison.png', dpi=300, bbox_inches='tight')
print(f"  Saved: residual_analysis_comparison.png")
plt.close()

# 6. Integrated comparison dashboard
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Row 1: Model fits
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(x_obs, y_obs, alpha=0.6, s=80, color='black', label='Observed', zorder=6)
ax1.plot(x_grid, y_grid1.mean(axis=0), 'C0', linewidth=2.5, label='Exp1: Asymptotic', zorder=4)
ax1.fill_between(x_grid,
                 np.percentile(y_grid1, 5, axis=0),
                 np.percentile(y_grid1, 95, axis=0),
                 alpha=0.2, color='C0', zorder=2)
ax1.plot(x_grid, y_grid3.mean(axis=0), 'C1', linewidth=2.5, label='Exp3: Log-Log', zorder=5, linestyle='--')
ax1.fill_between(x_grid,
                 np.percentile(y_grid3, 5, axis=0),
                 np.percentile(y_grid3, 95, axis=0),
                 alpha=0.2, color='C1', zorder=1)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_title('Model Comparison: Predictive Fits', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(True, alpha=0.3)

# Row 1: Summary metrics table
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
metrics_text = f"""
SUMMARY METRICS

Experiment 1 (Asymptotic):
  ELPD: {loo1.elpd_loo:.1f} ± {loo1.se:.1f}
  p_loo: {loo1.p_loo:.1f}
  RMSE: {rmse1:.4f}
  MAE: {mae1:.4f}
  Coverage: {coverage1:.1%}
  Bad k: {n_high_k1}/{n_obs}

Experiment 3 (Log-Log):
  ELPD: {loo3.elpd_loo:.1f} ± {loo3.se:.1f}
  p_loo: {loo3.p_loo:.1f}
  RMSE: {rmse3:.4f}
  MAE: {mae3:.4f}
  Coverage: {coverage3:.1%}
  Bad k: {n_high_k3}/{n_obs}

ΔELPD: {delta_elpd:.1f} ± {delta_se:.1f}
Threshold: {2*delta_se:.1f}
Winner: {best_model}
"""
ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Row 2: Pareto k
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(range(len(k1)), k1, alpha=0.6, s=40, color='C0', label='Exp1')
ax3.axhline(y=0.7, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7)
ax3.set_xlabel('Observation Index', fontsize=11)
ax3.set_ylabel('Pareto k', fontsize=11)
ax3.set_title('Exp1: Pareto k Values', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(range(len(k3)), k3, alpha=0.6, s=40, color='C1', label='Exp3')
ax4.axhline(y=0.7, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax4.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7)
ax4.set_xlabel('Observation Index', fontsize=11)
ax4.set_ylabel('Pareto k', fontsize=11)
ax4.set_title('Exp3: Pareto k Values', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Row 2: Coverage comparison
ax5 = fig.add_subplot(gs[1, 2])
in_interval1 = (y_obs >= y_lower1) & (y_obs <= y_upper1)
in_interval3 = (y_obs >= y_lower3) & (y_obs <= y_upper3)
coverage_comparison = pd.DataFrame({
    'Model': ['Exp1\nAsymptotic', 'Exp3\nLog-Log', 'Target\n90%'],
    'Coverage': [coverage1, coverage3, 0.90]
})
bars = ax5.bar(coverage_comparison['Model'], coverage_comparison['Coverage'],
               color=['C0', 'C1', 'green'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.axhline(y=0.90, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target')
ax5.set_ylabel('Coverage', fontsize=11)
ax5.set_title('90% Posterior Interval Coverage', fontsize=11, fontweight='bold')
ax5.set_ylim([0.7, 1.0])
ax5.grid(True, alpha=0.3, axis='y')
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Row 3: Residuals comparison
ax6 = fig.add_subplot(gs[2, 0])
ax6.scatter(y_mean1, residuals1, alpha=0.6, s=50, color='C0', label='Exp1')
ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax6.set_xlabel('Predicted Y', fontsize=11)
ax6.set_ylabel('Residuals', fontsize=11)
ax6.set_title('Exp1: Residuals vs Fitted', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)

ax7 = fig.add_subplot(gs[2, 1])
ax7.scatter(y_mean3, residuals3, alpha=0.6, s=50, color='C1', label='Exp3')
ax7.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax7.set_xlabel('Predicted Y', fontsize=11)
ax7.set_ylabel('Residuals', fontsize=11)
ax7.set_title('Exp3: Residuals vs Fitted', fontsize=11, fontweight='bold')
ax7.grid(True, alpha=0.3)

# Row 3: Error metrics comparison
ax8 = fig.add_subplot(gs[2, 2])
metrics_df = pd.DataFrame({
    'Metric': ['RMSE', 'RMSE', 'MAE', 'MAE'],
    'Model': ['Exp1', 'Exp3', 'Exp1', 'Exp3'],
    'Value': [rmse1, rmse3, mae1, mae3]
})
x_pos = np.arange(2)
width = 0.35
ax8.bar(x_pos - width/2, [rmse1, mae1], width, label='Exp1', color='C0', alpha=0.7, edgecolor='black')
ax8.bar(x_pos + width/2, [rmse3, mae3], width, label='Exp3', color='C1', alpha=0.7, edgecolor='black')
ax8.set_ylabel('Error', fontsize=11)
ax8.set_title('Prediction Error Metrics', fontsize=11, fontweight='bold')
ax8.set_xticks(x_pos)
ax8.set_xticklabels(['RMSE', 'MAE'])
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

plt.suptitle('Integrated Model Comparison Dashboard', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / 'integrated_comparison_dashboard.png', dpi=300, bbox_inches='tight')
print(f"  Saved: integrated_comparison_dashboard.png")
plt.close()

# 7. Parameter comparison
print("\n[7/7] Generating summary statistics...")
print("\nParameter summaries:")
print("\nExperiment 1 (Asymptotic Exponential):")
print(az.summary(idata1, var_names=['alpha', 'beta', 'gamma', 'sigma']))

print("\nExperiment 3 (Log-Log Power Law):")
print(az.summary(idata3, var_names=['alpha', 'beta', 'sigma']))

# Create parameter comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Experiment 1 parameters
az.plot_forest(idata1, var_names=['alpha', 'beta', 'gamma', 'sigma'],
               combined=True, ax=axes[0], figsize=(7, 5))
axes[0].set_title('Experiment 1: Parameter Estimates\n(Asymptotic Exponential)',
                  fontsize=12, fontweight='bold')

# Experiment 3 parameters
az.plot_forest(idata3, var_names=['alpha', 'beta', 'sigma'],
               combined=True, ax=axes[1], figsize=(7, 5))
axes[1].set_title('Experiment 3: Parameter Estimates\n(Log-Log Power Law)',
                  fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_comparison.png', dpi=300, bbox_inches='tight')
print(f"  Saved: parameter_comparison.png")
plt.close()

# Create summary dictionary
summary = {
    'Experiment 1 (Asymptotic)': {
        'ELPD': f"{loo1.elpd_loo:.2f} ± {loo1.se:.2f}",
        'p_loo': f"{loo1.p_loo:.2f}",
        'RMSE': f"{rmse1:.4f}",
        'MAE': f"{mae1:.4f}",
        'Coverage_90': f"{coverage1:.2%}",
        'Pareto_k_bad': f"{n_high_k1}/{n_obs}",
        'Pareto_k_max': f"{k1.max():.3f}",
        'Parameters': 4  # alpha, beta, gamma, sigma
    },
    'Experiment 3 (Log-Log)': {
        'ELPD': f"{loo3.elpd_loo:.2f} ± {loo3.se:.2f}",
        'p_loo': f"{loo3.p_loo:.2f}",
        'RMSE': f"{rmse3:.4f}",
        'MAE': f"{mae3:.4f}",
        'Coverage_90': f"{coverage3:.2%}",
        'Pareto_k_bad': f"{n_high_k3}/{n_obs}",
        'Pareto_k_max': f"{k3.max():.3f}",
        'Parameters': 3  # alpha, beta, sigma
    },
    'Comparison': {
        'Best_model': best_model,
        'DELTA_ELPD': f"{delta_elpd:.2f} ± {delta_se:.2f}",
        'Decision_threshold_2SE': f"{2*delta_se:.2f}",
        'Significant_difference': abs(delta_elpd) > 2 * delta_se
    }
}

# Save summary to file
summary_df = pd.DataFrame(summary).T
summary_df.to_csv(OUTPUT_DIR / 'comparison_summary.csv')
print(f"\nSummary saved to: comparison_summary.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  - Code: {OUTPUT_DIR / 'code/model_comparison.py'}")
print(f"  - Plots: {PLOTS_DIR}/")
print(f"  - Summary: {OUTPUT_DIR / 'comparison_summary.csv'}")
print("\nRecommendation will be written to comparison_report.md")
