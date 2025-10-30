"""
Comprehensive Model Comparison: Experiment 1 (Hierarchical) vs Experiment 3 (Beta-Binomial)

This script performs:
1. LOO cross-validation comparison (with Pareto k diagnostics)
2. WAIC computation (alternative to LOO)
3. Calibration assessment (LOO-PIT)
4. Predictive metrics (RMSE, MAE, Deviance)
5. Comprehensive comparison visualizations
"""

import sys
sys.path.append('/tmp/agent-home/.local/lib/python3.13/site-packages')

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_DIR = Path("/workspace")
EXP1_DIR = BASE_DIR / "experiments/experiment_1"
EXP3_DIR = BASE_DIR / "experiments/experiment_3"
COMP_DIR = BASE_DIR / "experiments/model_comparison"
DATA_PATH = BASE_DIR / "data/data.csv"

# Create output directories
(COMP_DIR / "code").mkdir(parents=True, exist_ok=True)
(COMP_DIR / "plots").mkdir(parents=True, exist_ok=True)
(COMP_DIR / "diagnostics").mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MODEL COMPARISON: Experiment 1 vs Experiment 3")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA AND MODELS
# ============================================================================
print("\n1. Loading data and models...")

# Load observed data
data = pd.read_csv(DATA_PATH)
n_groups = len(data)
n_obs = data['n'].values
r_obs = data['r'].values
p_obs = r_obs / n_obs

print(f"   - Loaded data: {n_groups} groups, N={n_obs.sum()} trials, r={r_obs.sum()} successes")

# Load InferenceData objects
idata_exp1 = az.from_netcdf(EXP1_DIR / "posterior_inference/diagnostics/posterior_inference.netcdf")
idata_exp3 = az.from_netcdf(EXP3_DIR / "posterior_inference/diagnostics/posterior_inference.netcdf")

print("   - Loaded Experiment 1 InferenceData")
print("   - Loaded Experiment 3 InferenceData")

# Verify log_likelihood exists
assert 'log_likelihood' in idata_exp1.groups(), "Exp 1 missing log_likelihood"
assert 'log_likelihood' in idata_exp3.groups(), "Exp 3 missing log_likelihood"
print("   - Verified log_likelihood groups present")

# ============================================================================
# 2. LOO CROSS-VALIDATION COMPARISON
# ============================================================================
print("\n2. Computing LOO cross-validation...")

# Compute LOO for both models
loo_exp1 = az.loo(idata_exp1, pointwise=True)
loo_exp3 = az.loo(idata_exp3, pointwise=True)

print("\n   EXPERIMENT 1 (Hierarchical Binomial):")
print(f"   - ELPD_loo: {loo_exp1.elpd_loo:.2f} ± {loo_exp1.se:.2f}")
print(f"   - p_loo: {loo_exp1.p_loo:.2f}")
print(f"   - Pareto k > 0.7: {np.sum(loo_exp1.pareto_k > 0.7)}/{n_groups} groups")
print(f"   - Max Pareto k: {np.max(loo_exp1.pareto_k):.3f}")

print("\n   EXPERIMENT 3 (Beta-Binomial):")
print(f"   - ELPD_loo: {loo_exp3.elpd_loo:.2f} ± {loo_exp3.se:.2f}")
print(f"   - p_loo: {loo_exp3.p_loo:.2f}")
print(f"   - Pareto k > 0.7: {np.sum(loo_exp3.pareto_k > 0.7)}/{n_groups} groups")
print(f"   - Max Pareto k: {np.max(loo_exp3.pareto_k):.3f}")

# Compute difference
delta_elpd = loo_exp3.elpd_loo - loo_exp1.elpd_loo
se_diff = np.sqrt(loo_exp1.se**2 + loo_exp3.se**2)  # Approximate SE of difference

print("\n   COMPARISON:")
print(f"   - ΔELPD (Exp3 - Exp1): {delta_elpd:.2f} ± {se_diff:.2f}")
print(f"   - Magnitude: {abs(delta_elpd) / se_diff:.2f} × SE")

if abs(delta_elpd) < 2 * se_diff:
    print("   - Decision: Models EQUIVALENT (|Δ| < 2×SE) → Choose simpler")
elif abs(delta_elpd) > 4 * se_diff:
    winner = "Exp 3" if delta_elpd > 0 else "Exp 1"
    print(f"   - Decision: {winner} STRONGLY PREFERRED (|Δ| > 4×SE)")
else:
    winner = "Exp 3" if delta_elpd > 0 else "Exp 1"
    print(f"   - Decision: {winner} WEAKLY PREFERRED (2×SE < |Δ| < 4×SE)")

# Use az.compare for formal comparison
print("\n   Using az.compare()...")
compare_df = az.compare({"Exp1_Hierarchical": idata_exp1, "Exp3_BetaBinomial": idata_exp3})
print(compare_df)

# ============================================================================
# 3. WAIC COMPUTATION (Alternative to LOO)
# ============================================================================
print("\n3. Computing WAIC (more robust to high Pareto k)...")

waic_exp1 = az.waic(idata_exp1, pointwise=True)
waic_exp3 = az.waic(idata_exp3, pointwise=True)

print("\n   EXPERIMENT 1 (Hierarchical Binomial):")
print(f"   - ELPD_waic: {waic_exp1.elpd_waic:.2f} ± {waic_exp1.se:.2f}")
print(f"   - p_waic: {waic_exp1.p_waic:.2f}")

print("\n   EXPERIMENT 3 (Beta-Binomial):")
print(f"   - ELPD_waic: {waic_exp3.elpd_waic:.2f} ± {waic_exp3.se:.2f}")
print(f"   - p_waic: {waic_exp3.p_waic:.2f}")

delta_waic = waic_exp3.elpd_waic - waic_exp1.elpd_waic
se_diff_waic = np.sqrt(waic_exp1.se**2 + waic_exp3.se**2)

print("\n   COMPARISON:")
print(f"   - ΔELPD_waic (Exp3 - Exp1): {delta_waic:.2f} ± {se_diff_waic:.2f}")

# ============================================================================
# 4. PREDICTIVE METRICS (RMSE, MAE, Deviance)
# ============================================================================
print("\n4. Computing predictive metrics...")

# Extract posterior predictive samples
def get_predictions(idata, var_name='r'):
    """Extract posterior predictive mean and credible intervals"""
    if 'posterior_predictive' in idata.groups():
        ppc = idata.posterior_predictive[var_name].values
        # Flatten chains and draws
        ppc_flat = ppc.reshape(-1, ppc.shape[-1])
        pred_mean = ppc_flat.mean(axis=0)
        pred_std = ppc_flat.std(axis=0)
        pred_lower = np.percentile(ppc_flat, 2.5, axis=0)
        pred_upper = np.percentile(ppc_flat, 97.5, axis=0)
        return pred_mean, pred_std, pred_lower, pred_upper
    return None, None, None, None

# Get predictions for both models
pred_exp1_mean, pred_exp1_std, pred_exp1_lower, pred_exp1_upper = get_predictions(idata_exp1)
pred_exp3_mean, pred_exp3_std, pred_exp3_lower, pred_exp3_upper = get_predictions(idata_exp3)

# Compute metrics
def compute_metrics(obs, pred_mean, pred_std):
    """Compute RMSE, MAE, and mean log predictive density"""
    rmse = np.sqrt(np.mean((obs - pred_mean)**2))
    mae = np.mean(np.abs(obs - pred_mean))

    # Normalized metrics (as proportion of observed)
    rmse_norm = rmse / np.mean(obs)
    mae_norm = mae / np.mean(obs)

    return rmse, mae, rmse_norm, mae_norm

if pred_exp1_mean is not None:
    rmse_exp1, mae_exp1, rmse_norm_exp1, mae_norm_exp1 = compute_metrics(r_obs, pred_exp1_mean, pred_exp1_std)
    print("\n   EXPERIMENT 1 (Hierarchical):")
    print(f"   - RMSE: {rmse_exp1:.3f} successes")
    print(f"   - MAE: {mae_exp1:.3f} successes")
    print(f"   - RMSE (normalized): {rmse_norm_exp1:.1%}")
    print(f"   - MAE (normalized): {mae_norm_exp1:.1%}")

if pred_exp3_mean is not None:
    rmse_exp3, mae_exp3, rmse_norm_exp3, mae_norm_exp3 = compute_metrics(r_obs, pred_exp3_mean, pred_exp3_std)
    print("\n   EXPERIMENT 3 (Beta-Binomial):")
    print(f"   - RMSE: {rmse_exp3:.3f} successes")
    print(f"   - MAE: {mae_exp3:.3f} successes")
    print(f"   - RMSE (normalized): {rmse_norm_exp3:.1%}")
    print(f"   - MAE (normalized): {mae_norm_exp3:.1%}")

if pred_exp1_mean is not None and pred_exp3_mean is not None:
    print("\n   COMPARISON:")
    print(f"   - ΔRMSE (Exp3 - Exp1): {rmse_exp3 - rmse_exp1:+.3f}")
    print(f"   - ΔMAE (Exp3 - Exp1): {mae_exp3 - mae_exp1:+.3f}")
    print(f"   - Better predictor: {'Exp1' if rmse_exp1 < rmse_exp3 else 'Exp3'} (by RMSE)")

# ============================================================================
# 5. CALIBRATION ASSESSMENT (Coverage)
# ============================================================================
print("\n5. Computing calibration (posterior predictive interval coverage)...")

def compute_coverage(obs, lower, upper):
    """Compute 95% interval coverage"""
    coverage = np.mean((obs >= lower) & (obs <= upper))
    return coverage

if pred_exp1_mean is not None:
    coverage_exp1 = compute_coverage(r_obs, pred_exp1_lower, pred_exp1_upper)
    print(f"   - Exp1: 95% interval coverage = {coverage_exp1:.1%} (target: 95%)")

if pred_exp3_mean is not None:
    coverage_exp3 = compute_coverage(r_obs, pred_exp3_lower, pred_exp3_upper)
    print(f"   - Exp3: 95% interval coverage = {coverage_exp3:.1%} (target: 95%)")

# ============================================================================
# 6. LOO-PIT CALIBRATION (if possible)
# ============================================================================
print("\n6. Computing LOO-PIT calibration...")

try:
    # LOO-PIT for Exp1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    az.plot_loo_pit(idata_exp1, y='y_obs', legend=False, ax=axes[0])
    axes[0].set_title('Exp1: Hierarchical Binomial LOO-PIT\n(Unreliable: 10/12 bad k)',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('LOO Probability Integral Transform', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)

    az.plot_loo_pit(idata_exp3, y='y_obs', legend=False, ax=axes[1])
    axes[1].set_title('Exp3: Beta-Binomial LOO-PIT\n(Reliable: 0/12 bad k)',
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('LOO Probability Integral Transform', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)

    plt.tight_layout()
    plt.savefig(COMP_DIR / "plots/loo_pit_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   - LOO-PIT plots saved successfully")

except Exception as e:
    print(f"   - Warning: Could not compute LOO-PIT: {e}")

# ============================================================================
# 7. VISUALIZATION 1: LOO COMPARISON WITH PARETO K
# ============================================================================
print("\n7. Creating comprehensive comparison visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: ELPD comparison
ax = axes[0, 0]
models = ['Exp1\nHierarchical', 'Exp3\nBeta-Binomial']
elpd_vals = [loo_exp1.elpd_loo, loo_exp3.elpd_loo]
se_vals = [loo_exp1.se, loo_exp3.se]
colors = ['#e74c3c', '#2ecc71']

bars = ax.bar(models, elpd_vals, yerr=se_vals, capsize=10,
              color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('ELPD (Expected Log Predictive Density)', fontsize=12, fontweight='bold')
ax.set_title('A. LOO Cross-Validation Comparison', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add text annotations
for i, (val, se) in enumerate(zip(elpd_vals, se_vals)):
    ax.text(i, val + se + 1, f'{val:.1f}±{se:.1f}',
           ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add difference
ax.text(0.5, min(elpd_vals) - 3,
       f'Δ = {delta_elpd:.1f}±{se_diff:.1f}\n({abs(delta_elpd)/se_diff:.1f}×SE)',
       ha='center', fontsize=11,
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Panel B: Pareto k comparison
ax = axes[0, 1]
x = np.arange(n_groups)
width = 0.35

k_exp1 = loo_exp1.pareto_k
k_exp3 = loo_exp3.pareto_k

bars1 = ax.bar(x - width/2, k_exp1, width, label='Exp1 (Hierarchical)',
               color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, k_exp3, width, label='Exp3 (Beta-Binomial)',
               color='#2ecc71', alpha=0.7, edgecolor='black')

ax.axhline(0.7, color='orange', linestyle='--', linewidth=2, label='k=0.7 (threshold)')
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='k=1.0 (very bad)')
ax.set_xlabel('Group', fontsize=12, fontweight='bold')
ax.set_ylabel('Pareto k', fontsize=12, fontweight='bold')
ax.set_title('B. LOO Reliability (Pareto k Diagnostics)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{i+1}' for i in range(n_groups)])
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add summary text
n_bad_exp1 = np.sum(k_exp1 > 0.7)
n_bad_exp3 = np.sum(k_exp3 > 0.7)
ax.text(0.02, 0.98, f'Exp1: {n_bad_exp1}/12 bad k\nExp3: {n_bad_exp3}/12 bad k',
       transform=ax.transAxes, va='top', fontsize=11,
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel C: Predictive accuracy (RMSE/MAE)
ax = axes[1, 0]
metrics = ['RMSE', 'MAE']
exp1_vals = [rmse_exp1, mae_exp1] if pred_exp1_mean is not None else [0, 0]
exp3_vals = [rmse_exp3, mae_exp3] if pred_exp3_mean is not None else [0, 0]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, exp1_vals, width, label='Exp1 (Hierarchical)',
              color='#e74c3c', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, exp3_vals, width, label='Exp3 (Beta-Binomial)',
              color='#2ecc71', alpha=0.7, edgecolor='black')

ax.set_ylabel('Prediction Error (successes)', fontsize=12, fontweight='bold')
ax.set_title('C. Predictive Accuracy', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=10)

# Panel D: Model complexity vs performance
ax = axes[1, 1]

# Create scatter plot
params = [14, 2]  # Exp1 has 14 params, Exp3 has 2
sampling_times = [90, 6]  # seconds
elpd = [loo_exp1.elpd_loo, loo_exp3.elpd_loo]
pareto_bad = [n_bad_exp1, n_bad_exp3]

# Size by sampling time, color by bad k values
sizes = [t * 10 for t in sampling_times]
colors_scatter = ['#e74c3c' if n > 5 else '#2ecc71' for n in pareto_bad]

scatter = ax.scatter(params, elpd, s=sizes, c=colors_scatter, alpha=0.6,
                    edgecolors='black', linewidths=2)

# Add labels
ax.text(14, loo_exp1.elpd_loo + 1, 'Exp1\n14 params\n90s sampling\n10 bad k',
       ha='center', va='bottom', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text(2, loo_exp3.elpd_loo - 1, 'Exp3\n2 params\n6s sampling\n0 bad k',
       ha='center', va='top', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
ax.set_ylabel('ELPD LOO', fontsize=12, fontweight='bold')
ax.set_title('D. Complexity-Performance Trade-off', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add diagonal reference line
ax.plot([2, 14], [loo_exp3.elpd_loo, loo_exp1.elpd_loo],
       'k--', alpha=0.3, linewidth=1, label='Trend')

plt.tight_layout()
plt.savefig(COMP_DIR / "plots/comprehensive_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("   - Comprehensive comparison plot saved")

# ============================================================================
# 8. VISUALIZATION 2: OBSERVED VS PREDICTED COMPARISON
# ============================================================================

if pred_exp1_mean is not None and pred_exp3_mean is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Exp1
    ax = axes[0]
    ax.errorbar(r_obs, pred_exp1_mean,
               yerr=[pred_exp1_mean - pred_exp1_lower, pred_exp1_upper - pred_exp1_mean],
               fmt='o', markersize=8, capsize=5, alpha=0.7, color='#e74c3c',
               ecolor='gray', label='Predictions with 95% CI')
    ax.plot([0, max(r_obs)], [0, max(r_obs)], 'k--', linewidth=2, alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('Observed Successes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Successes', fontsize=12, fontweight='bold')
    ax.set_title(f'Exp1: Hierarchical Binomial\nRMSE={rmse_exp1:.2f}, Coverage={coverage_exp1:.1%}',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Exp3
    ax = axes[1]
    ax.errorbar(r_obs, pred_exp3_mean,
               yerr=[pred_exp3_mean - pred_exp3_lower, pred_exp3_upper - pred_exp3_mean],
               fmt='o', markersize=8, capsize=5, alpha=0.7, color='#2ecc71',
               ecolor='gray', label='Predictions with 95% CI')
    ax.plot([0, max(r_obs)], [0, max(r_obs)], 'k--', linewidth=2, alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('Observed Successes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Successes', fontsize=12, fontweight='bold')
    ax.set_title(f'Exp3: Beta-Binomial\nRMSE={rmse_exp3:.2f}, Coverage={coverage_exp3:.1%}',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(COMP_DIR / "plots/observed_vs_predicted_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   - Observed vs predicted comparison saved")

# ============================================================================
# 9. VISUALIZATION 3: SPIDER/RADAR PLOT OF TRADE-OFFS
# ============================================================================

# Create spider plot showing trade-offs
categories = ['Predictive\nAccuracy', 'LOO\nReliability', 'Simplicity\n(1/params)',
              'Computational\nSpeed', 'Calibration']

# Normalize scores to 0-10 scale
exp1_scores = [
    10 * (1 - rmse_exp1 / max(rmse_exp1, rmse_exp3)) if pred_exp1_mean is not None else 5,  # Accuracy
    10 * (1 - n_bad_exp1 / n_groups),  # LOO reliability (inverted bad k proportion)
    10 * (2 / 14),  # Simplicity (inverted params)
    10 * (6 / 90),  # Speed (inverted time)
    10 * coverage_exp1 / 0.95 if pred_exp1_mean is not None else 5,  # Calibration
]

exp3_scores = [
    10 * (1 - rmse_exp3 / max(rmse_exp1, rmse_exp3)) if pred_exp3_mean is not None else 5,
    10 * (1 - n_bad_exp3 / n_groups),
    10 * (2 / 2),
    10 * (6 / 6),
    10 * coverage_exp3 / 0.95 if pred_exp3_mean is not None else 5,
]

# Number of variables
N = len(categories)

# Compute angle for each axis
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Initialize figure
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Plot data
exp1_scores += exp1_scores[:1]
exp3_scores += exp3_scores[:1]

ax.plot(angles, exp1_scores, 'o-', linewidth=2, color='#e74c3c', label='Exp1: Hierarchical', markersize=8)
ax.fill(angles, exp1_scores, alpha=0.15, color='#e74c3c')

ax.plot(angles, exp3_scores, 'o-', linewidth=2, color='#2ecc71', label='Exp3: Beta-Binomial', markersize=8)
ax.fill(angles, exp3_scores, alpha=0.15, color='#2ecc71')

# Fix axis to go in the right order
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)

# Set y-axis limits
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)
ax.grid(True)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)

plt.title('Model Trade-offs: Multi-Criteria Assessment\n(Higher = Better)',
         fontsize=14, fontweight='bold', pad=20)

plt.savefig(COMP_DIR / "plots/model_trade_offs_spider.png", dpi=300, bbox_inches='tight')
plt.close()
print("   - Spider plot of trade-offs saved")

# ============================================================================
# 10. SAVE NUMERICAL RESULTS
# ============================================================================
print("\n8. Saving numerical results...")

# Create comparison table
comparison_data = {
    'Metric': [
        'Parameters', 'Sampling_time_sec', 'ELPD_loo', 'SE_loo', 'p_loo',
        'Pareto_k_max', 'Pareto_k_gt_0.7', 'ELPD_waic', 'SE_waic', 'p_waic',
        'RMSE', 'MAE', 'Coverage_95pct'
    ],
    'Exp1_Hierarchical': [
        14, 90,
        loo_exp1.elpd_loo, loo_exp1.se, loo_exp1.p_loo,
        np.max(loo_exp1.pareto_k), np.sum(loo_exp1.pareto_k > 0.7),
        waic_exp1.elpd_waic, waic_exp1.se, waic_exp1.p_waic,
        rmse_exp1 if pred_exp1_mean is not None else np.nan,
        mae_exp1 if pred_exp1_mean is not None else np.nan,
        coverage_exp1 if pred_exp1_mean is not None else np.nan,
    ],
    'Exp3_BetaBinomial': [
        2, 6,
        loo_exp3.elpd_loo, loo_exp3.se, loo_exp3.p_loo,
        np.max(loo_exp3.pareto_k), np.sum(loo_exp3.pareto_k > 0.7),
        waic_exp3.elpd_waic, waic_exp3.se, waic_exp3.p_waic,
        rmse_exp3 if pred_exp3_mean is not None else np.nan,
        mae_exp3 if pred_exp3_mean is not None else np.nan,
        coverage_exp3 if pred_exp3_mean is not None else np.nan,
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(COMP_DIR / "diagnostics/comparison_metrics.csv", index=False)
print("   - Comparison metrics saved to CSV")

# Save detailed LOO comparison
loo_comparison_data = {
    'group': range(1, n_groups + 1),
    'n': n_obs,
    'r_obs': r_obs,
    'pareto_k_exp1': loo_exp1.pareto_k,
    'pareto_k_exp3': loo_exp3.pareto_k,
    'elpd_exp1': loo_exp1.loo_i.values,
    'elpd_exp3': loo_exp3.loo_i.values,
}
loo_df = pd.DataFrame(loo_comparison_data)
loo_df.to_csv(COMP_DIR / "diagnostics/loo_comparison_table.csv", index=False)
print("   - LOO comparison table saved")

# Save JSON summary
summary = {
    'experiment_1': {
        'model': 'Hierarchical Binomial (Logit-Normal)',
        'parameters': 14,
        'sampling_time_sec': 90,
        'loo': {
            'elpd': float(loo_exp1.elpd_loo),
            'se': float(loo_exp1.se),
            'p_loo': float(loo_exp1.p_loo),
            'pareto_k_max': float(np.max(loo_exp1.pareto_k)),
            'pareto_k_bad_count': int(np.sum(loo_exp1.pareto_k > 0.7)),
            'reliability': 'UNRELIABLE (10/12 bad k)'
        },
        'waic': {
            'elpd': float(waic_exp1.elpd_waic),
            'se': float(waic_exp1.se),
            'p_waic': float(waic_exp1.p_waic)
        },
        'predictive': {
            'rmse': float(rmse_exp1) if pred_exp1_mean is not None else None,
            'mae': float(mae_exp1) if pred_exp1_mean is not None else None,
            'coverage': float(coverage_exp1) if pred_exp1_mean is not None else None,
        }
    },
    'experiment_3': {
        'model': 'Beta-Binomial (Population-level)',
        'parameters': 2,
        'sampling_time_sec': 6,
        'loo': {
            'elpd': float(loo_exp3.elpd_loo),
            'se': float(loo_exp3.se),
            'p_loo': float(loo_exp3.p_loo),
            'pareto_k_max': float(np.max(loo_exp3.pareto_k)),
            'pareto_k_bad_count': int(np.sum(loo_exp3.pareto_k > 0.7)),
            'reliability': 'RELIABLE (0/12 bad k)'
        },
        'waic': {
            'elpd': float(waic_exp3.elpd_waic),
            'se': float(waic_exp3.se),
            'p_waic': float(waic_exp3.p_waic)
        },
        'predictive': {
            'rmse': float(rmse_exp3) if pred_exp3_mean is not None else None,
            'mae': float(mae_exp3) if pred_exp3_mean is not None else None,
            'coverage': float(coverage_exp3) if pred_exp3_mean is not None else None,
        }
    },
    'comparison': {
        'elpd_difference': float(delta_elpd),
        'se_difference': float(se_diff),
        'magnitude_in_se': float(abs(delta_elpd) / se_diff),
        'decision': 'EQUIVALENT' if abs(delta_elpd) < 2 * se_diff else
                   ('STRONG_PREFERENCE' if abs(delta_elpd) > 4 * se_diff else 'WEAK_PREFERENCE'),
        'preferred_by_elpd': 'Exp3' if delta_elpd > 0 else 'Exp1',
        'preferred_by_parsimony': 'Exp3 (2 params vs 14 params)',
        'preferred_by_loo_reliability': 'Exp3 (0/12 bad k vs 10/12 bad k)',
        'key_insight': 'Exp3 dramatically superior in LOO reliability despite similar predictive accuracy'
    }
}

with open(COMP_DIR / "diagnostics/comparison_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print("   - JSON summary saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON COMPLETE")
print("=" * 80)

print("\nKEY FINDINGS:")
print(f"1. LOO Comparison: ΔELPD = {delta_elpd:.2f} ± {se_diff:.2f} ({abs(delta_elpd)/se_diff:.1f}×SE)")
print(f"2. LOO Reliability: Exp1 has {n_bad_exp1}/12 bad k, Exp3 has {n_bad_exp3}/12 bad k")
print(f"3. Predictive Accuracy: ΔRMSE = {(rmse_exp3 - rmse_exp1):+.3f} successes")
print(f"4. Parsimony: Exp3 is 7× simpler (2 vs 14 parameters)")
print(f"5. Speed: Exp3 is 15× faster (6 vs 90 seconds)")

print("\nRECOMMENDATION:")
if abs(delta_elpd) < 2 * se_diff and n_bad_exp3 < n_bad_exp1:
    print("   Choose EXP3 (Beta-Binomial) based on:")
    print("   - Equivalent predictive performance (ELPD within 2×SE)")
    print("   - Dramatically superior LOO reliability (0 vs 10 bad k)")
    print("   - Greater parsimony (2 vs 14 parameters)")
    print("   - Faster computation (6 vs 90 seconds)")
    print("   - Simpler interpretation (probability vs logit scale)")
elif delta_elpd > 2 * se_diff:
    print("   EXP3 (Beta-Binomial) is preferred by predictive accuracy AND reliability")
else:
    print("   TRADE-OFF: Consider research goals")
    print("   - Use Exp1 for group-specific inference")
    print("   - Use Exp3 for population-level inference and reliable LOO")

print("\nOUTPUT FILES:")
print(f"   - {COMP_DIR}/plots/comprehensive_comparison.png")
print(f"   - {COMP_DIR}/plots/observed_vs_predicted_comparison.png")
print(f"   - {COMP_DIR}/plots/model_trade_offs_spider.png")
print(f"   - {COMP_DIR}/plots/loo_pit_comparison.png")
print(f"   - {COMP_DIR}/diagnostics/comparison_metrics.csv")
print(f"   - {COMP_DIR}/diagnostics/loo_comparison_table.csv")
print(f"   - {COMP_DIR}/diagnostics/comparison_summary.json")

print("\n" + "=" * 80)
