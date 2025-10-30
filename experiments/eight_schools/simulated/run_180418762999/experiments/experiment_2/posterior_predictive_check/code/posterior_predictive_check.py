"""
Posterior Predictive Check for Experiment 2: Hierarchical Partial Pooling Model

This script performs comprehensive posterior predictive checks including:
1. LOO-CV comparison with Model 1 (complete pooling)
2. Pareto k diagnostics
3. Observation-level posterior predictive checks
4. Test statistics (mean, SD, min, max)
5. Residual analysis
6. Calibration checks

Key Question: Does the hierarchical model improve predictive performance over Model 1?
"""

import numpy as np
import pandas as pd
import arviz as az
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
MODEL1_PATH = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
MODEL2_PATH = Path("/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf")
PLOTS_DIR = Path("/workspace/experiments/experiment_2/posterior_predictive_check/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("POSTERIOR PREDICTIVE CHECK: EXPERIMENT 2")
print("Hierarchical Partial Pooling Model with Known Measurement Error")
print("="*80)

# Load data
print("\n1. Loading observed data...")
data = pd.read_csv(DATA_PATH)
y_obs = data['y'].values
sigma_obs = data['sigma'].values
n_obs = len(y_obs)
print(f"   Observations: {n_obs}")
print(f"   y range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")
print(f"   sigma range: [{sigma_obs.min():.2f}, {sigma_obs.max():.2f}]")

# Load posterior samples
print("\n2. Loading posterior samples...")
print("   Model 1 (Complete Pooling)...")
trace_model1 = az.from_netcdf(MODEL1_PATH)
print("   Model 2 (Hierarchical Partial Pooling)...")
trace_model2 = az.from_netcdf(MODEL2_PATH)

# Extract posterior samples
mu_samples = trace_model2.posterior['mu'].values.flatten()
tau_samples = trace_model2.posterior['tau'].values.flatten()
theta_samples = trace_model2.posterior['theta'].values.reshape(-1, n_obs)

# Get chain and draw dimensions
n_chains = trace_model2.posterior.dims['chain']
n_draws = trace_model2.posterior.dims['draw']
n_samples = len(mu_samples)

print(f"   Total posterior samples: {n_samples} ({n_chains} chains x {n_draws} draws)")
print(f"   mu: {mu_samples.mean():.3f} ± {mu_samples.std():.3f}")
print(f"   tau: {tau_samples.mean():.3f} ± {tau_samples.std():.3f}")
print(f"   tau 95% HDI: [{np.percentile(tau_samples, 2.5):.3f}, {np.percentile(tau_samples, 97.5):.3f}]")

# Generate posterior predictive samples
print("\n3. Generating posterior predictive samples...")
np.random.seed(42)
y_pred_flat = np.zeros((n_samples, n_obs))

for i in range(n_samples):
    for j in range(n_obs):
        y_pred_flat[i, j] = np.random.normal(theta_samples[i, j], sigma_obs[j])

print(f"   Generated {n_samples} replicated datasets")
print(f"   y_pred shape: {y_pred_flat.shape}")

# Reshape to match trace structure and add to trace
y_pred = y_pred_flat.reshape(n_chains, n_draws, n_obs)
y_pred_da = xr.DataArray(
    y_pred,
    dims=['chain', 'draw', 'observation'],
    coords={
        'chain': trace_model2.posterior.chain,
        'draw': trace_model2.posterior.draw,
        'observation': range(n_obs)
    }
)
trace_model2.add_groups({'posterior_predictive': xr.Dataset({'y': y_pred_da})})
print(f"   Added posterior_predictive to trace")

# Compute residuals (using flattened predictions)
y_pred_mean = y_pred_flat.mean(axis=0)
residuals = y_obs - y_pred_mean
standardized_residuals = residuals / sigma_obs

print(f"   Mean absolute residual: {np.abs(residuals).mean():.3f}")
print(f"   Mean standardized residual: {standardized_residuals.mean():.3f}")

# ============================================================================
# 4. LOO-CV COMPARISON (CRITICAL)
# ============================================================================
print("\n" + "="*80)
print("4. LOO-CV ANALYSIS: MODEL COMPARISON")
print("="*80)

# Compute LOO for both models
print("\nComputing LOO for Model 1 (Complete Pooling)...")
loo_model1 = az.loo(trace_model1, pointwise=True)
print("\nComputing LOO for Model 2 (Hierarchical Partial Pooling)...")
loo_model2 = az.loo(trace_model2, pointwise=True)

# Compare models
print("\n" + "-"*80)
print("MODEL COMPARISON")
print("-"*80)
compare_df = az.compare({"Model_1_Complete_Pooling": trace_model1,
                         "Model_2_Hierarchical": trace_model2})
print(compare_df)
print("-"*80)

# Extract key metrics
elpd_model1 = loo_model1.elpd_loo
elpd_model2 = loo_model2.elpd_loo
se_model1 = loo_model1.se
se_model2 = loo_model2.se
delta_elpd = elpd_model2 - elpd_model1
se_diff = compare_df.loc["Model_2_Hierarchical", "dse"]

print(f"\nModel 1 ELPD: {elpd_model1:.2f} ± {se_model1:.2f}")
print(f"Model 2 ELPD: {elpd_model2:.2f} ± {se_model2:.2f}")
print(f"Δ ELPD (Model 2 - Model 1): {delta_elpd:.2f} ± {se_diff:.2f}")
print(f"Significance threshold (2×SE): {2*se_diff:.2f}")

if abs(delta_elpd) > 2 * se_diff:
    if delta_elpd > 0:
        print("\n*** MODEL 2 SIGNIFICANTLY BETTER (Δ ELPD > 2×SE) ***")
        preference = "Model 2"
    else:
        print("\n*** MODEL 1 SIGNIFICANTLY BETTER (Δ ELPD < -2×SE) ***")
        preference = "Model 1"
else:
    print("\n*** MODELS STATISTICALLY EQUIVALENT (|Δ ELPD| < 2×SE) ***")
    print("*** PREFER MODEL 1 BY PARSIMONY (1 parameter vs 10 parameters) ***")
    preference = "Model 1 (by parsimony)"

# Pareto k diagnostics
print("\n" + "-"*80)
print("PARETO k DIAGNOSTICS")
print("-"*80)
pareto_k_model1 = loo_model1.pareto_k
pareto_k_model2 = loo_model2.pareto_k

print("\nModel 1 Pareto k:")
for i, k in enumerate(pareto_k_model1):
    status = "GOOD" if k < 0.5 else ("OK" if k < 0.7 else "BAD")
    print(f"  Obs {i+1}: k = {k:.4f} [{status}]")
print(f"  Max k: {pareto_k_model1.max():.4f}")

print("\nModel 2 Pareto k:")
for i, k in enumerate(pareto_k_model2):
    status = "GOOD" if k < 0.5 else ("OK" if k < 0.7 else "BAD")
    print(f"  Obs {i+1}: k = {k:.4f} [{status}]")
print(f"  Max k: {pareto_k_model2.max():.4f}")

# ============================================================================
# 5. VISUALIZATION: LOO COMPARISON
# ============================================================================
print("\n5. Creating LOO comparison visualizations...")

# Plot 1: LOO comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: ELPD comparison with error bars
models = ['Model 1\n(Complete\nPooling)', 'Model 2\n(Hierarchical\nPartial Pooling)']
elpds = [elpd_model1, elpd_model2]
ses = [se_model1, se_model2]
colors = ['#2E86AB' if preference == "Model 1 (by parsimony)" or preference == "Model 1" else '#A23B72',
          '#A23B72' if preference == "Model 2" else '#2E86AB']

axes[0].bar(models, elpds, yerr=ses, capsize=10, color=colors, alpha=0.7, edgecolor='black')
axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
axes[0].set_ylabel('ELPD LOO', fontsize=12, fontweight='bold')
axes[0].set_title('Expected Log Predictive Density\n(Higher is Better)', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Add significance test
y_min = min(elpds) - max(ses) - 2
y_max = max(elpds) + max(ses) + 2
axes[0].set_ylim(y_min, y_max)

# Add delta text
delta_text = f'Δ ELPD = {delta_elpd:.2f} ± {se_diff:.2f}\n'
if abs(delta_elpd) > 2 * se_diff:
    delta_text += f'|Δ| = {abs(delta_elpd):.2f} > 2×SE = {2*se_diff:.2f}\n'
    if delta_elpd > 0:
        delta_text += 'Model 2 significantly better'
    else:
        delta_text += 'Model 1 significantly better'
else:
    delta_text += f'|Δ| = {abs(delta_elpd):.2f} < 2×SE = {2*se_diff:.2f}\n'
    delta_text += 'Models equivalent\nPrefer simpler Model 1'

axes[0].text(0.5, 0.95, delta_text, transform=axes[0].transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Right panel: Pointwise ELPD comparison
elpd_pointwise_1 = loo_model1.loo_i
elpd_pointwise_2 = loo_model2.loo_i

x_pos = np.arange(n_obs) + 1
width = 0.35

axes[1].bar(x_pos - width/2, elpd_pointwise_1, width, label='Model 1',
            color='#2E86AB', alpha=0.7, edgecolor='black')
axes[1].bar(x_pos + width/2, elpd_pointwise_2, width, label='Model 2',
            color='#A23B72', alpha=0.7, edgecolor='black')
axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
axes[1].set_xlabel('Observation', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Pointwise ELPD', fontsize=12, fontweight='bold')
axes[1].set_title('Pointwise ELPD Comparison\n(Higher is Better)', fontsize=13, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(x_pos)
axes[1].legend(frameon=True, fancybox=True, shadow=True)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "loo_comparison.png")
print(f"   Saved: {PLOTS_DIR / 'loo_comparison.png'}")
plt.close()

# Plot 2: Pareto k diagnostics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Model 1 Pareto k
axes[0].scatter(range(1, n_obs+1), pareto_k_model1, s=100, c='#2E86AB',
                alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0].axhline(0.5, color='green', linestyle='--', label='k=0.5 (good)', linewidth=2)
axes[0].axhline(0.7, color='orange', linestyle='--', label='k=0.7 (ok)', linewidth=2)
axes[0].axhline(1.0, color='red', linestyle='--', label='k=1.0 (bad)', linewidth=2)
axes[0].set_xlabel('Observation', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Pareto k', fontsize=12, fontweight='bold')
axes[0].set_title('Model 1: Pareto k Diagnostics\n(Lower is Better)', fontsize=13, fontweight='bold')
axes[0].legend(frameon=True, fancybox=True, shadow=True)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(-0.1, max(1.1, pareto_k_model1.max() + 0.1))
axes[0].set_xticks(range(1, n_obs+1))

# Right: Model 2 Pareto k
axes[1].scatter(range(1, n_obs+1), pareto_k_model2, s=100, c='#A23B72',
                alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1].axhline(0.5, color='green', linestyle='--', label='k=0.5 (good)', linewidth=2)
axes[1].axhline(0.7, color='orange', linestyle='--', label='k=0.7 (ok)', linewidth=2)
axes[1].axhline(1.0, color='red', linestyle='--', label='k=1.0 (bad)', linewidth=2)
axes[1].set_xlabel('Observation', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Pareto k', fontsize=12, fontweight='bold')
axes[1].set_title('Model 2: Pareto k Diagnostics\n(Lower is Better)', fontsize=13, fontweight='bold')
axes[1].legend(frameon=True, fancybox=True, shadow=True)
axes[1].grid(alpha=0.3)
axes[1].set_ylim(-0.1, max(1.1, pareto_k_model2.max() + 0.1))
axes[1].set_xticks(range(1, n_obs+1))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "loo_pareto_k.png")
print(f"   Saved: {PLOTS_DIR / 'loo_pareto_k.png'}")
plt.close()

# ============================================================================
# 6. OBSERVATION-LEVEL PPC
# ============================================================================
print("\n6. Creating observation-level posterior predictive checks...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(n_obs):
    ax = axes[i]

    # Histogram of posterior predictive
    ax.hist(y_pred_flat[:, i], bins=50, density=True, alpha=0.6,
            color='skyblue', edgecolor='black', label='Posterior Predictive')

    # Observed value
    ax.axvline(y_obs[i], color='red', linewidth=3, label=f'Observed: {y_obs[i]:.2f}')

    # Posterior mean
    ax.axvline(y_pred_mean[i], color='blue', linewidth=2, linestyle='--',
               label=f'Pred Mean: {y_pred_mean[i]:.2f}')

    # Calculate p-value
    p_value = np.mean(y_pred_flat[:, i] >= y_obs[i])
    p_value = min(p_value, 1 - p_value) * 2  # Two-tailed

    ax.set_title(f'Observation {i+1}\n(sigma={sigma_obs[i]:.0f}, p={p_value:.3f})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('y', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

plt.suptitle('Posterior Predictive Check: Observation-Level Comparisons\nModel 2: Hierarchical Partial Pooling',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_observations.png")
print(f"   Saved: {PLOTS_DIR / 'ppc_observations.png'}")
plt.close()

# ============================================================================
# 7. TEST STATISTICS
# ============================================================================
print("\n7. Computing test statistics...")

# Compute test statistics for observed data
def compute_test_stats(data):
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }

# Observed statistics
obs_stats = compute_test_stats(y_obs)

# Replicated statistics
rep_stats = {stat: [] for stat in obs_stats.keys()}
for i in range(n_samples):
    stats = compute_test_stats(y_pred_flat[i, :])
    for stat, value in stats.items():
        rep_stats[stat].append(value)

# Convert to arrays
for stat in rep_stats.keys():
    rep_stats[stat] = np.array(rep_stats[stat])

# Compute p-values (two-tailed)
p_values = {}
for stat in obs_stats.keys():
    p = np.mean(rep_stats[stat] >= obs_stats[stat])
    p_values[stat] = min(p, 1-p) * 2

# Print results
print("\nTest Statistics:")
print("-" * 80)
print(f"{'Statistic':<12} {'Observed':<12} {'Pred Mean':<12} {'Pred SD':<12} {'p-value':<10}")
print("-" * 80)
for stat in obs_stats.keys():
    print(f"{stat:<12} {obs_stats[stat]:>11.3f} {rep_stats[stat].mean():>11.3f} "
          f"{rep_stats[stat].std():>11.3f} {p_values[stat]:>9.3f}")
print("-" * 80)

# Plot test statistics
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, stat in enumerate(obs_stats.keys()):
    ax = axes[i]

    # Histogram of replicated statistics
    ax.hist(rep_stats[stat], bins=50, density=True, alpha=0.6,
            color='lightgreen', edgecolor='black', label='Replicated')

    # Observed statistic
    ax.axvline(obs_stats[stat], color='red', linewidth=3,
               label=f'Observed: {obs_stats[stat]:.2f}')

    # Mean of replicated
    ax.axvline(rep_stats[stat].mean(), color='blue', linewidth=2,
               linestyle='--', label=f'Pred Mean: {rep_stats[stat].mean():.2f}')

    ax.set_title(f'{stat.upper()}\n(p-value: {p_values[stat]:.3f})',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)

plt.suptitle('Test Statistics: Observed vs Posterior Predictive\nModel 2: Hierarchical Partial Pooling',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_test_statistics.png")
print(f"   Saved: {PLOTS_DIR / 'ppc_test_statistics.png'}")
plt.close()

# ============================================================================
# 8. RESIDUAL ANALYSIS
# ============================================================================
print("\n8. Creating residual plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Residuals vs Observation
axes[0, 0].scatter(range(1, n_obs+1), residuals, s=100, c='purple',
                   alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Observation', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Residual (y_obs - y_pred_mean)', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Residuals vs Observation', fontsize=12, fontweight='bold')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_xticks(range(1, n_obs+1))

# 2. Standardized residuals vs Observation
axes[0, 1].scatter(range(1, n_obs+1), standardized_residuals, s=100, c='purple',
                   alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=2)
axes[0, 1].axhline(2, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='±2 SD')
axes[0, 1].axhline(-2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
axes[0, 1].set_xlabel('Observation', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Standardized Residual', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Standardized Residuals vs Observation', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xticks(range(1, n_obs+1))

# 3. Residuals vs Predicted
axes[0, 2].scatter(y_pred_mean, residuals, s=100, c='purple',
                   alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0, 2].axhline(0, color='black', linestyle='--', linewidth=2)
axes[0, 2].set_xlabel('Predicted Value', fontsize=11, fontweight='bold')
axes[0, 2].set_ylabel('Residual', fontsize=11, fontweight='bold')
axes[0, 2].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
axes[0, 2].grid(alpha=0.3)

# 4. Residuals vs Measurement Error
axes[1, 0].scatter(sigma_obs, residuals, s=100, c='purple',
                   alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Measurement Error (sigma)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Residual', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Residuals vs Measurement Error', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# 5. QQ plot of standardized residuals
from scipy import stats
stats.probplot(standardized_residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Standardized Residuals', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

# 6. Histogram of standardized residuals
axes[1, 2].hist(standardized_residuals, bins=8, density=True, alpha=0.6,
                color='purple', edgecolor='black', label='Standardized Residuals')
x_norm = np.linspace(-3, 3, 100)
axes[1, 2].plot(x_norm, stats.norm.pdf(x_norm), 'r-', linewidth=2, label='N(0,1)')
axes[1, 2].set_xlabel('Standardized Residual', fontsize=11, fontweight='bold')
axes[1, 2].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[1, 2].set_title('Distribution of Standardized Residuals', fontsize=12, fontweight='bold')
axes[1, 2].legend(fontsize=9)
axes[1, 2].grid(alpha=0.3)

plt.suptitle('Residual Analysis\nModel 2: Hierarchical Partial Pooling',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_residuals.png")
print(f"   Saved: {PLOTS_DIR / 'ppc_residuals.png'}")
plt.close()

# ============================================================================
# 9. CALIBRATION CHECK (LOO-PIT)
# ============================================================================
print("\n9. Creating calibration plots...")

# LOO-PIT for Model 2
fig = plt.figure(figsize=(14, 5))
gs = fig.add_gridspec(1, 2, hspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
az.plot_loo_pit(trace_model2, y="y", ecdf=True, ax=ax1)
ax1.set_title('Model 2: LOO-PIT ECDF\nHierarchical Partial Pooling',
              fontsize=12, fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
az.plot_loo_pit(trace_model2, y="y", ecdf=False, ax=ax2)
ax2.set_title('Model 2: LOO-PIT Histogram\nHierarchical Partial Pooling',
              fontsize=12, fontweight='bold')

plt.suptitle('Calibration Check: LOO-PIT (Probability Integral Transform)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "ppc_calibration.png")
print(f"   Saved: {PLOTS_DIR / 'ppc_calibration.png'}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: POSTERIOR PREDICTIVE CHECK RESULTS")
print("="*80)

print("\n1. LOO-CV COMPARISON:")
print(f"   Model 1 ELPD: {elpd_model1:.2f} ± {se_model1:.2f}")
print(f"   Model 2 ELPD: {elpd_model2:.2f} ± {se_model2:.2f}")
print(f"   Δ ELPD: {delta_elpd:.2f} ± {se_diff:.2f}")
print(f"   Significance: |Δ ELPD| {'>' if abs(delta_elpd) > 2*se_diff else '<'} 2×SE")
print(f"   PREFERRED MODEL: {preference}")

print("\n2. PARETO k DIAGNOSTICS:")
print(f"   Model 1 max k: {pareto_k_model1.max():.4f}")
print(f"   Model 2 max k: {pareto_k_model2.max():.4f}")
print(f"   Both models: {'GOOD' if max(pareto_k_model1.max(), pareto_k_model2.max()) < 0.7 else 'PROBLEMATIC'}")

print("\n3. OBSERVATION-LEVEL FIT:")
p_values_obs = []
for i in range(n_obs):
    p = np.mean(y_pred_flat[:, i] >= y_obs[i])
    p = min(p, 1-p) * 2
    p_values_obs.append(p)
print(f"   Min p-value: {min(p_values_obs):.3f}")
print(f"   Max p-value: {max(p_values_obs):.3f}")
print(f"   Extreme p-values (< 0.05): {sum(1 for p in p_values_obs if p < 0.05)}/8")

print("\n4. TEST STATISTICS:")
extreme_stats = sum(1 for p in p_values.values() if p < 0.05)
print(f"   Extreme test statistics (p < 0.05): {extreme_stats}/{len(p_values)}")
if extreme_stats == 0:
    print("   Model captures all summary statistics well")
elif extreme_stats <= 2:
    print("   Minor discrepancies in some statistics")
else:
    print("   Substantial discrepancies in multiple statistics")

print("\n5. RESIDUALS:")
print(f"   Mean absolute residual: {np.abs(residuals).mean():.3f}")
print(f"   Mean standardized residual: {standardized_residuals.mean():.3f}")
print(f"   Extreme residuals (|z| > 2): {sum(np.abs(standardized_residuals) > 2)}/8")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"\nBased on LOO-CV comparison: {preference}")
print(f"\nJustification:")
if abs(delta_elpd) > 2 * se_diff:
    if delta_elpd > 0:
        print("  - Model 2 shows significantly better predictive performance")
        print("  - Hierarchical structure captures genuine heterogeneity")
        print("  - Added complexity is justified by improved predictions")
    else:
        print("  - Model 1 shows significantly better predictive performance")
        print("  - Hierarchical structure adds unnecessary complexity")
        print("  - Overfitting likely occurring in Model 2")
else:
    print("  - No significant difference in predictive performance")
    print("  - Models are statistically equivalent for prediction")
    print(f"  - tau uncertain (95% HDI: [{np.percentile(tau_samples, 2.5):.3f}, {np.percentile(tau_samples, 97.5):.3f}])")
    print("  - Prefer simpler Model 1 by principle of parsimony")
    print("  - Model 1: 1 parameter (mu)")
    print("  - Model 2: 10 parameters (mu, tau, theta[1:8])")

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECK COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: /workspace/experiments/experiment_2/posterior_predictive_check/")
print(f"  - Code: code/posterior_predictive_check.py")
print(f"  - Plots: plots/*.png (5 plots)")
print(f"  - Report: ppc_findings.md (to be created)")
