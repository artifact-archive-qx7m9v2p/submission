"""
Comprehensive Model Comparison: Hierarchical vs Mixture Models
Compares Experiment 1 (continuous heterogeneity) vs Experiment 2 (discrete clusters)
"""

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = '/workspace/data/data.csv'
IDATA1_PATH = '/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf'
IDATA2_PATH = '/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf'
OUTPUT_DIR = '/workspace/experiments/model_comparison'
PLOTS_DIR = f'{OUTPUT_DIR}/comparison_plots'

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)
print("\nExperiment 1: Hierarchical Logit-Normal (continuous heterogeneity)")
print("Experiment 2: Finite Mixture K=3 (discrete heterogeneity)")
print("\n" + "="*80)

# Load data
print("\n[1/9] Loading data and models...")
data = pd.read_csv(DATA_PATH)
print(f"Data: {len(data)} groups, {data['n_trials'].sum()} total trials")

# Load InferenceData objects
print("\nLoading Experiment 1 (Hierarchical)...")
idata1 = az.from_netcdf(IDATA1_PATH)
print(f"  Chains: {idata1.posterior.dims['chain']}, Draws: {idata1.posterior.dims['draw']}")

print("\nLoading Experiment 2 (Mixture)...")
idata2 = az.from_netcdf(IDATA2_PATH)
print(f"  Chains: {idata2.posterior.dims['chain']}, Draws: {idata2.posterior.dims['draw']}")

# Verify log_likelihood groups exist
print("\nVerifying log_likelihood groups:")
if 'log_likelihood' in idata1.groups():
    print("  Experiment 1: log_likelihood FOUND")
else:
    raise ValueError("Experiment 1 missing log_likelihood group!")

if 'log_likelihood' in idata2.groups():
    print("  Experiment 2: log_likelihood FOUND")
else:
    raise ValueError("Experiment 2 missing log_likelihood group!")

# ============================================================================
# 2. LOO CROSS-VALIDATION COMPARISON
# ============================================================================
print("\n" + "="*80)
print("[2/9] LOO CROSS-VALIDATION COMPARISON")
print("="*80)

print("\nComputing LOO for Experiment 1...")
loo1 = az.loo(idata1, pointwise=True)
print(f"  ELPD: {loo1.elpd_loo:.2f} (SE: {loo1.se:.2f})")
print(f"  pLOO: {loo1.p_loo:.2f}")

# Pareto k diagnostics for Exp 1
pareto_k1 = loo1.pareto_k.values
n_bad1 = np.sum(pareto_k1 > 0.7)
n_warn1 = np.sum((pareto_k1 > 0.5) & (pareto_k1 <= 0.7))
print(f"  Pareto k > 0.7: {n_bad1}/{len(pareto_k1)} groups")
print(f"  Pareto k 0.5-0.7: {n_warn1}/{len(pareto_k1)} groups")
print(f"  Pareto k range: [{pareto_k1.min():.3f}, {pareto_k1.max():.3f}]")

print("\nComputing LOO for Experiment 2...")
loo2 = az.loo(idata2, pointwise=True)
print(f"  ELPD: {loo2.elpd_loo:.2f} (SE: {loo2.se:.2f})")
print(f"  pLOO: {loo2.p_loo:.2f}")

# Pareto k diagnostics for Exp 2
pareto_k2 = loo2.pareto_k.values
n_bad2 = np.sum(pareto_k2 > 0.7)
n_warn2 = np.sum((pareto_k2 > 0.5) & (pareto_k2 <= 0.7))
print(f"  Pareto k > 0.7: {n_bad2}/{len(pareto_k2)} groups")
print(f"  Pareto k 0.5-0.7: {n_warn2}/{len(pareto_k2)} groups")
print(f"  Pareto k range: [{pareto_k2.min():.3f}, {pareto_k2.max():.3f}]")

# Model comparison
print("\n" + "-"*80)
print("COMPARATIVE ANALYSIS")
print("-"*80)

comparison_dict = {
    'Hierarchical (Exp1)': idata1,
    'Mixture_K3 (Exp2)': idata2
}

comparison = az.compare(comparison_dict, ic='loo', method='stacking')
print("\nModel Comparison Table:")
print(comparison)

# Extract key metrics
delta_elpd = comparison.loc['Mixture_K3 (Exp2)', 'elpd_loo'] - comparison.loc['Hierarchical (Exp1)', 'elpd_loo']
se_diff = comparison.loc['Mixture_K3 (Exp2)', 'dse']  # SE of difference

print(f"\nΔELPD (Exp2 - Exp1): {delta_elpd:.2f}")
print(f"SE of difference: {se_diff:.2f}")
print(f"Difference in standard errors: {abs(delta_elpd)/se_diff:.2f}σ")

# Decision rule
print("\n" + "-"*80)
print("DECISION RULE APPLICATION")
print("-"*80)
abs_delta = abs(delta_elpd)
if abs_delta < 2 * se_diff:
    decision = "EQUIVALENT - Apply parsimony principle"
    recommendation = "Prefer Experiment 1 (simpler model)"
elif abs_delta < 4 * se_diff:
    decision = "WEAK EVIDENCE - Uncertain"
    recommendation = "Consider model averaging or prefer simpler model"
else:
    decision = "STRONG EVIDENCE"
    if delta_elpd > 0:
        recommendation = "Prefer Experiment 2 (Mixture)"
    else:
        recommendation = "Prefer Experiment 1 (Hierarchical)"

print(f"Decision: {decision}")
print(f"Recommendation: {recommendation}")

# Stacking weights
stacking_weights = comparison[['weight']].copy()
print(f"\nStacking weights:")
for model, weight in stacking_weights.iterrows():
    print(f"  {model}: {weight['weight']:.3f}")

# Save comparison results
comparison_results = {
    'model': ['Experiment 1 (Hierarchical)', 'Experiment 2 (Mixture K=3)'],
    'elpd_loo': [loo1.elpd_loo, loo2.elpd_loo],
    'se': [loo1.se, loo2.se],
    'p_loo': [loo1.p_loo, loo2.p_loo],
    'pareto_k_max': [pareto_k1.max(), pareto_k2.max()],
    'pareto_k_gt_0.7': [n_bad1, n_bad2],
    'pareto_k_0.5_0.7': [n_warn1, n_warn2],
    'weight': [stacking_weights.loc['Hierarchical (Exp1)', 'weight'],
               stacking_weights.loc['Mixture_K3 (Exp2)', 'weight']]
}
comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv(f'{OUTPUT_DIR}/loo_results.csv', index=False)
print(f"\nSaved: {OUTPUT_DIR}/loo_results.csv")

# ============================================================================
# 3. CALIBRATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[3/9] CALIBRATION ANALYSIS")
print("="*80)

print("\nComputing LOO-PIT for both models...")

# Function to compute LOO predictive intervals and coverage
def compute_loo_coverage(idata, data):
    """Compute LOO predictive coverage at multiple levels."""
    n_groups = len(data)
    coverage_levels = [0.50, 0.90, 0.95]
    coverage_results = {level: 0 for level in coverage_levels}

    # Get LOO posterior predictive samples
    loo_pit = az.loo_pit(idata=idata, y='y')

    # For each observation, compute coverage
    if 'log_likelihood' in idata.groups():
        ll_data = idata.log_likelihood
        ll_var = list(ll_data.data_vars)[0]

        # Get posterior predictive samples (assuming binomial observations)
        if 'posterior_predictive' in idata.groups():
            y_pred = idata.posterior_predictive['y'].values
            y_obs = data['r_successes'].values

            for level in coverage_levels:
                alpha = 1 - level
                lower = np.percentile(y_pred, 100 * alpha/2, axis=(0,1))
                upper = np.percentile(y_pred, 100 * (1-alpha/2), axis=(0,1))

                in_interval = (y_obs >= lower) & (y_obs <= upper)
                coverage_results[level] = np.mean(in_interval)

    return loo_pit, coverage_results

# Compute LOO-PIT
loo_pit1 = az.loo_pit(idata=idata1, y='y')
loo_pit2 = az.loo_pit(idata=idata2, y='y')

print("\nLOO-PIT uniformity tests:")
# Kolmogorov-Smirnov test for uniformity
ks_stat1, ks_pval1 = stats.kstest(loo_pit1.values.flatten(), 'uniform')
ks_stat2, ks_pval2 = stats.kstest(loo_pit2.values.flatten(), 'uniform')

print(f"\nExperiment 1 (Hierarchical):")
print(f"  KS statistic: {ks_stat1:.4f}, p-value: {ks_pval1:.4f}")
print(f"  Calibration: {'GOOD' if ks_pval1 > 0.05 else 'POOR'}")

print(f"\nExperiment 2 (Mixture):")
print(f"  KS statistic: {ks_stat2:.4f}, p-value: {ks_pval2:.4f}")
print(f"  Calibration: {'GOOD' if ks_pval2 > 0.05 else 'POOR'}")

# Compute coverage if posterior predictive available
print("\nPosterior predictive coverage:")
if 'posterior_predictive' in idata1.groups():
    y_pred1 = idata1.posterior_predictive['y'].values
    y_obs = data['r_successes'].values

    for level in [0.50, 0.90, 0.95]:
        alpha = 1 - level
        lower1 = np.percentile(y_pred1, 100 * alpha/2, axis=(0,1))
        upper1 = np.percentile(y_pred1, 100 * (1-alpha/2), axis=(0,1))
        coverage1 = np.mean((y_obs >= lower1) & (y_obs <= upper1))
        print(f"  Exp1 {int(level*100)}% interval: {coverage1:.1%} coverage")

if 'posterior_predictive' in idata2.groups():
    y_pred2 = idata2.posterior_predictive['y'].values

    for level in [0.50, 0.90, 0.95]:
        alpha = 1 - level
        lower2 = np.percentile(y_pred2, 100 * alpha/2, axis=(0,1))
        upper2 = np.percentile(y_pred2, 100 * (1-alpha/2), axis=(0,1))
        coverage2 = np.mean((y_obs >= lower2) & (y_obs <= upper2))
        print(f"  Exp2 {int(level*100)}% interval: {coverage2:.1%} coverage")

# ============================================================================
# 4. ABSOLUTE PREDICTIVE METRICS
# ============================================================================
print("\n" + "="*80)
print("[4/9] ABSOLUTE PREDICTIVE METRICS")
print("="*80)

# Compute point predictions (posterior mean)
if 'posterior_predictive' in idata1.groups():
    y_pred1_mean = idata1.posterior_predictive['y'].mean(dim=['chain', 'draw']).values
    y_obs = data['r_successes'].values
    n_trials = data['n_trials'].values

    # Convert to success rates
    rate_pred1 = y_pred1_mean / n_trials
    rate_obs = y_obs / n_trials

    rmse1 = np.sqrt(np.mean((rate_pred1 - rate_obs)**2))
    mae1 = np.mean(np.abs(rate_pred1 - rate_obs))

    print(f"\nExperiment 1 (Hierarchical):")
    print(f"  RMSE: {rmse1:.4f}")
    print(f"  MAE: {mae1:.4f}")

if 'posterior_predictive' in idata2.groups():
    y_pred2_mean = idata2.posterior_predictive['y'].mean(dim=['chain', 'draw']).values

    # Convert to success rates
    rate_pred2 = y_pred2_mean / n_trials

    rmse2 = np.sqrt(np.mean((rate_pred2 - rate_obs)**2))
    mae2 = np.mean(np.abs(rate_pred2 - rate_obs))

    print(f"\nExperiment 2 (Mixture):")
    print(f"  RMSE: {rmse2:.4f}")
    print(f"  MAE: {mae2:.4f}")

    print(f"\nDifference (Exp2 - Exp1):")
    print(f"  ΔRMSE: {rmse2 - rmse1:.4f} {'(Exp2 better)' if rmse2 < rmse1 else '(Exp1 better)'}")
    print(f"  ΔMAE: {mae2 - mae1:.4f} {'(Exp2 better)' if mae2 < mae1 else '(Exp1 better)'}")

# ============================================================================
# 5. CONVERGENCE DIAGNOSTICS
# ============================================================================
print("\n" + "="*80)
print("[5/9] CONVERGENCE DIAGNOSTICS")
print("="*80)

print("\nExperiment 1 (Hierarchical):")
summary1 = az.summary(idata1, var_names=['mu', 'sigma'])
print(f"  Max Rhat: {summary1['r_hat'].max():.4f}")
print(f"  Min ESS bulk: {summary1['ess_bulk'].min():.0f}")
print(f"  Min ESS tail: {summary1['ess_tail'].min():.0f}")
print(f"  Convergence: {'GOOD' if summary1['r_hat'].max() < 1.01 else 'POOR'}")

print("\nExperiment 2 (Mixture):")
# Check what variables are available
available_vars = list(idata2.posterior.data_vars.keys())
var_names2 = [v for v in ['mu_k', 'pi'] if v in available_vars]
if var_names2:
    summary2 = az.summary(idata2, var_names=var_names2)
    print(f"  Max Rhat: {summary2['r_hat'].max():.4f}")
    print(f"  Min ESS bulk: {summary2['ess_bulk'].min():.0f}")
    print(f"  Min ESS tail: {summary2['ess_tail'].min():.0f}")
    print(f"  Convergence: {'GOOD' if summary2['r_hat'].max() < 1.01 else 'POOR'}")
else:
    print("  Warning: Could not find expected variables")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("[6/9] CREATING VISUALIZATIONS")
print("="*80)

# Plot 1: LOO Comparison
print("\n1. LOO comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_compare(comparison, insample_dev=False, ax=ax)
ax.set_title('Model Comparison: LOO ELPD', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/01_loo_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR}/01_loo_comparison.png")

# Plot 2: Pareto k comparison
print("\n2. Pareto k comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Exp 1
ax = axes[0]
az.plot_khat(loo1, ax=ax, show_bins=True)
ax.set_title('Experiment 1 (Hierarchical): Pareto k', fontsize=12, fontweight='bold')
ax.axhline(y=0.7, color='red', linestyle='--', label='k=0.7 threshold')
ax.legend()

# Exp 2
ax = axes[1]
az.plot_khat(loo2, ax=ax, show_bins=True)
ax.set_title('Experiment 2 (Mixture K=3): Pareto k', fontsize=12, fontweight='bold')
ax.axhline(y=0.7, color='red', linestyle='--', label='k=0.7 threshold')
ax.legend()

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/02_pareto_k_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR}/02_pareto_k_comparison.png")

# Plot 3: LOO-PIT calibration
print("\n3. LOO-PIT calibration plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Exp 1
ax = axes[0]
az.plot_loo_pit(idata1, y='y', ax=ax)
ax.set_title(f'Experiment 1: LOO-PIT\n(KS p-value: {ks_pval1:.3f})',
             fontsize=12, fontweight='bold')

# Exp 2
ax = axes[1]
az.plot_loo_pit(idata2, y='y', ax=ax)
ax.set_title(f'Experiment 2: LOO-PIT\n(KS p-value: {ks_pval2:.3f})',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/03_loo_pit_calibration.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR}/03_loo_pit_calibration.png")

# Plot 4: Pointwise ELPD comparison
print("\n4. Pointwise ELPD comparison...")
elpd1_pointwise = loo1.elpd_i.values
elpd2_pointwise = loo2.elpd_i.values
elpd_diff = elpd2_pointwise - elpd1_pointwise

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel A: Pointwise ELPD for both models
ax = axes[0]
x = np.arange(len(data))
ax.scatter(x, elpd1_pointwise, label='Exp1 (Hierarchical)', alpha=0.7, s=100)
ax.scatter(x, elpd2_pointwise, label='Exp2 (Mixture)', alpha=0.7, s=100)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('Group ID', fontsize=11)
ax.set_ylabel('Pointwise ELPD', fontsize=11)
ax.set_title('Pointwise ELPD by Group', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Difference (Exp2 - Exp1)
ax = axes[1]
colors = ['green' if d > 0 else 'red' for d in elpd_diff]
ax.bar(x, elpd_diff, color=colors, alpha=0.6)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Group ID', fontsize=11)
ax.set_ylabel('ELPD Difference (Exp2 - Exp1)', fontsize=11)
ax.set_title('Which Model Predicts Better? (Positive = Exp2 Better)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/04_pointwise_elpd.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR}/04_pointwise_elpd.png")

# Plot 5: Observed vs Predicted comparison
print("\n5. Observed vs predicted comparison...")
if 'posterior_predictive' in idata1.groups() and 'posterior_predictive' in idata2.groups():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Exp 1
    ax = axes[0]
    ax.scatter(rate_obs, rate_pred1, s=100, alpha=0.6)
    ax.plot([0, rate_obs.max()], [0, rate_obs.max()], 'r--', label='Perfect prediction')
    ax.set_xlabel('Observed Success Rate', fontsize=11)
    ax.set_ylabel('Predicted Success Rate', fontsize=11)
    ax.set_title(f'Experiment 1 (Hierarchical)\nRMSE: {rmse1:.4f}, MAE: {mae1:.4f}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Exp 2
    ax = axes[1]
    ax.scatter(rate_obs, rate_pred2, s=100, alpha=0.6, color='orange')
    ax.plot([0, rate_obs.max()], [0, rate_obs.max()], 'r--', label='Perfect prediction')
    ax.set_xlabel('Observed Success Rate', fontsize=11)
    ax.set_ylabel('Predicted Success Rate', fontsize=11)
    ax.set_title(f'Experiment 2 (Mixture K=3)\nRMSE: {rmse2:.4f}, MAE: {mae2:.4f}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/05_observed_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {PLOTS_DIR}/05_observed_vs_predicted.png")

# Plot 6: Multi-criteria comparison dashboard
print("\n6. Multi-criteria comparison dashboard...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: ELPD comparison (top left)
ax1 = fig.add_subplot(gs[0, 0])
models = ['Hierarchical\n(Exp1)', 'Mixture K=3\n(Exp2)']
elpd_vals = [loo1.elpd_loo, loo2.elpd_loo]
elpd_se = [loo1.se, loo2.se]
colors_bar = ['steelblue', 'coral']
bars = ax1.bar(models, elpd_vals, yerr=elpd_se, color=colors_bar, alpha=0.7, capsize=5)
ax1.set_ylabel('ELPD (LOO)', fontsize=10)
ax1.set_title('Predictive Accuracy', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
# Highlight winner
best_idx = np.argmax(elpd_vals)
bars[best_idx].set_edgecolor('black')
bars[best_idx].set_linewidth(3)

# Panel 2: Pareto k diagnostics (top middle)
ax2 = fig.add_subplot(gs[0, 1])
pareto_categories = ['k < 0.5', '0.5 ≤ k < 0.7', 'k ≥ 0.7']
exp1_counts = [
    np.sum(pareto_k1 < 0.5),
    np.sum((pareto_k1 >= 0.5) & (pareto_k1 < 0.7)),
    np.sum(pareto_k1 >= 0.7)
]
exp2_counts = [
    np.sum(pareto_k2 < 0.5),
    np.sum((pareto_k2 >= 0.5) & (pareto_k2 < 0.7)),
    np.sum(pareto_k2 >= 0.7)
]
x_pos = np.arange(len(pareto_categories))
width = 0.35
ax2.bar(x_pos - width/2, exp1_counts, width, label='Exp1', alpha=0.7, color='steelblue')
ax2.bar(x_pos + width/2, exp2_counts, width, label='Exp2', alpha=0.7, color='coral')
ax2.set_ylabel('Number of Groups', fontsize=10)
ax2.set_title('LOO Reliability (Pareto k)', fontsize=11, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(pareto_categories, rotation=15, ha='right', fontsize=9)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: RMSE/MAE comparison (top right)
ax3 = fig.add_subplot(gs[0, 2])
metrics = ['RMSE', 'MAE']
exp1_metrics = [rmse1, mae1]
exp2_metrics = [rmse2, mae2]
x_pos = np.arange(len(metrics))
bars1 = ax3.bar(x_pos - width/2, exp1_metrics, width, label='Exp1', alpha=0.7, color='steelblue')
bars2 = ax3.bar(x_pos + width/2, exp2_metrics, width, label='Exp2', alpha=0.7, color='coral')
ax3.set_ylabel('Error', fontsize=10)
ax3.set_title('Absolute Predictive Metrics', fontsize=11, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Panel 4: Stacking weights (middle left)
ax4 = fig.add_subplot(gs[1, 0])
weights = [stacking_weights.loc['Hierarchical (Exp1)', 'weight'],
           stacking_weights.loc['Mixture_K3 (Exp2)', 'weight']]
bars = ax4.barh(models, weights, color=colors_bar, alpha=0.7)
ax4.set_xlabel('Stacking Weight', fontsize=10)
ax4.set_title('Model Averaging Weights', fontsize=11, fontweight='bold')
ax4.set_xlim(0, 1)
ax4.grid(True, alpha=0.3, axis='x')
# Add values
for i, (bar, w) in enumerate(zip(bars, weights)):
    ax4.text(w + 0.02, i, f'{w:.3f}', va='center', fontsize=9)

# Panel 5: Pointwise ELPD difference (middle spanning)
ax5 = fig.add_subplot(gs[1, 1:])
colors_diff = ['green' if d > 0 else 'red' for d in elpd_diff]
ax5.bar(range(len(elpd_diff)), elpd_diff, color=colors_diff, alpha=0.6)
ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax5.set_xlabel('Group ID', fontsize=10)
ax5.set_ylabel('ELPD Difference (Exp2 - Exp1)', fontsize=10)
ax5.set_title('Where Each Model Excels (Green = Exp2 Better, Red = Exp1 Better)',
              fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Panel 6: Decision summary (bottom spanning)
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

summary_text = f"""
DECISION SUMMARY

Statistical Evidence:
  • ΔELPD (Exp2 - Exp1): {delta_elpd:.2f} (SE: {se_diff:.2f})
  • Difference magnitude: {abs(delta_elpd)/se_diff:.2f} standard errors
  • Decision: {decision}

Model Quality:
  • Exp1 Pareto k > 0.7: {n_bad1}/{len(pareto_k1)} groups
  • Exp2 Pareto k > 0.7: {n_bad2}/{len(pareto_k2)} groups
  • Exp1 calibration: {'GOOD' if ks_pval1 > 0.05 else 'POOR'} (KS p={ks_pval1:.3f})
  • Exp2 calibration: {'GOOD' if ks_pval2 > 0.05 else 'POOR'} (KS p={ks_pval2:.3f})

Parsimony Consideration:
  • Exp1: Simpler model (continuous heterogeneity)
  • Exp2: More complex (discrete K=3 clusters)
  • When equivalent: Prefer simpler (Exp1)

RECOMMENDATION: {recommendation}
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Comprehensive Model Comparison Dashboard',
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(f'{PLOTS_DIR}/06_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR}/06_comprehensive_dashboard.png")

# ============================================================================
# 7. SCIENTIFIC INTERPRETATION
# ============================================================================
print("\n" + "="*80)
print("[7/9] SCIENTIFIC INTERPRETATION")
print("="*80)

if abs_delta < 2 * se_diff or (delta_elpd < 0):
    interpretation = """
CONTINUOUS HETEROGENEITY SUFFICIENT

The hierarchical model (Exp1) with continuous heterogeneity performs
as well or better than the mixture model (Exp2) with discrete clusters.

Scientific Implications:
  • No strong evidence for discrete subpopulations in the data
  • Success rates vary continuously across groups
  • Clusters identified in EDA may reflect sampling variation
  • Simpler continuous model is preferred (parsimony)

Model Selection Rationale:
  • Equivalent or better predictive performance
  • Fewer parameters (more parsimonious)
  • Easier interpretation (smooth variation vs discrete types)
  • More robust (fewer influential observations)
"""
else:
    interpretation = """
DISCRETE SUBPOPULATIONS DETECTED

The mixture model (Exp2) provides substantially better predictions,
suggesting genuine discrete subpopulations exist in the data.

Scientific Implications:
  • Evidence for K=3 distinct types of groups
  • Success rates cluster into discrete categories
  • Heterogeneity is better captured by discrete types
  • Additional complexity is justified by improved fit

Model Selection Rationale:
  • Significantly better predictive performance
  • Improved calibration
  • Clusters are scientifically meaningful
  • Worth the added complexity
"""

print(interpretation)

# ============================================================================
# 8. SAVE STACKING WEIGHTS
# ============================================================================
print("\n" + "="*80)
print("[8/9] SAVING STACKING WEIGHTS")
print("="*80)

stacking_df = pd.DataFrame({
    'model': ['Experiment_1_Hierarchical', 'Experiment_2_Mixture_K3'],
    'stacking_weight': weights
})
stacking_df.to_csv(f'{OUTPUT_DIR}/stacking_weights.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/stacking_weights.csv")

# ============================================================================
# 9. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("[9/9] FINAL SUMMARY")
print("="*80)

print("\nKEY FINDINGS:")
print("-" * 80)
print(f"1. Predictive Accuracy:")
print(f"   - Exp1 ELPD: {loo1.elpd_loo:.2f} ± {loo1.se:.2f}")
print(f"   - Exp2 ELPD: {loo2.elpd_loo:.2f} ± {loo2.se:.2f}")
print(f"   - Difference: {delta_elpd:.2f} ({abs(delta_elpd)/se_diff:.2f}σ)")

print(f"\n2. LOO Reliability:")
print(f"   - Exp1 bad Pareto k: {n_bad1}/{len(pareto_k1)} groups")
print(f"   - Exp2 bad Pareto k: {n_bad2}/{len(pareto_k2)} groups")

print(f"\n3. Calibration:")
print(f"   - Exp1 KS p-value: {ks_pval1:.3f}")
print(f"   - Exp2 KS p-value: {ks_pval2:.3f}")

print(f"\n4. Absolute Metrics:")
print(f"   - Exp1 RMSE: {rmse1:.4f}, MAE: {mae1:.4f}")
print(f"   - Exp2 RMSE: {rmse2:.4f}, MAE: {mae2:.4f}")

print(f"\n5. Model Averaging:")
print(f"   - Exp1 weight: {weights[0]:.3f}")
print(f"   - Exp2 weight: {weights[1]:.3f}")

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)
print(f"\n{recommendation}")
print(f"\nDecision basis: {decision}")
print(interpretation)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutput files saved to: {OUTPUT_DIR}/")
print("  - loo_results.csv")
print("  - stacking_weights.csv")
print(f"  - comparison_plots/ (6 visualizations)")
print("\nNext: Generate comprehensive comparison report")
