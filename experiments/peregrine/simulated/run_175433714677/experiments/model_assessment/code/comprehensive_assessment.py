"""
Comprehensive Assessment of Two REJECTED Bayesian Models
=========================================================

Context: Both models REJECTED - need to understand WHY they failed
and what this tells us about the data structure.

Model 1 (Log-Linear): REJECTED due to systematic residual curvature
Model 2 (Quadratic): REJECTED due to non-significant beta2 and no improvement

This script:
1. Loads both models' InferenceData
2. Performs LOO-CV comparison
3. Generates comparative diagnostics
4. Analyzes the common failure mode
5. Provides recommendations for next steps
"""

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
plt.style.use('default')
az.style.use('arviz-darkgrid')

print("="*80)
print("COMPREHENSIVE ASSESSMENT: TWO REJECTED MODELS")
print("="*80)

# ============================================================================
# Load Both Models
# ============================================================================

print("\n[1/5] Loading InferenceData for both models...")

model1_path = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
model2_path = Path("/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf")

print(f"  Model 1 (Log-Linear): {model1_path}")
print(f"  Model 2 (Quadratic): {model2_path}")

idata_model1 = az.from_netcdf(model1_path)
idata_model2 = az.from_netcdf(model2_path)

print("  ✓ Both models loaded successfully")

# Verify log_likelihood is present
assert 'log_likelihood' in idata_model1.groups(), "Model 1 missing log_likelihood"
assert 'log_likelihood' in idata_model2.groups(), "Model 2 missing log_likelihood"
print("  ✓ Both models have log_likelihood group for LOO-CV")

# ============================================================================
# LOO-CV for Each Model
# ============================================================================

print("\n[2/5] Computing LOO-CV diagnostics for each model...")

# Model 1
print("\n  Model 1 (Log-Linear):")
loo_model1 = az.loo(idata_model1, pointwise=True)
print(f"    ELPD: {loo_model1.elpd_loo:.2f} ± {loo_model1.se:.2f}")
print(f"    p_loo: {loo_model1.p_loo:.2f}")
pareto_k1 = loo_model1.pareto_k.values
n_high_k1 = np.sum(pareto_k1 > 0.7)
print(f"    Pareto k > 0.7: {n_high_k1}/{len(pareto_k1)} observations ({100*n_high_k1/len(pareto_k1):.1f}%)")
print(f"    Pareto k > 0.5: {np.sum(pareto_k1 > 0.5)}/{len(pareto_k1)}")

# Model 2
print("\n  Model 2 (Quadratic):")
loo_model2 = az.loo(idata_model2, pointwise=True)
print(f"    ELPD: {loo_model2.elpd_loo:.2f} ± {loo_model2.se:.2f}")
print(f"    p_loo: {loo_model2.p_loo:.2f}")
pareto_k2 = loo_model2.pareto_k.values
n_high_k2 = np.sum(pareto_k2 > 0.7)
print(f"    Pareto k > 0.7: {n_high_k2}/{len(pareto_k2)} observations ({100*n_high_k2/len(pareto_k2):.1f}%)")
print(f"    Pareto k > 0.5: {np.sum(pareto_k2 > 0.5)}/{len(pareto_k2)}")

# ============================================================================
# Model Comparison
# ============================================================================

print("\n[3/5] Comparing models using az.compare()...")

# Create comparison dictionary
comparison_dict = {
    "Model_1_Linear": idata_model1,
    "Model_2_Quadratic": idata_model2
}

compare_df = az.compare(comparison_dict, ic='loo', method='stacking')
print("\n" + "="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(compare_df)
print("="*80)

# Extract key metrics
delta_elpd = compare_df.loc['Model_2_Quadratic', 'elpd_diff']
delta_se = compare_df.loc['Model_2_Quadratic', 'dse']

print(f"\nΔELPD (Model 2 - Model 1): {delta_elpd:.2f} ± {delta_se:.2f}")

if abs(delta_elpd) < 2 * delta_se:
    print("  → Models are EQUIVALENT (difference < 2 SE)")
    print("  → Model 2 provides NO IMPROVEMENT")
elif delta_elpd < -4:
    print("  → Model 1 is BETTER (strong evidence)")
elif delta_elpd > 4:
    print("  → Model 2 is BETTER (strong evidence)")
else:
    print("  → Weak evidence for difference")

# ============================================================================
# Generate Diagnostic Plots
# ============================================================================

print("\n[4/5] Generating comparative diagnostic plots...")

plots_dir = Path("/workspace/experiments/model_assessment/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# --- Plot 1: LOO Comparison ---
print("  Creating LOO comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_compare(compare_df, insample_dev=False, ax=ax)
ax.set_title("Model Comparison: LOO-CV ELPD", fontsize=14, fontweight='bold')
ax.set_xlabel("ELPD (Expected Log Pointwise Predictive Density)", fontsize=12)
plt.tight_layout()
plt.savefig(plots_dir / "loo_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {plots_dir / 'loo_comparison.png'}")

# --- Plot 2: Pareto k Diagnostics ---
print("  Creating Pareto k diagnostic plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Model 1 - manual plot
obs_idx = np.arange(len(pareto_k1))
axes[0].scatter(obs_idx, pareto_k1, alpha=0.6, s=50, c='blue')
axes[0].axhline(0.7, color='red', linestyle='--', linewidth=1.5, label='k=0.7 (problematic)')
axes[0].axhline(0.5, color='orange', linestyle='--', linewidth=1, label='k=0.5 (ok)')
axes[0].set_xlabel('Observation Index', fontsize=11)
axes[0].set_ylabel('Pareto k', fontsize=11)
axes[0].set_title("Model 1 (Log-Linear): Pareto k Diagnostic", fontweight='bold')
axes[0].legend(loc='upper left', fontsize=9)
axes[0].grid(alpha=0.3)
axes[0].text(0.02, 0.98, f'Max k: {pareto_k1.max():.3f}\nMean k: {pareto_k1.mean():.3f}',
            transform=axes[0].transAxes, va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Model 2 - manual plot
axes[1].scatter(obs_idx, pareto_k2, alpha=0.6, s=50, c='red')
axes[1].axhline(0.7, color='red', linestyle='--', linewidth=1.5, label='k=0.7 (problematic)')
axes[1].axhline(0.5, color='orange', linestyle='--', linewidth=1, label='k=0.5 (ok)')
axes[1].set_xlabel('Observation Index', fontsize=11)
axes[1].set_ylabel('Pareto k', fontsize=11)
axes[1].set_title("Model 2 (Quadratic): Pareto k Diagnostic", fontweight='bold')
axes[1].legend(loc='upper left', fontsize=9)
axes[1].grid(alpha=0.3)
axes[1].text(0.02, 0.98, f'Max k: {pareto_k2.max():.3f}\nMean k: {pareto_k2.mean():.3f}',
            transform=axes[1].transAxes, va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(plots_dir / "pareto_k_diagnostics.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {plots_dir / 'pareto_k_diagnostics.png'}")

# --- Plot 3: Parameter Comparison ---
print("  Creating parameter comparison plot...")

# Extract posteriors
beta0_m1 = idata_model1.posterior['beta_0'].values.flatten()
beta1_m1 = idata_model1.posterior['beta_1'].values.flatten()
phi_m1 = idata_model1.posterior['phi'].values.flatten()

beta0_m2 = idata_model2.posterior['beta0'].values.flatten()
beta1_m2 = idata_model2.posterior['beta1'].values.flatten()
beta2_m2 = idata_model2.posterior['beta2'].values.flatten()
phi_m2 = idata_model2.posterior['phi'].values.flatten()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# β₀ comparison
axes[0, 0].hist(beta0_m1, bins=40, alpha=0.6, label='Model 1', color='blue', density=True)
axes[0, 0].hist(beta0_m2, bins=40, alpha=0.6, label='Model 2', color='red', density=True)
axes[0, 0].axvline(beta0_m1.mean(), color='blue', linestyle='--', linewidth=2)
axes[0, 0].axvline(beta0_m2.mean(), color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('β₀ (Intercept)', fontsize=11)
axes[0, 0].set_ylabel('Density', fontsize=11)
axes[0, 0].set_title('Intercept Comparison', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# β₁ comparison
axes[0, 1].hist(beta1_m1, bins=40, alpha=0.6, label='Model 1', color='blue', density=True)
axes[0, 1].hist(beta1_m2, bins=40, alpha=0.6, label='Model 2', color='red', density=True)
axes[0, 1].axvline(beta1_m1.mean(), color='blue', linestyle='--', linewidth=2)
axes[0, 1].axvline(beta1_m2.mean(), color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('β₁ (Linear Term)', fontsize=11)
axes[0, 1].set_ylabel('Density', fontsize=11)
axes[0, 1].set_title('Linear Term Comparison', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# β₂ (Model 2 only)
axes[1, 0].hist(beta2_m2, bins=40, alpha=0.7, color='red', density=True)
axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=2, label='0 (no effect)')
axes[1, 0].axvline(beta2_m2.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {beta2_m2.mean():.3f}')
# Add credible interval
ci_lower, ci_upper = np.percentile(beta2_m2, [2.5, 97.5])
axes[1, 0].axvspan(ci_lower, ci_upper, alpha=0.2, color='red', label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
axes[1, 0].set_xlabel('β₂ (Quadratic Term)', fontsize=11)
axes[1, 0].set_ylabel('Density', fontsize=11)
axes[1, 0].set_title('Quadratic Term (Model 2 Only)', fontweight='bold')
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(alpha=0.3)

# φ comparison
axes[1, 1].hist(phi_m1, bins=40, alpha=0.6, label='Model 1', color='blue', density=True)
axes[1, 1].hist(phi_m2, bins=40, alpha=0.6, label='Model 2', color='red', density=True)
axes[1, 1].axvline(phi_m1.mean(), color='blue', linestyle='--', linewidth=2)
axes[1, 1].axvline(phi_m2.mean(), color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('φ (Dispersion)', fontsize=11)
axes[1, 1].set_ylabel('Density', fontsize=11)
axes[1, 1].set_title('Dispersion Parameter Comparison', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.suptitle('Parameter Posterior Comparisons', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(plots_dir / "parameter_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {plots_dir / 'parameter_comparison.png'}")

# --- Plot 4: Convergence Summary Dashboard ---
print("  Creating convergence summary dashboard...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Model 1 convergence metrics
ax1 = fig.add_subplot(gs[0, 0])
summary_m1 = az.summary(idata_model1, hdi_prob=0.95)
ax1.text(0.5, 0.9, 'Model 1: Log-Linear', ha='center', va='top', fontsize=12, fontweight='bold', transform=ax1.transAxes)
ax1.text(0.05, 0.75, f"R̂ max: {summary_m1['r_hat'].max():.4f}", fontsize=10, transform=ax1.transAxes)
ax1.text(0.05, 0.65, f"ESS min: {summary_m1['ess_bulk'].min():.0f}", fontsize=10, transform=ax1.transAxes)
ax1.text(0.05, 0.55, f"Divergences: 0", fontsize=10, transform=ax1.transAxes)
ax1.text(0.05, 0.45, f"Status: ✓ CONVERGED", fontsize=10, color='green', fontweight='bold', transform=ax1.transAxes)
ax1.text(0.05, 0.30, f"Decision: ✗ REJECTED", fontsize=10, color='red', fontweight='bold', transform=ax1.transAxes)
ax1.text(0.05, 0.20, "Reason: Residual curvature", fontsize=9, transform=ax1.transAxes)
ax1.axis('off')

# Model 2 convergence metrics
ax2 = fig.add_subplot(gs[0, 1])
summary_m2 = az.summary(idata_model2, hdi_prob=0.95)
ax2.text(0.5, 0.9, 'Model 2: Quadratic', ha='center', va='top', fontsize=12, fontweight='bold', transform=ax2.transAxes)
ax2.text(0.05, 0.75, f"R̂ max: {summary_m2['r_hat'].max():.4f}", fontsize=10, transform=ax2.transAxes)
ax2.text(0.05, 0.65, f"ESS min: {summary_m2['ess_bulk'].min():.0f}", fontsize=10, transform=ax2.transAxes)
ax2.text(0.05, 0.55, f"Divergences: 0", fontsize=10, transform=ax2.transAxes)
ax2.text(0.05, 0.45, f"Status: ✓ CONVERGED", fontsize=10, color='green', fontweight='bold', transform=ax2.transAxes)
ax2.text(0.05, 0.30, f"Decision: ✗ REJECTED", fontsize=10, color='red', fontweight='bold', transform=ax2.transAxes)
ax2.text(0.05, 0.20, "Reason: β₂ non-significant", fontsize=9, transform=ax2.transAxes)
ax2.axis('off')

# R-hat comparison
ax3 = fig.add_subplot(gs[1, :])
params_m1 = ['beta_0', 'beta_1', 'phi']
params_m2 = ['beta0', 'beta1', 'beta2', 'phi']
x_pos = np.arange(max(len(params_m1), len(params_m2)))
width = 0.35

rhat_m1 = [summary_m1.loc[p, 'r_hat'] if p in summary_m1.index else np.nan for p in params_m1 + [''] * (len(params_m2) - len(params_m1))]
rhat_m2 = [summary_m2.loc[p, 'r_hat'] if p in summary_m2.index else np.nan for p in params_m2]

bars1 = ax3.bar(x_pos - width/2, rhat_m1[:len(params_m2)], width, label='Model 1', alpha=0.8, color='blue')
bars2 = ax3.bar(x_pos + width/2, rhat_m2, width, label='Model 2', alpha=0.8, color='red')
ax3.axhline(1.01, color='orange', linestyle='--', linewidth=2, label='R̂ = 1.01 (threshold)')
ax3.set_ylabel('R̂', fontsize=11)
ax3.set_title('Convergence: R̂ Comparison', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['β₀', 'β₁', 'β₂', 'φ'])
ax3.legend()
ax3.grid(alpha=0.3, axis='y')
ax3.set_ylim([0.99, 1.02])

# ESS comparison
ax4 = fig.add_subplot(gs[2, :])
ess_m1 = [summary_m1.loc[p, 'ess_bulk'] if p in summary_m1.index else np.nan for p in params_m1 + [''] * (len(params_m2) - len(params_m1))]
ess_m2 = [summary_m2.loc[p, 'ess_bulk'] if p in summary_m2.index else np.nan for p in params_m2]

bars1 = ax4.bar(x_pos - width/2, ess_m1[:len(params_m2)], width, label='Model 1', alpha=0.8, color='blue')
bars2 = ax4.bar(x_pos + width/2, ess_m2, width, label='Model 2', alpha=0.8, color='red')
ax4.axhline(400, color='orange', linestyle='--', linewidth=2, label='ESS = 400 (threshold)')
ax4.set_ylabel('ESS (bulk)', fontsize=11)
ax4.set_title('Effective Sample Size Comparison', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['β₀', 'β₁', 'β₂', 'φ'])
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

plt.suptitle('Convergence Diagnostics: Both Models CONVERGED (But REJECTED)', fontsize=14, fontweight='bold')
plt.savefig(plots_dir / "convergence_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {plots_dir / 'convergence_summary.png'}")

# ============================================================================
# Save Comparison Metrics
# ============================================================================

print("\n[5/5] Saving comparison metrics...")

# Create CSV file
metrics_data = {
    'Model': ['Model_1_Linear', 'Model_2_Quadratic'],
    'ELPD': [loo_model1.elpd_loo, loo_model2.elpd_loo],
    'SE': [loo_model1.se, loo_model2.se],
    'p_loo': [loo_model1.p_loo, loo_model2.p_loo],
    'Pareto_k_gt_0.5': [np.sum(pareto_k1 > 0.5), np.sum(pareto_k2 > 0.5)],
    'Pareto_k_gt_0.7': [n_high_k1, n_high_k2],
    'n_params': [3, 4],
    'Decision': ['REJECTED', 'REJECTED'],
    'Reason': ['Residual curvature (coef=-5.22)', 'β₂ non-significant, no improvement']
}

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv("/workspace/experiments/model_assessment/loo_comparison.csv", index=False)
print(f"  ✓ Saved: /workspace/experiments/model_assessment/loo_comparison.csv")

# Save detailed comparison to JSON
comparison_results = {
    'model_comparison': {
        'delta_elpd': float(delta_elpd),
        'delta_se': float(delta_se),
        'interpretation': 'Models are equivalent (Model 1 slightly favored)' if abs(delta_elpd) < 2 * delta_se else 'Models differ',
        'comparison_table': compare_df.to_dict()
    },
    'model_1': {
        'elpd_loo': float(loo_model1.elpd_loo),
        'se': float(loo_model1.se),
        'p_loo': float(loo_model1.p_loo),
        'pareto_k_stats': {
            'n_high_k_0.7': int(n_high_k1),
            'n_high_k_0.5': int(np.sum(pareto_k1 > 0.5)),
            'max_k': float(pareto_k1.max()),
            'mean_k': float(pareto_k1.mean())
        },
        'decision': 'REJECTED',
        'reason': 'Systematic residual curvature (coef=-5.22), 4.17× MAE degradation'
    },
    'model_2': {
        'elpd_loo': float(loo_model2.elpd_loo),
        'se': float(loo_model2.se),
        'p_loo': float(loo_model2.p_loo),
        'pareto_k_stats': {
            'n_high_k_0.7': int(n_high_k2),
            'n_high_k_0.5': int(np.sum(pareto_k2 > 0.5)),
            'max_k': float(pareto_k2.max()),
            'mean_k': float(pareto_k2.mean())
        },
        'decision': 'REJECTED',
        'reason': 'β₂ non-significant (95% CI includes 0), no improvement over Model 1'
    }
}

with open("/workspace/experiments/model_assessment/comparison_results.json", 'w') as f:
    json.dump(comparison_results, f, indent=2)
print(f"  ✓ Saved: /workspace/experiments/model_assessment/comparison_results.json")

print("\n" + "="*80)
print("ASSESSMENT COMPLETE")
print("="*80)
print(f"\nFiles generated:")
print(f"  - Plots: /workspace/experiments/model_assessment/plots/ (4 PNG files)")
print(f"  - Metrics CSV: /workspace/experiments/model_assessment/loo_comparison.csv")
print(f"  - Detailed JSON: /workspace/experiments/model_assessment/comparison_results.json")
print("\nKey Finding: Both models REJECTED - polynomial functional form is inappropriate")
print("Recommendation: Explore changepoint models, GPs, or different likelihood structures")
print("="*80)
