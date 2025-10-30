"""
LOO-CV Model Comparison: Model 1 (Log) vs Model 2 (Change-Point)

Compare predictive performance using Pareto-smoothed importance sampling
leave-one-out cross-validation (LOO-CV).
"""

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

print("="*70)
print("LOO-CV MODEL COMPARISON")
print("="*70)

# Load both models
print("\nLoading models...")

idata1_path = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
idata2_path = Path("/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf")

idata1 = az.from_netcdf(idata1_path)
idata2 = az.from_netcdf(idata2_path)

print(f"✓ Model 1 (Log): {idata1_path}")
print(f"  Groups: {list(idata1.groups())}")
print(f"  Posterior shape: {idata1.posterior.dims}")

print(f"\n✓ Model 2 (Change-Point): {idata2_path}")
print(f"  Groups: {list(idata2.groups())}")
print(f"  Posterior shape: {idata2.posterior.dims}")

# Compute LOO for both models
print("\n" + "="*70)
print("COMPUTING LOO-CV")
print("="*70)

print("\nModel 1 (Logarithmic)...")
loo1 = az.loo(idata1, pointwise=True)
print(f"  ELPD_LOO = {loo1.elpd_loo:.2f} ± {loo1.se:.2f}")
print(f"  p_loo = {loo1.p_loo:.2f}")
print(f"  LOO-IC = {-2 * loo1.elpd_loo:.2f}")

print("\nModel 2 (Change-Point)...")
loo2 = az.loo(idata2, pointwise=True)
print(f"  ELPD_LOO = {loo2.elpd_loo:.2f} ± {loo2.se:.2f}")
print(f"  p_loo = {loo2.p_loo:.2f}")
print(f"  LOO-IC = {-2 * loo2.elpd_loo:.2f}")

# Compare models
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

comparison = az.compare({'Model1_Log': idata1, 'Model2_ChangePoint': idata2}, ic='loo')
print("\n", comparison)

# Interpret results
delta_loo = comparison.loc['Model2_ChangePoint', 'elpd_loo'] - comparison.loc['Model1_Log', 'elpd_loo']
se_diff = comparison.loc['Model2_ChangePoint', 'se']

# Get the better model
if delta_loo > 0:
    better_model = "Model 2 (Change-Point)"
    worse_model = "Model 1 (Log)"
else:
    better_model = "Model 1 (Log)"
    worse_model = "Model 2 (Change-Point)"
    delta_loo = -delta_loo  # Make positive for clarity

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

print(f"\nΔELPD_LOO = {delta_loo:.2f} ± {se_diff:.2f}")
print(f"Better model: {better_model}")

# Decision criteria
if abs(delta_loo) < 2:
    decision = "COMPARABLE"
    strength = "Models are essentially equivalent"
    recommendation = "Choose simpler model by parsimony principle"
elif abs(delta_loo) < 6:
    decision = "WEAK PREFERENCE"
    strength = f"{better_model} is slightly better"
    recommendation = "Prefer simpler model unless strong theoretical reason"
else:
    decision = "STRONG PREFERENCE"
    strength = f"{better_model} is clearly better"
    recommendation = f"Use {better_model} for inference"

print(f"\nDecision: {decision}")
print(f"Strength: {strength}")
print(f"Recommendation: {recommendation}")

# Model weights
weight1 = comparison.loc['Model1_Log', 'weight']
weight2 = comparison.loc['Model2_ChangePoint', 'weight']

print(f"\n\nModel Weights (Bayesian Model Averaging):")
print(f"  Model 1 (Log): {weight1:.3f}")
print(f"  Model 2 (Change-Point): {weight2:.3f}")

# Check for problematic Pareto k values
print("\n" + "="*70)
print("PARETO-K DIAGNOSTIC")
print("="*70)

def check_pareto_k(loo, model_name):
    k_values = loo.pareto_k.values
    n_high = np.sum(k_values > 0.7)
    n_medium = np.sum((k_values > 0.5) & (k_values <= 0.7))
    n_good = np.sum(k_values <= 0.5)

    print(f"\n{model_name}:")
    print(f"  Good (k ≤ 0.5): {n_good}/{len(k_values)}")
    print(f"  Medium (0.5 < k ≤ 0.7): {n_medium}/{len(k_values)}")
    print(f"  High (k > 0.7): {n_high}/{len(k_values)}")

    if n_high > 0:
        print(f"  ⚠ WARNING: {n_high} observations with high Pareto k")
        print(f"     LOO may be unreliable for these points")
    else:
        print(f"  ✓ All Pareto k values acceptable")

check_pareto_k(loo1, "Model 1 (Log)")
check_pareto_k(loo2, "Model 2 (Change-Point)")

# Visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

output_dir = Path("/workspace/experiments/experiment_2/posterior_inference/plots")

# Plot 1: LOO comparison
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_compare(comparison, insample_dev=False, ax=ax)
ax.set_title('LOO-CV Model Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
comparison_plot_path = output_dir / "loo_comparison.png"
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {comparison_plot_path}")

# Plot 2: Pareto k values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

az.plot_khat(loo1, ax=axes[0], show_bins=True)
axes[0].set_title('Model 1 (Log): Pareto k Diagnostic', fontsize=12, fontweight='bold')

az.plot_khat(loo2, ax=axes[1], show_bins=True)
axes[1].set_title('Model 2 (Change-Point): Pareto k Diagnostic', fontsize=12, fontweight='bold')

plt.tight_layout()
pareto_plot_path = output_dir / "pareto_k_diagnostic.png"
plt.savefig(pareto_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {pareto_plot_path}")

# Plot 3: Pointwise LOO comparison
fig, ax = plt.subplots(figsize=(10, 6))

elpd_diff = loo2.pareto_k.values - loo1.pareto_k.values  # Actually get ELPD diff
# Get actual pointwise differences
if hasattr(comparison, 'elpd_data'):
    elpd1_i = loo1.elpd_i.values
    elpd2_i = loo2.elpd_i.values
    elpd_diff = elpd2_i - elpd1_i

    ax.scatter(range(len(elpd_diff)), elpd_diff, alpha=0.6, s=80)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax.axhline(elpd_diff.mean(), color='green', linestyle='-', linewidth=2,
               label=f'Mean diff = {elpd_diff.mean():.2f}')
    ax.set_xlabel('Observation index', fontsize=12)
    ax.set_ylabel('ΔELPD (Model 2 - Model 1)', fontsize=12)
    ax.set_title('Pointwise ELPD Differences', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pointwise_plot_path = output_dir / "pointwise_loo_comparison.png"
    plt.savefig(pointwise_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {pointwise_plot_path}")

# Save comparison results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_path = Path("/workspace/experiments/experiment_2/posterior_inference/diagnostics/loo_comparison.csv")
comparison.to_csv(results_path)
print(f"✓ Saved comparison table: {results_path}")

print("\n" + "="*70)
print("LOO-CV COMPARISON COMPLETE")
print("="*70)
print(f"\nFinal Decision: {decision}")
print(f"\n{recommendation}")
