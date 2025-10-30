"""
LOO Comparison: Experiment 1 (Hierarchical) vs Experiment 2 (Complete Pooling)
===============================================================================

This is the critical test for model selection between nested models.
"""

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import os

print("="*70)
print("LOO COMPARISON: HIERARCHICAL vs COMPLETE POOLING")
print("="*70)

# Load both models
print("\nLoading inference data...")
idata_exp1 = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
idata_exp2 = az.from_netcdf('/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf')

print("✓ Experiment 1 (Hierarchical) loaded")
print("✓ Experiment 2 (Complete Pooling) loaded")

# Compute LOO for both models
print("\n" + "="*70)
print("COMPUTING LOO-CV")
print("="*70)

print("\nExperiment 1 (Hierarchical)...")
loo_exp1 = az.loo(idata_exp1, pointwise=True)
print(f"  ELPD_loo: {loo_exp1.elpd_loo:.2f}")
print(f"  SE: {loo_exp1.se:.2f}")
print(f"  p_loo: {loo_exp1.p_loo:.2f}")

print("\nExperiment 2 (Complete Pooling)...")
loo_exp2 = az.loo(idata_exp2, pointwise=True)
print(f"  ELPD_loo: {loo_exp2.elpd_loo:.2f}")
print(f"  SE: {loo_exp2.se:.2f}")
print(f"  p_loo: {loo_exp2.p_loo:.2f}")

# Compare models
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

comp = az.compare({
    'Exp1_Hierarchical': idata_exp1,
    'Exp2_CompletPooling': idata_exp2
}, ic='loo', scale='deviance')

print("\nComparison table:")
print(comp)

# Extract key metrics
# The best model has rank 0
best_model = comp.index[0]
delta_elpd = comp.loc['Exp2_CompletPooling', 'elpd_diff'] if 'Exp2_CompletPooling' in comp.index else -comp.loc['Exp1_Hierarchical', 'elpd_diff']
se_diff = comp.loc[comp.index[1], 'dse'] if len(comp) > 1 else 0

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

print(f"\nBest model (rank 0): {best_model}")
print(f"ΔLOO (difference from best): {abs(delta_elpd):.2f} ± {se_diff:.2f}")

# Decision rules
print("\n**DECISION RULES:**")
print(f"  |ΔLOO| = {abs(delta_elpd):.2f}")
print(f"  2 × SE = {2 * se_diff:.2f}")

if abs(delta_elpd) < 2 * se_diff:
    print(f"\n  ✓ |ΔLOO| < 2×SE: Models perform similarly")
    print(f"  → Apply PARSIMONY RULE: Prefer simpler model (Complete Pooling)")
    recommendation = "ACCEPT Complete Pooling (Exp 2)"
    rationale = "Models perform similarly, prefer simpler model by parsimony"
elif best_model == 'Exp2_CompletPooling' and abs(delta_elpd) > 2 * se_diff:
    print(f"\n  ✓ |ΔLOO| > 2×SE: Complete Pooling significantly better")
    recommendation = "ACCEPT Complete Pooling (Exp 2)"
    rationale = "Complete pooling has better predictive performance"
elif best_model == 'Exp1_Hierarchical' and abs(delta_elpd) > 2 * se_diff:
    print(f"\n  ✓ |ΔLOO| > 2×SE: Hierarchical significantly better")
    recommendation = "REJECT Complete Pooling, use Hierarchical (Exp 1)"
    rationale = "Hierarchical model has better predictive performance"
else:
    print(f"\n  Marginal difference")
    recommendation = "MARGINAL: Report both models"
    rationale = "Models have similar but distinguishable performance"

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print(f"\n{recommendation}")
print(f"\nRationale: {rationale}")

# Check Pareto k diagnostics
print("\n" + "="*70)
print("PARETO K DIAGNOSTICS")
print("="*70)

print("\nExperiment 1 (Hierarchical):")
k_exp1 = loo_exp1.pareto_k.values
print(f"  All k < 0.5: {np.all(k_exp1 < 0.5)}")
print(f"  Max k: {np.max(k_exp1):.3f}")
print(f"  k values: {k_exp1}")

print("\nExperiment 2 (Complete Pooling):")
k_exp2 = loo_exp2.pareto_k.values
print(f"  All k < 0.5: {np.all(k_exp2 < 0.5)}")
print(f"  Max k: {np.max(k_exp2):.3f}")
print(f"  k values: {k_exp2}")

if np.all(k_exp1 < 0.5) and np.all(k_exp2 < 0.5):
    print("\n✓ Both models: All Pareto k < 0.5 (LOO reliable)")
elif np.any(k_exp1 > 0.7) or np.any(k_exp2 > 0.7):
    print("\n⚠ Warning: Some Pareto k > 0.7 (LOO may be unreliable)")
else:
    print("\n⚠ Caution: Some Pareto k between 0.5-0.7")

# Pointwise comparison
print("\n" + "="*70)
print("POINTWISE COMPARISON")
print("="*70)

loo_exp1_i = loo_exp1.loo_i.values
loo_exp2_i = loo_exp2.loo_i.values
loo_diff_pointwise = loo_exp2_i - loo_exp1_i

print("\nStudy | LOO(Exp1) | LOO(Exp2) | Diff | Favors")
print("-" * 70)
for i in range(len(loo_diff_pointwise)):
    l1 = loo_exp1_i[i]
    l2 = loo_exp2_i[i]
    diff = loo_diff_pointwise[i]
    favors = "Exp2" if diff > 0 else "Exp1" if diff < 0 else "Equal"

    print(f"{i+1:5d} | {l1:9.2f} | {l2:9.2f} | {diff:4.2f} | {favors}")

print(f"\nSummary:")
print(f"  Studies favoring Exp1: {np.sum(loo_diff_pointwise < 0)}")
print(f"  Studies favoring Exp2: {np.sum(loo_diff_pointwise > 0)}")
print(f"  Mean pointwise diff: {np.mean(loo_diff_pointwise):.2f}")

# Visualizations
print("\n" + "="*70)
print("VISUALIZATIONS")
print("="*70)

os.makedirs('/workspace/experiments/experiment_2/posterior_predictive_check/plots', exist_ok=True)

# Plot 1: Comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# LOO comparison
ax = axes[0, 0]
az.plot_compare(comp, ax=ax, insample_dev=False)
ax.set_title('LOO Comparison', fontweight='bold', fontsize=13)

# Pareto k values - Exp 1
ax = axes[0, 1]
ax.scatter(range(1, len(k_exp1)+1), k_exp1, s=100, alpha=0.7, color='red', edgecolor='black')
ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, label='k=0.5 threshold')
ax.axhline(0.7, color='red', linestyle='--', linewidth=2, label='k=0.7 threshold')
ax.set_xlabel('Study', fontsize=11)
ax.set_ylabel('Pareto k', fontsize=11)
ax.set_title('Pareto k - Exp 1 (Hierarchical)', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Pareto k values - Exp 2
ax = axes[1, 0]
ax.scatter(range(1, len(k_exp2)+1), k_exp2, s=100, alpha=0.7, color='blue', edgecolor='black')
ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, label='k=0.5 threshold')
ax.axhline(0.7, color='red', linestyle='--', linewidth=2, label='k=0.7 threshold')
ax.set_xlabel('Study', fontsize=11)
ax.set_ylabel('Pareto k', fontsize=11)
ax.set_title('Pareto k - Exp 2 (Complete Pooling)', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(alpha=0.3)

# Pointwise LOO difference
ax = axes[1, 1]
colors = ['red' if d < 0 else 'blue' for d in loo_diff_pointwise]
ax.bar(range(1, len(loo_diff_pointwise)+1), loo_diff_pointwise,
       color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Study', fontsize=11)
ax.set_ylabel('LOO Difference (Exp2 - Exp1)', fontsize=11)
ax.set_title('Pointwise Predictive Performance', fontweight='bold', fontsize=13)
ax.grid(alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Favors Exp2'),
                   Patch(facecolor='red', alpha=0.7, label='Favors Exp1')]
ax.legend(handles=legend_elements)

plt.suptitle('LOO Cross-Validation Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/posterior_predictive_check/plots/loo_comparison.png',
            dpi=150, bbox_inches='tight')
print("\nSaved: plots/loo_comparison.png")
plt.close()

# Plot 2: LOO performance comparison with uncertainty
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Exp1\n(Hierarchical)', 'Exp2\n(Complete Pooling)']
elpd = [loo_exp1.elpd_loo, loo_exp2.elpd_loo]
se = [loo_exp1.se, loo_exp2.se]

x_pos = np.arange(len(models))
colors = ['red' if 'Hierarchical' in m else 'blue' for m in models]

ax.bar(x_pos, elpd, yerr=[2*s for s in se], alpha=0.7, color=colors,
       edgecolor='black', linewidth=2, capsize=10)

# Add values
for i, (e, s) in enumerate(zip(elpd, se)):
    ax.text(i, e + 2*s + 0.5, f'{e:.1f} ± {s:.1f}',
           ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Expected Log Pointwise Predictive Density', fontsize=12)
ax.set_title('LOO-CV: Predictive Performance Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=11)
ax.grid(alpha=0.3, axis='y')

# Add interpretation text
textstr = f'ΔLOO = {abs(delta_elpd):.2f} ± {se_diff:.2f}\n'
if abs(delta_elpd) < 2 * se_diff:
    textstr += 'Similar performance → Prefer simpler model'
    color = 'wheat'
elif best_model == 'Exp2_CompletPooling':
    textstr += 'Exp2 significantly better'
    color = 'lightblue'
else:
    textstr += 'Exp1 significantly better'
    color = 'lightcoral'

props = dict(boxstyle='round', facecolor=color, alpha=0.8)
ax.text(0.5, 0.05, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='center', bbox=props)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/posterior_predictive_check/plots/loo_performance.png',
            dpi=150, bbox_inches='tight')
print("Saved: plots/loo_performance.png")
plt.close()

print("\n" + "="*70)
print("LOO COMPARISON COMPLETE")
print("="*70)
print(f"\n**FINAL RECOMMENDATION: {recommendation}**")
print(f"\nRationale: {rationale}")
print(f"\nKey findings:")
print(f"  - Best model: {best_model}")
print(f"  - ΔLOO: {abs(delta_elpd):.2f} ± {se_diff:.2f}")
print(f"  - Pareto k warnings: {'Yes' if np.any(k_exp1 > 0.7) or np.any(k_exp2 > 0.7) else 'No'}")
print("\nFiles generated:")
print("  - plots/loo_comparison.png - Detailed comparison with Pareto k")
print("  - plots/loo_performance.png - Overall performance with uncertainty")
