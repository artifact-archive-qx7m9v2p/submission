"""
Comprehensive Model Comparison: Experiment 1 vs Experiment 3
Streamlined version focusing on LOO, WAIC, and visualizations
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

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_DIR = Path("/workspace")
EXP1_DIR = BASE_DIR / "experiments/experiment_1"
EXP3_DIR = BASE_DIR / "experiments/experiment_3"
COMP_DIR = BASE_DIR / "experiments/model_comparison"
DATA_PATH = BASE_DIR / "data/data.csv"

print("=" * 80)
print("BAYESIAN MODEL COMPARISON: Hierarchical vs Beta-Binomial")
print("=" * 80)

# Load data
data = pd.read_csv(DATA_PATH)
n_groups = len(data)
n_obs = data['n'].values
r_obs = data['r'].values
p_obs = r_obs / n_obs

print(f"\nData: {n_groups} groups, N={n_obs.sum()} trials, r={r_obs.sum()} successes")

# Load InferenceData
print("\nLoading models...")
idata_exp1 = az.from_netcdf(EXP1_DIR / "posterior_inference/diagnostics/posterior_inference.netcdf")
idata_exp3 = az.from_netcdf(EXP3_DIR / "posterior_inference/diagnostics/posterior_inference.netcdf")

# ============================================================================
# LOO CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("1. LOO CROSS-VALIDATION COMPARISON")
print("=" * 80)

loo_exp1 = az.loo(idata_exp1, pointwise=True)
loo_exp3 = az.loo(idata_exp3, pointwise=True)

print("\nEXP 1 (Hierarchical Binomial):")
print(f"  ELPD_loo: {loo_exp1.elpd_loo:.2f} ± {loo_exp1.se:.2f}")
print(f"  p_loo: {loo_exp1.p_loo:.2f} (effective parameters)")
k_exp1 = loo_exp1.pareto_k.values
n_bad_exp1 = np.sum(k_exp1 > 0.7)
n_verybad_exp1 = np.sum(k_exp1 > 1.0)
print(f"  Pareto k > 0.7: {n_bad_exp1}/{n_groups} groups ({100*n_bad_exp1/n_groups:.0f}%)")
print(f"  Pareto k > 1.0: {n_verybad_exp1}/{n_groups} groups")
print(f"  Max Pareto k: {np.max(k_exp1):.3f}")
print(f"  LOO Status: {'UNRELIABLE' if n_bad_exp1 > 2 else 'RELIABLE'}")

print("\nEXP 3 (Beta-Binomial):")
print(f"  ELPD_loo: {loo_exp3.elpd_loo:.2f} ± {loo_exp3.se:.2f}")
print(f"  p_loo: {loo_exp3.p_loo:.2f} (effective parameters)")
k_exp3 = loo_exp3.pareto_k.values
n_bad_exp3 = np.sum(k_exp3 > 0.7)
n_verybad_exp3 = np.sum(k_exp3 > 1.0)
print(f"  Pareto k > 0.7: {n_bad_exp3}/{n_groups} groups ({100*n_bad_exp3/n_groups:.0f}%)")
print(f"  Pareto k > 1.0: {n_verybad_exp3}/{n_groups} groups")
print(f"  Max Pareto k: {np.max(k_exp3):.3f}")
print(f"  LOO Status: {'UNRELIABLE' if n_bad_exp3 > 2 else 'RELIABLE'}")

# Comparison
delta_elpd = loo_exp3.elpd_loo - loo_exp1.elpd_loo
se_diff = np.sqrt(loo_exp1.se**2 + loo_exp3.se**2)

print("\nCOMPARISON:")
print(f"  ΔELPD (Exp3 - Exp1): {delta_elpd:.2f} ± {se_diff:.2f}")
print(f"  Magnitude: {abs(delta_elpd) / se_diff:.2f} × SE")

if abs(delta_elpd) < 2 * se_diff:
    decision = "EQUIVALENT"
    print(f"  Decision: Models EQUIVALENT (|Δ| < 2×SE)")
    print(f"  → Apply parsimony principle: Choose simpler model")
elif abs(delta_elpd) > 4 * se_diff:
    winner = "Exp 3" if delta_elpd > 0 else "Exp 1"
    decision = f"{winner} STRONGLY PREFERRED"
    print(f"  Decision: {winner} STRONGLY PREFERRED (|Δ| > 4×SE)")
else:
    winner = "Exp 3" if delta_elpd > 0 else "Exp 1"
    decision = f"{winner} WEAKLY PREFERRED"
    print(f"  Decision: {winner} WEAKLY PREFERRED (2×SE < |Δ| < 4×SE)")

# Formal comparison
print("\nFormal comparison (az.compare):")
compare_df = az.compare({"Exp1_Hierarchical": idata_exp1, "Exp3_BetaBinomial": idata_exp3})
print(compare_df[['rank', 'elpd_loo', 'p_loo', 'weight', 'se', 'dse', 'warning']])

# ============================================================================
# WAIC COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("2. WAIC COMPARISON (Alternative to LOO)")
print("=" * 80)

waic_exp1 = az.waic(idata_exp1, pointwise=True)
waic_exp3 = az.waic(idata_exp3, pointwise=True)

print("\nEXP 1 (Hierarchical):")
print(f"  ELPD_waic: {waic_exp1.elpd_waic:.2f} ± {waic_exp1.se:.2f}")
print(f"  p_waic: {waic_exp1.p_waic:.2f}")
print(f"  Warning: {waic_exp1.warning}")

print("\nEXP 3 (Beta-Binomial):")
print(f"  ELPD_waic: {waic_exp3.elpd_waic:.2f} ± {waic_exp3.se:.2f}")
print(f"  p_waic: {waic_exp3.p_waic:.2f}")
print(f"  Warning: {waic_exp3.warning}")

delta_waic = waic_exp3.elpd_waic - waic_exp1.elpd_waic
se_diff_waic = np.sqrt(waic_exp1.se**2 + waic_exp3.se**2)

print("\nCOMPARISON:")
print(f"  ΔELPD_waic (Exp3 - Exp1): {delta_waic:.2f} ± {se_diff_waic:.2f}")
print(f"  Note: WAIC may be more robust given Exp1's high Pareto k values")

# ============================================================================
# SAVE COMPARISON METRICS
# ============================================================================

comparison_data = {
    'Metric': [
        'Parameters', 'Sampling_time_sec',
        'ELPD_loo', 'SE_loo', 'p_loo',
        'Pareto_k_max', 'Pareto_k_bad_count', 'LOO_reliable',
        'ELPD_waic', 'SE_waic', 'p_waic', 'WAIC_warning'
    ],
    'Exp1_Hierarchical': [
        14, 90,
        float(loo_exp1.elpd_loo), float(loo_exp1.se), float(loo_exp1.p_loo),
        float(np.max(k_exp1)), int(n_bad_exp1), 'NO' if n_bad_exp1 > 2 else 'YES',
        float(waic_exp1.elpd_waic), float(waic_exp1.se), float(waic_exp1.p_waic), str(waic_exp1.warning)
    ],
    'Exp3_BetaBinomial': [
        2, 6,
        float(loo_exp3.elpd_loo), float(loo_exp3.se), float(loo_exp3.p_loo),
        float(np.max(k_exp3)), int(n_bad_exp3), 'NO' if n_bad_exp3 > 2 else 'YES',
        float(waic_exp3.elpd_waic), float(waic_exp3.se), float(waic_exp3.p_waic), str(waic_exp3.warning)
    ]
}

comparison_df_save = pd.DataFrame(comparison_data)
comparison_df_save.to_csv(COMP_DIR / "diagnostics/comparison_metrics.csv", index=False)

# LOO comparison table
loo_comparison_data = {
    'group': range(1, n_groups + 1),
    'n': n_obs,
    'r_obs': r_obs,
    'pareto_k_exp1': k_exp1,
    'pareto_k_exp3': k_exp3,
    'loo_i_exp1': loo_exp1.loo_i.values,
    'loo_i_exp3': loo_exp3.loo_i.values,
}
loo_df = pd.DataFrame(loo_comparison_data)
loo_df.to_csv(COMP_DIR / "diagnostics/loo_comparison_table.csv", index=False)

print("\n" + "=" * 80)
print("3. CREATING VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# VISUALIZATION 1: Comprehensive 4-Panel Comparison
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: ELPD Comparison
ax = axes[0, 0]
models = ['Exp1\nHierarchical', 'Exp3\nBeta-Binomial']
elpd_vals = [loo_exp1.elpd_loo, loo_exp3.elpd_loo]
se_vals = [loo_exp1.se, loo_exp3.se]
colors = ['#e74c3c', '#2ecc71']

bars = ax.bar(models, elpd_vals, yerr=se_vals, capsize=10,
              color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('ELPD LOO', fontsize=13, fontweight='bold')
ax.set_title('A. Predictive Performance (LOO-CV)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for i, (val, se) in enumerate(zip(elpd_vals, se_vals)):
    ax.text(i, val + se + 1, f'{val:.1f}±{se:.1f}',
           ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.text(0.5, min(elpd_vals) - 3,
       f'Δ = {delta_elpd:.1f}±{se_diff:.1f}\n({abs(delta_elpd)/se_diff:.1f}×SE)',
       ha='center', fontsize=11,
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# Panel B: Pareto k Comparison (KEY PLOT)
ax = axes[0, 1]
x = np.arange(n_groups)
width = 0.35

bars1 = ax.bar(x - width/2, k_exp1, width, label='Exp1 (Hierarchical)',
               color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, k_exp3, width, label='Exp3 (Beta-Binomial)',
               color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.axhline(0.7, color='orange', linestyle='--', linewidth=2, label='k=0.7 (threshold)', zorder=10)
ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='k=1.0 (very bad)', zorder=10)
ax.set_xlabel('Group', fontsize=13, fontweight='bold')
ax.set_ylabel('Pareto k', fontsize=13, fontweight='bold')
ax.set_title('B. LOO Reliability (Pareto k Diagnostics)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{i+1}' for i in range(n_groups)])
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

ax.text(0.02, 0.98, f'Exp1: {n_bad_exp1}/12 bad k\nExp3: {n_bad_exp3}/12 bad k',
       transform=ax.transAxes, va='top', fontsize=11,
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

# Panel C: Model Complexity
ax = axes[1, 0]
metrics = ['Parameters', 'Sampling\nTime (s)', 'Effective\nParameters\n(p_loo)']
exp1_vals = [14, 90, float(loo_exp1.p_loo)]
exp3_vals = [2, 6, float(loo_exp3.p_loo)]

x_pos = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x_pos - width/2, exp1_vals, width, label='Exp1',
              color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x_pos + width/2, exp3_vals, width, label='Exp3',
              color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Value', fontsize=13, fontweight='bold')
ax.set_title('C. Model Complexity', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics)
ax.legend(loc='upper left', fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_yscale('log')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height*1.1,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel D: Point-wise LOO comparison
ax = axes[1, 1]
loo_i_exp1 = loo_exp1.loo_i.values
loo_i_exp3 = loo_exp3.loo_i.values

# Scatter plot with color by bad k
colors_scatter = ['red' if k > 0.7 else 'green' for k in k_exp1]
sizes = [100 + 50*(k if k < 2 else 2) for k in k_exp1]  # Size by Pareto k

ax.scatter(loo_i_exp1, loo_i_exp3, c=colors_scatter, s=sizes, alpha=0.6,
          edgecolors='black', linewidths=1.5)

# Add diagonal line
min_val = min(np.min(loo_i_exp1), np.min(loo_i_exp3))
max_val = max(np.max(loo_i_exp1), np.max(loo_i_exp3))
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='Equal LOO')

ax.set_xlabel('Exp1: LOO pointwise', fontsize=13, fontweight='bold')
ax.set_ylabel('Exp3: LOO pointwise', fontsize=13, fontweight='bold')
ax.set_title('D. Group-Level LOO Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10)

# Add annotation for bad k groups
ax.text(0.98, 0.02, 'Red = Exp1 has bad k\nSize = Pareto k magnitude',
       transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

plt.tight_layout()
plt.savefig(COMP_DIR / "plots/comprehensive_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Comprehensive comparison plot saved")

# ============================================================================
# VISUALIZATION 2: Spider Plot (Trade-offs)
# ============================================================================

categories = ['LOO\nReliability', 'Simplicity', 'Computational\nSpeed',
              'ELPD\n(Predictive)', 'Parsimony\n(p_loo)']

# Normalize scores to 0-10 scale (higher is better)
exp1_scores = [
    10 * (1 - n_bad_exp1 / n_groups),  # LOO reliability (fewer bad k = better)
    10 * (2 / 14),  # Simplicity (fewer params = better)
    10 * (6 / 90),  # Speed (less time = better)
    10 * (1 + (loo_exp1.elpd_loo - min(loo_exp1.elpd_loo, loo_exp3.elpd_loo)) /
          abs(loo_exp1.elpd_loo - loo_exp3.elpd_loo + 0.001)),  # Relative ELPD
    10 * (2 / max(loo_exp1.p_loo, loo_exp3.p_loo)),  # Lower p_loo = more parsimonious
]

exp3_scores = [
    10 * (1 - n_bad_exp3 / n_groups),
    10 * (2 / 2),
    10 * (6 / 6),
    10 * (1 + (loo_exp3.elpd_loo - min(loo_exp1.elpd_loo, loo_exp3.elpd_loo)) /
          abs(loo_exp1.elpd_loo - loo_exp3.elpd_loo + 0.001)),
    10 * (loo_exp3.p_loo / max(loo_exp1.p_loo, loo_exp3.p_loo)),
]

N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

exp1_scores += exp1_scores[:1]
exp3_scores += exp3_scores[:1]

ax.plot(angles, exp1_scores, 'o-', linewidth=3, color='#e74c3c',
       label='Exp1: Hierarchical', markersize=10)
ax.fill(angles, exp1_scores, alpha=0.2, color='#e74c3c')

ax.plot(angles, exp3_scores, 'o-', linewidth=3, color='#2ecc71',
       label='Exp3: Beta-Binomial', markersize=10)
ax.fill(angles, exp3_scores, alpha=0.2, color='#2ecc71')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=13, fontweight='bold')
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=11)
ax.grid(True, linewidth=1.5, alpha=0.5)

plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=13, frameon=True)
plt.title('Model Trade-offs: Multi-Criteria Assessment\n(Higher = Better)',
         fontsize=15, fontweight='bold', pad=30)

plt.savefig(COMP_DIR / "plots/model_trade_offs_spider.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Spider plot saved")

# ============================================================================
# VISUALIZATION 3: Detailed Pareto k Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(n_groups)
width = 0.35

# Color bars by threshold
colors_exp1 = ['darkred' if k > 1.0 else 'red' if k > 0.7 else 'orange' if k > 0.5 else 'green'
              for k in k_exp1]
colors_exp3 = ['darkred' if k > 1.0 else 'red' if k > 0.7 else 'orange' if k > 0.5 else 'green'
              for k in k_exp3]

for i in range(n_groups):
    ax.bar(x[i] - width/2, k_exp1[i], width, color=colors_exp1[i],
          alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar(x[i] + width/2, k_exp3[i], width, color=colors_exp3[i],
          alpha=0.8, edgecolor='black', linewidth=1.5)

# Add sample sizes as text
for i in range(n_groups):
    ax.text(i, -0.15, f'n={n_obs[i]}', ha='center', fontsize=8, rotation=0)

ax.axhline(0.5, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='k=0.5 (ok)')
ax.axhline(0.7, color='red', linestyle='--', linewidth=2, alpha=0.7, label='k=0.7 (bad)')
ax.axhline(1.0, color='darkred', linestyle='-', linewidth=2, alpha=0.7, label='k=1.0 (very bad)')

ax.set_xlabel('Group (with sample size)', fontsize=13, fontweight='bold')
ax.set_ylabel('Pareto k', fontsize=13, fontweight='bold')
ax.set_title('Detailed LOO Reliability Comparison: Pareto k by Group\n' +
            f'Exp1 (left bars): {n_bad_exp1}/12 unreliable | Exp3 (right bars): {n_bad_exp3}/12 unreliable',
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{i+1}' for i in range(n_groups)])
ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(-0.2, max(np.max(k_exp1), np.max(k_exp3)) * 1.1)

# Add legend for bar colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='Good (k < 0.5)'),
    Patch(facecolor='orange', label='Ok (0.5 ≤ k < 0.7)'),
    Patch(facecolor='red', label='Bad (0.7 ≤ k < 1.0)'),
    Patch(facecolor='darkred', label='Very Bad (k ≥ 1.0)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, frameon=True)

plt.tight_layout()
plt.savefig(COMP_DIR / "plots/pareto_k_detailed_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Detailed Pareto k plot saved")

# ============================================================================
# SAVE SUMMARY JSON
# ============================================================================

summary = {
    'comparison_date': '2025-10-30',
    'experiment_1': {
        'model': 'Hierarchical Binomial (Logit-Normal)',
        'parameters': 14,
        'sampling_time_sec': 90,
        'loo': {
            'elpd': float(loo_exp1.elpd_loo),
            'se': float(loo_exp1.se),
            'p_loo': float(loo_exp1.p_loo),
            'pareto_k_max': float(np.max(k_exp1)),
            'pareto_k_bad_count': int(n_bad_exp1),
            'reliability': 'UNRELIABLE' if n_bad_exp1 > 2 else 'RELIABLE'
        },
        'waic': {
            'elpd': float(waic_exp1.elpd_waic),
            'se': float(waic_exp1.se),
            'p_waic': float(waic_exp1.p_waic),
            'warning': str(waic_exp1.warning)
        },
        'status': 'CONDITIONAL ACCEPT',
        'strengths': [
            'Group-specific inference',
            'Estimates heterogeneity (tau)',
            'Hierarchical shrinkage',
            'Slightly better ELPD (unreliable estimate)'
        ],
        'weaknesses': [
            '10/12 groups have unreliable LOO',
            'Complex (14 parameters)',
            'Slow sampling (90 seconds)',
            'Sensitive to extreme groups'
        ]
    },
    'experiment_3': {
        'model': 'Beta-Binomial (Population-level)',
        'parameters': 2,
        'sampling_time_sec': 6,
        'loo': {
            'elpd': float(loo_exp3.elpd_loo),
            'se': float(loo_exp3.se),
            'p_loo': float(loo_exp3.p_loo),
            'pareto_k_max': float(np.max(k_exp3)),
            'pareto_k_bad_count': int(n_bad_exp3),
            'reliability': 'UNRELIABLE' if n_bad_exp3 > 2 else 'RELIABLE'
        },
        'waic': {
            'elpd': float(waic_exp3.elpd_waic),
            'se': float(waic_exp3.se),
            'p_waic': float(waic_exp3.p_waic),
            'warning': str(waic_exp3.warning)
        },
        'status': 'ACCEPT',
        'strengths': [
            'Perfect LOO reliability (0/12 bad k)',
            'Very simple (2 parameters)',
            'Fast sampling (6 seconds)',
            'Easy interpretation (probability scale)'
        ],
        'weaknesses': [
            'No group-specific estimates',
            'Cannot quantify between-group variation directly',
            'Slightly worse ELPD (but reliable estimate)'
        ]
    },
    'comparison': {
        'loo_difference': {
            'delta_elpd': float(delta_elpd),
            'se': float(se_diff),
            'magnitude_in_se': float(abs(delta_elpd) / se_diff),
            'decision': decision
        },
        'waic_difference': {
            'delta_elpd': float(delta_waic),
            'se': float(se_diff_waic)
        },
        'key_insight': 'Models have equivalent predictive performance, but Exp3 has dramatically superior LOO reliability',
        'parsimony_winner': 'Exp3 (2 vs 14 parameters)',
        'speed_winner': 'Exp3 (6 vs 90 seconds)',
        'reliability_winner': 'Exp3 (0 vs 10 bad k)',
        'elpd_winner': 'Exp1 (by 1.5 ELPD, but unreliable)',
        'overall_recommendation': 'Exp3 for most purposes; Exp1 only if group-specific inference essential'
    }
}

with open(COMP_DIR / "diagnostics/comparison_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("  ✓ Summary JSON saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

print("\nKEY FINDINGS:")
print(f"  1. LOO Comparison: ΔELPD = {delta_elpd:.2f} ± {se_diff:.2f} ({abs(delta_elpd)/se_diff:.1f}×SE)")
print(f"     → {decision}")
print(f"  2. LOO Reliability: Exp1 {n_bad_exp1}/12 bad k vs Exp3 {n_bad_exp3}/12 bad k")
print(f"     → Exp3 DRAMATICALLY SUPERIOR")
print(f"  3. Parsimony: Exp3 7× simpler (2 vs 14 parameters)")
print(f"  4. Speed: Exp3 15× faster (6 vs 90 seconds)")
print(f"  5. Interpretability: Exp3 easier (probability vs logit scale)")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if abs(delta_elpd) < 2 * se_diff and n_bad_exp3 == 0 and n_bad_exp1 > 5:
    print("\n  CHOOSE EXP3 (Beta-Binomial)")
    print("\n  Justification:")
    print("    ✓ Equivalent predictive performance (ELPD within 2×SE)")
    print("    ✓ Perfect LOO reliability (0/12 bad k vs 10/12 bad k)")
    print("    ✓ 7× simpler (2 vs 14 parameters)")
    print("    ✓ 15× faster (6 vs 90 seconds)")
    print("    ✓ Easier interpretation")
    print("\n  When to use Exp1 instead:")
    print("    • Need group-specific rate estimates")
    print("    • Want to understand between-group heterogeneity")
    print("    • Willing to accept LOO limitations")

print("\nOUTPUT FILES:")
print(f"  • {COMP_DIR}/plots/comprehensive_comparison.png")
print(f"  • {COMP_DIR}/plots/model_trade_offs_spider.png")
print(f"  • {COMP_DIR}/plots/pareto_k_detailed_comparison.png")
print(f"  • {COMP_DIR}/diagnostics/comparison_metrics.csv")
print(f"  • {COMP_DIR}/diagnostics/loo_comparison_table.csv")
print(f"  • {COMP_DIR}/diagnostics/comparison_summary.json")

print("\n" + "=" * 80)
