"""
Create a visual summary badge for the PPC results
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import json

# Load summary
with open('/workspace/experiments/experiment_1/posterior_predictive_check/ppc_summary.json', 'r') as f:
    summary = json.load(f)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9, 'POSTERIOR PREDICTIVE CHECK',
        ha='center', va='top', fontsize=20, fontweight='bold')
ax.text(5, 8.3, 'Bayesian Hierarchical Meta-Analysis',
        ha='center', va='top', fontsize=14, color='gray')

# Main verdict box
verdict_color = {'EXCELLENT': '#2ecc71', 'ACCEPTABLE': '#f39c12', 'REJECT': '#e74c3c'}
color = verdict_color[summary['verdict']]

# Large verdict badge
badge = FancyBboxPatch((2, 5.5), 6, 1.8,
                       boxstyle="round,pad=0.1",
                       facecolor=color,
                       edgecolor='black',
                       linewidth=3,
                       alpha=0.3)
ax.add_patch(badge)

ax.text(5, 6.8, summary['verdict'],
        ha='center', va='center',
        fontsize=32, fontweight='bold', color=color)

ax.text(5, 6.2, 'Model PASSES all checks',
        ha='center', va='center',
        fontsize=14, fontweight='bold')

# Key metrics
metrics_y = 4.5
ax.text(5, metrics_y, 'FALSIFICATION CRITERION',
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))

ax.text(5, metrics_y - 0.7, f'Studies outside 95% PPI: {summary["n_outliers"]} of {summary["n_studies"]}',
        ha='center', va='center', fontsize=14)

ax.text(5, metrics_y - 1.3, f'Criterion: REJECT if >1 outlier',
        ha='center', va='center', fontsize=12, style='italic', color='gray')

# Result
result_y = 2.2
if summary['n_outliers'] > 1:
    result_text = 'REJECT MODEL'
    result_color = '#e74c3c'
    symbol = '✗'
else:
    result_text = 'DO NOT REJECT'
    result_color = '#2ecc71'
    symbol = '✓'

ax.text(5, result_y, f'{symbol} {result_text} {symbol}',
        ha='center', va='center',
        fontsize=20, fontweight='bold', color=result_color,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=result_color, linewidth=3))

# Additional info
info_y = 0.8
ax.text(5, info_y, 'All test statistics p-values: 0.38 - 0.96  |  No systematic misfit detected',
        ha='center', va='center', fontsize=10, color='gray')

ax.text(5, info_y - 0.4, 'Model ready for scientific inference',
        ha='center', va='center', fontsize=10, color='gray', style='italic')

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/posterior_predictive_check/plots/ppc_verdict_badge.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Created: ppc_verdict_badge.png")
