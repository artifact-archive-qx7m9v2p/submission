"""Create sensitivity analysis visualization"""
import matplotlib.pyplot as plt
import numpy as np
import json

# Load results
with open('/workspace/experiments/experiment_1/model_critique/code/sensitivity_results.json', 'r') as f:
    results = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Prior sensitivity
ax1 = axes[0, 0]
prior_ess = results['prior_sensitivity']['prior_ess']
prior_ess_pct = results['prior_sensitivity']['prior_ess_pct']
ax1.bar(['Prior ESS'], [prior_ess_pct], color='lightgreen', alpha=0.7, edgecolor='black', linewidth=2)
ax1.axhline(5, color='red', linestyle='--', linewidth=2, label='Min threshold (5%)')
ax1.set_ylabel('% of Total Samples', fontsize=12)
ax1.set_ylim([0, 105])
ax1.set_title('Prior Sensitivity: Prior-Posterior Overlap', fontsize=13, fontweight='bold')
ax1.text(0, prior_ess_pct + 2, f'{prior_ess_pct:.1f}%\n({prior_ess:.0f} / 40,000)',
        ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.text(0, 50, 'GOOD\nData dominates\ninference', ha='center', va='center',
        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3, axis='y')

# Panel 2: Influential point
ax2 = axes[0, 1]
params = ['α', 'β', 'σ']
changes = [results['influential_point_x31_5']['alpha_change_pct'],
          results['influential_point_x31_5']['beta_change_pct'],
          results['influential_point_x31_5']['sigma_change_pct']]
colors = ['green' if abs(c) < 10 else 'orange' if abs(c) < 30 else 'red' for c in changes]
bars = ax2.bar(params, changes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axhline(0, color='black', linestyle='-', linewidth=1)
ax2.axhline(30, color='red', linestyle='--', linewidth=2, label='Rejection (+30%)')
ax2.axhline(-30, color='red', linestyle='--', linewidth=2, label='Rejection (-30%)')
ax2.set_ylabel('% Change', fontsize=12)
ax2.set_ylim([-40, 40])
ax2.set_title('Influential Point Test: Without x=31.5', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3, axis='y')
for bar, change in zip(bars, changes):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{change:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
            fontsize=11, fontweight='bold')
ax2.text(0.5, 0.95, f'Criterion: {results["influential_point_x31_5"]["falsification_criterion"]}',
        ha='center', va='top', transform=ax2.transAxes, fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Panel 3: Gap region uncertainty
ax3 = axes[1, 0]
regions = ['Dense\n(x ≤ 22.5)', 'Gap\n(23-29)']
widths = [results['gap_region']['dense_width'], results['gap_region']['gap_width']]
bars3 = ax3.bar(regions, widths, color=['lightblue', 'orange'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('95% CI Width', fontsize=12)
ax3.set_title('Gap Region Uncertainty', fontsize=13, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')
for bar, val in zip(bars3, widths):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ratio = results['gap_region']['ratio']
ax3.text(0.5, 0.95, f'Ratio: {ratio:.2f}x', ha='center', va='top',
        transform=ax3.transAxes, fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Panel 4: Extrapolation
ax4 = axes[1, 1]
extrap = results['extrapolation']
x_vals = [e['x'] for e in extrap]
y_vals = [e['mean'] for e in extrap]
ci_lower = [e['ci_lower'] for e in extrap]
ci_upper = [e['ci_upper'] for e in extrap]
yerr = [[y - l for y, l in zip(y_vals, ci_lower)],
        [u - y for y, u in zip(y_vals, ci_upper)]]
ax4.errorbar(x_vals, y_vals, yerr=yerr, fmt='o-', markersize=10,
            capsize=5, capthick=2, linewidth=2, color='red', ecolor='red',
            alpha=0.7, label='Extrapolations')
ax4.axvline(31.5, color='blue', linestyle='--', linewidth=2, label='Max observed x')
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('Predicted Y', fontsize=12)
ax4.set_title('Extrapolation Beyond x=31.5', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)
ax4.text(0.5, 0.05, 'NOTE: Assumes unbounded logarithmic growth',
        ha='center', va='bottom', transform=ax4.transAxes, fontsize=10,
        style='italic', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.suptitle('Sensitivity Analysis: Logarithmic Regression Model',
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/model_critique/plots/sensitivity_analysis.png',
           dpi=300, bbox_inches='tight')
print("Created: sensitivity_analysis.png")
plt.close()
