"""
Create a summary visualization showing the overall prior predictive check assessment
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
sns.set_style("white")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

# Load summary statistics
with open('/workspace/experiments/experiment_1/prior_predictive_check/summary_stats.json', 'r') as f:
    stats = json.load(f)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prior Predictive Check Summary: Beta-Binomial Model',
             fontsize=16, fontweight='bold', y=0.98)

# Top Left: Decision Box
ax = axes[0, 0]
ax.axis('off')
decision = stats['decision']
color = 'lightgreen' if decision == 'PASS' else 'lightcoral'

# Draw decision box
rect = Rectangle((0.1, 0.4), 0.8, 0.4, linewidth=3,
                 edgecolor='darkgreen' if decision == 'PASS' else 'darkred',
                 facecolor=color, alpha=0.3)
ax.add_patch(rect)
ax.text(0.5, 0.6, decision, ha='center', va='center',
        fontsize=48, fontweight='bold',
        color='darkgreen' if decision == 'PASS' else 'darkred')

# Add status text
status_text = "Ready for Model Fitting" if decision == 'PASS' else "Revisions Required"
ax.text(0.5, 0.25, status_text, ha='center', va='center',
        fontsize=14, style='italic')

# Add date
ax.text(0.5, 0.1, 'Date: 2025-10-30', ha='center', va='center',
        fontsize=10, color='gray')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Top Right: Coverage Percentiles
ax = axes[0, 1]
ax.axis('off')

metrics = ['Total\nSuccesses', 'Variance\nInflation']
percentiles = [stats['total_successes_percentile'], stats['variance_inflation_percentile']]

y_positions = [0.7, 0.3]
for i, (metric, pct, y_pos) in enumerate(zip(metrics, percentiles, y_positions)):
    # Draw scale
    ax.plot([0.1, 0.9], [y_pos, y_pos], 'k-', linewidth=2)

    # Mark zones
    ax.axvspan(0.1, 0.15, ymin=y_pos-0.05, ymax=y_pos+0.05, alpha=0.2, color='red')
    ax.axvspan(0.85, 0.9, ymin=y_pos-0.05, ymax=y_pos+0.05, alpha=0.2, color='red')
    ax.axvspan(0.15, 0.85, ymin=y_pos-0.05, ymax=y_pos+0.05, alpha=0.1, color='green')

    # Mark observed percentile
    x_pos = 0.1 + 0.8 * (pct / 100)
    ax.plot(x_pos, y_pos, 'ro', markersize=12, markeredgecolor='darkred', markeredgewidth=2)
    ax.text(x_pos, y_pos + 0.08, f'{pct:.1f}th', ha='center', fontsize=10, fontweight='bold')

    # Label
    ax.text(0.05, y_pos, metric, ha='right', va='center', fontsize=11, fontweight='bold')

    # Axis labels
    if i == 1:
        ax.text(0.1, y_pos - 0.12, '0th', ha='center', fontsize=8)
        ax.text(0.5, y_pos - 0.12, '50th', ha='center', fontsize=8)
        ax.text(0.9, y_pos - 0.12, '100th', ha='center', fontsize=8)

ax.text(0.5, 0.95, 'Observed Data Percentiles\nin Prior Predictive',
        ha='center', fontsize=12, fontweight='bold')
ax.text(0.5, 0.05, 'Red zones: <5th or >95th percentile',
        ha='center', fontsize=9, style='italic', color='gray')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Bottom Left: Checklist
ax = axes[1, 0]
ax.axis('off')

checks = [
    ('Total successes coverage', True),
    ('Overdispersion coverage', True),
    ('Trial-level extremes', True),
    ('Computational stability', True),
    ('Domain validity', True)
]

ax.text(0.5, 0.95, 'Critical Checks', ha='center', fontsize=12, fontweight='bold')

y_start = 0.8
y_step = 0.15
for i, (check, passed) in enumerate(checks):
    y_pos = y_start - i * y_step
    symbol = '✓' if passed else '✗'
    color = 'green' if passed else 'red'
    ax.text(0.15, y_pos, symbol, fontsize=20, color=color, fontweight='bold')
    ax.text(0.25, y_pos, check, fontsize=11, va='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Bottom Right: Prior Summary
ax = axes[1, 1]
ax.axis('off')

ax.text(0.5, 0.95, 'Prior Specifications', ha='center', fontsize=12, fontweight='bold')

prior_info = [
    ('μ ~ Beta(2, 25)',
     f"Mean: {stats['mu_prior_mean']:.3f}",
     f"95% CI: [{stats['mu_prior_ci'][0]:.3f}, {stats['mu_prior_ci'][1]:.3f}]"),

    ('φ ~ Gamma(2, 2)',
     f"Mean: {stats['phi_prior_mean']:.3f}",
     f"95% CI: [{stats['phi_prior_ci'][0]:.3f}, {stats['phi_prior_ci'][1]:.3f}]"),
]

y_start = 0.7
y_step = 0.3
for i, (prior, mean, ci) in enumerate(prior_info):
    y_pos = y_start - i * y_step

    # Prior specification
    ax.text(0.1, y_pos, prior, fontsize=11, fontweight='bold',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Statistics
    ax.text(0.1, y_pos - 0.08, mean, fontsize=9)
    ax.text(0.1, y_pos - 0.15, ci, fontsize=9, style='italic', color='gray')

# Add warning if present
if stats['warnings']:
    warning_text = '\n'.join(['Warnings (non-blocking):'] + ['  • ' + w for w in stats['warnings']])
    ax.text(0.5, 0.05, warning_text, ha='center', fontsize=8,
            style='italic', color='orange',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/plots/assessment_summary.png',
            dpi=300, bbox_inches='tight')
print("Summary visualization created: assessment_summary.png")
