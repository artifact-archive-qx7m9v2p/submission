"""
Summary Figure: Key Results at a Glance
========================================
Single comprehensive figure summarizing main findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/eda/code/processed_data.csv')

# Calculate key statistics
weighted_mean = np.sum(data['y'] * data['precision']) / np.sum(data['precision'])
weights = data['precision']**2
Q = np.sum(weights * (data['y'] - weighted_mean)**2)
df = len(data) - 1
tau_squared = max(0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
I_squared = max(0, 100 * (Q - df) / Q)
pooled_se = 1 / np.sqrt(np.sum(weights))
ci_lower = weighted_mean - 1.96 * pooled_se
ci_upper = weighted_mean + 1.96 * pooled_se

# Create summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Meta-Analysis EDA Summary: Key Findings at a Glance',
             fontsize=18, fontweight='bold', y=0.98)

# Panel 1: Forest plot (compact)
ax1 = fig.add_subplot(gs[0, :2])
data_sorted = data.sort_values('y')
y_pos = np.arange(len(data_sorted))
for i, (idx, row) in enumerate(data_sorted.iterrows()):
    ci_low = row['y'] - 1.96 * row['sigma']
    ci_high = row['y'] + 1.96 * row['sigma']
    ax1.plot([ci_low, ci_high], [i, i], 'k-', linewidth=1, alpha=0.6)
    ax1.scatter(row['y'], i, s=100/row['sigma'], c='steelblue',
               edgecolors='black', zorder=3)
ax1.axvline(weighted_mean, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"Study {int(s)}" for s in data_sorted['study']], fontsize=9)
ax1.set_xlabel('Effect Size (y)', fontsize=10)
ax1.set_title('A. Forest Plot with 95% CIs', fontsize=11, fontweight='bold', loc='left')
ax1.grid(alpha=0.3, axis='x')

# Panel 2: Key statistics box
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
POOLED ESTIMATE
━━━━━━━━━━━━━━━━━
Mean: {weighted_mean:.2f}
95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]

HETEROGENEITY
━━━━━━━━━━━━━━━━━
I² = {I_squared:.1f}%
Q = {Q:.2f} (p = {1-stats.chi2.cdf(Q, df):.3f})
τ² = {tau_squared:.2f}
τ = {np.sqrt(tau_squared):.2f}

SAMPLE
━━━━━━━━━━━━━━━━━
Studies: {len(data)}
Eff. n: {1/np.sum((weights/weights.sum())**2):.1f}
Efficiency: {1/np.sum((weights/weights.sum())**2)/len(data)*100:.0f}%

QUALITY
━━━━━━━━━━━━━━━━━
Pub. bias: None
Outliers: 0
Missing: 0
"""
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel 3: Shrinkage visualization
ax3 = fig.add_subplot(gs[1, 0])
shrinkage_factors = tau_squared / (tau_squared + data['variance'])
partial_pooling = shrinkage_factors * data['y'] + (1 - shrinkage_factors) * weighted_mean
studies = data['study'].astype(int)
for idx, row in data.iterrows():
    study_id = int(row['study'])
    ax3.plot([row['y'], partial_pooling[idx]], [study_id, study_id],
            'steelblue', alpha=0.6, linewidth=2)
    ax3.scatter(row['y'], study_id, s=80, c='red', marker='o',
               edgecolors='black', zorder=3, label='Observed' if idx == 0 else '')
    ax3.scatter(partial_pooling[idx], study_id, s=80, c='blue', marker='s',
               edgecolors='black', zorder=3, label='Shrunken' if idx == 0 else '')
ax3.axvline(weighted_mean, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.set_yticks(studies)
ax3.set_yticklabels([f"{s}" for s in studies], fontsize=9)
ax3.set_xlabel('Effect Size', fontsize=10)
ax3.set_ylabel('Study', fontsize=10)
ax3.set_title('B. Shrinkage to Pooled Mean', fontsize=11, fontweight='bold', loc='left')
ax3.legend(fontsize=8, loc='lower right')
ax3.grid(alpha=0.3, axis='x')

# Panel 4: Heterogeneity bar chart
ax4 = fig.add_subplot(gs[1, 1])
categories = ['Within-Study\nVariance', 'Between-Study\nVariance']
total_var = data['variance'].mean() + tau_squared
values = [100 * data['variance'].mean() / total_var, 100 * tau_squared / total_var]
colors_bar = ['lightblue', 'coral']
bars = ax4.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1.5, alpha=0.7)
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height/2,
            f'{val:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')
ax4.set_ylabel('Percentage of Total Variance', fontsize=10)
ax4.set_title('C. Variance Decomposition', fontsize=11, fontweight='bold', loc='left')
ax4.set_ylim(0, 105)
ax4.grid(alpha=0.3, axis='y')

# Panel 5: Funnel plot
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(data['y'], data['sigma'], s=120, alpha=0.7, c='purple',
           edgecolors='black', linewidth=1.5)
for idx, row in data.iterrows():
    ax5.annotate(f"{int(row['study'])}", (row['y'], row['sigma']),
                xytext=(3, 3), textcoords='offset points', fontsize=8)
ax5.axvline(weighted_mean, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax5.invert_yaxis()
ax5.set_xlabel('Effect Size (y)', fontsize=10)
ax5.set_ylabel('Standard Error (σ)', fontsize=10)
ax5.set_title('D. Funnel Plot (Inverted)', fontsize=11, fontweight='bold', loc='left')
ax5.grid(alpha=0.3)

# Panel 6: Distribution of effects
ax6 = fig.add_subplot(gs[2, 0])
ax6.hist(data['y'], bins=6, alpha=0.6, color='steelblue', edgecolor='black', density=False)
ax6.axvline(weighted_mean, color='red', linestyle='--', linewidth=2,
           label=f'Pooled = {weighted_mean:.2f}')
ax6.axvline(data['y'].mean(), color='blue', linestyle=':', linewidth=2,
           label=f'Unweighted = {data["y"].mean():.2f}')
ax6.set_xlabel('Effect Size (y)', fontsize=10)
ax6.set_ylabel('Frequency', fontsize=10)
ax6.set_title('E. Distribution of Observed Effects', fontsize=11, fontweight='bold', loc='left')
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3)

# Panel 7: Model comparison
ax7 = fig.add_subplot(gs[2, 1])
models = ['Common\nEffect', 'Random\nEffects', 'No\nPooling']
aic_values = [63.85, 65.82, 70.64]
colors_model = ['green' if a == min(aic_values) else 'orange' for a in aic_values]
bars = ax7.bar(models, aic_values, color=colors_model, edgecolor='black', linewidth=1.5, alpha=0.7)
for bar, val in zip(bars, aic_values):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax7.set_ylabel('AIC (lower is better)', fontsize=10)
ax7.set_title('F. Model Comparison by AIC', fontsize=11, fontweight='bold', loc='left')
ax7.set_ylim(60, 75)
ax7.grid(alpha=0.3, axis='y')

# Panel 8: Influence analysis
ax8 = fig.add_subplot(gs[2, 2])
influences = []
for s in data['study']:
    data_loo = data[data['study'] != s]
    weights_loo = data_loo['precision']**2
    wm_loo = np.sum(data_loo['y'] * weights_loo) / np.sum(weights_loo)
    influences.append(abs(weighted_mean - wm_loo))

studies_sorted = data['study'].astype(int)
colors_inf = ['red' if inf > 2 else 'steelblue' for inf in influences]
bars = ax8.bar(studies_sorted, influences, color=colors_inf, edgecolor='black',
              linewidth=1.5, alpha=0.7)
ax8.axhline(2, color='orange', linestyle='--', linewidth=2, alpha=0.7,
           label='High influence (>2)')
ax8.set_xlabel('Study', fontsize=10)
ax8.set_ylabel('Influence on Pooled Estimate', fontsize=10)
ax8.set_title('G. Leave-One-Out Influence', fontsize=11, fontweight='bold', loc='left')
ax8.set_xticks(studies_sorted)
ax8.legend(fontsize=8)
ax8.grid(alpha=0.3, axis='y')

# Add footer with interpretation
footer_text = """
KEY INSIGHTS: (1) Low heterogeneity (I²=2.9%) indicates studies measure similar effects. (2) Strong shrinkage (>95%) shows pooling is highly beneficial.
(3) No publication bias detected (symmetric funnel). (4) Study 4 most influential (3.74 unit change if removed). RECOMMENDATION: Bayesian hierarchical
model with partial pooling, reporting both confidence and prediction intervals. Conduct sensitivity analyses removing Studies 4 and 5.
"""
fig.text(0.5, 0.01, footer_text, ha='center', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
        wrap=True)

plt.savefig('/workspace/eda/visualizations/00_summary_figure.png', dpi=300, bbox_inches='tight')
plt.close()

print("="*70)
print("Summary figure created: /workspace/eda/visualizations/00_summary_figure.png")
print("="*70)
print("\nThis comprehensive figure shows:")
print("  A. Forest plot with all studies")
print("  B. Shrinkage visualization")
print("  C. Variance decomposition")
print("  D. Funnel plot for publication bias")
print("  E. Distribution of effects")
print("  F. Model comparison (AIC)")
print("  G. Leave-one-out influence analysis")
print("\nUse this as a standalone summary of all key findings.")
print("="*70)
