"""
Summary Visualization: Model Comparison and Key Findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'eda' / 'analyst_3' / 'code' / 'data_with_diagnostics.csv'
VIZ_DIR = BASE_DIR / 'eda' / 'analyst_3' / 'visualizations'

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
df = pd.read_csv(DATA_PATH)

# Create summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# ============================================================================
# Panel 1: Model Comparison (AIC/BIC)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

models = ['Beta-Binomial\n(RECOMMENDED)', 'Heterogeneous\nBinomial', 'Pooled\nBinomial']
aic_values = [47.69, 73.76, 90.29]
bic_values = [48.66, 79.58, 90.78]
params = [2, 12, 1]

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, aic_values, width, label='AIC', color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, bic_values, width, label='BIC', color='coral', alpha=0.8, edgecolor='black')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add parameter count annotations
for i, (model, param) in enumerate(zip(models, params)):
    ax1.text(i, 95, f'{param} param{"s" if param > 1 else ""}',
            ha='center', va='top', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.set_ylabel('Information Criterion', fontsize=12, fontweight='bold')
ax1.set_title('Model Comparison: Lower is Better\nBeta-Binomial Wins by 26+ AIC Points',
             fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11, fontweight='bold')
ax1.legend(fontsize=11, frameon=True, loc='upper right')
ax1.set_ylim([0, 100])
ax1.grid(axis='y', alpha=0.3)

# Highlight best model
ax1.axvspan(-0.5, 0.5, alpha=0.1, color='green')

# ============================================================================
# Panel 2: Overdispersion Evidence
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

chi_square = 38.56
df_resid = 11
dispersion = 3.51

categories = ['Observed\nChi-square', 'Expected\n(if binomial)', 'Dispersion\nParameter']
values = [chi_square, df_resid, dispersion]
colors = ['red', 'green', 'orange']

bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

ax2.set_ylabel('Value', fontsize=11, fontweight='bold')
ax2.set_title('Evidence of Overdispersion\n(p < 0.0001)', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 45])
ax2.grid(axis='y', alpha=0.3)

# Add reference line for dispersion
ax2.axhline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Expected (φ=1)')
ax2.legend(fontsize=9, loc='upper right')

# ============================================================================
# Panel 3: Sample Size Distribution
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

ax3.boxplot([df['n_trials']], widths=0.5, patch_artist=True,
           boxprops=dict(facecolor='steelblue', alpha=0.7),
           medianprops=dict(color='red', linewidth=2),
           whiskerprops=dict(linewidth=1.5),
           capprops=dict(linewidth=1.5))

# Add individual points
ax3.scatter([1]*len(df), df['n_trials'], alpha=0.5, s=50, color='darkblue', zorder=3)

# Annotate outliers
outliers = df[df['n_trials'] > 352]
for idx, row in outliers.iterrows():
    ax3.annotate(f"Group {row['group']}",
                xy=(1, row['n_trials']),
                xytext=(1.15, row['n_trials']),
                fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax3.set_ylabel('Number of Trials', fontsize=11, fontweight='bold')
ax3.set_title(f'Sample Size Variability\n(CV = {df["n_trials"].std()/df["n_trials"].mean():.2f})',
             fontsize=12, fontweight='bold')
ax3.set_xticks([1])
ax3.set_xticklabels(['All Groups'])
ax3.grid(axis='y', alpha=0.3)

# Add statistics
stats_text = f"Min: {df['n_trials'].min()}\nMedian: {df['n_trials'].median():.0f}\nMax: {df['n_trials'].max()}"
ax3.text(0.65, 0.95, stats_text, transform=ax3.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# Panel 4: Success Rate Distribution
# ============================================================================
ax4 = fig.add_subplot(gs[1, 2])

ax4.boxplot([df['success_rate']*100], widths=0.5, patch_artist=True,
           boxprops=dict(facecolor='coral', alpha=0.7),
           medianprops=dict(color='darkred', linewidth=2),
           whiskerprops=dict(linewidth=1.5),
           capprops=dict(linewidth=1.5))

# Add individual points
ax4.scatter([1]*len(df), df['success_rate']*100, alpha=0.5, s=50, color='darkred', zorder=3)

# Annotate extreme values
group1 = df[df['group'] == 1].iloc[0]
group8 = df[df['group'] == 8].iloc[0]

ax4.annotate('Group 1\n(0%)',
            xy=(1, group1['success_rate']*100),
            xytext=(0.7, 2),
            fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

ax4.annotate('Group 8\n(14.4%)',
            xy=(1, group8['success_rate']*100),
            xytext=(1.2, 12),
            fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax4.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax4.set_title('Success Rate Heterogeneity\n(Groups Clearly Differ)',
             fontsize=12, fontweight='bold')
ax4.set_xticks([1])
ax4.set_xticklabels(['All Groups'])
ax4.grid(axis='y', alpha=0.3)

# Add statistics
stats_text = f"Min: {df['success_rate'].min()*100:.1f}%\nMedian: {df['success_rate'].median()*100:.1f}%\nMax: {df['success_rate'].max()*100:.1f}%"
ax4.text(0.65, 0.95, stats_text, transform=ax4.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# Panel 5: Assumption Testing Summary
# ============================================================================
ax5 = fig.add_subplot(gs[2, :])

assumptions = [
    'Binary outcomes',
    'Fixed sample sizes',
    'Independence\nacross groups',
    'Homogeneous\nsuccess rates',
    'No overdispersion',
    'Normal\napproximation'
]

statuses = ['PASS', 'PASS', 'PASS', 'FAIL', 'FAIL', 'PARTIAL']
colors_status = ['green', 'green', 'green', 'red', 'red', 'orange']
evidence = ['Data structure', 'Known n_i', 'p=0.29 (NS)', 'p<0.001', 'φ=3.5', '11/12 groups']

y_pos = np.arange(len(assumptions))

# Create horizontal bar chart
bars = ax5.barh(y_pos, [1]*len(assumptions), height=0.6, color=colors_status, alpha=0.7, edgecolor='black')

# Add assumption labels
for i, (assumption, status, evid) in enumerate(zip(assumptions, statuses, evidence)):
    ax5.text(0.02, i, assumption, ha='left', va='center', fontsize=10, fontweight='bold')
    ax5.text(0.5, i, status, ha='center', va='center', fontsize=11, fontweight='bold',
            color='white' if status == 'FAIL' else 'black')
    ax5.text(0.98, i, evid, ha='right', va='center', fontsize=9, style='italic')

ax5.set_yticks([])
ax5.set_xticks([])
ax5.set_xlim([0, 1])
ax5.set_title('Binomial Assumption Testing Summary', fontsize=14, fontweight='bold', pad=15)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.spines['left'].set_visible(False)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='✓ PASS'),
    Patch(facecolor='orange', alpha=0.7, label='⚠ PARTIAL'),
    Patch(facecolor='red', alpha=0.7, label='✗ FAIL')
]
ax5.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, frameon=True,
          bbox_to_anchor=(0.5, -0.15))

# Add overall title
fig.suptitle('EDA Summary: Model Assumptions and Data Quality Assessment\n' +
             'Recommendation: Bayesian Hierarchical Beta-Binomial Model',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(VIZ_DIR / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {VIZ_DIR / 'summary_dashboard.png'}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nKey Findings:")
print("1. Data quality: EXCELLENT (no missing values, no errors)")
print("2. Overdispersion: SIGNIFICANT (φ = 3.51, p < 0.0001)")
print("3. Group heterogeneity: STRONG (chi-square p < 0.0001)")
print("4. Best model: Beta-Binomial (AIC = 47.69)")
print("5. Critical issue: Group 1 has zero successes (requires shrinkage)")
print("\nRecommendation: Use Bayesian hierarchical beta-binomial model")
print("                with Beta(α=3.33, β=41.88) prior on success probabilities")
print("\nAll outputs saved to: /workspace/eda/analyst_3/")
