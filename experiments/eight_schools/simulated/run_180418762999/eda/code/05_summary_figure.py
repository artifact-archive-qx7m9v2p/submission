"""
Summary Figure: Complete EDA Overview
======================================
Goal: Create a single comprehensive figure summarizing all key findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")

# Load data
data = pd.read_csv('/workspace/eda/code/data_with_metrics.csv')

# Create comprehensive summary figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
fig.suptitle('EDA Summary: Hierarchical Dataset with Known Measurement Error (n=8)',
             fontsize=16, fontweight='bold', y=0.995)

# ============================================================================
# Panel 1: Data Overview
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.text(0.5, 0.95, 'DATA OVERVIEW', ha='center', fontsize=12, fontweight='bold',
         transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
info_text = f"""
Sample Size: n = 8 groups

Response Variable (y):
  Range: [{data['y'].min():.1f}, {data['y'].max():.1f}]
  Mean: {data['y'].mean():.2f} ± {data['y'].std():.2f}
  Median: {data['y'].median():.2f}

Measurement Error (σ):
  Range: [{data['sigma'].min()}, {data['sigma'].max()}]
  Mean: {data['sigma'].mean():.2f} ± {data['sigma'].std():.2f}

Signal-to-Noise:
  Mean SNR: {data['snr'].mean():.2f}
  Median SNR: {data['snr'].median():.2f}
  Groups with SNR>1: 4
  Groups with SNR<1: 4

Data Quality:
  Missing values: 0
  Outliers: 0
  Normality: p = 0.67 ✓
"""
ax1.text(0.05, 0.85, info_text, ha='left', va='top', fontsize=9,
         transform=ax1.transAxes, family='monospace')
ax1.axis('off')

# ============================================================================
# Panel 2: Key Finding - Measurement Error Dominates
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.text(0.5, 0.95, 'KEY FINDING', ha='center', fontsize=12, fontweight='bold',
         transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='coral', alpha=0.5))

variance_data = [
    ('Observed\nVariance', 124.27),
    ('Expected\nMeasurement\nVariance', 166.00),
    ('Between-\nGroup\nVariance', 0.0)
]
x_pos = range(len(variance_data))
colors = ['steelblue', 'coral', 'red']
bars = ax2.bar(x_pos, [v[1] for v in variance_data], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Variance', fontweight='bold', fontsize=10)
ax2.set_title('Variance Decomposition', fontsize=11, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([v[0] for v in variance_data], fontsize=8)
ax2.axhline(y=0, color='black', linewidth=1)
ax2.text(1, 166, 'Error explains\nall variation!', ha='center', va='bottom',
         fontsize=9, fontweight='bold', color='red',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax2.grid(alpha=0.3, axis='y')

# ============================================================================
# Panel 3: Hypothesis Test Results
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])
ax3.text(0.5, 0.95, 'HYPOTHESIS TESTS', ha='center', fontsize=12, fontweight='bold',
         transform=ax3.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

test_results = """
Test                    P-value   Result
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Homogeneity             0.42      ✓ Pooling
(Chi-square)                      supported

Mean vs Zero            0.016     ✓ Mean is
(t-test)                          positive

Mean vs Zero            0.014     ✓ Mean is
(weighted)                        positive

Normality               0.67      ✓ Normal
(Shapiro-Wilk)                    assumed

Correlation y-σ         0.39      ✓ Independent
(Pearson)                         error

Outliers                All >0.05 ✓ None
(Leave-one-out)                   detected
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONCLUSION: Complete pooling strongly
supported with positive mean.
"""
ax3.text(0.05, 0.85, test_results, ha='left', va='top', fontsize=7.5,
         transform=ax3.transAxes, family='monospace')
ax3.axis('off')

# ============================================================================
# Panel 4: Model Recommendation
# ============================================================================
ax4 = fig.add_subplot(gs[0, 3])
ax4.text(0.5, 0.95, 'MODEL RECOMMENDATION', ha='center', fontsize=12, fontweight='bold',
         transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5))

model_text = """
PRIMARY MODEL:
Complete Pooling with Known Error

  y_i ~ Normal(μ, σ_i)
  μ ~ Normal(10, 20)

where σ_i are known (given in data)

JUSTIFICATION:
• Between-group variance = 0
• Chi-square test: p = 0.42
• Simplest model supported by data
• Maximum precision from pooling

EXPECTED POSTERIOR:
• μ ≈ N(10, 4)
• 95% CI: [2, 18]

ALTERNATIVE:
Hierarchical model for sensitivity
(expect τ ≈ 0, reduces to pooling)

NOT RECOMMENDED:
No pooling (wastes information,
not supported by data)
"""
ax4.text(0.05, 0.85, model_text, ha='left', va='top', fontsize=8.5,
         transform=ax4.transAxes, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax4.axis('off')

# ============================================================================
# Panel 5: Observed Data with Error Bars
# ============================================================================
ax5 = fig.add_subplot(gs[1, :2])
ax5.errorbar(data['group'], data['y'], yerr=data['sigma'],
             fmt='o', markersize=12, capsize=8, capthick=3,
             color='steelblue', ecolor='coral', elinewidth=3, alpha=0.7,
             label='Observed ± 1σ')
# Add weighted mean line
weights = 1 / (data['sigma']**2)
weighted_mean = np.sum(data['y'] * weights) / np.sum(weights)
ax5.axhline(y=weighted_mean, color='green', linestyle='--', linewidth=3,
            label=f'Weighted mean: {weighted_mean:.2f}', alpha=0.7)
ax5.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
# Color code by SNR
for i, row in data.iterrows():
    color = 'green' if row['snr'] > 1 else 'red'
    ax5.plot(row['group'], row['y'], 'o', markersize=12, color=color, zorder=5, alpha=0.8)
ax5.set_xlabel('Group', fontweight='bold', fontsize=11)
ax5.set_ylabel('Response Variable (y)', fontweight='bold', fontsize=11)
ax5.set_title('Observed Values with Measurement Error (green=good SNR, red=poor SNR)',
              fontsize=11, fontweight='bold')
ax5.set_xticks(data['group'])
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3)

# ============================================================================
# Panel 6: Signal-to-Noise Ratio
# ============================================================================
ax6 = fig.add_subplot(gs[1, 2])
colors_snr = ['green' if snr >= 1 else 'red' for snr in data['snr']]
bars = ax6.bar(data['group'], data['snr'], color=colors_snr, alpha=0.7, edgecolor='black', linewidth=2)
ax6.axhline(y=1, color='black', linestyle='--', linewidth=2, label='SNR = 1')
ax6.axhline(y=data['snr'].mean(), color='blue', linestyle=':', linewidth=2,
            label=f'Mean = {data["snr"].mean():.2f}')
ax6.set_xlabel('Group', fontweight='bold', fontsize=10)
ax6.set_ylabel('Signal-to-Noise Ratio', fontweight='bold', fontsize=10)
ax6.set_title('Signal-to-Noise Ratio by Group', fontsize=11, fontweight='bold')
ax6.set_xticks(data['group'])
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3, axis='y')
ax6.text(0.5, 0.95, 'Half of observations\nhave SNR < 1', ha='center', va='top',
         transform=ax6.transAxes, fontsize=9, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ============================================================================
# Panel 7: Distribution Analysis
# ============================================================================
ax7 = fig.add_subplot(gs[1, 3])
ax7.hist(data['y'], bins=6, alpha=0.6, color='steelblue', edgecolor='black',
         density=True, label='Observed data')
# Overlay normal distribution
x_range = np.linspace(data['y'].min() - 5, data['y'].max() + 5, 100)
normal_dist = stats.norm(loc=weighted_mean, scale=data['y'].std())
ax7.plot(x_range, normal_dist.pdf(x_range), 'r-', linewidth=2,
         label='Normal distribution')
ax7.axvline(weighted_mean, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax7.set_xlabel('Response Variable (y)', fontweight='bold', fontsize=10)
ax7.set_ylabel('Density', fontweight='bold', fontsize=10)
ax7.set_title('Distribution Check\n(Shapiro-Wilk p=0.67)', fontsize=11, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(alpha=0.3)

# ============================================================================
# Panel 8: Model Comparison Visualization
# ============================================================================
ax8 = fig.add_subplot(gs[2, :2])
x = np.arange(len(data))
# Complete pooling estimates
complete_pool = [weighted_mean] * len(data)
# No pooling estimates
no_pool = data['y'].values
# Partial pooling (simple shrinkage)
shrinkage = 0.5
partial_pool = shrinkage * data['y'] + (1 - shrinkage) * weighted_mean

ax8.plot(x, data['y'], 'ko', markersize=10, label='Observed data', zorder=5)
ax8.plot(x, complete_pool, 's-', color='blue', linewidth=2, markersize=8,
         label='Complete pooling', alpha=0.7)
ax8.plot(x, no_pool, '^--', color='red', linewidth=2, markersize=8,
         label='No pooling', alpha=0.7)
ax8.plot(x, partial_pool, 'D:', color='purple', linewidth=2, markersize=8,
         label='Partial pooling', alpha=0.7)
ax8.set_xlabel('Group', fontweight='bold', fontsize=11)
ax8.set_ylabel('Estimated Value', fontweight='bold', fontsize=11)
ax8.set_title('Comparison of Three Modeling Approaches', fontsize=11, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(data['group'])
ax8.legend(fontsize=10, loc='best')
ax8.grid(alpha=0.3)
ax8.text(0.5, 0.02, 'Complete pooling supported: τ² = 0 (no between-group variance)',
         ha='center', va='bottom', transform=ax8.transAxes, fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# ============================================================================
# Panel 9: Power Analysis
# ============================================================================
ax9 = fig.add_subplot(gs[2, 2])
true_effects = np.linspace(0, 40, 100)
avg_sigma = data['sigma'].mean()
power = stats.norm.sf(1.96 - true_effects / avg_sigma)
ax9.plot(true_effects, power, 'b-', linewidth=3)
ax9.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80% power')
ax9.axvline(x=24.5, color='orange', linestyle=':', linewidth=2,
            label=f'Effect = 24.5\n(80% power)')
ax9.fill_between(true_effects, 0, power, alpha=0.3, color='steelblue')
ax9.set_xlabel('True Effect Size', fontweight='bold', fontsize=10)
ax9.set_ylabel('Statistical Power', fontweight='bold', fontsize=10)
ax9.set_title('Statistical Power\n(avg σ = 12.5)', fontsize=11, fontweight='bold')
ax9.legend(fontsize=9)
ax9.grid(alpha=0.3)
ax9.set_ylim(0, 1)
ax9.text(0.5, 0.5, 'Low power!\nNeed large\neffects', ha='center', va='center',
         transform=ax9.transAxes, fontsize=10, fontweight='bold', color='red',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# ============================================================================
# Panel 10: Key Takeaways
# ============================================================================
ax10 = fig.add_subplot(gs[2, 3])
ax10.text(0.5, 0.95, 'KEY TAKEAWAYS', ha='center', fontsize=12, fontweight='bold',
          transform=ax10.transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

takeaways = """
1. MEASUREMENT ERROR DOMINATES
   • SNR ≈ 1 (signal = noise)
   • Observed var < measurement var
   • Individual estimates unreliable

2. GROUPS ARE HOMOGENEOUS
   • Between-group variance = 0
   • Chi-square test: p = 0.42
   • Complete pooling supported

3. MEAN IS POSITIVE
   • Weighted mean: 10.02 ± 4.07
   • One-sample test: p = 0.014
   • Likely range: [5, 15]

4. USE COMPLETE POOLING
   • Simplest supported model
   • Maximum precision
   • Properly account for known σ

5. ACKNOWLEDGE UNCERTAINTY
   • Wide confidence intervals
   • Low statistical power
   • Need effects >25 to detect
"""
ax10.text(0.05, 0.85, takeaways, ha='left', va='top', fontsize=8,
          transform=ax10.transAxes, family='monospace')
ax10.axis('off')

plt.savefig('/workspace/eda/visualizations/00_eda_summary.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/visualizations/00_eda_summary.png")
print("\nThis comprehensive summary figure includes:")
print("  - Data overview and quality assessment")
print("  - Key finding: measurement error dominates")
print("  - All hypothesis test results")
print("  - Model recommendations")
print("  - Observed data with uncertainty")
print("  - Signal-to-noise analysis")
print("  - Distribution validation")
print("  - Model comparison")
print("  - Statistical power analysis")
print("  - Key takeaways")
plt.close()

print("\n" + "="*80)
print("EDA COMPLETE!")
print("="*80)
print("\nAll analyses finished successfully.")
print(f"\nTotal visualizations created: 9 figures")
print("\nStart with: /workspace/eda/visualizations/00_eda_summary.png")
print("For details, see: /workspace/eda/eda_report.md")
print("For quick reference: /workspace/eda/findings.md")
print("="*80)
