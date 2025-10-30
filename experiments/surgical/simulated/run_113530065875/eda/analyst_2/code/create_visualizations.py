"""
Create visualizations for binomial data EDA
Focus: Pooling comparisons, hierarchical structure, and prior elicitation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Load processed data
data = pd.read_csv('/workspace/eda/analyst_2/code/processed_data.csv')

# Calculate pooled rate
total_trials = data['n'].sum()
total_successes = data['r'].sum()
pooled_rate = total_successes / total_trials

print("Creating visualizations...")

# =============================================================================
# 1. CATERPILLAR PLOT: Individual rates vs pooled rate
# =============================================================================
print("  1. Caterpillar plot...")

fig, ax = plt.subplots(figsize=(12, 8))

# Sort by observed rate
data_sorted = data.sort_values('p_hat')
y_pos = np.arange(len(data_sorted))

# Plot confidence intervals
for i, row in data_sorted.iterrows():
    ax.plot([row['ci_lower'], row['ci_upper']], [y_pos[data_sorted.index == i][0]] * 2,
            'o-', color='steelblue', linewidth=2, markersize=4, alpha=0.7)

# Plot point estimates
ax.scatter(data_sorted['p_hat'], y_pos, s=100, color='darkblue', zorder=3, alpha=0.8, label='Observed rate')

# Plot pooled rate
ax.axvline(pooled_rate, color='red', linestyle='--', linewidth=2, label=f'Pooled rate ({pooled_rate:.4f})', zorder=2)

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Group {g}" for g in data_sorted['group']])
ax.set_xlabel('Success Rate', fontsize=12)
ax.set_ylabel('Group', fontsize=12)
ax.set_title('Individual Group Success Rates vs Pooled Rate\n(95% Wilson Confidence Intervals)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/01_caterpillar_plot.png')
plt.close()

# =============================================================================
# 2. FOREST PLOT: Rates with sample size information
# =============================================================================
print("  2. Forest plot with sample sizes...")

fig, ax = plt.subplots(figsize=(12, 8))

# Sort by rate
data_sorted = data.sort_values('p_hat')
y_pos = np.arange(len(data_sorted))

# Plot confidence intervals with varying line width by sample size
max_n = data_sorted['n'].max()
for i, row in data_sorted.iterrows():
    linewidth = 1 + 4 * (row['n'] / max_n)  # Scale line width by sample size
    ax.plot([row['ci_lower'], row['ci_upper']], [y_pos[data_sorted.index == i][0]] * 2,
            'o-', color='steelblue', linewidth=linewidth, markersize=6, alpha=0.7)

# Plot point estimates with size proportional to n
sizes = 50 + 300 * (data_sorted['n'] / max_n)
ax.scatter(data_sorted['p_hat'], y_pos, s=sizes, color='darkblue', zorder=3, alpha=0.7, edgecolors='black', linewidths=1)

# Plot pooled rate
ax.axvline(pooled_rate, color='red', linestyle='--', linewidth=2, label=f'Pooled rate ({pooled_rate:.4f})')

# Add sample size labels
for i, row in data_sorted.iterrows():
    ax.text(row['ci_upper'] + 0.005, y_pos[data_sorted.index == i][0],
            f"n={row['n']}", fontsize=8, va='center')

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Group {g}" for g in data_sorted['group']])
ax.set_xlabel('Success Rate', fontsize=12)
ax.set_ylabel('Group', fontsize=12)
ax.set_title('Forest Plot: Success Rates by Group\n(Marker and line size proportional to sample size)',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/02_forest_plot_sample_sizes.png')
plt.close()

# =============================================================================
# 3. POOLED VS UNPOOLED VS PARTIAL POOLING
# =============================================================================
print("  3. Three-way pooling comparison...")

fig, ax = plt.subplots(figsize=(12, 8))

# Sort by group number to show structure
data_sorted = data.sort_values('group')
y_pos = np.arange(len(data_sorted))

# Plot unpooled estimates
ax.scatter(data_sorted['p_hat'], y_pos, s=150, color='blue',
           label='Unpooled (independent)', marker='o', alpha=0.7, zorder=3)

# Plot pooled estimate
ax.scatter([pooled_rate] * len(y_pos), y_pos, s=150, color='red',
           label='Completely pooled', marker='s', alpha=0.7, zorder=3)

# Plot partially pooled estimates
ax.scatter(data_sorted['p_hat_pooled'], y_pos, s=150, color='green',
           label='Partially pooled (empirical Bayes)', marker='^', alpha=0.7, zorder=3)

# Draw arrows showing shrinkage
for i, row in data_sorted.iterrows():
    y = y_pos[data_sorted.index == i][0]
    ax.annotate('', xy=(row['p_hat_pooled'], y), xytext=(row['p_hat'], y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Group {g}" for g in data_sorted['group']])
ax.set_xlabel('Success Rate', fontsize=12)
ax.set_ylabel('Group', fontsize=12)
ax.set_title('Comparison: Unpooled vs Pooled vs Partial Pooling\n(Arrows show shrinkage direction)',
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/03_pooling_comparison.png')
plt.close()

# =============================================================================
# 4. SHRINKAGE ANALYSIS
# =============================================================================
print("  4. Shrinkage analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Shrinkage percentage by group
data_sorted = data.sort_values('shrinkage_pct', ascending=True)
y_pos = np.arange(len(data_sorted))

colors = plt.cm.RdYlGn_r(data_sorted['shrinkage_pct'] / data_sorted['shrinkage_pct'].max())
ax1.barh(y_pos, data_sorted['shrinkage_pct'], color=colors, edgecolor='black', linewidth=0.5)

ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"Group {g}" for g in data_sorted['group']])
ax1.set_xlabel('Shrinkage Toward Grand Mean (%)', fontsize=11)
ax1.set_ylabel('Group', fontsize=11)
ax1.set_title('Shrinkage by Group\n(Higher = more shrinkage)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Right panel: Shrinkage vs sample size
ax2.scatter(data['n'], data['shrinkage_pct'], s=150, alpha=0.7, edgecolors='black', linewidths=1)

# Add labels for groups with high shrinkage
high_shrink = data[data['shrinkage_pct'] > 50]
for _, row in high_shrink.iterrows():
    ax2.annotate(f"Grp {int(row['group'])}",
                xy=(row['n'], row['shrinkage_pct']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax2.set_xlabel('Sample Size (n)', fontsize=11)
ax2.set_ylabel('Shrinkage (%)', fontsize=11)
ax2.set_title('Shrinkage vs Sample Size\n(Smaller samples shrink more)', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/04_shrinkage_analysis.png')
plt.close()

# =============================================================================
# 5. PRIOR PREDICTIVE EXPLORATION
# =============================================================================
print("  5. Prior predictive exploration...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Different prior options
priors = [
    (1, 1, "Beta(1,1)\nUniform"),
    (0.5, 0.5, "Beta(0.5,0.5)\nJeffreys"),
    (2, 2, "Beta(2,2)\nWeak"),
    (5, 50, "Beta(5,50)\nData-informed"),
    (4.65, 54.36, "Beta(4.65,54.36)\nMethod of Moments"),
    (1, 10, "Beta(1,10)\nSkewed Low")
]

x = np.linspace(0, 0.3, 1000)

for idx, (alpha, beta, label) in enumerate(priors):
    ax = axes[idx]

    # Plot prior density
    y = stats.beta.pdf(x, alpha, beta)
    ax.plot(x, y, 'b-', linewidth=2, label='Prior density')
    ax.fill_between(x, y, alpha=0.3)

    # Add vertical lines for observed data quantiles
    ax.axvline(data['p_hat'].min(), color='red', linestyle='--', alpha=0.5, label='Data range')
    ax.axvline(data['p_hat'].max(), color='red', linestyle='--', alpha=0.5)
    ax.axvline(data['p_hat'].median(), color='green', linestyle=':', linewidth=2, label='Data median')

    # Shade the region covering observed data
    obs_min, obs_max = data['p_hat'].min(), data['p_hat'].max()
    ax.axvspan(obs_min, obs_max, alpha=0.1, color='yellow')

    # Prior statistics
    prior_mean = alpha / (alpha + beta)
    prior_std = np.sqrt(alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1)))

    ax.set_xlabel('Success Rate', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{label}\nMean={prior_mean:.3f}, SD={prior_std:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 0.3)
    if idx == 0:
        ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

plt.suptitle('Prior Predictive Exploration for Success Rate\n(Yellow shading = observed data range)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/05_prior_predictive.png')
plt.close()

# =============================================================================
# 6. DEVIATION FROM POOLED RATE
# =============================================================================
print("  6. Deviation analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Absolute deviation
data_sorted = data.sort_values('abs_deviation', ascending=False)
y_pos = np.arange(len(data_sorted))

colors = ['red' if dev > 0 else 'blue' for dev in data_sorted['deviation_from_pooled']]
ax1.barh(y_pos, data_sorted['abs_deviation'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"Group {g}" for g in data_sorted['group']])
ax1.set_xlabel('Absolute Deviation from Pooled Rate', fontsize=11)
ax1.set_ylabel('Group', fontsize=11)
ax1.set_title('Deviation from Pooled Rate\n(Red=above, Blue=below)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Right: Scatter of deviation vs sample size
above_pooled = data[data['p_hat'] > pooled_rate]
below_pooled = data[data['p_hat'] <= pooled_rate]

ax2.scatter(above_pooled['n'], above_pooled['abs_deviation'],
           s=150, color='red', alpha=0.7, label='Above pooled', edgecolors='black', linewidths=1)
ax2.scatter(below_pooled['n'], below_pooled['abs_deviation'],
           s=150, color='blue', alpha=0.7, label='Below pooled', edgecolors='black', linewidths=1)

# Add group labels
for _, row in data.iterrows():
    if row['abs_deviation'] > 0.04:  # Label extreme deviations
        ax2.annotate(f"Grp {int(row['group'])}",
                    xy=(row['n'], row['abs_deviation']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

ax2.set_xlabel('Sample Size (n)', fontsize=11)
ax2.set_ylabel('Absolute Deviation from Pooled Rate', fontsize=11)
ax2.set_title('Deviation vs Sample Size\n(Larger samples not necessarily closer to pooled)',
             fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/06_deviation_analysis.png')
plt.close()

# =============================================================================
# 7. HIERARCHICAL STRUCTURE EVIDENCE
# =============================================================================
print("  7. Hierarchical structure evidence...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Distribution of rates on probability scale
ax1.hist(data['p_hat'], bins=8, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(data['p_hat'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax1.axvline(pooled_rate, color='orange', linestyle=':', linewidth=2, label='Pooled')
ax1.set_xlabel('Success Rate', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Success Rates\n(Probability Scale)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Panel 2: Distribution of rates on logit scale
ax2.hist(data['logit_p'], bins=8, edgecolor='black', alpha=0.7, color='lightcoral')
ax2.axvline(data['logit_p'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax2.set_xlabel('log(p/(1-p))', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Distribution of Success Rates\n(Logit Scale - more symmetric)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Panel 3: Variance components
components = ['Within-group\n(mean)', 'Between-group\n(tau²)']
values = [data['se_logit'].pow(2).mean(), data['logit_p'].var() - data['se_logit'].pow(2).mean()]
colors_comp = ['lightblue', 'lightcoral']

ax3.bar(components, values, color=colors_comp, edgecolor='black', linewidth=1, alpha=0.7)
ax3.set_ylabel('Variance (logit scale)', fontsize=11)
ax3.set_title('Variance Components\n(Higher between-group variance supports hierarchical model)',
             fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add values on bars
for i, v in enumerate(values):
    ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

# Panel 4: ICC visualization
icc = values[1] / sum(values) if sum(values) > 0 else 0
ax4.pie([icc, 1-icc], labels=['Between-group', 'Within-group'],
        colors=['lightcoral', 'lightblue'], autopct='%1.1f%%', startangle=90)
ax4.set_title(f'Intraclass Correlation (ICC) = {icc:.3f}\n(Proportion of variance between groups)',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/07_hierarchical_evidence.png')
plt.close()

# =============================================================================
# 8. SAMPLE SIZE AND PRECISION
# =============================================================================
print("  8. Sample size and precision analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: CI width vs sample size
ax1.scatter(data['n'], data['ci_width'], s=150, alpha=0.7, edgecolors='black', linewidths=1)

# Add power law fit
log_n = np.log(data['n'])
log_ci = np.log(data['ci_width'])
slope, intercept = np.polyfit(log_n, log_ci, 1)
n_pred = np.linspace(data['n'].min(), data['n'].max(), 100)
ci_pred = np.exp(intercept) * n_pred**slope
ax1.plot(n_pred, ci_pred, 'r--', linewidth=2, alpha=0.7, label=f'Power law fit (slope={slope:.2f})')

# Label smallest sample groups
small_groups = data.nsmallest(3, 'n')
for _, row in small_groups.iterrows():
    ax1.annotate(f"Grp {int(row['group'])}\nn={int(row['n'])}",
                xy=(row['n'], row['ci_width']),
                xytext=(10, 10), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax1.set_xlabel('Sample Size (n)', fontsize=11)
ax1.set_ylabel('Confidence Interval Width', fontsize=11)
ax1.set_title('Precision vs Sample Size\n(Smaller n = wider CI)', fontsize=12, fontweight='bold')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(alpha=0.3, which='both')

# Right: Sample size distribution
ax2.bar(data['group'], data['n'], color='steelblue', edgecolor='black', linewidth=1, alpha=0.7)
ax2.axhline(data['n'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax2.axhline(data['n'].median(), color='orange', linestyle=':', linewidth=2, label='Median')

ax2.set_xlabel('Group', fontsize=11)
ax2.set_ylabel('Sample Size (n)', fontsize=11)
ax2.set_title('Sample Size by Group\n(Wide variation suggests differential precision)',
             fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/08_sample_size_precision.png')
plt.close()

# =============================================================================
# 9. EXTREME GROUPS IDENTIFICATION
# =============================================================================
print("  9. Extreme groups identification...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Z-scores for rates
data_sorted = data.sort_values('z_score_rate')
y_pos = np.arange(len(data_sorted))
colors = ['red' if abs(z) > 1.5 else 'gray' for z in data_sorted['z_score_rate']]

ax1.barh(y_pos, data_sorted['z_score_rate'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax1.axvline(1.5, color='red', linestyle='--', alpha=0.5)
ax1.axvline(-1.5, color='red', linestyle='--', alpha=0.5)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"Group {g}" for g in data_sorted['group']])
ax1.set_xlabel('Z-score (standardized rate)', fontsize=11)
ax1.set_ylabel('Group', fontsize=11)
ax1.set_title('Extreme Success Rates\n(|Z| > 1.5 highlighted)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Panel 2: Z-scores for sample sizes
data_sorted = data.sort_values('z_score_n')
y_pos = np.arange(len(data_sorted))
colors = ['red' if abs(z) > 1.5 else 'gray' for z in data_sorted['z_score_n']]

ax2.barh(y_pos, data_sorted['z_score_n'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axvline(1.5, color='red', linestyle='--', alpha=0.5)
ax2.axvline(-1.5, color='red', linestyle='--', alpha=0.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"Group {g}" for g in data_sorted['group']])
ax2.set_xlabel('Z-score (standardized sample size)', fontsize=11)
ax2.set_ylabel('Group', fontsize=11)
ax2.set_title('Extreme Sample Sizes\n(|Z| > 1.5 highlighted)', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Panel 3: Influence score
data_sorted = data.sort_values('influence_score', ascending=False)
y_pos = np.arange(len(data_sorted))

ax3.barh(y_pos, data_sorted['influence_score'], color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"Group {g}" for g in data_sorted['group']])
ax3.set_xlabel('Influence Score', fontsize=11)
ax3.set_ylabel('Group', fontsize=11)
ax3.set_title('Group Influence on Pooled Estimate\n(|Z-score| × proportion of total n)',
             fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Panel 4: Scatter showing both dimensions
ax4.scatter(data['z_score_rate'], data['z_score_n'], s=200, alpha=0.7, edgecolors='black', linewidths=1)

# Add quadrant lines
ax4.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax4.axvline(0, color='gray', linestyle='-', alpha=0.3)
ax4.axhline(1.5, color='red', linestyle='--', alpha=0.3)
ax4.axhline(-1.5, color='red', linestyle='--', alpha=0.3)
ax4.axvline(1.5, color='red', linestyle='--', alpha=0.3)
ax4.axvline(-1.5, color='red', linestyle='--', alpha=0.3)

# Label extreme groups
extreme = data[(abs(data['z_score_rate']) > 1.5) | (abs(data['z_score_n']) > 1.5)]
for _, row in extreme.iterrows():
    ax4.annotate(f"Grp {int(row['group'])}",
                xy=(row['z_score_rate'], row['z_score_n']),
                xytext=(5, 5), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax4.set_xlabel('Z-score: Success Rate', fontsize=11)
ax4.set_ylabel('Z-score: Sample Size', fontsize=11)
ax4.set_title('Joint Distribution of Extremeness\n(Both dimensions)', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/09_extreme_groups.png')
plt.close()

# =============================================================================
# 10. COMPREHENSIVE SUMMARY DASHBOARD
# =============================================================================
print("  10. Comprehensive summary dashboard...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Top row: Main comparison
ax1 = fig.add_subplot(gs[0, :])
data_sorted = data.sort_values('group')
x = np.arange(len(data_sorted))

ax1.plot(x, data_sorted['p_hat'], 'o-', label='Unpooled', markersize=8, linewidth=2, color='blue')
ax1.plot(x, [pooled_rate] * len(x), '--', label='Pooled', linewidth=2, color='red')
ax1.plot(x, data_sorted['p_hat_pooled'], '^-', label='Partial pooling', markersize=8, linewidth=2, color='green')

# Add error bars
ax1.errorbar(x, data_sorted['p_hat'],
            yerr=[data_sorted['p_hat'] - data_sorted['ci_lower'],
                  data_sorted['ci_upper'] - data_sorted['p_hat']],
            fmt='none', color='blue', alpha=0.3, capsize=3)

ax1.set_xticks(x)
ax1.set_xticklabels([f"{int(g)}" for g in data_sorted['group']])
ax1.set_xlabel('Group', fontsize=11)
ax1.set_ylabel('Success Rate', fontsize=11)
ax1.set_title('Pooling Strategy Comparison Across All Groups', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(alpha=0.3)

# Middle left: Sample sizes
ax2 = fig.add_subplot(gs[1, 0])
ax2.bar(data_sorted['group'], data_sorted['n'], color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Group', fontsize=10)
ax2.set_ylabel('Sample Size', fontsize=10)
ax2.set_title('Sample Size by Group', fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Middle center: Success count
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(data_sorted['group'], data_sorted['r'], color='darkgreen', edgecolor='black', alpha=0.7)
ax3.set_xlabel('Group', fontsize=10)
ax3.set_ylabel('Success Count', fontsize=10)
ax3.set_title('Successes by Group', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Middle right: CI width
ax4 = fig.add_subplot(gs[1, 2])
ax4.bar(data_sorted['group'], data_sorted['ci_width'], color='coral', edgecolor='black', alpha=0.7)
ax4.set_xlabel('Group', fontsize=10)
ax4.set_ylabel('CI Width', fontsize=10)
ax4.set_title('Uncertainty by Group', fontsize=11, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Bottom left: Shrinkage
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(data['n'], data['shrinkage_pct'], s=100, alpha=0.7, edgecolors='black')
ax5.set_xlabel('Sample Size', fontsize=10)
ax5.set_ylabel('Shrinkage (%)', fontsize=10)
ax5.set_title('Shrinkage vs Sample Size', fontsize=11, fontweight='bold')
ax5.grid(alpha=0.3)

# Bottom center: Rate distribution
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(data['p_hat'], bins=8, edgecolor='black', alpha=0.7, color='purple')
ax6.axvline(pooled_rate, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('Success Rate', fontsize=10)
ax6.set_ylabel('Count', fontsize=10)
ax6.set_title('Distribution of Rates', fontsize=11, fontweight='bold')
ax6.grid(alpha=0.3)

# Bottom right: Summary statistics table
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

summary_stats = [
    ['Metric', 'Value'],
    ['Total groups', f'{len(data)}'],
    ['Total trials', f'{data["n"].sum()}'],
    ['Total successes', f'{data["r"].sum()}'],
    ['', ''],
    ['Pooled rate', f'{pooled_rate:.4f}'],
    ['Mean unpooled rate', f'{data["p_hat"].mean():.4f}'],
    ['SD of rates', f'{data["p_hat"].std():.4f}'],
    ['', ''],
    ['Estimated tau (logit)', f'{np.sqrt(max(0, data["logit_p"].var() - data["se_logit"].pow(2).mean())):.4f}'],
    ['ICC', f'{max(0, data["logit_p"].var() - data["se_logit"].pow(2).mean()) / data["logit_p"].var():.4f}'],
    ['', ''],
    ['Chi² test p-value', f'{stats.chi2_contingency(data[["r", "failures"]].values.T)[1]:.4f}'],
]

table = ax7.table(cellText=summary_stats, loc='center', cellLoc='left',
                 colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(summary_stats)):
    for j in range(2):
        if summary_stats[i][0] == '':
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('#ffffff' if i % 2 else '#f9f9f9')

ax7.set_title('Summary Statistics', fontsize=11, fontweight='bold', pad=20)

plt.suptitle('Binomial Data EDA - Comprehensive Summary Dashboard',
            fontsize=15, fontweight='bold', y=0.995)

plt.savefig('/workspace/eda/analyst_2/visualizations/10_summary_dashboard.png')
plt.close()

print("\nAll visualizations created successfully!")
print("Saved to: /workspace/eda/analyst_2/visualizations/")
