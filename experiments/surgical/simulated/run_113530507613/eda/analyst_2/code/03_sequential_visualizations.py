"""
Visualizations for Sequential and Temporal Patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/data/data_analyst_2.csv')
df['failure_count'] = df['n_trials'] - df['r_successes']
df['logit_success_rate'] = np.log((df['r_successes'] + 0.5) / (df['failure_count'] + 0.5))

# Set style
sns.set_style("whitegrid")

# ============================================================================
# FIGURE 1: Sequential Pattern Analysis (Multi-panel)
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sequential Pattern Analysis Across Group ID', fontsize=14, fontweight='bold', y=0.995)

# Panel 1: Success Rate over Group ID with trend line
ax1 = axes[0, 0]
ax1.plot(df['group_id'], df['success_rate'], 'o-', color='steelblue', markersize=8, linewidth=2, label='Observed')
# Add trend line
z = np.polyfit(df['group_id'], df['success_rate'], 1)
p = np.poly1d(z)
ax1.plot(df['group_id'], p(df['group_id']), "--", color='red', alpha=0.7, linewidth=1.5, label=f'Linear trend (p=0.30)')
ax1.axhline(y=df['success_rate'].mean(), color='gray', linestyle=':', alpha=0.5, label='Mean')
ax1.set_xlabel('Group ID', fontsize=11)
ax1.set_ylabel('Success Rate', fontsize=11)
ax1.set_title('A. Success Rate Sequence', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: n_trials over Group ID
ax2 = axes[0, 1]
ax2.plot(df['group_id'], df['n_trials'], 's-', color='darkgreen', markersize=8, linewidth=2)
ax2.axhline(y=df['n_trials'].mean(), color='gray', linestyle=':', alpha=0.5, label='Mean')
ax2.set_xlabel('Group ID', fontsize=11)
ax2.set_ylabel('Number of Trials', fontsize=11)
ax2.set_title('B. Sample Size Sequence', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
# Highlight outlier
outlier_idx = df['n_trials'].idxmax()
ax2.annotate(f'Group {df.loc[outlier_idx, "group_id"]}\n(n={df.loc[outlier_idx, "n_trials"]})',
             xy=(df.loc[outlier_idx, 'group_id'], df.loc[outlier_idx, 'n_trials']),
             xytext=(df.loc[outlier_idx, 'group_id']+1, df.loc[outlier_idx, 'n_trials']+50),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=9, color='red')

# Panel 3: Autocorrelation plot for success_rate
ax3 = axes[1, 0]
max_lag = 5
acf_values = []
for lag in range(1, max_lag + 1):
    if lag < len(df):
        acf = np.corrcoef(df['success_rate'][:-lag], df['success_rate'][lag:])[0, 1]
        acf_values.append(acf)
    else:
        acf_values.append(np.nan)

lags = range(1, len(acf_values) + 1)
ax3.bar(lags, acf_values, color='steelblue', alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
# Add confidence interval (approximate)
conf_int = 1.96 / np.sqrt(len(df))
ax3.axhline(y=conf_int, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'95% CI')
ax3.axhline(y=-conf_int, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax3.set_xlabel('Lag', fontsize=11)
ax3.set_ylabel('Autocorrelation', fontsize=11)
ax3.set_title('C. Autocorrelation Function (Success Rate)', fontsize=12, fontweight='bold')
ax3.set_ylim(-1, 1)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Runs visualization
ax4 = axes[1, 1]
median = df['success_rate'].median()
colors = ['green' if x > median else 'orange' for x in df['success_rate']]
ax4.bar(df['group_id'], df['success_rate'], color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax4.axhline(y=median, color='red', linestyle='--', linewidth=2, label=f'Median = {median:.3f}')
ax4.set_xlabel('Group ID', fontsize=11)
ax4.set_ylabel('Success Rate', fontsize=11)
ax4.set_title('D. Runs Around Median (Runs test p=0.23)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9, loc='upper right')
ax4.grid(True, alpha=0.3, axis='y')
# Add text
ax4.text(0.02, 0.98, 'Green: Above median\nOrange: Below median',
         transform=ax4.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/sequential_patterns.png', dpi=300, bbox_inches='tight')
print("Saved: sequential_patterns.png")
plt.close()

# ============================================================================
# FIGURE 2: Segment Comparison
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Temporal Segmentation Analysis', fontsize=14, fontweight='bold')

# Panel 1: Boxplot by segments (thirds)
ax1 = axes[0]
segment_labels = ['Early\n(1-4)', 'Middle\n(5-8)', 'Late\n(9-12)']
segment_data = [
    df.iloc[:4]['success_rate'].values,
    df.iloc[4:8]['success_rate'].values,
    df.iloc[8:]['success_rate'].values
]
bp = ax1.boxplot(segment_data, labels=segment_labels, patch_artist=True,
                 medianprops=dict(color='red', linewidth=2),
                 boxprops=dict(facecolor='lightblue', alpha=0.7))
ax1.set_ylabel('Success Rate', fontsize=11)
ax1.set_xlabel('Temporal Segment', fontsize=11)
ax1.set_title('A. Success Rate by Temporal Segments\n(Kruskal-Wallis p=0.87)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add mean markers
means = [np.mean(data) for data in segment_data]
ax1.plot(range(1, 4), means, 'D', color='darkred', markersize=8, label='Mean', zorder=5)
ax1.legend(fontsize=9)

# Panel 2: Cumulative success rate
ax2 = axes[1]
cumulative_successes = df['r_successes'].cumsum()
cumulative_trials = df['n_trials'].cumsum()
cumulative_rate = cumulative_successes / cumulative_trials

ax2.plot(df['group_id'], cumulative_rate, 'o-', color='purple', linewidth=2.5, markersize=8)
ax2.axhline(y=df['success_rate'].mean(), color='red', linestyle='--', linewidth=1.5,
            label=f'Overall mean = {df["success_rate"].mean():.3f}')
ax2.set_xlabel('Group ID', fontsize=11)
ax2.set_ylabel('Cumulative Success Rate', fontsize=11)
ax2.set_title('B. Cumulative Success Rate Evolution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Add annotations for stabilization
ax2.annotate('Early volatility', xy=(2, cumulative_rate.iloc[1]),
             xytext=(3, cumulative_rate.iloc[1]+0.02),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1),
             fontsize=9, color='gray')
ax2.annotate('Stabilization', xy=(10, cumulative_rate.iloc[9]),
             xytext=(8, cumulative_rate.iloc[9]+0.015),
             arrowprops=dict(arrowstyle='->', color='gray', lw=1),
             fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/temporal_segments.png', dpi=300, bbox_inches='tight')
print("Saved: temporal_segments.png")
plt.close()

print("\nVisualization generation complete!")
