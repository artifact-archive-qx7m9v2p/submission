"""
Caterpillar Plots: Visualizing group-level variation with confidence intervals
Key question: Do confidence intervals overlap substantially, or is there clear separation?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
BASE_DIR = Path("/workspace/eda/analyst_2")
VIZ_DIR = BASE_DIR / "visualizations"
df = pd.read_csv(BASE_DIR / "code" / "group_data_with_ci.csv")

# Calculate pooled rate
pooled_rate = df['r_successes'].sum() / df['n_trials'].sum()

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# Plot 1: Classic Caterpillar Plot (sorted by success rate)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Sort by success rate for better visualization
df_sorted = df.sort_values('success_rate')

# Plot confidence intervals
y_positions = range(len(df_sorted))
ax.hlines(y=y_positions, xmin=df_sorted['ci_lower'], xmax=df_sorted['ci_upper'],
          color='steelblue', alpha=0.6, linewidth=2, label='95% CI')

# Plot point estimates
ax.scatter(df_sorted['success_rate'], y_positions,
           s=df_sorted['n_trials']/5,  # Size by number of trials
           color='darkblue', alpha=0.7, zorder=5, label='Observed rate (sized by n)')

# Add pooled rate line
ax.axvline(pooled_rate, color='red', linestyle='--', linewidth=2,
           label=f'Pooled rate ({pooled_rate:.3f})', alpha=0.7)

# Formatting
ax.set_yticks(y_positions)
ax.set_yticklabels([f"Group {g}" for g in df_sorted['group']])
ax.set_xlabel('Success Rate', fontsize=12)
ax.set_ylabel('Group', fontsize=12)
ax.set_title('Caterpillar Plot: Group Success Rates with 95% Confidence Intervals\n(sorted by rate, point size = sample size)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / "caterpillar_plot_sorted.png", dpi=150, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'caterpillar_plot_sorted.png'}")
plt.close()

# ============================================================================
# Plot 2: Caterpillar Plot by Group ID (original ordering)
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

y_positions = range(len(df))
ax.hlines(y=y_positions, xmin=df['ci_lower'], xmax=df['ci_upper'],
          color='steelblue', alpha=0.6, linewidth=2, label='95% CI')

ax.scatter(df['success_rate'], y_positions,
           s=df['n_trials']/5,
           color='darkblue', alpha=0.7, zorder=5, label='Observed rate (sized by n)')

ax.axvline(pooled_rate, color='red', linestyle='--', linewidth=2,
           label=f'Pooled rate ({pooled_rate:.3f})', alpha=0.7)

ax.set_yticks(y_positions)
ax.set_yticklabels([f"Group {g}" for g in df['group']])
ax.set_xlabel('Success Rate', fontsize=12)
ax.set_ylabel('Group', fontsize=12)
ax.set_title('Caterpillar Plot: Group Success Rates (by group ID)\n(point size = sample size)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / "caterpillar_plot_by_id.png", dpi=150, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'caterpillar_plot_by_id.png'}")
plt.close()

# ============================================================================
# Plot 3: Shrinkage Visualization - Observed vs Pooled
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Create shrinkage arrows
for idx, row in df.iterrows():
    # Arrow from observed to pooled
    ax.annotate('', xy=(pooled_rate, idx), xytext=(row['success_rate'], idx),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1.5))

# Plot observed rates
ax.scatter(df['success_rate'], range(len(df)),
           s=df['n_trials']/5,
           color='darkblue', alpha=0.7, zorder=5, label='Observed rate')

# Plot pooled rate
ax.scatter([pooled_rate] * len(df), range(len(df)),
           s=50, color='red', marker='s', alpha=0.7, zorder=5,
           label='Complete pooling (same for all)')

ax.axvline(pooled_rate, color='red', linestyle='--', linewidth=1, alpha=0.3)

ax.set_yticks(range(len(df)))
ax.set_yticklabels([f"Group {g}" for g in df['group']])
ax.set_xlabel('Success Rate', fontsize=12)
ax.set_ylabel('Group', fontsize=12)
ax.set_title('Shrinkage Potential: Observed Rates vs Complete Pooling\n(arrows show direction of shrinkage)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / "shrinkage_visualization.png", dpi=150, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'shrinkage_visualization.png'}")
plt.close()

# ============================================================================
# Analysis: Count overlapping confidence intervals
# ============================================================================
print("\n" + "=" * 80)
print("CONFIDENCE INTERVAL OVERLAP ANALYSIS")
print("=" * 80)

# Check how many CIs include the pooled rate
includes_pooled = ((df['ci_lower'] <= pooled_rate) & (df['ci_upper'] >= pooled_rate)).sum()
print(f"\nGroups whose 95% CI includes pooled rate: {includes_pooled}/{len(df)} ({includes_pooled/len(df)*100:.1f}%)")

groups_including_pooled = df[(df['ci_lower'] <= pooled_rate) & (df['ci_upper'] >= pooled_rate)]['group'].values
print(f"Groups: {groups_including_pooled}")

groups_below_pooled = df[df['ci_upper'] < pooled_rate]['group'].values
groups_above_pooled = df[df['ci_lower'] > pooled_rate]['group'].values
print(f"\nGroups with CI entirely below pooled rate: {groups_below_pooled}")
print(f"Groups with CI entirely above pooled rate: {groups_above_pooled}")

# Pairwise overlap analysis
print("\n" + "=" * 80)
print("PAIRWISE CI OVERLAP MATRIX")
print("=" * 80)

overlap_matrix = np.zeros((len(df), len(df)))
for i in range(len(df)):
    for j in range(len(df)):
        # Check if CIs overlap
        overlap = not (df.iloc[i]['ci_upper'] < df.iloc[j]['ci_lower'] or
                      df.iloc[j]['ci_upper'] < df.iloc[i]['ci_lower'])
        overlap_matrix[i, j] = int(overlap)

overlap_df = pd.DataFrame(overlap_matrix,
                          index=[f"G{g}" for g in df['group']],
                          columns=[f"G{g}" for g in df['group']])
print("\nOverlap matrix (1 = overlap, 0 = no overlap):")
print(overlap_df.to_string())

# Count non-overlapping pairs
n_pairs = len(df) * (len(df) - 1) / 2
n_overlapping = (overlap_matrix.sum() - len(df)) / 2  # Subtract diagonal, divide by 2
pct_overlapping = n_overlapping / n_pairs * 100

print(f"\nTotal pairs: {int(n_pairs)}")
print(f"Overlapping pairs: {int(n_overlapping)} ({pct_overlapping:.1f}%)")
print(f"Non-overlapping pairs: {int(n_pairs - n_overlapping)} ({100-pct_overlapping:.1f}%)")
