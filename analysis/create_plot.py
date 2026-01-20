#!/usr/bin/env python3
"""
Clean ELPD comparison plot with clear annotations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read data
df = pd.read_csv('elpd_comparison_data.csv')

# Create dataset-type labels
unique_datasets = df.groupby(['Dataset', 'Type']).first().reset_index()
unique_datasets['Dataset_Label'] = unique_datasets['Dataset'].str.replace('_', ' ') + '\n(' + unique_datasets['Type'] + ')'

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Set up x positions
x_positions = {}
current_x = 0
for dataset, dtype in zip(unique_datasets['Dataset'], unique_datasets['Type']):
    key = f"{dataset}_{dtype}"
    x_positions[key] = current_x
    current_x += 1

offset = 0.15

# Plot reference scores
for i, row in unique_datasets.iterrows():
    key = f"{row['Dataset']}_{row['Type']}"
    x = x_positions[key]

    # Error bar
    ax.errorbar(x, row['Ref_ELPD'], yerr=row['Ref_SE'],
               fmt='none', ecolor='#3498db', elinewidth=3,
               capsize=6, capthick=3, alpha=0.7, zorder=3)

    # Point
    ax.scatter(x, row['Ref_ELPD'],
              c='#3498db', marker='s', s=180,
              edgecolors='white', linewidth=2,
              label='Expert Reference' if i == 0 else '', zorder=10)

# Plot our scores
for i, row in df.iterrows():
    key = f"{row['Dataset']}_{row['Type']}"
    x = x_positions[key]

    # Calculate offset for multiple runs
    same_dataset = df[(df['Dataset'] == row['Dataset']) & (df['Type'] == row['Type'])]
    if len(same_dataset) > 1:
        run_idx = list(same_dataset.index).index(i)
        total_runs = len(same_dataset)
        offset_x = (run_idx - (total_runs-1)/2) * offset
    else:
        offset_x = 0

    x_pos = x + offset_x

    # Error bar
    ax.errorbar(x_pos, row['Our_ELPD'], yerr=row['Our_SE'],
               fmt='none', ecolor='#e67e22', elinewidth=2.5,
               capsize=5, capthick=2.5, alpha=0.6, zorder=4)

    # Point
    ax.scatter(x_pos, row['Our_ELPD'],
              c='#e67e22', marker='o', s=120,
              edgecolors='white', linewidth=2,
              label='Agent Models' if i == 0 else '', zorder=11)

# Formatting
ax.set_xticks(range(len(unique_datasets)))
ax.set_xticklabels(unique_datasets['Dataset_Label'], fontsize=12)
ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
ax.set_ylabel('ELPD-LOO (± SE)    [Higher is Better]', fontsize=14, fontweight='bold')
ax.set_title('ELPD Comparison', fontsize=16, fontweight='bold', pad=15)
ax.grid(True, alpha=0.2, axis='y', linestyle='--', zorder=0)
ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1, zorder=1)

# Legend
ax.legend(loc='lower left', fontsize=13, framealpha=0.95)

# Vertical separators
prev_dataset = None
for i, row in unique_datasets.iterrows():
    if prev_dataset is not None and row['Dataset'] != prev_dataset:
        key = f"{row['Dataset']}_{row['Type']}"
        x = x_positions[key] - 0.5
        ax.axvline(x, color='lightgray', linestyle='-', alpha=0.4, linewidth=1.5, zorder=0)
    prev_dataset = row['Dataset']

# Add upward arrow on y-axis to emphasize "higher is better"
y_lim = ax.get_ylim()
y_arrow_start = y_lim[0] + 0.15 * (y_lim[1] - y_lim[0])
y_arrow_end = y_lim[0] + 0.35 * (y_lim[1] - y_lim[0])
ax.annotate('',
           xy=(-0.8, y_arrow_end), xytext=(-0.8, y_arrow_start),
           arrowprops=dict(arrowstyle='->', lw=3, color='darkgreen'))

# Add note about excluded runs
n_total = 24
n_shown = len(df)
n_excluded = n_total - n_shown
note_text = f'{n_excluded} runs excluded:\nerrors or incomplete'
ax.text(0.98, 0.02, note_text,
       transform=ax.transAxes,
       fontsize=10, ha='right', va='bottom',
       style='italic',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffebee',
                edgecolor='#c62828', linewidth=1.5, alpha=0.9))

plt.tight_layout()
plt.savefig('elpd_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"Plot saved to: elpd_comparison.png")
print(f"Showing {n_shown} successful runs (excluded {n_excluded} problematic runs)")
