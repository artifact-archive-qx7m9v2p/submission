"""
Distribution Visualization
Focus: Histogram, density plots, and theoretical distribution comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Setup
sns.set_style("whitegrid")
output_dir = Path('/workspace/eda/analyst_2/visualizations')

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
C = data['C'].values

# Create multi-panel distribution overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram with KDE
ax = axes[0, 0]
ax.hist(C, bins=20, density=True, alpha=0.7, edgecolor='black', color='steelblue', label='Observed')
# Add KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(C)
x_range = np.linspace(C.min(), C.max(), 200)
ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
ax.set_xlabel('Count (C)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Count Variable', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add statistics box
stats_text = f'n = {len(C)}\nMean = {C.mean():.1f}\nSD = {C.std(ddof=1):.1f}\nVar/Mean = {C.var(ddof=1)/C.mean():.1f}'
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

# 2. Box plot with violin overlay
ax = axes[0, 1]
parts = ax.violinplot([C], positions=[0], widths=0.7, showmeans=True, showmedians=True)
ax.boxplot([C], positions=[0], widths=0.3, patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.7))
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Distribution Shape', fontsize=12, fontweight='bold')
ax.set_xticks([0])
ax.set_xticklabels(['Count Data'])
ax.grid(True, alpha=0.3, axis='y')

# Add quartile labels
q1, median, q3 = np.percentile(C, [25, 50, 75])
ax.axhline(median, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.text(-0.4, median, f'Median: {median:.0f}', fontsize=9)
ax.text(-0.4, q1, f'Q1: {q1:.0f}', fontsize=9)
ax.text(-0.4, q3, f'Q3: {q3:.0f}', fontsize=9)

# 3. Empirical CDF
ax = axes[1, 0]
sorted_c = np.sort(C)
y = np.arange(1, len(C) + 1) / len(C)
ax.step(sorted_c, y, where='post', linewidth=2, color='steelblue', label='Empirical CDF')
ax.set_xlabel('Count (C)', fontsize=11)
ax.set_ylabel('Cumulative Probability', fontsize=11)
ax.set_title('Empirical Cumulative Distribution Function', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Add percentile lines
for p in [0.25, 0.5, 0.75]:
    val = np.percentile(C, p * 100)
    ax.axhline(p, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(val, color='red', linestyle='--', alpha=0.3, linewidth=1)

# 4. Counts over time (to show context)
ax = axes[1, 1]
year = data['year'].values
ax.scatter(year, C, alpha=0.6, s=50, color='steelblue')
ax.plot(year, C, alpha=0.3, color='gray', linestyle='-')
ax.set_xlabel('Standardized Year', fontsize=11)
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Counts Over Time (for context)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(year, C, 1)
p = np.poly1d(z)
ax.plot(year, p(year), "r--", alpha=0.8, linewidth=2, label=f'Linear trend')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'distribution_overview.png', dpi=300, bbox_inches='tight')
print(f"Saved: distribution_overview.png")
plt.close()

# Create histogram with bin analysis
fig, ax = plt.subplots(figsize=(10, 6))
n_bins = int(np.sqrt(len(C))) + 5  # Sturges' rule + buffer
counts, bins, patches = ax.hist(C, bins=n_bins, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Count (C)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Count Distribution (Frequency Histogram)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add statistics
stats_text = (f'n = {len(C)}\n'
              f'Mean = {C.mean():.2f}\n'
              f'Median = {np.median(C):.2f}\n'
              f'Std = {C.std(ddof=1):.2f}\n'
              f'Skew = {stats.skew(C):.3f}\n'
              f'Kurtosis = {stats.kurtosis(C):.3f}')
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'count_histogram.png', dpi=300, bbox_inches='tight')
print(f"Saved: count_histogram.png")
plt.close()

print("\nDistribution summary:")
print(f"  - Moderate positive skewness ({stats.skew(C):.3f})")
print(f"  - Negative excess kurtosis ({stats.kurtosis(C):.3f}) - platykurtic (flatter than normal)")
print(f"  - Strong right tail with max = {C.max()}")
