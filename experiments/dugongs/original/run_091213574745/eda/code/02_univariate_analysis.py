"""
Univariate Analysis
===================
Examine distributions of x and Y independently
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/data/data.csv')

# Set up figure for x distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution Analysis: x (Predictor)', fontsize=16, fontweight='bold')

# x: Histogram
ax = axes[0, 0]
ax.hist(df['x'], bins=15, edgecolor='black', alpha=0.7)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Histogram', fontsize=12)
ax.grid(True, alpha=0.3)

# x: Density plot
ax = axes[0, 1]
df['x'].plot(kind='density', ax=ax, linewidth=2)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Kernel Density Estimate', fontsize=12)
ax.grid(True, alpha=0.3)

# x: Box plot
ax = axes[0, 2]
ax.boxplot(df['x'], vert=True)
ax.set_ylabel('x', fontsize=12)
ax.set_title('Box Plot', fontsize=12)
ax.grid(True, alpha=0.3)

# x: Q-Q plot for normality
ax = axes[1, 0]
stats.probplot(df['x'], dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normal)', fontsize=12)
ax.grid(True, alpha=0.3)

# x: Empirical CDF
ax = axes[1, 1]
x_sorted = np.sort(df['x'])
y_ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
ax.step(x_sorted, y_ecdf, where='post', linewidth=2)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('ECDF', fontsize=12)
ax.set_title('Empirical Cumulative Distribution', fontsize=12)
ax.grid(True, alpha=0.3)

# x: Sequential plot
ax = axes[1, 2]
ax.plot(range(len(df)), df['x'], marker='o', linestyle='-', markersize=4)
ax.set_xlabel('Observation Index', fontsize=12)
ax.set_ylabel('x', fontsize=12)
ax.set_title('Sequential Plot (Order in Dataset)', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/01_x_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Set up figure for Y distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution Analysis: Y (Response)', fontsize=16, fontweight='bold')

# Y: Histogram
ax = axes[0, 0]
ax.hist(df['Y'], bins=15, edgecolor='black', alpha=0.7, color='coral')
ax.set_xlabel('Y', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Histogram', fontsize=12)
ax.grid(True, alpha=0.3)

# Y: Density plot
ax = axes[0, 1]
df['Y'].plot(kind='density', ax=ax, linewidth=2, color='coral')
ax.set_xlabel('Y', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Kernel Density Estimate', fontsize=12)
ax.grid(True, alpha=0.3)

# Y: Box plot
ax = axes[0, 2]
ax.boxplot(df['Y'], vert=True)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Box Plot', fontsize=12)
ax.grid(True, alpha=0.3)

# Y: Q-Q plot for normality
ax = axes[1, 0]
stats.probplot(df['Y'], dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normal)', fontsize=12)
ax.grid(True, alpha=0.3)

# Y: Empirical CDF
ax = axes[1, 1]
y_sorted = np.sort(df['Y'])
y_ecdf = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
ax.step(y_sorted, y_ecdf, where='post', linewidth=2, color='coral')
ax.set_xlabel('Y', fontsize=12)
ax.set_ylabel('ECDF', fontsize=12)
ax.set_title('Empirical Cumulative Distribution', fontsize=12)
ax.grid(True, alpha=0.3)

# Y: Sequential plot
ax = axes[1, 2]
ax.plot(range(len(df)), df['Y'], marker='o', linestyle='-', markersize=4, color='coral')
ax.set_xlabel('Observation Index', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Sequential Plot (Order in Dataset)', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/02_Y_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Statistical tests
print("=" * 80)
print("NORMALITY TESTS")
print("=" * 80)

print("\nVariable: x")
print("-" * 80)
shapiro_stat, shapiro_p = stats.shapiro(df['x'])
print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
ks_stat, ks_p = stats.kstest(df['x'], 'norm', args=(df['x'].mean(), df['x'].std()))
print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")

print("\nVariable: Y")
print("-" * 80)
shapiro_stat, shapiro_p = stats.shapiro(df['Y'])
print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
ks_stat, ks_p = stats.kstest(df['Y'], 'norm', args=(df['Y'].mean(), df['Y'].std()))
print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("For Shapiro-Wilk test: p < 0.05 suggests departure from normality")
print("For K-S test: p < 0.05 suggests departure from normality")
