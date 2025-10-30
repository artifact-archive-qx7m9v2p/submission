"""
Distribution Analysis - Analyst 1
==================================
Purpose: Examine univariate distributions of Y and x
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Create multi-panel plot for distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Univariate Distributions: Y and x', fontsize=16, fontweight='bold')

# Row 1: Y variable
# Histogram
axes[0, 0].hist(data['Y'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(data['Y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data["Y"].mean():.3f}')
axes[0, 0].axvline(data['Y'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data["Y"].median():.3f}')
axes[0, 0].set_xlabel('Y', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Y: Histogram', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Boxplot
bp = axes[0, 1].boxplot(data['Y'], vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('steelblue')
axes[0, 1].set_ylabel('Y', fontsize=11)
axes[0, 1].set_title('Y: Boxplot', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3, axis='y')

# Q-Q plot
stats.probplot(data['Y'], dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('Y: Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
axes[0, 2].grid(alpha=0.3)

# Row 2: x variable
# Histogram
axes[1, 0].hist(data['x'], bins=15, edgecolor='black', alpha=0.7, color='coral')
axes[1, 0].axvline(data['x'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data["x"].mean():.2f}')
axes[1, 0].axvline(data['x'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data["x"].median():.2f}')
axes[1, 0].set_xlabel('x', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('x: Histogram', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Boxplot
bp = axes[1, 1].boxplot(data['x'], vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('coral')
axes[1, 1].set_ylabel('x', fontsize=11)
axes[1, 1].set_title('x: Boxplot', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3, axis='y')

# Q-Q plot
stats.probplot(data['x'], dist="norm", plot=axes[1, 2])
axes[1, 2].set_title('x: Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/01_univariate_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: 01_univariate_distributions.png")
plt.close()

# Test for normality
print("\nNORMALITY TESTS")
print("="*60)
print("\nY variable:")
shapiro_y = stats.shapiro(data['Y'])
print(f"  Shapiro-Wilk test: W = {shapiro_y.statistic:.4f}, p-value = {shapiro_y.pvalue:.4f}")
if shapiro_y.pvalue > 0.05:
    print("  => Y appears normally distributed (p > 0.05)")
else:
    print("  => Y deviates from normality (p < 0.05)")

print("\nx variable:")
shapiro_x = stats.shapiro(data['x'])
print(f"  Shapiro-Wilk test: W = {shapiro_x.statistic:.4f}, p-value = {shapiro_x.pvalue:.4f}")
if shapiro_x.pvalue > 0.05:
    print("  => x appears normally distributed (p > 0.05)")
else:
    print("  => x deviates from normality (p < 0.05)")

# Create density plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Kernel Density Estimates', fontsize=14, fontweight='bold')

# Y density
axes[0].hist(data['Y'], bins=15, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Histogram')
axes[0].plot(np.linspace(data['Y'].min(), data['Y'].max(), 100),
             stats.norm.pdf(np.linspace(data['Y'].min(), data['Y'].max(), 100),
                           data['Y'].mean(), data['Y'].std()),
             'r-', linewidth=2, label='Normal fit')
from scipy.stats import gaussian_kde
kde_y = gaussian_kde(data['Y'])
x_range = np.linspace(data['Y'].min(), data['Y'].max(), 100)
axes[0].plot(x_range, kde_y(x_range), 'g-', linewidth=2, label='KDE')
axes[0].set_xlabel('Y', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Y: Density Plot', fontsize=12)
axes[0].legend()
axes[0].grid(alpha=0.3)

# x density
axes[1].hist(data['x'], bins=15, density=True, alpha=0.6, color='coral', edgecolor='black', label='Histogram')
axes[1].plot(np.linspace(data['x'].min(), data['x'].max(), 100),
             stats.norm.pdf(np.linspace(data['x'].min(), data['x'].max(), 100),
                           data['x'].mean(), data['x'].std()),
             'r-', linewidth=2, label='Normal fit')
kde_x = gaussian_kde(data['x'])
x_range = np.linspace(data['x'].min(), data['x'].max(), 100)
axes[1].plot(x_range, kde_x(x_range), 'g-', linewidth=2, label='KDE')
axes[1].set_xlabel('x', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].set_title('x: Density Plot', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/02_density_plots.png', dpi=300, bbox_inches='tight')
print("Saved: 02_density_plots.png")
plt.close()

print("\nDistribution analysis complete!")
