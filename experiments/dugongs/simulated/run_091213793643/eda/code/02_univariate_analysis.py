"""
Script 2: Univariate Analysis
Analyzes the distribution of each variable independently
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Set up paths
DATA_PATH = Path("/workspace/data/data.csv")
VIZ_DIR = Path("/workspace/eda/visualizations")

# Load data
df = pd.read_csv(DATA_PATH)

print("="*80)
print("UNIVARIATE ANALYSIS")
print("="*80)

# ============================================================================
# DISTRIBUTION OF X
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
axes[0, 0].hist(df['x'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(df['x'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {df["x"].mean():.2f}')
axes[0, 0].axvline(df['x'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {df["x"].median():.2f}')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of x (Histogram)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# KDE plot
axes[0, 1].hist(df['x'], bins=15, density=True, edgecolor='black', alpha=0.5, color='steelblue', label='Histogram')
df['x'].plot(kind='kde', ax=axes[0, 1], color='darkblue', linewidth=2, label='KDE')
axes[0, 1].axvline(df['x'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {df["x"].mean():.2f}')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Distribution of x (Kernel Density Estimate)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Box plot
box = axes[1, 0].boxplot(df['x'], vert=True, patch_artist=True)
box['boxes'][0].set_facecolor('lightblue')
axes[1, 0].set_ylabel('x')
axes[1, 0].set_title('Box Plot of x')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(df['x'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot of x (vs Normal Distribution)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'univariate_x_distribution.png', bbox_inches='tight')
plt.close()

print("\nX Distribution Analysis:")
print(f"  Saved: univariate_x_distribution.png")

# Test normality for x
shapiro_x = stats.shapiro(df['x'])
ks_x = stats.kstest(df['x'], 'norm', args=(df['x'].mean(), df['x'].std()))
print(f"  Shapiro-Wilk test: statistic={shapiro_x.statistic:.6f}, p-value={shapiro_x.pvalue:.6f}")
print(f"  Kolmogorov-Smirnov test: statistic={ks_x.statistic:.6f}, p-value={ks_x.pvalue:.6f}")
if shapiro_x.pvalue < 0.05:
    print(f"  -> x is NOT normally distributed (p < 0.05)")
else:
    print(f"  -> x may be normally distributed (p >= 0.05)")

# ============================================================================
# DISTRIBUTION OF Y
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
axes[0, 0].hist(df['Y'], bins=12, edgecolor='black', alpha=0.7, color='forestgreen')
axes[0, 0].axvline(df['Y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {df["Y"].mean():.2f}')
axes[0, 0].axvline(df['Y'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {df["Y"].median():.2f}')
axes[0, 0].set_xlabel('Y')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Y (Histogram)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# KDE plot
axes[0, 1].hist(df['Y'], bins=12, density=True, edgecolor='black', alpha=0.5, color='forestgreen', label='Histogram')
df['Y'].plot(kind='kde', ax=axes[0, 1], color='darkgreen', linewidth=2, label='KDE')
axes[0, 1].axvline(df['Y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {df["Y"].mean():.2f}')
axes[0, 1].set_xlabel('Y')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Distribution of Y (Kernel Density Estimate)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Box plot
box = axes[1, 0].boxplot(df['Y'], vert=True, patch_artist=True)
box['boxes'][0].set_facecolor('lightgreen')
axes[1, 0].set_ylabel('Y')
axes[1, 0].set_title('Box Plot of Y')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(df['Y'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot of Y (vs Normal Distribution)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'univariate_y_distribution.png', bbox_inches='tight')
plt.close()

print("\nY Distribution Analysis:")
print(f"  Saved: univariate_y_distribution.png")

# Test normality for Y
shapiro_y = stats.shapiro(df['Y'])
ks_y = stats.kstest(df['Y'], 'norm', args=(df['Y'].mean(), df['Y'].std()))
print(f"  Shapiro-Wilk test: statistic={shapiro_y.statistic:.6f}, p-value={shapiro_y.pvalue:.6f}")
print(f"  Kolmogorov-Smirnov test: statistic={ks_y.statistic:.6f}, p-value={ks_y.pvalue:.6f}")
if shapiro_y.pvalue < 0.05:
    print(f"  -> Y is NOT normally distributed (p < 0.05)")
else:
    print(f"  -> Y may be normally distributed (p >= 0.05)")

# ============================================================================
# COMBINED DISTRIBUTION OVERVIEW
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# X distribution
axes[0].hist(df['x'], bins=15, density=True, edgecolor='black', alpha=0.5, color='steelblue')
df['x'].plot(kind='kde', ax=axes[0], color='darkblue', linewidth=2.5)
axes[0].axvline(df['x'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {df["x"].mean():.2f}')
axes[0].axvline(df['x'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {df["x"].median():.2f}')
axes[0].set_xlabel('x (Predictor)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Distribution of Predictor (x)', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Y distribution
axes[1].hist(df['Y'], bins=12, density=True, edgecolor='black', alpha=0.5, color='forestgreen')
df['Y'].plot(kind='kde', ax=axes[1], color='darkgreen', linewidth=2.5)
axes[1].axvline(df['Y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {df["Y"].mean():.2f}')
axes[1].axvline(df['Y'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {df["Y"].median():.2f}')
axes[1].set_xlabel('Y (Response)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Distribution of Response (Y)', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'univariate_combined_distributions.png', bbox_inches='tight')
plt.close()

print("\nCombined Distribution Overview:")
print(f"  Saved: univariate_combined_distributions.png")

print("\n" + "="*80)
print("UNIVARIATE ANALYSIS COMPLETE")
print("="*80)
