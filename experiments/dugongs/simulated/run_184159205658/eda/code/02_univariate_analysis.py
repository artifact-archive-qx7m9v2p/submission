"""
Univariate Distribution Analysis
=================================
Author: EDA Specialist Agent
Date: 2025-10-27

This script creates comprehensive visualizations of individual variable distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data.csv'
VIZ_DIR = BASE_DIR / 'eda' / 'visualizations'

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("UNIVARIATE ANALYSIS")
print("=" * 80)

# Figure 1: Distribution of x variable
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Predictor Variable (x)', fontsize=16, fontweight='bold')

# Histogram
axes[0, 0].hist(df['x'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(df['x'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["x"].mean():.2f}')
axes[0, 0].axvline(df['x'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["x"].median():.2f}')
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Histogram of x', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# KDE plot
axes[0, 1].hist(df['x'], bins=15, density=True, edgecolor='black', alpha=0.5, color='steelblue', label='Histogram')
df['x'].plot(kind='kde', ax=axes[0, 1], color='darkblue', linewidth=2, label='KDE')
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_ylabel('Density', fontsize=12)
axes[0, 1].set_title('Density Plot of x', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Box plot
bp = axes[1, 0].boxplot(df['x'], vert=False, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('black')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_title('Box Plot of x', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Q-Q plot
stats.probplot(df['x'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot of x (vs Normal)', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'distribution_x.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'distribution_x.png'}")
plt.close()

# Figure 2: Distribution of Y variable
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Response Variable (Y)', fontsize=16, fontweight='bold')

# Histogram
axes[0, 0].hist(df['Y'], bins=12, edgecolor='black', alpha=0.7, color='coral')
axes[0, 0].axvline(df['Y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["Y"].mean():.2f}')
axes[0, 0].axvline(df['Y'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["Y"].median():.2f}')
axes[0, 0].set_xlabel('Y', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Histogram of Y', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# KDE plot
axes[0, 1].hist(df['Y'], bins=12, density=True, edgecolor='black', alpha=0.5, color='coral', label='Histogram')
df['Y'].plot(kind='kde', ax=axes[0, 1], color='darkred', linewidth=2, label='KDE')
axes[0, 1].set_xlabel('Y', fontsize=12)
axes[0, 1].set_ylabel('Density', fontsize=12)
axes[0, 1].set_title('Density Plot of Y', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Box plot
bp = axes[1, 0].boxplot(df['Y'], vert=False, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightsalmon')
bp['boxes'][0].set_edgecolor('black')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)
axes[1, 0].set_xlabel('Y', fontsize=12)
axes[1, 0].set_title('Box Plot of Y', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Q-Q plot
stats.probplot(df['Y'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot of Y (vs Normal)', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'distribution_Y.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'distribution_Y.png'}")
plt.close()

# Figure 3: Combined distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Comparative Distribution Analysis', fontsize=16, fontweight='bold')

# Standardized distributions
x_std = (df['x'] - df['x'].mean()) / df['x'].std()
y_std = (df['Y'] - df['Y'].mean()) / df['Y'].std()

axes[0].hist(x_std, bins=15, alpha=0.5, label='x (standardized)', edgecolor='black', color='steelblue')
axes[0].hist(y_std, bins=12, alpha=0.5, label='Y (standardized)', edgecolor='black', color='coral')
axes[0].set_xlabel('Standardized Value', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Standardized Distributions', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# KDE comparison
x_std_series = pd.Series(x_std)
y_std_series = pd.Series(y_std)
x_std_series.plot(kind='kde', ax=axes[1], label='x (standardized)', linewidth=2, color='steelblue')
y_std_series.plot(kind='kde', ax=axes[1], label='Y (standardized)', linewidth=2, color='coral')
axes[1].set_xlabel('Standardized Value', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Density Comparison', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'distribution_comparison.png'}")
plt.close()

# Statistical tests for normality
print("\n" + "=" * 80)
print("NORMALITY TESTS")
print("=" * 80)

for col in ['x', 'Y']:
    print(f"\n{col}:")
    # Shapiro-Wilk test
    stat, p = stats.shapiro(df[col])
    print(f"  Shapiro-Wilk test: statistic={stat:.4f}, p-value={p:.4f}")
    if p > 0.05:
        print(f"    -> Fail to reject normality (p > 0.05)")
    else:
        print(f"    -> Reject normality (p <= 0.05)")

    # Anderson-Darling test
    result = stats.anderson(df[col])
    print(f"  Anderson-Darling test: statistic={result.statistic:.4f}")
    print(f"    Critical values: {result.critical_values}")
    print(f"    Significance levels: {result.significance_level}")

print("\n" + "=" * 80)
print("Univariate analysis complete!")
print("=" * 80)
