"""
Distribution Analysis: Success Rates and Sample Sizes
EDA Analyst 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set up paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data_analyst_1.csv'
VIZ_DIR = BASE_DIR / 'eda' / 'analyst_1' / 'visualizations'

# Load data
df = pd.read_csv(DATA_PATH)

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

print("=" * 70)
print("DISTRIBUTION ANALYSIS")
print("=" * 70)

# ============================================================================
# FIGURE 1: Success Rate Distribution
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Success Rate Distribution Analysis', fontsize=16, fontweight='bold', y=0.995)

# Histogram
axes[0, 0].hist(df['success_rate'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(df['success_rate'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {df["success_rate"].mean():.4f}')
axes[0, 0].axvline(df['success_rate'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {df["success_rate"].median():.4f}')
axes[0, 0].set_xlabel('Success Rate', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('(A) Histogram of Success Rates', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot
bp = axes[0, 1].boxplot(df['success_rate'], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('black')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)
axes[0, 1].set_ylabel('Success Rate', fontsize=11)
axes[0, 1].set_title('(B) Box Plot of Success Rates', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Add statistics to box plot
q1, median, q3 = df['success_rate'].quantile([0.25, 0.5, 0.75])
iqr = q3 - q1
axes[0, 1].text(1.3, q1, f'Q1: {q1:.4f}', fontsize=9)
axes[0, 1].text(1.3, median, f'Median: {median:.4f}', fontsize=9, color='red', fontweight='bold')
axes[0, 1].text(1.3, q3, f'Q3: {q3:.4f}', fontsize=9)
axes[0, 1].text(1.3, df['success_rate'].min(), f'Min: {df["success_rate"].min():.4f}', fontsize=9)
axes[0, 1].text(1.3, df['success_rate'].max(), f'Max: {df["success_rate"].max():.4f}', fontsize=9)

# QQ plot for normality check
stats.probplot(df['success_rate'], dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('(C) Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Kernel Density Estimate
axes[1, 1].hist(df['success_rate'], bins=15, density=True, alpha=0.5, color='steelblue', edgecolor='black', label='Histogram')
df['success_rate'].plot(kind='kde', ax=axes[1, 1], color='darkblue', linewidth=2, label='KDE')
axes[1, 1].axvline(df['success_rate'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
axes[1, 1].set_xlabel('Success Rate', fontsize=11)
axes[1, 1].set_ylabel('Density', fontsize=11)
axes[1, 1].set_title('(D) Kernel Density Estimate', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'success_rate_distribution.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: success_rate_distribution.png")
plt.close()

# ============================================================================
# FIGURE 2: Sample Size Distribution
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Sample Size (n_trials) Distribution', fontsize=16, fontweight='bold')

# Histogram
axes[0].hist(df['n_trials'], bins=12, edgecolor='black', alpha=0.7, color='darkgreen')
axes[0].axvline(df['n_trials'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {df["n_trials"].mean():.1f}')
axes[0].axvline(df['n_trials'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {df["n_trials"].median():.1f}')
axes[0].set_xlabel('Number of Trials', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('(A) Histogram of Sample Sizes', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bar chart by group
axes[1].bar(df['group'], df['n_trials'], color='darkgreen', alpha=0.7, edgecolor='black')
axes[1].axhline(df['n_trials'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
axes[1].set_xlabel('Group', fontsize=11)
axes[1].set_ylabel('Number of Trials', fontsize=11)
axes[1].set_title('(B) Sample Size by Group', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(VIZ_DIR / 'sample_size_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved: sample_size_distribution.png")
plt.close()

# ============================================================================
# Statistical Tests
# ============================================================================
print("\n" + "=" * 70)
print("NORMALITY TESTS FOR SUCCESS RATE")
print("=" * 70)

# Shapiro-Wilk test
shapiro_stat, shapiro_p = stats.shapiro(df['success_rate'])
print(f"\nShapiro-Wilk Test:")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.4f}")
print(f"  Interpretation: {'Reject normality' if shapiro_p < 0.05 else 'Cannot reject normality'} (α=0.05)")

# Anderson-Darling test
anderson_result = stats.anderson(df['success_rate'], dist='norm')
print(f"\nAnderson-Darling Test:")
print(f"  Statistic: {anderson_result.statistic:.4f}")
print(f"  Critical values: {anderson_result.critical_values}")
print(f"  Significance levels: {anderson_result.significance_level}")

# Skewness and Kurtosis
skewness = stats.skew(df['success_rate'])
kurtosis = stats.kurtosis(df['success_rate'])
print(f"\nDistribution Shape:")
print(f"  Skewness: {skewness:.4f} {'(right-skewed)' if skewness > 0 else '(left-skewed)' if skewness < 0 else '(symmetric)'}")
print(f"  Kurtosis: {kurtosis:.4f} {'(heavy-tailed)' if kurtosis > 0 else '(light-tailed)' if kurtosis < 0 else '(normal-tailed)'}")
