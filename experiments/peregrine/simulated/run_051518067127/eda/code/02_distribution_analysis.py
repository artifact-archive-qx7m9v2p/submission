"""
Distribution Analysis: Count and Predictor Variables
====================================================
Goal: Visualize and assess distributional properties
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/workspace/data/data.csv')

# Create comprehensive distribution plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution Analysis of Count Time Series Data', fontsize=16, y=1.00)

# 1. Histogram of C
ax = axes[0, 0]
ax.hist(data['C'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(data['C'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={data["C"].mean():.1f}')
ax.axvline(data['C'].median(), color='green', linestyle='--', linewidth=2, label=f'Median={data["C"].median():.1f}')
ax.set_xlabel('Count (C)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Count Variable')
ax.legend()
ax.grid(alpha=0.3)

# 2. Box plot of C
ax = axes[0, 1]
bp = ax.boxplot(data['C'], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('black')
ax.set_ylabel('Count (C)')
ax.set_title('Box Plot of Count Variable')
ax.grid(alpha=0.3, axis='y')

# Add quartile values
q1, q2, q3 = data['C'].quantile([0.25, 0.5, 0.75])
ax.text(1.3, q1, f'Q1={q1:.1f}', fontsize=9)
ax.text(1.3, q2, f'Q2={q2:.1f}', fontsize=9)
ax.text(1.3, q3, f'Q3={q3:.1f}', fontsize=9)

# 3. Q-Q plot for normality check
ax = axes[0, 2]
stats.probplot(data['C'], dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Count vs Normal Distribution')
ax.grid(alpha=0.3)

# 4. Histogram of year
ax = axes[1, 0]
ax.hist(data['year'], bins=20, edgecolor='black', alpha=0.7, color='coral')
ax.axvline(data['year'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={data["year"].mean():.3f}')
ax.set_xlabel('Standardized Year')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Predictor (year)')
ax.legend()
ax.grid(alpha=0.3)

# 5. Log-transformed counts
ax = axes[1, 1]
log_C = np.log(data['C'])
ax.hist(log_C, bins=20, edgecolor='black', alpha=0.7, color='mediumseagreen')
ax.axvline(log_C.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={log_C.mean():.3f}')
ax.axvline(log_C.median(), color='green', linestyle='--', linewidth=2, label=f'Median={log_C.median():.3f}')
ax.set_xlabel('log(C)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Log-Transformed Counts')
ax.legend()
ax.grid(alpha=0.3)

# 6. Q-Q plot for log-transformed counts
ax = axes[1, 2]
stats.probplot(log_C, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: log(C) vs Normal Distribution')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/01_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: 01_distribution_analysis.png")
plt.close()

# Normality tests
print("\n" + "="*80)
print("NORMALITY TESTS")
print("="*80)

# Test for C
shapiro_C = stats.shapiro(data['C'])
anderson_C = stats.anderson(data['C'], dist='norm')
print(f"\nCount variable (C):")
print(f"  Shapiro-Wilk test: W={shapiro_C.statistic:.4f}, p={shapiro_C.pvalue:.4f}")
print(f"  Conclusion: {'Reject normality' if shapiro_C.pvalue < 0.05 else 'Cannot reject normality'}")
print(f"  Anderson-Darling: statistic={anderson_C.statistic:.4f}")
print(f"  Critical values: {anderson_C.critical_values}")
print(f"  Significance levels: {anderson_C.significance_level}")

# Test for log(C)
shapiro_logC = stats.shapiro(log_C)
anderson_logC = stats.anderson(log_C, dist='norm')
print(f"\nLog-transformed counts log(C):")
print(f"  Shapiro-Wilk test: W={shapiro_logC.statistic:.4f}, p={shapiro_logC.pvalue:.4f}")
print(f"  Conclusion: {'Reject normality' if shapiro_logC.pvalue < 0.05 else 'Cannot reject normality'}")
print(f"  Anderson-Darling: statistic={anderson_logC.statistic:.4f}")

# Test for year
shapiro_year = stats.shapiro(data['year'])
print(f"\nPredictor (year):")
print(f"  Shapiro-Wilk test: W={shapiro_year.statistic:.4f}, p={shapiro_year.pvalue:.4f}")
print(f"  Conclusion: {'Reject normality' if shapiro_year.pvalue < 0.05 else 'Cannot reject normality'}")

# Skewness and kurtosis interpretation
print("\n" + "="*80)
print("DISTRIBUTION SHAPE INTERPRETATION")
print("="*80)
print(f"\nCount variable (C):")
print(f"  Skewness = {stats.skew(data['C']):.4f}")
print(f"    Interpretation: {'Right-skewed (positive)' if stats.skew(data['C']) > 0 else 'Left-skewed (negative)'}")
print(f"  Kurtosis = {stats.kurtosis(data['C']):.4f}")
print(f"    Interpretation: {'Platykurtic (flatter than normal)' if stats.kurtosis(data['C']) < 0 else 'Leptokurtic (heavier tails)'}")

print(f"\nLog-transformed counts log(C):")
print(f"  Skewness = {stats.skew(log_C):.4f}")
print(f"  Kurtosis = {stats.kurtosis(log_C):.4f}")
print(f"    Log transformation {'improved' if abs(stats.skew(log_C)) < abs(stats.skew(data['C'])) else 'did not improve'} symmetry")

print("\n" + "="*80)
