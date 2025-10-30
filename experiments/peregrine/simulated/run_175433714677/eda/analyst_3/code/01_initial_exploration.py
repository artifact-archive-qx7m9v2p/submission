"""
Initial Exploration: Data Overview and Basic Patterns
Focus: Understanding the raw data structure before model fitting
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
with open('/workspace/data/data_analyst_3.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({'year': data['year'], 'C': data['C']})
print("Dataset Overview:")
print(df.describe())
print(f"\nData shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Basic statistics
print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print(f"Count range: [{df['C'].min()}, {df['C'].max()}]")
print(f"Year range: [{df['year'].min():.3f}, {df['year'].max():.3f}]")
print(f"Mean count: {df['C'].mean():.2f}")
print(f"Std count: {df['C'].std():.2f}")
print(f"CV (coefficient of variation): {df['C'].std()/df['C'].mean():.3f}")

# Check for overdispersion (variance > mean for count data)
print(f"\nOverdispersion check (count data):")
print(f"  Mean: {df['C'].mean():.2f}")
print(f"  Variance: {df['C'].var():.2f}")
print(f"  Var/Mean ratio: {df['C'].var()/df['C'].mean():.2f}")
print(f"  {'OVERDISPERSED' if df['C'].var() > df['C'].mean() else 'Not overdispersed'}")

# Visual exploration
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Time series plot
axes[0, 0].plot(df['year'], df['C'], 'o-', alpha=0.6, linewidth=1.5)
axes[0, 0].set_xlabel('Year (standardized)')
axes[0, 0].set_ylabel('Count (C)')
axes[0, 0].set_title('A. Time Series: Count vs Year')
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribution of counts
axes[0, 1].hist(df['C'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Count (C)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('B. Distribution of Counts')
axes[0, 1].axvline(df['C'].mean(), color='red', linestyle='--', label=f'Mean: {df['C'].mean():.1f}')
axes[0, 1].axvline(df['C'].median(), color='orange', linestyle='--', label=f'Median: {df['C'].median():.1f}')
axes[0, 1].legend()

# 3. Log-scale plot (to check for exponential growth)
axes[1, 0].plot(df['year'], df['C'], 'o-', alpha=0.6, linewidth=1.5)
axes[1, 0].set_xlabel('Year (standardized)')
axes[1, 0].set_ylabel('Count (C)')
axes[1, 0].set_yscale('log')
axes[1, 0].set_title('C. Log-Scale: Count vs Year')
axes[1, 0].grid(True, alpha=0.3)

# 4. Rate of change
diff_C = np.diff(df['C'])
diff_year = np.diff(df['year'])
rate_of_change = diff_C / diff_year
year_midpoints = (df['year'].values[:-1] + df['year'].values[1:]) / 2

axes[1, 1].plot(year_midpoints, rate_of_change, 'o-', alpha=0.6, linewidth=1.5)
axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('Year (standardized)')
axes[1, 1].set_ylabel('Rate of Change (dC/dyear)')
axes[1, 1].set_title('D. Rate of Change Over Time')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/01_initial_exploration.png', dpi=150, bbox_inches='tight')
print("\n[SAVED] Initial exploration plot: visualizations/01_initial_exploration.png")
plt.close()

# Check for structural breaks in the data
print("\n" + "="*60)
print("STRUCTURAL ANALYSIS")
print("="*60)

# Split data into early and late periods
mid_point = len(df) // 2
early = df.iloc[:mid_point]
late = df.iloc[mid_point:]

print(f"\nEarly period (n={len(early)}):")
print(f"  Mean: {early['C'].mean():.2f}")
print(f"  Std: {early['C'].std():.2f}")
print(f"  Range: [{early['C'].min()}, {early['C'].max()}]")

print(f"\nLate period (n={len(late)}):")
print(f"  Mean: {late['C'].mean():.2f}")
print(f"  Std: {late['C'].std():.2f}")
print(f"  Range: [{late['C'].min()}, {late['C'].max()}]")

print(f"\nMean ratio (late/early): {late['C'].mean()/early['C'].mean():.2f}x")

# Correlation
correlation = df['year'].corr(df['C'])
print(f"\nPearson correlation (year, C): {correlation:.4f}")

# Log-transform correlation (for exponential growth hypothesis)
log_correlation = df['year'].corr(np.log(df['C']))
print(f"Pearson correlation (year, log(C)): {log_correlation:.4f}")

print("\n" + "="*60)
print("INITIAL OBSERVATIONS")
print("="*60)
print("""
Key patterns to investigate:
1. Strong increasing trend visible in raw data
2. Potential exponential growth (note log-scale linearity)
3. Significant overdispersion suggests Negative Binomial may be better than Poisson
4. Rate of change appears to accelerate over time
5. Late period shows much higher variance than early period
""")
