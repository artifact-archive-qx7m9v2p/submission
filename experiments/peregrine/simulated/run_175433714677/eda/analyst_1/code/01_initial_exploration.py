"""
Initial Exploration: Time Series & Temporal Patterns
Focus: Understanding basic temporal structure and growth patterns
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import UnivariateSpline

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Load data
with open('/workspace/data/data_analyst_1.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({
    'year': data['year'],
    'C': data['C']
})

print("="*60)
print("INITIAL DATA EXPLORATION")
print("="*60)
print(f"\nDataset shape: {df.shape}")
print(f"\nBasic statistics:")
print(df.describe())

print(f"\nData range:")
print(f"Year: [{df['year'].min():.3f}, {df['year'].max():.3f}]")
print(f"Count C: [{df['C'].min()}, {df['C'].max()}]")

print(f"\nChange in counts:")
print(f"First 5 observations mean: {df['C'].iloc[:5].mean():.2f}")
print(f"Last 5 observations mean: {df['C'].iloc[-5:].mean():.2f}")
print(f"Ratio (end/start): {df['C'].iloc[-5:].mean() / df['C'].iloc[:5:].mean():.2f}x")

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Basic time series plot
ax1 = axes[0, 0]
ax1.plot(df['year'], df['C'], 'o-', linewidth=2, markersize=6, alpha=0.7)
ax1.set_xlabel('Year (standardized)', fontsize=11)
ax1.set_ylabel('Count (C)', fontsize=11)
ax1.set_title('A. Time Series: Count vs Year', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Log-scale plot to assess exponential growth
ax2 = axes[0, 1]
ax2.semilogy(df['year'], df['C'], 'o-', linewidth=2, markersize=6, alpha=0.7, color='darkgreen')
ax2.set_xlabel('Year (standardized)', fontsize=11)
ax2.set_ylabel('Count (C) - Log Scale', fontsize=11)
ax2.set_title('B. Log-Scale View (Exponential Growth Test)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Distribution of counts
ax3 = axes[1, 0]
ax3.hist(df['C'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(df['C'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["C"].mean():.1f}')
ax3.axvline(df['C'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["C"].median():.1f}')
ax3.set_xlabel('Count (C)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('C. Distribution of Counts', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. First differences (changes between consecutive observations)
df['diff'] = df['C'].diff()
ax4 = axes[1, 1]
ax4.plot(df['year'].iloc[1:], df['diff'].iloc[1:], 'o-', linewidth=2, markersize=6, alpha=0.7, color='crimson')
ax4.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax4.axhline(df['diff'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {df["diff"].mean():.2f}')
ax4.set_xlabel('Year (standardized)', fontsize=11)
ax4.set_ylabel('First Difference (ΔC)', fontsize=11)
ax4.set_title('D. First Differences (Rate of Change)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/01_initial_exploration.png', dpi=300, bbox_inches='tight')
print("\nSaved: 01_initial_exploration.png")

# Calculate basic temporal statistics
print("\n" + "="*60)
print("TEMPORAL PATTERNS")
print("="*60)

# Linear correlation
corr_pearson = stats.pearsonr(df['year'], df['C'])
corr_spearman = stats.spearmanr(df['year'], df['C'])
print(f"\nPearson correlation (year, C): r={corr_pearson[0]:.4f}, p={corr_pearson[1]:.2e}")
print(f"Spearman correlation (year, C): rho={corr_spearman[0]:.4f}, p={corr_spearman[1]:.2e}")

# First differences analysis
print(f"\nFirst differences statistics:")
print(f"Mean change per step: {df['diff'].mean():.2f}")
print(f"Std of changes: {df['diff'].std():.2f}")
print(f"Min change: {df['diff'].min():.0f}")
print(f"Max change: {df['diff'].max():.0f}")

# Check if changes are increasing (accelerating growth)
df['diff_of_diff'] = df['diff'].diff()
print(f"\nSecond derivative (acceleration):")
print(f"Mean: {df['diff_of_diff'].mean():.2f}")
print(f"Positive changes: {(df['diff_of_diff'] > 0).sum()} / {len(df['diff_of_diff']) - 1}")

plt.close()
