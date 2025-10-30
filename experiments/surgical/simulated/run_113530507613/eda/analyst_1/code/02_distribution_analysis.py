"""
Distribution Analysis - Success Rates and Sample Sizes
Focus: Understanding the distributional characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Create comprehensive distribution plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution Analysis: Success Rates and Sample Sizes', fontsize=16, fontweight='bold')

# 1. Histogram of success rates
ax = axes[0, 0]
ax.hist(data['success_rate'], bins=8, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(data['success_rate'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data["success_rate"].mean():.4f}')
ax.axvline(data['success_rate'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data["success_rate"].median():.4f}')
ax.set_xlabel('Success Rate', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Histogram of Success Rates', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2. Box plot of success rates
ax = axes[0, 1]
bp = ax.boxplot(data['success_rate'], vert=True, patch_artist=True,
                showmeans=True, meanline=False,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
ax.set_ylabel('Success Rate', fontsize=11)
ax.set_title('Box Plot of Success Rates', fontsize=12, fontweight='bold')
ax.set_xticklabels(['All Groups'])
ax.grid(alpha=0.3)

# Calculate outliers using IQR method
Q1 = data['success_rate'].quantile(0.25)
Q3 = data['success_rate'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data[(data['success_rate'] < lower_bound) | (data['success_rate'] > upper_bound)]
ax.text(0.5, 0.98, f'Outliers (IQR): {len(outliers)}', transform=ax.transAxes,
        ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Q-Q plot for normality
ax = axes[0, 2]
stats.probplot(data['success_rate'], dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Success Rates vs Normal', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# 4. Histogram of n_trials
ax = axes[1, 0]
ax.hist(data['n_trials'], bins=8, edgecolor='black', alpha=0.7, color='coral')
ax.axvline(data['n_trials'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data["n_trials"].mean():.1f}')
ax.axvline(data['n_trials'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data["n_trials"].median():.1f}')
ax.set_xlabel('Number of Trials', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Histogram of Sample Sizes', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 5. Box plot of n_trials
ax = axes[1, 1]
bp = ax.boxplot(data['n_trials'], vert=True, patch_artist=True,
                showmeans=True, meanline=False,
                boxprops=dict(facecolor='lightsalmon', alpha=0.7),
                meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
ax.set_ylabel('Number of Trials', fontsize=11)
ax.set_title('Box Plot of Sample Sizes', fontsize=12, fontweight='bold')
ax.set_xticklabels(['All Groups'])
ax.grid(alpha=0.3)

# Calculate outliers for n_trials
Q1_n = data['n_trials'].quantile(0.25)
Q3_n = data['n_trials'].quantile(0.75)
IQR_n = Q3_n - Q1_n
lower_bound_n = Q1_n - 1.5 * IQR_n
upper_bound_n = Q3_n + 1.5 * IQR_n
outliers_n = data[(data['n_trials'] < lower_bound_n) | (data['n_trials'] > upper_bound_n)]
ax.text(0.5, 0.98, f'Outliers (IQR): {len(outliers_n)}', transform=ax.transAxes,
        ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6. Kernel Density Estimate of success rates
ax = axes[1, 2]
data['success_rate'].plot(kind='kde', ax=ax, linewidth=2, color='steelblue')
ax.axvline(data['success_rate'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax.axvline(data['success_rate'].median(), color='green', linestyle='--', linewidth=2, label='Median')
ax.set_xlabel('Success Rate', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Kernel Density Estimate', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/01_distribution_overview.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/analyst_1/visualizations/01_distribution_overview.png")

# Print detailed findings
print("\n" + "="*80)
print("DISTRIBUTION ANALYSIS FINDINGS")
print("="*80)

print("\n1. SUCCESS RATE OUTLIERS (IQR Method)")
if len(outliers) > 0:
    print(f"Found {len(outliers)} outlier(s):")
    print(outliers[['group_id', 'success_rate', 'n_trials', 'r_successes']])
else:
    print("No outliers detected using IQR method (1.5 * IQR threshold)")

print("\n2. SAMPLE SIZE OUTLIERS (IQR Method)")
if len(outliers_n) > 0:
    print(f"Found {len(outliers_n)} outlier(s):")
    print(outliers_n[['group_id', 'n_trials', 'r_successes', 'success_rate']])
else:
    print("No outliers detected")

print("\n3. NORMALITY TEST FOR SUCCESS RATES")
shapiro_stat, shapiro_p = stats.shapiro(data['success_rate'])
print(f"Shapiro-Wilk test: statistic={shapiro_stat:.6f}, p-value={shapiro_p:.6f}")
if shapiro_p > 0.05:
    print("   -> Cannot reject normality (p > 0.05)")
else:
    print("   -> Reject normality (p < 0.05)")

print("\n4. DISTRIBUTION SHAPE METRICS")
print(f"Success rate skewness: {data['success_rate'].skew():.4f}")
print(f"   -> Interpretation: {'Right-skewed' if data['success_rate'].skew() > 0 else 'Left-skewed'}")
print(f"Success rate kurtosis: {data['success_rate'].kurtosis():.4f}")
print(f"   -> Interpretation: {'Heavier tails than normal' if data['success_rate'].kurtosis() > 0 else 'Lighter tails than normal'}")

print("\n5. GROUPS AT EXTREMES")
print(f"\nLowest success rates:")
print(data.nsmallest(3, 'success_rate')[['group_id', 'success_rate', 'n_trials', 'r_successes']])
print(f"\nHighest success rates:")
print(data.nlargest(3, 'success_rate')[['group_id', 'success_rate', 'n_trials', 'r_successes']])

print("\n" + "="*80)
