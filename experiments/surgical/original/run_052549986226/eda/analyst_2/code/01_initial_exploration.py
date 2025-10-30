"""
Initial Exploration: Group-level descriptive statistics and variance decomposition
Focus: Understanding the basic structure and variability patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Setup paths
BASE_DIR = Path("/workspace/eda/analyst_2")
VIZ_DIR = BASE_DIR / "visualizations"
DATA_PATH = Path("/workspace/data/data_analyst_2.csv")

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("INITIAL DATA EXPLORATION")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

# Basic statistics per group
print("\n" + "=" * 80)
print("GROUP-LEVEL STATISTICS")
print("=" * 80)
print(df.to_string(index=False))

# Calculate confidence intervals for each group (Wilson score interval)
def wilson_score_interval(successes, trials, confidence=0.95):
    """Calculate Wilson score confidence interval for binomial proportion"""
    if trials == 0:
        return 0, 0

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / trials

    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator

    return max(0, center - margin), min(1, center + margin)

# Add confidence intervals
ci_lower = []
ci_upper = []
for _, row in df.iterrows():
    lower, upper = wilson_score_interval(row['r_successes'], row['n_trials'])
    ci_lower.append(lower)
    ci_upper.append(upper)

df['ci_lower'] = ci_lower
df['ci_upper'] = ci_upper
df['ci_width'] = df['ci_upper'] - df['ci_lower']

print("\n" + "=" * 80)
print("GROUP STATISTICS WITH CONFIDENCE INTERVALS")
print("=" * 80)
print(df[['group', 'n_trials', 'r_successes', 'success_rate', 'ci_lower', 'ci_upper', 'ci_width']].to_string(index=False))

# Overall pooled statistics
total_trials = df['n_trials'].sum()
total_successes = df['r_successes'].sum()
pooled_rate = total_successes / total_trials

print("\n" + "=" * 80)
print("OVERALL POOLED STATISTICS")
print("=" * 80)
print(f"Total trials: {total_trials}")
print(f"Total successes: {total_successes}")
print(f"Pooled success rate: {pooled_rate:.6f}")

# Variability metrics
print("\n" + "=" * 80)
print("VARIABILITY METRICS")
print("=" * 80)
print(f"\nSuccess rate statistics:")
print(f"  Mean: {df['success_rate'].mean():.6f}")
print(f"  Median: {df['success_rate'].median():.6f}")
print(f"  Std Dev: {df['success_rate'].std():.6f}")
print(f"  Min: {df['success_rate'].min():.6f}")
print(f"  Max: {df['success_rate'].max():.6f}")
print(f"  Range: {df['success_rate'].max() - df['success_rate'].min():.6f}")
print(f"  Coefficient of Variation: {df['success_rate'].std() / df['success_rate'].mean():.6f}")

print(f"\nTrial count statistics:")
print(f"  Mean: {df['n_trials'].mean():.1f}")
print(f"  Median: {df['n_trials'].median():.1f}")
print(f"  Min: {df['n_trials'].min()}")
print(f"  Max: {df['n_trials'].max()}")
print(f"  Ratio (max/min): {df['n_trials'].max() / df['n_trials'].min():.1f}")

# Check for potential outliers in success rates
z_scores = np.abs(stats.zscore(df['success_rate']))
print(f"\nGroups with |z-score| > 2:")
outlier_groups = df[z_scores > 2]['group'].values
if len(outlier_groups) > 0:
    print(f"  Groups: {outlier_groups}")
else:
    print("  None detected")

# Save augmented dataframe for later use
df.to_csv(BASE_DIR / "code" / "group_data_with_ci.csv", index=False)
print(f"\n\nSaved augmented data to: {BASE_DIR / 'code' / 'group_data_with_ci.csv'}")
