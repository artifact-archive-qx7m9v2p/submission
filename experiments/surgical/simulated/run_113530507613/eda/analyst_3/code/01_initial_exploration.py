"""
Initial Data Exploration - Data Quality and Structure
Focus: Understanding data quality, binomial constraints, and basic characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/data/data_analyst_3.csv')

print("="*80)
print("INITIAL DATA EXPLORATION - ANALYST 3")
print("Focus: Data Quality & Model-Relevant Features")
print("="*80)

print("\n1. BASIC DATA STRUCTURE")
print("-" * 80)
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"\nData types:")
print(data.dtypes)
print(f"\nFirst few rows:")
print(data.head(10))
print(f"\nLast few rows:")
print(data.tail(5))

print("\n2. DATA QUALITY CHECKS")
print("-" * 80)

# Missing values
print("\nMissing values:")
print(data.isnull().sum())
print(f"Total missing: {data.isnull().sum().sum()}")

# Duplicates
print(f"\nDuplicate rows: {data.duplicated().sum()}")
print(f"Duplicate group_ids: {data['group_id'].duplicated().sum()}")

print("\n3. BASIC STATISTICS")
print("-" * 80)
print(data.describe())

print("\n4. CRITICAL BINOMIAL CONSTRAINT CHECKS")
print("-" * 80)

# Check r_successes <= n_trials (fundamental binomial constraint)
violations = data[data['r_successes'] > data['n_trials']]
print(f"Violations where r_successes > n_trials: {len(violations)}")
if len(violations) > 0:
    print("CRITICAL ISSUE - Violations found:")
    print(violations)

# Check for negative values
neg_trials = data[data['n_trials'] < 0]
neg_successes = data[data['r_successes'] < 0]
print(f"\nNegative n_trials: {len(neg_trials)}")
print(f"Negative r_successes: {len(neg_successes)}")

# Check for zero trials (undefined rate)
zero_trials = data[data['n_trials'] == 0]
print(f"\nZero n_trials: {len(zero_trials)}")
if len(zero_trials) > 0:
    print("Groups with zero trials:")
    print(zero_trials)

# Verify success_rate calculation
data['calculated_rate'] = data['r_successes'] / data['n_trials']
data['rate_discrepancy'] = np.abs(data['success_rate'] - data['calculated_rate'])
max_discrepancy = data['rate_discrepancy'].max()
print(f"\nMax discrepancy between provided and calculated success_rate: {max_discrepancy:.10f}")
if max_discrepancy > 1e-6:
    print("WARNING: Success rate calculation discrepancies found!")
    print(data[data['rate_discrepancy'] > 1e-6][['group_id', 'n_trials', 'r_successes', 'success_rate', 'calculated_rate']])

print("\n5. RANGE CHECKS")
print("-" * 80)
print(f"n_trials range: [{data['n_trials'].min()}, {data['n_trials'].max()}]")
print(f"r_successes range: [{data['r_successes'].min()}, {data['r_successes'].max()}]")
print(f"success_rate range: [{data['success_rate'].min():.6f}, {data['success_rate'].max():.6f}]")

# Check if success_rate is in [0, 1]
invalid_rates = data[(data['success_rate'] < 0) | (data['success_rate'] > 1)]
print(f"\nSuccess rates outside [0, 1]: {len(invalid_rates)}")
if len(invalid_rates) > 0:
    print("CRITICAL ISSUE - Invalid rates found:")
    print(invalid_rates)

print("\n6. SPARSITY AND DATA DENSITY")
print("-" * 80)
print(f"Total groups: {len(data)}")
print(f"Total trials: {data['n_trials'].sum()}")
print(f"Total successes: {data['r_successes'].sum()}")
print(f"Overall pooled success rate: {data['r_successes'].sum() / data['n_trials'].sum():.6f}")

# Small sample sizes
print(f"\nGroups with n_trials < 10: {(data['n_trials'] < 10).sum()}")
print(f"Groups with n_trials < 50: {(data['n_trials'] < 50).sum()}")
print(f"Groups with n_trials < 100: {(data['n_trials'] < 100).sum()}")

# Rare events
print(f"\nGroups with r_successes = 0: {(data['r_successes'] == 0).sum()}")
print(f"Groups with r_successes = 1: {(data['r_successes'] == 1).sum()}")
print(f"Groups with r_successes <= 5: {(data['r_successes'] <= 5).sum()}")

# Perfect success
print(f"\nGroups with success_rate = 1.0: {(data['success_rate'] == 1.0).sum()}")

print("\n7. DISTRIBUTION CHARACTERISTICS")
print("-" * 80)
print("\nn_trials distribution:")
print(f"  Mean: {data['n_trials'].mean():.2f}")
print(f"  Median: {data['n_trials'].median():.2f}")
print(f"  Std: {data['n_trials'].std():.2f}")
print(f"  CV: {data['n_trials'].std() / data['n_trials'].mean():.3f}")
print(f"  Skewness: {data['n_trials'].skew():.3f}")

print("\nr_successes distribution:")
print(f"  Mean: {data['r_successes'].mean():.2f}")
print(f"  Median: {data['r_successes'].median():.2f}")
print(f"  Std: {data['r_successes'].std():.2f}")
print(f"  CV: {data['r_successes'].std() / data['r_successes'].mean():.3f}")
print(f"  Skewness: {data['r_successes'].skew():.3f}")

print("\nsuccess_rate distribution:")
print(f"  Mean: {data['success_rate'].mean():.6f}")
print(f"  Median: {data['success_rate'].median():.6f}")
print(f"  Std: {data['success_rate'].std():.6f}")
print(f"  CV: {data['success_rate'].std() / data['success_rate'].mean():.3f}")
print(f"  Skewness: {data['success_rate'].skew():.3f}")

print("\n8. QUANTILES FOR PRIOR SPECIFICATION")
print("-" * 80)
print("\nsuccess_rate quantiles:")
quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
for q in quantiles:
    print(f"  {q:4.2f}: {data['success_rate'].quantile(q):.6f}")

print("\n" + "="*80)
print("INITIAL EXPLORATION COMPLETE")
print("="*80)
