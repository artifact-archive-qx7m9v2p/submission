"""
Initial Data Exploration and Quality Assessment
Focus: Data quality, completeness, basic statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data_analyst_3.csv'
OUTPUT_DIR = BASE_DIR / 'eda' / 'analyst_3'
VIZ_DIR = OUTPUT_DIR / 'visualizations'
CODE_DIR = OUTPUT_DIR / 'code'

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
df = pd.read_csv(DATA_PATH)

print("="*80)
print("DATA QUALITY ASSESSMENT")
print("="*80)

print("\n1. BASIC DATA STRUCTURE")
print("-" * 80)
print(f"Shape: {df.shape}")
print(f"\nColumn types:")
print(df.dtypes)
print(f"\nFirst few rows:")
print(df.head())
print(f"\nLast few rows:")
print(df.tail())

print("\n2. MISSING DATA CHECK")
print("-" * 80)
print("Missing values per column:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")
print(f"Percentage missing: {100 * df.isnull().sum().sum() / df.size:.2f}%")

print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 80)
print(df.describe())

print("\n4. DATA CONSISTENCY CHECKS")
print("-" * 80)

# Check if success_rate matches r_successes/n_trials
df['calculated_rate'] = df['r_successes'] / df['n_trials']
df['rate_diff'] = abs(df['success_rate'] - df['calculated_rate'])
max_diff = df['rate_diff'].max()
print(f"Maximum difference between recorded and calculated success_rate: {max_diff}")

if max_diff > 1e-10:
    print("WARNING: Inconsistencies found!")
    print(df[df['rate_diff'] > 1e-10][['group', 'n_trials', 'r_successes', 'success_rate', 'calculated_rate']])
else:
    print("SUCCESS: All success_rates match r_successes/n_trials")

# Check if r_successes <= n_trials
invalid_counts = df[df['r_successes'] > df['n_trials']]
print(f"\nRecords where r_successes > n_trials: {len(invalid_counts)}")
if len(invalid_counts) > 0:
    print("WARNING: Invalid data found!")
    print(invalid_counts)

# Check if r_successes is non-negative
negative_successes = df[df['r_successes'] < 0]
print(f"\nRecords with negative r_successes: {len(negative_successes)}")
if len(negative_successes) > 0:
    print("WARNING: Negative successes found!")
    print(negative_successes)

# Check if n_trials is positive
non_positive_trials = df[df['n_trials'] <= 0]
print(f"\nRecords with non-positive n_trials: {len(non_positive_trials)}")
if len(non_positive_trials) > 0:
    print("WARNING: Non-positive trial counts found!")
    print(non_positive_trials)

# Check for duplicate groups
duplicate_groups = df[df.duplicated(subset=['group'], keep=False)]
print(f"\nDuplicate groups: {len(duplicate_groups)}")
if len(duplicate_groups) > 0:
    print("WARNING: Duplicate groups found!")
    print(duplicate_groups)

# Check group numbering
expected_groups = set(range(1, df['group'].max() + 1))
actual_groups = set(df['group'].unique())
missing_groups = expected_groups - actual_groups
extra_groups = actual_groups - expected_groups
print(f"\nExpected groups (1 to {df['group'].max()}): {sorted(expected_groups)}")
print(f"Actual groups: {sorted(actual_groups)}")
if missing_groups:
    print(f"WARNING: Missing groups: {sorted(missing_groups)}")
if extra_groups:
    print(f"WARNING: Extra/unexpected groups: {sorted(extra_groups)}")

print("\n5. EXTREME VALUES CHECK")
print("-" * 80)

# Use IQR method for outlier detection
for col in ['n_trials', 'r_successes', 'success_rate']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n{col}:")
    print(f"  Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f}")
    print(f"  Lower bound={lower_bound:.4f}, Upper bound={upper_bound:.4f}")
    print(f"  Number of outliers (IQR method): {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Outlier groups: {sorted(outliers['group'].values)}")

print("\n6. SAMPLE SIZE DISTRIBUTION")
print("-" * 80)
print(f"Minimum n_trials: {df['n_trials'].min()}")
print(f"Maximum n_trials: {df['n_trials'].max()}")
print(f"Mean n_trials: {df['n_trials'].mean():.2f}")
print(f"Median n_trials: {df['n_trials'].median():.2f}")
print(f"Std n_trials: {df['n_trials'].std():.2f}")
print(f"CV (coefficient of variation): {df['n_trials'].std() / df['n_trials'].mean():.2f}")

# Check for very small samples (< 30 often considered small)
small_samples = df[df['n_trials'] < 30]
print(f"\nGroups with n_trials < 30: {len(small_samples)}")
if len(small_samples) > 0:
    print(small_samples[['group', 'n_trials', 'r_successes', 'success_rate']])

# Check for extreme success rates (0 or 1)
extreme_rates = df[(df['success_rate'] == 0) | (df['success_rate'] == 1)]
print(f"\nGroups with extreme success rates (0 or 1): {len(extreme_rates)}")
if len(extreme_rates) > 0:
    print(extreme_rates[['group', 'n_trials', 'r_successes', 'success_rate']])

print("\n" + "="*80)
print("INITIAL DATA QUALITY SUMMARY")
print("="*80)
print(f"Total records: {len(df)}")
print(f"Complete cases: {df.dropna().shape[0]}")
print(f"Data consistency: {'PASS' if max_diff < 1e-10 else 'FAIL'}")
print(f"Logical validity: {'PASS' if len(invalid_counts) == 0 and len(negative_successes) == 0 and len(non_positive_trials) == 0 else 'FAIL'}")

# Save cleaned dataframe for further analysis
df.to_csv(OUTPUT_DIR / 'code' / 'data_with_checks.csv', index=False)
print(f"\nData saved to: {OUTPUT_DIR / 'code' / 'data_with_checks.csv'}")
