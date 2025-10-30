"""
Initial Data Exploration - Distributional Properties and Outlier Detection
Focus: Understanding the structure and quality of the binomial outcome data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

print("="*80)
print("INITIAL DATA EXPLORATION")
print("="*80)

# Basic structure
print("\n1. DATA STRUCTURE")
print("-"*40)
print(f"Shape: {data.shape}")
print(f"\nColumn types:\n{data.dtypes}")
print(f"\nFirst few rows:\n{data.head()}")
print(f"\nLast few rows:\n{data.tail()}")

# Data quality checks
print("\n2. DATA QUALITY")
print("-"*40)
print(f"Missing values:\n{data.isnull().sum()}")
print(f"\nDuplicate rows: {data.duplicated().sum()}")

# Verify data consistency
print("\n3. DATA CONSISTENCY CHECKS")
print("-"*40)
print("Checking if proportion = r/n:")
calculated_prop = data['r'] / data['n']
prop_match = np.allclose(data['proportion'], calculated_prop)
print(f"  Proportions match: {prop_match}")

print("\nChecking if failures = n - r:")
calculated_failures = data['n'] - data['r']
failures_match = (data['failures'] == calculated_failures).all()
print(f"  Failures match: {failures_match}")

print("\nChecking for impossible values:")
print(f"  Any r > n: {(data['r'] > data['n']).any()}")
print(f"  Any r < 0: {(data['r'] < 0).any()}")
print(f"  Any n <= 0: {(data['n'] <= 0).any()}")

# Summary statistics
print("\n4. SUMMARY STATISTICS")
print("-"*40)
print("\nSample sizes (n):")
print(data['n'].describe())

print("\nNumber of events (r):")
print(data['r'].describe())

print("\nObserved proportions:")
print(data['proportion'].describe())

# Total sample size
print(f"\nTotal sample size: {data['n'].sum()}")
print(f"Total events: {data['r'].sum()}")
print(f"Overall proportion: {data['r'].sum() / data['n'].sum():.4f}")

# Range of values
print("\n5. RANGES AND EXTREMES")
print("-"*40)
print(f"Sample size range: {data['n'].min()} to {data['n'].max()}")
print(f"  Ratio (max/min): {data['n'].max() / data['n'].min():.2f}x")

print(f"\nProportion range: {data['proportion'].min():.4f} to {data['proportion'].max():.4f}")
print(f"  Difference: {data['proportion'].max() - data['proportion'].min():.4f}")

# Identify extremes
print("\n6. EXTREME GROUPS")
print("-"*40)
print("Groups with smallest sample sizes:")
print(data.nsmallest(3, 'n')[['group', 'n', 'r', 'proportion']])

print("\nGroups with largest sample sizes:")
print(data.nlargest(3, 'n')[['group', 'n', 'r', 'proportion']])

print("\nGroups with lowest proportions:")
print(data.nsmallest(3, 'proportion')[['group', 'n', 'r', 'proportion']])

print("\nGroups with highest proportions:")
print(data.nlargest(3, 'proportion')[['group', 'n', 'r', 'proportion']])

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
