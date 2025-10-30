"""
Initial Data Exploration - Analyst 2
Focus: Patterns, structure, and relationships in binomial outcome data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("=" * 80)
print("INITIAL DATA EXPLORATION")
print("=" * 80)

print("\n1. DATA STRUCTURE")
print("-" * 80)
print(f"Shape: {data.shape}")
print(f"\nColumns: {list(data.columns)}")
print(f"\nData types:\n{data.dtypes}")

print("\n2. DATA OVERVIEW")
print("-" * 80)
print(data)

print("\n3. BASIC STATISTICS")
print("-" * 80)
print(data.describe())

print("\n4. DATA QUALITY CHECKS")
print("-" * 80)
print(f"Missing values:\n{data.isnull().sum()}")
print(f"\nDuplicate rows: {data.duplicated().sum()}")

# Verify calculations
print("\n5. CALCULATION VERIFICATION")
print("-" * 80)
data['calc_proportion'] = data['r'] / data['n']
data['calc_failures'] = data['n'] - data['r']
print(f"Proportion calculation correct: {np.allclose(data['proportion'], data['calc_proportion'])}")
print(f"Failures calculation correct: {np.allclose(data['failures'], data['calc_failures'])}")

print("\n6. SAMPLE SIZE DISTRIBUTION")
print("-" * 80)
print(f"Total observations: {data['n'].sum()}")
print(f"Total events: {data['r'].sum()}")
print(f"Overall proportion: {data['r'].sum() / data['n'].sum():.4f}")
print(f"\nSample size range: [{data['n'].min()}, {data['n'].max()}]")
print(f"Sample size CV: {data['n'].std() / data['n'].mean():.3f}")

print("\n7. PROPORTION DISTRIBUTION")
print("-" * 80)
print(f"Proportion range: [{data['proportion'].min():.4f}, {data['proportion'].max():.4f}]")
print(f"Proportion mean: {data['proportion'].mean():.4f}")
print(f"Proportion median: {data['proportion'].median():.4f}")
print(f"Proportion std: {data['proportion'].std():.4f}")
print(f"Proportion CV: {data['proportion'].std() / data['proportion'].mean():.3f}")

print("\n8. ZERO/RARE EVENT ANALYSIS")
print("-" * 80)
zero_groups = data[data['r'] == 0]
print(f"Groups with zero events: {len(zero_groups)}")
if len(zero_groups) > 0:
    print(f"Zero event groups: {zero_groups['group'].tolist()}")
    print(f"Sample sizes for zero groups: {zero_groups['n'].tolist()}")

low_count = data[data['r'] <= 5]
print(f"\nGroups with ≤5 events: {len(low_count)}")
print(low_count[['group', 'n', 'r', 'proportion']])

print("\n9. HETEROGENEITY METRICS")
print("-" * 80)
# Calculate weighted mean
weighted_mean = data['r'].sum() / data['n'].sum()
print(f"Weighted (pooled) proportion: {weighted_mean:.4f}")
print(f"Unweighted mean proportion: {data['proportion'].mean():.4f}")
print(f"Difference: {abs(weighted_mean - data['proportion'].mean()):.4f}")

# Range
prop_range = data['proportion'].max() - data['proportion'].min()
print(f"\nProportion range: {prop_range:.4f}")
print(f"Ratio (max/min non-zero): {data[data['r'] > 0]['proportion'].max() / data[data['r'] > 0]['proportion'].min():.2f}")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
