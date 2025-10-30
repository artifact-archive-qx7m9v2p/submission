"""
Initial Data Exploration - Analyst 1
=====================================
Purpose: Load data and examine basic characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

print("="*60)
print("INITIAL DATA EXPLORATION")
print("="*60)

# Basic info
print("\n1. DATA STRUCTURE")
print("-"*60)
print(f"Number of observations: {len(data)}")
print(f"Number of variables: {len(data.columns)}")
print(f"\nColumn names: {data.columns.tolist()}")
print(f"\nData types:\n{data.dtypes}")

# Check for missing values
print("\n2. MISSING VALUES")
print("-"*60)
print(data.isnull().sum())
print(f"Total missing: {data.isnull().sum().sum()}")

# Check for duplicates
print("\n3. DUPLICATE ROWS")
print("-"*60)
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
if duplicates > 0:
    print("\nDuplicate rows:")
    print(data[data.duplicated(keep=False)].sort_values(by=['x', 'Y']))

# Summary statistics
print("\n4. SUMMARY STATISTICS")
print("-"*60)
print(data.describe())

# Additional statistics
print("\n5. ADDITIONAL STATISTICS")
print("-"*60)
for col in ['Y', 'x']:
    print(f"\n{col}:")
    print(f"  Range: [{data[col].min():.3f}, {data[col].max():.3f}]")
    print(f"  IQR: {data[col].quantile(0.75) - data[col].quantile(0.25):.3f}")
    print(f"  Variance: {data[col].var():.6f}")
    print(f"  Skewness: {stats.skew(data[col]):.3f}")
    print(f"  Kurtosis: {stats.kurtosis(data[col]):.3f}")

# Display first and last rows
print("\n6. DATA PREVIEW")
print("-"*60)
print("\nFirst 5 rows:")
print(data.head())
print("\nLast 5 rows:")
print(data.tail())

# Check spacing of x values
print("\n7. X-VALUE DISTRIBUTION")
print("-"*60)
x_unique = data['x'].unique()
print(f"Unique x values: {len(x_unique)}")
print(f"X values: {sorted(x_unique)}")
print(f"\nObservations per x value:")
print(data['x'].value_counts().sort_index())

# Compute differences between consecutive x values
x_sorted = np.sort(x_unique)
x_diffs = np.diff(x_sorted)
print(f"\nSpacing between consecutive x values:")
print(f"  Min: {x_diffs.min():.1f}")
print(f"  Max: {x_diffs.max():.1f}")
print(f"  Mean: {x_diffs.mean():.2f}")
print(f"  Median: {np.median(x_diffs):.2f}")

print("\n" + "="*60)
print("EXPLORATION COMPLETE")
print("="*60)
