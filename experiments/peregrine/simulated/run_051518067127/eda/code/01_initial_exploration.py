"""
Initial Data Exploration: Count Time Series
============================================
Goal: Understand basic structure, quality, and distributions
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
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Load data
data = pd.read_csv('/workspace/data/data.csv')

print("="*80)
print("INITIAL DATA EXPLORATION")
print("="*80)

# Basic structure
print("\n1. DATA STRUCTURE")
print("-" * 80)
print(f"Shape: {data.shape}")
print(f"\nColumn types:\n{data.dtypes}")
print(f"\nFirst few rows:\n{data.head(10)}")
print(f"\nLast few rows:\n{data.tail(5)}")

# Missing values
print("\n2. DATA QUALITY")
print("-" * 80)
print(f"Missing values:\n{data.isnull().sum()}")
print(f"\nDuplicate rows: {data.duplicated().sum()}")

# Descriptive statistics
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 80)
print("\nSummary statistics:")
print(data.describe())

print("\nAdditional statistics:")
for col in data.columns:
    print(f"\n{col}:")
    print(f"  Range: [{data[col].min():.4f}, {data[col].max():.4f}]")
    print(f"  IQR: {data[col].quantile(0.75) - data[col].quantile(0.25):.4f}")
    print(f"  Skewness: {stats.skew(data[col]):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(data[col]):.4f}")
    print(f"  CV (Coefficient of Variation): {data[col].std() / data[col].mean():.4f}")

# Count-specific properties
print("\n4. COUNT DATA PROPERTIES")
print("-" * 80)
print(f"Mean of C: {data['C'].mean():.4f}")
print(f"Variance of C: {data['C'].var():.4f}")
print(f"Variance-to-Mean Ratio: {data['C'].var() / data['C'].mean():.4f}")
print(f"  (Ratio > 1 suggests overdispersion)")
print(f"\nZero counts: {(data['C'] == 0).sum()}")
print(f"Minimum count: {data['C'].min()}")
print(f"Maximum count: {data['C'].max()}")

# Percentiles
print("\nPercentiles of C:")
percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
for p in percentiles:
    print(f"  {p:3d}%: {np.percentile(data['C'], p):6.1f}")

# Check year standardization
print("\n5. PREDICTOR STANDARDIZATION CHECK")
print("-" * 80)
print(f"Mean of year: {data['year'].mean():.6f} (should be ~0)")
print(f"Std of year: {data['year'].std():.6f} (should be ~1)")
print(f"Std (ddof=1): {data['year'].std(ddof=1):.6f}")

# Reconstruct original years if possible
# year = (original - mean) / std
# If std uses n-1, we can estimate
year_std_corrected = data['year'].std(ddof=1)
year_spacing = np.diff(data['year']).mean()
print(f"\nMean spacing between consecutive year values: {year_spacing:.6f}")
print(f"This suggests original years were evenly spaced")

# Save to file
with open('/workspace/eda/initial_summary.txt', 'w') as f:
    f.write("INITIAL DATA SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Shape: {data.shape}\n")
    f.write(f"Columns: {list(data.columns)}\n")
    f.write(f"Missing values: {data.isnull().sum().sum()}\n\n")
    f.write("Summary Statistics:\n")
    f.write(str(data.describe()) + "\n\n")
    f.write(f"Variance-to-Mean Ratio for C: {data['C'].var() / data['C'].mean():.4f}\n")

print("\n" + "="*80)
print("Initial exploration complete. Summary saved to initial_summary.txt")
print("="*80)
