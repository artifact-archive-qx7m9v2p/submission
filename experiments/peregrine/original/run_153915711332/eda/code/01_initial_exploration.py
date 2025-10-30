"""
Initial Data Exploration and Validation
========================================
Purpose: Load data, validate structure, compute descriptive statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set styling
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/data/data.csv')

print("="*80)
print("INITIAL DATA EXPLORATION")
print("="*80)

# Basic structure
print("\n1. DATA STRUCTURE")
print("-"*80)
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"\nData types:\n{data.dtypes}")

# Missing values
print("\n2. DATA QUALITY")
print("-"*80)
print(f"Missing values:\n{data.isnull().sum()}")
print(f"\nDuplicate rows: {data.duplicated().sum()}")

# Preview
print("\n3. DATA PREVIEW")
print("-"*80)
print("\nFirst 5 rows:")
print(data.head())
print("\nLast 5 rows:")
print(data.tail())

# Descriptive statistics
print("\n4. DESCRIPTIVE STATISTICS")
print("-"*80)
print("\nNumeric summary:")
print(data.describe())

# Additional statistics for counts
print("\n5. COUNT VARIABLE (C) - DETAILED STATISTICS")
print("-"*80)
C = data['C'].values
print(f"Mean: {np.mean(C):.2f}")
print(f"Median: {np.median(C):.2f}")
print(f"Mode: {stats.mode(C, keepdims=True).mode[0]}")
print(f"Std Dev: {np.std(C, ddof=1):.2f}")
print(f"Variance: {np.var(C, ddof=1):.2f}")
print(f"Variance-to-Mean Ratio: {np.var(C, ddof=1) / np.mean(C):.2f}")
print(f"  (>1 suggests overdispersion for Poisson)")
print(f"\nMin: {np.min(C)}")
print(f"Max: {np.max(C)}")
print(f"Range: {np.max(C) - np.min(C)}")
print(f"IQR: {np.percentile(C, 75) - np.percentile(C, 25):.2f}")
print(f"\nSkewness: {stats.skew(C):.3f}")
print(f"Kurtosis (excess): {stats.kurtosis(C):.3f}")

# Check if counts are integers
print(f"\nAll counts are integers: {all(C == C.astype(int))}")
print(f"Any negative counts: {any(C < 0)}")
print(f"Any zero counts: {any(C == 0)}")

# Time variable statistics
print("\n6. TIME VARIABLE (year) - STATISTICS")
print("-"*80)
year = data['year'].values
print(f"Mean: {np.mean(year):.6f}")
print(f"Median: {np.median(year):.6f}")
print(f"Std Dev: {np.std(year, ddof=1):.6f}")
print(f"Min: {np.min(year):.6f}")
print(f"Max: {np.max(year):.6f}")
print(f"Range: {np.max(year) - np.min(year):.6f}")

# Check if evenly spaced
diffs = np.diff(year)
print(f"\nTime spacing:")
print(f"  Mean diff: {np.mean(diffs):.6f}")
print(f"  Std diff: {np.std(diffs):.6f}")
print(f"  All equal spacing: {np.allclose(diffs, diffs[0])}")

# Growth metrics
print("\n7. GROWTH CHARACTERISTICS")
print("-"*80)
initial_count = C[0]
final_count = C[-1]
total_growth = final_count - initial_count
pct_growth = (final_count - initial_count) / initial_count * 100

print(f"Initial count (year={year[0]:.2f}): {initial_count}")
print(f"Final count (year={year[-1]:.2f}): {final_count}")
print(f"Absolute growth: {total_growth}")
print(f"Percentage growth: {pct_growth:.1f}%")
print(f"Growth factor: {final_count / initial_count:.2f}x")

# Simple linear trend
slope, intercept, r_value, p_value, std_err = stats.linregress(year, C)
print(f"\nSimple linear trend:")
print(f"  Slope: {slope:.2f} counts per standardized year")
print(f"  R-squared: {r_value**2:.4f}")
print(f"  P-value: {p_value:.2e}")

print("\n" + "="*80)
print("INITIAL EXPLORATION COMPLETE")
print("="*80)
