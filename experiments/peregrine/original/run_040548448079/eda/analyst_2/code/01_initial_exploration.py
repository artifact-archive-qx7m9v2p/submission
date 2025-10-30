"""
Initial Exploration of Distributional Properties
Focus: Basic statistics, distribution shape, variance-mean relationship
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {data.shape}")
print(f"\nFirst few rows:\n{data.head(10)}")
print(f"\nLast few rows:\n{data.tail(10)}")
print(f"\nData types:\n{data.dtypes}")
print(f"\nMissing values:\n{data.isnull().sum()}")

# Basic statistics
print("\n" + "=" * 80)
print("BASIC STATISTICS")
print("=" * 80)
print(data.describe())

# Count-specific statistics
C = data['C'].values
print("\n" + "=" * 80)
print("COUNT VARIABLE (C) PROPERTIES")
print("=" * 80)
print(f"Sample size: {len(C)}")
print(f"Mean: {C.mean():.2f}")
print(f"Median: {np.median(C):.2f}")
print(f"Std Dev: {C.std(ddof=1):.2f}")
print(f"Variance: {C.var(ddof=1):.2f}")
print(f"Min: {C.min()}")
print(f"Max: {C.max()}")
print(f"Range: {C.max() - C.min()}")
print(f"IQR: {np.percentile(C, 75) - np.percentile(C, 25):.2f}")
print(f"Skewness: {stats.skew(C):.3f}")
print(f"Kurtosis (excess): {stats.kurtosis(C):.3f}")

# Variance-to-mean ratio (key diagnostic for Poisson vs overdispersion)
var_mean_ratio = C.var(ddof=1) / C.mean()
print(f"\n*** Variance-to-Mean Ratio: {var_mean_ratio:.3f} ***")
print(f"    (Poisson assumption: ratio = 1)")
print(f"    (Ratio > 1: overdispersion)")
print(f"    (Ratio < 1: underdispersion)")

# Zero counts
n_zeros = (C == 0).sum()
print(f"\nZero counts: {n_zeros} ({n_zeros/len(C)*100:.1f}%)")
print(f"Min non-zero count: {C[C > 0].min()}")

# Percentiles
print("\n" + "=" * 80)
print("PERCENTILES")
print("=" * 80)
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    print(f"{p:3d}th percentile: {np.percentile(C, p):6.1f}")

print("\n" + "=" * 80)
print("TEMPORAL SUMMARY")
print("=" * 80)
# Split into thirds by time
n = len(data)
early = data.iloc[:n//3]['C']
middle = data.iloc[n//3:2*n//3]['C']
late = data.iloc[2*n//3:]['C']

print(f"Early period (n={len(early)}):")
print(f"  Mean: {early.mean():.2f}, Var: {early.var(ddof=1):.2f}, Var/Mean: {early.var(ddof=1)/early.mean():.3f}")
print(f"Middle period (n={len(middle)}):")
print(f"  Mean: {middle.mean():.2f}, Var: {middle.var(ddof=1):.2f}, Var/Mean: {middle.var(ddof=1)/middle.mean():.3f}")
print(f"Late period (n={len(late)}):")
print(f"  Mean: {late.mean():.2f}, Var: {late.var(ddof=1):.2f}, Var/Mean: {late.var(ddof=1)/late.mean():.3f}")
