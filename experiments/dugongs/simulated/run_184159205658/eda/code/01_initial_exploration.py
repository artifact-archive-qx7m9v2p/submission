"""
Initial Data Exploration and Quality Assessment
================================================
Author: EDA Specialist Agent
Date: 2025-10-27

This script performs initial data quality checks and basic statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data.csv'
OUTPUT_DIR = BASE_DIR / 'eda'

# Load data
print("=" * 80)
print("INITIAL DATA EXPLORATION")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape: {df.shape}")
print(f"Variables: {list(df.columns)}")

# Basic info
print("\n" + "-" * 80)
print("DATA TYPES AND MISSING VALUES")
print("-" * 80)
print(df.info())

# Missing values
print("\n" + "-" * 80)
print("MISSING VALUE ANALYSIS")
print("-" * 80)
missing = df.isnull().sum()
missing_pct = 100 * df.isnull().sum() / len(df)
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})
print(missing_df)

# Descriptive statistics
print("\n" + "-" * 80)
print("DESCRIPTIVE STATISTICS")
print("-" * 80)
print(df.describe())

# Extended statistics
print("\n" + "-" * 80)
print("EXTENDED STATISTICS")
print("-" * 80)
for col in df.columns:
    print(f"\n{col}:")
    print(f"  Count: {df[col].count()}")
    print(f"  Mean: {df[col].mean():.4f}")
    print(f"  Median: {df[col].median():.4f}")
    print(f"  Std Dev: {df[col].std():.4f}")
    print(f"  Variance: {df[col].var():.4f}")
    print(f"  Min: {df[col].min():.4f}")
    print(f"  Max: {df[col].max():.4f}")
    print(f"  Range: {df[col].max() - df[col].min():.4f}")
    print(f"  IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.4f}")
    print(f"  Skewness: {df[col].skew():.4f}")
    print(f"  Kurtosis: {df[col].kurtosis():.4f}")

# Check for duplicates
print("\n" + "-" * 80)
print("DUPLICATE ANALYSIS")
print("-" * 80)
print(f"Duplicate rows (exact): {df.duplicated().sum()}")
print(f"Duplicate x values: {df['x'].duplicated().sum()}")

# X value distribution - check for clustering
print("\n" + "-" * 80)
print("X VALUE DISTRIBUTION")
print("-" * 80)
x_counts = df['x'].value_counts().sort_index()
print("Frequency of x values:")
print(x_counts)

# Check for potential outliers using IQR method
print("\n" + "-" * 80)
print("OUTLIER DETECTION (IQR Method)")
print("-" * 80)
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n{col}:")
    print(f"  Lower bound: {lower_bound:.4f}")
    print(f"  Upper bound: {upper_bound:.4f}")
    print(f"  Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Outlier values: {sorted(outliers[col].values)}")

# Correlation
print("\n" + "-" * 80)
print("CORRELATION ANALYSIS")
print("-" * 80)
corr = df.corr()
print(corr)
print(f"\nPearson correlation (x, Y): {df['x'].corr(df['Y']):.4f}")
print(f"Spearman correlation (x, Y): {df['x'].corr(df['Y'], method='spearman'):.4f}")

print("\n" + "=" * 80)
print("Initial exploration complete!")
print("=" * 80)
