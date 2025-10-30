"""
Initial Data Exploration for Meta-Analysis Dataset
===================================================
This script performs basic data loading, validation, and summary statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
data = pd.read_csv('/workspace/data/data.csv')

print("="*80)
print("INITIAL DATA EXPLORATION")
print("="*80)

# Basic information
print("\n1. DATA STRUCTURE")
print("-"*80)
print(f"Number of observations (J): {len(data)}")
print(f"Variables: {list(data.columns)}")
print(f"\nData types:\n{data.dtypes}")

# Display data
print("\n2. RAW DATA")
print("-"*80)
print(data.to_string(index=False))

# Check for missing values
print("\n3. DATA QUALITY CHECKS")
print("-"*80)
missing = data.isnull().sum()
print(f"Missing values:\n{missing}")
print(f"Total missing: {missing.sum()}")

# Check for duplicates
duplicates = data.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Check data validity
print(f"\nData validity checks:")
print(f"  - All sigma values positive: {(data['sigma'] > 0).all()}")
print(f"  - All sigma values finite: {np.isfinite(data['sigma']).all()}")
print(f"  - All y values finite: {np.isfinite(data['y']).all()}")

# Summary statistics
print("\n4. SUMMARY STATISTICS")
print("-"*80)
print(data.describe())

# Additional statistics
print("\n5. DETAILED STATISTICS")
print("-"*80)
for col in ['y', 'sigma']:
    print(f"\n{col.upper()}:")
    values = data[col]
    print(f"  Mean: {values.mean():.3f}")
    print(f"  Median: {values.median():.3f}")
    print(f"  Std Dev: {values.std():.3f}")
    print(f"  Variance: {values.var():.3f}")
    print(f"  Min: {values.min():.3f}")
    print(f"  Max: {values.max():.3f}")
    print(f"  Range: {values.max() - values.min():.3f}")
    print(f"  IQR: {values.quantile(0.75) - values.quantile(0.25):.3f}")
    print(f"  Skewness: {stats.skew(values):.3f}")
    print(f"  Kurtosis: {stats.kurtosis(values):.3f}")
    print(f"  CV (coefficient of variation): {values.std() / abs(values.mean()):.3f}")

# Outlier detection using IQR method
print("\n6. OUTLIER DETECTION (IQR METHOD)")
print("-"*80)
for col in ['y', 'sigma']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    print(f"\n{col.upper()}:")
    print(f"  Lower bound: {lower_bound:.3f}")
    print(f"  Upper bound: {upper_bound:.3f}")
    print(f"  Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Outlier indices: {outliers.index.tolist()}")
        print(f"  Outlier values: {outliers[col].tolist()}")

# Normality tests
print("\n7. NORMALITY TESTS")
print("-"*80)
for col in ['y', 'sigma']:
    stat, p_value = stats.shapiro(data[col])
    print(f"\n{col.upper()} - Shapiro-Wilk test:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Normally distributed (p > 0.05): {p_value > 0.05}")

# Correlation between y and sigma
print("\n8. CORRELATION ANALYSIS")
print("-"*80)
corr_pearson = data['y'].corr(data['sigma'])
corr_spearman = data['y'].corr(data['sigma'], method='spearman')
print(f"Pearson correlation (y vs sigma): {corr_pearson:.4f}")
print(f"Spearman correlation (y vs sigma): {corr_spearman:.4f}")

# Save processed data with additional columns
data['obs_id'] = range(1, len(data) + 1)
data['y_standardized'] = (data['y'] - data['y'].mean()) / data['y'].std()
data['sigma_standardized'] = (data['sigma'] - data['sigma'].mean()) / data['sigma'].std()
data['precision'] = 1 / data['sigma']**2
data['weight'] = 1 / data['sigma']**2

print("\n9. DERIVED QUANTITIES")
print("-"*80)
print("\nPrecisions (1/sigma^2):")
print(data['precision'].to_string(index=False))
print(f"\nPrecision range: {data['precision'].min():.4f} to {data['precision'].max():.4f}")
print(f"Precision ratio (max/min): {data['precision'].max() / data['precision'].min():.2f}")

# Weighted mean
weighted_mean = np.sum(data['y'] * data['weight']) / np.sum(data['weight'])
weighted_var = 1 / np.sum(data['weight'])
weighted_se = np.sqrt(weighted_var)

print(f"\n10. WEIGHTED STATISTICS")
print("-"*80)
print(f"Weighted mean: {weighted_mean:.3f}")
print(f"Weighted SE: {weighted_se:.3f}")
print(f"Simple mean: {data['y'].mean():.3f}")
print(f"Difference: {weighted_mean - data['y'].mean():.3f}")

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
