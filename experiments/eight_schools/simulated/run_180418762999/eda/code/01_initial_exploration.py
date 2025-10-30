"""
Initial Data Exploration
========================
Goal: Understand basic data structure, quality, and descriptive statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
data = pd.read_csv('/workspace/data/data.csv')

print("="*80)
print("INITIAL DATA EXPLORATION")
print("="*80)

# Basic structure
print("\n1. DATA STRUCTURE")
print("-"*80)
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"\nData types:\n{data.dtypes}")

# Display all rows (only 8 observations)
print("\n2. COMPLETE DATASET")
print("-"*80)
print(data.to_string(index=False))

# Missing values
print("\n3. DATA QUALITY")
print("-"*80)
print(f"Missing values:\n{data.isnull().sum()}")
print(f"Total missing: {data.isnull().sum().sum()}")
print(f"\nDuplicate rows: {data.duplicated().sum()}")

# Descriptive statistics
print("\n4. DESCRIPTIVE STATISTICS")
print("-"*80)
print(data.describe())

# Additional statistics
print("\n5. DETAILED STATISTICS FOR KEY VARIABLES")
print("-"*80)
for col in ['y', 'sigma']:
    print(f"\n{col.upper()}:")
    print(f"  Mean: {data[col].mean():.4f}")
    print(f"  Median: {data[col].median():.4f}")
    print(f"  Std Dev: {data[col].std():.4f}")
    print(f"  Variance: {data[col].var():.4f}")
    print(f"  Min: {data[col].min():.4f}")
    print(f"  Max: {data[col].max():.4f}")
    print(f"  Range: {data[col].max() - data[col].min():.4f}")
    print(f"  IQR: {data[col].quantile(0.75) - data[col].quantile(0.25):.4f}")
    print(f"  Coef. of Variation: {data[col].std() / data[col].mean():.4f}")
    print(f"  Skewness: {stats.skew(data[col]):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(data[col]):.4f}")

# Measurement error analysis
print("\n6. MEASUREMENT ERROR CHARACTERISTICS")
print("-"*80)
print(f"Sigma range: [{data['sigma'].min()}, {data['sigma'].max()}]")
print(f"Sigma values: {sorted(data['sigma'].unique())}")
print(f"Relative error (sigma/|y|):")
data['rel_error'] = data['sigma'] / np.abs(data['y'])
print(data[['group', 'y', 'sigma', 'rel_error']].to_string(index=False))
print(f"\nMean relative error: {data['rel_error'].mean():.4f}")
print(f"Median relative error: {data['rel_error'].median():.4f}")

# Signal-to-noise ratio
print("\n7. SIGNAL-TO-NOISE RATIO")
print("-"*80)
data['snr'] = np.abs(data['y']) / data['sigma']
print(data[['group', 'y', 'sigma', 'snr']].to_string(index=False))
print(f"\nMean SNR: {data['snr'].mean():.4f}")
print(f"Median SNR: {data['snr'].median():.4f}")

# Correlation
print("\n8. CORRELATION ANALYSIS")
print("-"*80)
print(f"Correlation between y and sigma: {data['y'].corr(data['sigma']):.4f}")
print(f"Spearman correlation: {stats.spearmanr(data['y'], data['sigma'])[0]:.4f}")

# Check for potential outliers
print("\n9. OUTLIER DETECTION")
print("-"*80)
for col in ['y', 'sigma']:
    q1, q3 = data[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    print(f"\n{col.upper()} outliers (IQR method):")
    if len(outliers) > 0:
        print(outliers[['group', col]].to_string(index=False))
    else:
        print("  None detected")

# Z-score outliers
print("\nZ-score outliers (|z| > 2):")
for col in ['y', 'sigma']:
    z_scores = np.abs(stats.zscore(data[col]))
    outliers = data[z_scores > 2]
    print(f"\n{col.upper()}:")
    if len(outliers) > 0:
        print(outliers[['group', col]].to_string(index=False))
    else:
        print("  None detected")

# Save processed data
data.to_csv('/workspace/eda/code/data_with_metrics.csv', index=False)
print("\n" + "="*80)
print("Processed data saved to: /workspace/eda/code/data_with_metrics.csv")
print("="*80)
