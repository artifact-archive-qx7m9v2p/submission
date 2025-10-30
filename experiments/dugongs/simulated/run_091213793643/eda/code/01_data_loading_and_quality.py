"""
Script 1: Data Loading and Quality Assessment
Loads the dataset and performs initial quality checks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Set up paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/eda")
CODE_DIR = OUTPUT_DIR / "code"
VIZ_DIR = OUTPUT_DIR / "visualizations"

# Load data
print("="*80)
print("DATA LOADING AND QUALITY ASSESSMENT")
print("="*80)

df = pd.read_csv(DATA_PATH)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())

# Data types
print("\n" + "="*80)
print("DATA TYPES")
print("="*80)
print(df.dtypes)

# Missing values
print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing: {missing.sum()}")

# Duplicates
print("\n" + "="*80)
print("DUPLICATE ROWS")
print("="*80)
n_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {n_duplicates}")

# Basic statistics
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)
print(df.describe())

# Additional statistics
print("\n" + "="*80)
print("ADDITIONAL STATISTICS")
print("="*80)
for col in df.columns:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.6f}")
    print(f"  Median: {df[col].median():.6f}")
    print(f"  Std Dev: {df[col].std():.6f}")
    print(f"  Variance: {df[col].var():.6f}")
    print(f"  Skewness: {df[col].skew():.6f}")
    print(f"  Kurtosis: {df[col].kurtosis():.6f}")
    print(f"  Range: [{df[col].min():.6f}, {df[col].max():.6f}]")
    print(f"  IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.6f}")

# Check for outliers using IQR method
print("\n" + "="*80)
print("OUTLIER DETECTION (IQR Method)")
print("="*80)
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n{col}:")
    print(f"  Lower bound: {lower_bound:.6f}")
    print(f"  Upper bound: {upper_bound:.6f}")
    print(f"  Number of outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Outlier indices: {outliers.index.tolist()}")
        print(f"  Outlier values: {outliers[col].values}")

# Check spacing of x values
print("\n" + "="*80)
print("X-VARIABLE SPACING")
print("="*80)
x_sorted = df['x'].sort_values()
x_diffs = x_sorted.diff().dropna()
print(f"Unique x values: {df['x'].nunique()}")
print(f"Total observations: {len(df)}")
print(f"Repeated x values:")
x_counts = df['x'].value_counts().sort_index()
repeated = x_counts[x_counts > 1]
if len(repeated) > 0:
    print(repeated)
else:
    print("  None")
print(f"\nSpacing between consecutive x values:")
print(f"  Mean spacing: {x_diffs.mean():.6f}")
print(f"  Median spacing: {x_diffs.median():.6f}")
print(f"  Min spacing: {x_diffs.min():.6f}")
print(f"  Max spacing: {x_diffs.max():.6f}")

# Save summary statistics to JSON
summary_stats = {
    'n_observations': len(df),
    'n_variables': len(df.columns),
    'missing_values': missing.to_dict(),
    'n_duplicates': int(n_duplicates),
    'statistics': {
        col: {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'skewness': float(df[col].skew()),
            'kurtosis': float(df[col].kurtosis())
        }
        for col in df.columns
    }
}

with open(OUTPUT_DIR / 'data_quality_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\n" + "="*80)
print("Summary statistics saved to data_quality_summary.json")
print("="*80)
