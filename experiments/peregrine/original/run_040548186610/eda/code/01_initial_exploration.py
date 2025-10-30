"""
Initial Data Exploration
========================
Load data, compute basic statistics, and validate structure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = '/workspace/data/data.csv'
VIZ_DIR = Path('/workspace/eda/visualizations')
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("INITIAL DATA EXPLORATION")
print("=" * 80)

# Basic structure
print("\n1. DATA STRUCTURE")
print("-" * 40)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst few rows:\n{df.head(10)}")
print(f"\nLast few rows:\n{df.tail(5)}")

# Missing values
print("\n2. DATA QUALITY")
print("-" * 40)
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Descriptive statistics
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 40)
print(df.describe())

# Additional statistics for year
print("\n4. YEAR VARIABLE (NORMALIZED TIME)")
print("-" * 40)
print(f"Min: {df['year'].min():.4f}")
print(f"Max: {df['year'].max():.4f}")
print(f"Range: {df['year'].max() - df['year'].min():.4f}")
print(f"Mean: {df['year'].mean():.4f}")
print(f"Median: {df['year'].median():.4f}")
print(f"Std Dev: {df['year'].std():.4f}")

# Check if year is evenly spaced
year_diffs = np.diff(df['year'])
print(f"\nYear spacing (differences):")
print(f"  Mean: {year_diffs.mean():.6f}")
print(f"  Std Dev: {year_diffs.std():.10f}")
print(f"  Min: {year_diffs.min():.6f}")
print(f"  Max: {year_diffs.max():.6f}")
print(f"  Is evenly spaced: {np.allclose(year_diffs, year_diffs[0])}")

# Count variable statistics
print("\n5. COUNT VARIABLE (C)")
print("-" * 40)
print(f"Min: {df['C'].min()}")
print(f"Max: {df['C'].max()}")
print(f"Range: {df['C'].max() - df['C'].min()}")
print(f"Mean: {df['C'].mean():.2f}")
print(f"Median: {df['C'].median():.2f}")
print(f"Std Dev: {df['C'].std():.2f}")
print(f"Variance: {df['C'].var():.2f}")
print(f"Variance-to-Mean Ratio: {df['C'].var() / df['C'].mean():.2f}")

# Check for integer counts
print(f"\nAll values are integers: {(df['C'] == df['C'].astype(int)).all()}")
print(f"All values are positive: {(df['C'] > 0).all()}")

# Quantiles
print("\nQuantiles of C:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"  {q*100:>5.1f}%: {df['C'].quantile(q):>6.1f}")

# Skewness and kurtosis
print(f"\nSkewness: {stats.skew(df['C']):.3f}")
print(f"Kurtosis: {stats.kurtosis(df['C']):.3f}")

print("\n" + "=" * 80)
