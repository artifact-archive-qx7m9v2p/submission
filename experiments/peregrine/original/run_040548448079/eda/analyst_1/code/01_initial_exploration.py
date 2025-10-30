"""
Initial Temporal Exploration
Analyst 1: Temporal Patterns and Trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('/workspace/data/data_analyst_1.csv')

print("=" * 80)
print("INITIAL DATA EXPLORATION")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic statistics:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Time span
print(f"\nYear range (standardized): {df['year'].min():.3f} to {df['year'].max():.3f}")
print(f"Number of observations: {len(df)}")
print(f"Average spacing: {df['year'].diff().mean():.4f}")
print(f"Spacing std dev: {df['year'].diff().std():.4f}")

# Count variable
print(f"\nCount (C) statistics:")
print(f"  Range: {df['C'].min()} to {df['C'].max()}")
print(f"  Mean: {df['C'].mean():.2f}")
print(f"  Median: {df['C'].median():.2f}")
print(f"  Std Dev: {df['C'].std():.2f}")
print(f"  Coefficient of Variation: {df['C'].std() / df['C'].mean():.2f}")

# Check for zeros (important for log transformations)
print(f"  Zero values: {(df['C'] == 0).sum()}")
print(f"  Min non-zero: {df[df['C'] > 0]['C'].min()}")

# Basic temporal characteristics
print("\n" + "=" * 80)
print("TEMPORAL CHARACTERISTICS")
print("=" * 80)

# Overall growth
total_growth = df['C'].iloc[-1] - df['C'].iloc[0]
pct_growth = (df['C'].iloc[-1] / df['C'].iloc[0] - 1) * 100
print(f"\nAbsolute growth: {total_growth} (from {df['C'].iloc[0]} to {df['C'].iloc[-1]})")
print(f"Percentage growth: {pct_growth:.1f}%")

# Mean growth rates
first_half_mean = df.iloc[:20]['C'].mean()
second_half_mean = df.iloc[20:]['C'].mean()
print(f"\nFirst half mean: {first_half_mean:.2f}")
print(f"Second half mean: {second_half_mean:.2f}")
print(f"Ratio: {second_half_mean / first_half_mean:.2f}x")

print("\nAnalysis complete.")
