"""
Initial Data Exploration - Temporal Patterns and Growth Dynamics
Analyst 2: Focus on time series characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("="*80)
print("INITIAL DATA EXPLORATION")
print("="*80)

print("\nDataset Shape:", data.shape)
print("\nFirst 10 rows:")
print(data.head(10))
print("\nLast 10 rows:")
print(data.tail(10))

print("\nBasic Statistics:")
print(data.describe())

print("\nData Types:")
print(data.dtypes)

print("\nMissing Values:")
print(data.isnull().sum())

print("\n" + "="*80)
print("TEMPORAL CHARACTERISTICS")
print("="*80)

# Check if data is sorted by year
print(f"\nIs data sorted by year? {data['year'].is_monotonic_increasing}")

# Time span
print(f"Time span: {data['year'].min():.3f} to {data['year'].max():.3f}")
print(f"Range: {data['year'].max() - data['year'].min():.3f}")

# Check for regular spacing
time_diffs = data['year'].diff().dropna()
print(f"\nTime intervals between observations:")
print(f"  Mean: {time_diffs.mean():.6f}")
print(f"  Std: {time_diffs.std():.6f}")
print(f"  Min: {time_diffs.min():.6f}")
print(f"  Max: {time_diffs.max():.6f}")
print(f"  Regular spacing? {time_diffs.std() < 0.0001}")

print("\n" + "="*80)
print("VARIABLE C CHARACTERISTICS")
print("="*80)

print(f"\nC statistics:")
print(f"  Min: {data['C'].min()}")
print(f"  Max: {data['C'].max()}")
print(f"  Mean: {data['C'].mean():.2f}")
print(f"  Median: {data['C'].median():.2f}")
print(f"  Std: {data['C'].std():.2f}")
print(f"  CV (coefficient of variation): {data['C'].std()/data['C'].mean():.2f}")

# Range over time
first_half = data.iloc[:20]['C'].mean()
second_half = data.iloc[20:]['C'].mean()
print(f"\nMean C in first half: {first_half:.2f}")
print(f"Mean C in second half: {second_half:.2f}")
print(f"Ratio (second/first): {second_half/first_half:.2f}")

# Basic correlation
corr = data['year'].corr(data['C'])
print(f"\nPearson correlation (year, C): {corr:.4f}")

# Save basic info
with open('/workspace/eda/analyst_2/eda_log.md', 'w') as f:
    f.write("# EDA Log - Analyst 2: Temporal Patterns and Growth Dynamics\n\n")
    f.write("## Initial Data Exploration\n\n")
    f.write(f"- Dataset: 40 observations, 2 variables (year, C)\n")
    f.write(f"- No missing values\n")
    f.write(f"- Regular time spacing: {time_diffs.mean():.6f} units\n")
    f.write(f"- Time range: [{data['year'].min():.3f}, {data['year'].max():.3f}]\n")
    f.write(f"- C range: [{data['C'].min()}, {data['C'].max()}]\n")
    f.write(f"- Mean C increases {second_half/first_half:.2f}x from first to second half\n")
    f.write(f"- Strong positive correlation: {corr:.4f}\n\n")

print("\n" + "="*80)
print("Initial exploration complete. Results saved to eda_log.md")
print("="*80)
