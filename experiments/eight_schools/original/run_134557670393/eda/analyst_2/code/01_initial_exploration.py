"""
Initial Exploration: Uncertainty Structure Analysis
Meta-analysis dataset with 8 studies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
df = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("="*60)
print("DATASET OVERVIEW")
print("="*60)
print(df)
print("\n")

print("="*60)
print("BASIC STATISTICS")
print("="*60)
print(df.describe())
print("\n")

print("="*60)
print("DATA QUALITY CHECKS")
print("="*60)
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nNumber of studies: {len(df)}")
print("\n")

# Calculate derived uncertainty metrics
df['precision'] = 1 / df['sigma']
df['variance'] = df['sigma'] ** 2
df['snr'] = df['y'] / df['sigma']  # Signal-to-noise ratio
df['standardized_effect'] = df['y'] / df['sigma']  # z-score
df['ci_lower'] = df['y'] - 1.96 * df['sigma']
df['ci_upper'] = df['y'] + 1.96 * df['sigma']
df['ci_width'] = df['ci_upper'] - df['ci_lower']

print("="*60)
print("UNCERTAINTY METRICS")
print("="*60)
print("\nStandard Error (sigma) statistics:")
print(df['sigma'].describe())
print(f"CV of sigma: {df['sigma'].std() / df['sigma'].mean():.3f}")
print(f"Range: {df['sigma'].min():.1f} to {df['sigma'].max():.1f}")

print("\nPrecision (1/sigma) statistics:")
print(df['precision'].describe())
print(f"CV of precision: {df['precision'].std() / df['precision'].mean():.3f}")

print("\nSignal-to-Noise Ratio statistics:")
print(df['snr'].describe())

print("\n")
print("="*60)
print("EFFECT SIZE CHARACTERISTICS")
print("="*60)
print(f"Effect size range: {df['y'].min():.1f} to {df['y'].max():.1f}")
print(f"Mean effect: {df['y'].mean():.2f} (SD: {df['y'].std():.2f})")
print(f"Median effect: {df['y'].median():.2f}")
print(f"Positive effects: {(df['y'] > 0).sum()}/{len(df)}")
print(f"Statistically significant (|z| > 1.96): {(np.abs(df['snr']) > 1.96).sum()}/{len(df)}")

print("\n")
print("="*60)
print("DETAILED STUDY BREAKDOWN")
print("="*60)
print(df[['study', 'y', 'sigma', 'precision', 'snr', 'ci_lower', 'ci_upper']].to_string(index=False))

# Save enhanced dataframe
df.to_csv('/workspace/eda/analyst_2/code/enhanced_data.csv', index=False)
print("\n Enhanced data saved to: /workspace/eda/analyst_2/code/enhanced_data.csv")
