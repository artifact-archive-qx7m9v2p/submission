"""
Initial Exploration of Binomial Dataset
========================================
This script performs initial data loading and basic descriptive statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Setup paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/eda")

# Load data
print("="*60)
print("BINOMIAL DATASET - INITIAL EXPLORATION")
print("="*60)

df = pd.read_csv(DATA_PATH)

print("\n1. DATA STRUCTURE")
print("-"*60)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

print("\n2. FIRST FEW ROWS")
print("-"*60)
print(df.head(12))

print("\n3. BASIC DESCRIPTIVE STATISTICS")
print("-"*60)
print(df.describe())

print("\n4. MISSING VALUES CHECK")
print("-"*60)
print(df.isnull().sum())

print("\n5. DATA VALIDATION")
print("-"*60)
# Check if r <= n (successes can't exceed trials)
invalid_rows = df[df['r'] > df['n']]
print(f"Rows where r > n: {len(invalid_rows)}")

# Check if proportion matches r/n
df['calculated_proportion'] = df['r'] / df['n']
df['proportion_diff'] = np.abs(df['proportion'] - df['calculated_proportion'])
max_diff = df['proportion_diff'].max()
print(f"Max difference between given proportion and r/n: {max_diff:.2e}")

# Check for negative values
print(f"Negative values in n: {(df['n'] < 0).sum()}")
print(f"Negative values in r: {(df['r'] < 0).sum()}")

print("\n6. SAMPLE SIZE CHARACTERISTICS")
print("-"*60)
print(f"Total trials across all observations: {df['n'].sum()}")
print(f"Total successes across all observations: {df['r'].sum()}")
print(f"Overall success rate: {df['r'].sum() / df['n'].sum():.4f}")
print(f"\nSample size range: [{df['n'].min()}, {df['n'].max()}]")
print(f"Sample size variance: {df['n'].var():.2f}")
print(f"Sample size CV: {df['n'].std() / df['n'].mean():.4f}")

print("\n7. PROPORTION CHARACTERISTICS")
print("-"*60)
print(f"Proportion range: [{df['proportion'].min():.4f}, {df['proportion'].max():.4f}]")
print(f"Mean proportion: {df['proportion'].mean():.4f}")
print(f"Median proportion: {df['proportion'].median():.4f}")
print(f"Proportion variance: {df['proportion'].var():.6f}")
print(f"Proportion std: {df['proportion'].std():.6f}")

print("\n8. INDIVIDUAL OBSERVATION DETAILS")
print("-"*60)
for idx, row in df.iterrows():
    print(f"Trial {row['trial_id']:2.0f}: n={row['n']:3.0f}, r={row['r']:2.0f}, "
          f"p={row['proportion']:.4f}")

print("\n" + "="*60)
print("EXPLORATION COMPLETE")
print("="*60)
