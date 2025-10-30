"""
Initial Data Exploration: Distributional Characteristics and Sample Size Effects
EDA Analyst 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set up paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data_analyst_1.csv'
OUTPUT_DIR = BASE_DIR / 'eda' / 'analyst_1'
VIZ_DIR = OUTPUT_DIR / 'visualizations'
CODE_DIR = OUTPUT_DIR / 'code'

# Ensure directories exist
VIZ_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("INITIAL DATA EXPLORATION")
print("=" * 70)

print("\nDataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print("\nDescriptive Statistics:")
print(df.describe())

# Additional statistics
print("\nSuccess Rate Statistics:")
print(f"Mean: {df['success_rate'].mean():.4f}")
print(f"Median: {df['success_rate'].median():.4f}")
print(f"Std Dev: {df['success_rate'].std():.4f}")
print(f"Variance: {df['success_rate'].var():.6f}")
print(f"CV (Coefficient of Variation): {df['success_rate'].std() / df['success_rate'].mean():.4f}")
print(f"Min: {df['success_rate'].min():.4f}")
print(f"Max: {df['success_rate'].max():.4f}")
print(f"Range: {df['success_rate'].max() - df['success_rate'].min():.4f}")

print("\nSample Size (n_trials) Statistics:")
print(f"Mean: {df['n_trials'].mean():.2f}")
print(f"Median: {df['n_trials'].median():.2f}")
print(f"Std Dev: {df['n_trials'].std():.2f}")
print(f"Min: {df['n_trials'].min()}")
print(f"Max: {df['n_trials'].max()}")
print(f"Range: {df['n_trials'].max() - df['n_trials'].min()}")

print("\nSuccesses (r_successes) Statistics:")
print(f"Mean: {df['r_successes'].mean():.2f}")
print(f"Median: {df['r_successes'].median():.2f}")
print(f"Total: {df['r_successes'].sum()}")

print("\n" + "=" * 70)
print("GROUP-LEVEL DETAILS")
print("=" * 70)
print("\nAll Groups:")
print(df.to_string(index=False))
