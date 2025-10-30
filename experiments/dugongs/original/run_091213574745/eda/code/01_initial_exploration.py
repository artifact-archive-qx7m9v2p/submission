"""
Initial Data Exploration
========================
Load data and perform basic quality checks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('/workspace/data/data.csv')

print("=" * 80)
print("DATA STRUCTURE AND QUALITY CHECKS")
print("=" * 80)

print("\n1. BASIC INFORMATION")
print("-" * 80)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

print("\n2. FIRST/LAST ROWS")
print("-" * 80)
print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())

print("\n3. MISSING VALUES")
print("-" * 80)
print(f"Missing values per column:\n{df.isnull().sum()}")
print(f"Total missing: {df.isnull().sum().sum()}")
print(f"Percentage missing: {100 * df.isnull().sum().sum() / (df.shape[0] * df.shape[1]):.2f}%")

print("\n4. DUPLICATES")
print("-" * 80)
print(f"Duplicate rows: {df.duplicated().sum()}")
if df.duplicated().sum() > 0:
    print("Duplicate rows:")
    print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)))

# Check for duplicate x values (legitimate replicates)
print(f"\nDuplicate x values (potential replicates): {df['x'].duplicated().sum()}")
if df['x'].duplicated().sum() > 0:
    print("x values with replicates:")
    dup_x = df[df['x'].duplicated(keep=False)].sort_values('x')
    print(dup_x)

print("\n5. BASIC STATISTICS")
print("-" * 80)
print(df.describe())

print("\n6. DATA RANGES")
print("-" * 80)
print(f"x range: [{df['x'].min()}, {df['x'].max()}]")
print(f"Y range: [{df['Y'].min()}, {df['Y'].max()}]")
print(f"x span: {df['x'].max() - df['x'].min()}")
print(f"Y span: {df['Y'].max() - df['Y'].min()}")

print("\n7. ADDITIONAL STATISTICS")
print("-" * 80)
for col in df.columns:
    skew = df[col].skew()
    kurt = df[col].kurtosis()
    print(f"\n{col}:")
    print(f"  Skewness: {skew:.3f}")
    print(f"  Kurtosis: {kurt:.3f}")
    print(f"  CV (coefficient of variation): {df[col].std() / df[col].mean():.3f}")
