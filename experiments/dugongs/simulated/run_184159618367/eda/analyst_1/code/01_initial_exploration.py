"""
Initial Exploration of x-Y Relationship
Analyst 1 - Round 1
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
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

print("=" * 60)
print("INITIAL DATA EXPLORATION")
print("=" * 60)

# Basic info
print("\n1. DATA STRUCTURE")
print(f"Observations: {len(data)}")
print(f"Variables: {list(data.columns)}")
print(f"\nData types:\n{data.dtypes}")

# Descriptive statistics
print("\n2. DESCRIPTIVE STATISTICS")
print(data.describe())

# Missing values
print("\n3. DATA QUALITY")
print(f"Missing values:\n{data.isnull().sum()}")
print(f"Duplicate rows: {data.duplicated().sum()}")

# Range and spread
print("\n4. VARIABLE RANGES")
print(f"x range: [{data['x'].min():.2f}, {data['x'].max():.2f}]")
print(f"Y range: [{data['Y'].min():.2f}, {data['Y'].max():.2f}]")
print(f"x span: {data['x'].max() - data['x'].min():.2f}")
print(f"Y span: {data['Y'].max() - data['Y'].min():.2f}")

# Distribution properties
print("\n5. DISTRIBUTION PROPERTIES")
print(f"\nx - Skewness: {stats.skew(data['x']):.3f}, Kurtosis: {stats.kurtosis(data['x']):.3f}")
print(f"Y - Skewness: {stats.skew(data['Y']):.3f}, Kurtosis: {stats.kurtosis(data['Y']):.3f}")

# Test for normality
_, p_x = stats.shapiro(data['x'])
_, p_y = stats.shapiro(data['Y'])
print(f"\nShapiro-Wilk normality test:")
print(f"x: p-value = {p_x:.4f} {'(Normal)' if p_x > 0.05 else '(Non-normal)'}")
print(f"Y: p-value = {p_y:.4f} {'(Normal)' if p_y > 0.05 else '(Non-normal)'}")

# Correlation
print("\n6. CORRELATION ANALYSIS")
pearson_r, pearson_p = stats.pearsonr(data['x'], data['Y'])
spearman_r, spearman_p = stats.spearmanr(data['x'], data['Y'])
print(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4e}")
print(f"Spearman correlation: rho = {spearman_r:.4f}, p = {spearman_p:.4e}")

# Check for duplicates in x values
print("\n7. X VALUE DISTRIBUTION")
x_counts = data['x'].value_counts().sort_index()
print(f"Unique x values: {data['x'].nunique()}")
print(f"Repeated x values: {sum(x_counts > 1)}")
if sum(x_counts > 1) > 0:
    print(f"\nX values with replicates:")
    print(x_counts[x_counts > 1])

print("\n" + "=" * 60)
