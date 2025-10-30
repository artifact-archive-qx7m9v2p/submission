"""
Initial Data Exploration
Focus: Basic statistics, distributions, and initial visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

print("=" * 80)
print("INITIAL DATA EXPLORATION")
print("=" * 80)

# Basic info
print("\nDataset Shape:", data.shape)
print("\nFirst few rows:")
print(data.head(10))
print("\nLast few rows:")
print(data.tail(5))

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(data.describe())

# Missing values
print("\nMissing values:")
print(data.isnull().sum())

# Data types
print("\nData types:")
print(data.dtypes)

# Univariate statistics
print("\n" + "=" * 80)
print("UNIVARIATE ANALYSIS")
print("=" * 80)

for col in ['Y', 'x']:
    print(f"\n{col}:")
    print(f"  Range: [{data[col].min():.4f}, {data[col].max():.4f}]")
    print(f"  Mean: {data[col].mean():.4f}")
    print(f"  Median: {data[col].median():.4f}")
    print(f"  Std Dev: {data[col].std():.4f}")
    print(f"  Skewness: {stats.skew(data[col]):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(data[col]):.4f}")
    print(f"  CV (Coefficient of Variation): {data[col].std()/data[col].mean():.4f}")

# Correlation
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
print(f"Pearson correlation: {data['Y'].corr(data['x']):.4f}")
print(f"Spearman correlation: {data['Y'].corr(data['x'], method='spearman'):.4f}")
print(f"Kendall tau: {data['Y'].corr(data['x'], method='kendall'):.4f}")

# Check for duplicates in x
print("\n" + "=" * 80)
print("DUPLICATE AND REPLICATE ANALYSIS")
print("=" * 80)
print(f"Duplicate x values: {data['x'].duplicated().sum()}")
print("\nValues of x with replicates:")
x_counts = data['x'].value_counts().sort_index()
print(x_counts[x_counts > 1])

print("\nData points at duplicate x values:")
duplicate_x = data[data['x'].duplicated(keep=False)].sort_values('x')
print(duplicate_x)

# Create initial visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Scatter plot
axes[0, 0].scatter(data['x'], data['Y'], alpha=0.6, s=50)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Y')
axes[0, 0].set_title('Y vs x (Raw Data)')
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribution of Y
axes[0, 1].hist(data['Y'], bins=15, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Y')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Y')
axes[0, 1].axvline(data['Y'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 1].axvline(data['Y'].median(), color='green', linestyle='--', label='Median')
axes[0, 1].legend()

# 3. Distribution of x
axes[0, 2].hist(data['x'], bins=15, edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Distribution of x')
axes[0, 2].axvline(data['x'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 2].axvline(data['x'].median(), color='green', linestyle='--', label='Median')
axes[0, 2].legend()

# 4. Q-Q plot for Y
stats.probplot(data['Y'], dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot for Y')

# 5. Q-Q plot for x
stats.probplot(data['x'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot for x')

# 6. Box plots
bp_data = [data['Y'], data['x']]
axes[1, 2].boxplot(bp_data, labels=['Y', 'x'])
axes[1, 2].set_title('Box Plots')
axes[1, 2].set_ylabel('Value')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/01_initial_exploration.png', dpi=300, bbox_inches='tight')
print("\nSaved: /workspace/eda/analyst_2/visualizations/01_initial_exploration.png")
plt.close()

print("\n" + "=" * 80)
print("INITIAL EXPLORATION COMPLETE")
print("=" * 80)
