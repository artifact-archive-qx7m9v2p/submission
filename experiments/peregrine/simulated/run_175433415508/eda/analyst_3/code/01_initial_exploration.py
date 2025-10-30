"""
Initial Data Exploration for Model Assumption Testing
Analyst 3 - Diagnostic Preparation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
data = pd.read_csv('/workspace/data/data_analyst_3.csv')

print("="*80)
print("INITIAL DATA EXPLORATION")
print("="*80)

# Basic info
print("\nDataset Shape:", data.shape)
print("\nColumn Names:", data.columns.tolist())
print("\nData Types:")
print(data.dtypes)

# Missing values
print("\n" + "="*80)
print("DATA QUALITY CHECKS")
print("="*80)
print("\nMissing Values:")
print(data.isnull().sum())
print(f"\nMissing Value Percentage: {(data.isnull().sum().sum() / data.size) * 100:.2f}%")

# Duplicate rows
print(f"\nDuplicate Rows: {data.duplicated().sum()}")

# Basic statistics
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)
print("\n", data.describe())

# Count variable specifics
print("\n" + "="*80)
print("COUNT VARIABLE (C) ANALYSIS")
print("="*80)
print(f"\nRange: [{data['C'].min()}, {data['C'].max()}]")
print(f"Mean: {data['C'].mean():.4f}")
print(f"Variance: {data['C'].var():.4f}")
print(f"Variance-to-Mean Ratio: {data['C'].var() / data['C'].mean():.4f}")
print(f"  (Poisson assumption: ratio ≈ 1, observed: {data['C'].var() / data['C'].mean():.4f})")

# Check if counts are integers
print(f"\nAll integers?: {data['C'].apply(lambda x: x == int(x)).all()}")
print(f"Non-negative?: {(data['C'] >= 0).all()}")

# Skewness and Kurtosis
print(f"\nSkewness: {stats.skew(data['C']):.4f}")
print(f"Kurtosis: {stats.kurtosis(data['C']):.4f}")

# Year variable
print("\n" + "="*80)
print("YEAR VARIABLE ANALYSIS")
print("="*80)
print(f"\nRange: [{data['year'].min():.4f}, {data['year'].max():.4f}]")
print(f"Mean: {data['year'].mean():.4f}")
print(f"Std: {data['year'].std():.4f}")

# Check spacing
year_diffs = data['year'].diff().dropna()
print(f"\nYear spacing (differences):")
print(f"  Mean: {year_diffs.mean():.6f}")
print(f"  Std: {year_diffs.std():.6f}")
print(f"  Min: {year_diffs.min():.6f}")
print(f"  Max: {year_diffs.max():.6f}")
print(f"  Uniform spacing?: {np.allclose(year_diffs, year_diffs.mean(), rtol=1e-6)}")

# Save summary
with open('/workspace/eda/analyst_3/eda_log.md', 'w') as f:
    f.write("# EDA Log - Analyst 3: Model Assumptions & Diagnostics\n\n")
    f.write("## Objective\n")
    f.write("Systematically evaluate model assumptions for count data, focusing on:\n")
    f.write("- Poisson distributional assumptions\n")
    f.write("- Variance patterns (heteroscedasticity)\n")
    f.write("- Residual diagnostics from simple models\n")
    f.write("- Transformation effects\n\n")
    f.write("## Data Quality Assessment\n\n")
    f.write(f"- **Sample size**: {data.shape[0]} observations\n")
    f.write(f"- **Variables**: {', '.join(data.columns.tolist())}\n")
    f.write(f"- **Missing values**: {data.isnull().sum().sum()} ({(data.isnull().sum().sum() / data.size) * 100:.2f}%)\n")
    f.write(f"- **Duplicates**: {data.duplicated().sum()}\n")
    f.write(f"- **Data integrity**: All counts are non-negative integers: {(data['C'] >= 0).all()}\n\n")
    f.write("## Initial Count Distribution Characteristics\n\n")
    f.write(f"- **Mean**: {data['C'].mean():.4f}\n")
    f.write(f"- **Variance**: {data['C'].var():.4f}\n")
    f.write(f"- **Variance-to-Mean Ratio**: {data['C'].var() / data['C'].mean():.4f}\n")
    f.write(f"  - **Interpretation**: Ratio >> 1 indicates **overdispersion** (variance exceeds mean)\n")
    f.write(f"  - **Implication**: Simple Poisson model likely inappropriate; consider Negative Binomial\n")
    f.write(f"- **Skewness**: {stats.skew(data['C']):.4f}\n")
    f.write(f"- **Kurtosis**: {stats.kurtosis(data['C']):.4f}\n\n")

print("\n" + "="*80)
print("Initial exploration complete. Log saved.")
print("="*80)
