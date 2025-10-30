"""
Initial Exploration of Meta-Analysis Dataset
=============================================
Analyzes basic distributional properties and relationships
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
data = pd.read_csv('/workspace/data/data.csv')

print("="*60)
print("BASIC DATA SUMMARY")
print("="*60)
print(f"\nDataset shape: {data.shape}")
print(f"\nColumn types:\n{data.dtypes}")
print(f"\nMissing values:\n{data.isnull().sum()}")

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)
print(data.describe())

# Calculate additional statistics
print("\n" + "="*60)
print("ADDITIONAL STATISTICS")
print("="*60)

# For observed effects
print("\nObserved Effects (y):")
print(f"  Range: [{data['y'].min():.2f}, {data['y'].max():.2f}]")
print(f"  IQR: {data['y'].quantile(0.75) - data['y'].quantile(0.25):.2f}")
print(f"  Skewness: {stats.skew(data['y']):.3f}")
print(f"  Kurtosis: {stats.kurtosis(data['y']):.3f}")

# For standard errors
print("\nStandard Errors (sigma):")
print(f"  Range: [{data['sigma'].min():.2f}, {data['sigma'].max():.2f}]")
print(f"  IQR: {data['sigma'].quantile(0.75) - data['sigma'].quantile(0.25):.2f}")
print(f"  Coefficient of Variation: {data['sigma'].std() / data['sigma'].mean():.3f}")

# Calculate precision
data['precision'] = 1 / data['sigma']
data['variance'] = data['sigma']**2

print("\nPrecision (1/sigma):")
print(f"  Range: [{data['precision'].min():.4f}, {data['precision'].max():.4f}]")

# Calculate weighted mean
weighted_mean = np.sum(data['y'] * data['precision']) / np.sum(data['precision'])
print(f"\nWeighted mean effect (precision-weighted): {weighted_mean:.3f}")
print(f"Unweighted mean effect: {data['y'].mean():.3f}")

# Test for outliers using z-scores
z_scores = np.abs(stats.zscore(data['y']))
print(f"\nZ-scores for observed effects:")
for i, (idx, row) in enumerate(data.iterrows()):
    print(f"  Study {row['study']}: {z_scores[i]:.3f}")

# Identify potential outliers (|z| > 2)
outliers = data[z_scores > 2]
if len(outliers) > 0:
    print(f"\nPotential outliers (|z| > 2):")
    print(outliers[['study', 'y', 'sigma']])
else:
    print("\nNo strong outliers detected using |z| > 2 criterion")

# Heterogeneity statistics
print("\n" + "="*60)
print("HETEROGENEITY ASSESSMENT")
print("="*60)

# Q statistic for heterogeneity
weights = data['precision']**2
Q = np.sum(weights * (data['y'] - weighted_mean)**2)
df = len(data) - 1
p_value = 1 - stats.chi2.cdf(Q, df)

print(f"\nCochran's Q test:")
print(f"  Q = {Q:.3f}")
print(f"  df = {df}")
print(f"  p-value = {p_value:.4f}")

if p_value < 0.05:
    print("  --> Significant heterogeneity detected (p < 0.05)")
else:
    print("  --> No significant heterogeneity detected (p >= 0.05)")

# I-squared statistic
I_squared = max(0, 100 * (Q - df) / Q)
print(f"\nI² statistic: {I_squared:.1f}%")
if I_squared < 25:
    print("  --> Low heterogeneity")
elif I_squared < 75:
    print("  --> Moderate heterogeneity")
else:
    print("  --> High heterogeneity")

# Tau-squared (DerSimonian-Laird estimator)
tau_squared = max(0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
print(f"\nTau² (between-study variance): {tau_squared:.3f}")
print(f"Tau (between-study SD): {np.sqrt(tau_squared):.3f}")

# Calculate ratio of total variance to within-study variance
total_var = data['y'].var()
avg_within_var = data['variance'].mean()
print(f"\nVariance decomposition:")
print(f"  Total variance in y: {total_var:.3f}")
print(f"  Average within-study variance: {avg_within_var:.3f}")
print(f"  Ratio (total/within): {total_var/avg_within_var:.3f}")

# Save processed data
data.to_csv('/workspace/eda/code/processed_data.csv', index=False)
print("\n" + "="*60)
print("Processed data saved to: /workspace/eda/code/processed_data.csv")
print("="*60)
