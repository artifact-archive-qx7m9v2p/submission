"""
Initial Exploration of Eight Schools Dataset
============================================
Classic hierarchical meta-analysis problem
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
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Load data
data = pd.read_csv('/workspace/data/data.csv')

print("=" * 70)
print("EIGHT SCHOOLS DATASET - INITIAL EXPLORATION")
print("=" * 70)

# Basic info
print("\n1. DATASET STRUCTURE")
print("-" * 70)
print(f"Number of schools: {len(data)}")
print(f"Columns: {list(data.columns)}")
print(f"\nData types:")
print(data.dtypes)

# Display full dataset (it's small)
print("\n2. COMPLETE DATASET")
print("-" * 70)
print(data.to_string(index=False))

# Missing values
print("\n3. DATA QUALITY")
print("-" * 70)
print(f"Missing values:\n{data.isnull().sum()}")
print(f"\nDuplicate rows: {data.duplicated().sum()}")

# Summary statistics
print("\n4. SUMMARY STATISTICS")
print("-" * 70)
print("\nObserved Effects (y):")
print(data['y'].describe())
print(f"Range: [{data['y'].min():.1f}, {data['y'].max():.1f}]")
print(f"IQR: {data['y'].quantile(0.75) - data['y'].quantile(0.25):.1f}")
print(f"Variance: {data['y'].var():.1f}")
print(f"Std Dev: {data['y'].std():.1f}")

print("\nStandard Errors (sigma):")
print(data['sigma'].describe())
print(f"Range: [{data['sigma'].min():.1f}, {data['sigma'].max():.1f}]")
print(f"IQR: {data['sigma'].quantile(0.75) - data['sigma'].quantile(0.25):.1f}")
print(f"Coefficient of Variation: {data['sigma'].std() / data['sigma'].mean():.3f}")

# Calculate precision (inverse variance)
data['precision'] = 1 / (data['sigma'] ** 2)
print("\nPrecision (1/sigma^2):")
print(data['precision'].describe())

print("\n5. DISTRIBUTION CHARACTERISTICS")
print("-" * 70)

# Test normality
shapiro_y = stats.shapiro(data['y'])
shapiro_sigma = stats.shapiro(data['sigma'])
print(f"\nShapiro-Wilk test for y: W={shapiro_y.statistic:.4f}, p={shapiro_y.pvalue:.4f}")
print(f"Shapiro-Wilk test for sigma: W={shapiro_sigma.statistic:.4f}, p={shapiro_sigma.pvalue:.4f}")

# Skewness and kurtosis
print(f"\nSkewness of y: {stats.skew(data['y']):.3f}")
print(f"Kurtosis of y: {stats.kurtosis(data['y']):.3f}")
print(f"Skewness of sigma: {stats.skew(data['sigma']):.3f}")
print(f"Kurtosis of sigma: {stats.kurtosis(data['sigma']):.3f}")

print("\n6. CORRELATION ANALYSIS")
print("-" * 70)
corr_y_sigma = data['y'].corr(data['sigma'])
print(f"\nPearson correlation (y, sigma): {corr_y_sigma:.3f}")
spearman_corr = data['y'].corr(data['sigma'], method='spearman')
print(f"Spearman correlation (y, sigma): {spearman_corr:.3f}")

# Statistical test
corr_test = stats.pearsonr(data['y'], data['sigma'])
print(f"Pearson test: r={corr_test.statistic:.3f}, p={corr_test.pvalue:.3f}")

print("\n7. HETEROGENEITY ASSESSMENT")
print("-" * 70)

# Simple weighted mean
weights = 1 / (data['sigma'] ** 2)
weighted_mean = np.sum(data['y'] * weights) / np.sum(weights)
print(f"\nWeighted mean effect: {weighted_mean:.2f}")
print(f"Unweighted mean effect: {data['y'].mean():.2f}")

# Q-statistic for heterogeneity
Q = np.sum(weights * (data['y'] - weighted_mean) ** 2)
df = len(data) - 1
p_value_Q = 1 - stats.chi2.cdf(Q, df)
print(f"\nCochran's Q test:")
print(f"  Q = {Q:.2f}")
print(f"  df = {df}")
print(f"  p-value = {p_value_Q:.4f}")

# I-squared statistic
I_squared = max(0, 100 * (Q - df) / Q)
print(f"\nI² statistic: {I_squared:.1f}%")
if I_squared < 25:
    print("  Interpretation: Low heterogeneity")
elif I_squared < 50:
    print("  Interpretation: Moderate heterogeneity")
elif I_squared < 75:
    print("  Interpretation: Substantial heterogeneity")
else:
    print("  Interpretation: Considerable heterogeneity")

# Between-study variance estimate (DerSimonian-Laird)
tau_squared = max(0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
print(f"\nEstimated between-study variance (tau²): {tau_squared:.2f}")
print(f"Estimated between-study SD (tau): {np.sqrt(tau_squared):.2f}")

print("\n8. OUTLIER DETECTION")
print("-" * 70)

# Standardized residuals
z_scores = (data['y'] - weighted_mean) / data['sigma']
print("\nStandardized residuals (z-scores):")
for i, (school, z) in enumerate(zip(data['school'], z_scores), 1):
    flag = "***" if abs(z) > 2 else ("**" if abs(z) > 1.5 else "")
    print(f"  School {school}: {z:6.2f} {flag}")

# Identify potential outliers
outliers = data[np.abs(z_scores) > 2]
print(f"\nPotential outliers (|z| > 2): {len(outliers)}")
if len(outliers) > 0:
    print(outliers[['school', 'y', 'sigma']].to_string(index=False))

print("\n9. INFLUENCE ANALYSIS")
print("-" * 70)

# Leave-one-out analysis
print("\nLeave-one-out weighted means:")
for idx in range(len(data)):
    loo_data = data.drop(idx)
    loo_weights = 1 / (loo_data['sigma'] ** 2)
    loo_mean = np.sum(loo_data['y'] * loo_weights) / np.sum(loo_weights)
    influence = loo_mean - weighted_mean
    print(f"  Without School {data.iloc[idx]['school']}: {loo_mean:6.2f} (change: {influence:+.2f})")

# Save augmented data
data['z_score'] = z_scores
data.to_csv('/workspace/eda/code/data_with_diagnostics.csv', index=False)
print("\n" + "=" * 70)
print("Analysis complete. Augmented data saved.")
print("=" * 70)
