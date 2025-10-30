"""
Eight Schools Dataset - Initial Exploration
Classic hierarchical modeling dataset analysis
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

print("=" * 80)
print("EIGHT SCHOOLS DATASET - INITIAL EXPLORATION")
print("=" * 80)

# Basic info
print("\n1. DATASET STRUCTURE")
print("-" * 80)
print(f"Number of schools: {len(data)}")
print(f"Columns: {list(data.columns)}")
print(f"\nData types:\n{data.dtypes}")
print(f"\nDataset shape: {data.shape}")

# Display full dataset (it's small)
print("\n2. COMPLETE DATASET")
print("-" * 80)
print(data.to_string(index=False))

# Missing values check
print("\n3. DATA QUALITY CHECKS")
print("-" * 80)
print(f"Missing values:\n{data.isnull().sum()}")
print(f"\nAny duplicates: {data.duplicated().any()}")

# Check sigma values are positive
print(f"\nAll sigma values positive: {(data['sigma'] > 0).all()}")
print(f"Sigma range: [{data['sigma'].min()}, {data['sigma'].max()}]")

# Descriptive statistics
print("\n4. DESCRIPTIVE STATISTICS")
print("-" * 80)
print("\nEffect (Observed Treatment Effects):")
print(data['effect'].describe())
print(f"\nVariance of effects: {data['effect'].var():.2f}")
print(f"Standard deviation of effects: {data['effect'].std():.2f}")
print(f"Range: {data['effect'].max() - data['effect'].min():.2f}")
print(f"IQR: {data['effect'].quantile(0.75) - data['effect'].quantile(0.25):.2f}")

print("\nSigma (Standard Errors):")
print(data['sigma'].describe())
print(f"\nVariance of sigma: {data['sigma'].var():.2f}")
print(f"Standard deviation of sigma: {data['sigma'].std():.2f}")

# Variance components analysis
print("\n5. VARIANCE COMPONENTS ANALYSIS")
print("-" * 80)
between_school_var = data['effect'].var()
within_school_var_mean = (data['sigma']**2).mean()
within_school_var_median = (data['sigma']**2).median()

print(f"Between-school variance (empirical): {between_school_var:.2f}")
print(f"Mean within-school variance (sigma²): {within_school_var_mean:.2f}")
print(f"Median within-school variance (sigma²): {within_school_var_median:.2f}")
print(f"\nRatio (between/mean-within): {between_school_var/within_school_var_mean:.3f}")

# If ratio < 1, suggests pooling might be appropriate
# If ratio >> 1, suggests substantial between-school variation

# Weighted vs unweighted mean
unweighted_mean = data['effect'].mean()
weights = 1 / (data['sigma']**2)
weighted_mean = np.average(data['effect'], weights=weights)

print(f"\n6. POOLING ESTIMATES")
print("-" * 80)
print(f"Unweighted mean (complete pooling): {unweighted_mean:.2f}")
print(f"Weighted mean (inverse variance): {weighted_mean:.2f}")
print(f"Difference: {abs(weighted_mean - unweighted_mean):.2f}")

# Precision-weighted statistics
print(f"\nPrecision (1/sigma²) range: [{weights.min():.4f}, {weights.max():.4f}]")
print(f"Ratio of max to min precision: {weights.max()/weights.min():.2f}x")

# Identify extreme values
print("\n7. OUTLIER ANALYSIS")
print("-" * 80)

# Standardized effects (z-scores based on sample distribution)
z_scores = (data['effect'] - data['effect'].mean()) / data['effect'].std()
data_analysis = data.copy()
data_analysis['z_score'] = z_scores

# Schools with |z| > 1.5
outlier_threshold = 1.5
potential_outliers = data_analysis[abs(data_analysis['z_score']) > outlier_threshold]

print(f"Schools with |z-score| > {outlier_threshold}:")
if len(potential_outliers) > 0:
    print(potential_outliers[['school', 'effect', 'sigma', 'z_score']].to_string(index=False))
else:
    print("None")

# Effect relative to uncertainty
data_analysis['effect_to_sigma_ratio'] = data_analysis['effect'] / data_analysis['sigma']
print(f"\nEffect/Sigma ratios (signal-to-noise):")
print(data_analysis[['school', 'effect', 'sigma', 'effect_to_sigma_ratio']].to_string(index=False))

# Schools where effect is more than 2 sigma from 0
significant_schools = data_analysis[abs(data_analysis['effect_to_sigma_ratio']) > 2]
print(f"\nSchools with |effect| > 2*sigma (nominally significant): {len(significant_schools)}")
if len(significant_schools) > 0:
    print(significant_schools[['school', 'effect', 'sigma', 'effect_to_sigma_ratio']].to_string(index=False))

# Test for normality of effects
print("\n8. DISTRIBUTIONAL TESTS")
print("-" * 80)
shapiro_stat, shapiro_p = stats.shapiro(data['effect'])
print(f"Shapiro-Wilk test for normality of effects:")
print(f"  Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
print(f"  Interpretation: {'Consistent with normality' if shapiro_p > 0.05 else 'Evidence against normality'} (alpha=0.05)")

# Correlation between effect and sigma
corr_pearson, p_pearson = stats.pearsonr(data['effect'], data['sigma'])
corr_spearman, p_spearman = stats.spearmanr(data['effect'], data['sigma'])

print(f"\nCorrelation between effect and sigma:")
print(f"  Pearson r: {corr_pearson:.3f} (p={p_pearson:.3f})")
print(f"  Spearman rho: {corr_spearman:.3f} (p={p_spearman:.3f})")
print(f"  Interpretation: {'Significant' if p_pearson < 0.05 else 'No significant'} linear relationship")

print("\n" + "=" * 80)
print("INITIAL EXPLORATION COMPLETE")
print("=" * 80)
