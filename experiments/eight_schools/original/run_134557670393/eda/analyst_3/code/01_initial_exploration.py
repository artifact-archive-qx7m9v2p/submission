"""
Initial Data Exploration - Meta-Analysis Structure & Context
Focus: Data structure, study ordering, extreme values, data quality
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
data = pd.read_csv('/workspace/data/data_analyst_3.csv')

print("="*80)
print("META-ANALYSIS DATASET: INITIAL EXPLORATION")
print("="*80)

# =============================================================================
# 1. BASIC STRUCTURE
# =============================================================================
print("\n" + "="*80)
print("1. BASIC DATA STRUCTURE")
print("="*80)

print(f"\nDataset shape: {data.shape}")
print(f"Number of studies (J): {len(data)}")
print(f"\nColumns: {list(data.columns)}")
print(f"Data types:\n{data.dtypes}")

print("\n--- First few rows ---")
print(data.head())

print("\n--- Last few rows ---")
print(data.tail())

print("\n--- Full dataset ---")
print(data)

# =============================================================================
# 2. DATA QUALITY CHECKS
# =============================================================================
print("\n" + "="*80)
print("2. DATA QUALITY ASSESSMENT")
print("="*80)

# Missing values
print("\n--- Missing Values ---")
print(f"Missing values per column:\n{data.isnull().sum()}")
print(f"Total missing values: {data.isnull().sum().sum()}")
print(f"Percentage missing: {100 * data.isnull().sum().sum() / (data.shape[0] * data.shape[1]):.2f}%")

# Check for duplicates
print("\n--- Duplicate Studies ---")
duplicates = data.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")

# Check study ID continuity
print("\n--- Study ID Continuity ---")
expected_ids = list(range(1, len(data) + 1))
actual_ids = sorted(data['study'].unique())
print(f"Expected study IDs: {expected_ids}")
print(f"Actual study IDs: {actual_ids}")
print(f"IDs continuous and complete: {expected_ids == actual_ids}")

# Check for negative or zero standard errors (implausible)
print("\n--- Standard Error Validity ---")
print(f"Minimum sigma: {data['sigma'].min()}")
print(f"Any sigma <= 0: {(data['sigma'] <= 0).any()}")
print(f"Any sigma == 0: {(data['sigma'] == 0).any()}")

# Check for infinite values
print("\n--- Infinite Values ---")
print(f"Infinite values in y: {np.isinf(data['y']).sum()}")
print(f"Infinite values in sigma: {np.isinf(data['sigma']).sum()}")

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("3. DESCRIPTIVE STATISTICS")
print("="*80)

print("\n--- Summary Statistics ---")
print(data.describe())

print("\n--- Additional Statistics ---")
print(f"\nEffect Size (y):")
print(f"  Mean: {data['y'].mean():.2f}")
print(f"  Median: {data['y'].median():.2f}")
print(f"  Std Dev: {data['y'].std():.2f}")
print(f"  Range: [{data['y'].min():.2f}, {data['y'].max():.2f}]")
print(f"  IQR: {data['y'].quantile(0.75) - data['y'].quantile(0.25):.2f}")
print(f"  Skewness: {stats.skew(data['y']):.2f}")
print(f"  Kurtosis: {stats.kurtosis(data['y']):.2f}")

print(f"\nStandard Error (sigma):")
print(f"  Mean: {data['sigma'].mean():.2f}")
print(f"  Median: {data['sigma'].median():.2f}")
print(f"  Std Dev: {data['sigma'].std():.2f}")
print(f"  Range: [{data['sigma'].min():.2f}, {data['sigma'].max():.2f}]")
print(f"  IQR: {data['sigma'].quantile(0.75) - data['sigma'].quantile(0.25):.2f}")
print(f"  Skewness: {stats.skew(data['sigma']):.2f}")
print(f"  Kurtosis: {stats.kurtosis(data['sigma']):.2f}")

# =============================================================================
# 4. EXTREME VALUE IDENTIFICATION
# =============================================================================
print("\n" + "="*80)
print("4. EXTREME VALUE ANALYSIS")
print("="*80)

# Z-scores for effect sizes
data['y_zscore'] = (data['y'] - data['y'].mean()) / data['y'].std()
data['sigma_zscore'] = (data['sigma'] - data['sigma'].mean()) / data['sigma'].std()

print("\n--- Z-scores (|z| > 2 considered extreme) ---")
print("\nEffect Size Z-scores:")
print(data[['study', 'y', 'y_zscore']].sort_values('y_zscore', key=abs, ascending=False))

print("\nStandard Error Z-scores:")
print(data[['study', 'sigma', 'sigma_zscore']].sort_values('sigma_zscore', key=abs, ascending=False))

# Identify extreme studies
extreme_y = data[np.abs(data['y_zscore']) > 2]
extreme_sigma = data[np.abs(data['sigma_zscore']) > 2]

print(f"\n--- Studies with Extreme Values ---")
print(f"Studies with |y_zscore| > 2: {list(extreme_y['study'].values)}")
print(f"Studies with |sigma_zscore| > 2: {list(extreme_sigma['study'].values)}")

# Precision-weighted analysis
data['precision'] = 1 / data['sigma']
data['weight'] = data['precision']**2

print("\n--- Precision Analysis ---")
print(data[['study', 'sigma', 'precision', 'weight']].sort_values('precision', ascending=False))

print(f"\nMost precise study (lowest sigma): Study {data.loc[data['sigma'].idxmin(), 'study']}")
print(f"Least precise study (highest sigma): Study {data.loc[data['sigma'].idxmax(), 'study']}")

# =============================================================================
# 5. STUDY ORDERING PATTERNS
# =============================================================================
print("\n" + "="*80)
print("5. STUDY ORDERING PATTERNS")
print("="*80)

# Correlations with study ID
y_corr = data['study'].corr(data['y'])
sigma_corr = data['study'].corr(data['sigma'])

print(f"\nCorrelation between study ID and y: {y_corr:.3f}")
print(f"Correlation between study ID and sigma: {sigma_corr:.3f}")

# Spearman correlation (non-parametric)
y_spearman = stats.spearmanr(data['study'], data['y'])
sigma_spearman = stats.spearmanr(data['study'], data['sigma'])

print(f"\nSpearman correlation (study ID vs y): rho={y_spearman.correlation:.3f}, p={y_spearman.pvalue:.3f}")
print(f"Spearman correlation (study ID vs sigma): rho={sigma_spearman.correlation:.3f}, p={sigma_spearman.pvalue:.3f}")

print("\n--- Potential Temporal/Quality Trends ---")
if abs(y_corr) > 0.3 or y_spearman.pvalue < 0.10:
    print("WARNING: Possible trend in effect sizes by study order")
else:
    print("No strong evidence of trends in effect sizes by study order")

if abs(sigma_corr) > 0.3 or sigma_spearman.pvalue < 0.10:
    print("WARNING: Possible trend in standard errors by study order")
else:
    print("No strong evidence of trends in standard errors by study order")

# =============================================================================
# 6. RELATIONSHIP BETWEEN Y AND SIGMA
# =============================================================================
print("\n" + "="*80)
print("6. RELATIONSHIP BETWEEN EFFECT SIZE AND STANDARD ERROR")
print("="*80)

corr_y_sigma = data['y'].corr(data['sigma'])
spearman_y_sigma = stats.spearmanr(data['y'], data['sigma'])

print(f"\nPearson correlation (y vs sigma): {corr_y_sigma:.3f}")
print(f"Spearman correlation (y vs sigma): rho={spearman_y_sigma.correlation:.3f}, p={spearman_y_sigma.pvalue:.3f}")

if abs(corr_y_sigma) > 0.5 or (abs(spearman_y_sigma.correlation) > 0.5 and spearman_y_sigma.pvalue < 0.10):
    print("\nWARNING: Strong correlation between effect size and standard error")
    print("This could indicate:")
    print("  - Small-study effects (publication bias)")
    print("  - Heteroscedasticity in the underlying process")
    print("  - Data quality issues")
else:
    print("\nNo strong correlation between effect size and standard error")

# =============================================================================
# 7. META-ANALYSIS CONTEXT
# =============================================================================
print("\n" + "="*80)
print("7. META-ANALYSIS CONTEXT & COMPARISON")
print("="*80)

print(f"\nSample size characteristics:")
print(f"  Number of studies (J): {len(data)}")
print(f"  Classification: ", end="")
if len(data) < 5:
    print("Very small meta-analysis (J < 5)")
elif len(data) < 10:
    print("Small meta-analysis (5 <= J < 10)")
elif len(data) < 20:
    print("Medium meta-analysis (10 <= J < 20)")
else:
    print("Large meta-analysis (J >= 20)")

print(f"\nPrecision heterogeneity:")
cv_sigma = data['sigma'].std() / data['sigma'].mean()
print(f"  Coefficient of variation (CV) for sigma: {cv_sigma:.2f}")
if cv_sigma > 0.5:
    print("  High heterogeneity in study precision")
elif cv_sigma > 0.3:
    print("  Moderate heterogeneity in study precision")
else:
    print("  Low heterogeneity in study precision")

print(f"\nEffect size heterogeneity:")
# Simple heterogeneity measure
I2_approx = max(0, 100 * (data['y'].var() - data['sigma'].mean()**2) / data['y'].var())
print(f"  Approximate I²: {I2_approx:.1f}%")

print(f"\nSign consistency:")
positive_effects = (data['y'] > 0).sum()
negative_effects = (data['y'] < 0).sum()
zero_effects = (data['y'] == 0).sum()
print(f"  Positive effects: {positive_effects} ({100*positive_effects/len(data):.1f}%)")
print(f"  Negative effects: {negative_effects} ({100*negative_effects/len(data):.1f}%)")
print(f"  Zero effects: {zero_effects}")

# =============================================================================
# 8. CONFIDENCE INTERVAL ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("8. CONFIDENCE INTERVAL ANALYSIS")
print("="*80)

# Calculate 95% CIs for each study
data['ci_lower'] = data['y'] - 1.96 * data['sigma']
data['ci_upper'] = data['y'] + 1.96 * data['sigma']
data['ci_width'] = data['ci_upper'] - data['ci_lower']

print("\n--- 95% Confidence Intervals ---")
print(data[['study', 'y', 'ci_lower', 'ci_upper', 'ci_width']].to_string(index=False))

# Check how many CIs include zero
includes_zero = ((data['ci_lower'] <= 0) & (data['ci_upper'] >= 0)).sum()
print(f"\nStudies with CIs including zero: {includes_zero}/{len(data)} ({100*includes_zero/len(data):.1f}%)")

# Check CI overlap
print("\nCI Width analysis:")
print(f"  Mean CI width: {data['ci_width'].mean():.2f}")
print(f"  Range: [{data['ci_width'].min():.2f}, {data['ci_width'].max():.2f}]")
print(f"  CV: {data['ci_width'].std() / data['ci_width'].mean():.2f}")

print("\n" + "="*80)
print("INITIAL EXPLORATION COMPLETE")
print("="*80)

# Save data with calculated fields
data.to_csv('/workspace/eda/analyst_3/code/data_with_diagnostics.csv', index=False)
print("\nEnhanced data saved to: /workspace/eda/analyst_3/code/data_with_diagnostics.csv")
