"""
Initial Exploration: Distributional Properties and Count Characteristics
EDA Analyst 1 - Round 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import poisson, nbinom
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

print("="*80)
print("INITIAL DATA OVERVIEW")
print("="*80)
print(f"\nDataset shape: {data.shape}")
print(f"\nColumn names: {data.columns.tolist()}")
print(f"\nData types:\n{data.dtypes}")
print(f"\nFirst few rows:\n{data.head(10)}")
print(f"\nLast few rows:\n{data.tail(10)}")

print("\n" + "="*80)
print("BASIC DESCRIPTIVE STATISTICS")
print("="*80)
print(f"\n{data.describe()}")

print("\n" + "="*80)
print("COUNT VARIABLE (C) - DETAILED STATISTICS")
print("="*80)
C = data['C'].values

# Basic statistics
print(f"\nCount: {len(C)}")
print(f"Min: {C.min()}")
print(f"Max: {C.max()}")
print(f"Range: {C.max() - C.min()}")
print(f"Mean: {C.mean():.4f}")
print(f"Median: {np.median(C):.4f}")
print(f"Std Dev: {C.std(ddof=1):.4f}")
print(f"Variance: {C.var(ddof=1):.4f}")

# Moments
print(f"\nSkewness: {stats.skew(C):.4f}")
print(f"Kurtosis (excess): {stats.kurtosis(C):.4f}")

# Quartiles and IQR
q1, q2, q3 = np.percentile(C, [25, 50, 75])
iqr = q3 - q1
print(f"\nQ1 (25th percentile): {q1:.2f}")
print(f"Q2 (50th percentile/median): {q2:.2f}")
print(f"Q3 (75th percentile): {q3:.2f}")
print(f"IQR: {iqr:.2f}")

# Percentiles
print(f"\nAdditional Percentiles:")
for p in [5, 10, 90, 95]:
    print(f"  {p}th percentile: {np.percentile(C, p):.2f}")

print("\n" + "="*80)
print("COUNT VARIABLE PROPERTIES")
print("="*80)

# Check for zeros
n_zeros = (C == 0).sum()
pct_zeros = n_zeros / len(C) * 100
print(f"\nNumber of zeros: {n_zeros} ({pct_zeros:.2f}%)")

# Check for negative values (shouldn't happen for counts)
n_negative = (C < 0).sum()
print(f"Number of negative values: {n_negative}")

# Check if all values are integers
all_integers = np.all(C == C.astype(int))
print(f"All values are integers: {all_integers}")

# Unique values
n_unique = len(np.unique(C))
print(f"Number of unique values: {n_unique}")

print("\n" + "="*80)
print("OVERDISPERSION ANALYSIS")
print("="*80)

mean_C = C.mean()
var_C = C.var(ddof=1)
dispersion_ratio = var_C / mean_C

print(f"\nMean: {mean_C:.4f}")
print(f"Variance: {var_C:.4f}")
print(f"Variance-to-Mean Ratio (Dispersion): {dispersion_ratio:.4f}")

if dispersion_ratio > 1:
    print(f"\n*** OVERDISPERSION DETECTED ***")
    print(f"Variance is {dispersion_ratio:.2f}x the mean")
    print("This suggests:")
    print("  - Poisson distribution may be inappropriate")
    print("  - Negative Binomial or other models may be better")
elif dispersion_ratio < 1:
    print(f"\n*** UNDERDISPERSION DETECTED ***")
    print(f"Variance is {dispersion_ratio:.2f}x the mean")
else:
    print(f"\nEquidispersion (variance ≈ mean)")
    print("Poisson distribution may be appropriate")

# Cameron-Trivedi test for overdispersion (manual calculation)
# Test if variance = mean (H0) vs variance > mean (H1)
# Using a simple chi-square approximation
chi2_stat = (len(C) - 1) * var_C / mean_C
df = len(C) - 1
p_value = 1 - stats.chi2.cdf(chi2_stat, df)
print(f"\nChi-square test for overdispersion:")
print(f"  Chi-square statistic: {chi2_stat:.4f}")
print(f"  Degrees of freedom: {df}")
print(f"  P-value: {p_value:.6f}")

print("\n" + "="*80)
print("OUTLIER DETECTION")
print("="*80)

# IQR method
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers_iqr = (C < lower_fence) | (C > upper_fence)
n_outliers_iqr = outliers_iqr.sum()

print(f"\nIQR Method (1.5 * IQR):")
print(f"  Lower fence: {lower_fence:.2f}")
print(f"  Upper fence: {upper_fence:.2f}")
print(f"  Number of outliers: {n_outliers_iqr}")
if n_outliers_iqr > 0:
    print(f"  Outlier values: {C[outliers_iqr]}")
    print(f"  Outlier indices: {np.where(outliers_iqr)[0]}")

# Z-score method
z_scores = np.abs(stats.zscore(C))
outliers_z = z_scores > 3
n_outliers_z = outliers_z.sum()

print(f"\nZ-score Method (|z| > 3):")
print(f"  Number of outliers: {n_outliers_z}")
if n_outliers_z > 0:
    print(f"  Outlier values: {C[outliers_z]}")
    print(f"  Outlier indices: {np.where(outliers_z)[0]}")
    print(f"  Z-scores: {z_scores[outliers_z]}")

print("\n" + "="*80)
print("TEMPORAL PATTERN (C vs Year)")
print("="*80)

year = data['year'].values
correlation = np.corrcoef(year, C)[0, 1]
print(f"\nPearson correlation (year, C): {correlation:.4f}")

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(year, C)
print(f"\nLinear regression:")
print(f"  Slope: {slope:.4f}")
print(f"  Intercept: {intercept:.4f}")
print(f"  R-squared: {r_value**2:.4f}")
print(f"  P-value: {p_value:.6f}")
print(f"  Std error: {std_err:.4f}")

# Spearman correlation (non-parametric)
spearman_corr, spearman_p = stats.spearmanr(year, C)
print(f"\nSpearman correlation: {spearman_corr:.4f} (p={spearman_p:.6f})")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
