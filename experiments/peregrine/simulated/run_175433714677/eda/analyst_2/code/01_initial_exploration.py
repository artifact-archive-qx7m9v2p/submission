"""
Initial Exploration: Count Distribution & Statistical Properties
Focus: Understanding basic distributional characteristics
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import poisson, nbinom
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
with open('/workspace/data/data_analyst_2.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({'C': data['C'], 'year': data['year']})
df['time_index'] = np.arange(len(df))

print("=" * 80)
print("INITIAL DATA EXPLORATION")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few observations:")
print(df.head(10))
print(f"\nLast few observations:")
print(df.tail(10))

print("\n" + "=" * 80)
print("BASIC STATISTICS")
print("=" * 80)

print(f"\nCount variable (C) summary:")
print(df['C'].describe())

print(f"\nAdditional statistics:")
print(f"Mean: {df['C'].mean():.2f}")
print(f"Variance: {df['C'].var(ddof=1):.2f}")
print(f"Std Dev: {df['C'].std(ddof=1):.2f}")
print(f"Variance-to-Mean Ratio: {df['C'].var(ddof=1) / df['C'].mean():.2f}")
print(f"Coefficient of Variation: {df['C'].std(ddof=1) / df['C'].mean():.2f}")

print(f"\nSkewness: {stats.skew(df['C']):.3f}")
print(f"Kurtosis (excess): {stats.kurtosis(df['C']):.3f}")

print(f"\nRange: {df['C'].min()} to {df['C'].max()}")
print(f"IQR: {df['C'].quantile(0.75) - df['C'].quantile(0.25):.2f}")

print("\n" + "=" * 80)
print("ZERO-INFLATION CHECK")
print("=" * 80)
print(f"Number of zeros: {(df['C'] == 0).sum()}")
print(f"Proportion of zeros: {(df['C'] == 0).mean():.4f}")
print(f"Expected zeros under Poisson(mean={df['C'].mean():.2f}): {poisson.pmf(0, df['C'].mean()) * len(df):.2f}")

print("\n" + "=" * 80)
print("DISTRIBUTION MOMENTS")
print("=" * 80)
print(f"1st moment (mean): {df['C'].mean():.2f}")
print(f"2nd moment (variance): {df['C'].var(ddof=1):.2f}")
print(f"3rd standardized moment (skewness): {stats.skew(df['C']):.2f}")
print(f"4th standardized moment (kurtosis): {stats.kurtosis(df['C'], fisher=True):.2f}")

print("\n" + "=" * 80)
print("OVERDISPERSION ANALYSIS")
print("=" * 80)
print(f"\nFor Poisson distribution, mean = variance")
print(f"Observed mean: {df['C'].mean():.2f}")
print(f"Observed variance: {df['C'].var(ddof=1):.2f}")
print(f"Variance/Mean ratio: {df['C'].var(ddof=1) / df['C'].mean():.2f}")
print(f"Interpretation: {'OVERDISPERSED (variance >> mean)' if df['C'].var(ddof=1) / df['C'].mean() > 1.5 else 'Approximately equidispersed' if 0.67 < df['C'].var(ddof=1) / df['C'].mean() < 1.5 else 'UNDERDISPERSED (variance << mean)'}")

# Calculate dispersion index (chi-square test for Poisson)
mean_count = df['C'].mean()
chi_sq_stat = ((df['C'] - mean_count)**2 / mean_count).sum()
df_test = len(df) - 1
p_value = 1 - stats.chi2.cdf(chi_sq_stat, df_test)
print(f"\nDispersion test (chi-square):")
print(f"  Chi-square statistic: {chi_sq_stat:.2f}")
print(f"  Degrees of freedom: {df_test}")
print(f"  p-value: {p_value:.4f}")
print(f"  Conclusion: {'Reject Poisson assumption (overdispersed)' if p_value < 0.05 else 'Cannot reject Poisson assumption'}")

print("\n" + "=" * 80)
print("TEMPORAL PATTERNS IN COUNTS")
print("=" * 80)
print("\nQuartile analysis:")
for q in [0, 0.25, 0.5, 0.75, 1.0]:
    idx = int(q * (len(df) - 1))
    print(f"  {q*100:.0f}% through time: C = {df.iloc[idx]['C']}, year = {df.iloc[idx]['year']:.2f}")

print("\nFirst half vs Second half:")
mid = len(df) // 2
first_half = df.iloc[:mid]['C']
second_half = df.iloc[mid:]['C']
print(f"  First half: mean = {first_half.mean():.2f}, var = {first_half.var(ddof=1):.2f}, var/mean = {first_half.var(ddof=1)/first_half.mean():.2f}")
print(f"  Second half: mean = {second_half.mean():.2f}, var = {second_half.var(ddof=1):.2f}, var/mean = {second_half.var(ddof=1)/second_half.mean():.2f}")

print("\n" + "=" * 80)
print("OUTLIER DETECTION")
print("=" * 80)
Q1 = df['C'].quantile(0.25)
Q3 = df['C'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['C'] < lower_bound) | (df['C'] > upper_bound)]
print(f"\nIQR method (1.5 * IQR):")
print(f"  Lower bound: {lower_bound:.2f}")
print(f"  Upper bound: {upper_bound:.2f}")
print(f"  Number of outliers: {len(outliers)}")
if len(outliers) > 0:
    print(f"  Outlier values: {outliers['C'].tolist()}")
    print(f"  Outlier indices: {outliers['time_index'].tolist()}")

# Z-score method
z_scores = np.abs(stats.zscore(df['C']))
z_outliers = df[z_scores > 2.5]
print(f"\nZ-score method (|z| > 2.5):")
print(f"  Number of outliers: {len(z_outliers)}")
if len(z_outliers) > 0:
    print(f"  Outlier values: {z_outliers['C'].tolist()}")
    print(f"  Outlier indices: {z_outliers['time_index'].tolist()}")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
