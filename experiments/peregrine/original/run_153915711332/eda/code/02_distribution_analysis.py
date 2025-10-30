"""
Distribution Analysis
=====================
Purpose: Understand the distribution of counts and test for overdispersion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/workspace/data/data.csv')
C = data['C'].values
year = data['year'].values

# Create comprehensive distribution plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution Analysis of Count Variable (C)', fontsize=16, y=1.00)

# 1. Histogram with KDE
ax = axes[0, 0]
ax.hist(C, bins=15, density=True, alpha=0.7, color='steelblue', edgecolor='black')
kde_x = np.linspace(C.min(), C.max(), 200)
kde = stats.gaussian_kde(C)
ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
ax.axvline(np.mean(C), color='darkred', linestyle='--', linewidth=2, label=f'Mean={np.mean(C):.1f}')
ax.axvline(np.median(C), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(C):.1f}')
ax.set_xlabel('Count (C)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Histogram with KDE', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Q-Q plot for normality
ax = axes[0, 1]
stats.probplot(C, dist="norm", plot=ax)
ax.set_title('Q-Q Plot vs Normal Distribution', fontsize=12)
ax.grid(True, alpha=0.3)

# 3. Box plot
ax = axes[0, 2]
bp = ax.boxplot(C, vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('black')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)
# Add points
ax.scatter(np.ones(len(C)) + np.random.normal(0, 0.02, len(C)), C,
           alpha=0.4, s=30, color='steelblue')
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Box Plot with Individual Points', fontsize=12)
ax.set_xticks([])
ax.grid(True, alpha=0.3, axis='y')

# Calculate outliers
Q1 = np.percentile(C, 25)
Q3 = np.percentile(C, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = C[(C < lower_bound) | (C > upper_bound)]
ax.text(0.5, 0.98, f'Outliers (1.5*IQR): {len(outliers)}',
        transform=ax.transAxes, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Empirical CDF
ax = axes[1, 0]
sorted_C = np.sort(C)
ecdf = np.arange(1, len(C) + 1) / len(C)
ax.step(sorted_C, ecdf, where='post', linewidth=2, color='steelblue')
ax.set_xlabel('Count (C)', fontsize=11)
ax.set_ylabel('Cumulative Probability', fontsize=11)
ax.set_title('Empirical Cumulative Distribution', fontsize=12)
ax.grid(True, alpha=0.3)
# Add quartiles
for q, label in [(0.25, 'Q1'), (0.5, 'Q2'), (0.75, 'Q3')]:
    val = np.percentile(C, q*100)
    ax.axhline(q, color='red', linestyle='--', alpha=0.5)
    ax.axvline(val, color='red', linestyle='--', alpha=0.5)
    ax.text(val, q, f' {label}={val:.0f}', fontsize=9)

# 5. Log-scale histogram
ax = axes[1, 1]
log_C = np.log(C)
ax.hist(log_C, bins=15, density=True, alpha=0.7, color='forestgreen', edgecolor='black')
kde_log = stats.gaussian_kde(log_C)
kde_x_log = np.linspace(log_C.min(), log_C.max(), 200)
ax.plot(kde_x_log, kde_log(kde_x_log), 'r-', linewidth=2, label='KDE')
ax.axvline(np.mean(log_C), color='darkred', linestyle='--', linewidth=2, label=f'Mean={np.mean(log_C):.2f}')
ax.set_xlabel('log(Count)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of log(C)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Count frequency table
ax = axes[1, 2]
unique_counts, frequencies = np.unique(C, return_counts=True)
# Show only if reasonable number of unique values
if len(unique_counts) <= 20:
    ax.bar(unique_counts, frequencies, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Count Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Count Frequency (n={len(unique_counts)} unique)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
else:
    # Too many unique values, show histogram instead
    ax.hist(C, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Count (C)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Count Histogram (n={len(unique_counts)} unique)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/01_distribution_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("="*80)
print("DISTRIBUTION ANALYSIS")
print("="*80)

# Normality tests
print("\n1. NORMALITY TESTS")
print("-"*80)
shapiro_stat, shapiro_p = stats.shapiro(C)
print(f"Shapiro-Wilk Test:")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.4e}")
print(f"  Conclusion: {'Reject normality' if shapiro_p < 0.05 else 'Cannot reject normality'} (α=0.05)")

ks_stat, ks_p = stats.kstest(C, 'norm', args=(np.mean(C), np.std(C, ddof=1)))
print(f"\nKolmogorov-Smirnov Test (vs Normal):")
print(f"  Statistic: {ks_stat:.4f}")
print(f"  P-value: {ks_p:.4e}")
print(f"  Conclusion: {'Reject normality' if ks_p < 0.05 else 'Cannot reject normality'} (α=0.05)")

# Log-normality test
log_shapiro_stat, log_shapiro_p = stats.shapiro(log_C)
print(f"\nShapiro-Wilk Test on log(C):")
print(f"  Statistic: {log_shapiro_stat:.4f}")
print(f"  P-value: {log_shapiro_p:.4e}")
print(f"  Conclusion: {'Reject log-normality' if log_shapiro_p < 0.05 else 'Cannot reject log-normality'} (α=0.05)")

# Overdispersion analysis
print("\n2. OVERDISPERSION ANALYSIS")
print("-"*80)
mean_C = np.mean(C)
var_C = np.var(C, ddof=1)
dispersion_ratio = var_C / mean_C

print(f"Mean: {mean_C:.2f}")
print(f"Variance: {var_C:.2f}")
print(f"Variance-to-Mean Ratio: {dispersion_ratio:.2f}")
print(f"\nInterpretation:")
if dispersion_ratio < 0.9:
    print(f"  UNDERDISPERSED (ratio < 1): Data is more regular than Poisson")
elif dispersion_ratio < 1.1:
    print(f"  EQUIDISPERSED (ratio ≈ 1): Consistent with Poisson")
else:
    print(f"  OVERDISPERSED (ratio >> 1): Variance greatly exceeds mean")
    print(f"  → Negative Binomial or quasi-Poisson models recommended")

# Index of dispersion test (approximate)
# Under Poisson assumption, (n-1)*s²/mean ~ Chi-square(n-1)
n = len(C)
test_stat = (n - 1) * var_C / mean_C
chi2_lower = stats.chi2.ppf(0.025, n - 1)
chi2_upper = stats.chi2.ppf(0.975, n - 1)

print(f"\nIndex of Dispersion Test:")
print(f"  Test statistic: {test_stat:.2f}")
print(f"  95% CI under Poisson: [{chi2_lower:.2f}, {chi2_upper:.2f}]")
print(f"  Conclusion: {'REJECT Poisson assumption' if (test_stat < chi2_lower or test_stat > chi2_upper) else 'Cannot reject Poisson'}")

# Coefficient of variation
cv = np.std(C, ddof=1) / mean_C
print(f"\nCoefficient of Variation: {cv:.2f}")
print(f"  (High CV indicates high relative variability)")

print("\n3. DISTRIBUTION SHAPE")
print("-"*80)
print(f"Skewness: {stats.skew(C):.3f}")
print(f"  Interpretation: {'Right-skewed' if stats.skew(C) > 0.5 else 'Left-skewed' if stats.skew(C) < -0.5 else 'Approximately symmetric'}")
print(f"\nKurtosis (excess): {stats.kurtosis(C):.3f}")
print(f"  Interpretation: {'Heavy tails' if stats.kurtosis(C) > 1 else 'Light tails' if stats.kurtosis(C) < -1 else 'Near-normal tails'}")

print("\n4. OUTLIER DETECTION")
print("-"*80)
print(f"IQR Method (1.5*IQR):")
print(f"  Q1: {Q1:.2f}")
print(f"  Q3: {Q3:.2f}")
print(f"  IQR: {IQR:.2f}")
print(f"  Lower bound: {lower_bound:.2f}")
print(f"  Upper bound: {upper_bound:.2f}")
print(f"  Number of outliers: {len(outliers)}")
if len(outliers) > 0:
    print(f"  Outlier values: {sorted(outliers)}")

# Z-score method
z_scores = np.abs(stats.zscore(C))
z_outliers = C[z_scores > 3]
print(f"\nZ-score Method (|z| > 3):")
print(f"  Number of outliers: {len(z_outliers)}")
if len(z_outliers) > 0:
    print(f"  Outlier values: {sorted(z_outliers)}")

print("\n" + "="*80)
print("Visualization saved: /workspace/eda/visualizations/01_distribution_analysis.png")
print("="*80)
