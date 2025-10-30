"""
Extreme Value and Alternative Distribution Analysis
Focus: Testing robustness and alternative model specifications
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import poisson, nbinom, gamma, lognorm
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
print("EXTREME VALUE AND ALTERNATIVE DISTRIBUTION ANALYSIS")
print("=" * 80)

# ============================================================================
# EXTREME VALUE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("EXTREME VALUE DETECTION AND CHARACTERIZATION")
print("=" * 80)

# Multiple methods for outlier detection
mean_C = df['C'].mean()
std_C = df['C'].std(ddof=1)

# Method 1: Z-score
z_scores = np.abs((df['C'] - mean_C) / std_C)
outliers_zscore = df[z_scores > 2.5]

# Method 2: IQR
Q1 = df['C'].quantile(0.25)
Q3 = df['C'].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df[(df['C'] < Q1 - 1.5*IQR) | (df['C'] > Q3 + 1.5*IQR)]

# Method 3: Modified Z-score (more robust)
median_C = df['C'].median()
mad = np.median(np.abs(df['C'] - median_C))
modified_z = 0.6745 * (df['C'] - median_C) / mad if mad > 0 else np.zeros(len(df))
outliers_modified_z = df[np.abs(modified_z) > 3.5]

print("\nOutlier Detection Results:")
print(f"  Z-score method (|z| > 2.5): {len(outliers_zscore)} outliers")
if len(outliers_zscore) > 0:
    print(f"    Values: {outliers_zscore['C'].tolist()}")
    print(f"    Time indices: {outliers_zscore['time_index'].tolist()}")

print(f"\n  IQR method (1.5*IQR): {len(outliers_iqr)} outliers")
if len(outliers_iqr) > 0:
    print(f"    Values: {outliers_iqr['C'].tolist()}")

print(f"\n  Modified Z-score (|mz| > 3.5): {len(outliers_modified_z)} outliers")
if len(outliers_modified_z) > 0:
    print(f"    Values: {outliers_modified_z['C'].tolist()}")

# Extreme value statistics
print("\nExtreme Value Statistics:")
print(f"  Maximum: {df['C'].max()} (at index {df['C'].idxmax()})")
print(f"  Minimum: {df['C'].min()} (at index {df['C'].idxmin()})")
print(f"  Range: {df['C'].max() - df['C'].min()}")
print(f"  Max/Min ratio: {df['C'].max() / df['C'].min():.2f}")

# Top 5 and bottom 5
print("\nTop 5 counts:")
top5 = df.nlargest(5, 'C')[['time_index', 'C', 'year']]
print(top5.to_string(index=False))

print("\nBottom 5 counts:")
bottom5 = df.nsmallest(5, 'C')[['time_index', 'C', 'year']]
print(bottom5.to_string(index=False))

# ============================================================================
# ALTERNATIVE DISTRIBUTIONS
# ============================================================================
print("\n" + "=" * 80)
print("TESTING ALTERNATIVE DISTRIBUTIONS")
print("=" * 80)

# Fit alternative continuous distributions for comparison
# Even though data is discrete, continuous approximations can be informative

# 1. Log-Normal Distribution
# Log-transform (add small constant if there are zeros)
log_data = np.log(df['C'])
mu_lognorm, std_lognorm = log_data.mean(), log_data.std()
shape_lognorm = std_lognorm
scale_lognorm = np.exp(mu_lognorm)

# 2. Gamma Distribution
# Fit gamma distribution
shape_gamma, loc_gamma, scale_gamma = gamma.fit(df['C'], floc=0)

print("\nAlternative Distribution Fits:")
print("\n1. LOG-NORMAL DISTRIBUTION:")
print(f"   Location parameter (mu): {mu_lognorm:.4f}")
print(f"   Scale parameter (sigma): {std_lognorm:.4f}")
print(f"   Implied mean: {np.exp(mu_lognorm + std_lognorm**2/2):.2f}")
print(f"   Implied variance: {(np.exp(std_lognorm**2) - 1) * np.exp(2*mu_lognorm + std_lognorm**2):.2f}")

# KS test for log-normal
ks_lognorm, pval_lognorm = stats.kstest(df['C'], lambda x: lognorm.cdf(x, std_lognorm, scale=scale_lognorm))
print(f"   KS test: statistic = {ks_lognorm:.4f}, p-value = {pval_lognorm:.6f}")

# Log-likelihood
ll_lognorm = np.sum(lognorm.logpdf(df['C'], std_lognorm, scale=scale_lognorm))
aic_lognorm = 2 * 2 - 2 * ll_lognorm
print(f"   Log-likelihood: {ll_lognorm:.2f}")
print(f"   AIC: {aic_lognorm:.2f}")

print("\n2. GAMMA DISTRIBUTION:")
print(f"   Shape parameter (alpha): {shape_gamma:.4f}")
print(f"   Scale parameter (beta): {scale_gamma:.4f}")
print(f"   Implied mean: {shape_gamma * scale_gamma:.2f}")
print(f"   Implied variance: {shape_gamma * scale_gamma**2:.2f}")

# KS test for gamma
ks_gamma, pval_gamma = stats.kstest(df['C'], lambda x: gamma.cdf(x, shape_gamma, loc=loc_gamma, scale=scale_gamma))
print(f"   KS test: statistic = {ks_gamma:.4f}, p-value = {pval_gamma:.6f}")

# Log-likelihood
ll_gamma = np.sum(gamma.logpdf(df['C'], shape_gamma, loc=loc_gamma, scale=scale_gamma))
aic_gamma = 2 * 2 - 2 * ll_gamma  # 2 parameters (shape, scale; loc fixed at 0)
print(f"   Log-likelihood: {ll_gamma:.2f}")
print(f"   AIC: {aic_gamma:.2f}")

# ============================================================================
# MODEL COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE MODEL COMPARISON")
print("=" * 80)

# Previously calculated
ll_poisson = np.sum(poisson.logpmf(df['C'], df['C'].mean()))
aic_poisson = 2 * 1 - 2 * ll_poisson

r_nb = 1.5493
p_nb = 0.0140
ll_nb = np.sum(nbinom.logpmf(df['C'], r_nb, p_nb))
aic_nb = 2 * 2 - 2 * ll_nb

print(f"\n{'Distribution':<25} {'Log-Lik':<12} {'AIC':<12} {'Parameters':<12} {'Type'}")
print("-" * 90)
print(f"{'Poisson':<25} {ll_poisson:>11.2f} {aic_poisson:>11.2f} {1:>11} {'Discrete'}")
print(f"{'Negative Binomial':<25} {ll_nb:>11.2f} {aic_nb:>11.2f} {2:>11} {'Discrete'}")
print(f"{'Log-Normal':<25} {ll_lognorm:>11.2f} {aic_lognorm:>11.2f} {2:>11} {'Continuous'}")
print(f"{'Gamma':<25} {ll_gamma:>11.2f} {aic_gamma:>11.2f} {2:>11} {'Continuous'}")

print("\nAIC Rankings (lower is better):")
models = [
    ('Poisson', aic_poisson),
    ('Negative Binomial', aic_nb),
    ('Log-Normal', aic_lognorm),
    ('Gamma', aic_gamma)
]
models_sorted = sorted(models, key=lambda x: x[1])
for i, (name, aic_val) in enumerate(models_sorted, 1):
    delta_aic = aic_val - models_sorted[0][1]
    print(f"  {i}. {name:<25} AIC = {aic_val:.2f}  (Delta = {delta_aic:.2f})")

# ============================================================================
# ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# Check if extreme values drive the overdispersion
print("\n1. IMPACT OF EXTREME VALUES:")

# Remove top 5% and recompute statistics
threshold_95 = df['C'].quantile(0.95)
df_trimmed = df[df['C'] <= threshold_95]

print(f"\n   Original data (n={len(df)}):")
print(f"     Mean: {df['C'].mean():.2f}")
print(f"     Variance: {df['C'].var(ddof=1):.2f}")
print(f"     Var/Mean: {df['C'].var(ddof=1) / df['C'].mean():.2f}")

print(f"\n   Trimmed data (removed top 5%, n={len(df_trimmed)}):")
print(f"     Mean: {df_trimmed['C'].mean():.2f}")
print(f"     Variance: {df_trimmed['C'].var(ddof=1):.2f}")
print(f"     Var/Mean: {df_trimmed['C'].var(ddof=1) / df_trimmed['C'].mean():.2f}")

print("\n   Conclusion:", end=" ")
if df_trimmed['C'].var(ddof=1) / df_trimmed['C'].mean() > 10:
    print("Overdispersion persists even after removing extremes")
else:
    print("Overdispersion is driven by extreme values")

# Check temporal stability of distribution parameters
print("\n2. TEMPORAL STABILITY:")
n_windows = 4
window_stats = []

for i in range(n_windows):
    start = i * len(df) // n_windows
    end = (i + 1) * len(df) // n_windows if i < n_windows - 1 else len(df)
    window_data = df['C'].iloc[start:end]

    mean_w = window_data.mean()
    var_w = window_data.var(ddof=1)

    # Fit NB to window
    r_w = mean_w**2 / (var_w - mean_w) if var_w > mean_w else 1

    window_stats.append({
        'window': i+1,
        'mean': mean_w,
        'variance': var_w,
        'var_mean_ratio': var_w / mean_w,
        'nb_r': r_w
    })

print(f"\n{'Window':<10} {'Mean':<12} {'Variance':<12} {'Var/Mean':<12} {'NB r':<12}")
print("-" * 70)
for ws in window_stats:
    print(f"{ws['window']:<10} {ws['mean']:>11.2f} {ws['variance']:>11.2f} "
          f"{ws['var_mean_ratio']:>11.2f} {ws['nb_r']:>11.2f}")

# Test for significant change in r parameter
r_values = [ws['nb_r'] for ws in window_stats]
print(f"\n   Range of r values: {min(r_values):.4f} to {max(r_values):.4f}")
print(f"   Ratio (max/min): {max(r_values)/min(r_values):.2f}")

if max(r_values)/min(r_values) > 3:
    print("   Conclusion: Dispersion parameter is NOT stable over time")
else:
    print("   Conclusion: Dispersion parameter is relatively stable")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
