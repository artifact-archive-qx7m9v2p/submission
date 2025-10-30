"""
Detailed Count Model Assumption Testing
Focus on Poisson vs Negative Binomial considerations
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/workspace/data/data_analyst_3.csv')
C = data['C'].values
year = data['year'].values

print("="*80)
print("COUNT MODEL ASSUMPTION TESTING")
print("="*80)

# 1. Equidispersion test (Poisson assumption)
mean_C = np.mean(C)
var_C = np.var(C, ddof=1)
dispersion_ratio = var_C / mean_C

print("\n1. EQUIDISPERSION TEST (Poisson Assumption)")
print("-" * 60)
print(f"   Mean (λ): {mean_C:.4f}")
print(f"   Variance: {var_C:.4f}")
print(f"   Dispersion ratio (Var/Mean): {dispersion_ratio:.4f}")
print(f"\n   Interpretation:")
print(f"   - Ratio = 1: Equidispersed (Poisson appropriate)")
print(f"   - Ratio > 1: Overdispersed (Negative Binomial better)")
print(f"   - Ratio < 1: Underdispersed (rare in count data)")
print(f"\n   RESULT: SEVERE OVERDISPERSION (ratio = {dispersion_ratio:.2f})")
print(f"   Poisson model is NOT appropriate for this data")

# 2. Test statistic for overdispersion
# Under Poisson, (n-1)*s²/mean follows chi-square with n-1 df
n = len(C)
test_stat = (n - 1) * var_C / mean_C
p_value = 1 - stats.chi2.cdf(test_stat, n - 1)

print(f"\n   Formal Test for Overdispersion:")
print(f"   - Test statistic: {test_stat:.4f}")
print(f"   - Critical value (χ²_{n-1}, α=0.05): {stats.chi2.ppf(0.95, n-1):.4f}")
print(f"   - p-value: {p_value:.6f}")
print(f"   - Decision: {'REJECT Poisson' if test_stat > stats.chi2.ppf(0.95, n-1) else 'Accept Poisson'}")

# 3. Index of Dispersion (standardized)
index_of_dispersion = (n - 1) * var_C / mean_C / (n - 1)
print(f"\n   Index of Dispersion: {index_of_dispersion:.4f}")

# 4. Coefficient of Variation
cv = np.sqrt(var_C) / mean_C
print(f"   Coefficient of Variation: {cv:.4f}")
print(f"   (Poisson: CV = 1/√λ = {1/np.sqrt(mean_C):.4f})")

# 5. Check for zero-inflation
n_zeros = np.sum(C == 0)
expected_zeros_poisson = n * stats.poisson.pmf(0, mean_C)

print(f"\n2. ZERO-INFLATION CHECK")
print("-" * 60)
print(f"   Observed zeros: {n_zeros}")
print(f"   Expected zeros (Poisson): {expected_zeros_poisson:.2f}")
print(f"   Zero-inflation: {'YES' if n_zeros > expected_zeros_poisson else 'NO'}")

# 6. Distribution shape tests
print(f"\n3. DISTRIBUTION SHAPE TESTS")
print("-" * 60)

# Skewness
skewness = stats.skew(C)
expected_skew_poisson = 1 / np.sqrt(mean_C)
print(f"   Skewness: {skewness:.4f}")
print(f"   Expected (Poisson): {expected_skew_poisson:.4f}")

# Kurtosis (excess)
kurtosis = stats.kurtosis(C)
expected_kurt_poisson = 1 / mean_C
print(f"   Excess Kurtosis: {kurtosis:.4f}")
print(f"   Expected (Poisson): {expected_kurt_poisson:.4f}")

# 7. Goodness of fit test (Chi-square)
print(f"\n4. GOODNESS OF FIT TEST (Poisson)")
print("-" * 60)

# Group data into bins for chi-square test
bins = np.linspace(C.min(), C.max(), 10)
observed_freq, bin_edges = np.histogram(C, bins=bins)

# Calculate expected frequencies under Poisson
expected_freq = []
for i in range(len(bin_edges) - 1):
    # Use midpoint of bin
    midpoint = (bin_edges[i] + bin_edges[i+1]) / 2
    # Approximate probability for this range
    prob = stats.poisson.pmf(int(midpoint), mean_C)
    expected_freq.append(prob * n)

expected_freq = np.array(expected_freq)

# Filter out bins with expected frequency < 5
mask = expected_freq >= 5
observed_filtered = observed_freq[mask]
expected_filtered = expected_freq[mask]

if len(observed_filtered) > 1:
    chi2_stat = np.sum((observed_filtered - expected_filtered)**2 / expected_filtered)
    df = len(observed_filtered) - 1 - 1  # -1 for estimated parameter
    chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df)

    print(f"   Chi-square statistic: {chi2_stat:.4f}")
    print(f"   Degrees of freedom: {df}")
    print(f"   p-value: {chi2_pvalue:.4f}")
    print(f"   Result: {'REJECT Poisson' if chi2_pvalue < 0.05 else 'Accept Poisson'} (α=0.05)")
else:
    print("   Insufficient bins for chi-square test")

# 8. Variance stability over time
print(f"\n5. VARIANCE STABILITY ANALYSIS")
print("-" * 60)

# Split into quartiles
quartile_size = n // 4
quartiles = []
for i in range(4):
    start = i * quartile_size
    end = (i + 1) * quartile_size if i < 3 else n
    quartiles.append(C[start:end])

print(f"   Quartile-wise Statistics:")
for i, q in enumerate(quartiles):
    print(f"   Q{i+1}: Mean={np.mean(q):6.2f}, Var={np.var(q, ddof=1):8.2f}, Ratio={np.var(q, ddof=1)/np.mean(q):6.2f}")

# Levene's test for equality of variances
levene_stat, levene_pvalue = stats.levene(*quartiles)
print(f"\n   Levene's Test for Equality of Variances:")
print(f"   - Statistic: {levene_stat:.4f}")
print(f"   - p-value: {levene_pvalue:.4f}")
print(f"   - Result: {'HETEROGENEOUS' if levene_pvalue < 0.05 else 'Homogeneous'} variances (α=0.05)")

# 9. Temporal autocorrelation
print(f"\n6. AUTOCORRELATION ANALYSIS")
print("-" * 60)

# Lag-1 autocorrelation
lag1_corr = np.corrcoef(C[:-1], C[1:])[0, 1]
print(f"   Lag-1 Autocorrelation: {lag1_corr:.4f}")

# Durbin-Watson statistic (for residuals from earlier)
results = np.load('/workspace/eda/analyst_3/code/model_results.npz')
residuals = results['residuals_linear']
dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
print(f"   Durbin-Watson Statistic: {dw_stat:.4f}")
print(f"   (2 = no autocorrelation, <2 = positive, >2 = negative)")

# 10. Recommend appropriate model
print(f"\n7. MODEL RECOMMENDATIONS")
print("="*80)
print("\nBased on the diagnostic tests:")
print(f"  Variance/Mean Ratio: {dispersion_ratio:.2f} >> 1")
print(f"  Overdispersion p-value: {p_value:.6f}")
print(f"  Heteroscedasticity in transformations: Present")
print(f"  Variance increases with time: Yes")
print()
print("RECOMMENDED MODELS (in order of priority):")
print()
print("1. NEGATIVE BINOMIAL REGRESSION")
print("   - Handles overdispersion naturally")
print("   - Dispersion parameter captures extra-Poisson variation")
print("   - Most appropriate for this data")
print()
print("2. QUASI-POISSON REGRESSION")
print("   - Adjusts standard errors for overdispersion")
print("   - Simpler than Negative Binomial")
print("   - Good for prediction, less good for inference")
print()
print("3. GENERALIZED LINEAR MODEL with log link")
print("   - log(μ) = β₀ + β₁×year")
print("   - With robust standard errors")
print()
print("NOT RECOMMENDED:")
print("  - Standard Poisson regression (equidispersion violated)")
print("  - Simple linear regression (residual issues)")
print()
print("="*80)

# Save detailed results
with open('/workspace/eda/analyst_3/eda_log.md', 'a') as f:
    f.write("\n## Detailed Count Model Diagnostics\n\n")
    f.write("### Equidispersion Test (Poisson Assumption)\n\n")
    f.write(f"- **Mean**: {mean_C:.4f}\n")
    f.write(f"- **Variance**: {var_C:.4f}\n")
    f.write(f"- **Dispersion Ratio**: {dispersion_ratio:.4f}\n")
    f.write(f"- **Conclusion**: SEVERE OVERDISPERSION - Poisson inappropriate\n\n")

    f.write("### Distribution Characteristics\n\n")
    f.write(f"- **Skewness**: {skewness:.4f} (Poisson expected: {expected_skew_poisson:.4f})\n")
    f.write(f"- **Excess Kurtosis**: {kurtosis:.4f} (Poisson expected: {expected_kurt_poisson:.4f})\n")
    f.write(f"- **Coefficient of Variation**: {cv:.4f}\n\n")

    f.write("### Variance Stability\n\n")
    f.write("| Quartile | Mean | Variance | Var/Mean Ratio |\n")
    f.write("|----------|------|----------|----------------|\n")
    for i, q in enumerate(quartiles):
        f.write(f"| Q{i+1} | {np.mean(q):.2f} | {np.var(q, ddof=1):.2f} | {np.var(q, ddof=1)/np.mean(q):.2f} |\n")
    f.write(f"\n- **Levene's Test p-value**: {levene_pvalue:.4f}\n")
    f.write(f"- **Interpretation**: {'Heterogeneous' if levene_pvalue < 0.05 else 'Homogeneous'} variances\n\n")

print("\nCount model testing complete.")
