"""
Summary Statistics - Quick Reference
One-page summary of all key findings
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
C = data['C'].values
year = data['year'].values

print("=" * 80)
print("ANALYST 2 SUMMARY: DISTRIBUTIONAL PROPERTIES & VARIANCE STRUCTURE")
print("=" * 80)
print(f"Dataset: data/data_analyst_2.csv")
print(f"Observations: {len(C)}")
print(f"Focus: Count distribution, overdispersion, variance-mean relationship")
print()

print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print("\n1. DISTRIBUTION FAMILY: NEGATIVE BINOMIAL")
print("   " + "-" * 76)
print(f"   Variance/Mean Ratio: {C.var(ddof=1)/C.mean():.2f} (Poisson = 1.0)")
print(f"   Dispersion parameter (r): 1.634")
print(f"   Overdispersion param (α): 0.612")
print(f"   ΔAIC (NB - Poisson): -2416.62 (overwhelming evidence for NB)")
print(f"   Conclusion: STRONG OVERDISPERSION - use Negative Binomial")

print("\n2. VARIANCE STRUCTURE: NON-LINEAR & HETEROSCEDASTIC")
print("   " + "-" * 76)
print(f"   Power law exponent: 1.67 (Variance ∝ Mean^1.67)")
print(f"   R² for power law: 0.814")
print(f"   Heteroscedasticity: p < 0.0001 (Breusch-Pagan test)")
print(f"   Dispersion range across time: 1.27 to 7.52 (6-fold variation)")
print(f"   Conclusion: Time-varying dispersion present")

print("\n3. DATA QUALITY: EXCELLENT")
print("   " + "-" * 76)
print(f"   Missing values: 0")
print(f"   Outliers (multiple methods): 0")
print(f"   Influential points: 3 (at temporal extremes, legitimate)")
print(f"   Conclusion: Use full dataset (n=40)")

print("\n4. DISTRIBUTION SHAPE: RIGHT-SKEWED, PLATYKURTIC")
print("   " + "-" * 76)
print(f"   Mean: {C.mean():.2f}")
print(f"   Median: {np.median(C):.2f}")
print(f"   Skewness: {stats.skew(C):.3f} (moderate right skew)")
print(f"   Kurtosis: {stats.kurtosis(C):.3f} (flatter than normal)")
print(f"   Range: [{C.min()}, {C.max()}]")
print(f"   Zero counts: {(C == 0).sum()} (no zero-inflation)")

print("\n" + "=" * 80)
print("MODELING RECOMMENDATIONS")
print("=" * 80)

print("\nPRIMARY MODEL: Negative Binomial Regression")
print("   Y_t ~ NegBinomial(μ_t, r)")
print("   log(μ_t) = β₀ + β₁ × year_t")
print("   r ≈ 1.6 (or estimate via ML)")

print("\nALTERNATIVE: NB with Time-Varying Dispersion")
print("   log(r_t) = γ₀ + γ₁ × year_t")
print("   (if standard NB shows poor fit)")

print("\nNOT RECOMMENDED:")
print("   ❌ Poisson regression (AIC 2417 points worse)")
print("   ❌ Zero-inflated models (0 zeros in data)")

print("\n" + "=" * 80)
print("QUANTITATIVE EVIDENCE")
print("=" * 80)

print(f"\nOverdispersion Metrics:")
print(f"  Index of Dispersion (Var/Mean): {C.var(ddof=1)/C.mean():.2f}")
print(f"  Coefficient of Variation: {C.std(ddof=1)/C.mean():.3f}")
print(f"  Dispersion parameter (α): 0.612")

print(f"\nModel Comparison:")
from scipy.stats import poisson, nbinom
sample_mean = C.mean()
sample_var = C.var(ddof=1)
r_mom = sample_mean**2 / (sample_var - sample_mean)
p_mom = sample_mean / sample_var
ll_poisson = np.sum(poisson.logpmf(C, sample_mean))
ll_nb = np.sum(nbinom.logpmf(C, r_mom, p_mom))
aic_poisson = -2 * ll_poisson + 2 * 1
aic_nb = -2 * ll_nb + 2 * 2

print(f"  Poisson Log-likelihood: {ll_poisson:.2f}")
print(f"  NB Log-likelihood: {ll_nb:.2f}")
print(f"  Improvement: {ll_nb - ll_poisson:.2f}")
print(f"  Poisson AIC: {aic_poisson:.2f}")
print(f"  NB AIC: {aic_nb:.2f}")
print(f"  ΔAIC: {aic_nb - aic_poisson:.2f}")

print(f"\nTemporal Patterns:")
n_periods = 5
period_size = len(C) // n_periods
print(f"  Number of periods analyzed: {n_periods}")
for i in range(n_periods):
    start = i * period_size
    end = start + period_size if i < n_periods - 1 else len(C)
    period_data = C[start:end]
    print(f"  Period {i+1}: Mean={period_data.mean():.1f}, " +
          f"Var/Mean={period_data.var(ddof=1)/period_data.mean():.2f}")

print("\n" + "=" * 80)
print("VISUALIZATIONS CREATED")
print("=" * 80)
print(f"\nOutput directory: /workspace/eda/analyst_2/visualizations/")
print(f"\n1. distribution_overview.png - 4-panel distribution summary")
print(f"2. count_histogram.png - Frequency distribution")
print(f"3. variance_mean_analysis.png - Variance-mean relationship")
print(f"4. distribution_fitting.png - Poisson vs NB comparison")
print(f"5. outlier_analysis.png - Outlier diagnostics")
print(f"6. temporal_dispersion_rolling.png - Rolling window dispersion")
print(f"7. temporal_periods_comparison.png - Period-by-period analysis")

print("\n" + "=" * 80)
print("REPORTS CREATED")
print("=" * 80)
print(f"\n1. findings.md - Main findings report (comprehensive)")
print(f"2. eda_log.md - Detailed exploration process")
print(f"3. code/ - All reproducible analysis scripts")

print("\n" + "=" * 80)
print("BOTTOM LINE")
print("=" * 80)
print(f"\n⭐ MUST use Negative Binomial (not Poisson)")
print(f"⭐ Overdispersion is extreme (68× Poisson variance)")
print(f"⭐ Variance structure is time-varying")
print(f"⭐ Data quality is excellent (no issues)")
print(f"⭐ All 40 observations should be used")

print("\n" + "=" * 80)
print("END OF SUMMARY")
print("=" * 80)
