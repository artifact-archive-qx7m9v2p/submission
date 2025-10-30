"""
Eight Schools Dataset - Hypothesis Testing and Modeling Considerations
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('/workspace/data/data.csv')
effects = data['effect'].values
sigmas = data['sigma'].values

print("=" * 80)
print("HYPOTHESIS TESTING AND MODELING CONSIDERATIONS")
print("=" * 80)

# ============================================================================
# HYPOTHESIS 1: Are all school effects equal? (Complete Pooling)
# ============================================================================
print("\nHYPOTHESIS 1: Complete Pooling (All effects are equal)")
print("-" * 80)

# Chi-square test for homogeneity
# If effects are truly equal, weighted sum of squared deviations follows chi-square
pooled_mean = np.average(effects, weights=1/(sigmas**2))
chi_sq_stat = np.sum((effects - pooled_mean)**2 / sigmas**2)
df = len(effects) - 1
p_value_homogeneity = 1 - stats.chi2.cdf(chi_sq_stat, df)

print(f"Weighted pooled estimate: {pooled_mean:.2f}")
print(f"\nChi-square test for homogeneity:")
print(f"  H0: All school effects are equal")
print(f"  Test statistic: {chi_sq_stat:.2f}")
print(f"  Degrees of freedom: {df}")
print(f"  p-value: {p_value_homogeneity:.4f}")

if p_value_homogeneity < 0.05:
    print(f"  Decision: REJECT H0 (alpha=0.05)")
    print(f"  Interpretation: Evidence against complete pooling")
else:
    print(f"  Decision: FAIL TO REJECT H0 (alpha=0.05)")
    print(f"  Interpretation: Data consistent with complete pooling")

# Q statistic (Cochran's Q)
Q_stat = chi_sq_stat  # Same as chi-square in this context
I_squared = max(0, (Q_stat - df) / Q_stat) * 100  # Heterogeneity measure

print(f"\nHeterogeneity assessment:")
print(f"  Q-statistic: {Q_stat:.2f}")
print(f"  I² statistic: {I_squared:.1f}%")
print(f"  Interpretation: {I_squared:.1f}% of variation due to heterogeneity")
if I_squared < 25:
    print("    -> Low heterogeneity")
elif I_squared < 50:
    print("    -> Moderate heterogeneity")
elif I_squared < 75:
    print("    -> Substantial heterogeneity")
else:
    print("    -> Considerable heterogeneity")

# ============================================================================
# HYPOTHESIS 2: Are effects completely independent? (No Pooling)
# ============================================================================
print("\n\nHYPOTHESIS 2: No Pooling (Effects are completely independent)")
print("-" * 80)

# Test for correlation in standardized effects
standardized_effects = (effects - effects.mean()) / effects.std()

print("If effects are truly independent with no common structure:")
print("  - Variance of effects should match sampling variance")
print("  - No systematic patterns in effect sizes")

# Variance ratio test
empirical_var = effects.var()
expected_var_under_independence = (sigmas**2).mean()
variance_ratio = empirical_var / expected_var_under_independence

print(f"\nEmpirical variance of effects: {empirical_var:.2f}")
print(f"Mean sampling variance (σ²): {expected_var_under_independence:.2f}")
print(f"Variance ratio: {variance_ratio:.3f}")

if variance_ratio < 0.5:
    print("  Interpretation: Observed variance LESS than expected")
    print("  -> Suggests effects may be more similar than independent draws")
    print("  -> Complete pooling might be appropriate")
elif variance_ratio > 1.5:
    print("  Interpretation: Observed variance MORE than expected")
    print("  -> Suggests true between-school variation exists")
    print("  -> Hierarchical model would be appropriate")
else:
    print("  Interpretation: Observed variance roughly matches expected")
    print("  -> Ambiguous case, partial pooling might help")

# Manual runs test for randomness (ordering test)
# Sort by school order and check for patterns
sorted_idx = np.argsort(data['school'].values)
sorted_effects = effects[sorted_idx]
median_effect = np.median(sorted_effects)
above = sorted_effects > median_effect

# Count runs
runs = 1
for i in range(1, len(above)):
    if above[i] != above[i-1]:
        runs += 1

n1 = np.sum(above)
n2 = len(above) - n1
expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))

print(f"\nRuns test for randomness (by school order):")
print(f"  Number of runs: {runs}")
print(f"  Expected runs: {expected_runs:.2f}")
print(f"  Z-statistic: {z_runs:.3f}")
print(f"  p-value: {p_runs:.3f}")
print(f"  Interpretation: {'Evidence of non-randomness' if p_runs < 0.05 else 'Consistent with random variation'}")

# ============================================================================
# HYPOTHESIS 3: Does uncertainty correlate with effect size?
# ============================================================================
print("\n\nHYPOTHESIS 3: Relationship between Effect Size and Uncertainty")
print("-" * 80)

print("Testing for heteroscedasticity patterns:")

# Correlation tests
corr_pearson, p_pearson = stats.pearsonr(effects, sigmas)
corr_spearman, p_spearman = stats.spearmanr(effects, sigmas)

print(f"\nCorrelation between effect and sigma:")
print(f"  Pearson correlation: {corr_pearson:.3f} (p={p_pearson:.3f})")
print(f"  Spearman correlation: {corr_spearman:.3f} (p={p_spearman:.3f})")

if p_pearson < 0.05:
    print(f"  Decision: Significant correlation detected")
    print(f"  Implication: Effect size and uncertainty are related")
    print(f"  -> May need to account for this in modeling")
else:
    print(f"  Decision: No significant correlation")
    print(f"  Implication: Effect size and uncertainty appear independent")
    print(f"  -> Standard hierarchical model assumptions reasonable")

# Test for publication bias / funnel plot asymmetry
# Egger's test: regress standardized effect on precision
precision = 1 / sigmas
standardized_effect = effects / sigmas
slope, intercept, r_value, p_value_egger, std_err = stats.linregress(precision, standardized_effect)

print(f"\nEgger's test for funnel plot asymmetry:")
print(f"  Intercept: {intercept:.3f}")
print(f"  p-value: {p_value_egger:.3f}")
if p_value_egger < 0.10:  # Liberal threshold for small sample
    print(f"  Decision: Evidence of asymmetry")
    print(f"  Interpretation: Possible publication bias or small-study effects")
else:
    print(f"  Decision: No evidence of asymmetry")
    print(f"  Interpretation: No obvious publication bias")

# ============================================================================
# HYPOTHESIS 4: Are effects normally distributed?
# ============================================================================
print("\n\nHYPOTHESIS 4: Normality of Effects Distribution")
print("-" * 80)

# Multiple normality tests
shapiro_stat, shapiro_p = stats.shapiro(effects)
anderson_result = stats.anderson(effects, dist='norm')
jarque_bera_stat, jarque_bera_p = stats.jarque_bera(effects)

print("Multiple tests for normality:")
print(f"\n1. Shapiro-Wilk test:")
print(f"   Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
print(f"   Decision: {'Reject normality' if shapiro_p < 0.05 else 'Consistent with normality'} (alpha=0.05)")

print(f"\n2. Anderson-Darling test:")
print(f"   Statistic: {anderson_result.statistic:.4f}")
for i, (crit, sig) in enumerate(zip(anderson_result.critical_values, anderson_result.significance_level)):
    result = "Reject" if anderson_result.statistic > crit else "Accept"
    print(f"   {sig}% significance: {result} (critical value: {crit:.3f})")

print(f"\n3. Jarque-Bera test:")
print(f"   Statistic: {jarque_bera_stat:.4f}, p-value: {jarque_bera_p:.4f}")
print(f"   Decision: {'Reject normality' if jarque_bera_p < 0.05 else 'Consistent with normality'} (alpha=0.05)")

# Skewness and kurtosis
skewness = stats.skew(effects)
kurtosis_excess = stats.kurtosis(effects)

print(f"\nDistributional moments:")
print(f"  Skewness: {skewness:.3f} (normal=0)")
print(f"  Excess kurtosis: {kurtosis_excess:.3f} (normal=0)")

if abs(skewness) < 0.5 and abs(kurtosis_excess) < 1:
    print(f"  Interpretation: Close to normal distribution")
elif abs(skewness) < 1 and abs(kurtosis_excess) < 2:
    print(f"  Interpretation: Approximately normal")
else:
    print(f"  Interpretation: Notable departure from normality")

# ============================================================================
# SUMMARY: Which Model is Most Appropriate?
# ============================================================================
print("\n" + "=" * 80)
print("MODELING RECOMMENDATIONS BASED ON HYPOTHESIS TESTS")
print("=" * 80)

recommendations = []

# Recommendation 1: Hierarchical vs alternatives
if p_value_homogeneity > 0.10 and variance_ratio < 0.8:
    recommendations.append(
        "1. COMPLETE POOLING may be appropriate:\n"
        "   - No strong evidence against homogeneity\n"
        "   - Low variance ratio suggests minimal between-school variation\n"
        "   - Model: theta_i = mu for all i"
    )
elif p_value_homogeneity < 0.05 and variance_ratio > 1.3:
    recommendations.append(
        "1. HIERARCHICAL MODEL (Partial Pooling) is recommended:\n"
        "   - Evidence against complete homogeneity\n"
        "   - Variance ratio suggests true between-school variation\n"
        "   - Model: theta_i ~ N(mu, tau^2), y_i ~ N(theta_i, sigma_i^2)"
    )
else:
    recommendations.append(
        "1. HIERARCHICAL MODEL (Partial Pooling) provides best balance:\n"
        "   - Ambiguous evidence from tests\n"
        "   - Partial pooling adapts based on variance components\n"
        "   - Model: theta_i ~ N(mu, tau^2), y_i ~ N(theta_i, sigma_i^2)"
    )

# Recommendation 2: Distributional assumptions
if shapiro_p > 0.10 and abs(skewness) < 0.5:
    recommendations.append(
        "2. NORMAL DISTRIBUTION assumptions are reasonable:\n"
        "   - Effects pass normality tests\n"
        "   - Standard Gaussian hierarchical model appropriate"
    )
else:
    recommendations.append(
        "2. Consider ROBUST alternatives:\n"
        "   - Some evidence against normality\n"
        "   - Consider t-distributed errors or mixture models"
    )

# Recommendation 3: Heteroscedasticity
if abs(corr_pearson) > 0.3 and p_pearson < 0.15:
    recommendations.append(
        "3. ACCOUNT FOR RELATIONSHIP between effect and sigma:\n"
        "   - Moderate correlation detected\n"
        "   - Consider modeling tau^2 as function of sigma_i"
    )
else:
    recommendations.append(
        "3. STANDARD VARIANCE STRUCTURE is adequate:\n"
        "   - No strong effect-uncertainty correlation\n"
        "   - Constant between-school variance tau^2 appropriate"
    )

print("\n")
for rec in recommendations:
    print(rec)
    print()

print("=" * 80)
print("HYPOTHESIS TESTING COMPLETE")
print("=" * 80)
