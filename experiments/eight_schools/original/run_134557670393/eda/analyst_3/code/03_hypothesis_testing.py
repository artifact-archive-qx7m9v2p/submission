"""
Hypothesis Testing and Contextual Analysis
Testing competing hypotheses about data structure
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/workspace/eda/analyst_3/code/data_with_diagnostics.csv')

print("="*80)
print("HYPOTHESIS TESTING & CONTEXTUAL ANALYSIS")
print("="*80)

# =============================================================================
# HYPOTHESIS 1: Is there evidence of publication bias?
# =============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 1: PUBLICATION BIAS DETECTION")
print("="*80)

print("\nH1a: Small-study effects (correlation between effect size and SE)")
print("-" * 70)

# Test 1: Correlation test
y_sigma_corr = data['y'].corr(data['sigma'])
y_sigma_spearman = stats.spearmanr(data['y'], data['sigma'])

print(f"Pearson correlation (y vs sigma): r = {y_sigma_corr:.3f}")
print(f"Spearman correlation (y vs sigma): rho = {y_sigma_spearman.correlation:.3f}, p = {y_sigma_spearman.pvalue:.3f}")

if y_sigma_spearman.pvalue < 0.10:
    print("\nResult: Some evidence of correlation (p < 0.10)")
    print("Interpretation: Possible small-study effects, but weak with J=8")
else:
    print("\nResult: No significant correlation (p >= 0.10)")
    print("Interpretation: No strong evidence of small-study effects")

# Test 2: Egger's regression test (intercept test)
print("\nH1b: Egger's regression test for funnel plot asymmetry")
print("-" * 70)

# Standard Egger's: regress standardized effect on precision
data['std_effect'] = data['y'] / data['sigma']  # Standardized effect
data['inv_se'] = 1 / data['sigma']  # Precision

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(data['inv_se'], data['std_effect'])

print(f"Egger's intercept: {intercept:.3f}")
print(f"Egger's p-value: {p_value:.3f}")
print(f"Standard error: {std_err:.3f}")

if p_value < 0.10:
    print("\nResult: Evidence of funnel asymmetry (p < 0.10)")
    print("Interpretation: Possible publication bias")
else:
    print("\nResult: No significant funnel asymmetry (p >= 0.10)")
    print("Interpretation: Limited evidence of publication bias")

print("\nCAVEAT: With J=8, power to detect publication bias is very low.")
print("These tests are unreliable with small meta-analyses.")

# =============================================================================
# HYPOTHESIS 2: Is there true heterogeneity vs. sampling variation?
# =============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 2: HETEROGENEITY ASSESSMENT")
print("="*80)

print("\nH2: Cochran's Q test for heterogeneity")
print("-" * 70)

# Calculate Q statistic
weights = 1 / (data['sigma']**2)
weighted_mean = np.sum(weights * data['y']) / np.sum(weights)
Q = np.sum(weights * (data['y'] - weighted_mean)**2)
df = len(data) - 1
p_value_Q = 1 - stats.chi2.cdf(Q, df)

print(f"Cochran's Q: {Q:.3f}")
print(f"Degrees of freedom: {df}")
print(f"p-value: {p_value_Q:.3f}")

if p_value_Q < 0.10:
    print("\nResult: Significant heterogeneity detected (p < 0.10)")
    print("Interpretation: Effect sizes vary more than expected by chance")
else:
    print("\nResult: No significant heterogeneity detected (p >= 0.10)")
    print("Interpretation: Variation consistent with sampling error")

# I² statistic
I2 = max(0, 100 * (Q - df) / Q) if Q > 0 else 0
print(f"\nI² statistic: {I2:.1f}%")
if I2 > 75:
    print("Interpretation: High heterogeneity")
elif I2 > 50:
    print("Interpretation: Moderate heterogeneity")
elif I2 > 25:
    print("Interpretation: Low heterogeneity")
else:
    print("Interpretation: Minimal heterogeneity")

# Tau² (between-study variance)
tau2 = max(0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
print(f"Tau² (between-study variance): {tau2:.3f}")

# =============================================================================
# HYPOTHESIS 3: Are there study quality/temporal trends?
# =============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 3: TEMPORAL/QUALITY TRENDS")
print("="*80)

print("\nH3a: Trend in effect sizes over study sequence")
print("-" * 70)

# Linear regression: y ~ study
from scipy.stats import linregress
slope_y, intercept_y, r_y, p_y, se_y = linregress(data['study'], data['y'])

print(f"Slope (effect per study): {slope_y:.3f}")
print(f"R²: {r_y**2:.3f}")
print(f"p-value: {p_y:.3f}")

if p_y < 0.10:
    if slope_y > 0:
        print("\nResult: Significant positive trend (p < 0.10)")
        print("Interpretation: Effect sizes increase with study order")
    else:
        print("\nResult: Significant negative trend (p < 0.10)")
        print("Interpretation: Effect sizes decrease with study order")
else:
    print("\nResult: No significant trend (p >= 0.10)")
    print("Interpretation: No evidence of temporal/quality effects")

print("\nH3b: Trend in precision over study sequence")
print("-" * 70)

slope_prec, intercept_prec, r_prec, p_prec, se_prec = linregress(data['study'], data['precision'])

print(f"Slope (precision per study): {slope_prec:.5f}")
print(f"R²: {r_prec**2:.3f}")
print(f"p-value: {p_prec:.3f}")

if p_prec < 0.10:
    if slope_prec > 0:
        print("\nResult: Significant improvement in precision (p < 0.10)")
        print("Interpretation: Later studies are more precise (better quality)")
    else:
        print("\nResult: Significant decline in precision (p < 0.10)")
        print("Interpretation: Later studies are less precise (declining quality?)")
else:
    print("\nResult: No significant trend (p >= 0.10)")
    print("Interpretation: No evidence of quality changes over time")

# =============================================================================
# HYPOTHESIS 4: Is the effect significantly different from zero?
# =============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 4: OVERALL EFFECT TESTING")
print("="*80)

print("\nH4: Fixed-effect meta-analysis test")
print("-" * 70)

# Fixed-effect model
weights = 1 / (data['sigma']**2)
fixed_effect = np.sum(weights * data['y']) / np.sum(weights)
fixed_se = np.sqrt(1 / np.sum(weights))
fixed_z = fixed_effect / fixed_se
fixed_p = 2 * (1 - stats.norm.cdf(abs(fixed_z)))

print(f"Fixed-effect estimate: {fixed_effect:.3f}")
print(f"Standard error: {fixed_se:.3f}")
print(f"95% CI: [{fixed_effect - 1.96*fixed_se:.3f}, {fixed_effect + 1.96*fixed_se:.3f}]")
print(f"Z-score: {fixed_z:.3f}")
print(f"p-value: {fixed_p:.4f}")

if fixed_p < 0.05:
    print("\nResult: Significant effect (p < 0.05)")
    print("Interpretation: Overall effect is significantly different from zero")
else:
    print("\nResult: Non-significant effect (p >= 0.05)")
    print("Interpretation: Cannot reject null hypothesis of no effect")

# Random-effects model (DerSimonian-Laird)
print("\nH4b: Random-effects meta-analysis test")
print("-" * 70)

# Calculate tau² (already done above)
random_weights = 1 / (data['sigma']**2 + tau2)
random_effect = np.sum(random_weights * data['y']) / np.sum(random_weights)
random_se = np.sqrt(1 / np.sum(random_weights))
random_z = random_effect / random_se
random_p = 2 * (1 - stats.norm.cdf(abs(random_z)))

print(f"Random-effect estimate: {random_effect:.3f}")
print(f"Standard error: {random_se:.3f}")
print(f"95% CI: [{random_effect - 1.96*random_se:.3f}, {random_effect + 1.96*random_se:.3f}]")
print(f"Z-score: {random_z:.3f}")
print(f"p-value: {random_p:.4f}")

if random_p < 0.05:
    print("\nResult: Significant effect (p < 0.05)")
    print("Interpretation: Overall effect is significantly different from zero")
else:
    print("\nResult: Non-significant effect (p >= 0.05)")
    print("Interpretation: Cannot reject null hypothesis of no effect")

# =============================================================================
# HYPOTHESIS 5: Are there influential studies (outliers)?
# =============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 5: INFLUENTIAL STUDIES & OUTLIERS")
print("="*80)

print("\nH5: Leave-one-out sensitivity analysis")
print("-" * 70)

print("\nFixed-effect estimates with each study removed:")
print(f"{'Study':<8} {'Estimate':<10} {'SE':<10} {'95% CI':<25} {'Change':<10}")
print("-" * 70)

for i in range(len(data)):
    # Remove study i
    temp_data = data.drop(i)
    temp_weights = 1 / (temp_data['sigma']**2)
    temp_effect = np.sum(temp_weights * temp_data['y']) / np.sum(temp_weights)
    temp_se = np.sqrt(1 / np.sum(temp_weights))
    temp_ci_lower = temp_effect - 1.96 * temp_se
    temp_ci_upper = temp_effect + 1.96 * temp_se
    change = temp_effect - fixed_effect

    print(f"{int(data.loc[i, 'study']):<8} {temp_effect:<10.3f} {temp_se:<10.3f} "
          f"[{temp_ci_lower:>6.3f}, {temp_ci_upper:>6.3f}] {change:>9.3f}")

# Identify most influential study
influences = []
for i in range(len(data)):
    temp_data = data.drop(i)
    temp_weights = 1 / (temp_data['sigma']**2)
    temp_effect = np.sum(temp_weights * temp_data['y']) / np.sum(temp_weights)
    influences.append(abs(temp_effect - fixed_effect))

most_influential = data.loc[np.argmax(influences), 'study']
max_influence = max(influences)

print(f"\nMost influential study: Study {int(most_influential)}")
print(f"Maximum influence: {max_influence:.3f}")
print(f"As % of fixed effect: {100 * max_influence / abs(fixed_effect):.1f}%")

# =============================================================================
# HYPOTHESIS 6: Sample size adequacy
# =============================================================================
print("\n" + "="*80)
print("HYPOTHESIS 6: SAMPLE SIZE & POWER CONSIDERATIONS")
print("="*80)

print("\nH6: Is J=8 adequate for this meta-analysis?")
print("-" * 70)

J = len(data)
print(f"Number of studies (J): {J}")

print("\nGeneral guidelines:")
print("  - J < 5: Too small for reliable meta-analysis")
print("  - 5 <= J < 10: Small, limited power for heterogeneity tests")
print("  - 10 <= J < 20: Medium, reasonable for basic meta-analysis")
print("  - J >= 20: Large, good power for most analyses")

print(f"\nCurrent classification: Small meta-analysis (J={J})")

print("\nImplications:")
print("  1. Low power to detect heterogeneity (Q test unreliable)")
print("  2. Low power to detect publication bias (Egger test unreliable)")
print("  3. Random-effects estimates may be unstable")
print("  4. Subgroup analyses not recommended")
print("  5. Meta-regression not feasible")
print("  6. Sensitivity analyses limited")

# Precision analysis
total_weight = np.sum(1 / data['sigma']**2)
avg_precision = data['precision'].mean()

print(f"\nPrecision characteristics:")
print(f"  Total weight (sum of 1/sigma²): {total_weight:.3f}")
print(f"  Average precision: {avg_precision:.3f}")
print(f"  Effective sample size: ~{total_weight:.1f} (precision-adjusted)")

# =============================================================================
# SUMMARY OF FINDINGS
# =============================================================================
print("\n" + "="*80)
print("SUMMARY OF HYPOTHESIS TESTING")
print("="*80)

print("\n1. PUBLICATION BIAS:")
if abs(y_sigma_corr) < 0.3 and y_sigma_spearman.pvalue >= 0.10:
    print("   - No strong evidence detected")
else:
    print("   - Some evidence, but J=8 is too small for reliable assessment")

print("\n2. HETEROGENEITY:")
if p_value_Q < 0.10:
    print(f"   - Significant heterogeneity detected (Q={Q:.2f}, p={p_value_Q:.3f})")
    print(f"   - I²={I2:.1f}%, suggesting true between-study variation")
else:
    print(f"   - No significant heterogeneity (Q={Q:.2f}, p={p_value_Q:.3f})")
    print(f"   - I²={I2:.1f}%, variation consistent with sampling error")

print("\n3. TEMPORAL/QUALITY TRENDS:")
if p_y < 0.10 or p_prec < 0.10:
    print("   - Some evidence of trends over study sequence")
else:
    print("   - No evidence of temporal or quality trends")

print("\n4. OVERALL EFFECT:")
if fixed_p < 0.05:
    print(f"   - Significant overall effect (Fixed: {fixed_effect:.2f}, p={fixed_p:.4f})")
else:
    print(f"   - Non-significant overall effect (Fixed: {fixed_effect:.2f}, p={fixed_p:.4f})")

print("\n5. INFLUENTIAL STUDIES:")
if max_influence > 0.3 * abs(fixed_effect):
    print(f"   - Study {int(most_influential)} is highly influential")
else:
    print("   - No single study dominates the meta-analysis")

print("\n6. SAMPLE SIZE:")
print("   - J=8 is small but acceptable for basic meta-analysis")
print("   - Limited power for complex analyses")
print("   - Results should be interpreted cautiously")

print("\n" + "="*80)
print("HYPOTHESIS TESTING COMPLETE")
print("="*80)
