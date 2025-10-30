"""
Hypothesis Testing for Meta-Analysis Dataset
=============================================
Tests competing hypotheses about data structure
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('/workspace/eda/code/processed_data.csv')

print("="*70)
print("HYPOTHESIS TESTING: Competing Models for Data Structure")
print("="*70)

# Calculate weighted mean
weighted_mean = np.sum(data['y'] * data['precision']) / np.sum(data['precision'])

print("\n" + "="*70)
print("HYPOTHESIS 1: Common Effect Model (No Heterogeneity)")
print("="*70)
print("Assumption: All studies measure the same underlying effect")
print("H0: tau² = 0 (no between-study variance)")
print()

# Under common effect model, each y_i ~ N(theta, sigma_i²)
# Test if observed spread is consistent with sampling error alone

# Calculate expected variance under common effect
weights = data['precision']**2
expected_var = 1 / np.sum(weights)

# Calculate observed variance of weighted effects
observed_var_weighted = np.sum(weights * (data['y'] - weighted_mean)**2) / np.sum(weights)

print(f"Weighted mean effect: {weighted_mean:.3f}")
print(f"Expected variance (under H0): {expected_var:.3f}")
print(f"Observed weighted variance: {observed_var_weighted:.3f}")
print(f"Ratio (observed/expected): {observed_var_weighted/expected_var:.3f}")

# Q test (already computed but let's be explicit)
Q = np.sum(weights * (data['y'] - weighted_mean)**2)
df = len(data) - 1
p_value_Q = 1 - stats.chi2.cdf(Q, df)

print(f"\nCochran's Q = {Q:.3f} (df={df})")
print(f"p-value = {p_value_Q:.4f}")
print(f"Critical value (α=0.05): {stats.chi2.ppf(0.95, df):.3f}")

if p_value_Q > 0.05:
    print("\nConclusion: CANNOT REJECT common effect model")
    print("Data is consistent with all studies measuring the same effect")
else:
    print("\nConclusion: REJECT common effect model")
    print("Significant heterogeneity detected")

print("\n" + "="*70)
print("HYPOTHESIS 2: Random Effects Model (Moderate Heterogeneity)")
print("="*70)
print("Assumption: Effects vary across studies, y_i ~ N(theta_i, sigma_i²)")
print("             theta_i ~ N(mu, tau²)")
print()

# DerSimonian-Laird estimator for tau²
tau_squared = max(0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
tau = np.sqrt(tau_squared)

print(f"Between-study variance (tau²): {tau_squared:.3f}")
print(f"Between-study SD (tau): {tau:.3f}")
print(f"Average within-study variance: {data['variance'].mean():.3f}")
print(f"Ratio tau²/avg(sigma²): {tau_squared/data['variance'].mean():.3f}")

# I² statistic interpretation
I_squared = max(0, 100 * (Q - df) / Q)
print(f"\nI² statistic: {I_squared:.1f}%")
print(f"Interpretation: {I_squared:.1f}% of variation is due to heterogeneity")
print(f"               {100-I_squared:.1f}% is due to sampling error")

# Prediction interval for new study
random_effects_var = tau_squared + (1/np.sum(weights))
random_effects_se = np.sqrt(random_effects_var)
pred_interval = (weighted_mean - 1.96*random_effects_se,
                 weighted_mean + 1.96*random_effects_se)

print(f"\n95% Prediction interval for new study: [{pred_interval[0]:.2f}, {pred_interval[1]:.2f}]")
print(f"Width: {pred_interval[1] - pred_interval[0]:.2f}")

print("\n" + "="*70)
print("HYPOTHESIS 3: Study-Specific Effects (High Heterogeneity)")
print("="*70)
print("Assumption: Each study measures a different effect (no pooling)")
print()

# Check if studies are so different that pooling is inappropriate
# Calculate range of confidence intervals
ci_ranges = []
for idx, row in data.iterrows():
    ci_lower = row['y'] - 1.96 * row['sigma']
    ci_upper = row['y'] + 1.96 * row['sigma']
    ci_ranges.append((ci_lower, ci_upper))

# Check for non-overlapping CIs
non_overlapping_pairs = 0
total_pairs = 0
for i in range(len(ci_ranges)):
    for j in range(i+1, len(ci_ranges)):
        total_pairs += 1
        if ci_ranges[i][1] < ci_ranges[j][0] or ci_ranges[j][1] < ci_ranges[i][0]:
            non_overlapping_pairs += 1
            print(f"  Study {data.iloc[i]['study']:.0f} and Study {data.iloc[j]['study']:.0f} have non-overlapping CIs")

print(f"\nNon-overlapping CI pairs: {non_overlapping_pairs} out of {total_pairs}")
print(f"Percentage: {100*non_overlapping_pairs/total_pairs:.1f}%")

if non_overlapping_pairs == 0:
    print("Conclusion: All CIs overlap - some degree of pooling seems reasonable")
elif non_overlapping_pairs / total_pairs < 0.3:
    print("Conclusion: Most CIs overlap - partial pooling appropriate")
else:
    print("Conclusion: Many non-overlapping CIs - consider no pooling or investigate subgroups")

print("\n" + "="*70)
print("HYPOTHESIS 4: Publication Bias / Small-Study Effects")
print("="*70)
print("Tests for relationship between effect size and study precision")
print()

# Egger's test: regression of standardized effect on precision
# standardized_effect = y/sigma, predictor = 1/sigma
standardized_effects = data['y'] / data['sigma']
precision = data['precision']

# Linear regression
slope, intercept, r_value, p_value_egger, std_err = stats.linregress(precision, standardized_effects)

print(f"Egger's regression test:")
print(f"  Standardized effect = {intercept:.3f} + {slope:.3f} × precision")
print(f"  Intercept (bias): {intercept:.3f} (SE: {std_err:.3f})")
print(f"  p-value: {p_value_egger:.4f}")

if p_value_egger < 0.05:
    print("  --> Significant asymmetry detected (p < 0.05)")
    print("  --> Possible small-study effects or publication bias")
else:
    print("  --> No significant asymmetry (p >= 0.05)")
    print("  --> No strong evidence for publication bias")

# Also test correlation between |y| and sigma (do smaller studies show larger effects?)
abs_effects = np.abs(data['y'])
corr_abs, p_corr_abs = stats.pearsonr(abs_effects, data['sigma'])
print(f"\nCorrelation between |effect| and SE: r = {corr_abs:.3f}, p = {p_corr_abs:.4f}")

# Correlation between y and sigma
corr_y_sigma, p_corr_y_sigma = stats.pearsonr(data['y'], data['sigma'])
print(f"Correlation between effect and SE: r = {corr_y_sigma:.3f}, p = {p_corr_y_sigma:.4f}")

if abs(corr_y_sigma) > 0.5 and p_corr_y_sigma < 0.05:
    print("  --> Significant correlation - investigate potential bias")
else:
    print("  --> No strong correlation between effect size and precision")

print("\n" + "="*70)
print("HYPOTHESIS 5: Outlier Influence")
print("="*70)
print("Tests sensitivity to potential outliers")
print()

# Identify potential outliers (more than 2 SD from mean)
z_scores = np.abs((data['y'] - data['y'].mean()) / data['y'].std())
potential_outliers = data[z_scores > 2]

print(f"Studies with |z-score| > 2:")
if len(potential_outliers) > 0:
    for idx, row in potential_outliers.iterrows():
        z = (row['y'] - data['y'].mean()) / data['y'].std()
        print(f"  Study {row['study']:.0f}: y={row['y']:.2f}, z={z:.2f}")
else:
    print("  None")

# Leave-one-out analysis
print("\nLeave-one-out sensitivity analysis:")
print(f"{'Study':<8} {'Full':<12} {'Without':<12} {'Difference':<12}")
print("-" * 50)

for idx, row in data.iterrows():
    study_id = int(row['study'])
    data_loo = data[data['study'] != study_id]

    # Recalculate weighted mean without this study
    weights_loo = data_loo['precision']**2
    weighted_mean_loo = np.sum(data_loo['y'] * weights_loo / weights_loo.sum())

    diff = weighted_mean - weighted_mean_loo
    diff_pct = 100 * diff / weighted_mean

    print(f"{study_id:<8} {weighted_mean:>11.3f} {weighted_mean_loo:>11.3f} {diff:>11.3f} ({diff_pct:+.1f}%)")

# Find most influential study
influences = []
for idx, row in data.iterrows():
    data_loo = data[data['study'] != row['study']]
    weights_loo = data_loo['precision']**2
    weighted_mean_loo = np.sum(data_loo['y'] * weights_loo / weights_loo.sum())
    influences.append(abs(weighted_mean - weighted_mean_loo))

most_influential_idx = np.argmax(influences)
most_influential_study = int(data.iloc[most_influential_idx]['study'])
max_influence = influences[most_influential_idx]

print(f"\nMost influential study: Study {most_influential_study}")
print(f"Maximum influence on pooled estimate: {max_influence:.3f}")

if max_influence / weighted_mean > 0.1:
    print("  --> Study has substantial influence (>10% change)")
else:
    print("  --> No single study dominates the pooled estimate")

print("\n" + "="*70)
print("SUMMARY OF HYPOTHESIS TESTS")
print("="*70)
print()
print("1. Common Effect Model:")
print(f"   Status: {'REJECTED' if p_value_Q < 0.05 else 'NOT REJECTED'}")
print(f"   Q-test p-value: {p_value_Q:.4f}")
print()
print("2. Random Effects Model:")
print(f"   Tau²: {tau_squared:.3f}")
print(f"   I²: {I_squared:.1f}%")
print(f"   Status: {'Appropriate' if I_squared > 0 and I_squared < 75 else 'Questionable'}")
print()
print("3. Study-Specific Effects:")
print(f"   Non-overlapping CIs: {100*non_overlapping_pairs/total_pairs:.1f}%")
print(f"   Status: {'Not needed' if non_overlapping_pairs == 0 else 'Consider'}")
print()
print("4. Publication Bias:")
print(f"   Egger's test p-value: {p_value_egger:.4f}")
print(f"   Status: {'Detected' if p_value_egger < 0.05 else 'Not detected'}")
print()
print("5. Outlier Influence:")
print(f"   Number of outliers (|z|>2): {len(potential_outliers)}")
print(f"   Max influence: {max_influence:.3f} ({100*max_influence/weighted_mean:.1f}%)")
print(f"   Status: {'Robust' if max_influence/weighted_mean < 0.1 else 'Sensitive'}")
print()
print("="*70)
