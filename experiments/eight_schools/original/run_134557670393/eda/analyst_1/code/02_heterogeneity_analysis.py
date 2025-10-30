"""
Round 1 (continued): Heterogeneity Analysis
Focus: Assessing between-study variability and consistency
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load data
data = pd.read_csv('/workspace/eda/analyst_1/code/processed_data.csv')

print("="*70)
print("HETEROGENEITY ASSESSMENT")
print("="*70)

# Calculate weights (inverse variance)
data['weight'] = 1 / (data['sigma'] ** 2)
data['weighted_y'] = data['weight'] * data['y']

# Weighted mean (fixed effect estimate)
fixed_effect = data['weighted_y'].sum() / data['weight'].sum()
fixed_effect_se = np.sqrt(1 / data['weight'].sum())

print("\n1. FIXED EFFECT META-ANALYSIS")
print("-"*70)
print(f"Weighted mean effect: {fixed_effect:.3f}")
print(f"Standard error: {fixed_effect_se:.3f}")
print(f"95% CI: [{fixed_effect - 1.96*fixed_effect_se:.3f}, {fixed_effect + 1.96*fixed_effect_se:.3f}]")

# Simple unweighted mean for comparison
unweighted_mean = data['y'].mean()
unweighted_se = data['y'].sem()
print(f"\nUnweighted mean effect: {unweighted_mean:.3f}")
print(f"Standard error: {unweighted_se:.3f}")
print(f"Difference from weighted mean: {abs(fixed_effect - unweighted_mean):.3f}")

# Cochran's Q statistic
Q = sum(data['weight'] * (data['y'] - fixed_effect) ** 2)
df = len(data) - 1
p_value = 1 - stats.chi2.cdf(Q, df)

print("\n2. COCHRAN'S Q TEST FOR HETEROGENEITY")
print("-"*70)
print(f"Q statistic: {Q:.3f}")
print(f"Degrees of freedom: {df}")
print(f"P-value: {p_value:.4f}")
print(f"Critical value (alpha=0.05): {stats.chi2.ppf(0.95, df):.3f}")
print(f"Interpretation: {'Significant heterogeneity detected' if p_value < 0.05 else 'No significant heterogeneity'}")

# I-squared statistic
I_squared = max(0, 100 * (Q - df) / Q)
print("\n3. I-SQUARED STATISTIC")
print("-"*70)
print(f"I²: {I_squared:.1f}%")
if I_squared < 25:
    interpretation = "Low heterogeneity"
elif I_squared < 50:
    interpretation = "Moderate heterogeneity"
elif I_squared < 75:
    interpretation = "Substantial heterogeneity"
else:
    interpretation = "Considerable heterogeneity"
print(f"Interpretation: {interpretation}")
print("\nGuidelines (Higgins et al., 2003):")
print("  0-40%: Might not be important")
print("  30-60%: May represent moderate heterogeneity")
print("  50-90%: May represent substantial heterogeneity")
print("  75-100%: Considerable heterogeneity")

# H-squared statistic
H_squared = Q / df if df > 0 else 0
print("\n4. H-SQUARED STATISTIC")
print("-"*70)
print(f"H²: {H_squared:.3f}")
print(f"H: {np.sqrt(H_squared):.3f}")
print(f"Interpretation: H=1 indicates no heterogeneity, H>1.5 suggests heterogeneity")

# Tau-squared (DerSimonian-Laird estimator)
C = data['weight'].sum() - (data['weight'] ** 2).sum() / data['weight'].sum()
tau_squared = max(0, (Q - df) / C)
tau = np.sqrt(tau_squared)

print("\n5. BETWEEN-STUDY VARIANCE (TAU²)")
print("-"*70)
print(f"τ²: {tau_squared:.3f}")
print(f"τ: {tau:.3f}")
print(f"Interpretation: τ represents the standard deviation of true effects")

# Random effects estimate
random_weights = 1 / (data['sigma'] ** 2 + tau_squared)
random_effect = sum(random_weights * data['y']) / sum(random_weights)
random_effect_se = np.sqrt(1 / sum(random_weights))

print("\n6. RANDOM EFFECTS META-ANALYSIS")
print("-"*70)
print(f"Random effects mean: {random_effect:.3f}")
print(f"Standard error: {random_effect_se:.3f}")
print(f"95% CI: [{random_effect - 1.96*random_effect_se:.3f}, {random_effect + 1.96*random_effect_se:.3f}]")
print(f"Difference from fixed effect: {abs(random_effect - fixed_effect):.3f}")

# Prediction interval
prediction_se = np.sqrt(random_effect_se**2 + tau_squared)
prediction_lower = random_effect - 1.96 * prediction_se
prediction_upper = random_effect + 1.96 * prediction_se

print("\n7. PREDICTION INTERVAL")
print("-"*70)
print(f"95% Prediction interval: [{prediction_lower:.3f}, {prediction_upper:.3f}]")
print(f"Width: {prediction_upper - prediction_lower:.3f}")
print(f"Interpretation: Range in which we expect 95% of true effects to fall")
print(f"Note: {'Crosses zero' if prediction_lower < 0 < prediction_upper else 'Does not cross zero'}")

# Study-level analysis
print("\n8. STUDY-LEVEL CONTRIBUTION TO HETEROGENEITY")
print("-"*70)
study_contributions = data['weight'] * (data['y'] - fixed_effect) ** 2
data['Q_contribution'] = study_contributions
data['Q_contribution_pct'] = 100 * study_contributions / Q

print("\nStudies ranked by contribution to Q:")
ranked = data.sort_values('Q_contribution', ascending=False)
for idx, row in ranked.iterrows():
    print(f"  Study {row['study']}: {row['Q_contribution']:.3f} ({row['Q_contribution_pct']:.1f}%)")

# Identify studies outside prediction interval
print("\n9. STUDIES OUTSIDE PREDICTION INTERVAL")
print("-"*70)
outlier_studies = data[
    (data['y'] < prediction_lower) | (data['y'] > prediction_upper)
]
if len(outlier_studies) > 0:
    print(f"Number of studies: {len(outlier_studies)}")
    for idx, row in outlier_studies.iterrows():
        print(f"  Study {row['study']}: y={row['y']}, CI=[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]")
else:
    print("All studies fall within the prediction interval")

# Calculate standardized residuals
data['std_residual'] = (data['y'] - fixed_effect) / data['sigma']

print("\n10. STANDARDIZED RESIDUALS")
print("-"*70)
for idx, row in data.iterrows():
    print(f"  Study {row['study']}: {row['std_residual']:.3f}")

large_residuals = data[np.abs(data['std_residual']) > 2]
if len(large_residuals) > 0:
    print(f"\nStudies with |residual| > 2: {list(large_residuals['study'])}")
else:
    print("\nNo studies with |residual| > 2")

# Save heterogeneity results
results = {
    'fixed_effect': fixed_effect,
    'fixed_effect_se': fixed_effect_se,
    'random_effect': random_effect,
    'random_effect_se': random_effect_se,
    'Q': Q,
    'Q_pvalue': p_value,
    'I_squared': I_squared,
    'H_squared': H_squared,
    'tau_squared': tau_squared,
    'tau': tau,
    'prediction_lower': prediction_lower,
    'prediction_upper': prediction_upper
}

results_df = pd.DataFrame([results])
results_df.to_csv('/workspace/eda/analyst_1/code/heterogeneity_results.csv', index=False)

data.to_csv('/workspace/eda/analyst_1/code/processed_data_with_metrics.csv', index=False)

print("\n" + "="*70)
print("Results saved to: /workspace/eda/analyst_1/code/heterogeneity_results.csv")
print("="*70)
