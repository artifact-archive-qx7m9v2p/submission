"""
Round 2 Analysis: Detrending and Residual Analysis
EDA Analyst 1
Testing competing hypotheses about overdispersion sources
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')
C = data['C'].values
year = data['year'].values

print("="*80)
print("ROUND 2: DETRENDING AND RESIDUAL ANALYSIS")
print("="*80)

# ============================================================================
# Linear detrending
# ============================================================================
print("\n" + "="*80)
print("LINEAR DETRENDING ANALYSIS")
print("="*80)

# Fit linear model
slope, intercept, r_value, p_value, std_err = stats.linregress(year, C)
predicted = slope * year + intercept
residuals_linear = C - predicted

print(f"\nLinear Model: C = {slope:.4f} * year + {intercept:.4f}")
print(f"R-squared: {r_value**2:.4f}")

print(f"\nOriginal C statistics:")
print(f"  Mean: {C.mean():.4f}")
print(f"  Variance: {C.var(ddof=1):.4f}")
print(f"  Std Dev: {C.std(ddof=1):.4f}")
print(f"  Var/Mean ratio: {C.var(ddof=1) / C.mean():.4f}")

print(f"\nLinear Residuals statistics:")
print(f"  Mean: {residuals_linear.mean():.4f}")
print(f"  Variance: {residuals_linear.var(ddof=1):.4f}")
print(f"  Std Dev: {residuals_linear.std(ddof=1):.4f}")

# For residuals, variance-to-mean ratio doesn't make sense if mean ≈ 0
# Instead, compare variance to predicted mean
residual_var_ratio = residuals_linear.var(ddof=1) / predicted.mean()
print(f"  Residual Var / Predicted Mean: {residual_var_ratio:.4f}")

print(f"\nVariance reduction:")
original_var = C.var(ddof=1)
residual_var = residuals_linear.var(ddof=1)
var_reduction_pct = (1 - residual_var / original_var) * 100
print(f"  Original variance: {original_var:.4f}")
print(f"  Residual variance: {residual_var:.4f}")
print(f"  Variance explained by trend: {var_reduction_pct:.2f}%")

# Test if residuals are normally distributed
shapiro_stat, shapiro_p = stats.shapiro(residuals_linear)
print(f"\nShapiro-Wilk test for normality of residuals:")
print(f"  W statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.6f}")
if shapiro_p > 0.05:
    print("  Result: Cannot reject normality (p > 0.05)")
else:
    print("  Result: Residuals are NOT normally distributed (p < 0.05)")

# ============================================================================
# Heteroscedasticity analysis
# ============================================================================
print("\n" + "="*80)
print("HETEROSCEDASTICITY ANALYSIS")
print("="*80)

# Split into quartiles by predicted values
quartiles = np.percentile(predicted, [25, 50, 75])
q1_mask = predicted < quartiles[0]
q2_mask = (predicted >= quartiles[0]) & (predicted < quartiles[1])
q3_mask = (predicted >= quartiles[1]) & (predicted < quartiles[2])
q4_mask = predicted >= quartiles[2]

print("\nResidual variance by predicted value quartiles:")
for i, (mask, label) in enumerate([(q1_mask, 'Q1 (lowest)'), (q2_mask, 'Q2'),
                                      (q3_mask, 'Q3'), (q4_mask, 'Q4 (highest)')]):
    q_residuals = residuals_linear[mask]
    q_predicted = predicted[mask]
    print(f"\n{label}:")
    print(f"  N observations: {mask.sum()}")
    print(f"  Mean predicted: {q_predicted.mean():.2f}")
    print(f"  Residual variance: {q_residuals.var(ddof=1):.2f}")
    print(f"  Residual std dev: {q_residuals.std(ddof=1):.2f}")

# Breusch-Pagan test for heteroscedasticity
# Test if squared residuals correlate with predictor
squared_residuals = residuals_linear**2
bp_corr, bp_p = stats.pearsonr(year, squared_residuals)
print(f"\nBreusch-Pagan style test (correlation of squared residuals with year):")
print(f"  Correlation: {bp_corr:.4f}")
print(f"  P-value: {bp_p:.6f}")
if abs(bp_corr) > 0.3 and bp_p < 0.05:
    print("  Result: Evidence of heteroscedasticity")
else:
    print("  Result: No strong evidence of heteroscedasticity")

# ============================================================================
# Examine if variance scales with mean (common in count data)
# ============================================================================
print("\n" + "="*80)
print("MEAN-VARIANCE RELATIONSHIP IN WINDOWS")
print("="*80)

# Create rolling windows
window_size = 10
n_windows = len(C) - window_size + 1

window_means = []
window_vars = []
window_centers = []

for i in range(n_windows):
    window_C = C[i:i+window_size]
    window_means.append(window_C.mean())
    window_vars.append(window_C.var(ddof=1))
    window_centers.append(year[i:i+window_size].mean())

window_means = np.array(window_means)
window_vars = np.array(window_vars)

# Fit power law: Var = a * Mean^b
# Log transform: log(Var) = log(a) + b * log(Mean)
log_means = np.log(window_means)
log_vars = np.log(window_vars)

slope_mv, intercept_mv, r_mv, p_mv, se_mv = stats.linregress(log_means, log_vars)

print(f"\nPower law fit: Variance = a * Mean^b")
print(f"  b (power): {slope_mv:.4f}")
print(f"  a (constant): {np.exp(intercept_mv):.4f}")
print(f"  R-squared: {r_mv**2:.4f}")
print(f"  P-value: {p_mv:.6f}")

print(f"\nInterpretation:")
if abs(slope_mv - 1) < 0.1:
    print(f"  Power ≈ 1: Variance proportional to mean (like Poisson)")
elif abs(slope_mv - 2) < 0.2:
    print(f"  Power ≈ 2: Variance proportional to mean squared (like NB with small r)")
else:
    print(f"  Power = {slope_mv:.2f}: Non-standard relationship")

# ============================================================================
# Alternative: Log-linear model (common for count data with exponential growth)
# ============================================================================
print("\n" + "="*80)
print("LOG-LINEAR MODEL ANALYSIS")
print("="*80)

# Fit: log(C) = a + b*year
log_C = np.log(C)
slope_log, intercept_log, r_log, p_log, se_log = stats.linregress(year, log_C)

print(f"\nLog-linear model: log(C) = {intercept_log:.4f} + {slope_log:.4f} * year")
print(f"R-squared: {r_log**2:.4f}")
print(f"Growth rate: {(np.exp(slope_log) - 1) * 100:.2f}% per standardized year unit")

# Residuals from log-linear model
predicted_log = slope_log * year + intercept_log
residuals_log = log_C - predicted_log

print(f"\nLog-scale residuals:")
print(f"  Mean: {residuals_log.mean():.6f}")
print(f"  Variance: {residuals_log.var(ddof=1):.6f}")
print(f"  Std Dev: {residuals_log.std(ddof=1):.6f}")

# Back-transform to original scale
predicted_original_scale = np.exp(predicted_log)
residuals_original_scale = C - predicted_original_scale

print(f"\nOriginal scale residuals (from log model):")
print(f"  Mean: {residuals_original_scale.mean():.4f}")
print(f"  Variance: {residuals_original_scale.var(ddof=1):.4f}")
print(f"  Std Dev: {residuals_original_scale.std(ddof=1):.4f}")

# Compare model fits
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

# Calculate AIC-like measure (not true AIC but useful for comparison)
def calc_mse(residuals):
    return np.mean(residuals**2)

mse_linear = calc_mse(residuals_linear)
mse_log = calc_mse(residuals_original_scale)

print(f"\nMean Squared Error:")
print(f"  Linear model: {mse_linear:.2f}")
print(f"  Log-linear model: {mse_log:.2f}")

if mse_log < mse_linear:
    print(f"\nLog-linear model fits better (lower MSE)")
else:
    print(f"\nLinear model fits better (lower MSE)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
