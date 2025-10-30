"""
Variance Analysis and Temporal Dependencies
Focus: Heteroscedasticity, autocorrelation, and count data properties
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import levene, chi2
import warnings
warnings.filterwarnings('ignore')

# Load data
with open('/workspace/data/data_analyst_1.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({
    'year': data['year'],
    'C': data['C']
})

X = df['year'].values
y = df['C'].values

print("="*60)
print("VARIANCE STRUCTURE ANALYSIS")
print("="*60)

# Fit quadratic model for residual analysis (best simple model from previous)
coeffs_quad = np.polyfit(X, y, 2)
y_pred_quad = np.polyval(coeffs_quad, X)
residuals = y - y_pred_quad

# 1. Test for heteroscedasticity
# Split data into thirds and test variance differences
n = len(y)
third = n // 3

early_resid = residuals[:third]
middle_resid = residuals[third:2*third]
late_resid = residuals[2*third:]

print("\n1. RESIDUAL VARIANCE BY TIME PERIOD")
print(f"   Early period variance: {np.var(early_resid, ddof=1):.2f}")
print(f"   Middle period variance: {np.var(middle_resid, ddof=1):.2f}")
print(f"   Late period variance: {np.var(late_resid, ddof=1):.2f}")

# Levene's test for homogeneity of variance
levene_stat, levene_p = levene(early_resid, middle_resid, late_resid)
print(f"\n   Levene's test for equal variances:")
print(f"   Statistic={levene_stat:.4f}, p-value={levene_p:.4f}")
if levene_p < 0.05:
    print("   -> REJECT null hypothesis: Evidence of heteroscedasticity")
else:
    print("   -> Cannot reject null: Variance appears homogeneous")

# 2. Mean-Variance relationship (critical for count data)
# Split into bins based on fitted values
n_bins = 5
bins = np.linspace(y_pred_quad.min(), y_pred_quad.max(), n_bins + 1)
bin_means = []
bin_vars = []

print("\n2. MEAN-VARIANCE RELATIONSHIP")
print("   (Critical for assessing Poisson vs Negative Binomial)")
for i in range(n_bins):
    mask = (y_pred_quad >= bins[i]) & (y_pred_quad < bins[i+1])
    if i == n_bins - 1:  # Include upper bound in last bin
        mask = (y_pred_quad >= bins[i]) & (y_pred_quad <= bins[i+1])

    if mask.sum() > 0:
        bin_mean = y[mask].mean()
        bin_var = y[mask].var(ddof=1)
        bin_means.append(bin_mean)
        bin_vars.append(bin_var)
        print(f"   Bin {i+1}: mean={bin_mean:.2f}, variance={bin_var:.2f}, ratio={bin_var/bin_mean:.2f}")

# Calculate dispersion index
dispersion_ratio = np.mean(bin_vars) / np.mean(bin_means)
print(f"\n   Overall variance-to-mean ratio: {dispersion_ratio:.2f}")
if dispersion_ratio > 1.5:
    print("   -> Strong OVERDISPERSION: Negative Binomial preferred over Poisson")
elif dispersion_ratio < 0.8:
    print("   -> Evidence of UNDERDISPERSION: Consider Zero-inflated models")
else:
    print("   -> Near-equidispersion: Poisson may be adequate")

# 3. Autocorrelation analysis
print("\n" + "="*60)
print("TEMPORAL DEPENDENCIES")
print("="*60)

# Calculate ACF for raw counts
max_lag = min(15, len(y) // 2)
acf_values = []
for lag in range(max_lag + 1):
    if lag == 0:
        acf_values.append(1.0)
    else:
        corr = np.corrcoef(y[:-lag], y[lag:])[0, 1]
        acf_values.append(corr)

print("\n3. AUTOCORRELATION FUNCTION (ACF) for raw counts")
for i, acf in enumerate(acf_values[:6]):
    print(f"   Lag {i}: {acf:.4f}")

# Calculate ACF for residuals
acf_resid = []
for lag in range(max_lag + 1):
    if lag == 0:
        acf_resid.append(1.0)
    else:
        corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
        acf_resid.append(corr)

print("\n4. AUTOCORRELATION FUNCTION (ACF) for residuals")
for i, acf in enumerate(acf_resid[:6]):
    print(f"   Lag {i}: {acf:.4f}")

# Test Ljung-Box statistic approximation for first lag
lb_stat = len(residuals) * acf_resid[1]**2
lb_p = 1 - chi2.cdf(lb_stat, 1)
print(f"\n   First-order autocorrelation test (approx):")
print(f"   Statistic={lb_stat:.4f}, p-value={lb_p:.4f}")

# Calculate first differences to check if stationary
first_diff = np.diff(y)
print("\n5. STATIONARITY ASSESSMENT")
print(f"   Original series mean: {y.mean():.2f}, std: {y.std():.2f}")
print(f"   First differences mean: {first_diff.mean():.2f}, std: {first_diff.std():.2f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Residuals vs Fitted
ax1 = axes[0, 0]
ax1.scatter(y_pred_quad, residuals, alpha=0.6, s=50, color='steelblue')
ax1.axhline(0, color='red', linestyle='--', linewidth=2)
# Add smooth trend
z = np.polyfit(y_pred_quad, residuals, 2)
p = np.poly1d(z)
x_smooth = np.linspace(y_pred_quad.min(), y_pred_quad.max(), 100)
ax1.plot(x_smooth, p(x_smooth), 'orange', linewidth=2, label='Trend')
ax1.set_xlabel('Fitted Values', fontsize=10)
ax1.set_ylabel('Residuals', fontsize=10)
ax1.set_title('A. Residual Plot (Heteroscedasticity Check)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals vs Time
ax2 = axes[0, 1]
ax2.scatter(X, residuals, alpha=0.6, s=50, color='green')
ax2.axhline(0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Year (standardized)', fontsize=10)
ax2.set_ylabel('Residuals', fontsize=10)
ax2.set_title('B. Residuals vs Time', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Absolute residuals vs Fitted (variance check)
ax3 = axes[0, 2]
abs_resid = np.abs(residuals)
ax3.scatter(y_pred_quad, abs_resid, alpha=0.6, s=50, color='purple')
# Add smooth trend
z = np.polyfit(y_pred_quad, abs_resid, 1)
p = np.poly1d(z)
ax3.plot(x_smooth, p(x_smooth), 'orange', linewidth=2, label='Trend')
ax3.set_xlabel('Fitted Values', fontsize=10)
ax3.set_ylabel('|Residuals|', fontsize=10)
ax3.set_title('C. Variance vs Fitted (Scale-Location)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Mean-Variance relationship
ax4 = axes[1, 0]
ax4.scatter(bin_means, bin_vars, s=100, alpha=0.7, color='darkred')
# Add reference lines
max_val = max(max(bin_means), max(bin_vars))
ax4.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Poisson (var=mean)', alpha=0.5)
ax4.plot([0, max_val], [0, 2*max_val], 'r--', linewidth=2, label='var=2*mean', alpha=0.5)
ax4.set_xlabel('Mean Count', fontsize=10)
ax4.set_ylabel('Variance', fontsize=10)
ax4.set_title('D. Mean-Variance Relationship', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: ACF for raw counts
ax5 = axes[1, 1]
lags = np.arange(len(acf_values))
ax5.bar(lags, acf_values, alpha=0.7, color='steelblue')
ax5.axhline(0, color='black', linewidth=1)
# Add confidence bands (95%)
conf_level = 1.96 / np.sqrt(len(y))
ax5.axhline(conf_level, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax5.axhline(-conf_level, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax5.set_xlabel('Lag', fontsize=10)
ax5.set_ylabel('ACF', fontsize=10)
ax5.set_title('E. Autocorrelation: Raw Counts', fontweight='bold')
ax5.set_ylim(-0.5, 1.1)
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: ACF for residuals
ax6 = axes[1, 2]
lags_resid = np.arange(len(acf_resid))
ax6.bar(lags_resid, acf_resid, alpha=0.7, color='green')
ax6.axhline(0, color='black', linewidth=1)
ax6.axhline(conf_level, color='red', linestyle='--', linewidth=1, alpha=0.5, label='95% CI')
ax6.axhline(-conf_level, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax6.set_xlabel('Lag', fontsize=10)
ax6.set_ylabel('ACF', fontsize=10)
ax6.set_title('F. Autocorrelation: Residuals', fontweight='bold')
ax6.set_ylim(-0.5, 1.1)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/03_variance_autocorrelation.png', dpi=300, bbox_inches='tight')
print("\nSaved: 03_variance_autocorrelation.png")

plt.close()
