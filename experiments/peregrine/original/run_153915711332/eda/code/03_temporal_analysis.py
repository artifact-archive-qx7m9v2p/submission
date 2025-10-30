"""
Temporal Trend Analysis
=======================
Purpose: Analyze the relationship between time and counts, test linearity vs nonlinearity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/workspace/data/data.csv')
C = data['C'].values
year = data['year'].values

# Polynomial fitting helper
def fit_polynomial(x, y, degree):
    """Fit polynomial and return predictions and R²"""
    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot
    return coeffs, r2

# Create comprehensive temporal analysis
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Temporal Trend Analysis: C vs Year', fontsize=16, y=1.00)

# 1. Scatter plot with linear fit
ax = axes[0, 0]
ax.scatter(year, C, s=60, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
# Linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(year, C)
x_line = np.array([year.min(), year.max()])
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Linear: R²={r_value**2:.3f}')
ax.set_xlabel('Year (standardized)', fontsize=11)
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Scatter Plot with Linear Trend', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Multiple polynomial fits comparison
ax = axes[0, 1]
ax.scatter(year, C, s=60, alpha=0.4, color='steelblue', edgecolors='black', linewidth=0.5, label='Data')
x_smooth = np.linspace(year.min(), year.max(), 200)

colors = ['red', 'green', 'orange', 'purple']
degrees = [1, 2, 3, 4]
r2_scores = []

for deg, color in zip(degrees, colors):
    coeffs, r2 = fit_polynomial(year, C, deg)
    r2_scores.append(r2)
    y_smooth = np.polyval(coeffs, x_smooth)
    ax.plot(x_smooth, y_smooth, color=color, linewidth=2, label=f'Degree {deg}: R²={r2:.3f}')

ax.set_xlabel('Year (standardized)', fontsize=11)
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Polynomial Fits Comparison', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Log-scale plot (test exponential growth)
ax = axes[0, 2]
ax.scatter(year, C, s=60, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
ax.set_xlabel('Year (standardized)', fontsize=11)
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_yscale('log')
ax.set_title('Log-scale Y (Test Exponential Growth)', fontsize=12)
ax.grid(True, alpha=0.3)

# Linear fit on log scale
log_C = np.log(C)
slope_log, intercept_log, r_log, p_log, se_log = stats.linregress(year, log_C)
y_exp = np.exp(intercept_log + slope_log * x_line)
ax.plot(x_line, y_exp, 'r-', linewidth=2, label=f'Exponential: R²={r_log**2:.3f}')
ax.legend()

# 4. Residuals plot (linear model)
ax = axes[1, 0]
y_pred = slope * year + intercept
residuals = C - y_pred
ax.scatter(y_pred, residuals, s=60, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
# Add lowess smoother to check for patterns
sort_idx = np.argsort(y_pred)
window_len = min(11, len(residuals)//3*2+1)
if window_len % 2 == 0:
    window_len += 1
smoothed = savgol_filter(residuals[sort_idx], window_length=window_len, polyorder=2)
ax.plot(y_pred[sort_idx], smoothed, 'orange', linewidth=2, label='Smoothed trend')
ax.set_xlabel('Fitted Values', fontsize=11)
ax.set_ylabel('Residuals', fontsize=11)
ax.set_title('Residual Plot (Linear Model)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Q-Q plot of residuals
ax = axes[1, 1]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot of Residuals', fontsize=12)
ax.grid(True, alpha=0.3)

# 6. Variance plot (test heteroscedasticity)
ax = axes[1, 2]
# Split data into bins
n_bins = 5
bins = np.linspace(year.min(), year.max(), n_bins + 1)
bin_means = []
bin_vars = []
bin_centers = []

for i in range(n_bins):
    mask = (year >= bins[i]) & (year < bins[i+1])
    if mask.sum() > 1:
        bin_means.append(np.mean(C[mask]))
        bin_vars.append(np.var(C[mask], ddof=1))
        bin_centers.append((bins[i] + bins[i+1]) / 2)

ax.scatter(bin_centers, bin_vars, s=150, alpha=0.6, color='steelblue', edgecolors='black', linewidth=2)
ax.plot(bin_centers, bin_vars, 'steelblue', linewidth=1, alpha=0.3)
ax.set_xlabel('Year (bin centers)', fontsize=11)
ax.set_ylabel('Variance within bins', fontsize=11)
ax.set_title('Variance Structure Over Time', fontsize=12)
ax.grid(True, alpha=0.3)

# Add mean-variance relationship line
if len(bin_means) > 1:
    ax_twin = ax.twinx()
    ax_twin.scatter(bin_centers, bin_means, s=150, alpha=0.6, color='coral',
                    edgecolors='black', linewidth=2, marker='^', label='Mean')
    ax_twin.plot(bin_centers, bin_means, 'coral', linewidth=1, alpha=0.3)
    ax_twin.set_ylabel('Mean within bins', fontsize=11, color='coral')
    ax_twin.tick_params(axis='y', labelcolor='coral')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/02_temporal_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("="*80)
print("TEMPORAL TREND ANALYSIS")
print("="*80)

# Linear model diagnostics
print("\n1. LINEAR MODEL")
print("-"*80)
print(f"Equation: C = {slope:.3f} * year + {intercept:.3f}")
print(f"Slope: {slope:.3f} counts per standardized year")
print(f"  (95% CI: [{slope - 1.96*std_err:.3f}, {slope + 1.96*std_err:.3f}])")
print(f"Intercept: {intercept:.3f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"Correlation: {r_value:.4f}")
print(f"P-value: {p_value:.2e}")
print(f"Standard error: {std_err:.3f}")

# Residual statistics
print(f"\nResidual Statistics:")
print(f"  Mean: {np.mean(residuals):.6f}")
print(f"  Std Dev: {np.std(residuals, ddof=1):.3f}")
print(f"  Min: {np.min(residuals):.3f}")
print(f"  Max: {np.max(residuals):.3f}")

# Test residuals for normality
shapiro_res, p_res = stats.shapiro(residuals)
print(f"\n  Shapiro-Wilk test on residuals:")
print(f"    Statistic: {shapiro_res:.4f}")
print(f"    P-value: {p_res:.4f}")
print(f"    Conclusion: {'Residuals NOT normal' if p_res < 0.05 else 'Residuals approximately normal'} (α=0.05)")

# Durbin-Watson test for autocorrelation
dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
print(f"\n  Durbin-Watson statistic: {dw:.3f}")
print(f"    Interpretation: {'Positive autocorrelation' if dw < 1.5 else 'Negative autocorrelation' if dw > 2.5 else 'No strong autocorrelation'}")

# 2. Exponential model
print("\n2. EXPONENTIAL MODEL (log-linear)")
print("-"*80)
print(f"Equation: log(C) = {slope_log:.3f} * year + {intercept_log:.3f}")
print(f"  → C = {np.exp(intercept_log):.3f} * exp({slope_log:.3f} * year)")
print(f"Growth rate: {slope_log:.3f} per standardized year")
print(f"  = {(np.exp(slope_log) - 1) * 100:.2f}% multiplicative increase per unit")
print(f"R-squared (on log scale): {r_log**2:.4f}")
print(f"P-value: {p_log:.2e}")

# 3. Polynomial comparison
print("\n3. POLYNOMIAL MODEL COMPARISON")
print("-"*80)
for deg, r2 in zip(degrees, r2_scores):
    print(f"Degree {deg}: R² = {r2:.4f}")

# AIC/BIC for model selection
n = len(C)
for deg, r2 in zip(degrees, r2_scores):
    coeffs, _ = fit_polynomial(year, C, deg)
    y_pred_poly = np.polyval(coeffs, year)
    RSS = np.sum((C - y_pred_poly)**2)
    k = deg + 1  # Number of parameters
    AIC = n * np.log(RSS/n) + 2*k
    BIC = n * np.log(RSS/n) + k*np.log(n)
    print(f"  Degree {deg}: AIC={AIC:.1f}, BIC={BIC:.1f}")

# 4. Heteroscedasticity tests
print("\n4. HETEROSCEDASTICITY ANALYSIS")
print("-"*80)

# Breusch-Pagan test (manual implementation)
# Regress squared residuals on predictor
residuals_sq = residuals**2
slope_bp, intercept_bp, r_bp, p_bp, se_bp = stats.linregress(year, residuals_sq)
print(f"Visual inspection of variance over time:")
print(f"  Slope of (residuals² vs year): {slope_bp:.3f}")
print(f"  P-value: {p_bp:.4f}")
print(f"  Conclusion: {'Evidence of heteroscedasticity' if p_bp < 0.05 else 'No strong evidence of heteroscedasticity'}")

# Compare early vs late variance
mid_point = np.median(year)
early_C = C[year < mid_point]
late_C = C[year >= mid_point]
var_early = np.var(early_C, ddof=1)
var_late = np.var(late_C, ddof=1)

print(f"\nEarly period (year < {mid_point:.2f}):")
print(f"  Mean: {np.mean(early_C):.2f}, Variance: {var_early:.2f}")
print(f"Late period (year >= {mid_point:.2f}):")
print(f"  Mean: {np.mean(late_C):.2f}, Variance: {var_late:.2f}")
print(f"Variance ratio (late/early): {var_late/var_early:.2f}")

# F-test for equality of variances
F_stat = var_late / var_early
df1 = len(late_C) - 1
df2 = len(early_C) - 1
p_f = 1 - stats.f.cdf(F_stat, df1, df2)
print(f"\nF-test for equal variances:")
print(f"  F-statistic: {F_stat:.3f}")
print(f"  P-value: {p_f:.4f}")
print(f"  Conclusion: {'Variances differ' if p_f < 0.05 else 'Cannot reject equal variances'}")

# 5. Mean-variance relationship
print("\n5. MEAN-VARIANCE RELATIONSHIP")
print("-"*80)
if len(bin_means) > 1:
    slope_mv, intercept_mv, r_mv, p_mv, se_mv = stats.linregress(bin_means, bin_vars)
    print(f"Regression: Variance = {slope_mv:.3f} * Mean + {intercept_mv:.3f}")
    print(f"  R-squared: {r_mv**2:.4f}")
    print(f"  P-value: {p_mv:.4f}")

    if slope_mv > 0.5:
        if abs(slope_mv - 1) < 0.3:
            print(f"\n  Interpretation: Linear mean-variance relationship (Poisson-like)")
        elif slope_mv > 1.5:
            print(f"\n  Interpretation: Quadratic mean-variance relationship")
            print(f"    → Suggests Negative Binomial or Gamma family")

print("\n" + "="*80)
print("Visualization saved: /workspace/eda/visualizations/02_temporal_analysis.png")
print("="*80)
