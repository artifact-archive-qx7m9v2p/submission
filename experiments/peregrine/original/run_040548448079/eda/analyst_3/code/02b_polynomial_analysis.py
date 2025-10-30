"""
Polynomial and exponential feature analysis (without sklearn)
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('/workspace/data/data_analyst_3.csv')
df_sorted = df.sort_values('year').reset_index(drop=True)

X = df_sorted['year'].values
y = df_sorted['C'].values
n = len(y)

print("="*60)
print("POLYNOMIAL FEATURE ANALYSIS")
print("="*60)

print("\nPolynomial fits on original scale:")
print(f"{'Degree':<8} {'R²':<8} {'RMSE':<10} {'AIC':<10} {'BIC':<10}")
print("-" * 50)

for degree in range(1, 6):
    # Fit polynomial
    coeffs = np.polyfit(X, y, degree)
    y_pred = np.polyval(coeffs, X)

    # Calculate metrics
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(ss_res / n)

    # Calculate AIC and BIC
    k = degree + 1  # number of parameters
    aic = n * np.log(ss_res/n) + 2*k
    bic = n * np.log(ss_res/n) + k*np.log(n)

    print(f"{degree:<8} {r2:7.4f} {rmse:9.2f} {aic:9.2f} {bic:9.2f}")

print("\n" + "="*60)
print("EXPONENTIAL GROWTH ASSESSMENT")
print("="*60)

# Test if log(C) is linear in year (exponential growth)
log_C = np.log(y)
slope, intercept = np.polyfit(X, log_C, 1)
log_C_fitted = slope * X + intercept
ss_res_log = np.sum((log_C - log_C_fitted)**2)
ss_tot_log = np.sum((log_C - np.mean(log_C))**2)
r2_log = 1 - (ss_res_log / ss_tot_log)

print(f"\nR² for log(C) ~ year: {r2_log:.4f}")
print(f"This implies: C = exp({intercept:.4f}) * exp({slope:.4f} * year)")
print(f"Growth rate per unit year: {np.exp(slope):.4f}x")

# Back-transform to original scale and check fit
C_pred_exp = np.exp(log_C_fitted)
r2_exp_original = 1 - np.sum((y - C_pred_exp)**2) / np.sum((y - np.mean(y))**2)
print(f"R² for exponential model on original scale: {r2_exp_original:.4f}")

# Check residuals for exponential model
residuals_log = log_C - log_C_fitted
print(f"\nResidual diagnostics for log-scale exponential model:")
print(f"  Mean: {np.mean(residuals_log):.6f}")
print(f"  Std: {np.std(residuals_log):.4f}")
print(f"  Shapiro-Wilk p-value: {stats.shapiro(residuals_log)[1]:.4f}")

print("\n" + "="*60)
print("POWER LAW ASSESSMENT")
print("="*60)

# Test if log(C) is linear in log(year_positive)
# Since year is standardized (can be negative), shift to positive
year_positive = X - X.min() + 0.1  # Shift to positive
log_year = np.log(year_positive)
slope_power, intercept_power = np.polyfit(log_year, log_C, 1)
log_C_fitted_power = slope_power * log_year + intercept_power

ss_res_power = np.sum((log_C - log_C_fitted_power)**2)
ss_tot_power = np.sum((log_C - np.mean(log_C))**2)
r2_power = 1 - (ss_res_power / ss_tot_power)

print(f"\nR² for log(C) ~ log(year_positive): {r2_power:.4f}")
print(f"This implies: C = exp({intercept_power:.4f}) * year_positive^{slope_power:.4f}")
print(f"Power law exponent: {slope_power:.4f}")

print("\n" + "="*60)
print("DERIVED FEATURE CANDIDATES")
print("="*60)

# Test various derived features
features = {
    'year': X,
    'year^2': X**2,
    'year^3': X**3,
    'exp(year)': np.exp(X),
    'exp(0.5*year)': np.exp(0.5*X),
    'year*exp(year)': X * np.exp(X),
}

print("\nCorrelation with C for various features:")
print(f"{'Feature':<20} {'Correlation':<12}")
print("-" * 35)
for name, feature in features.items():
    corr = np.corrcoef(feature, y)[0, 1]
    print(f"{name:<20} {corr:11.4f}")

print("\n" + "="*60)
print("COMPARING QUADRATIC vs EXPONENTIAL MODELS")
print("="*60)

# Quadratic model
coeffs_quad = np.polyfit(X, y, 2)
y_pred_quad = np.polyval(coeffs_quad, X)
r2_quad = 1 - np.sum((y - y_pred_quad)**2) / np.sum((y - np.mean(y))**2)
rmse_quad = np.sqrt(np.mean((y - y_pred_quad)**2))
aic_quad = n * np.log(np.sum((y - y_pred_quad)**2)/n) + 2*3  # 3 parameters

# Exponential model (already computed)
rmse_exp = np.sqrt(np.mean((y - C_pred_exp)**2))
aic_exp = n * np.log(np.sum((y - C_pred_exp)**2)/n) + 2*2  # 2 parameters

print(f"\nQuadratic model (C ~ year + year²):")
print(f"  R² = {r2_quad:.4f}, RMSE = {rmse_quad:.2f}, AIC = {aic_quad:.2f}")
print(f"  Coefficients: {coeffs_quad}")

print(f"\nExponential model (log(C) ~ year):")
print(f"  R² = {r2_exp_original:.4f}, RMSE = {rmse_exp:.2f}, AIC = {aic_exp:.2f}")

print(f"\nAIC difference (Quad - Exp): {aic_quad - aic_exp:.2f}")
if aic_quad < aic_exp:
    print("  -> Quadratic model preferred by AIC")
else:
    print("  -> Exponential model preferred by AIC")
