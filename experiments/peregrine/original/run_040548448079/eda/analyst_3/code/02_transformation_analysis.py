"""
Comprehensive transformation analysis:
- Multiple transformation types
- Variance stabilization assessment
- Linearity diagnostics
- Optimal Box-Cox parameter
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
df = pd.read_csv('/workspace/data/data_analyst_3.csv')
df_sorted = df.sort_values('year').reset_index(drop=True)

print("="*60)
print("TRANSFORMATION ANALYSIS")
print("="*60)

# Create various transformations
transformations = {
    'original': df_sorted['C'].values,
    'log': np.log(df_sorted['C'].values),
    'sqrt': np.sqrt(df_sorted['C'].values),
    'inverse': 1.0 / df_sorted['C'].values,
    'square': df_sorted['C'].values ** 2,
}

# Box-Cox transformation - find optimal lambda
def box_cox_transform(data, lam):
    """Apply Box-Cox transformation"""
    if lam == 0:
        return np.log(data)
    else:
        return (data**lam - 1) / lam

# Find optimal lambda for Box-Cox
print("\n" + "="*60)
print("BOX-COX TRANSFORMATION ANALYSIS")
print("="*60)

# Method 1: Using scipy's built-in
data_boxcox, lambda_optimal = stats.boxcox(df_sorted['C'].values)
print(f"Optimal lambda (scipy method): {lambda_optimal:.4f}")

# Method 2: Maximize correlation with year (for linearity)
def neg_correlation(lam):
    transformed = box_cox_transform(df_sorted['C'].values, lam)
    return -np.abs(np.corrcoef(df_sorted['year'].values, transformed)[0, 1])

result = minimize_scalar(neg_correlation, bounds=(-2, 2), method='bounded')
lambda_linearity = result.x
print(f"Lambda for max linearity with year: {lambda_linearity:.4f}")

# Method 3: Minimize variance of residuals from linear fit
def residual_variance(lam):
    transformed = box_cox_transform(df_sorted['C'].values, lam)
    slope, intercept = np.polyfit(df_sorted['year'].values, transformed, 1)
    residuals = transformed - (slope * df_sorted['year'].values + intercept)
    return np.var(residuals)

result = minimize_scalar(residual_variance, bounds=(-2, 2), method='bounded')
lambda_variance = result.x
print(f"Lambda for minimum residual variance: {lambda_variance:.4f}")

# Add Box-Cox transformations to our dict
transformations['boxcox_optimal'] = data_boxcox
transformations['boxcox_linearity'] = box_cox_transform(df_sorted['C'].values, lambda_linearity)

print("\n" + "="*60)
print("LINEARITY ASSESSMENT")
print("="*60)

# Calculate correlation for each transformation
for name, transformed in transformations.items():
    corr = np.corrcoef(df_sorted['year'].values, transformed)[0, 1]
    print(f"{name:20s}: r = {corr:7.4f}")

print("\n" + "="*60)
print("VARIANCE STABILIZATION ASSESSMENT")
print("="*60)

# For each transformation, split into thirds and check variance ratio
def variance_ratio(data):
    """Calculate ratio of variance in high vs low third"""
    n = len(data)
    third = n // 3
    var_low = np.var(data[:third])
    var_high = np.var(data[2*third:])
    return var_high / var_low if var_low > 0 else np.inf

print("\nVariance ratio (high/low thirds):")
print("Lower ratio = better variance stabilization")
print("-" * 40)
for name, transformed in transformations.items():
    ratio = variance_ratio(transformed)
    print(f"{name:20s}: {ratio:7.2f}")

print("\n" + "="*60)
print("RESIDUAL ANALYSIS FROM LINEAR FITS")
print("="*60)

residual_stats = {}
for name, transformed in transformations.items():
    # Fit linear model
    slope, intercept = np.polyfit(df_sorted['year'].values, transformed, 1)
    fitted = slope * df_sorted['year'].values + intercept
    residuals = transformed - fitted

    # Calculate residual statistics
    residual_stats[name] = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'skew': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals),
        'shapiro_p': stats.shapiro(residuals)[1]
    }

print("\nResidual statistics (for linear fit):")
print(f"{'Transformation':<20} {'Mean':<8} {'Std':<8} {'Skew':<8} {'Kurt':<8} {'Shapiro-p':<10}")
print("-" * 70)
for name, stats_dict in residual_stats.items():
    print(f"{name:<20} {stats_dict['mean']:7.4f} {stats_dict['std']:7.3f} "
          f"{stats_dict['skew']:7.3f} {stats_dict['kurtosis']:7.3f} {stats_dict['shapiro_p']:9.4f}")

print("\n" + "="*60)
print("POLYNOMIAL FEATURE ANALYSIS")
print("="*60)

# Test polynomial features on original scale
X = df_sorted['year'].values
y = df_sorted['C'].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

print("\nPolynomial fits on original scale:")
print(f"{'Degree':<8} {'R²':<8} {'RMSE':<10} {'AIC':<10}")
print("-" * 40)

n = len(y)
for degree in range(1, 6):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Calculate AIC
    rss = np.sum((y - y_pred)**2)
    k = degree + 1  # number of parameters
    aic = n * np.log(rss/n) + 2*k

    print(f"{degree:<8} {r2:7.4f} {rmse:9.2f} {aic:9.2f}")

print("\n" + "="*60)
print("EXPONENTIAL GROWTH ASSESSMENT")
print("="*60)

# Test if log(C) is linear in year (exponential growth)
log_C = np.log(df_sorted['C'].values)
slope, intercept = np.polyfit(X, log_C, 1)
log_C_fitted = slope * X + intercept
r2_log = r2_score(log_C, log_C_fitted)

print(f"R² for log(C) ~ year: {r2_log:.4f}")
print(f"This implies: C = exp({intercept:.4f}) * exp({slope:.4f} * year)")
print(f"Growth rate per unit year: {np.exp(slope):.4f}x")

# Check residuals for exponential model
residuals_log = log_C - log_C_fitted
print(f"\nResidual diagnostics for exponential model:")
print(f"  Mean: {np.mean(residuals_log):.6f}")
print(f"  Std: {np.std(residuals_log):.4f}")
print(f"  Shapiro-Wilk p-value: {stats.shapiro(residuals_log)[1]:.4f}")

print("\n" + "="*60)
print("SUMMARY RECOMMENDATIONS")
print("="*60)
print("\n1. Best transformation for LINEARITY:")
best_linearity = max([(name, np.abs(np.corrcoef(df_sorted['year'].values, trans)[0, 1]))
                      for name, trans in transformations.items()],
                     key=lambda x: x[1])
print(f"   {best_linearity[0]} (r = {best_linearity[1]:.4f})")

print("\n2. Best transformation for VARIANCE STABILIZATION:")
best_variance = min([(name, variance_ratio(trans))
                     for name, trans in transformations.items()],
                    key=lambda x: x[1])
print(f"   {best_variance[0]} (ratio = {best_variance[1]:.2f})")

print("\n3. Most NORMAL residuals (highest Shapiro-Wilk p):")
best_normality = max(residual_stats.items(),
                     key=lambda x: x[1]['shapiro_p'])
print(f"   {best_normality[0]} (p = {best_normality[1]['shapiro_p']:.4f})")
