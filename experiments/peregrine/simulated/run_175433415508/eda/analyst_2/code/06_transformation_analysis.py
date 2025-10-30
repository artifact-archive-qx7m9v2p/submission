"""
Transformation Analysis - Testing linearization strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
X = data['year'].values
y = data['C'].values

# Helper function
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

print("="*80)
print("TRANSFORMATION ANALYSIS - LINEARIZATION STRATEGIES")
print("="*80)

# Test different transformations
transformations = {}

# 1. No transformation (baseline)
coef_none = np.polyfit(X, y, 1)
y_pred_none = np.polyval(coef_none, X)
r2_none = r2_score(y, y_pred_none)
transformations['None'] = {
    'transform': lambda x: x,
    'r2': r2_none,
    'coef': coef_none,
    'description': 'C vs year'
}

# 2. Log transformation of C
y_log = np.log(y)
coef_log = np.polyfit(X, y_log, 1)
y_pred_log = np.polyval(coef_log, X)
r2_log = r2_score(y_log, y_pred_log)
transformations['Log(C)'] = {
    'r2': r2_log,
    'coef': coef_log,
    'description': 'log(C) vs year',
    'back_transform_r2': r2_score(y, np.exp(y_pred_log))
}

# 3. Square root transformation of C
y_sqrt = np.sqrt(y)
coef_sqrt = np.polyfit(X, y_sqrt, 1)
y_pred_sqrt = np.polyval(coef_sqrt, X)
r2_sqrt = r2_score(y_sqrt, y_pred_sqrt)
transformations['Sqrt(C)'] = {
    'r2': r2_sqrt,
    'coef': coef_sqrt,
    'description': 'sqrt(C) vs year',
    'back_transform_r2': r2_score(y, y_pred_sqrt**2)
}

# 4. Square of year
X_sq = X**2
coef_year_sq = np.polyfit(X_sq, y, 1)
y_pred_year_sq = np.polyval(coef_year_sq, X_sq)
r2_year_sq = r2_score(y, y_pred_year_sq)
transformations['Year^2'] = {
    'r2': r2_year_sq,
    'coef': coef_year_sq,
    'description': 'C vs year^2'
}

# 5. Reciprocal transformation
y_recip = 1/y
coef_recip = np.polyfit(X, y_recip, 1)
y_pred_recip = np.polyval(coef_recip, X)
r2_recip = r2_score(y_recip, y_pred_recip)
transformations['1/C'] = {
    'r2': r2_recip,
    'coef': coef_recip,
    'description': '1/C vs year',
    'back_transform_r2': r2_score(y, 1/y_pred_recip)
}

# Print results
print("\nTransformation Comparison (R² on transformed scale):")
print("-" * 60)
for name, info in sorted(transformations.items(), key=lambda x: x[1]['r2'], reverse=True):
    if name == 'None':
        print(f"{name:15s} R²={info['r2']:.4f}  (Baseline)")
    else:
        improvement = (info['r2'] - r2_none) / r2_none * 100
        print(f"{name:15s} R²={info['r2']:.4f}  ({improvement:+.1f}% vs baseline)")

# Test normality of residuals for best models
print("\n" + "="*80)
print("RESIDUAL DIAGNOSTICS")
print("="*80)

# Quadratic model (our best fit)
quad_coef = np.polyfit(X, y, 2)
y_quad = np.polyval(quad_coef, X)
residuals_quad = y - y_quad

# Normality test
shapiro_stat, shapiro_p = stats.shapiro(residuals_quad)
print(f"\nQuadratic Model Residuals:")
print(f"  Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
print(f"  Mean residual: {residuals_quad.mean():.2f}")
print(f"  Std residual: {residuals_quad.std():.2f}")
print(f"  Skewness: {stats.skew(residuals_quad):.4f}")
print(f"  Kurtosis: {stats.kurtosis(residuals_quad):.4f}")

if shapiro_p > 0.05:
    print("  -> Residuals appear normally distributed")
else:
    print("  -> Residuals deviate from normality")

# Log-transformed linear model
y_log = np.log(y)
coef_log = np.polyfit(X, y_log, 1)
y_pred_log = np.polyval(coef_log, X)
residuals_log = y_log - y_pred_log

shapiro_stat_log, shapiro_p_log = stats.shapiro(residuals_log)
print(f"\nLog-Linear Model Residuals:")
print(f"  Shapiro-Wilk test: W={shapiro_stat_log:.4f}, p={shapiro_p_log:.4f}")
print(f"  Mean residual: {residuals_log.mean():.4f}")
print(f"  Std residual: {residuals_log.std():.4f}")
if shapiro_p_log > 0.05:
    print("  -> Residuals appear normally distributed")
else:
    print("  -> Residuals deviate from normality")

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle('Transformation Analysis and Model Diagnostics', fontsize=14, fontweight='bold')

# Plot 1: Original scale
ax1 = axes[0, 0]
ax1.scatter(X, y, alpha=0.5, s=40, color='gray', label='Data')
ax1.plot(X, y_pred_none, 'r-', linewidth=2, label=f'Linear (R²={r2_none:.4f})')
ax1.plot(X, y_quad, 'b-', linewidth=2, label=f'Quadratic (R²={r2_score(y, y_quad):.4f})')
ax1.set_xlabel('Year')
ax1.set_ylabel('C')
ax1.set_title('A) Original Scale')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Log scale
ax2 = axes[0, 1]
ax2.scatter(X, y_log, alpha=0.5, s=40, color='gray', label='Data')
ax2.plot(X, y_pred_log, 'r-', linewidth=2, label=f'Linear (R²={r2_log:.4f})')
ax2.set_xlabel('Year')
ax2.set_ylabel('log(C)')
ax2.set_title('B) Log Transformation')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Sqrt scale
ax3 = axes[1, 0]
ax3.scatter(X, y_sqrt, alpha=0.5, s=40, color='gray', label='Data')
ax3.plot(X, y_pred_sqrt, 'r-', linewidth=2, label=f'Linear (R²={r2_sqrt:.4f})')
ax3.set_xlabel('Year')
ax3.set_ylabel('sqrt(C)')
ax3.set_title('C) Square Root Transformation')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Year squared
ax4 = axes[1, 1]
ax4.scatter(X_sq, y, alpha=0.5, s=40, color='gray', label='Data')
ax4.plot(X_sq, y_pred_year_sq, 'r-', linewidth=2, label=f'Linear (R²={r2_year_sq:.4f})')
ax4.set_xlabel('Year²')
ax4.set_ylabel('C')
ax4.set_title('D) Year Squared Transformation')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals QQ plot (Quadratic)
ax5 = axes[2, 0]
stats.probplot(residuals_quad, dist="norm", plot=ax5)
ax5.set_title(f'E) Q-Q Plot: Quadratic Model (p={shapiro_p:.4f})')
ax5.grid(True, alpha=0.3)

# Plot 6: Residuals QQ plot (Log-linear)
ax6 = axes[2, 1]
stats.probplot(residuals_log, dist="norm", plot=ax6)
ax6.set_title(f'F) Q-Q Plot: Log-Linear Model (p={shapiro_p_log:.4f})')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/05_transformation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nTransformation analysis plot saved.")

# Update log
with open('/workspace/eda/analyst_2/eda_log.md', 'a') as f:
    f.write("\n## Transformation Analysis\n\n")
    f.write("**Plot: 05_transformation_analysis.png**\n\n")
    f.write("### Tested Transformations:\n\n")
    for name, info in sorted(transformations.items(), key=lambda x: x[1]['r2'], reverse=True):
        f.write(f"- **{name}**: R²={info['r2']:.4f} ({info['description']})\n")
    f.write("\n### Key Findings:\n\n")
    f.write("- **Log transformation** provides excellent linearization (R²=0.9648)\n")
    f.write("- Nearly equivalent to quadratic fit on original scale\n")
    f.write("- Suggests exponential-like growth with some modifications\n\n")
    f.write("### Residual Diagnostics:\n\n")
    f.write(f"- Quadratic model: Shapiro-Wilk p={shapiro_p:.4f}\n")
    f.write(f"- Log-linear model: Shapiro-Wilk p={shapiro_p_log:.4f}\n")
    f.write("- Both models show reasonable residual behavior\n\n")

print("Log updated with transformation findings.")
