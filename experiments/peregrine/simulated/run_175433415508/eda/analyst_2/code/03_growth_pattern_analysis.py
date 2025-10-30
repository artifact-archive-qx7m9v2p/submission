"""
Growth Pattern Analysis - Testing different functional forms
Hypothesis testing: Linear vs Exponential vs Polynomial
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
X = data['year'].values
y = data['C'].values

# Helper functions
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

print("="*80)
print("GROWTH PATTERN ANALYSIS - FUNCTIONAL FORM COMPARISON")
print("="*80)

# 1. Linear Model: C = a + b*year
linear_coef = np.polyfit(X, y, 1)
y_linear = np.polyval(linear_coef, X)
r2_linear = r2_score(y, y_linear)
rmse_linear = rmse(y, y_linear)

print("\n1. LINEAR MODEL: C = a + b*year")
print(f"   Coefficients: a={linear_coef[1]:.2f}, b={linear_coef[0]:.2f}")
print(f"   R-squared: {r2_linear:.4f}")
print(f"   RMSE: {rmse_linear:.2f}")

# 2. Quadratic Model: C = a + b*year + c*year^2
quad_coef = np.polyfit(X, y, 2)
y_quad = np.polyval(quad_coef, X)
r2_quad = r2_score(y, y_quad)
rmse_quad = rmse(y, y_quad)

print("\n2. QUADRATIC MODEL: C = a + b*year + c*year^2")
print(f"   Coefficients: a={quad_coef[2]:.2f}, b={quad_coef[1]:.2f}, c={quad_coef[0]:.2f}")
print(f"   R-squared: {r2_quad:.4f}")
print(f"   RMSE: {rmse_quad:.2f}")
print(f"   Improvement over linear: {(r2_quad - r2_linear)/r2_linear * 100:.2f}%")

# 3. Cubic Model: C = a + b*year + c*year^2 + d*year^3
cubic_coef = np.polyfit(X, y, 3)
y_cubic = np.polyval(cubic_coef, X)
r2_cubic = r2_score(y, y_cubic)
rmse_cubic = rmse(y, y_cubic)

print("\n3. CUBIC MODEL: C = a + b*year + c*year^2 + d*year^3")
print(f"   Coefficients: a={cubic_coef[3]:.2f}, b={cubic_coef[2]:.2f}, c={cubic_coef[1]:.2f}, d={cubic_coef[0]:.2f}")
print(f"   R-squared: {r2_cubic:.4f}")
print(f"   RMSE: {rmse_cubic:.2f}")
print(f"   Improvement over quadratic: {(r2_cubic - r2_quad)/r2_quad * 100:.2f}%")

# 4. Exponential Model: C = a * exp(b*year)
# Use log transform: log(C) = log(a) + b*year
# Only use positive C values
y_positive = y[y > 0]
X_positive = X[y > 0]
log_y = np.log(y_positive)
exp_coef = np.polyfit(X_positive, log_y, 1)
y_exp = np.exp(exp_coef[1]) * np.exp(exp_coef[0] * X)
r2_exp = r2_score(y, y_exp)
rmse_exp = rmse(y, y_exp)

print("\n4. EXPONENTIAL MODEL: C = a * exp(b*year)")
print(f"   Coefficients: a={np.exp(exp_coef[1]):.2f}, b={exp_coef[0]:.4f}")
print(f"   R-squared: {r2_exp:.4f}")
print(f"   RMSE: {rmse_exp:.2f}")

# 5. Power Law Model: C = a * year^b (for positive year values)
# Shift year to make all positive
X_shifted = X - X.min() + 0.1  # Add small constant to avoid log(0)
log_X = np.log(X_shifted)
log_y_all = np.log(y)
power_coef = np.polyfit(log_X, log_y_all, 1)
y_power = np.exp(power_coef[1]) * X_shifted**power_coef[0]
r2_power = r2_score(y, y_power)
rmse_power = rmse(y, y_power)

print("\n5. POWER LAW MODEL: C = a * (year-shifted)^b")
print(f"   Coefficients: a={np.exp(power_coef[1]):.2f}, b={power_coef[0]:.4f}")
print(f"   R-squared: {r2_power:.4f}")
print(f"   RMSE: {rmse_power:.2f}")

# Model comparison summary
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY (ranked by R-squared)")
print("="*80)
models = [
    ('Linear', r2_linear, rmse_linear),
    ('Quadratic', r2_quad, rmse_quad),
    ('Cubic', r2_cubic, rmse_cubic),
    ('Exponential', r2_exp, rmse_exp),
    ('Power Law', r2_power, rmse_power)
]
models_sorted = sorted(models, key=lambda x: x[1], reverse=True)
for i, (name, r2, rmse_val) in enumerate(models_sorted, 1):
    print(f"{i}. {name:15s} R²={r2:.4f}  RMSE={rmse_val:.2f}")

# Visualization: Compare all models
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Growth Pattern Analysis: Comparing Functional Forms', fontsize=14, fontweight='bold')

# Create smooth line for predictions
X_smooth = np.linspace(X.min(), X.max(), 200)

# Plot 1: Linear
ax1 = axes[0, 0]
ax1.scatter(X, y, alpha=0.5, s=40, color='gray', label='Data')
ax1.plot(X_smooth, np.polyval(linear_coef, X_smooth), 'r-', linewidth=2, label='Linear fit')
ax1.set_xlabel('Year')
ax1.set_ylabel('C')
ax1.set_title(f'A) Linear Model (R²={r2_linear:.4f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Quadratic
ax2 = axes[0, 1]
ax2.scatter(X, y, alpha=0.5, s=40, color='gray', label='Data')
ax2.plot(X_smooth, np.polyval(quad_coef, X_smooth), 'b-', linewidth=2, label='Quadratic fit')
ax2.set_xlabel('Year')
ax2.set_ylabel('C')
ax2.set_title(f'B) Quadratic Model (R²={r2_quad:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Cubic
ax3 = axes[0, 2]
ax3.scatter(X, y, alpha=0.5, s=40, color='gray', label='Data')
ax3.plot(X_smooth, np.polyval(cubic_coef, X_smooth), 'g-', linewidth=2, label='Cubic fit')
ax3.set_xlabel('Year')
ax3.set_ylabel('C')
ax3.set_title(f'C) Cubic Model (R²={r2_cubic:.4f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Exponential
ax4 = axes[1, 0]
ax4.scatter(X, y, alpha=0.5, s=40, color='gray', label='Data')
y_exp_smooth = np.exp(exp_coef[1]) * np.exp(exp_coef[0] * X_smooth)
ax4.plot(X_smooth, y_exp_smooth, 'm-', linewidth=2, label='Exponential fit')
ax4.set_xlabel('Year')
ax4.set_ylabel('C')
ax4.set_title(f'D) Exponential Model (R²={r2_exp:.4f})')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Power Law
ax5 = axes[1, 1]
ax5.scatter(X, y, alpha=0.5, s=40, color='gray', label='Data')
X_smooth_shifted = X_smooth - X.min() + 0.1
y_power_smooth = np.exp(power_coef[1]) * X_smooth_shifted**power_coef[0]
ax5.plot(X_smooth, y_power_smooth, 'c-', linewidth=2, label='Power law fit')
ax5.set_xlabel('Year')
ax5.set_ylabel('C')
ax5.set_title(f'E) Power Law Model (R²={r2_power:.4f})')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Residual comparison for best models
ax6 = axes[1, 2]
ax6.scatter(X, y - y_linear, alpha=0.5, s=30, label='Linear', color='red')
ax6.scatter(X, y - y_quad, alpha=0.5, s=30, label='Quadratic', color='blue')
ax6.scatter(X, y - y_cubic, alpha=0.5, s=30, label='Cubic', color='green')
ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax6.set_xlabel('Year')
ax6.set_ylabel('Residuals')
ax6.set_title('F) Residual Comparison')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/02_functional_form_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nFunctional form comparison plot saved.")

# Store results for later use
results = {
    'linear': {'coef': linear_coef, 'r2': r2_linear, 'rmse': rmse_linear},
    'quadratic': {'coef': quad_coef, 'r2': r2_quad, 'rmse': rmse_quad},
    'cubic': {'coef': cubic_coef, 'r2': r2_cubic, 'rmse': rmse_cubic},
    'exponential': {'coef': exp_coef, 'r2': r2_exp, 'rmse': rmse_exp},
    'power': {'coef': power_coef, 'r2': r2_power, 'rmse': rmse_power}
}

# Save results
import pickle
with open('/workspace/eda/analyst_2/model_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Model results saved.")
