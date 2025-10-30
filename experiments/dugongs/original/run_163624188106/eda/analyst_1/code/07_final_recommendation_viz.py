"""
Final Recommendation Visualization - Analyst 1
==============================================
Purpose: Create publication-quality visualization of recommended model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')

# Fit logarithmic model
log_coef = np.polyfit(np.log(data['x']), data['Y'], 1)
y_pred_log = np.polyval(log_coef, np.log(data['x']))
residuals_log = data['Y'] - y_pred_log

# Calculate variance by x range for visualization
data_sorted = data.sort_values('x')
n = len(data_sorted)
third = n // 3
low_x = data_sorted.iloc[:third]
mid_x = data_sorted.iloc[third:2*third]
high_x = data_sorted.iloc[2*third:]

# Create comprehensive final figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main plot: Data with logarithmic fit and confidence bands
ax_main = fig.add_subplot(gs[0:2, 0:2])

# Generate prediction line
x_plot = np.linspace(data['x'].min(), data['x'].max(), 200)
y_plot = log_coef[0] * np.log(x_plot) + log_coef[1]

# Calculate confidence bands (approximate)
se = np.std(residuals_log)
ax_main.fill_between(x_plot, y_plot - 1.96*se, y_plot + 1.96*se,
                      alpha=0.2, color='red', label='95% prediction interval')

# Plot data points colored by x range
colors = ['blue' if x <= low_x['x'].max() else 'green' if x <= mid_x['x'].max() else 'purple'
          for x in data['x']]
sizes = [100 if i == 26 else 80 for i in range(len(data))]

scatter = ax_main.scatter(data['x'], data['Y'], c=colors, s=sizes, alpha=0.7,
                          edgecolors='black', linewidth=1.5, zorder=5)

# Annotate influential point
ax_main.annotate('Influential\n(x=31.5)', xy=(31.5, 2.57), xytext=(27, 2.3),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# Plot fitted line
ax_main.plot(x_plot, y_plot, 'r-', linewidth=3, label='Logarithmic fit', zorder=4)

ax_main.set_xlabel('x', fontsize=14, fontweight='bold')
ax_main.set_ylabel('Y', fontsize=14, fontweight='bold')
ax_main.set_title('Recommended Model: Logarithmic Regression\n' +
                  f'Y = {log_coef[1]:.3f} + {log_coef[0]:.3f}·log(x)  |  R² = 0.897',
                  fontsize=14, fontweight='bold')

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', edgecolor='black', label='Low x (1-7): High variance'),
    Patch(facecolor='green', edgecolor='black', label='Mid x (8-13): Med variance'),
    Patch(facecolor='purple', edgecolor='black', label='High x (13-31.5): Low variance'),
    plt.Line2D([0], [0], color='r', linewidth=3, label='Logarithmic fit'),
    Patch(facecolor='red', alpha=0.2, label='95% prediction interval')
]
ax_main.legend(handles=legend_elements, loc='lower right', fontsize=10)
ax_main.grid(alpha=0.3)

# Residual plot
ax_resid = fig.add_subplot(gs[0, 2])
ax_resid.scatter(y_pred_log, residuals_log, alpha=0.7, s=60, edgecolors='black')
ax_resid.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax_resid.set_xlabel('Fitted Values', fontsize=11)
ax_resid.set_ylabel('Residuals', fontsize=11)
ax_resid.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax_resid.grid(alpha=0.3)

# Q-Q plot
ax_qq = fig.add_subplot(gs[1, 2])
stats.probplot(residuals_log, dist="norm", plot=ax_qq)
ax_qq.set_title('Q-Q Plot\n(Normality Check)', fontsize=12, fontweight='bold')
ax_qq.grid(alpha=0.3)

# Variance structure
ax_var = fig.add_subplot(gs[2, 0])
variances = [low_x['Y'].var(), mid_x['Y'].var(), high_x['Y'].var()]
x_ranges = ['Low\n(1-7)', 'Mid\n(8-13)', 'High\n(13-31.5)']
bars = ax_var.bar(x_ranges, variances, color=['blue', 'green', 'purple'],
                  alpha=0.7, edgecolor='black', linewidth=2)
ax_var.set_ylabel('Y Variance', fontsize=11, fontweight='bold')
ax_var.set_xlabel('X Range', fontsize=11, fontweight='bold')
ax_var.set_title('Heteroscedasticity Evidence\n(Levene p=0.003)', fontsize=12, fontweight='bold')
ax_var.grid(alpha=0.3, axis='y')

# Add values on bars
for bar, val in zip(bars, variances):
    height = bar.get_height()
    ax_var.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Model comparison
ax_compare = fig.add_subplot(gs[2, 1])
models = ['Linear', 'Quadratic', 'Log', 'Power', 'Exp']
r2_values = [0.677, 0.874, 0.897, 0.889, 0.618]
colors_bar = ['red' if r2 < 0.85 else 'yellow' if r2 < 0.89 else 'green' for r2 in r2_values]

bars = ax_compare.barh(models, r2_values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
ax_compare.set_xlabel('R² Value', fontsize=11, fontweight='bold')
ax_compare.set_title('Model Comparison', fontsize=12, fontweight='bold')
ax_compare.set_xlim([0.5, 1.0])
ax_compare.grid(alpha=0.3, axis='x')

# Add values on bars
for bar, val in zip(bars, r2_values):
    width = bar.get_width()
    ax_compare.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                   f'{val:.3f}',
                   ha='left', va='center', fontsize=10, fontweight='bold')

# Key recommendations text box
ax_text = fig.add_subplot(gs[2, 2])
ax_text.axis('off')

recommendations = """
KEY RECOMMENDATIONS

Model: Logarithmic Regression
Y ~ Normal(μ, σ)
μ = β₀ + β₁·log(x)
σ = f(x)  # decreasing

Why Logarithmic?
✓ Best fit (R²=0.897)
✓ Captures diminishing returns
✓ Parsimonious (2 params)
✓ Normal residuals

Critical Features:
• Heteroscedastic variance
• 7.5x variance reduction
• 1 influential point (x=31.5)
• No outliers detected

Priors (suggested):
β₀ ~ N(1.8, 0.5)
β₁ ~ N(0.3, 0.2)
σ₀ ~ Exp(10)

Likelihood: Normal
Alternative: Student-t
"""

ax_text.text(0.05, 0.95, recommendations, transform=ax_text.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Overall title
fig.suptitle('EDA Summary: Recommended Bayesian Model Specification',
            fontsize=16, fontweight='bold', y=0.98)

plt.savefig('/workspace/eda/analyst_1/visualizations/10_final_recommendation.png',
            dpi=300, bbox_inches='tight')
print("Saved: 10_final_recommendation.png")
plt.close()

print("\n" + "="*60)
print("FINAL RECOMMENDATION VISUALIZATION COMPLETE")
print("="*60)
print("\nAll EDA deliverables are ready in:")
print("  Code: /workspace/eda/analyst_1/code/")
print("  Visualizations: /workspace/eda/analyst_1/visualizations/")
print("  Report: /workspace/eda/analyst_1/findings.md")
print("  Log: /workspace/eda/analyst_1/eda_log.md")
