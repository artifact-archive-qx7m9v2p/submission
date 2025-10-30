"""
Visualization: Trend Models and Time Series
Analyst 1: Temporal Patterns and Trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data and models
df = pd.read_csv('/workspace/data/data_analyst_1.csv')
df = df.sort_values('year').reset_index(drop=True)

with open('/workspace/eda/analyst_1/code/trend_models.pkl', 'rb') as f:
    results = pickle.load(f)

models = results['models']
X = results['X']
y = results['y']

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# PLOT 1: Time Series with All Trend Models
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Plot actual data
ax.scatter(X, y, s=80, alpha=0.7, color='black', zorder=5, label='Observed Data', edgecolors='white', linewidths=1.5)
ax.plot(X, y, alpha=0.3, color='gray', linewidth=1, zorder=1)

# Plot all models
colors = {'Linear': '#1f77b4', 'Quadratic': '#ff7f0e', 'Cubic': '#2ca02c',
          'Exponential': '#d62728', 'Log-Linear': '#9467bd'}
linestyles = {'Linear': '--', 'Quadratic': '-.', 'Cubic': '-',
              'Exponential': ':', 'Log-Linear': '--'}

for name, results_dict in models.items():
    ax.plot(X, results_dict['predictions'],
            label=f"{name} (R²={results_dict['r2']:.3f})",
            color=colors.get(name, 'gray'),
            linestyle=linestyles.get(name, '-'),
            linewidth=2.5, alpha=0.8)

ax.set_xlabel('Standardized Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Count (C)', fontsize=12, fontweight='bold')
ax.set_title('Temporal Trends: Comparison of Functional Forms', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/01_trend_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: 01_trend_comparison.png")
plt.close()

# ============================================================================
# PLOT 2: Multi-panel - Best Models Only
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Select top 3 models
top_models = ['Cubic', 'Quadratic', 'Exponential']

for idx, name in enumerate(top_models):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    results_dict = models[name]
    residuals = y - results_dict['predictions']

    # Scatter and fit
    ax.scatter(X, y, s=60, alpha=0.6, color='black', zorder=5, edgecolors='white', linewidths=1)
    ax.plot(X, results_dict['predictions'], color=colors.get(name, 'blue'), linewidth=3, label=name)

    # Add confidence band (simple std-based)
    std_resid = np.std(residuals)
    ax.fill_between(X,
                     results_dict['predictions'] - 1.96 * std_resid,
                     results_dict['predictions'] + 1.96 * std_resid,
                     alpha=0.2, color=colors.get(name, 'blue'))

    ax.set_xlabel('Standardized Year', fontsize=10, fontweight='bold')
    ax.set_ylabel('Count (C)', fontsize=10, fontweight='bold')
    ax.set_title(f"{name} Model\nR²={results_dict['r2']:.4f}, RMSE={results_dict['rmse']:.2f}",
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

# Use the 4th panel for model comparison summary
ax = axes[1, 1]
ax.axis('off')

# Create text summary
summary_text = "MODEL COMPARISON\n" + "="*40 + "\n\n"
for name in ['Cubic', 'Quadratic', 'Exponential', 'Linear']:
    r = models[name]
    summary_text += f"{name:12s}: R²={r['r2']:.4f}, RMSE={r['rmse']:.1f}\n"

summary_text += "\n" + "="*40 + "\n"
summary_text += "BEST MODEL: Cubic\n"
summary_text += f"AIC={models['Cubic']['aic']:.1f}\n\n"
summary_text += "KEY FINDINGS:\n"
summary_text += "- Non-linear growth pattern\n"
summary_text += "- Accelerating then decelerating\n"
summary_text += "- Strong autocorrelation present\n"

ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Top Trend Models: Detailed Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/02_top_models_panel.png', dpi=300, bbox_inches='tight')
print("Saved: 02_top_models_panel.png")
plt.close()

# ============================================================================
# PLOT 3: Growth Rates and Changes
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Absolute differences
ax = axes[0, 0]
abs_diff = np.diff(y)
ax.plot(X[1:], abs_diff, marker='o', linewidth=2, markersize=6, color='steelblue')
ax.axhline(y=np.mean(abs_diff), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(abs_diff):.2f}')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Standardized Year', fontsize=10, fontweight='bold')
ax.set_ylabel('Absolute Change', fontsize=10, fontweight='bold')
ax.set_title('First Differences (Absolute Change)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Percentage changes
ax = axes[0, 1]
pct_change = np.diff(y) / y[:-1] * 100
ax.plot(X[1:], pct_change, marker='o', linewidth=2, markersize=6, color='darkgreen')
ax.axhline(y=np.mean(pct_change), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pct_change):.1f}%')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax.set_xlabel('Standardized Year', fontsize=10, fontweight='bold')
ax.set_ylabel('Percentage Change (%)', fontsize=10, fontweight='bold')
ax.set_title('Growth Rate (% Change)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Rolling mean of absolute differences (window=5)
ax = axes[1, 0]
window = 5
rolling_mean = pd.Series(abs_diff).rolling(window=window, center=True).mean()
ax.plot(X[1:], abs_diff, alpha=0.3, color='steelblue', marker='o', markersize=4, label='Actual')
ax.plot(X[1:], rolling_mean, linewidth=3, color='darkblue', label=f'{window}-period MA')
ax.set_xlabel('Standardized Year', fontsize=10, fontweight='bold')
ax.set_ylabel('Absolute Change', fontsize=10, fontweight='bold')
ax.set_title(f'Smoothed Growth Pattern (Moving Average)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Log scale to check exponential growth
ax = axes[1, 1]
ax.semilogy(X, y, marker='o', linewidth=2, markersize=6, color='purple', label='Log(Count)')
# Fit linear on log scale
log_y = np.log(y)
params = np.polyfit(X, log_y, 1)
y_log_fit = np.exp(np.polyval(params, X))
ax.semilogy(X, y_log_fit, '--', linewidth=2, color='orange', label='Exponential Fit')
ax.set_xlabel('Standardized Year', fontsize=10, fontweight='bold')
ax.set_ylabel('Count (C) - Log Scale', fontsize=10, fontweight='bold')
ax.set_title('Log-Scale View (Testing Exponential Growth)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.suptitle('Growth Rate Analysis and Temporal Dynamics', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/03_growth_rates.png', dpi=300, bbox_inches='tight')
print("Saved: 03_growth_rates.png")
plt.close()

print("\nVisualization complete.")
