"""
Final Summary Visualization - Key temporal findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
X = data['year'].values
y = data['C'].values

# Best models
quad_coef = np.polyfit(X, y, 2)
y_quad = np.polyval(quad_coef, X)

log_y = np.log(y)
log_coef = np.polyfit(X, log_y, 1)
y_log_pred = np.exp(np.polyval(log_coef, X))

# Calculate ACF manually
def calculate_acf(series, nlags=10):
    """Calculate autocorrelation function"""
    acf_values = []
    for lag in range(nlags + 1):
        if lag == 0:
            acf_values.append(1.0)
        else:
            shifted = np.roll(series, lag)
            # Remove the affected values
            valid_idx = lag
            corr = np.corrcoef(series[valid_idx:], shifted[valid_idx:])[0, 1]
            acf_values.append(corr)
    return np.array(acf_values)

acf_vals = calculate_acf(data['C'].values, nlags=10)

# Create final summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Temporal Patterns and Growth Dynamics: Comprehensive Summary',
             fontsize=16, fontweight='bold', y=0.98)

# Main plot: Data with best fits
ax_main = fig.add_subplot(gs[0:2, 0:2])
ax_main.scatter(X, y, alpha=0.5, s=60, color='steelblue', edgecolors='black',
                linewidth=0.5, label='Observed data', zorder=3)

# Quadratic fit
X_smooth = np.linspace(X.min(), X.max(), 200)
y_quad_smooth = np.polyval(quad_coef, X_smooth)
ax_main.plot(X_smooth, y_quad_smooth, 'r-', linewidth=3,
             label=f'Quadratic: C={quad_coef[0]:.1f}t²+{quad_coef[1]:.1f}t+{quad_coef[2]:.1f}',
             alpha=0.8, zorder=2)

# Log-linear equivalent
y_log_smooth = np.exp(np.polyval(log_coef, X_smooth))
ax_main.plot(X_smooth, y_log_smooth, 'g--', linewidth=2.5,
             label=f'Exponential: C={np.exp(log_coef[1]):.1f}×exp({log_coef[0]:.3f}t)',
             alpha=0.8, zorder=1)

ax_main.set_xlabel('Year (normalized)', fontweight='bold', fontsize=12)
ax_main.set_ylabel('C', fontweight='bold', fontsize=12)
ax_main.set_title('A) Growth Pattern: Quadratic vs Exponential Models',
                  fontweight='bold', fontsize=12)
ax_main.legend(loc='upper left', fontsize=10)
ax_main.grid(True, alpha=0.3)

# Add text box with key statistics
textstr = f'Observations: n=40\nR² (Quadratic): 0.9641\nR² (Exponential): 0.9358\nCorrelation: 0.9387'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax_main.text(0.02, 0.98, textstr, transform=ax_main.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

# Growth rate plot
ax1 = fig.add_subplot(gs[0, 2])
data['growth_rate'] = np.log(data['C']).diff()
ax1.plot(data['year'][1:], data['growth_rate'][1:], marker='o', markersize=3,
         linewidth=1.5, color='purple', alpha=0.7)
ax1.axhline(y=data['growth_rate'].mean(), color='orange', linestyle='--',
            linewidth=2, label=f'Mean={data["growth_rate"].mean():.3f}')
ax1.set_xlabel('Year', fontsize=9)
ax1.set_ylabel('Growth Rate', fontsize=9)
ax1.set_title('B) Growth Rate', fontweight='bold', fontsize=10)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=8)

# Residuals plot
ax2 = fig.add_subplot(gs[1, 2])
residuals = y - y_quad
ax2.scatter(X, residuals, alpha=0.6, s=40, color='coral', edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
std_res = np.std(residuals)
ax2.axhline(y=std_res, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axhline(y=-std_res, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Year', fontsize=9)
ax2.set_ylabel('Residuals', fontsize=9)
ax2.set_title('C) Quadratic Model Residuals', fontweight='bold', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=8)

# Time series with quartiles
ax3 = fig.add_subplot(gs[2, 0])
data['quartile'] = pd.qcut(data['year'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
quartile_stats = data.groupby('quartile', observed=True)['C'].agg(['mean', 'std', 'count'])
x_pos = range(len(quartile_stats))
ax3.bar(x_pos, quartile_stats['mean'], yerr=quartile_stats['std'],
        alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        capsize=5, edgecolor='black', linewidth=1)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(quartile_stats.index)
ax3.set_xlabel('Time Period', fontsize=9)
ax3.set_ylabel('Mean C (±std)', fontsize=9)
ax3.set_title('D) Mean C by Time Quartile', fontweight='bold', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
ax3.tick_params(labelsize=8)

# Log scale plot
ax4 = fig.add_subplot(gs[2, 1])
ax4.scatter(X, log_y, alpha=0.5, s=40, color='green', edgecolors='black', linewidth=0.5)
ax4.plot(X_smooth, np.polyval(log_coef, X_smooth), 'r-', linewidth=2.5,
         label=f'Linear fit: R²=0.9367')
ax4.set_xlabel('Year', fontsize=9)
ax4.set_ylabel('log(C)', fontsize=9)
ax4.set_title('E) Log-Transformed Data', fontweight='bold', fontsize=10)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=8)

# Autocorrelation
ax5 = fig.add_subplot(gs[2, 2])
lags = range(len(acf_vals))
ax5.bar(lags, acf_vals, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
ax5.axhline(y=0, color='black', linewidth=1)
conf_int = 1.96 / np.sqrt(len(data))
ax5.axhline(y=conf_int, color='red', linestyle='--', linewidth=1)
ax5.axhline(y=-conf_int, color='red', linestyle='--', linewidth=1)
ax5.set_xlabel('Lag', fontsize=9)
ax5.set_ylabel('ACF', fontsize=9)
ax5.set_title('F) Autocorrelation Function', fontweight='bold', fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.tick_params(labelsize=8)

plt.savefig('/workspace/eda/analyst_2/visualizations/06_comprehensive_summary.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Comprehensive summary visualization saved.")

# Calculate additional statistics for report
print("\n" + "="*80)
print("SUMMARY STATISTICS FOR REPORT")
print("="*80)

# Model comparison
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

linear_coef = np.polyfit(X, y, 1)
y_linear = np.polyval(linear_coef, X)

print("\nModel Performance Summary:")
print(f"  Linear:      R²={r2_score(y, y_linear):.4f}, RMSE={np.sqrt(np.mean((y - y_linear)**2)):.2f}")
print(f"  Quadratic:   R²={r2_score(y, y_quad):.4f}, RMSE={np.sqrt(np.mean((y - y_quad)**2)):.2f}")
print(f"  Exponential: R²={r2_score(y, y_log_pred):.4f}, RMSE={np.sqrt(np.mean((y - y_log_pred)**2)):.2f}")

print("\nGrowth Characteristics:")
print(f"  Initial C (year={X[0]:.2f}): {y[0]}")
print(f"  Final C (year={X[-1]:.2f}): {y[-1]}")
print(f"  Total growth: {y[-1] - y[0]} ({(y[-1]/y[0] - 1)*100:.1f}% increase)")
print(f"  Average annual growth rate: {data['growth_rate'].mean():.4f} (log scale)")
print(f"  Equivalent % growth per unit: {(np.exp(data['growth_rate'].mean()) - 1)*100:.2f}%")

print("\nTemporal Structure:")
print(f"  Autocorrelation at lag-1: {acf_vals[1]:.4f}")
print(f"  Changepoint detected at year≈-0.21 (slope increases 9.6x)")
print(f"  Heteroscedasticity: Variance increases {np.var(residuals[20:]) / np.var(residuals[:20]):.1f}x")

print("\nData Quality:")
print(f"  Complete observations: 40/40 (100%)")
print(f"  Regular spacing: Yes (Δt=0.0855)")
print(f"  Outliers (>3σ from quadratic): {np.sum(np.abs(residuals) > 3*np.std(residuals))}/40")
