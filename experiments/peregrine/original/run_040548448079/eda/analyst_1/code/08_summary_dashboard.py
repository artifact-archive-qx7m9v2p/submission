"""
Summary Dashboard: Key Findings Overview
Analyst 1: Temporal Patterns and Trends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('/workspace/data/data_analyst_1.csv')
df = df.sort_values('year').reset_index(drop=True)

X = df['year'].values
y = df['C'].values
n = len(y)

with open('/workspace/eda/analyst_1/code/trend_models.pkl', 'rb') as f:
    model_results = pickle.load(f)

with open('/workspace/eda/analyst_1/code/structural_breaks.pkl', 'rb') as f:
    break_results = pickle.load(f)

models = model_results['models']

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# ============================================================================
# Panel 1: Data Overview (top left)
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(X, y, marker='o', linewidth=2, markersize=7, color='steelblue', markerfacecolor='white', markeredgewidth=2)
ax1.set_xlabel('Standardized Year', fontsize=10, fontweight='bold')
ax1.set_ylabel('Count (C)', fontsize=10, fontweight='bold')
ax1.set_title('A. Time Series Overview\n40 observations, range 19-272', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add annotations
ax1.annotate('Start: 29', xy=(X[0], y[0]), xytext=(X[0]-0.3, y[0]+30),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5), fontsize=9)
ax1.annotate('End: 245', xy=(X[-1], y[-1]), xytext=(X[-1]+0.3, y[-1]-30),
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5), fontsize=9)

# ============================================================================
# Panel 2: Model Comparison (top middle)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])
model_names = ['Linear', 'Quadratic', 'Cubic', 'Exponential', 'Log-Linear']
r2_values = [models[name]['r2'] for name in model_names]
colors_bar = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']

bars = ax2.barh(model_names, r2_values, color=colors_bar, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('R² Score', fontsize=10, fontweight='bold')
ax2.set_title('B. Model Performance (R²)', fontsize=11, fontweight='bold')
ax2.set_xlim(0.85, 1.0)
ax2.axvline(x=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add values
for bar, val in zip(bars, r2_values):
    ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
             va='center', fontsize=9, fontweight='bold')

# ============================================================================
# Panel 3: Autocorrelation Summary (top right)
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])
with open('/workspace/eda/analyst_1/code/acf_data.pkl', 'rb') as f:
    acf_data = pickle.load(f)

acf_raw = acf_data['raw']['acf']
acf_diff = acf_data['diff']['acf']
lags = np.arange(8)

width = 0.35
x = np.arange(len(lags))
ax3.bar(x - width/2, acf_raw[:8], width, label='Raw Data', color='darkred', alpha=0.7, edgecolor='black')
ax3.bar(x + width/2, acf_diff[:8], width, label='First Diff', color='darkgreen', alpha=0.7, edgecolor='black')

ax3.axhline(y=0, color='black', linewidth=1)
confidence = 1.96 / np.sqrt(len(y))
ax3.axhline(y=confidence, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.axhline(y=-confidence, color='red', linestyle='--', linewidth=1, alpha=0.5)

ax3.set_xlabel('Lag', fontsize=10, fontweight='bold')
ax3.set_ylabel('ACF', fontsize=10, fontweight='bold')
ax3.set_title('C. Autocorrelation Comparison', fontsize=11, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(lags)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Panel 4: Two-Regime Model (middle left, spans 2 columns)
# ============================================================================
ax4 = fig.add_subplot(gs[1, :2])
optimal_bp = break_results['optimal_break']['breakpoint']
params1 = break_results['optimal_break']['params1']
params2 = break_results['optimal_break']['params2']

# Plot data with regimes
ax4.scatter(X[:optimal_bp], y[:optimal_bp], s=100, alpha=0.7, color='blue',
           label=f'Regime 1 (n={optimal_bp})', edgecolors='white', linewidths=2, zorder=5)
ax4.scatter(X[optimal_bp:], y[optimal_bp:], s=100, alpha=0.7, color='red',
           label=f'Regime 2 (n={n-optimal_bp})', edgecolors='white', linewidths=2, zorder=5)

# Fitted lines
X1_fit = np.column_stack([np.ones(optimal_bp), X[:optimal_bp]])
X2_fit = np.column_stack([np.ones(n - optimal_bp), X[optimal_bp:]])
y1_fit = X1_fit @ params1
y2_fit = X2_fit @ params2

ax4.plot(X[:optimal_bp], y1_fit, linewidth=4, color='darkblue',
         label=f'Slope 1: {params1[1]:.1f}', zorder=4)
ax4.plot(X[optimal_bp:], y2_fit, linewidth=4, color='darkred',
         label=f'Slope 2: {params2[1]:.1f}', zorder=4)

# Breakpoint
ax4.axvline(x=X[optimal_bp], color='green', linestyle='--', linewidth=3, alpha=0.8,
           label=f'Break: obs {optimal_bp}', zorder=3)
ax4.axvspan(X[optimal_bp]-0.05, X[optimal_bp]+0.05, alpha=0.2, color='green', zorder=1)

ax4.set_xlabel('Standardized Year', fontsize=11, fontweight='bold')
ax4.set_ylabel('Count (C)', fontsize=11, fontweight='bold')
ax4.set_title('D. Structural Break: Two-Regime Model (730% slope increase)', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax4.grid(True, alpha=0.3)

# ============================================================================
# Panel 5: SSE by Breakpoint (middle right)
# ============================================================================
ax5 = fig.add_subplot(gs[1, 2])
breakpoints = break_results['optimal_break']['breakpoints']
sse_values = break_results['optimal_break']['sse']

ax5.plot(breakpoints, sse_values, linewidth=2.5, color='purple', marker='o', markersize=4)
ax5.axvline(x=optimal_bp, color='red', linestyle='--', linewidth=2.5, label=f'Optimal: {optimal_bp}')
ax5.scatter([optimal_bp], [min(sse_values)], s=200, color='red', marker='*',
            edgecolors='black', linewidths=2, zorder=10)

ax5.set_xlabel('Breakpoint Location (obs)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Sum of Squared Errors', fontsize=10, fontweight='bold')
ax5.set_title('E. Optimal Breakpoint Search', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# ============================================================================
# Panel 6: Growth Rates (bottom left)
# ============================================================================
ax6 = fig.add_subplot(gs[2, 0])
pct_change = np.diff(y) / y[:-1] * 100

ax6.plot(X[1:], pct_change, marker='o', linewidth=2, markersize=5, color='darkgreen', alpha=0.7)
ax6.axhline(y=np.mean(pct_change), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(pct_change):.1f}%')
ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax6.axvline(x=X[optimal_bp], color='green', linestyle='--', linewidth=2, alpha=0.5)

ax6.set_xlabel('Standardized Year', fontsize=10, fontweight='bold')
ax6.set_ylabel('% Change', fontsize=10, fontweight='bold')
ax6.set_title('F. Period-to-Period Growth Rate', fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# ============================================================================
# Panel 7: Residual ACF for Best Model (bottom middle)
# ============================================================================
ax7 = fig.add_subplot(gs[2, 1])
residuals_cubic = y - models['Cubic']['predictions']
residual_acf_cubic = acf_data['residuals']['acf']['Cubic']

lags_acf = np.arange(len(residual_acf_cubic))
ax7.stem(lags_acf, residual_acf_cubic, basefmt=' ', linefmt='steelblue', markerfmt='o')
ax7.axhline(y=0, color='black', linewidth=1)
confidence = 1.96 / np.sqrt(len(y))
ax7.axhline(y=confidence, color='red', linestyle='--', linewidth=1, alpha=0.7, label='95% CI')
ax7.axhline(y=-confidence, color='red', linestyle='--', linewidth=1, alpha=0.7)

ax7.set_xlabel('Lag', fontsize=10, fontweight='bold')
ax7.set_ylabel('ACF', fontsize=10, fontweight='bold')
ax7.set_title('G. Cubic Model: Residual ACF\n(ACF(1)=0.51, still significant)', fontsize=11, fontweight='bold')
ax7.set_xlim(-0.5, 15.5)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# ============================================================================
# Panel 8: Key Statistics (bottom right)
# ============================================================================
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

stats_text = f"""
KEY FINDINGS SUMMARY

DATA CHARACTERISTICS:
  Observations: {n}
  Range: {y.min()} to {y.max()}
  Mean: {np.mean(y):.1f}
  Total Growth: {y[-1] - y[0]} ({(y[-1]/y[0]-1)*100:.0f}%)

BEST SINGLE MODEL (Cubic):
  R²: {models['Cubic']['r2']:.4f}
  RMSE: {models['Cubic']['rmse']:.2f}
  Residual ACF(1): {residual_acf_cubic[1]:.3f}

STRUCTURAL BREAK:
  Location: Obs {optimal_bp} (year {X[optimal_bp]:.3f})
  Regime 1 Slope: {params1[1]:.2f}
  Regime 2 Slope: {params2[1]:.2f}
  Slope Increase: {((params2[1]/params1[1])-1)*100:.0f}%
  SSE Improvement: 79.9%

AUTOCORRELATION:
  Raw ACF(1): {acf_raw[1]:.3f}
  Ljung-Box p: < 0.001
  I(1) Process: Yes (stationary after 1 diff)

RECOMMENDATIONS:
  • Use two-regime framework
  • Account for count data (Poisson/NB)
  • Address autocorrelation (AR errors)
  • DO NOT ignore structural break
"""

ax8.text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
        verticalalignment='top', transform=ax8.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2))

# ============================================================================
# Main title
# ============================================================================
fig.suptitle('EDA Summary Dashboard: Temporal Patterns Analysis\nAnalyst 1 - Key Findings Overview',
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig('/workspace/eda/analyst_1/visualizations/00_summary_dashboard.png', dpi=300, bbox_inches='tight')
print("Saved: 00_summary_dashboard.png")
plt.close()

print("\nSummary dashboard complete!")
