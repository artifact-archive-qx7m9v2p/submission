"""
Comprehensive Visualizations for Model Diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
data = pd.read_csv('/workspace/data/data_analyst_3.csv')
results = np.load('/workspace/eda/analyst_3/code/model_results.npz')

# Extract results
residuals_linear = results['residuals_linear']
fitted_linear = results['fitted_linear']
residuals_log = results['residuals_log']
fitted_log = results['fitted_log']
residuals_sqrt = results['residuals_sqrt']
fitted_sqrt = results['fitted_sqrt']
year = results['year']
C = results['C']

# Plot 1: Residual plots for all three models (multi-panel)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# Model 1: Linear
axes[0, 0].scatter(fitted_linear, residuals_linear, alpha=0.6, s=50)
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Linear Model: Residuals vs Fitted')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(year, residuals_linear, alpha=0.6, s=50)
axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Year (standardized)')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Linear Model: Residuals vs Year')
axes[0, 1].grid(True, alpha=0.3)

stats.probplot(residuals_linear, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('Linear Model: Q-Q Plot')
axes[0, 2].grid(True, alpha=0.3)

# Model 2: Log
axes[1, 0].scatter(fitted_log, residuals_log, alpha=0.6, s=50, color='orange')
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Fitted Values (log scale)')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Log Model: Residuals vs Fitted')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(year, residuals_log, alpha=0.6, s=50, color='orange')
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Year (standardized)')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Log Model: Residuals vs Year')
axes[1, 1].grid(True, alpha=0.3)

stats.probplot(residuals_log, dist="norm", plot=axes[1, 2])
axes[1, 2].set_title('Log Model: Q-Q Plot')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].get_lines()[0].set_color('orange')
axes[1, 2].get_lines()[1].set_color('red')

# Model 3: Sqrt
axes[2, 0].scatter(fitted_sqrt, residuals_sqrt, alpha=0.6, s=50, color='green')
axes[2, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[2, 0].set_xlabel('Fitted Values (sqrt scale)')
axes[2, 0].set_ylabel('Residuals')
axes[2, 0].set_title('Sqrt Model: Residuals vs Fitted')
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].scatter(year, residuals_sqrt, alpha=0.6, s=50, color='green')
axes[2, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[2, 1].set_xlabel('Year (standardized)')
axes[2, 1].set_ylabel('Residuals')
axes[2, 1].set_title('Sqrt Model: Residuals vs Year')
axes[2, 1].grid(True, alpha=0.3)

stats.probplot(residuals_sqrt, dist="norm", plot=axes[2, 2])
axes[2, 2].set_title('Sqrt Model: Q-Q Plot')
axes[2, 2].grid(True, alpha=0.3)
axes[2, 2].get_lines()[0].set_color('green')
axes[2, 2].get_lines()[1].set_color('red')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/residual_diagnostics_all_models.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: residual_diagnostics_all_models.png")

# Plot 2: Variance-Mean relationship (critical for count data)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Calculate variance in moving windows
window_size = 10
n_windows = len(C) - window_size + 1
window_means = []
window_vars = []

for i in range(n_windows):
    window = C[i:i+window_size]
    window_means.append(np.mean(window))
    window_vars.append(np.var(window, ddof=1))

axes[0].scatter(window_means, window_vars, alpha=0.7, s=100, edgecolors='black', linewidth=1.5)
# Add reference lines
mean_range = np.array([min(window_means), max(window_means)])
axes[0].plot(mean_range, mean_range, 'r--', label='Poisson (var=mean)', linewidth=2)
axes[0].plot(mean_range, 2*mean_range, 'g--', label='var=2×mean', linewidth=2, alpha=0.7)
axes[0].plot(mean_range, 0.5*mean_range, 'b--', label='var=0.5×mean', linewidth=2, alpha=0.7)
axes[0].set_xlabel('Mean Count (sliding window)', fontsize=12)
axes[0].set_ylabel('Variance (sliding window)', fontsize=12)
axes[0].set_title('Variance-Mean Relationship\n(Window size=10)', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Alternative: scatter of fitted vs squared residuals
axes[1].scatter(fitted_linear, residuals_linear**2, alpha=0.6, s=50)
axes[1].set_xlabel('Fitted Values (Linear Model)', fontsize=12)
axes[1].set_ylabel('Squared Residuals', fontsize=12)
axes[1].set_title('Scale-Location Plot\n(Linear Model)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Add smoothed line
from scipy.interpolate import make_interp_spline
x_sorted_idx = np.argsort(fitted_linear)
x_sorted = fitted_linear[x_sorted_idx]
y_sorted = (residuals_linear**2)[x_sorted_idx]
spl = make_interp_spline(x_sorted, y_sorted, k=3)
x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
y_smooth = spl(x_smooth)
axes[1].plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smoothed trend')
axes[1].legend()

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/variance_mean_relationship.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: variance_mean_relationship.png")

# Plot 3: Dispersion check - comparing to theoretical Poisson
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Observed distribution
axes[0, 0].hist(C, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
axes[0, 0].set_xlabel('Count (C)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Observed Distribution', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Poisson distribution with same mean
mean_C = C.mean()
from scipy.stats import poisson
x_poisson = np.arange(0, int(C.max()) + 1)
poisson_probs = poisson.pmf(x_poisson, mean_C)
axes[0, 1].bar(x_poisson, poisson_probs * len(C), alpha=0.5, color='red', label='Poisson(λ={:.1f})'.format(mean_C))
axes[0, 1].hist(C, bins=20, alpha=0.5, edgecolor='black', color='steelblue', label='Observed')
axes[0, 1].set_xlabel('Count (C)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Observed vs Poisson Distribution', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Negative Binomial might be better - show variance structure over time
# Split data into thirds
n_obs = len(C)
third = n_obs // 3
segment1 = C[:third]
segment2 = C[third:2*third]
segment3 = C[2*third:]

segments = [segment1, segment2, segment3]
segment_labels = ['Early\n(low counts)', 'Middle\n(medium counts)', 'Late\n(high counts)']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

positions = [1, 2, 3]
bp = axes[1, 0].boxplot(segments, positions=positions, labels=segment_labels,
                        patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1, 0].set_ylabel('Count (C)', fontsize=12)
axes[1, 0].set_xlabel('Time Period', fontsize=12)
axes[1, 0].set_title('Distribution by Time Period\n(Equal-sized segments)', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Add mean and variance annotations
for i, (seg, label) in enumerate(zip(segments, segment_labels)):
    mean_val = np.mean(seg)
    var_val = np.var(seg, ddof=1)
    axes[1, 0].text(i+1, mean_val, f'μ={mean_val:.1f}\nσ²={var_val:.1f}',
                   ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Time series plot with variance bands
axes[1, 1].scatter(year, C, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
axes[1, 1].plot(year, fitted_linear, 'r-', linewidth=2, label='Linear fit')
# Add +/- 2 std bands
std_linear = np.std(residuals_linear)
axes[1, 1].fill_between(year, fitted_linear - 2*std_linear, fitted_linear + 2*std_linear,
                        alpha=0.2, color='red', label='±2 SD')
axes[1, 1].set_xlabel('Year (standardized)', fontsize=12)
axes[1, 1].set_ylabel('Count (C)', fontsize=12)
axes[1, 1].set_title('Time Series with Uncertainty Bands', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/dispersion_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: dispersion_analysis.png")

# Plot 4: Transformation comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original scale
axes[0, 0].scatter(year, C, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
axes[0, 0].plot(year, fitted_linear, 'r-', linewidth=2)
axes[0, 0].set_xlabel('Year', fontsize=12)
axes[0, 0].set_ylabel('Count (C)', fontsize=12)
axes[0, 0].set_title('Original Scale\nR²={:.4f}'.format(results['r2_linear']), fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Log scale
axes[0, 1].scatter(year, np.log(C), alpha=0.6, s=50, color='orange', edgecolors='black', linewidth=0.5)
axes[0, 1].plot(year, fitted_log, 'r-', linewidth=2)
axes[0, 1].set_xlabel('Year', fontsize=12)
axes[0, 1].set_ylabel('log(Count)', fontsize=12)
axes[0, 1].set_title('Log Transformation\nR²={:.4f}'.format(results['r2_log']), fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Sqrt scale
axes[1, 0].scatter(year, np.sqrt(C), alpha=0.6, s=50, color='green', edgecolors='black', linewidth=0.5)
axes[1, 0].plot(year, fitted_sqrt, 'r-', linewidth=2)
axes[1, 0].set_xlabel('Year', fontsize=12)
axes[1, 0].set_ylabel('sqrt(Count)', fontsize=12)
axes[1, 0].set_title('Square Root Transformation\nR²={:.4f}'.format(results['r2_sqrt']), fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Model comparison summary
models = ['Linear', 'Log', 'Sqrt']
r2_values = [results['r2_linear'], results['r2_log'], results['r2_sqrt']]
bp_pvalues = [results['bp_pvalue_linear'], results['bp_pvalue_log'], results['bp_pvalue_sqrt']]
sw_pvalues = [results['sw_pvalue_linear'], results['sw_pvalue_log'], results['sw_pvalue_sqrt']]

x_pos = np.arange(len(models))
width = 0.25

axes[1, 1].bar(x_pos - width, r2_values, width, label='R² (higher better)', alpha=0.8)
axes[1, 1].bar(x_pos, bp_pvalues, width, label='BP p-value (>0.05 better)', alpha=0.8)
axes[1, 1].bar(x_pos + width, sw_pvalues, width, label='SW p-value (>0.05 better)', alpha=0.8)
axes[1, 1].axhline(y=0.05, color='r', linestyle='--', linewidth=2, label='α=0.05')
axes[1, 1].set_xlabel('Model Type', fontsize=12)
axes[1, 1].set_ylabel('Value', fontsize=12)
axes[1, 1].set_title('Model Comparison Metrics', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(models)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/transformation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: transformation_comparison.png")

# Plot 5: Advanced diagnostic - influence and leverage
# Cook's distance approximation
n = len(C)
p = 2  # number of parameters (intercept + slope)
mse = np.sum(residuals_linear**2) / (n - p)
leverage = 1/n + (year - year.mean())**2 / np.sum((year - year.mean())**2)
cooks_d = (residuals_linear**2 / (p * mse)) * (leverage / (1 - leverage)**2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Cook's distance plot
axes[0].stem(range(len(cooks_d)), cooks_d, basefmt=' ')
axes[0].axhline(y=4/n, color='r', linestyle='--', linewidth=2, label='Threshold (4/n)')
axes[0].set_xlabel('Observation Index', fontsize=12)
axes[0].set_ylabel("Cook's Distance", fontsize=12)
axes[0].set_title("Cook's Distance (Influence Diagnostic)", fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Highlight influential points
influential = np.where(cooks_d > 4/n)[0]
if len(influential) > 0:
    for idx in influential:
        axes[0].annotate(f'{idx}', (idx, cooks_d[idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

# Leverage vs residuals
axes[1].scatter(leverage, residuals_linear, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].axvline(x=2*p/n, color='orange', linestyle='--', linewidth=2, label='High leverage (2p/n)')
axes[1].set_xlabel('Leverage', fontsize=12)
axes[1].set_ylabel('Residuals', fontsize=12)
axes[1].set_title('Leverage vs Residuals', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/influence_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: influence_diagnostics.png")

print("\nAll visualizations created successfully!")
