"""
Variance-Mean Relationship Analysis
Critical for distinguishing Poisson vs Negative Binomial
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Setup
sns.set_style("whitegrid")
output_dir = Path('/workspace/eda/analyst_2/visualizations')

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
year = data['year'].values
C = data['C'].values

print("=" * 80)
print("VARIANCE-MEAN RELATIONSHIP ANALYSIS")
print("=" * 80)

# Overall variance-mean ratio
overall_var = C.var(ddof=1)
overall_mean = C.mean()
var_mean_ratio = overall_var / overall_mean

print(f"\nOVERALL STATISTICS:")
print(f"  Mean: {overall_mean:.2f}")
print(f"  Variance: {overall_var:.2f}")
print(f"  Variance/Mean Ratio: {var_mean_ratio:.3f}")
print(f"  Standard Deviation: {C.std(ddof=1):.2f}")
print(f"  Coefficient of Variation: {C.std(ddof=1)/overall_mean:.3f}")

# Calculate variance-mean relationship using moving windows
print("\n" + "=" * 80)
print("MOVING WINDOW ANALYSIS (window size = 10)")
print("=" * 80)

window_size = 10
n_windows = len(C) - window_size + 1
window_means = []
window_vars = []
window_centers = []

for i in range(n_windows):
    window = C[i:i+window_size]
    window_means.append(window.mean())
    window_vars.append(window.var(ddof=1))
    window_centers.append(i + window_size / 2)

window_means = np.array(window_means)
window_vars = np.array(window_vars)

# Fit variance = a * mean^b model
# log(var) = log(a) + b * log(mean)
log_means = np.log(window_means)
log_vars = np.log(window_vars)
slope, intercept, r_value, p_value, std_err = stats.linregress(log_means, log_vars)

print(f"Power law fit: Variance = a * Mean^b")
print(f"  Exponent (b): {slope:.3f}")
print(f"  Intercept (log a): {intercept:.3f}")
print(f"  R-squared: {r_value**2:.3f}")
print(f"  p-value: {p_value:.4e}")
print(f"\nInterpretation:")
print(f"  b = 1: Poisson-like (variance proportional to mean)")
print(f"  b = 2: Variance proportional to mean^2")
print(f"  Observed b = {slope:.3f}")

# Temporal periods analysis
print("\n" + "=" * 80)
print("TEMPORAL PERIOD ANALYSIS")
print("=" * 80)

n_periods = 4
period_size = len(C) // n_periods
period_stats = []

for i in range(n_periods):
    start = i * period_size
    end = start + period_size if i < n_periods - 1 else len(C)
    period_data = C[start:end]
    period_mean = period_data.mean()
    period_var = period_data.var(ddof=1)
    period_ratio = period_var / period_mean

    period_stats.append({
        'period': i + 1,
        'n': len(period_data),
        'mean': period_mean,
        'var': period_var,
        'var_mean_ratio': period_ratio
    })

    print(f"\nPeriod {i+1} (obs {start+1}-{end}):")
    print(f"  n = {len(period_data)}")
    print(f"  Mean = {period_mean:.2f}")
    print(f"  Variance = {period_var:.2f}")
    print(f"  Var/Mean = {period_ratio:.3f}")

period_df = pd.DataFrame(period_stats)

# Create comprehensive variance-mean plot
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. Scatter plot with theoretical lines
ax = axes[0, 0]
ax.scatter(window_means, window_vars, alpha=0.6, s=60, color='steelblue', label='Observed (windows)')

# Add theoretical lines
mean_range = np.linspace(window_means.min(), window_means.max(), 100)
ax.plot(mean_range, mean_range, 'r--', linewidth=2, label='Poisson (Var = Mean)', alpha=0.7)
ax.plot(mean_range, mean_range * var_mean_ratio, 'orange', linestyle='--',
        linewidth=2, label=f'Scaled Poisson (Var = {var_mean_ratio:.1f} × Mean)', alpha=0.7)

# Add fitted power law
fitted_vars = np.exp(intercept) * mean_range ** slope
ax.plot(mean_range, fitted_vars, 'g-', linewidth=2,
        label=f'Power law (Var = Mean^{slope:.2f})', alpha=0.7)

ax.set_xlabel('Mean Count', fontsize=11)
ax.set_ylabel('Variance', fontsize=11)
ax.set_title('Variance-Mean Relationship\n(Moving Windows)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Add text box
text = f'Overall Var/Mean = {var_mean_ratio:.2f}\nPoisson assumption: Var/Mean = 1'
ax.text(0.05, 0.95, text, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6),
        fontsize=9)

# 2. Log-log plot
ax = axes[0, 1]
ax.scatter(log_means, log_vars, alpha=0.6, s=60, color='steelblue', label='Observed')
ax.plot(log_means, intercept + slope * log_means, 'r-', linewidth=2,
        label=f'Fitted: slope = {slope:.2f}')
# Add reference lines
log_mean_range = np.array([log_means.min(), log_means.max()])
ax.plot(log_mean_range, log_mean_range, 'g--', linewidth=2,
        label='Poisson (slope = 1)', alpha=0.7)

ax.set_xlabel('log(Mean Count)', fontsize=11)
ax.set_ylabel('log(Variance)', fontsize=11)
ax.set_title('Log-Log Variance-Mean Relationship', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

text = f'Slope = {slope:.3f}\nR² = {r_value**2:.3f}'
ax.text(0.05, 0.95, text, transform=ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6),
        fontsize=9)

# 3. Variance/Mean ratio over time
ax = axes[1, 0]
# Calculate ratio for each window
var_mean_ratios = window_vars / window_means
ax.plot(window_centers, var_mean_ratios, 'o-', color='steelblue', markersize=6, alpha=0.7)
ax.axhline(1, color='red', linestyle='--', linewidth=2, label='Poisson expectation', alpha=0.7)
ax.axhline(var_mean_ratio, color='orange', linestyle='--', linewidth=2,
           label=f'Overall ratio = {var_mean_ratio:.1f}', alpha=0.7)
ax.set_xlabel('Observation Index (window center)', fontsize=11)
ax.set_ylabel('Variance / Mean Ratio', fontsize=11)
ax.set_title('Dispersion Pattern Over Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 4. Period comparison
ax = axes[1, 1]
x_pos = np.arange(len(period_df))
bars = ax.bar(x_pos, period_df['var_mean_ratio'], alpha=0.7, color='steelblue', edgecolor='black')
ax.axhline(1, color='red', linestyle='--', linewidth=2, label='Poisson', alpha=0.7)
ax.set_xlabel('Time Period', fontsize=11)
ax.set_ylabel('Variance / Mean Ratio', fontsize=11)
ax.set_title('Dispersion by Time Period', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'P{i+1}\n(μ={row["mean"]:.0f})' for i, row in period_df.iterrows()], fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (idx, row) in enumerate(period_df.iterrows()):
    ax.text(i, row['var_mean_ratio'] + 0.5, f'{row["var_mean_ratio"]:.1f}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'variance_mean_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: variance_mean_analysis.png")
plt.close()

# Index of dispersion (variance/mean) analysis
print("\n" + "=" * 80)
print("INDEX OF DISPERSION ANALYSIS")
print("=" * 80)
print(f"Index of Dispersion (ID) = Variance / Mean = {var_mean_ratio:.3f}")
print(f"\nInterpretation:")
print(f"  ID = 1: Poisson distribution")
print(f"  ID > 1: Overdispersion (suggests Negative Binomial)")
print(f"  ID < 1: Underdispersion")
print(f"\nConclusion: STRONG OVERDISPERSION detected (ID = {var_mean_ratio:.1f})")
