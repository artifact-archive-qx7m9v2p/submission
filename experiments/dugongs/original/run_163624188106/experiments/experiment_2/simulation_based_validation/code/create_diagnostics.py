"""
Create diagnostic visualizations for SBC results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# Paths
BASE_DIR = Path('/workspace/experiments/experiment_2/simulation_based_validation')
RESULTS_DIR = BASE_DIR / 'code' / 'sbc_results'
PLOTS_DIR = BASE_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

# Load results
df = pd.read_csv(RESULTS_DIR / 'sbc_results.csv')

print(f"Loaded {len(df)} successful simulations")

# Define parameters
params = ['beta_0', 'beta_1', 'gamma_0', 'gamma_1']
param_labels = {
    'beta_0': r'$\beta_0$ (intercept)',
    'beta_1': r'$\beta_1$ (log-x slope)',
    'gamma_0': r'$\gamma_0$ (log-sigma intercept)',
    'gamma_1': r'$\gamma_1$ (log-sigma slope)'
}

# ============================================================
# 1. COMPREHENSIVE RECOVERY PLOT
# ============================================================
print("\nCreating parameter recovery plot...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Simulation-Based Calibration: Parameter Recovery Assessment',
             fontsize=14, fontweight='bold')

for idx, param in enumerate(params):
    row = idx // 2

    # Column 0-1: Recovery scatter plot
    ax_scatter = axes[row, idx % 2 * 2]
    true_col = f'true_{param}'
    mean_col = f'{param}_mean'
    q05_col = f'{param}_q05'
    q95_col = f'{param}_q95'

    # Plot identity line
    true_vals = df[true_col].values
    min_val, max_val = true_vals.min(), true_vals.max()
    range_buffer = (max_val - min_val) * 0.1
    ax_scatter.plot([min_val - range_buffer, max_val + range_buffer],
                    [min_val - range_buffer, max_val + range_buffer],
                    'k--', alpha=0.5, linewidth=1, label='Perfect recovery')

    # Plot points with error bars
    for i in range(len(df)):
        color = 'green' if (df[q05_col].iloc[i] <= df[true_col].iloc[i] <= df[q95_col].iloc[i]) else 'red'
        ax_scatter.errorbar(df[true_col].iloc[i], df[mean_col].iloc[i],
                          yerr=[[df[mean_col].iloc[i] - df[q05_col].iloc[i]],
                                [df[q95_col].iloc[i] - df[mean_col].iloc[i]]],
                          fmt='o', color=color, alpha=0.5, markersize=4, capsize=2)

    # Calculate coverage
    coverage = ((df[true_col] >= df[q05_col]) & (df[true_col] <= df[q95_col])).mean()

    ax_scatter.set_xlabel(f'True {param}', fontsize=10)
    ax_scatter.set_ylabel(f'Posterior mean {param}', fontsize=10)
    ax_scatter.set_title(f'{param_labels[param]}\nCoverage: {100*coverage:.1f}%',
                        fontsize=10, fontweight='bold')
    ax_scatter.grid(True, alpha=0.3)

    # Column 2-3: Rank histogram
    ax_rank = axes[row, idx % 2 * 2 + 1]
    rank_col = f'rank_{param}'
    ranks = df[rank_col].values

    # Bin the ranks
    n_bins = 20
    n_samples = 4000  # From our posterior sampling
    bin_edges = np.linspace(0, n_samples, n_bins + 1)

    counts, edges = np.histogram(ranks, bins=bin_edges)
    expected_count = len(ranks) / n_bins

    # Plot histogram
    ax_rank.bar(edges[:-1], counts, width=np.diff(edges),
                align='edge', alpha=0.7, edgecolor='black')
    ax_rank.axhline(expected_count, color='red', linestyle='--',
                   linewidth=2, label=f'Expected (uniform)')

    # Compute uniformity test statistic
    chi_sq = np.sum((counts - expected_count)**2 / expected_count)

    ax_rank.set_xlabel('Rank statistic', fontsize=10)
    ax_rank.set_ylabel('Count', fontsize=10)
    ax_rank.set_title(f'Rank histogram\n' + r'$\chi^2$=' + f'{chi_sq:.1f}',
                     fontsize=10)
    ax_rank.legend(fontsize=8)
    ax_rank.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_recovery_comprehensive.png', dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'parameter_recovery_comprehensive.png'}")
plt.close()

# ============================================================
# 2. BIAS AND CALIBRATION ASSESSMENT
# ============================================================
print("\nCreating bias and calibration plot...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Bias and Calibration Assessment', fontsize=14, fontweight='bold')

# Panel A: Bias by parameter
ax = axes[0]
biases = []
param_names = []
for param in params:
    true_col = f'true_{param}'
    mean_col = f'{param}_mean'
    bias = (df[mean_col] - df[true_col]).mean()
    rel_bias_pct = 100 * bias / df[true_col].abs().mean()
    biases.append(rel_bias_pct)
    param_names.append(param)

colors = ['green' if abs(b) < 5 else 'orange' if abs(b) < 10 else 'red' for b in biases]
bars = ax.barh(param_names, biases, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(0, color='black', linewidth=1)
ax.axvline(-5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Relative Bias (%)', fontsize=11)
ax.set_ylabel('Parameter', fontsize=11)
ax.set_title('(A) Posterior Mean Bias\n(target: |bias| < 5%)', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

# Panel B: Coverage by parameter
ax = axes[1]
coverages = []
for param in params:
    true_col = f'true_{param}'
    q05_col = f'{param}_q05'
    q95_col = f'{param}_q95'
    coverage = 100 * ((df[true_col] >= df[q05_col]) &
                      (df[true_col] <= df[q95_col])).mean()
    coverages.append(coverage)

colors = ['green' if 85 <= c <= 95 else 'orange' if 80 <= c <= 98 else 'red' for c in coverages]
bars = ax.barh(param_names, coverages, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(90, color='red', linewidth=2, linestyle='--', label='Target (90%)')
ax.axvline(85, color='gray', linewidth=1, linestyle=':', alpha=0.5)
ax.axvline(95, color='gray', linewidth=1, linestyle=':', alpha=0.5)
ax.set_xlabel('Coverage (%)', fontsize=11)
ax.set_ylabel('Parameter', fontsize=11)
ax.set_title('(B) 90% Credible Interval Coverage\n(target: 85-95%)', fontsize=11)
ax.set_xlim([75, 100])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')

# Panel C: RMSE
ax = axes[2]
rmses = []
for param in params:
    true_col = f'true_{param}'
    mean_col = f'{param}_mean'
    rmse = np.sqrt(((df[mean_col] - df[true_col])**2).mean())
    # Normalize by standard deviation of true values
    normalized_rmse = rmse / df[true_col].std()
    rmses.append(normalized_rmse)

bars = ax.barh(param_names, rmses, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Normalized RMSE', fontsize=11)
ax.set_ylabel('Parameter', fontsize=11)
ax.set_title('(C) Recovery Precision\n(RMSE / SD of true values)', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'bias_and_calibration.png', dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'bias_and_calibration.png'}")
plt.close()

# ============================================================
# 3. PARAMETER CORRELATIONS AND IDENTIFIABILITY
# ============================================================
print("\nCreating identifiability plot...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Parameter Identifiability: Posterior Correlation Structure',
             fontsize=14, fontweight='bold')

# Plot key parameter pairs
pairs = [
    ('beta_0', 'beta_1'),
    ('beta_0', 'gamma_0'),
    ('beta_1', 'gamma_1'),
    ('gamma_0', 'gamma_1'),
    ('beta_0', 'gamma_1'),
    ('beta_1', 'gamma_0')
]

for idx, (param1, param2) in enumerate(pairs):
    ax = axes[idx // 3, idx % 3]

    mean_col1 = f'{param1}_mean'
    mean_col2 = f'{param2}_mean'

    # Scatter plot
    ax.scatter(df[mean_col1], df[mean_col2], alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidth=0.5)

    # Calculate correlation
    corr = df[mean_col1].corr(df[mean_col2])

    ax.set_xlabel(param_labels[param1], fontsize=10)
    ax.set_ylabel(param_labels[param2], fontsize=10)
    ax.set_title(f'Correlation: {corr:.3f}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'parameter_identifiability.png', dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'parameter_identifiability.png'}")
plt.close()

# ============================================================
# 4. COVERAGE BY TRUE PARAMETER VALUE
# ============================================================
print("\nCreating coverage by true value plot...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Coverage Analysis: Does calibration depend on true value?',
             fontsize=14, fontweight='bold')

for idx, param in enumerate(params):
    ax = axes[idx // 2, idx % 2]

    true_col = f'true_{param}'
    q05_col = f'{param}_q05'
    q95_col = f'{param}_q95'

    # Sort by true value
    df_sorted = df.sort_values(true_col)

    # Calculate coverage in rolling windows
    window_size = 15
    if len(df) >= window_size:
        true_vals_window = []
        coverage_window = []

        for i in range(len(df_sorted) - window_size + 1):
            window = df_sorted.iloc[i:i+window_size]
            true_mean = window[true_col].mean()
            cov = ((window[true_col] >= window[q05_col]) &
                   (window[true_col] <= window[q95_col])).mean()
            true_vals_window.append(true_mean)
            coverage_window.append(100 * cov)

        ax.plot(true_vals_window, coverage_window, 'o-', color='steelblue',
                markersize=4, linewidth=2, alpha=0.7)

    ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target 90%')
    ax.axhline(85, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(95, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel(f'True {param}', fontsize=10)
    ax.set_ylabel('Coverage (%)', fontsize=10)
    ax.set_title(param_labels[param], fontsize=11, fontweight='bold')
    ax.set_ylim([70, 100])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'coverage_by_true_value.png', dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'coverage_by_true_value.png'}")
plt.close()

# ============================================================
# 5. SIMULATION SUCCESS ANALYSIS
# ============================================================
print("\nCreating simulation success summary...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Computational Performance Assessment', fontsize=14, fontweight='bold')

# Panel A: Success rate
ax = axes[0]
n_total = 100
n_success = len(df)
n_failed = n_total - n_success

categories = ['Successful\nFits', 'Failed\nFits']
counts = [n_success, n_failed]
colors = ['green', 'red']

bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'(A) Optimization Success Rate\n{n_success}/{n_total} = {100*n_success/n_total:.1f}%',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add text annotations
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Panel B: Summary metrics
ax = axes[1]
ax.axis('off')

summary_text = f"""
COMPUTATIONAL PERFORMANCE
{'='*45}

Total Simulations:           {n_total}
Successful Fits:             {n_success} ({100*n_success/n_total:.1f}%)
Failed Optimizations:        {n_failed} ({100*n_failed/n_total:.1f}%)

QUALITY METRICS (successful fits)
{'='*45}

Coverage Rates (target: 85-95%):
  beta_0:   {coverages[0]:.1f}% {'✓' if 85 <= coverages[0] <= 95 else '✗'}
  beta_1:   {coverages[1]:.1f}% {'✓' if 85 <= coverages[1] <= 95 else '✗'}
  gamma_0:  {coverages[2]:.1f}% {'✓' if 85 <= coverages[2] <= 95 else '✗'}
  gamma_1:  {coverages[3]:.1f}% {'✓' if 85 <= coverages[3] <= 95 else '✗'}

Relative Bias (target: |bias| < 10%):
  beta_0:   {biases[0]:+.2f}% {'✓' if abs(biases[0]) < 10 else '✗'}
  beta_1:   {biases[1]:+.2f}% {'✓' if abs(biases[1]) < 10 else '✗'}
  gamma_0:  {biases[2]:+.2f}% {'✓' if abs(biases[2]) < 10 else '✗'}
  gamma_1:  {biases[3]:+.2f}% {'✓' if abs(biases[3]) < 10 else '✗'}
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'simulation_success_summary.png', dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'simulation_success_summary.png'}")
plt.close()

print("\n" + "="*60)
print("All diagnostic plots created successfully!")
print("="*60)
