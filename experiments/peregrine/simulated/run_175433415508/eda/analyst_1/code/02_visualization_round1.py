"""
Visualization Round 1: Basic Distributional Plots
EDA Analyst 1
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

# Load data
data = pd.read_csv('/workspace/data/data_analyst_1.csv')
C = data['C'].values
year = data['year'].values

# ============================================================================
# PLOT 1: Multi-panel overview of count distribution
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histogram
axes[0, 0].hist(C, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(C.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {C.mean():.1f}')
axes[0, 0].axvline(np.median(C), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(C):.1f}')
axes[0, 0].set_xlabel('Count (C)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Histogram of Count Variable C', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Boxplot
bp = axes[0, 1].boxplot(C, vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('black')
axes[0, 1].set_ylabel('Count (C)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Boxplot of Count Variable C', fontsize=12, fontweight='bold')
axes[0, 1].grid(alpha=0.3, axis='y')
# Add statistics
stats_text = f"Min: {C.min()}\nQ1: {np.percentile(C, 25):.1f}\nMedian: {np.median(C):.1f}\nQ3: {np.percentile(C, 75):.1f}\nMax: {C.max()}"
axes[0, 1].text(1.3, C.mean(), stats_text, fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# QQ-plot (Normal)
stats.probplot(C, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Empirical CDF
sorted_C = np.sort(C)
cdf = np.arange(1, len(sorted_C) + 1) / len(sorted_C)
axes[1, 1].plot(sorted_C, cdf, marker='o', linestyle='-', markersize=4, color='steelblue')
axes[1, 1].set_xlabel('Count (C)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Empirical CDF', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Median')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/01_distribution_overview.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 01_distribution_overview.png")

# ============================================================================
# PLOT 2: Temporal pattern
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(year, C, alpha=0.6, s=80, color='steelblue', edgecolor='black', linewidth=0.5)

# Add regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(year, C)
line = slope * year + intercept
ax.plot(year, line, 'r-', linewidth=2, label=f'Linear fit: C = {slope:.2f}*year + {intercept:.2f}\nR² = {r_value**2:.4f}')

ax.set_xlabel('Year (Standardized)', fontsize=12, fontweight='bold')
ax.set_ylabel('Count (C)', fontsize=12, fontweight='bold')
ax.set_title('Count Variable C vs Year (Strong Positive Trend)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/02_temporal_pattern.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 02_temporal_pattern.png")

# ============================================================================
# PLOT 3: Variance-Mean relationship (diagnostic for count data)
# ============================================================================
# Split data into groups by year to examine local variance-mean relationship
n_groups = 4
data_sorted = data.sort_values('year')
group_size = len(data_sorted) // n_groups

group_means = []
group_vars = []
group_labels = []

for i in range(n_groups):
    start_idx = i * group_size
    end_idx = (i + 1) * group_size if i < n_groups - 1 else len(data_sorted)
    group_data = data_sorted.iloc[start_idx:end_idx]['C'].values
    group_means.append(group_data.mean())
    group_vars.append(group_data.var(ddof=1))
    group_labels.append(f'Group {i+1}')

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(group_means, group_vars, s=150, alpha=0.7, color='steelblue', edgecolor='black', linewidth=2)

# Add identity line (variance = mean, Poisson)
max_val = max(max(group_means), max(group_vars))
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Poisson (Var = Mean)')

# Add labels for each point
for i, (gm, gv, label) in enumerate(zip(group_means, group_vars, group_labels)):
    ax.annotate(label, (gm, gv), xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Group Mean', fontsize=12, fontweight='bold')
ax.set_ylabel('Group Variance', fontsize=12, fontweight='bold')
ax.set_title('Variance-Mean Relationship Across Groups\n(Points above red line indicate overdispersion)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Add text box with overall stats
textstr = f'Overall:\nMean = {C.mean():.2f}\nVar = {C.var(ddof=1):.2f}\nVar/Mean = {C.var(ddof=1)/C.mean():.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/03_variance_mean_relationship.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 03_variance_mean_relationship.png")

# ============================================================================
# PLOT 4: Distribution comparison with theoretical distributions
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Compare with Poisson
mean_C = C.mean()
x_range = np.arange(C.min(), C.max() + 1)

# Empirical distribution
axes[0].hist(C, bins=15, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Empirical')

# Poisson fit
poisson_pmf = stats.poisson.pmf(x_range, mean_C)
axes[0].plot(x_range, poisson_pmf, 'r-', linewidth=2, label=f'Poisson(λ={mean_C:.1f})')

axes[0].set_xlabel('Count (C)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[0].set_title('Comparison with Poisson Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: Compare with Negative Binomial
# Fit NB using method of moments
var_C = C.var(ddof=1)
# For NB: mean = mu, variance = mu + mu^2/r
# So: r = mu^2 / (var - mu)
if var_C > mean_C:
    r_est = mean_C**2 / (var_C - mean_C)
    p_est = r_est / (r_est + mean_C)

    axes[1].hist(C, bins=15, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Empirical')

    # NB PMF
    nb_pmf = stats.nbinom.pmf(x_range, r_est, p_est)
    axes[1].plot(x_range, nb_pmf, 'g-', linewidth=2, label=f'Neg. Binomial(r={r_est:.2f}, p={p_est:.3f})')

    axes[1].set_xlabel('Count (C)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[1].set_title('Comparison with Negative Binomial Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_1/visualizations/04_theoretical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved: 04_theoretical_distributions.png")

print("\n" + "="*80)
print("All Round 1 visualizations complete!")
print("="*80)
