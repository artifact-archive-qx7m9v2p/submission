"""
Distribution Visualizations
Focus: Visualizing count distributions and comparing to theoretical distributions
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import poisson, nbinom
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
with open('/workspace/data/data_analyst_2.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({'C': data['C'], 'year': data['year']})
df['time_index'] = np.arange(len(df))

# ============================================================================
# FIGURE 1: Basic Distribution Characteristics (4-panel)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Histogram with KDE
ax = axes[0, 0]
ax.hist(df['C'], bins=20, alpha=0.7, color='steelblue', edgecolor='black', density=True)
# Add KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(df['C'])
x_range = np.linspace(df['C'].min(), df['C'].max(), 200)
ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
ax.axvline(df['C'].mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean = {df['C'].mean():.1f}')
ax.axvline(df['C'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {df['C'].median():.1f}')
ax.set_xlabel('Count (C)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Count Values', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Boxplot with individual points
ax = axes[0, 1]
bp = ax.boxplot(df['C'], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_alpha(0.7)
# Add strip plot
np.random.seed(42)
x_jitter = np.random.normal(1, 0.04, len(df))
ax.scatter(x_jitter, df['C'], alpha=0.5, s=50, color='darkblue')
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Boxplot with Individual Observations', fontsize=12, fontweight='bold')
ax.set_xticks([])
ax.grid(True, alpha=0.3, axis='y')
# Add statistics text
stats_text = f"Mean: {df['C'].mean():.1f}\nMedian: {df['C'].median():.1f}\nStd: {df['C'].std():.1f}\nIQR: {df['C'].quantile(0.75) - df['C'].quantile(0.25):.1f}"
ax.text(1.4, df['C'].max() * 0.5, stats_text, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 3: Q-Q plot
ax = axes[1, 0]
stats.probplot(df['C'], dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 4: Empirical CDF
ax = axes[1, 1]
sorted_counts = np.sort(df['C'])
y = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
ax.step(sorted_counts, y, where='post', linewidth=2, color='steelblue', label='Empirical CDF')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Median')
ax.axvline(df['C'].median(), color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Count (C)', fontsize=11)
ax.set_ylabel('Cumulative Probability', fontsize=11)
ax.set_title('Empirical Cumulative Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/01_basic_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: 01_basic_distribution.png")

# ============================================================================
# FIGURE 2: Poisson vs Negative Binomial Comparison
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

mean_C = df['C'].mean()
var_C = df['C'].var(ddof=1)

# Fit Negative Binomial
# Parameterization: var = mu + mu^2/r
# Solving: r = mu^2 / (var - mu)
r_nb = mean_C**2 / (var_C - mean_C) if var_C > mean_C else 1
p_nb = r_nb / (r_nb + mean_C)

# Panel 1: Histogram vs Poisson PMF
ax = axes[0, 0]
counts, bins, patches = ax.hist(df['C'], bins=15, alpha=0.7, color='steelblue',
                                  edgecolor='black', density=True, label='Observed')
# Overlay Poisson PMF
x_vals = np.arange(df['C'].min(), df['C'].max() + 1)
poisson_pmf = poisson.pmf(x_vals, mean_C)
ax.plot(x_vals, poisson_pmf, 'ro-', linewidth=2, markersize=4, label=f'Poisson({mean_C:.1f})', alpha=0.7)
ax.set_xlabel('Count (C)', fontsize=11)
ax.set_ylabel('Density / PMF', fontsize=11)
ax.set_title('Observed vs Poisson Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Histogram vs Negative Binomial PMF
ax = axes[0, 1]
ax.hist(df['C'], bins=15, alpha=0.7, color='steelblue',
        edgecolor='black', density=True, label='Observed')
# Overlay Negative Binomial PMF
nb_pmf = nbinom.pmf(x_vals, r_nb, p_nb)
ax.plot(x_vals, nb_pmf, 'go-', linewidth=2, markersize=4, label=f'NegBinom(r={r_nb:.2f})', alpha=0.7)
ax.set_xlabel('Count (C)', fontsize=11)
ax.set_ylabel('Density / PMF', fontsize=11)
ax.set_title('Observed vs Negative Binomial Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: All three overlaid
ax = axes[1, 0]
ax.hist(df['C'], bins=15, alpha=0.5, color='gray',
        edgecolor='black', density=True, label='Observed')
ax.plot(x_vals, poisson_pmf, 'ro-', linewidth=2, markersize=4, label=f'Poisson({mean_C:.1f})', alpha=0.7)
ax.plot(x_vals, nb_pmf, 'go-', linewidth=2, markersize=4, label=f'NegBinom(r={r_nb:.2f})', alpha=0.7)
ax.set_xlabel('Count (C)', fontsize=11)
ax.set_ylabel('Density / PMF', fontsize=11)
ax.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
# Add text about fit
fit_text = f"Mean: {mean_C:.1f}\nVar: {var_C:.1f}\nVar/Mean: {var_C/mean_C:.2f}"
ax.text(0.98, 0.97, fit_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Panel 4: Log-scale comparison
ax = axes[1, 1]
ax.hist(df['C'], bins=15, alpha=0.5, color='gray',
        edgecolor='black', density=True, label='Observed')
ax.plot(x_vals, poisson_pmf, 'ro-', linewidth=2, markersize=4, label=f'Poisson', alpha=0.7)
ax.plot(x_vals, nb_pmf, 'go-', linewidth=2, markersize=4, label=f'NegBinom', alpha=0.7)
ax.set_xlabel('Count (C)', fontsize=11)
ax.set_ylabel('Density / PMF (log scale)', fontsize=11)
ax.set_yscale('log')
ax.set_title('Distribution Comparison (Log Scale)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/02_poisson_vs_negbinom.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: 02_poisson_vs_negbinom.png")

# ============================================================================
# FIGURE 3: Mean-Variance Relationship Over Time
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Count time series
ax = axes[0, 0]
ax.plot(df['time_index'], df['C'], 'o-', color='steelblue', linewidth=2, markersize=6)
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Count Values Over Time', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 2: Rolling mean and std (window=10)
ax = axes[0, 1]
window = 10
rolling_mean = df['C'].rolling(window=window, center=True).mean()
rolling_std = df['C'].rolling(window=window, center=True).std()
ax.plot(df['time_index'], rolling_mean, 'b-', linewidth=2, label=f'Rolling Mean (w={window})')
ax.fill_between(df['time_index'],
                 rolling_mean - rolling_std,
                 rolling_mean + rolling_std,
                 alpha=0.3, label='± 1 SD')
ax.plot(df['time_index'], df['C'], 'o', color='gray', alpha=0.4, markersize=4, label='Observed')
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Count (C)', fontsize=11)
ax.set_title('Rolling Statistics', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Sliding window mean-variance relationship
ax = axes[1, 0]
window_sizes = [5, 8, 10, 12]
colors = ['red', 'blue', 'green', 'purple']
for w, color in zip(window_sizes, colors):
    means = []
    variances = []
    for i in range(len(df) - w + 1):
        window_data = df['C'].iloc[i:i+w]
        means.append(window_data.mean())
        variances.append(window_data.var(ddof=1))
    ax.scatter(means, variances, alpha=0.6, s=50, color=color, label=f'Window={w}')

# Add reference lines
max_val = max(max(means), max(variances))
x_ref = np.linspace(0, max_val, 100)
ax.plot(x_ref, x_ref, 'k--', linewidth=2, alpha=0.5, label='Var = Mean (Poisson)')
ax.set_xlabel('Window Mean', fontsize=11)
ax.set_ylabel('Window Variance', fontsize=11)
ax.set_title('Mean-Variance Relationship (Sliding Windows)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Mean vs variance for temporal segments
ax = axes[1, 1]
n_segments = 8
segment_size = len(df) // n_segments
segment_means = []
segment_vars = []
segment_positions = []

for i in range(n_segments):
    start = i * segment_size
    end = start + segment_size if i < n_segments - 1 else len(df)
    segment = df['C'].iloc[start:end]
    segment_means.append(segment.mean())
    segment_vars.append(segment.var(ddof=1))
    segment_positions.append((start + end) / 2)

scatter = ax.scatter(segment_means, segment_vars, c=segment_positions,
                     s=150, alpha=0.7, cmap='viridis', edgecolors='black', linewidth=2)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Time Position', fontsize=10)

# Add reference lines
max_val = max(max(segment_means), max(segment_vars))
x_ref = np.linspace(0, max_val, 100)
ax.plot(x_ref, x_ref, 'k--', linewidth=2, alpha=0.5, label='Var = Mean')

# Add quadratic fit line (NB suggests var = mean + mean^2/r)
if len(segment_means) > 2:
    z = np.polyfit(segment_means, segment_vars, 2)
    p = np.poly1d(z)
    x_fit = np.linspace(min(segment_means), max(segment_means), 100)
    ax.plot(x_fit, p(x_fit), 'r-', linewidth=2, alpha=0.7, label='Quadratic fit')

ax.set_xlabel('Segment Mean', fontsize=11)
ax.set_ylabel('Segment Variance', fontsize=11)
ax.set_title('Mean-Variance by Time Segment', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/03_mean_variance_relationship.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: 03_mean_variance_relationship.png")

# ============================================================================
# FIGURE 4: Dispersion Analysis Over Time
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Variance-to-mean ratio by time segment
ax = axes[0, 0]
segment_vm_ratios = [v/m for v, m in zip(segment_vars, segment_means)]
ax.plot(range(n_segments), segment_vm_ratios, 'o-', color='darkred', linewidth=2, markersize=8)
ax.axhline(1, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Poisson (ratio=1)')
ax.set_xlabel('Time Segment', fontsize=11)
ax.set_ylabel('Variance / Mean', fontsize=11)
ax.set_title('Dispersion Ratio Over Time', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: CV (coefficient of variation) by segment
ax = axes[0, 1]
segment_cvs = [np.sqrt(v)/m for v, m in zip(segment_vars, segment_means)]
ax.plot(range(n_segments), segment_cvs, 'o-', color='darkblue', linewidth=2, markersize=8)
ax.set_xlabel('Time Segment', fontsize=11)
ax.set_ylabel('Coefficient of Variation', fontsize=11)
ax.set_title('Coefficient of Variation Over Time', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 3: Residuals from linear trend
ax = axes[1, 0]
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(df['time_index'], df['C'])
fitted = slope * df['time_index'] + intercept
residuals = df['C'] - fitted
ax.scatter(df['time_index'], residuals, alpha=0.7, s=50, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.fill_between(df['time_index'], -2*residuals.std(), 2*residuals.std(),
                 alpha=0.2, color='red', label='±2 SD')
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Residuals', fontsize=11)
ax.set_title('Residuals from Linear Trend', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Distribution of residuals
ax = axes[1, 1]
ax.hist(residuals, bins=15, alpha=0.7, color='steelblue', edgecolor='black', density=True)
# Overlay normal distribution
x_range = np.linspace(residuals.min(), residuals.max(), 100)
normal_fit = stats.norm.pdf(x_range, residuals.mean(), residuals.std())
ax.plot(x_range, normal_fit, 'r-', linewidth=2, label='Normal fit')
ax.set_xlabel('Residuals', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
# Add text
resid_text = f"Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}\nSkew: {stats.skew(residuals):.2f}"
ax.text(0.98, 0.97, resid_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/04_dispersion_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: 04_dispersion_analysis.png")

print("\n=== Visualization creation complete ===")
