"""
Diagnostic Plots for Distribution Fitting
Focus: Visual assessment of fit quality and assumptions
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

mean_C = df['C'].mean()
var_C = df['C'].var(ddof=1)

# Fit NB parameters
r_nb = 1.5493
p_nb = 0.0140

# ============================================================================
# FIGURE 5: Quantile-Quantile Plots
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Q-Q plot for Poisson
ax = axes[0]
theoretical_quantiles_poisson = poisson.ppf(np.linspace(0.01, 0.99, len(df)), mean_C)
observed_quantiles = np.sort(df['C'])
ax.scatter(theoretical_quantiles_poisson, observed_quantiles, alpha=0.7, s=60, color='steelblue')
# Add reference line
min_val = min(theoretical_quantiles_poisson.min(), observed_quantiles.min())
max_val = max(theoretical_quantiles_poisson.max(), observed_quantiles.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
ax.set_xlabel('Theoretical Quantiles (Poisson)', fontsize=11)
ax.set_ylabel('Observed Quantiles', fontsize=11)
ax.set_title('Q-Q Plot: Poisson Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Q-Q plot for Negative Binomial
ax = axes[1]
theoretical_quantiles_nb = nbinom.ppf(np.linspace(0.01, 0.99, len(df)), r_nb, p_nb)
ax.scatter(theoretical_quantiles_nb, observed_quantiles, alpha=0.7, s=60, color='steelblue')
min_val = min(theoretical_quantiles_nb.min(), observed_quantiles.min())
max_val = max(theoretical_quantiles_nb.max(), observed_quantiles.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
ax.set_xlabel('Theoretical Quantiles (Neg. Binomial)', fontsize=11)
ax.set_ylabel('Observed Quantiles', fontsize=11)
ax.set_title('Q-Q Plot: Negative Binomial Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/05_qq_plots.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: 05_qq_plots.png")

# ============================================================================
# FIGURE 6: Probability-Probability Plots
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# P-P plot for Poisson
ax = axes[0]
observed_cdf = np.arange(1, len(df) + 1) / len(df)
theoretical_cdf_poisson = poisson.cdf(np.sort(df['C']), mean_C)
ax.scatter(theoretical_cdf_poisson, observed_cdf, alpha=0.7, s=60, color='steelblue')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')
ax.set_xlabel('Theoretical CDF (Poisson)', fontsize=11)
ax.set_ylabel('Empirical CDF', fontsize=11)
ax.set_title('P-P Plot: Poisson Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# P-P plot for Negative Binomial
ax = axes[1]
theoretical_cdf_nb = nbinom.cdf(np.sort(df['C']), r_nb, p_nb)
ax.scatter(theoretical_cdf_nb, observed_cdf, alpha=0.7, s=60, color='steelblue')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')
ax.set_xlabel('Theoretical CDF (Neg. Binomial)', fontsize=11)
ax.set_ylabel('Empirical CDF', fontsize=11)
ax.set_title('P-P Plot: Negative Binomial Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/06_pp_plots.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: 06_pp_plots.png")

# ============================================================================
# FIGURE 7: Residual Analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Calculate Pearson residuals for Poisson and NB
# Pearson residual = (observed - expected) / sqrt(expected_variance)

# For Poisson: variance = mean
pearson_resid_poisson = (df['C'] - mean_C) / np.sqrt(mean_C)

# For NB: variance = mean + mean^2/r
nb_mean = r_nb * (1 - p_nb) / p_nb
nb_var = r_nb * (1 - p_nb) / p_nb**2
pearson_resid_nb = (df['C'] - nb_mean) / np.sqrt(nb_var)

# Panel 1: Pearson residuals for Poisson over time
ax = axes[0, 0]
ax.scatter(df['time_index'], pearson_resid_poisson, alpha=0.7, s=60, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Pearson Residual', fontsize=11)
ax.set_title('Pearson Residuals: Poisson Model', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 2: Pearson residuals for NB over time
ax = axes[0, 1]
ax.scatter(df['time_index'], pearson_resid_nb, alpha=0.7, s=60, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Pearson Residual', fontsize=11)
ax.set_title('Pearson Residuals: Negative Binomial Model', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 3: Distribution of Pearson residuals (Poisson)
ax = axes[1, 0]
ax.hist(pearson_resid_poisson, bins=15, alpha=0.7, color='steelblue', edgecolor='black', density=True)
x_range = np.linspace(pearson_resid_poisson.min(), pearson_resid_poisson.max(), 100)
normal_fit = stats.norm.pdf(x_range, 0, 1)
ax.plot(x_range, normal_fit, 'r-', linewidth=2, label='N(0,1)')
ax.set_xlabel('Pearson Residual', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Residuals: Poisson', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Distribution of Pearson residuals (NB)
ax = axes[1, 1]
ax.hist(pearson_resid_nb, bins=15, alpha=0.7, color='steelblue', edgecolor='black', density=True)
x_range = np.linspace(pearson_resid_nb.min(), pearson_resid_nb.max(), 100)
normal_fit = stats.norm.pdf(x_range, 0, 1)
ax.plot(x_range, normal_fit, 'r-', linewidth=2, label='N(0,1)')
ax.set_xlabel('Pearson Residual', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of Residuals: Neg. Binomial', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/07_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: 07_residual_analysis.png")

# ============================================================================
# FIGURE 8: Rootogram (specialized count data diagnostic)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Create bins for observed counts
count_range = np.arange(df['C'].min(), df['C'].max() + 1)
observed_freq = np.array([np.sum(df['C'] == c) for c in count_range])

# Expected frequencies
expected_freq_poisson = np.array([poisson.pmf(c, mean_C) * len(df) for c in count_range])
expected_freq_nb = np.array([nbinom.pmf(c, r_nb, p_nb) * len(df) for c in count_range])

# Hanging rootogram for Poisson
ax = axes[0]
hanging_bars = np.sqrt(observed_freq) - np.sqrt(expected_freq_poisson)
colors = ['red' if h < 0 else 'steelblue' for h in hanging_bars]
ax.bar(count_range, hanging_bars, width=5, alpha=0.7, color=colors, edgecolor='black')
ax.axhline(0, color='black', linewidth=2)
ax.set_xlabel('Count Value', fontsize=11)
ax.set_ylabel('sqrt(Observed) - sqrt(Expected)', fontsize=11)
ax.set_title('Hanging Rootogram: Poisson', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Hanging rootogram for Negative Binomial
ax = axes[1]
hanging_bars = np.sqrt(observed_freq) - np.sqrt(expected_freq_nb)
colors = ['red' if h < 0 else 'steelblue' for h in hanging_bars]
ax.bar(count_range, hanging_bars, width=5, alpha=0.7, color=colors, edgecolor='black')
ax.axhline(0, color='black', linewidth=2)
ax.set_xlabel('Count Value', fontsize=11)
ax.set_ylabel('sqrt(Observed) - sqrt(Expected)', fontsize=11)
ax.set_title('Hanging Rootogram: Negative Binomial', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/08_rootograms.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: 08_rootograms.png")

# ============================================================================
# FIGURE 9: Outlier and Influence Diagnostics
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Cook's distance-like measure
# For count data: influence = (residual^2 * leverage)
# Simple leverage = 1/n
ax = axes[0, 0]
leverage = np.ones(len(df)) / len(df)
influence_poisson = pearson_resid_poisson**2 * leverage
ax.scatter(df['time_index'], influence_poisson, alpha=0.7, s=60, color='steelblue')
threshold = 4 / len(df)  # Common threshold for Cook's D
ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Influence Measure', fontsize=11)
ax.set_title('Influence Diagnostic: Poisson', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Cook's distance-like for NB
ax = axes[0, 1]
influence_nb = pearson_resid_nb**2 * leverage
ax.scatter(df['time_index'], influence_nb, alpha=0.7, s=60, color='steelblue')
ax.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Influence Measure', fontsize=11)
ax.set_title('Influence Diagnostic: Neg. Binomial', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Standardized residuals vs fitted values (Poisson)
ax = axes[1, 0]
fitted_poisson = np.full(len(df), mean_C)
std_resid_poisson = pearson_resid_poisson
ax.scatter(fitted_poisson, std_resid_poisson, alpha=0.7, s=60, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Fitted Values', fontsize=11)
ax.set_ylabel('Standardized Residuals', fontsize=11)
ax.set_title('Residuals vs Fitted: Poisson', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 4: Standardized residuals vs fitted values (NB)
ax = axes[1, 1]
fitted_nb = np.full(len(df), nb_mean)
std_resid_nb = pearson_resid_nb
ax.scatter(fitted_nb, std_resid_nb, alpha=0.7, s=60, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Fitted Values', fontsize=11)
ax.set_ylabel('Standardized Residuals', fontsize=11)
ax.set_title('Residuals vs Fitted: Neg. Binomial', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/09_influence_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: 09_influence_diagnostics.png")

print("\n=== Diagnostic visualization creation complete ===")
