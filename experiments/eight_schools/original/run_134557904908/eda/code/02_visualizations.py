"""
Visualization Script for Meta-Analysis Dataset
===============================================
Creates comprehensive visualizations to understand distributions,
relationships, and patterns in the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed and plotting style
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
data = pd.read_csv('/workspace/data/data.csv')
data['obs_id'] = range(1, len(data) + 1)
data['precision'] = 1 / data['sigma']**2

# Create output directory
import os
os.makedirs('/workspace/eda/visualizations', exist_ok=True)

print("Creating visualizations...")

# ============================================================================
# PLOT 1: Distribution of Observed Outcomes (y)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
axes[0].hist(data['y'], bins=6, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(data['y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {data["y"].mean():.2f}')
axes[0].axvline(data['y'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median = {data["y"].median():.2f}')
axes[0].set_xlabel('Observed Outcome (y)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution of y', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
box_parts = axes[1].boxplot(data['y'], vert=True, patch_artist=True, widths=0.5)
box_parts['boxes'][0].set_facecolor('steelblue')
box_parts['boxes'][0].set_alpha(0.7)
axes[1].scatter([1]*len(data), data['y'], alpha=0.6, s=80, color='darkblue', zorder=3)
axes[1].set_ylabel('Observed Outcome (y)', fontsize=11)
axes[1].set_title('Boxplot of y', fontsize=12, fontweight='bold')
axes[1].set_xticks([])
axes[1].grid(True, alpha=0.3, axis='y')

# Q-Q plot
stats.probplot(data['y'], dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot: y vs Normal Distribution', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/01_distribution_y.png', dpi=300, bbox_inches='tight')
print("  [1/9] Created: 01_distribution_y.png")
plt.close()

# ============================================================================
# PLOT 2: Distribution of Standard Errors (sigma)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
axes[0].hist(data['sigma'], bins=6, edgecolor='black', alpha=0.7, color='coral')
axes[0].axvline(data['sigma'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {data["sigma"].mean():.2f}')
axes[0].axvline(data['sigma'].median(), color='darkred', linestyle='--', linewidth=2, label=f'Median = {data["sigma"].median():.2f}')
axes[0].set_xlabel('Standard Error (sigma)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution of sigma', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
box_parts = axes[1].boxplot(data['sigma'], vert=True, patch_artist=True, widths=0.5)
box_parts['boxes'][0].set_facecolor('coral')
box_parts['boxes'][0].set_alpha(0.7)
axes[1].scatter([1]*len(data), data['sigma'], alpha=0.6, s=80, color='darkred', zorder=3)
axes[1].set_ylabel('Standard Error (sigma)', fontsize=11)
axes[1].set_title('Boxplot of sigma', fontsize=12, fontweight='bold')
axes[1].set_xticks([])
axes[1].grid(True, alpha=0.3, axis='y')

# Q-Q plot
stats.probplot(data['sigma'], dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot: sigma vs Normal Distribution', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/02_distribution_sigma.png', dpi=300, bbox_inches='tight')
print("  [2/9] Created: 02_distribution_sigma.png")
plt.close()

# ============================================================================
# PLOT 3: Relationship between y and sigma
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Scatter plot with labels
ax.scatter(data['sigma'], data['y'], s=120, alpha=0.7, color='purple', edgecolors='black', linewidth=1.5)

# Add observation labels
for idx, row in data.iterrows():
    ax.annotate(f"  {row['obs_id']}",
                (row['sigma'], row['y']),
                fontsize=10,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points')

# Add correlation information
corr_pearson = data['y'].corr(data['sigma'])
corr_spearman = data['y'].corr(data['sigma'], method='spearman')
ax.text(0.05, 0.95, f'Pearson r = {corr_pearson:.3f}\nSpearman ρ = {corr_spearman:.3f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add regression line
z = np.polyfit(data['sigma'], data['y'], 1)
p = np.poly1d(z)
x_line = np.linspace(data['sigma'].min(), data['sigma'].max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.6, linewidth=2, label=f'Linear fit: y = {z[0]:.2f}σ + {z[1]:.2f}')

ax.set_xlabel('Standard Error (sigma)', fontsize=12)
ax.set_ylabel('Observed Outcome (y)', fontsize=12)
ax.set_title('Relationship between y and sigma', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/03_y_vs_sigma.png', dpi=300, bbox_inches='tight')
print("  [3/9] Created: 03_y_vs_sigma.png")
plt.close()

# ============================================================================
# PLOT 4: Forest Plot - Error Bars for Each Observation
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Sort by y value for better visualization
data_sorted = data.sort_values('y', ascending=True).reset_index(drop=True)

# Plot error bars
for i, row in data_sorted.iterrows():
    ax.errorbar(row['y'], i, xerr=row['sigma'], fmt='o',
                markersize=8, capsize=5, capthick=2,
                alpha=0.7, linewidth=2, label=f"Obs {row['obs_id']}")

# Add vertical line at weighted mean
weighted_mean = np.sum(data['y'] / data['sigma']**2) / np.sum(1 / data['sigma']**2)
ax.axvline(weighted_mean, color='red', linestyle='--', linewidth=2,
           label=f'Weighted Mean = {weighted_mean:.2f}')

# Add vertical line at simple mean
ax.axvline(data['y'].mean(), color='orange', linestyle='--', linewidth=2,
           label=f'Simple Mean = {data["y"].mean():.2f}')

# Add zero line
ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Zero')

ax.set_xlabel('Outcome Value (y ± sigma)', fontsize=12)
ax.set_ylabel('Observation (sorted by y)', fontsize=12)
ax.set_title('Forest Plot: Observed Outcomes with Uncertainty', fontsize=13, fontweight='bold')
ax.set_yticks(range(len(data_sorted)))
ax.set_yticklabels([f"Obs {row['obs_id']}" for _, row in data_sorted.iterrows()])
ax.legend(loc='best', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/04_forest_plot.png', dpi=300, bbox_inches='tight')
print("  [4/9] Created: 04_forest_plot.png")
plt.close()

# ============================================================================
# PLOT 5: Precision Analysis
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: y vs precision
axes[0].scatter(data['precision'], data['y'], s=120, alpha=0.7,
                color='teal', edgecolors='black', linewidth=1.5)
for idx, row in data.iterrows():
    axes[0].annotate(f"  {row['obs_id']}",
                     (row['precision'], row['y']),
                     fontsize=10, alpha=0.8,
                     xytext=(5, 5),
                     textcoords='offset points')

corr_prec = data['y'].corr(data['precision'])
axes[0].text(0.05, 0.95, f'Correlation = {corr_prec:.3f}',
             transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[0].set_xlabel('Precision (1/sigma²)', fontsize=12)
axes[0].set_ylabel('Observed Outcome (y)', fontsize=12)
axes[0].set_title('Outcome vs Precision', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: Distribution of precisions
axes[1].bar(range(1, len(data) + 1), data['precision'],
            color='teal', alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1].axhline(data['precision'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean = {data["precision"].mean():.4f}')
axes[1].set_xlabel('Observation ID', fontsize=12)
axes[1].set_ylabel('Precision (1/sigma²)', fontsize=12)
axes[1].set_title('Precision by Observation', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/05_precision_analysis.png', dpi=300, bbox_inches='tight')
print("  [5/9] Created: 05_precision_analysis.png")
plt.close()

# ============================================================================
# PLOT 6: Heterogeneity Assessment
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Individual observations with confidence intervals
ax = axes[0, 0]
x_pos = range(1, len(data) + 1)
ax.errorbar(x_pos, data['y'], yerr=1.96*data['sigma'], fmt='o',
            markersize=8, capsize=5, capthick=2, alpha=0.7, linewidth=2)
ax.axhline(weighted_mean, color='red', linestyle='--', linewidth=2,
           label=f'Weighted Mean = {weighted_mean:.2f}')
ax.fill_between([0.5, len(data) + 0.5],
                weighted_mean - 1.96*np.sqrt(1/np.sum(data['precision'])),
                weighted_mean + 1.96*np.sqrt(1/np.sum(data['precision'])),
                alpha=0.2, color='red', label='95% CI of pooled estimate')
ax.set_xlabel('Observation ID', fontsize=11)
ax.set_ylabel('Outcome (y ± 1.96σ)', fontsize=11)
ax.set_title('95% Confidence Intervals by Observation', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(x_pos)

# Plot 2: Standardized residuals from weighted mean
ax = axes[0, 1]
residuals = data['y'] - weighted_mean
standardized_residuals = residuals / data['sigma']
colors = ['red' if abs(sr) > 1.96 else 'blue' for sr in standardized_residuals]
ax.bar(x_pos, standardized_residuals, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(1.96, color='red', linestyle='--', linewidth=2, alpha=0.5, label='±1.96 (95% threshold)')
ax.axhline(-1.96, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Observation ID', fontsize=11)
ax.set_ylabel('Standardized Residual', fontsize=11)
ax.set_title('Standardized Residuals from Weighted Mean', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(x_pos)

# Plot 3: Funnel plot (y vs sigma) - publication bias check
ax = axes[1, 0]
ax.scatter(data['y'], 1/data['sigma'], s=120, alpha=0.7,
           color='green', edgecolors='black', linewidth=1.5)
for idx, row in data.iterrows():
    ax.annotate(f"  {row['obs_id']}",
                (row['y'], 1/row['sigma']),
                fontsize=9, alpha=0.8)
ax.axvline(weighted_mean, color='red', linestyle='--', linewidth=2,
           label=f'Weighted Mean = {weighted_mean:.2f}')
ax.set_xlabel('Observed Outcome (y)', fontsize=11)
ax.set_ylabel('Precision (1/sigma)', fontsize=11)
ax.set_title('Funnel Plot: Publication Bias Check', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Galbraith plot (radial plot)
ax = axes[1, 1]
z_scores = data['y'] / data['sigma']
ax.scatter(data['precision'], z_scores, s=120, alpha=0.7,
           color='orange', edgecolors='black', linewidth=1.5)
for idx, row in data.iterrows():
    ax.annotate(f"  {row['obs_id']}",
                (row['precision'], row['y']/row['sigma']),
                fontsize=9, alpha=0.8)

# Add regression line through origin
z_scores_arr = z_scores.values
precision_arr = data['precision'].values
slope = np.sum(z_scores_arr * precision_arr) / np.sum(precision_arr)
x_line = np.linspace(0, data['precision'].max(), 100)
ax.plot(x_line, slope * x_line, "r--", alpha=0.6, linewidth=2,
        label=f'Regression: slope = {slope:.2f}')

ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Precision (1/sigma²)', fontsize=11)
ax.set_ylabel('z-score (y/sigma)', fontsize=11)
ax.set_title('Galbraith Plot', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/06_heterogeneity_assessment.png', dpi=300, bbox_inches='tight')
print("  [6/9] Created: 06_heterogeneity_assessment.png")
plt.close()

# ============================================================================
# PLOT 7: Data Overview Panel
# ============================================================================
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Plot 1: Data table
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('tight')
ax1.axis('off')
table_data = []
table_data.append(['ID', 'y', 'sigma', 'Precision'])
for idx, row in data.iterrows():
    table_data.append([f"{row['obs_id']}", f"{row['y']}",
                       f"{row['sigma']}", f"{row['precision']:.4f}"])
table = ax1.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.15, 0.25, 0.25, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
for i in range(len(table_data)):
    if i == 0:
        for j in range(4):
            table[(i, j)].set_facecolor('#40466e')
            table[(i, j)].set_text_props(weight='bold', color='white')
ax1.set_title('Raw Data', fontsize=12, fontweight='bold', pad=10)

# Plot 2: y distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(data['y'], bins=6, edgecolor='black', alpha=0.7, color='steelblue')
ax2.axvline(data['y'].mean(), color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Observed Outcome (y)', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('Distribution of y', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: sigma distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(data['sigma'], bins=6, edgecolor='black', alpha=0.7, color='coral')
ax3.axvline(data['sigma'].mean(), color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Standard Error (sigma)', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title('Distribution of sigma', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: y vs sigma scatter
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(data['sigma'], data['y'], s=100, alpha=0.7, color='purple',
            edgecolors='black', linewidth=1.5)
for idx, row in data.iterrows():
    ax4.annotate(f"{row['obs_id']}", (row['sigma'], row['y']),
                fontsize=8, alpha=0.8, xytext=(3, 3), textcoords='offset points')
ax4.set_xlabel('Standard Error (sigma)', fontsize=10)
ax4.set_ylabel('Observed Outcome (y)', fontsize=10)
ax4.set_title(f'y vs sigma (r={corr_pearson:.3f})', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Forest plot mini version
ax5 = fig.add_subplot(gs[1, 1:])
data_sorted = data.sort_values('y').reset_index(drop=True)
for i, row in data_sorted.iterrows():
    ax5.errorbar(row['y'], i, xerr=row['sigma'], fmt='o',
                markersize=6, capsize=4, capthick=1.5,
                alpha=0.7, linewidth=1.5, color=f'C{i}')
ax5.axvline(weighted_mean, color='red', linestyle='--', linewidth=2, alpha=0.8)
ax5.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax5.set_xlabel('Outcome Value (y ± sigma)', fontsize=10)
ax5.set_ylabel('Observation', fontsize=10)
ax5.set_title('Forest Plot: All Observations', fontsize=12, fontweight='bold')
ax5.set_yticks(range(len(data_sorted)))
ax5.set_yticklabels([f"Obs {row['obs_id']}" for _, row in data_sorted.iterrows()], fontsize=8)
ax5.grid(True, alpha=0.3, axis='x')

plt.savefig('/workspace/eda/visualizations/07_data_overview.png', dpi=300, bbox_inches='tight')
print("  [7/9] Created: 07_data_overview.png")
plt.close()

# ============================================================================
# PLOT 8: Statistical Tests Visualization
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Cochran's Q test for heterogeneity
weighted_effects = data['y'] * data['precision']
Q = np.sum(data['precision'] * (data['y'] - weighted_mean)**2)
df = len(data) - 1
p_value_Q = 1 - stats.chi2.cdf(Q, df)
I2 = max(0, 100 * (Q - df) / Q)

# Plot 1: Q statistic visualization
ax = axes[0]
chi2_x = np.linspace(0, 20, 1000)
chi2_y = stats.chi2.pdf(chi2_x, df)
ax.plot(chi2_x, chi2_y, 'b-', linewidth=2, label=f'χ²({df})')
ax.axvline(Q, color='red', linestyle='--', linewidth=2,
           label=f'Q = {Q:.2f}\np = {p_value_Q:.4f}')
ax.fill_between(chi2_x[chi2_x >= Q], 0,
                stats.chi2.pdf(chi2_x[chi2_x >= Q], df),
                alpha=0.3, color='red')
ax.set_xlabel("Q statistic", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"Cochran's Q Test\nI² = {I2:.1f}%", fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Between-study variance estimation
ax = axes[1]
# DerSimonian-Laird estimator
tau2_DL = max(0, (Q - df) / (np.sum(data['precision']) -
                               np.sum(data['precision']**2) / np.sum(data['precision'])))
methods = ['Fixed\nEffect', 'DL Random\nEffect']
tau2_values = [0, tau2_DL]
colors_bar = ['lightblue', 'lightcoral']
bars = ax.bar(methods, tau2_values, color=colors_bar, alpha=0.7,
              edgecolor='black', linewidth=2)
ax.set_ylabel('τ² (Between-study variance)', fontsize=11)
ax.set_title('Heterogeneity Variance Estimates', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, tau2_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 3: Fixed vs Random effects comparison
ax = axes[2]
# Fixed effect estimate
fe_estimate = weighted_mean
fe_se = np.sqrt(1 / np.sum(data['precision']))

# Random effects estimate (DerSimonian-Laird)
re_weights = 1 / (data['sigma']**2 + tau2_DL)
re_estimate = np.sum(data['y'] * re_weights) / np.sum(re_weights)
re_se = np.sqrt(1 / np.sum(re_weights))

estimates = [fe_estimate, re_estimate]
se_values = [fe_se, re_se]
models = ['Fixed\nEffect', 'Random\nEffect (DL)']

x_pos_bar = range(len(models))
ax.errorbar(x_pos_bar, estimates, yerr=1.96*np.array(se_values),
            fmt='o', markersize=10, capsize=8, capthick=2,
            alpha=0.7, linewidth=2, color='darkgreen')
ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xticks(x_pos_bar)
ax.set_xticklabels(models)
ax.set_ylabel('Pooled Estimate ± 95% CI', fontsize=11)
ax.set_title('Fixed vs Random Effects Models', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add text with estimates
for i, (est, se) in enumerate(zip(estimates, se_values)):
    ax.text(i, est + 1.96*se + 2, f'{est:.2f}\n±{1.96*se:.2f}',
            ha='center', fontsize=9, bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/08_statistical_tests.png', dpi=300, bbox_inches='tight')
print("  [8/9] Created: 08_statistical_tests.png")
plt.close()

# ============================================================================
# PLOT 9: Sensitivity Analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Leave-one-out analysis
loo_estimates = []
loo_labels = []
for i in range(len(data)):
    data_loo = data.drop(i)
    loo_weights = 1 / data_loo['sigma']**2
    loo_est = np.sum(data_loo['y'] * loo_weights) / np.sum(loo_weights)
    loo_estimates.append(loo_est)
    loo_labels.append(f"Leave out\nObs {data.iloc[i]['obs_id']}")

# Plot 1: Leave-one-out estimates
ax = axes[0, 0]
y_pos = range(len(loo_estimates))
colors_loo = ['red' if abs(est - weighted_mean) > 2 else 'steelblue'
              for est in loo_estimates]
ax.barh(y_pos, loo_estimates, color=colors_loo, alpha=0.7, edgecolor='black', linewidth=1)
ax.axvline(weighted_mean, color='darkred', linestyle='--', linewidth=2,
           label=f'Full Model = {weighted_mean:.2f}')
ax.set_yticks(y_pos)
ax.set_yticklabels(loo_labels, fontsize=8)
ax.set_xlabel('Pooled Estimate', fontsize=11)
ax.set_title('Leave-One-Out Sensitivity Analysis', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Plot 2: Influence of each observation
ax = axes[0, 1]
influence = [abs(est - weighted_mean) for est in loo_estimates]
obs_ids = [f"Obs {data.iloc[i]['obs_id']}" for i in range(len(data))]
colors_inf = ['red' if inf > np.mean(influence) else 'steelblue' for inf in influence]
ax.bar(range(len(influence)), influence, color=colors_inf, alpha=0.7,
       edgecolor='black', linewidth=1)
ax.axhline(np.mean(influence), color='orange', linestyle='--', linewidth=2,
           label=f'Mean Influence = {np.mean(influence):.2f}')
ax.set_xticks(range(len(influence)))
ax.set_xticklabels(obs_ids, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Change in Estimate', fontsize=11)
ax.set_title('Influence of Each Observation', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Cumulative forest plot (adding observations sequentially)
ax = axes[1, 0]
data_sorted_by_precision = data.sort_values('precision', ascending=False).reset_index(drop=True)
cumulative_estimates = []
cumulative_ses = []

for i in range(1, len(data_sorted_by_precision) + 1):
    subset = data_sorted_by_precision.iloc[:i]
    cum_weights = 1 / subset['sigma']**2
    cum_est = np.sum(subset['y'] * cum_weights) / np.sum(cum_weights)
    cum_se = np.sqrt(1 / np.sum(cum_weights))
    cumulative_estimates.append(cum_est)
    cumulative_ses.append(cum_se)

n_obs = range(1, len(data) + 1)
ax.errorbar(n_obs, cumulative_estimates, yerr=1.96*np.array(cumulative_ses),
            fmt='o-', markersize=8, capsize=5, capthick=2,
            alpha=0.7, linewidth=2, color='purple')
ax.axhline(weighted_mean, color='red', linestyle='--', linewidth=2,
           label=f'Final Estimate = {weighted_mean:.2f}')
ax.set_xlabel('Number of Studies (ordered by precision)', fontsize=11)
ax.set_ylabel('Cumulative Estimate ± 95% CI', fontsize=11)
ax.set_title('Cumulative Meta-Analysis', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Precision vs residual
ax = axes[1, 1]
residuals_from_wm = data['y'] - weighted_mean
ax.scatter(data['precision'], residuals_from_wm, s=120, alpha=0.7,
           color='brown', edgecolors='black', linewidth=1.5)
for idx, row in data.iterrows():
    ax.annotate(f"  {row['obs_id']}",
                (row['precision'], residuals_from_wm[idx]),
                fontsize=9, alpha=0.8)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Precision (1/sigma²)', fontsize=11)
ax.set_ylabel('Residual from Weighted Mean', fontsize=11)
ax.set_title('Residuals vs Precision', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/09_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
print("  [9/9] Created: 09_sensitivity_analysis.png")
plt.close()

print("\nAll visualizations created successfully!")
print("\nKey Statistics Summary:")
print(f"  - Weighted mean: {weighted_mean:.3f}")
print(f"  - Cochran's Q: {Q:.3f} (p = {p_value_Q:.4f})")
print(f"  - I² statistic: {I2:.1f}%")
print(f"  - τ² (DL): {tau2_DL:.3f}")
print(f"  - Random effects estimate: {re_estimate:.3f} ± {1.96*re_se:.3f}")
