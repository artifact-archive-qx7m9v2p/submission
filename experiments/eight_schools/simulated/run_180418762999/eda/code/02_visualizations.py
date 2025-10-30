"""
Comprehensive Visualizations
=============================
Goal: Create visualizations to understand distributions, relationships, and patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/eda/code/data_with_metrics.csv')

# ============================================================================
# 1. OVERVIEW PANEL: Key distributions and relationships
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Hierarchical Data: Overview of Distributions and Relationships', fontsize=14, fontweight='bold')

# 1a. Distribution of y values
ax = axes[0, 0]
ax.hist(data['y'], bins=6, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(data['y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data["y"].mean():.2f}')
ax.axvline(data['y'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {data["y"].median():.2f}')
ax.set_xlabel('Response Variable (y)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Response Variable (y)')
ax.legend()
ax.grid(alpha=0.3)

# 1b. Distribution of sigma values
ax = axes[0, 1]
ax.hist(data['sigma'], bins=6, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(data['sigma'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data["sigma"].mean():.2f}')
ax.axvline(data['sigma'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {data["sigma"].median():.2f}')
ax.set_xlabel('Measurement Error (sigma)', fontweight='bold')
ax.set_ylabel('Frequency', fontweight='bold')
ax.set_title('Distribution of Measurement Error')
ax.legend()
ax.grid(alpha=0.3)

# 1c. Scatter plot: y vs sigma
ax = axes[0, 2]
ax.scatter(data['sigma'], data['y'], s=100, alpha=0.7, color='purple', edgecolor='black')
for i, row in data.iterrows():
    ax.annotate(f"G{int(row['group'])}", (row['sigma'], row['y']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
# Add correlation line
z = np.polyfit(data['sigma'], data['y'], 1)
p = np.poly1d(z)
ax.plot(data['sigma'], p(data['sigma']), "r--", alpha=0.8, linewidth=2)
corr = data['y'].corr(data['sigma'])
ax.set_xlabel('Measurement Error (sigma)', fontweight='bold')
ax.set_ylabel('Response Variable (y)', fontweight='bold')
ax.set_title(f'y vs sigma (r = {corr:.3f})')
ax.grid(alpha=0.3)

# 1d. Box plots comparison
ax = axes[1, 0]
box_data = [data['y'], data['sigma']]
bp = ax.boxplot(box_data, labels=['y', 'sigma'], patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
for patch in bp['boxes']:
    patch.set_alpha(0.7)
ax.set_ylabel('Value', fontweight='bold')
ax.set_title('Distributions Comparison')
ax.grid(alpha=0.3, axis='y')

# 1e. Signal-to-noise ratio by group
ax = axes[1, 1]
colors = ['green' if snr >= 1 else 'red' for snr in data['snr']]
bars = ax.bar(data['group'], data['snr'], color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=1, color='black', linestyle='--', linewidth=2, label='SNR = 1')
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Signal-to-Noise Ratio', fontweight='bold')
ax.set_title('Signal-to-Noise Ratio by Group')
ax.set_xticks(data['group'])
ax.legend()
ax.grid(alpha=0.3, axis='y')

# 1f. Relative error by group
ax = axes[1, 2]
bars = ax.bar(data['group'], data['rel_error'], color='orange', alpha=0.7, edgecolor='black')
ax.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Rel. Error = 1')
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Relative Error (sigma/|y|)', fontweight='bold')
ax.set_title('Relative Measurement Error by Group')
ax.set_xticks(data['group'])
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/01_overview_panel.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/visualizations/01_overview_panel.png")
plt.close()

# ============================================================================
# 2. DETAILED Y DISTRIBUTION ANALYSIS
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Detailed Analysis of Response Variable (y)', fontsize=14, fontweight='bold')

# 2a. Histogram with kernel density
ax = axes[0]
ax.hist(data['y'], bins=6, alpha=0.5, color='steelblue', edgecolor='black', density=True, label='Histogram')
# KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(data['y'])
x_range = np.linspace(data['y'].min() - 5, data['y'].max() + 5, 100)
ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
ax.axvline(data['y'].mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {data["y"].mean():.2f}')
ax.set_xlabel('Response Variable (y)', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('Distribution with KDE')
ax.legend()
ax.grid(alpha=0.3)

# 2b. Q-Q plot
ax = axes[1]
stats.probplot(data['y'], dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Testing Normality')
ax.grid(alpha=0.3)

# 2c. ECDF
ax = axes[2]
sorted_y = np.sort(data['y'])
ecdf = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
ax.step(sorted_y, ecdf, where='post', linewidth=2, color='steelblue')
ax.scatter(sorted_y, ecdf, s=50, color='steelblue', edgecolor='black', zorder=3)
ax.set_xlabel('Response Variable (y)', fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontweight='bold')
ax.set_title('Empirical CDF')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/02_y_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/visualizations/02_y_distribution_analysis.png")
plt.close()

# ============================================================================
# 3. GROUP-LEVEL VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Group-Level Analysis with Uncertainty', fontsize=14, fontweight='bold')

# 3a. Point estimates with error bars
ax = axes[0, 0]
ax.errorbar(data['group'], data['y'], yerr=data['sigma'],
            fmt='o', markersize=10, capsize=5, capthick=2,
            color='steelblue', ecolor='red', elinewidth=2, alpha=0.7)
ax.axhline(y=data['y'].mean(), color='green', linestyle='--', linewidth=2, label=f'Overall mean: {data["y"].mean():.2f}')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Response Variable (y)', fontweight='bold')
ax.set_title('Observed Values with Measurement Error (±1 sigma)')
ax.set_xticks(data['group'])
ax.legend()
ax.grid(alpha=0.3)

# 3b. Confidence intervals (±2 sigma)
ax = axes[0, 1]
ax.errorbar(data['group'], data['y'], yerr=2*data['sigma'],
            fmt='s', markersize=10, capsize=5, capthick=2,
            color='purple', ecolor='orange', elinewidth=2, alpha=0.7)
ax.axhline(y=data['y'].mean(), color='green', linestyle='--', linewidth=2, label=f'Overall mean: {data["y"].mean():.2f}')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Response Variable (y)', fontweight='bold')
ax.set_title('Observed Values with 95% CI (±2 sigma)')
ax.set_xticks(data['group'])
ax.legend()
ax.grid(alpha=0.3)

# 3c. Standardized values (y/sigma)
ax = axes[1, 0]
standardized = data['y'] / data['sigma']
colors = ['green' if abs(z) > 1 else 'red' for z in standardized]
bars = ax.bar(data['group'], standardized, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Standardized Value (y/sigma)', fontweight='bold')
ax.set_title('Standardized Effect Sizes')
ax.set_xticks(data['group'])
ax.grid(alpha=0.3, axis='y')

# 3d. Observed value and measurement error together
ax = axes[1, 1]
x = np.arange(len(data))
width = 0.35
bars1 = ax.bar(x - width/2, data['y'], width, label='Observed y', color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, data['sigma'], width, label='Measurement error (sigma)', color='coral', alpha=0.7, edgecolor='black')
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Value', fontweight='bold')
ax.set_title('Observed Values vs Measurement Errors')
ax.set_xticks(x)
ax.set_xticklabels(data['group'])
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/03_group_level_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/visualizations/03_group_level_analysis.png")
plt.close()

# ============================================================================
# 4. UNCERTAINTY VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Measurement Uncertainty Patterns', fontsize=14, fontweight='bold')

# 4a. Uncertainty ranges visualization
ax = axes[0]
# Sort by y value for better visualization
data_sorted = data.sort_values('y').reset_index(drop=True)
for i, row in data_sorted.iterrows():
    # Draw the range
    ax.plot([i, i], [row['y'] - row['sigma'], row['y'] + row['sigma']],
            'o-', linewidth=8, alpha=0.4, color='steelblue')
    # Draw the point estimate
    ax.plot(i, row['y'], 'o', markersize=10, color='red', zorder=3)
    # Add group label
    ax.text(i, row['y'] + row['sigma'] + 1, f"G{int(row['group'])}",
            ha='center', fontsize=9, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Groups (sorted by y value)', fontweight='bold')
ax.set_ylabel('Value Range', fontweight='bold')
ax.set_title('Uncertainty Ranges (y ± sigma) - Sorted by y')
ax.set_xticks(range(len(data_sorted)))
ax.set_xticklabels([f"G{int(g)}" for g in data_sorted['group']])
ax.grid(alpha=0.3)

# 4b. Scatter with proportional error bars
ax = axes[1]
# Create a plot showing the relationship between magnitude and uncertainty
scatter = ax.scatter(np.abs(data['y']), data['sigma'],
                     s=200, alpha=0.6, c=data['snr'],
                     cmap='RdYlGn', edgecolor='black', linewidth=2)
for i, row in data.iterrows():
    ax.annotate(f"G{int(row['group'])}",
                (np.abs(row['y']), row['sigma']),
                xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
ax.set_xlabel('Absolute Value of y', fontweight='bold')
ax.set_ylabel('Measurement Error (sigma)', fontweight='bold')
ax.set_title('Magnitude vs Uncertainty (colored by SNR)')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Signal-to-Noise Ratio', fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/04_uncertainty_patterns.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/visualizations/04_uncertainty_patterns.png")
plt.close()

# ============================================================================
# 5. STATISTICAL TESTS AND DIAGNOSTICS
# ============================================================================
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
fig.suptitle('Statistical Tests and Diagnostic Plots', fontsize=14, fontweight='bold')

# 5a. Normality test for y
ax = fig.add_subplot(gs[0, 0])
shapiro_stat, shapiro_p = stats.shapiro(data['y'])
ax.text(0.5, 0.7, 'Shapiro-Wilk Test for y', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.5, f'Statistic: {shapiro_stat:.4f}', ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.3, f'P-value: {shapiro_p:.4f}', ha='center', fontsize=11, transform=ax.transAxes)
interpretation = "Normal" if shapiro_p > 0.05 else "Not Normal"
color = 'green' if shapiro_p > 0.05 else 'red'
ax.text(0.5, 0.1, f'Interpretation: {interpretation}', ha='center', fontsize=11,
        fontweight='bold', color=color, transform=ax.transAxes)
ax.axis('off')

# 5b. Anderson-Darling test
ax = fig.add_subplot(gs[0, 1])
anderson_result = stats.anderson(data['y'], dist='norm')
ax.text(0.5, 0.8, 'Anderson-Darling Test for y', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.6, f'Statistic: {anderson_result.statistic:.4f}', ha='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.4, 'Critical values:', ha='center', fontsize=10, transform=ax.transAxes)
for i, (sig, crit) in enumerate(zip(anderson_result.significance_level, anderson_result.critical_values)):
    y_pos = 0.3 - i*0.08
    ax.text(0.5, y_pos, f'{sig}%: {crit:.3f}', ha='center', fontsize=9, transform=ax.transAxes)
ax.axis('off')

# 5c. Kernel Density Estimate comparison with Normal
ax = fig.add_subplot(gs[1, :])
ax.hist(data['y'], bins=6, alpha=0.4, color='steelblue', edgecolor='black', density=True, label='Data histogram')
kde = gaussian_kde(data['y'])
x_range = np.linspace(data['y'].min() - 10, data['y'].max() + 10, 200)
ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='Data KDE')
# Overlay normal distribution
norm_dist = stats.norm(loc=data['y'].mean(), scale=data['y'].std())
ax.plot(x_range, norm_dist.pdf(x_range), 'r--', linewidth=2, label='Normal distribution')
ax.set_xlabel('Response Variable (y)', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('Observed Distribution vs Normal Distribution')
ax.legend()
ax.grid(alpha=0.3)

# 5d. Residuals from mean
ax = fig.add_subplot(gs[2, 0])
residuals = data['y'] - data['y'].mean()
ax.scatter(data['group'], residuals, s=100, alpha=0.7, color='purple', edgecolor='black')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
# Add error bars
ax.errorbar(data['group'], residuals, yerr=data['sigma'],
            fmt='none', ecolor='gray', elinewidth=1, capsize=3, alpha=0.5)
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Residual from Overall Mean', fontweight='bold')
ax.set_title('Residuals with Measurement Uncertainty')
ax.set_xticks(data['group'])
ax.grid(alpha=0.3)

# 5e. Standardized residuals
ax = fig.add_subplot(gs[2, 1])
std_residuals = residuals / data['sigma']
colors = ['green' if abs(sr) > 1.96 else 'blue' for sr in std_residuals]
ax.scatter(data['group'], std_residuals, s=100, alpha=0.7, c=colors, edgecolor='black')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.axhline(y=1.96, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='±1.96 (95% CI)')
ax.axhline(y=-1.96, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Standardized Residual', fontweight='bold')
ax.set_title('Standardized Residuals (residual/sigma)')
ax.set_xticks(data['group'])
ax.legend()
ax.grid(alpha=0.3)

plt.savefig('/workspace/eda/visualizations/05_statistical_diagnostics.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/visualizations/05_statistical_diagnostics.png")
plt.close()

print("\nAll visualizations completed successfully!")
