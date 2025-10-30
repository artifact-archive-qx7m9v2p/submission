"""
Comprehensive Visualizations - Analyst 2
Focus areas:
1. Sequential patterns across groups
2. Sample size vs proportion relationships
3. Uncertainty quantification
4. Rare events analysis
5. Pooling considerations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')

# Calculate confidence intervals (Wilson score interval for binomial proportions)
def wilson_ci(r, n, alpha=0.05):
    """Calculate Wilson score confidence interval"""
    z = stats.norm.ppf(1 - alpha/2)
    p_hat = r / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
    return center - margin, center + margin

# Add confidence intervals to data
data['ci_lower'], data['ci_upper'] = zip(*data.apply(lambda row: wilson_ci(row['r'], row['n']), axis=1))
data['ci_width'] = data['ci_upper'] - data['ci_lower']

# Calculate standard errors
data['se'] = np.sqrt(data['proportion'] * (1 - data['proportion']) / data['n'])

print("Starting visualization creation...")

# ============================================================================
# VISUALIZATION 1: Sequential Pattern Analysis (Multi-panel)
# ============================================================================
print("Creating sequential pattern analysis...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Sequential Pattern Analysis Across Groups', fontsize=14, fontweight='bold', y=0.995)

# Panel 1: Proportions with confidence intervals
ax = axes[0]
ax.errorbar(data['group'], data['proportion'],
            yerr=[data['proportion'] - data['ci_lower'], data['ci_upper'] - data['proportion']],
            fmt='o-', markersize=8, capsize=5, linewidth=2, color='#2E86AB')
ax.axhline(data['proportion'].mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
ax.axhline(data['r'].sum()/data['n'].sum(), color='orange', linestyle='--', alpha=0.7, label='Pooled')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Proportion', fontsize=11)
ax.set_title('Proportions with 95% Confidence Intervals', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(data['group'])

# Panel 2: Sample sizes
ax = axes[1]
bars = ax.bar(data['group'], data['n'], color='#A23B72', alpha=0.7, edgecolor='black')
# Highlight groups with small sample sizes
for i, (idx, row) in enumerate(data.iterrows()):
    if row['n'] < 100:
        bars[i].set_color('#F18F01')
        bars[i].set_alpha(1.0)
ax.axhline(data['n'].mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
ax.axhline(data['n'].median(), color='blue', linestyle='--', alpha=0.7, label='Median')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Sample Size (n)', fontsize=11)
ax.set_title('Sample Size Distribution (Orange: n < 100)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_xticks(data['group'])

# Panel 3: Confidence interval widths (uncertainty)
ax = axes[2]
ax.plot(data['group'], data['ci_width'], 'o-', markersize=8, linewidth=2, color='#C73E1D')
ax.fill_between(data['group'], 0, data['ci_width'], alpha=0.3, color='#C73E1D')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('CI Width', fontsize=11)
ax.set_title('Uncertainty (95% CI Width) by Group', fontsize=12)
ax.grid(alpha=0.3)
ax.set_xticks(data['group'])

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/01_sequential_patterns.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 2: Sample Size vs Proportion Relationship
# ============================================================================
print("Creating sample size vs proportion analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Sample Size vs Proportion Relationship Analysis', fontsize=14, fontweight='bold')

# Panel 1: Scatter with confidence intervals
ax = axes[0, 0]
for idx, row in data.iterrows():
    ax.plot([row['n'], row['n']], [row['ci_lower'], row['ci_upper']],
            color='gray', alpha=0.5, linewidth=2)
scatter = ax.scatter(data['n'], data['proportion'], s=data['r']*3,
                     c=data['group'], cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)
# Add trend line
z = np.polyfit(data['n'], data['proportion'], 1)
p = np.poly1d(z)
x_trend = np.linspace(data['n'].min(), data['n'].max(), 100)
ax.plot(x_trend, p(x_trend), "r--", alpha=0.7, linewidth=2, label=f'Trend: y={z[0]:.2e}x+{z[1]:.3f}')
ax.axhline(data['r'].sum()/data['n'].sum(), color='orange', linestyle='--', alpha=0.7, label='Pooled')
ax.set_xlabel('Sample Size (n)', fontsize=11)
ax.set_ylabel('Proportion', fontsize=11)
ax.set_title('Proportion vs Sample Size (bubble size = events)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Group', fontsize=10)

# Panel 2: Log scale version
ax = axes[0, 1]
for idx, row in data.iterrows():
    ax.plot([row['n'], row['n']], [row['ci_lower'], row['ci_upper']],
            color='gray', alpha=0.5, linewidth=2)
scatter = ax.scatter(data['n'], data['proportion'], s=data['r']*3,
                     c=data['group'], cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)
ax.axhline(data['r'].sum()/data['n'].sum(), color='orange', linestyle='--', alpha=0.7, label='Pooled')
ax.set_xlabel('Sample Size (n)', fontsize=11)
ax.set_ylabel('Proportion', fontsize=11)
ax.set_title('Proportion vs Sample Size (Log Scale)', fontsize=12)
ax.set_xscale('log')
ax.legend()
ax.grid(alpha=0.3, which='both')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Group', fontsize=10)

# Panel 3: Absolute uncertainty vs sample size
ax = axes[1, 0]
scatter = ax.scatter(data['n'], data['ci_width'], s=100, c=data['proportion'],
                     cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=1)
# Theoretical relationship
n_theory = np.linspace(data['n'].min(), data['n'].max(), 100)
pooled_p = data['r'].sum()/data['n'].sum()
# Approximate CI width for moderate proportions
ci_theory = 2 * 1.96 * np.sqrt(pooled_p * (1-pooled_p) / n_theory)
ax.plot(n_theory, ci_theory, 'r--', linewidth=2, alpha=0.7, label='Theoretical (pooled p)')
ax.set_xlabel('Sample Size (n)', fontsize=11)
ax.set_ylabel('CI Width', fontsize=11)
ax.set_title('Uncertainty vs Sample Size', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Proportion', fontsize=10)

# Panel 4: Relative uncertainty (CV) vs sample size
ax = axes[1, 1]
# Calculate CV for non-zero proportions
data_nonzero = data[data['r'] > 0].copy()
data_nonzero['cv'] = data_nonzero['se'] / data_nonzero['proportion']
scatter = ax.scatter(data_nonzero['n'], data_nonzero['cv'], s=100,
                     c=data_nonzero['proportion'], cmap='RdYlBu_r',
                     alpha=0.7, edgecolors='black', linewidth=1)
ax.set_xlabel('Sample Size (n)', fontsize=11)
ax.set_ylabel('Coefficient of Variation (SE/p)', fontsize=11)
ax.set_title('Relative Uncertainty vs Sample Size (excluding zero events)', fontsize=12)
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Proportion', fontsize=10)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/02_sample_size_relationships.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 3: Uncertainty Quantification Details
# ============================================================================
print("Creating uncertainty quantification analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Uncertainty Quantification and Confidence Intervals', fontsize=14, fontweight='bold')

# Panel 1: Forest plot style
ax = axes[0, 0]
y_pos = np.arange(len(data))
colors = ['red' if r == 0 else 'steelblue' for r in data['r']]
for i, (idx, row) in enumerate(data.iterrows()):
    ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 'o-',
            linewidth=2, markersize=8, color=colors[i])
ax.scatter(data['proportion'], y_pos, s=100, c=colors, edgecolors='black', linewidth=1.5, zorder=3)
ax.axvline(data['r'].sum()/data['n'].sum(), color='orange', linestyle='--',
           linewidth=2, alpha=0.7, label='Pooled estimate')
ax.set_yticks(y_pos)
ax.set_yticklabels([f"Group {g} (n={n})" for g, n in zip(data['group'], data['n'])])
ax.set_xlabel('Proportion', fontsize=11)
ax.set_title('Forest Plot: All Groups with 95% CIs', fontsize=12)
ax.grid(alpha=0.3, axis='x')
ax.legend()

# Panel 2: Precision (inverse variance) by group
ax = axes[0, 1]
data['precision'] = 1 / (data['se']**2)
data_nonzero = data[data['r'] > 0]
bars = ax.bar(data_nonzero['group'], data_nonzero['precision'],
              color='#2E86AB', alpha=0.7, edgecolor='black')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Precision (1/SE²)', fontsize=11)
ax.set_title('Statistical Precision by Group (excluding zero events)', fontsize=12)
ax.grid(alpha=0.3, axis='y')
ax.set_xticks(data_nonzero['group'])

# Panel 3: Observed vs Expected variation
ax = axes[1, 0]
pooled_p = data['r'].sum()/data['n'].sum()
data['expected_se'] = np.sqrt(pooled_p * (1-pooled_p) / data['n'])
ax.scatter(data['expected_se'], data['se'], s=data['n']/5, alpha=0.7,
          c=data['proportion'], cmap='RdYlBu_r', edgecolors='black', linewidth=1)
# Add diagonal line
max_se = max(data['expected_se'].max(), data['se'].max())
ax.plot([0, max_se], [0, max_se], 'k--', alpha=0.5, label='Expected = Observed')
ax.set_xlabel('Expected SE (under pooled p)', fontsize=11)
ax.set_ylabel('Observed SE', fontsize=11)
ax.set_title('Observed vs Expected Standard Errors (bubble size = n)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Panel 4: Coverage probability check
ax = axes[1, 1]
# Simulate what proportion should be if all groups same
# Calculate z-scores
data_nonzero = data[data['r'] > 0].copy()
pooled_p = data['r'].sum()/data['n'].sum()
data_nonzero['z_score'] = (data_nonzero['proportion'] - pooled_p) / data_nonzero['expected_se']
ax.hist(data_nonzero['z_score'], bins=10, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Expected mean')
ax.axvline(-1.96, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='95% CI bounds')
ax.axvline(1.96, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Z-score (from pooled estimate)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Z-scores (excluding zero events)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/03_uncertainty_quantification.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 4: Rare Events and Zero Inflation Analysis
# ============================================================================
print("Creating rare events analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Rare Events and Zero Inflation Analysis', fontsize=14, fontweight='bold')

# Panel 1: Event counts
ax = axes[0, 0]
colors_events = ['red' if r == 0 else '#2E86AB' if r <= 5 else '#4CAF50' for r in data['r']]
bars = ax.bar(data['group'], data['r'], color=colors_events, alpha=0.7, edgecolor='black')
ax.axhline(5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Rare event threshold (5)')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Number of Events (r)', fontsize=11)
ax.set_title('Event Counts by Group (Red: zero, Blue: rare ≤5)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_xticks(data['group'])

# Panel 2: Expected events under pooled model
ax = axes[0, 1]
pooled_p = data['r'].sum()/data['n'].sum()
data['expected_r'] = data['n'] * pooled_p
ax.scatter(data['expected_r'], data['r'], s=150, alpha=0.7,
          c=data['group'], cmap='viridis', edgecolors='black', linewidth=1)
# Add diagonal line
max_r = max(data['expected_r'].max(), data['r'].max())
ax.plot([0, max_r], [0, max_r], 'k--', alpha=0.5, label='Expected = Observed')
# Add labels for outliers
for idx, row in data.iterrows():
    if abs(row['r'] - row['expected_r']) > 5:
        ax.annotate(f"G{row['group']}",
                   (row['expected_r'], row['r']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.set_xlabel('Expected Events (under pooled p)', fontsize=11)
ax.set_ylabel('Observed Events', fontsize=11)
ax.set_title('Observed vs Expected Event Counts', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Panel 3: Probability of observing zero under pooled model
ax = axes[1, 0]
data['p_zero_pooled'] = (1 - pooled_p) ** data['n']
colors_zero = ['red' if r == 0 else 'steelblue' for r in data['r']]
bars = ax.bar(data['group'], data['p_zero_pooled'], color=colors_zero, alpha=0.7, edgecolor='black')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('P(r=0 | pooled p)', fontsize=11)
ax.set_title('Probability of Zero Events Under Pooled Model (Red: observed zero)', fontsize=12)
ax.set_yscale('log')
ax.grid(alpha=0.3, which='both', axis='y')
ax.set_xticks(data['group'])

# Panel 4: Residual analysis
ax = axes[1, 1]
data['residual'] = data['r'] - data['expected_r']
data['std_residual'] = data['residual'] / np.sqrt(data['expected_r'] * (1 - pooled_p))
colors_res = ['red' if abs(sr) > 2 else 'steelblue' for sr in data['std_residual']]
ax.scatter(data['expected_r'], data['std_residual'], s=data['n']/3,
          c=colors_res, alpha=0.7, edgecolors='black', linewidth=1)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='±2 SD')
ax.axhline(-2, color='red', linestyle='--', linewidth=2, alpha=0.7)
# Label outliers
for idx, row in data.iterrows():
    if abs(row['std_residual']) > 2:
        ax.annotate(f"G{row['group']}",
                   (row['expected_r'], row['std_residual']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.set_xlabel('Expected Events', fontsize=11)
ax.set_ylabel('Standardized Residual', fontsize=11)
ax.set_title('Standardized Residuals (Red: outliers, bubble size = n)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/04_rare_events_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# VISUALIZATION 5: Pooling Considerations
# ============================================================================
print("Creating pooling considerations analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Pooling Considerations: Complete vs No Pooling vs Partial Pooling', fontsize=14, fontweight='bold')

# Panel 1: Complete pooling vs no pooling estimates
ax = axes[0, 0]
pooled_estimate = data['r'].sum() / data['n'].sum()
x_pos = np.arange(len(data))
width = 0.35
bars1 = ax.bar(x_pos - width/2, data['proportion'], width, label='No pooling (observed)',
              alpha=0.7, color='steelblue', edgecolor='black')
bars2 = ax.bar(x_pos + width/2, [pooled_estimate]*len(data), width, label='Complete pooling',
              alpha=0.7, color='orange', edgecolor='black')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Proportion', fontsize=11)
ax.set_title('No Pooling vs Complete Pooling Estimates', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(data['group'])
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Panel 2: Shrinkage illustration (simple empirical Bayes)
ax = axes[0, 1]
# Simple shrinkage: weight by precision
total_precision = data['precision'].sum()
data['weight'] = data['precision'] / total_precision
# Empirical Bayes estimate (simplified)
grand_mean = np.average(data['proportion'], weights=data['n'])
# Shrinkage factor based on within vs between group variance
data_nonzero = data[data['r'] > 0].copy()
within_var = np.average(data_nonzero['se']**2, weights=data_nonzero['n'])
between_var = max(0, np.var(data_nonzero['proportion']) - within_var/np.mean(data_nonzero['n']))
for idx, row in data.iterrows():
    shrinkage = within_var / (within_var + between_var * row['n'])
    shrunk_estimate = shrinkage * grand_mean + (1 - shrinkage) * row['proportion']
    ax.plot([row['group'], row['group']], [row['proportion'], shrunk_estimate],
           'o-', linewidth=2, markersize=8, color='gray', alpha=0.5)
    ax.scatter(row['group'], row['proportion'], s=100, color='steelblue',
              edgecolors='black', linewidth=1.5, zorder=3, label='No pooling' if idx == 0 else '')
    ax.scatter(row['group'], shrunk_estimate, s=100, color='green',
              edgecolors='black', linewidth=1.5, zorder=3, label='Partial pooling' if idx == 0 else '')
ax.axhline(grand_mean, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Complete pooling')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Proportion', fontsize=11)
ax.set_title('Shrinkage Effect: Partial Pooling (Empirical Bayes)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(data['group'])

# Panel 3: Between-group variance analysis
ax = axes[1, 0]
# Calculate group-level statistics for different models
models = ['No Pooling\n(12 params)', 'Complete Pooling\n(1 param)', 'Partial Pooling\n(~2-3 params)']
# Variance in estimates
no_pool_var = data['proportion'].var()
complete_pool_var = 0  # All same estimate
# For partial pooling, use the shrunk estimates
shrunk_estimates = []
for idx, row in data.iterrows():
    shrinkage = within_var / (within_var + between_var * row['n'])
    shrunk_estimate = shrinkage * grand_mean + (1 - shrinkage) * row['proportion']
    shrunk_estimates.append(shrunk_estimate)
partial_pool_var = np.var(shrunk_estimates)

variances = [no_pool_var, complete_pool_var, partial_pool_var]
colors_model = ['steelblue', 'orange', 'green']
bars = ax.bar(models, variances, color=colors_model, alpha=0.7, edgecolor='black')
ax.set_ylabel('Variance in Estimates', fontsize=11)
ax.set_title('Between-Group Variance by Pooling Strategy', fontsize=12)
ax.grid(alpha=0.3, axis='y')

# Panel 4: Effective sample size under different pooling
ax = axes[1, 1]
# No pooling: use actual sample size
# Complete pooling: use total sample size
# Partial pooling: somewhere in between based on shrinkage
effective_n_complete = [data['n'].sum()] * len(data)
effective_n_partial = []
for idx, row in data.iterrows():
    shrinkage = within_var / (within_var + between_var * row['n'])
    eff_n = row['n'] + shrinkage * (data['n'].sum() - row['n'])
    effective_n_partial.append(eff_n)

x_pos = np.arange(len(data))
width = 0.25
ax.bar(x_pos - width, data['n'], width, label='No pooling', alpha=0.7, color='steelblue', edgecolor='black')
ax.bar(x_pos, effective_n_partial, width, label='Partial pooling', alpha=0.7, color='green', edgecolor='black')
ax.bar(x_pos + width, [data['n'].sum()]*len(data), width, label='Complete pooling', alpha=0.7, color='orange', edgecolor='black')
ax.set_xlabel('Group', fontsize=11)
ax.set_ylabel('Effective Sample Size', fontsize=11)
ax.set_title('Effective Sample Size by Pooling Strategy', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels(data['group'])
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_2/visualizations/05_pooling_considerations.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll visualizations created successfully!")
print(f"Saved to: /workspace/eda/analyst_2/visualizations/")
