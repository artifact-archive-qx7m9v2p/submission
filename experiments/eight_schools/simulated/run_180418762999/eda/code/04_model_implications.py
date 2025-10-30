"""
Model Implications and Comparison
==================================
Goal: Visualize different modeling approaches and their implications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")

# Load data
data = pd.read_csv('/workspace/eda/code/data_with_metrics.csv')

# ============================================================================
# VISUALIZATION 1: Three Modeling Approaches
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Comparison of Three Modeling Approaches', fontsize=14, fontweight='bold')

# Calculate different estimates
# 1. Complete pooling: single mean for all groups
complete_pool_mean = data['y'].mean()
complete_pool_se = data['y'].std() / np.sqrt(len(data))

# 2. No pooling: each group separate
no_pool_means = data['y'].values
no_pool_se = data['sigma'].values

# 3. Weighted pooling (approximation of hierarchical)
weights = 1 / (data['sigma']**2)
weighted_mean = np.sum(data['y'] * weights) / np.sum(weights)
weighted_se = np.sqrt(1 / np.sum(weights))

# Plot 1: Complete Pooling
ax = axes[0]
ax.errorbar(data['group'], [complete_pool_mean]*len(data),
            yerr=[complete_pool_se]*len(data),
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='blue', ecolor='lightblue', elinewidth=2, alpha=0.7,
            label='Complete pooling estimate')
ax.scatter(data['group'], data['y'], s=100, alpha=0.5, color='red',
           edgecolor='black', zorder=5, label='Observed data')
ax.axhline(y=complete_pool_mean, color='blue', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Value', fontweight='bold')
ax.set_title('Complete Pooling\n(All groups share same mean)', fontsize=12)
ax.set_xticks(data['group'])
ax.legend(loc='best')
ax.grid(alpha=0.3)

# Plot 2: No Pooling
ax = axes[1]
ax.errorbar(data['group'], no_pool_means, yerr=no_pool_se,
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='green', ecolor='lightgreen', elinewidth=2, alpha=0.7,
            label='No pooling estimates')
ax.scatter(data['group'], data['y'], s=100, alpha=0.5, color='red',
           edgecolor='black', zorder=5, label='Observed data')
ax.axhline(y=data['y'].mean(), color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Value', fontweight='bold')
ax.set_title('No Pooling\n(Each group independent)', fontsize=12)
ax.set_xticks(data['group'])
ax.legend(loc='best')
ax.grid(alpha=0.3)

# Plot 3: Partial Pooling (shrinkage toward weighted mean)
ax = axes[2]
# Simple shrinkage estimate: weighted average of group mean and global mean
shrinkage_factor = 0.5  # simplified
partial_pool_means = shrinkage_factor * data['y'] + (1 - shrinkage_factor) * weighted_mean
ax.errorbar(data['group'], partial_pool_means, yerr=data['sigma']*0.7,
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='purple', ecolor='plum', elinewidth=2, alpha=0.7,
            label='Partial pooling estimates')
ax.scatter(data['group'], data['y'], s=100, alpha=0.5, color='red',
           edgecolor='black', zorder=5, label='Observed data')
ax.axhline(y=weighted_mean, color='purple', linestyle='--', linewidth=2, alpha=0.5)
# Draw shrinkage arrows
for i in range(len(data)):
    ax.annotate('', xy=(data['group'].iloc[i], partial_pool_means[i]),
                xytext=(data['group'].iloc[i], data['y'].iloc[i]),
                arrowprops=dict(arrowstyle='->', color='orange', lw=1.5, alpha=0.6))
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Value', fontweight='bold')
ax.set_title('Partial Pooling\n(Hierarchical shrinkage)', fontsize=12)
ax.set_xticks(data['group'])
ax.legend(loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/06_model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/visualizations/06_model_comparison.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Prior Implications
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prior Distribution Implications', fontsize=14, fontweight='bold')

# Based on data analysis, suggest reasonable priors
# For the population mean (mu)
mu_mean = weighted_mean
mu_sd = max(data['y'].std(), weighted_se * 2)  # at least 2x the SE

# For the population std (tau) - between-group variation
tau_observed = 0  # from variance decomposition
tau_upper = data['y'].std()  # upper bound

# Plot 1: Prior for population mean
ax = axes[0, 0]
x_mu = np.linspace(mu_mean - 3*mu_sd, mu_mean + 3*mu_sd, 200)
# Weakly informative prior
weak_prior = stats.norm(mu_mean, mu_sd * 2)
ax.plot(x_mu, weak_prior.pdf(x_mu), 'b-', linewidth=2, label=f'Weakly informative: N({mu_mean:.1f}, {mu_sd*2:.1f})')
# Informative prior
info_prior = stats.norm(mu_mean, mu_sd)
ax.plot(x_mu, info_prior.pdf(x_mu), 'r-', linewidth=2, label=f'Informative: N({mu_mean:.1f}, {mu_sd:.1f})')
# Mark observed data
ax.axvline(data['y'].mean(), color='green', linestyle='--', linewidth=2, label=f'Observed mean: {data["y"].mean():.1f}')
ax.axvline(weighted_mean, color='orange', linestyle='--', linewidth=2, label=f'Weighted mean: {weighted_mean:.1f}')
ax.set_xlabel('Population Mean (mu)', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('Prior Options for Population Mean')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Prior for population std (tau)
ax = axes[0, 1]
x_tau = np.linspace(0, tau_upper * 2, 200)
# Half-Cauchy prior (common for hierarchical std)
scale_param = tau_upper / 2
half_cauchy = stats.halfcauchy(scale=scale_param)
ax.plot(x_tau, half_cauchy.pdf(x_tau), 'b-', linewidth=2, label=f'Half-Cauchy(0, {scale_param:.1f})')
# Half-Normal prior
half_normal = stats.halfnorm(scale=tau_upper)
ax.plot(x_tau, half_normal.pdf(x_tau), 'r-', linewidth=2, label=f'Half-Normal(0, {tau_upper:.1f})')
# Exponential prior
exp_prior = stats.expon(scale=tau_upper/2)
ax.plot(x_tau, exp_prior.pdf(x_tau), 'g-', linewidth=2, label=f'Exponential({2/tau_upper:.2f})')
ax.set_xlabel('Between-Group Std (tau)', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('Prior Options for Between-Group Variation')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Posterior predictive simulation - complete pooling
ax = axes[1, 0]
np.random.seed(42)
n_samples = 1000
# Simulate from complete pooling model
post_samples = np.random.normal(complete_pool_mean, complete_pool_se, n_samples)
ax.hist(post_samples, bins=30, alpha=0.6, color='steelblue', density=True, edgecolor='black')
ax.axvline(data['y'].mean(), color='red', linestyle='--', linewidth=2, label='Observed mean')
# Overlay observed data points
for y_val in data['y']:
    ax.axvline(y_val, color='green', alpha=0.3, linewidth=1)
ax.set_xlabel('Value', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('Posterior Predictive: Complete Pooling\n(green lines = observed data)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Prior sensitivity illustration
ax = axes[1, 1]
# Show how different prior widths affect uncertainty
prior_scales = [5, 10, 20, 40]
colors = ['red', 'orange', 'blue', 'green']
x_range = np.linspace(-20, 40, 200)
for scale, color in zip(prior_scales, colors):
    prior = stats.norm(mu_mean, scale)
    ax.plot(x_range, prior.pdf(x_range), color=color, linewidth=2,
            label=f'SD = {scale}', alpha=0.7)
# Mark data range
ax.axvspan(data['y'].min(), data['y'].max(), alpha=0.2, color='gray', label='Data range')
ax.set_xlabel('Population Mean (mu)', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('Prior Sensitivity: Effect of Prior Width')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/07_prior_implications.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/visualizations/07_prior_implications.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Measurement Error Impact
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Impact of Measurement Error on Inference', fontsize=14, fontweight='bold')

# Plot 1: Confidence intervals ignoring vs accounting for measurement error
ax = axes[0, 0]
# Naive CI (ignoring measurement error)
naive_ci = 1.96 * data['y'].std() / np.sqrt(len(data))
# Proper CI (accounting for measurement error)
proper_se = np.sqrt((data['y'].std()**2 / len(data)) + (data['sigma'].mean()**2 / len(data)))
proper_ci = 1.96 * proper_se

x_pos = [0, 1]
means = [data['y'].mean(), data['y'].mean()]
cis = [naive_ci, proper_ci]
labels = ['Naive\n(ignore sigma)', 'Proper\n(account for sigma)']
colors = ['red', 'green']

for i, (mean, ci, label, color) in enumerate(zip(means, cis, labels, colors)):
    ax.errorbar(x_pos[i], mean, yerr=ci, fmt='o', markersize=12, capsize=8, capthick=3,
                color=color, ecolor=color, elinewidth=3, alpha=0.7)
    ax.text(x_pos[i], mean + ci + 2, f'±{ci:.2f}', ha='center', fontweight='bold')

ax.set_xlim(-0.5, 1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Population Mean Estimate', fontweight='bold')
ax.set_title('95% Confidence Intervals')
ax.axhline(data['y'].mean(), color='blue', linestyle='--', linewidth=2, alpha=0.5)
ax.grid(alpha=0.3, axis='y')

# Plot 2: Effective sample size
ax = axes[0, 1]
# Calculate effective n for each observation based on sigma
# Lower sigma = more information
effective_weights = 1 / (data['sigma']**2)
normalized_weights = effective_weights / effective_weights.sum() * len(data)

bars = ax.bar(data['group'], normalized_weights, color='steelblue', alpha=0.7, edgecolor='black')
ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Equal weight')
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Effective Weight (relative to equal weighting)', fontweight='bold')
ax.set_title('Information Content by Group\n(inversely proportional to sigma^2)')
ax.set_xticks(data['group'])
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: Posterior width comparison
ax = axes[1, 0]
# For each group, compare posterior width with and without measurement error
# Without measurement error: posterior ~ N(y, sigma_post)
# With measurement error: posterior ~ N(y, sqrt(sigma_post^2 + sigma^2))

# Assuming a vague prior, posterior is approximately Normal(y, sigma)
post_widths_proper = data['sigma'].values
# If we ignored measurement error, we'd use a fixed small value
post_widths_naive = np.ones(len(data)) * data['sigma'].min() * 0.5

x = np.arange(len(data))
width = 0.35
bars1 = ax.bar(x - width/2, post_widths_naive, width, label='Naive (ignoring sigma)',
               color='red', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, post_widths_proper, width, label='Proper (using sigma)',
               color='green', alpha=0.7, edgecolor='black')
ax.set_xlabel('Group', fontweight='bold')
ax.set_ylabel('Posterior Standard Deviation', fontweight='bold')
ax.set_title('Posterior Uncertainty: Naive vs Proper')
ax.set_xticks(x)
ax.set_xticklabels(data['group'])
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 4: Power analysis - ability to detect true effects
ax = axes[1, 1]
# For different true effect sizes, what's the power to detect them?
true_effects = np.linspace(0, 30, 100)
# Power with average measurement error
avg_sigma = data['sigma'].mean()
power_avg = stats.norm.sf(1.96 - true_effects / avg_sigma)
ax.plot(true_effects, power_avg, 'b-', linewidth=2, label=f'Avg sigma = {avg_sigma:.1f}')

# Power with best measurement error (smallest sigma)
best_sigma = data['sigma'].min()
power_best = stats.norm.sf(1.96 - true_effects / best_sigma)
ax.plot(true_effects, power_best, 'g-', linewidth=2, label=f'Best sigma = {best_sigma:.1f}')

# Power with worst measurement error (largest sigma)
worst_sigma = data['sigma'].max()
power_worst = stats.norm.sf(1.96 - true_effects / worst_sigma)
ax.plot(true_effects, power_worst, 'r-', linewidth=2, label=f'Worst sigma = {worst_sigma:.1f}')

ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='80% power')
ax.set_xlabel('True Effect Size', fontweight='bold')
ax.set_ylabel('Statistical Power', fontweight='bold')
ax.set_title('Detection Power Given Measurement Error')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/08_measurement_error_impact.png', dpi=300, bbox_inches='tight')
print("Saved: /workspace/eda/visualizations/08_measurement_error_impact.png")
plt.close()

print("\nAll model implication visualizations completed!")
