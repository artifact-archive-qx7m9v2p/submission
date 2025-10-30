"""
Prior Specification Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
data = pd.read_csv('/workspace/data/data_analyst_3.csv')
output_dir = Path('/workspace/eda/analyst_3/visualizations')

print("Creating prior specification visualizations...")

# Calculate key statistics
pooled_rate = data['r_successes'].sum() / data['n_trials'].sum()
mean_rate = data['success_rate'].mean()
std_rate = data['success_rate'].std()

# Beta parameters from method of moments
def beta_params_from_moments(mean, variance):
    """Estimate Beta(alpha, beta) parameters from mean and variance"""
    if variance >= mean * (1 - mean):
        return None, None
    common = mean * (1 - mean) / variance - 1
    alpha = mean * common
    beta = (1 - mean) * common
    return alpha, beta

alpha_obs, beta_obs = beta_params_from_moments(mean_rate, std_rate**2)

# Figure 1: Prior options comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prior Distribution Options for Success Rate', fontsize=14, fontweight='bold')

p_range = np.linspace(0, 0.3, 1000)

# 1. Non-informative priors
axes[0, 0].plot(p_range, stats.beta.pdf(p_range, 1, 1), label='Beta(1, 1) - Uniform', linewidth=2)
axes[0, 0].plot(p_range, stats.beta.pdf(p_range, 0.5, 0.5), label='Beta(0.5, 0.5) - Jeffreys', linewidth=2)
axes[0, 0].plot(p_range, stats.beta.pdf(p_range, 2, 2), label='Beta(2, 2) - Weak', linewidth=2)
axes[0, 0].axvline(pooled_rate, color='red', linestyle='--', label=f'Pooled rate={pooled_rate:.3f}')
axes[0, 0].set_xlabel('Success Rate')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Non-informative Prior Options')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 0.3)
axes[0, 0].grid(True, alpha=0.3)

# 2. Informative priors based on data
if alpha_obs is not None and alpha_obs > 0:
    axes[0, 1].plot(p_range, stats.beta.pdf(p_range, alpha_obs, beta_obs),
                    label=f'Beta({alpha_obs:.1f}, {beta_obs:.1f}) - From data variation',
                    linewidth=2, color='purple')
else:
    # Use a different informative prior
    alpha_alt, beta_alt = 5, 60
    axes[0, 1].plot(p_range, stats.beta.pdf(p_range, alpha_alt, beta_alt),
                    label=f'Beta({alpha_alt}, {beta_alt}) - Moderately informative',
                    linewidth=2, color='purple')

# Weakly informative centered on pooled
alpha_weak = 2
beta_weak = (1 - pooled_rate) / pooled_rate * alpha_weak
axes[0, 1].plot(p_range, stats.beta.pdf(p_range, alpha_weak, beta_weak),
                label=f'Beta({alpha_weak}, {beta_weak:.1f}) - Weakly informative',
                linewidth=2, color='orange')

# Histogram of observed rates
axes[0, 1].hist(data['success_rate'], bins=15, density=True, alpha=0.3,
                color='gray', edgecolor='black', label='Observed rates')
axes[0, 1].axvline(pooled_rate, color='red', linestyle='--', label=f'Pooled rate={pooled_rate:.3f}')
axes[0, 1].set_xlabel('Success Rate')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Informative Prior Options vs Observed Data')
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 0.3)
axes[0, 1].grid(True, alpha=0.3)

# 3. Prior vs posterior (simulated) for one group
# Simulate what posterior might look like for a typical group
typical_group = data.iloc[5]  # Group 6
n, r = typical_group['n_trials'], typical_group['r_successes']

# Different priors
priors = [
    ('Beta(1, 1) - Uniform', 1, 1),
    (f'Beta(2, {beta_weak:.1f}) - Weak', 2, beta_weak),
]

for label, alpha_prior, beta_prior in priors:
    # Posterior is Beta(alpha + r, beta + n - r)
    alpha_post = alpha_prior + r
    beta_post = beta_prior + n - r
    axes[1, 0].plot(p_range, stats.beta.pdf(p_range, alpha_post, beta_post),
                    label=f'{label} → Post', linewidth=2)

axes[1, 0].axvline(r/n, color='black', linestyle='--', label=f'MLE={r/n:.3f}', linewidth=2)
axes[1, 0].set_xlabel('Success Rate')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title(f'Prior Influence on Posterior (Group {int(typical_group["group_id"])}: n={int(n)}, r={int(r)})')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, 0.2)
axes[1, 0].grid(True, alpha=0.3)

# 4. Hierarchical model illustration
# Show how individual groups relate to population distribution
axes[1, 1].hist(data['success_rate'], bins=15, density=True, alpha=0.3,
                color='gray', edgecolor='black', label='Observed group rates')

# Fit a normal to logit-transformed rates (approximation)
logit_rates = np.log(data['success_rate'] / (1 - data['success_rate']))
mu_logit = logit_rates.mean()
tau_logit = logit_rates.std()

# Transform back to probability scale for visualization
p_sim = np.linspace(0.01, 0.99, 1000)
logit_p = np.log(p_sim / (1 - p_sim))
normal_density = stats.norm.pdf(logit_p, mu_logit, tau_logit)
# Jacobian adjustment for transformation
jacobian = 1 / (p_sim * (1 - p_sim))
adjusted_density = normal_density * jacobian
# Normalize
adjusted_density = adjusted_density / np.trapz(adjusted_density, p_sim) * np.trapz(adjusted_density, p_sim)

axes[1, 1].plot(p_sim, adjusted_density / adjusted_density.max() * 15,
                label='Implied population dist\n(logit-normal)', linewidth=2, color='red')
axes[1, 1].axvline(pooled_rate, color='blue', linestyle='--',
                   label=f'Population mean={pooled_rate:.3f}', linewidth=2)
axes[1, 1].set_xlabel('Success Rate')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Hierarchical Model: Group Rates around Population Mean')
axes[1, 1].legend()
axes[1, 1].set_xlim(0, 0.3)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prior_options.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: prior_options.png")

# Figure 2: Prior sensitivity illustration
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Prior Sensitivity Analysis Preview', fontsize=14, fontweight='bold')

# Small sample group (most influenced by prior)
small_group = data[data['n_trials'] < 100].iloc[0]
n_small, r_small = small_group['n_trials'], small_group['r_successes']

# Large sample group (less influenced by prior)
large_group = data[data['n_trials'] > 500].iloc[0]
n_large, r_large = large_group['n_trials'], large_group['r_successes']

# Priors to compare
priors_to_test = [
    ('Uniform', 1, 1, 'blue'),
    ('Weak', 2, beta_weak, 'orange'),
    ('Moderate', 5, 65, 'purple'),
]

# Small sample group
for label, a, b, color in priors_to_test:
    alpha_post = a + r_small
    beta_post = b + n_small - r_small
    axes[0].plot(p_range, stats.beta.pdf(p_range, alpha_post, beta_post),
                 label=f'{label} prior', linewidth=2, color=color, alpha=0.7)

axes[0].axvline(r_small/n_small, color='black', linestyle='--',
                label=f'MLE={r_small/n_small:.3f}', linewidth=2)
axes[0].set_xlabel('Success Rate')
axes[0].set_ylabel('Density')
axes[0].set_title(f'Small Sample Group (n={int(n_small)}, r={int(r_small)})\nPrior Has Strong Influence')
axes[0].legend()
axes[0].set_xlim(0, 0.3)
axes[0].grid(True, alpha=0.3)

# Large sample group
for label, a, b, color in priors_to_test:
    alpha_post = a + r_large
    beta_post = b + n_large - r_large
    axes[1].plot(p_range, stats.beta.pdf(p_range, alpha_post, beta_post),
                 label=f'{label} prior', linewidth=2, color=color, alpha=0.7)

axes[1].axvline(r_large/n_large, color='black', linestyle='--',
                label=f'MLE={r_large/n_large:.3f}', linewidth=2)
axes[1].set_xlabel('Success Rate')
axes[1].set_ylabel('Density')
axes[1].set_title(f'Large Sample Group (n={int(n_large)}, r={int(r_large)})\nPrior Has Weak Influence')
axes[1].legend()
axes[1].set_xlim(0.02, 0.08)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'prior_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: prior_sensitivity.png")

print("\nPrior visualizations complete!")
