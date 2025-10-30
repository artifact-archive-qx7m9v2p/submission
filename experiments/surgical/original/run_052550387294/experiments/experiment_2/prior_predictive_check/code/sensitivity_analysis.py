"""
Sensitivity Analysis: Alternative Priors for Hierarchical Logit Model

Tests robustness of prior predictive checks to alternative prior specifications.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Set random seed
np.random.seed(123)

# Configuration
N_PRIOR_DRAWS = 2000
N_TRIALS = 12

# Load observed data
data = pd.read_csv('/workspace/data/data.csv')
n_obs = data['n'].values
r_obs = data['r'].values
total_successes_obs = r_obs.sum()

print("=" * 70)
print("SENSITIVITY ANALYSIS: ALTERNATIVE PRIORS")
print("=" * 70)


def sample_prior_predictive_alt(n_draws, mu_loc, mu_scale, sigma_scale, seed=None):
    """Sample from alternative prior specification."""
    if seed is not None:
        np.random.seed(seed)

    mu_logit = stats.norm.rvs(loc=mu_loc, scale=mu_scale, size=n_draws)
    sigma = stats.halfnorm.rvs(loc=0, scale=sigma_scale, size=n_draws)
    eta = stats.norm.rvs(loc=0, scale=1, size=(n_draws, N_TRIALS))

    logit_theta = mu_logit[:, np.newaxis] + sigma[:, np.newaxis] * eta
    theta = expit(logit_theta)

    r = np.zeros((n_draws, N_TRIALS), dtype=int)
    for i in range(n_draws):
        for j in range(N_TRIALS):
            r[i, j] = stats.binom.rvs(n=n_obs[j], p=theta[i, j])

    total_r = r.sum(axis=1)

    return {
        'mu_logit': mu_logit,
        'sigma': sigma,
        'theta': theta,
        'r': r,
        'total_r': total_r
    }


# Define alternative prior specifications
prior_specs = {
    'Baseline': {
        'mu_loc': -2.53,
        'mu_scale': 1.0,
        'sigma_scale': 1.0,
        'description': 'Original: N(-2.53, 1), HalfN(0, 1)'
    },
    'Wider μ': {
        'mu_loc': -2.53,
        'mu_scale': 2.0,
        'sigma_scale': 1.0,
        'description': 'N(-2.53, 2), HalfN(0, 1)'
    },
    'More dispersed σ': {
        'mu_loc': -2.53,
        'mu_scale': 1.0,
        'sigma_scale': 2.0,
        'description': 'N(-2.53, 1), HalfN(0, 2)'
    },
    'Both wider': {
        'mu_loc': -2.53,
        'mu_scale': 2.0,
        'sigma_scale': 2.0,
        'description': 'N(-2.53, 2), HalfN(0, 2)'
    },
    'Tighter μ': {
        'mu_loc': -2.53,
        'mu_scale': 0.5,
        'sigma_scale': 1.0,
        'description': 'N(-2.53, 0.5), HalfN(0, 1)'
    }
}

# Sample from each prior specification
results = {}
for name, spec in prior_specs.items():
    print(f"\nSampling: {name}")
    print(f"  {spec['description']}")
    results[name] = sample_prior_predictive_alt(
        N_PRIOR_DRAWS,
        spec['mu_loc'],
        spec['mu_scale'],
        spec['sigma_scale'],
        seed=123 + list(prior_specs.keys()).index(name)
    )

# Print summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS BY PRIOR SPECIFICATION")
print("=" * 70)

summary_data = []
for name, res in results.items():
    mu_prob = expit(res['mu_logit'])

    summary_data.append({
        'Prior': name,
        'μ_prob mean': mu_prob.mean(),
        'μ_prob SD': mu_prob.std(),
        'μ_prob [2.5%, 97.5%]': f"[{np.quantile(mu_prob, 0.025):.3f}, {np.quantile(mu_prob, 0.975):.3f}]",
        'σ mean': res['sigma'].mean(),
        'σ SD': res['sigma'].std(),
        'σ [2.5%, 97.5%]': f"[{np.quantile(res['sigma'], 0.025):.3f}, {np.quantile(res['sigma'], 0.975):.3f}]",
        'Total r mean': res['total_r'].mean(),
        'Total r SD': res['total_r'].std(),
        'Obs percentile': stats.percentileofscore(res['total_r'], total_successes_obs)
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Check coverage of observed data
print("\n" + "=" * 70)
print("COVERAGE OF OBSERVED TOTAL SUCCESSES")
print("=" * 70)
print(f"Observed: {total_successes_obs} successes")

for name, res in results.items():
    percentile = stats.percentileofscore(res['total_r'], total_successes_obs)
    q025, q975 = np.quantile(res['total_r'], [0.025, 0.975])
    covered = q025 <= total_successes_obs <= q975
    print(f"\n{name}:")
    print(f"  95% interval: [{q025:.0f}, {q975:.0f}]")
    print(f"  Covered: {covered}")
    print(f"  Percentile: {percentile:.1f}%")

# Check extreme trials coverage
print("\n" + "=" * 70)
print("COVERAGE OF EXTREME TRIALS")
print("=" * 70)

for trial_idx, trial_name in [(0, "Trial 1 (0/47)"), (7, "Trial 8 (31/215)")]:
    print(f"\n{trial_name}:")
    for name, res in results.items():
        r_trial = res['r'][:, trial_idx]
        percentile = stats.percentileofscore(r_trial, r_obs[trial_idx])
        q025, q975 = np.quantile(r_trial, [0.025, 0.975])
        covered = q025 <= r_obs[trial_idx] <= q975
        print(f"  {name}: {percentile:.1f}th percentile, 95% [{q025:.0f}, {q975:.0f}], covered={covered}")

# Create visualization
plots_dir = Path('/workspace/experiments/experiment_2/prior_predictive_check/plots')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Total successes by prior
ax = axes[0, 0]
colors = plt.cm.Set2(range(len(results)))
for i, (name, res) in enumerate(results.items()):
    ax.hist(res['total_r'], bins=50, alpha=0.4, density=True,
           label=name, color=colors[i])
ax.axvline(total_successes_obs, color='red', linestyle='--', lw=3,
          label='Observed')
ax.set_xlabel('Total successes', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Total Successes: Prior Sensitivity', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: μ_prob by prior
ax = axes[0, 1]
mu_prob_data = []
for name, res in results.items():
    mu_prob = expit(res['mu_logit'])
    mu_prob_data.extend([(name, x) for x in mu_prob])
mu_prob_df = pd.DataFrame(mu_prob_data, columns=['Prior', 'μ_prob'])
sns.violinplot(data=mu_prob_df, x='Prior', y='μ_prob', ax=ax)
ax.axhline(0.074, color='red', linestyle='--', lw=2, label='Target (0.074)')
ax.set_xlabel('Prior specification', fontsize=11)
ax.set_ylabel('μ_prob (probability scale)', fontsize=11)
ax.set_title('Population Mean Distribution', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3, axis='y')
ax.legend(fontsize=8)

# Plot 3: σ by prior
ax = axes[0, 2]
sigma_data = []
for name, res in results.items():
    sigma_data.extend([(name, x) for x in res['sigma']])
sigma_df = pd.DataFrame(sigma_data, columns=['Prior', 'σ'])
sns.violinplot(data=sigma_df, x='Prior', y='σ', ax=ax)
ax.set_xlabel('Prior specification', fontsize=11)
ax.set_ylabel('σ (scale parameter)', fontsize=11)
ax.set_title('Scale Parameter Distribution', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3, axis='y')

# Plot 4: Trial 1 coverage
ax = axes[1, 0]
trial_1_data = []
for name, res in results.items():
    trial_1_data.extend([(name, x) for x in res['r'][:, 0]])
trial_1_df = pd.DataFrame(trial_1_data, columns=['Prior', 'Successes'])
sns.violinplot(data=trial_1_df, x='Prior', y='Successes', ax=ax)
ax.axhline(r_obs[0], color='red', linestyle='--', lw=2, label='Observed (0)')
ax.set_xlabel('Prior specification', fontsize=11)
ax.set_ylabel('Successes', fontsize=11)
ax.set_title('Trial 1 (0/47): Prior Sensitivity', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3, axis='y')
ax.legend(fontsize=8)
ax.set_ylim(-2, 50)

# Plot 5: Trial 8 coverage
ax = axes[1, 1]
trial_8_data = []
for name, res in results.items():
    trial_8_data.extend([(name, x) for x in res['r'][:, 7]])
trial_8_df = pd.DataFrame(trial_8_data, columns=['Prior', 'Successes'])
sns.violinplot(data=trial_8_df, x='Prior', y='Successes', ax=ax)
ax.axhline(r_obs[7], color='red', linestyle='--', lw=2, label='Observed (31)')
ax.set_xlabel('Prior specification', fontsize=11)
ax.set_ylabel('Successes', fontsize=11)
ax.set_title('Trial 8 (31/215): Prior Sensitivity', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3, axis='y')
ax.legend(fontsize=8)

# Plot 6: Observed percentiles by prior
ax = axes[1, 2]
percentile_data = []
for trial_idx in range(N_TRIALS):
    for name, res in results.items():
        percentile = stats.percentileofscore(res['r'][:, trial_idx], r_obs[trial_idx])
        percentile_data.append({
            'Prior': name,
            'Trial': trial_idx + 1,
            'Percentile': percentile
        })
percentile_df = pd.DataFrame(percentile_data)

for i, (name, color) in enumerate(zip(results.keys(), colors)):
    subset = percentile_df[percentile_df['Prior'] == name]
    ax.scatter(subset['Trial'], subset['Percentile'],
              label=name, color=color, alpha=0.7, s=60)

ax.axhline(50, color='gray', linestyle='--', lw=1, alpha=0.5)
ax.axhspan(2.5, 97.5, alpha=0.1, color='green', label='Expected range')
ax.set_xlabel('Trial', fontsize=11)
ax.set_ylabel('Observed percentile', fontsize=11)
ax.set_title('Calibration: All Trials', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(plots_dir / 'sensitivity_analysis.png')
print(f"\n" + "=" * 70)
print(f"Sensitivity analysis plot saved to:")
print(f"  {plots_dir / 'sensitivity_analysis.png'}")
print("=" * 70)
plt.close()

# Save summary
summary_df.to_csv(
    Path('/workspace/experiments/experiment_2/prior_predictive_check/code') / 'sensitivity_summary.csv',
    index=False
)
print(f"\nSummary table saved to:")
print(f"  /workspace/experiments/experiment_2/prior_predictive_check/code/sensitivity_summary.csv")
