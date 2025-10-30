"""
Prior Predictive Check for Random-Effects Hierarchical Model

Tests whether observed data is plausible under the joint prior distribution.
Also assesses prior sensitivity to tau hyperprior choices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Load observed data
data = pd.read_csv('/workspace/data/data.csv')
y_obs = data['y'].values
sigma_obs = data['sigma'].values
J = len(y_obs)

print("="*70)
print("PRIOR PREDICTIVE CHECK - RANDOM EFFECTS HIERARCHICAL MODEL")
print("="*70)
print(f"\nObserved data:")
print(f"  y_obs: {y_obs}")
print(f"  sigma: {sigma_obs}")
print(f"  J (number of studies): {J}")

# Prior specifications to test
prior_specs = {
    'baseline': {'tau_scale': 5, 'label': 'τ ~ Half-Normal(0, 5²)'},
    'tight': {'tau_scale': 3, 'label': 'τ ~ Half-Normal(0, 3²)'},
    'wide': {'tau_scale': 10, 'label': 'τ ~ Half-Normal(0, 10²)'}
}

# Number of prior samples
n_prior_samples = 5000

results = {}

for spec_name, spec in prior_specs.items():
    print(f"\n{'='*70}")
    print(f"Prior specification: {spec['label']}")
    print(f"{'='*70}")

    # Sample from joint prior
    mu_prior = np.random.normal(0, 20, n_prior_samples)
    tau_prior = np.abs(np.random.normal(0, spec['tau_scale'], n_prior_samples))

    # Sample study-specific effects (non-centered parameterization)
    theta_raw_prior = np.random.normal(0, 1, (n_prior_samples, J))
    theta_prior = mu_prior[:, None] + tau_prior[:, None] * theta_raw_prior

    # Generate prior predictive observations
    y_prior_pred = np.random.normal(theta_prior, sigma_obs[None, :])

    # Summary statistics
    print(f"\nPrior samples (hyperparameters):")
    print(f"  μ: mean={mu_prior.mean():.2f}, sd={mu_prior.std():.2f}")
    print(f"  τ: mean={tau_prior.mean():.2f}, sd={tau_prior.std():.2f}")
    print(f"  τ: median={np.median(tau_prior):.2f}, 95% CI=[{np.percentile(tau_prior, 2.5):.2f}, {np.percentile(tau_prior, 97.5):.2f}]")

    print(f"\nPrior predictive y:")
    print(f"  mean={y_prior_pred.mean():.2f}, sd={y_prior_pred.std():.2f}")
    print(f"  range: [{y_prior_pred.min():.2f}, {y_prior_pred.max():.2f}]")

    # Check if observed data is plausible
    # For each observation, compute percentile rank in prior predictive
    percentile_ranks = []
    for i in range(J):
        rank = (y_prior_pred[:, i] < y_obs[i]).mean() * 100
        percentile_ranks.append(rank)
        print(f"  y[{i}] = {y_obs[i]:3.0f}: {rank:.1f}th percentile")

    # Check for extreme values (< 1% or > 99%)
    extreme = [(i, rank) for i, rank in enumerate(percentile_ranks)
               if rank < 1 or rank > 99]
    if extreme:
        print(f"\n  WARNING: {len(extreme)} observations at extreme prior percentiles:")
        for i, rank in extreme:
            print(f"    Study {i}: {rank:.1f}th percentile")
    else:
        print(f"\n  All observations within [1%, 99%] prior predictive range ✓")

    # Store results
    results[spec_name] = {
        'mu_prior': mu_prior,
        'tau_prior': tau_prior,
        'theta_prior': theta_prior,
        'y_prior_pred': y_prior_pred,
        'percentile_ranks': percentile_ranks,
        'label': spec['label']
    }

# Visualizations
print(f"\n{'='*70}")
print("Creating visualizations...")
print(f"{'='*70}")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# Plot prior distributions for hyperparameters
for col, (spec_name, spec) in enumerate(prior_specs.items()):
    res = results[spec_name]

    # μ prior
    ax = fig.add_subplot(gs[0, col])
    ax.hist(res['mu_prior'], bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Prior mean')
    ax.set_xlabel('μ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{spec["label"]}\nμ ~ Normal(0, 20²)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # τ prior
    ax = fig.add_subplot(gs[1, col])
    ax.hist(res['tau_prior'], bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(res['tau_prior'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean={res["tau_prior"].mean():.1f}')
    ax.set_xlabel('τ', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(spec['label'], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Prior predictive distribution vs observed data
    ax = fig.add_subplot(gs[2, col])
    for i in range(J):
        ax.hist(res['y_prior_pred'][:, i], bins=50, alpha=0.15, color='gray')
    # Overlay all y's together
    ax.hist(res['y_prior_pred'].flatten(), bins=100, alpha=0.4, color='skyblue',
            edgecolor='black', label='Prior predictive')
    ax.scatter(y_obs, np.zeros(J), color='red', s=100, marker='D',
               label='Observed data', zorder=10, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('y', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Prior Predictive vs Observed', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Percentile ranks (should be roughly uniform if prior is reasonable)
    ax = fig.add_subplot(gs[3, col])
    ax.bar(range(J), res['percentile_ranks'], color='steelblue', edgecolor='black', alpha=0.7)
    ax.axhline(50, color='red', linestyle='--', linewidth=2, label='Median')
    ax.axhline(2.5, color='orange', linestyle=':', linewidth=1.5, label='2.5th/97.5th percentile')
    ax.axhline(97.5, color='orange', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Study Index', fontsize=11)
    ax.set_ylabel('Percentile Rank', fontsize=11)
    ax.set_title('Observed Data Percentile Ranks', fontsize=10)
    ax.set_ylim([0, 100])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Prior Predictive Check - Random-Effects Hierarchical Model',
             fontsize=14, fontweight='bold', y=0.995)

plot_path = '/workspace/experiments/experiment_2/prior_predictive_check/plots/prior_predictive_check.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot: {plot_path}")
plt.close()

# Create comparison plot for prior sensitivity
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Compare tau priors
ax = axes[0, 0]
for spec_name, res in results.items():
    ax.hist(res['tau_prior'], bins=50, alpha=0.5, label=res['label'], density=True)
ax.set_xlabel('τ', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Sensitivity: τ Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Compare prior predictive ranges
ax = axes[0, 1]
for spec_name, res in results.items():
    y_flat = res['y_prior_pred'].flatten()
    ax.hist(y_flat, bins=100, alpha=0.3, label=res['label'], density=True)
ax.scatter(y_obs, np.zeros(J), color='red', s=100, marker='D',
           label='Observed', zorder=10, edgecolor='black', linewidth=1.5)
ax.set_xlabel('y', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prior Predictive Sensitivity', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Compare percentile ranks
ax = axes[1, 0]
for spec_name, res in results.items():
    ax.plot(range(J), res['percentile_ranks'], marker='o', label=res['label'], linewidth=2)
ax.axhline(50, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax.axhline(2.5, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
ax.axhline(97.5, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
ax.set_xlabel('Study Index', fontsize=12)
ax.set_ylabel('Percentile Rank', fontsize=12)
ax.set_title('Percentile Rank Comparison', fontsize=13, fontweight='bold')
ax.set_ylim([0, 100])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Summary statistics table
ax = axes[1, 1]
ax.axis('off')
summary_text = "Prior Sensitivity Summary\n" + "="*40 + "\n\n"
for spec_name, res in results.items():
    tau_mean = res['tau_prior'].mean()
    tau_median = np.median(res['tau_prior'])
    tau_95 = np.percentile(res['tau_prior'], 97.5)
    y_range = res['y_prior_pred'].max() - res['y_prior_pred'].min()
    n_extreme = sum(1 for r in res['percentile_ranks'] if r < 1 or r > 99)

    summary_text += f"{res['label']}\n"
    summary_text += f"  τ mean: {tau_mean:.2f}\n"
    summary_text += f"  τ median: {tau_median:.2f}\n"
    summary_text += f"  τ 97.5%: {tau_95:.2f}\n"
    summary_text += f"  y range: {y_range:.1f}\n"
    summary_text += f"  # extreme obs: {n_extreme}\n\n"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Prior Sensitivity Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()

plot_path = '/workspace/experiments/experiment_2/prior_predictive_check/plots/prior_sensitivity.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Saved plot: {plot_path}")
plt.close()

# Final assessment
print(f"\n{'='*70}")
print("ASSESSMENT")
print(f"{'='*70}")

# Check baseline prior
baseline = results['baseline']
n_extreme = sum(1 for r in baseline['percentile_ranks'] if r < 1 or r > 99)
y_range = baseline['y_prior_pred'].max() - baseline['y_prior_pred'].min()

print(f"\nBaseline prior (τ ~ Half-Normal(0, 5²)):")
print(f"  - Observed data range: [{y_obs.min()}, {y_obs.max()}]")
print(f"  - Prior predictive range: [{baseline['y_prior_pred'].min():.1f}, {baseline['y_prior_pred'].max():.1f}]")
print(f"  - Number of extreme observations: {n_extreme}/{J}")
print(f"  - All observations plausible: {'YES' if n_extreme == 0 else 'NO'}")

# Prior sensitivity
tau_ranges = [np.percentile(results[spec]['tau_prior'], [2.5, 97.5]) for spec in prior_specs]
tau_sensitivity = max([r[1] - r[0] for r in tau_ranges]) / min([r[1] - r[0] for r in tau_ranges])
print(f"\nPrior sensitivity:")
print(f"  - τ prior range sensitivity ratio: {tau_sensitivity:.2f}")
print(f"  - Moderate sensitivity expected for hierarchical models: {'YES' if tau_sensitivity < 3 else 'NO'}")

# Decision
if n_extreme == 0 and tau_sensitivity < 3:
    decision = "PASS"
    reasoning = "All observed data plausible under prior, moderate prior sensitivity"
elif n_extreme == 0:
    decision = "PASS (with note)"
    reasoning = "Data plausible but consider prior sensitivity in interpretation"
else:
    decision = "REVIEW"
    reasoning = "Some observations at extreme prior percentiles"

print(f"\n{'='*70}")
print(f"DECISION: {decision}")
print(f"REASONING: {reasoning}")
print(f"{'='*70}")

# Save results
output = {
    'decision': decision,
    'reasoning': reasoning,
    'n_extreme': n_extreme,
    'tau_sensitivity': tau_sensitivity,
    'percentile_ranks': baseline['percentile_ranks']
}

import json
with open('/workspace/experiments/experiment_2/prior_predictive_check/prior_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: /workspace/experiments/experiment_2/prior_predictive_check/prior_results.json")
print("\nPrior predictive check complete!")
