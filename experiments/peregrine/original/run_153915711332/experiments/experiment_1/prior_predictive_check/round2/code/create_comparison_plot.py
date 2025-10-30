"""
Create Round 1 vs Round 2 Comparison Plot
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

# Load Round 1 samples
samples_r1 = np.load('/workspace/experiments/experiment_1/prior_predictive_check/code/prior_samples.npz')
sigma_eta_r1 = samples_r1['sigma_eta']
phi_r1 = samples_r1['phi']
C_prior_r1 = samples_r1['C_prior']
delta_r1 = samples_r1['delta']

# Compute Round 1 summary statistics
prior_mean_counts_r1 = np.mean(C_prior_r1, axis=1)
prior_max_counts_r1 = np.max(C_prior_r1, axis=1)

# Load Round 2 samples
samples_r2 = np.load('/workspace/experiments/experiment_1/prior_predictive_check/round2/code/prior_samples.npz')
sigma_eta_r2 = samples_r2['sigma_eta']
phi_r2 = samples_r2['phi']
C_prior_r2 = samples_r2['C_prior']
delta_r2 = samples_r2['delta']
prior_mean_counts_r2 = samples_r2['prior_mean_counts']
prior_max_counts_r2 = samples_r2['prior_max_counts']

# Observed data
obs_mean = 109.45
obs_max = 272

print("Creating Round 1 vs Round 2 Comparison Plot...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel A: Sigma_eta comparison
ax = axes[0, 0]
ax.hist(sigma_eta_r1, bins=50, alpha=0.6, label='Round 1: Exp(10)', density=True,
        edgecolor='black', color='red')
ax.hist(sigma_eta_r2, bins=50, alpha=0.6, label='Round 2: Exp(20)', density=True,
        edgecolor='black', color='blue')
ax.axvline(np.median(sigma_eta_r1), color='darkred', linestyle='--', linewidth=2, alpha=0.8)
ax.axvline(np.median(sigma_eta_r2), color='darkblue', linestyle='--', linewidth=2, alpha=0.8)
ax.set_xlabel('Sigma_eta', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Sigma_eta Prior: Round 1 vs Round 2', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

stats_text = f'R1 median: {np.median(sigma_eta_r1):.3f}\nR2 median: {np.median(sigma_eta_r2):.3f}'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel B: Phi comparison
ax = axes[0, 1]
ax.hist(phi_r1, bins=50, alpha=0.6, label='Round 1: Exp(0.1)', density=True,
        edgecolor='black', color='red')
ax.hist(phi_r2, bins=50, alpha=0.6, label='Round 2: Exp(0.05)', density=True,
        edgecolor='black', color='blue')
ax.axvline(np.median(phi_r1), color='darkred', linestyle='--', linewidth=2, alpha=0.8)
ax.axvline(np.median(phi_r2), color='darkblue', linestyle='--', linewidth=2, alpha=0.8)
ax.set_xlabel('Phi', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Phi Prior: Round 1 vs Round 2', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

stats_text = f'R1 median: {np.median(phi_r1):.1f}\nR2 median: {np.median(phi_r2):.1f}'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel C: Delta (unchanged)
ax = axes[0, 2]
ax.hist(delta_r1, bins=50, alpha=0.6, label='Round 1', density=True,
        edgecolor='black', color='gray')
ax.hist(delta_r2, bins=50, alpha=0.6, label='Round 2', density=True,
        edgecolor='black', color='gray')
ax.set_xlabel('Delta', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Delta (UNCHANGED): Normal(0.05, 0.02)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

stats_text = f'R1 mean: {np.mean(delta_r1):.4f}\nR2 mean: {np.mean(delta_r2):.4f}'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel D: Mean counts comparison
ax = axes[1, 0]
bins_range = (0, 1500)
ax.hist(prior_mean_counts_r1, bins=50, alpha=0.6, label='Round 1',
        edgecolor='black', color='red', range=bins_range)
ax.hist(prior_mean_counts_r2, bins=50, alpha=0.6, label='Round 2',
        edgecolor='black', color='blue', range=bins_range)
ax.axvline(obs_mean, color='black', linestyle='--', linewidth=3, label='Observed')
ax.set_xlabel('Mean Count', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Prior Predictive Mean: Round 1 vs Round 2', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

pct_r1 = 100 * np.sum(prior_mean_counts_r1 < obs_mean) / len(prior_mean_counts_r1)
pct_r2 = 100 * np.sum(prior_mean_counts_r2 < obs_mean) / len(prior_mean_counts_r2)
stats_text = f'Obs @ R1: {pct_r1:.1f}%ile\nObs @ R2: {pct_r2:.1f}%ile'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel E: Max counts comparison (log scale)
ax = axes[1, 1]
ax.hist(np.log10(prior_max_counts_r1 + 1), bins=50, alpha=0.6, label='Round 1',
        edgecolor='black', color='red')
ax.hist(np.log10(prior_max_counts_r2 + 1), bins=50, alpha=0.6, label='Round 2',
        edgecolor='black', color='blue')
ax.axvline(np.log10(obs_max + 1), color='black', linestyle='--', linewidth=3, label='Observed')
ax.set_xlabel('Log10(Max Count + 1)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Prior Predictive Max: Round 1 vs Round 2', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

pct_r1 = 100 * np.sum(prior_max_counts_r1 < obs_max) / len(prior_max_counts_r1)
pct_r2 = 100 * np.sum(prior_max_counts_r2 < obs_max) / len(prior_max_counts_r2)
stats_text = f'Obs @ R1: {pct_r1:.1f}%ile\nObs @ R2: {pct_r2:.1f}%ile'
ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel F: Extreme counts comparison
ax = axes[1, 2]
all_counts_r1 = C_prior_r1.flatten()
all_counts_r2 = C_prior_r2.flatten()

thresholds = [100, 500, 1000, 5000, 10000]
pct_r1_list = [100 * np.sum(all_counts_r1 > t) / len(all_counts_r1) for t in thresholds]
pct_r2_list = [100 * np.sum(all_counts_r2 > t) / len(all_counts_r2) for t in thresholds]

x = np.arange(len(thresholds))
width = 0.35
bars1 = ax.bar(x - width/2, pct_r1_list, width, label='Round 1', alpha=0.7, color='red')
bars2 = ax.bar(x + width/2, pct_r2_list, width, label='Round 2', alpha=0.7, color='blue')

ax.set_xticks(x)
ax.set_xticklabels([f'>{t}' for t in thresholds], rotation=45)
ax.set_xlabel('Count Threshold', fontsize=12)
ax.set_ylabel('Percentage of All Counts', fontsize=12)
ax.set_title('Extreme Count Frequency: Round 1 vs Round 2', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_1/prior_predictive_check/round2/plots/round1_vs_round2_comparison.png',
            dpi=300, bbox_inches='tight')
print("Saved: round1_vs_round2_comparison.png")

# Print summary statistics
print("\n" + "="*80)
print("ROUND 1 vs ROUND 2 COMPARISON SUMMARY")
print("="*80)

print("\nPRIOR PARAMETERS:")
print("-"*80)
print(f"Sigma_eta:")
print(f"  Round 1: median={np.median(sigma_eta_r1):.4f}, 95% CI=[{np.percentile(sigma_eta_r1, 2.5):.4f}, {np.percentile(sigma_eta_r1, 97.5):.4f}]")
print(f"  Round 2: median={np.median(sigma_eta_r2):.4f}, 95% CI=[{np.percentile(sigma_eta_r2, 2.5):.4f}, {np.percentile(sigma_eta_r2, 97.5):.4f}]")
print(f"  Change: {((np.median(sigma_eta_r2) - np.median(sigma_eta_r1)) / np.median(sigma_eta_r1) * 100):+.1f}%")

print(f"\nPhi:")
print(f"  Round 1: median={np.median(phi_r1):.2f}, 95% CI=[{np.percentile(phi_r1, 2.5):.2f}, {np.percentile(phi_r1, 97.5):.2f}]")
print(f"  Round 2: median={np.median(phi_r2):.2f}, 95% CI=[{np.percentile(phi_r2, 2.5):.2f}, {np.percentile(phi_r2, 97.5):.2f}]")
print(f"  Change: {((np.median(phi_r2) - np.median(phi_r1)) / np.median(phi_r1) * 100):+.1f}%")

print("\nPRIOR PREDICTIVE SUMMARIES:")
print("-"*80)
print(f"Mean Counts:")
print(f"  Round 1: mean={np.mean(prior_mean_counts_r1):.1f}, median={np.median(prior_mean_counts_r1):.1f}")
print(f"  Round 2: mean={np.mean(prior_mean_counts_r2):.1f}, median={np.median(prior_mean_counts_r2):.1f}")
print(f"  Observed: {obs_mean:.1f}")
print(f"  Improvement: {((np.mean(prior_mean_counts_r1) - np.mean(prior_mean_counts_r2)) / np.mean(prior_mean_counts_r1) * 100):.1f}% closer")

print(f"\nMax Counts (95% CI upper):")
print(f"  Round 1: {np.percentile(prior_max_counts_r1, 97.5):.0f}")
print(f"  Round 2: {np.percentile(prior_max_counts_r2, 97.5):.0f}")
print(f"  Observed: {obs_max}")
print(f"  Reduction: {((np.percentile(prior_max_counts_r1, 97.5) - np.percentile(prior_max_counts_r2, 97.5)) / np.percentile(prior_max_counts_r1, 97.5) * 100):.1f}%")

print(f"\nExtreme Counts (>10,000):")
pct_extreme_r1 = 100 * np.sum(all_counts_r1 > 10000) / len(all_counts_r1)
pct_extreme_r2 = 100 * np.sum(all_counts_r2 > 10000) / len(all_counts_r2)
print(f"  Round 1: {pct_extreme_r1:.4f}%")
print(f"  Round 2: {pct_extreme_r2:.4f}%")
print(f"  Reduction: {((pct_extreme_r1 - pct_extreme_r2) / pct_extreme_r1 * 100):.1f}%")

print("\n" + "="*80)
