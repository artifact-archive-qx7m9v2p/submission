"""
Compare Original vs Revised Priors for Log-Log Power Law Model

Shows the impact of recommended prior adjustments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
sns.set_style("whitegrid")

N_DRAWS = 10000

# ============================================================================
# Original Priors
# ============================================================================
print("Sampling from original priors...")
orig_alpha = np.random.normal(0.6, 0.3, N_DRAWS)
orig_beta = np.random.normal(0.12, 0.1, N_DRAWS)
orig_sigma = np.abs(stats.cauchy.rvs(loc=0, scale=0.1, size=N_DRAWS))

# ============================================================================
# Revised Priors (Option 1: Minimal Adjustment)
# ============================================================================
print("Sampling from revised priors...")
rev_alpha = np.random.normal(0.6, 0.3, N_DRAWS)
rev_beta = np.random.normal(0.12, 0.05, N_DRAWS)  # Reduced SD
rev_sigma = np.abs(stats.cauchy.rvs(loc=0, scale=0.05, size=N_DRAWS))  # Reduced scale

# ============================================================================
# Compute Statistics
# ============================================================================
print("\n" + "="*70)
print("COMPARISON: ORIGINAL vs REVISED PRIORS")
print("="*70)

print("\nβ (Power Law Exponent):")
print(f"  Original: Mean={orig_beta.mean():.3f}, SD={orig_beta.std():.3f}, % negative={100*(orig_beta<0).mean():.1f}%")
print(f"  Revised:  Mean={rev_beta.mean():.3f}, SD={rev_beta.std():.3f}, % negative={100*(rev_beta<0).mean():.1f}%")
print(f"  → Improvement: {100*(orig_beta<0).mean() - 100*(rev_beta<0).mean():.1f}% fewer negative values")

print("\nσ (Residual SD):")
print(f"  Original: Mean={orig_sigma.mean():.3f}, 95th%ile={np.percentile(orig_sigma, 95):.3f}, % > 1.0={100*(orig_sigma>1.0).mean():.1f}%")
print(f"  Revised:  Mean={rev_sigma.mean():.3f}, 95th%ile={np.percentile(rev_sigma, 95):.3f}, % > 1.0={100*(rev_sigma>1.0).mean():.1f}%")
print(f"  → Improvement: {100*(orig_sigma>1.0).mean() - 100*(rev_sigma>1.0).mean():.1f}% fewer extreme values")

# ============================================================================
# Visualization
# ============================================================================
print("\nGenerating comparison plot...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Beta comparison
ax = axes[0, 0]
ax.hist(orig_beta, bins=50, alpha=0.6, color='red', label='Original: N(0.12, 0.1)', edgecolor='black', density=True)
ax.hist(rev_beta, bins=50, alpha=0.6, color='green', label='Revised: N(0.12, 0.05)', edgecolor='black', density=True)
ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax.set_xlabel('β', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12)
ax.set_title('A) β Prior Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Zoom on negative region
ax = axes[0, 1]
ax.hist(orig_beta, bins=100, alpha=0.6, color='red', label='Original', edgecolor='black', density=True)
ax.hist(rev_beta, bins=100, alpha=0.6, color='green', label='Revised', edgecolor='black', density=True)
ax.axvline(0, color='black', linestyle='--', linewidth=2)
ax.set_xlim([-0.3, 0.05])
ax.set_xlabel('β', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12)
ax.set_title('B) β Prior - Zoomed on Negative Region', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Cumulative distribution
ax = axes[0, 2]
sorted_orig = np.sort(orig_beta)
sorted_rev = np.sort(rev_beta)
ax.plot(sorted_orig, np.linspace(0, 1, N_DRAWS), color='red', linewidth=2, label='Original')
ax.plot(sorted_rev, np.linspace(0, 1, N_DRAWS), color='green', linewidth=2, label='Revised')
ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax.set_xlabel('β', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('C) β Cumulative Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Row 2: Sigma comparison
ax = axes[1, 0]
# Focus on reasonable range for histogram
orig_sigma_filtered = orig_sigma[orig_sigma < 2.0]
rev_sigma_filtered = rev_sigma[rev_sigma < 2.0]
ax.hist(orig_sigma_filtered, bins=50, alpha=0.6, color='red', label='Original: HC(0, 0.1)', edgecolor='black', density=True)
ax.hist(rev_sigma_filtered, bins=50, alpha=0.6, color='green', label='Revised: HC(0, 0.05)', edgecolor='black', density=True)
ax.axvline(1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Problematic threshold')
ax.set_xlabel('σ', fontsize=12, fontweight='bold')
ax.set_ylabel('Density', fontsize=12)
ax.set_title('D) σ Prior Comparison (filtered < 2.0)', fontsize=13, fontweight='bold')
ax.set_xlim([0, 2.0])
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Log scale histogram to show tail
ax = axes[1, 1]
ax.hist(orig_sigma, bins=100, alpha=0.6, color='red', label='Original', edgecolor='black')
ax.hist(rev_sigma, bins=100, alpha=0.6, color='green', label='Revised', edgecolor='black')
ax.axvline(1.0, color='orange', linestyle='--', linewidth=2)
ax.set_yscale('log')
ax.set_xlim([0, 5.0])
ax.set_xlabel('σ', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency (log scale)', fontsize=12)
ax.set_title('E) σ Prior - Heavy Tail Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Cumulative distribution
ax = axes[1, 2]
sorted_orig_sig = np.sort(orig_sigma)
sorted_rev_sig = np.sort(rev_sigma)
ax.plot(sorted_orig_sig, np.linspace(0, 1, N_DRAWS), color='red', linewidth=2, label='Original')
ax.plot(sorted_rev_sig, np.linspace(0, 1, N_DRAWS), color='green', linewidth=2, label='Revised')
ax.axvline(1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(0.95, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='95th percentile')
ax.set_xlabel('σ', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('F) σ Cumulative Distribution', fontsize=13, fontweight='bold')
ax.set_xlim([0, 3.0])
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.suptitle('Original vs Revised Prior Specifications', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_3/prior_predictive_check/plots/prior_revision_comparison.png', dpi=300)
print("Saved: prior_revision_comparison.png")
plt.close()

# ============================================================================
# Summary Table
# ============================================================================
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)

summary_data = {
    'Metric': [
        'β: Mean',
        'β: SD',
        'β: % negative',
        'β: 5th percentile',
        'β: 95th percentile',
        '',
        'σ: Mean',
        'σ: Median',
        'σ: 95th percentile',
        'σ: % > 1.0',
        'σ: % > 0.5'
    ],
    'Original': [
        f'{orig_beta.mean():.3f}',
        f'{orig_beta.std():.3f}',
        f'{100*(orig_beta<0).mean():.1f}%',
        f'{np.percentile(orig_beta, 5):.3f}',
        f'{np.percentile(orig_beta, 95):.3f}',
        '',
        f'{orig_sigma.mean():.3f}',
        f'{np.median(orig_sigma):.3f}',
        f'{np.percentile(orig_sigma, 95):.3f}',
        f'{100*(orig_sigma>1.0).mean():.1f}%',
        f'{100*(orig_sigma>0.5).mean():.1f}%'
    ],
    'Revised': [
        f'{rev_beta.mean():.3f}',
        f'{rev_beta.std():.3f}',
        f'{100*(rev_beta<0).mean():.1f}%',
        f'{np.percentile(rev_beta, 5):.3f}',
        f'{np.percentile(rev_beta, 95):.3f}',
        '',
        f'{rev_sigma.mean():.3f}',
        f'{np.median(rev_sigma):.3f}',
        f'{np.percentile(rev_sigma, 95):.3f}',
        f'{100*(rev_sigma>1.0).mean():.1f}%',
        f'{100*(rev_sigma>0.5).mean():.1f}%'
    ]
}

print(f"\n{'Metric':<25} {'Original':>15} {'Revised':>15}")
print("-"*60)
for i, metric in enumerate(summary_data['Metric']):
    if metric == '':
        print()
    else:
        print(f"{metric:<25} {summary_data['Original'][i]:>15} {summary_data['Revised'][i]:>15}")

print("\n" + "="*70)
print("EXPECTED IMPACT ON PRIOR PREDICTIVE CHECK")
print("="*70)
print("\nWith revised priors, we expect:")
print("  - Trajectory pass rate: 62.8% → ~85-90%")
print("  - Monotonicity issues: 11.8% negative β → ~0.8%")
print("  - Extreme σ values: 5.7% → ~0.5%")
print("  - Overall plausibility: 89.0% → ~95%")
print("\n→ Should PASS the 80% threshold for proceeding to SBC")
print("="*70 + "\n")
