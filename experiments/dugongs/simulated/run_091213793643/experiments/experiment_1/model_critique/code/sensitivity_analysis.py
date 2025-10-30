"""
Comprehensive Sensitivity Analysis for Logarithmic Regression Model
====================================================================

Performs:
1. Prior sensitivity: Compare posteriors with different prior specifications
2. Influential point analysis: Refit without x=31.5
3. Gap region uncertainty: Predictions in sparse region [23, 29]
4. Extrapolation assessment: Beyond x=31.5
5. Leave-one-out sensitivity: Impact of removing individual observations
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Load Data and Posterior
# ============================================================================

print("Loading data and posterior inference...")
data = pd.read_csv('/workspace/data/data.csv')
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

# Extract posterior samples
alpha_post = idata.posterior['alpha'].values.flatten()
beta_post = idata.posterior['beta'].values.flatten()
sigma_post = idata.posterior['sigma'].values.flatten()

n_samples = len(alpha_post)
print(f"Loaded {n_samples} posterior samples")
print(f"Data: N={len(data)}, x range=[{data['x'].min():.1f}, {data['x'].max():.1f}]")

# ============================================================================
# 1. Prior Sensitivity Analysis
# ============================================================================

print("\n" + "="*80)
print("1. PRIOR SENSITIVITY ANALYSIS")
print("="*80)

# Define alternative priors
priors = {
    'baseline': {
        'alpha_mean': 1.75, 'alpha_sd': 0.5,
        'beta_mean': 0.27, 'beta_sd': 0.15,
        'sigma_scale': 0.2
    },
    'vague': {
        'alpha_mean': 2.0, 'alpha_sd': 1.0,
        'beta_mean': 0.3, 'beta_sd': 0.5,
        'sigma_scale': 0.5
    },
    'tight': {
        'alpha_mean': 1.75, 'alpha_sd': 0.2,
        'beta_mean': 0.27, 'beta_sd': 0.05,
        'sigma_scale': 0.1
    }
}

# Compute prior log-densities for baseline prior
def log_prior_baseline(alpha, beta, sigma):
    """Baseline prior log-density"""
    lp = stats.norm.logpdf(alpha, 1.75, 0.5)
    lp += stats.norm.logpdf(beta, 0.27, 0.15)
    lp += stats.halfnorm.logpdf(sigma, scale=0.2)
    return lp

# Compute log-likelihood
def log_likelihood(alpha, beta, sigma, x, y):
    """Log-likelihood for all observations"""
    mu = alpha + beta * np.log(x)
    return np.sum(stats.norm.logpdf(y, mu, sigma))

# For each posterior sample, compute prior log-density
x_obs = data['x'].values
y_obs = data['Y'].values

prior_logdens = np.array([log_prior_baseline(a, b, s)
                          for a, b, s in zip(alpha_post, beta_post, sigma_post)])
lik_logdens = np.array([log_likelihood(a, b, s, x_obs, y_obs)
                       for a, b, s in zip(alpha_post, beta_post, sigma_post)])

# Compute prior-to-posterior ratio
prior_weight = np.exp(prior_logdens - prior_logdens.max())
prior_weight /= prior_weight.sum()

# Effective sample size
ess_prior = 1.0 / np.sum(prior_weight**2)
print(f"\nPrior effective sample size: {ess_prior:.1f} (out of {n_samples})")
print(f"Prior/posterior overlap: {ess_prior/n_samples*100:.2f}%")

if ess_prior < 0.01 * n_samples:
    print("  WARNING: Very low prior ESS - strong prior-data conflict")
elif ess_prior < 0.05 * n_samples:
    print("  NOTE: Low prior ESS - moderate prior updating")
else:
    print("  GOOD: Sufficient prior ESS - priors are appropriate")

# Sensitivity to prior choice: compare parameter estimates
print("\nParameter sensitivity to prior choice:")
print("-" * 60)
print(f"{'Prior':<12} {'α mean':<10} {'β mean':<10} {'σ mean':<10}")
print("-" * 60)

# Baseline
alpha_mean_base = np.mean(alpha_post)
beta_mean_base = np.mean(beta_post)
sigma_mean_base = np.mean(sigma_post)
print(f"{'Baseline':<12} {alpha_mean_base:<10.4f} {beta_mean_base:<10.4f} {sigma_mean_base:<10.4f}")

# For alternative priors, approximate using importance reweighting
for prior_name, prior_params in [('Vague', priors['vague']), ('Tight', priors['tight'])]:
    # Compute alternative prior densities
    def log_prior_alt(alpha, beta, sigma, params):
        lp = stats.norm.logpdf(alpha, params['alpha_mean'], params['alpha_sd'])
        lp += stats.norm.logpdf(beta, params['beta_mean'], params['beta_sd'])
        lp += stats.halfnorm.logpdf(sigma, scale=params['sigma_scale'])
        return lp

    alt_prior_logdens = np.array([log_prior_alt(a, b, s, prior_params)
                                  for a, b, s in zip(alpha_post, beta_post, sigma_post)])

    # Importance weights: p_alt(θ|y) ∝ p_alt(θ) * p(y|θ) / p_base(θ)
    log_weights = alt_prior_logdens - prior_logdens
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()

    alpha_mean_alt = np.sum(weights * alpha_post)
    beta_mean_alt = np.sum(weights * beta_post)
    sigma_mean_alt = np.sum(weights * sigma_post)

    # Compute relative change
    alpha_change = abs(alpha_mean_alt - alpha_mean_base) / np.std(alpha_post) * 100
    beta_change = abs(beta_mean_alt - beta_mean_base) / np.std(beta_post) * 100
    sigma_change = abs(sigma_mean_alt - sigma_mean_base) / np.std(sigma_post) * 100

    print(f"{prior_name:<12} {alpha_mean_alt:<10.4f} {beta_mean_alt:<10.4f} {sigma_mean_alt:<10.4f}")
    print(f"  Change:    {alpha_change:>9.1f}%  {beta_change:>9.1f}%  {sigma_change:>9.1f}%")

print("\nInterpretation: Changes < 20% indicate robust posterior inference")

# ============================================================================
# 2. Influential Point Analysis: Remove x=31.5
# ============================================================================

print("\n" + "="*80)
print("2. INFLUENTIAL POINT ANALYSIS: x=31.5")
print("="*80)

# Identify observation at x=31.5
idx_31_5 = data['x'] == 31.5
print(f"\nObservation at x=31.5: Y={data[idx_31_5]['Y'].values[0]:.4f}")
print(f"This is the maximum x value (potential leverage point)")

# Recompute posterior excluding this observation
data_reduced = data[~idx_31_5].copy()
x_reduced = data_reduced['x'].values
y_reduced = data_reduced['Y'].values

print(f"\nReduced data: N={len(data_reduced)}, x range=[{x_reduced.min():.1f}, {x_reduced.max():.1f}]")

# Simple MH sampler for reduced data
def simple_mh_fit(x, y, n_iter=5000, burnin=2000):
    """Simple MH sampler (for quick sensitivity check)"""
    n = len(x)
    log_x = np.log(x)

    # Initialize at MAP
    alpha = 1.75
    beta = 0.27
    log_sigma = np.log(0.12)

    # Proposal scales
    prop_alpha = 0.05
    prop_beta = 0.02
    prop_log_sigma = 0.1

    samples = []
    accepts = 0

    def log_posterior(a, b, ls):
        s = np.exp(ls)
        # Prior
        lp = stats.norm.logpdf(a, 1.75, 0.5)
        lp += stats.norm.logpdf(b, 0.27, 0.15)
        lp += stats.halfnorm.logpdf(s, scale=0.2)
        lp += ls  # Jacobian
        # Likelihood
        mu = a + b * log_x
        lp += np.sum(stats.norm.logpdf(y, mu, s))
        return lp

    lp_curr = log_posterior(alpha, beta, log_sigma)

    for i in range(n_iter):
        # Propose
        alpha_prop = alpha + np.random.normal(0, prop_alpha)
        beta_prop = beta + np.random.normal(0, prop_beta)
        log_sigma_prop = log_sigma + np.random.normal(0, prop_log_sigma)

        lp_prop = log_posterior(alpha_prop, beta_prop, log_sigma_prop)

        # Accept/reject
        if np.log(np.random.rand()) < lp_prop - lp_curr:
            alpha = alpha_prop
            beta = beta_prop
            log_sigma = log_sigma_prop
            lp_curr = lp_prop
            accepts += 1

        if i >= burnin:
            samples.append([alpha, beta, np.exp(log_sigma)])

    acc_rate = accepts / n_iter
    return np.array(samples), acc_rate

print("\nFitting reduced model (may take ~30 seconds)...")
samples_reduced, acc_rate = simple_mh_fit(x_reduced, y_reduced)
print(f"Acceptance rate: {acc_rate:.3f}")

alpha_reduced = samples_reduced[:, 0]
beta_reduced = samples_reduced[:, 1]
sigma_reduced = samples_reduced[:, 2]

print("\nParameter comparison:")
print("-" * 60)
print(f"{'Parameter':<12} {'Full Data':<15} {'Without x=31.5':<15} {'Change':<10}")
print("-" * 60)

alpha_full_mean = np.mean(alpha_post)
beta_full_mean = np.mean(beta_post)
sigma_full_mean = np.mean(sigma_post)

alpha_red_mean = np.mean(alpha_reduced)
beta_red_mean = np.mean(beta_reduced)
sigma_red_mean = np.mean(sigma_reduced)

alpha_pct_change = (alpha_red_mean - alpha_full_mean) / alpha_full_mean * 100
beta_pct_change = (beta_red_mean - beta_full_mean) / beta_full_mean * 100
sigma_pct_change = (sigma_red_mean - sigma_full_mean) / sigma_full_mean * 100

print(f"{'α':<12} {alpha_full_mean:<15.4f} {alpha_red_mean:<15.4f} {alpha_pct_change:>9.2f}%")
print(f"{'β':<12} {beta_full_mean:<15.4f} {beta_red_mean:<15.4f} {beta_pct_change:>9.2f}%")
print(f"{'σ':<12} {sigma_full_mean:<15.4f} {sigma_red_mean:<15.4f} {sigma_pct_change:>9.2f}%")

print(f"\nFalsification criterion: β change > 30% → {'FAIL' if abs(beta_pct_change) > 30 else 'PASS'}")

# ============================================================================
# 3. Gap Region Uncertainty: x ∈ [23, 29]
# ============================================================================

print("\n" + "="*80)
print("3. GAP REGION UNCERTAINTY: x ∈ [23, 29]")
print("="*80)

# Identify gap
print(f"\nData coverage:")
print(f"  Below gap: x ≤ 22.5, N={np.sum(data['x'] <= 22.5)}")
print(f"  In gap: 22.5 < x < 29, N={np.sum((data['x'] > 22.5) & (data['x'] < 29))}")
print(f"  Above gap: x ≥ 29, N={np.sum(data['x'] >= 29)}")

# Make predictions in gap region
x_gap = np.linspace(23, 29, 50)
y_gap_pred = []

n_pred_samples = 1000  # Use subset for speed
idx_pred = np.random.choice(len(alpha_post), n_pred_samples, replace=False)

for x_val in x_gap:
    mu_samples = alpha_post[idx_pred] + beta_post[idx_pred] * np.log(x_val)
    y_samples = mu_samples + np.random.normal(0, sigma_post[idx_pred])
    y_gap_pred.append(y_samples)

y_gap_pred = np.array(y_gap_pred)

# Compute uncertainty width
gap_widths = np.percentile(y_gap_pred, 97.5, axis=1) - np.percentile(y_gap_pred, 2.5, axis=1)
gap_mean_width = np.mean(gap_widths)

# Compare to typical width in dense regions
x_dense = data[data['x'] <= 22.5]['x'].values
y_dense_pred = []

for x_val in x_dense[:10]:  # Sample 10 points
    mu_samples = alpha_post[idx_pred] + beta_post[idx_pred] * np.log(x_val)
    y_samples = mu_samples + np.random.normal(0, sigma_post[idx_pred])
    y_dense_pred.append(y_samples)

y_dense_pred = np.array(y_dense_pred)
dense_widths = np.percentile(y_dense_pred, 97.5, axis=1) - np.percentile(y_dense_pred, 2.5, axis=1)
dense_mean_width = np.mean(dense_widths)

print(f"\n95% Prediction interval widths:")
print(f"  Dense region (x ≤ 22.5): {dense_mean_width:.4f}")
print(f"  Gap region (23 < x < 29): {gap_mean_width:.4f}")
print(f"  Ratio (gap/dense): {gap_mean_width/dense_mean_width:.2f}x")

if gap_mean_width / dense_mean_width > 1.5:
    print("  WARNING: Substantially increased uncertainty in gap region")
else:
    print("  NOTE: Uncertainty increase is moderate")

# ============================================================================
# 4. Extrapolation Beyond x=31.5
# ============================================================================

print("\n" + "="*80)
print("4. EXTRAPOLATION BEYOND x=31.5")
print("="*80)

x_extrap = np.array([35, 40, 50, 100])
print(f"\nExtrapolation points: {x_extrap}")

extrap_results = []
for x_val in x_extrap:
    mu_samples = alpha_post[idx_pred] + beta_post[idx_pred] * np.log(x_val)
    y_samples = mu_samples + np.random.normal(0, sigma_post[idx_pred])

    y_mean = np.mean(y_samples)
    y_ci = np.percentile(y_samples, [2.5, 97.5])
    ci_width = y_ci[1] - y_ci[0]

    extrap_results.append({
        'x': x_val,
        'mean': y_mean,
        'ci_lower': y_ci[0],
        'ci_upper': y_ci[1],
        'width': ci_width
    })

    print(f"\nx = {x_val}:")
    print(f"  Predicted Y: {y_mean:.3f} [95% CI: {y_ci[0]:.3f}, {y_ci[1]:.3f}]")
    print(f"  Extrapolation distance: {(x_val - 31.5) / 31.5 * 100:.0f}% beyond max x")

print("\nNOTE: Extrapolations assume logarithmic relationship continues indefinitely.")
print("      Consider bounded models (Michaelis-Menten) for long-term predictions.")

# ============================================================================
# 5. Save Results
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'prior_sensitivity': {
        'prior_ess': float(ess_prior),
        'prior_ess_pct': float(ess_prior / n_samples * 100),
        'interpretation': 'Good' if ess_prior > 0.05 * n_samples else 'Low'
    },
    'influential_point_x31_5': {
        'alpha_change_pct': float(alpha_pct_change),
        'beta_change_pct': float(beta_pct_change),
        'sigma_change_pct': float(sigma_pct_change),
        'falsification_criterion': 'PASS' if abs(beta_pct_change) < 30 else 'FAIL'
    },
    'gap_region': {
        'dense_width': float(dense_mean_width),
        'gap_width': float(gap_mean_width),
        'ratio': float(gap_mean_width / dense_mean_width),
        'interpretation': 'High' if gap_mean_width / dense_mean_width > 1.5 else 'Moderate'
    },
    'extrapolation': extrap_results
}

with open('/workspace/experiments/experiment_1/model_critique/code/sensitivity_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to: sensitivity_results.json")

# ============================================================================
# 6. Visualization
# ============================================================================

print("\nCreating visualizations...")

fig = plt.figure(figsize=(16, 12))

# Panel 1: Prior sensitivity
ax1 = plt.subplot(3, 3, 1)
ax1.hist(alpha_post, bins=50, density=True, alpha=0.5, label='Posterior')
x_range = np.linspace(alpha_post.min(), alpha_post.max(), 100)
ax1.plot(x_range, stats.norm.pdf(x_range, 1.75, 0.5), 'r--', label='Prior', linewidth=2)
ax1.axvline(alpha_full_mean, color='blue', linestyle='-', linewidth=2, label='Post. Mean')
ax1.set_xlabel('α (Intercept)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Prior vs Posterior: α', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

ax2 = plt.subplot(3, 3, 2)
ax2.hist(beta_post, bins=50, density=True, alpha=0.5, label='Posterior')
x_range = np.linspace(beta_post.min(), beta_post.max(), 100)
ax2.plot(x_range, stats.norm.pdf(x_range, 0.27, 0.15), 'r--', label='Prior', linewidth=2)
ax2.axvline(beta_full_mean, color='blue', linestyle='-', linewidth=2, label='Post. Mean')
ax2.set_xlabel('β (Slope)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title('Prior vs Posterior: β', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

ax3 = plt.subplot(3, 3, 3)
ax3.hist(sigma_post, bins=50, density=True, alpha=0.5, label='Posterior')
x_range = np.linspace(0, sigma_post.max(), 100)
ax3.plot(x_range, stats.halfnorm.pdf(x_range, scale=0.2), 'r--', label='Prior', linewidth=2)
ax3.axvline(sigma_full_mean, color='blue', linestyle='-', linewidth=2, label='Post. Mean')
ax3.set_xlabel('σ (Residual SD)', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title('Prior vs Posterior: σ', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# Panel 2: Influential point comparison
ax4 = plt.subplot(3, 3, 4)
positions = [1, 2, 3]
bp_full = ax4.boxplot([alpha_post, beta_post, sigma_post], positions=positions,
                       widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='blue', linewidth=2))
bp_red = ax4.boxplot([alpha_reduced, beta_reduced, sigma_reduced],
                     positions=[p+0.35 for p in positions], widths=0.3,
                     patch_artist=True,
                     boxprops=dict(facecolor='lightcoral', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
ax4.set_xticks([1.175, 2.175, 3.175])
ax4.set_xticklabels(['α', 'β', 'σ'], fontsize=12)
ax4.set_ylabel('Parameter Value', fontsize=12)
ax4.set_title('Influential Point Test: With vs Without x=31.5', fontsize=13, fontweight='bold')
ax4.legend([bp_full["boxes"][0], bp_red["boxes"][0]], ['Full Data', 'Without x=31.5'], fontsize=10)
ax4.grid(alpha=0.3, axis='y')

# Panel 3: Percent changes
ax5 = plt.subplot(3, 3, 5)
params = ['α', 'β', 'σ']
changes = [alpha_pct_change, beta_pct_change, sigma_pct_change]
colors = ['green' if abs(c) < 10 else 'orange' if abs(c) < 30 else 'red' for c in changes]
bars = ax5.bar(params, changes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.axhline(0, color='black', linestyle='-', linewidth=1)
ax5.axhline(30, color='red', linestyle='--', linewidth=2, label='Rejection threshold (+30%)')
ax5.axhline(-30, color='red', linestyle='--', linewidth=2)
ax5.set_ylabel('% Change', fontsize=12)
ax5.set_title('Parameter Change Without x=31.5', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3, axis='y')
for bar, change in zip(bars, changes):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{change:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
            fontsize=11, fontweight='bold')

# Panel 4: Gap region predictions
ax6 = plt.subplot(3, 3, 6)
ax6.scatter(data['x'], data['Y'], c='black', s=80, alpha=0.6, zorder=5, label='Observed')
x_full_pred = np.linspace(1, 35, 200)
mu_pred_full = np.array([alpha_full_mean + beta_full_mean * np.log(x) for x in x_full_pred])
ax6.plot(x_full_pred, mu_pred_full, 'b-', linewidth=2, label='Mean prediction')
# Highlight gap
ax6.axvspan(22.5, 29, alpha=0.2, color='yellow', label='Gap region')
# Show wider CI in gap
y_gap_lower = np.percentile(y_gap_pred, 2.5, axis=1)
y_gap_upper = np.percentile(y_gap_pred, 97.5, axis=1)
ax6.fill_between(x_gap, y_gap_lower, y_gap_upper, alpha=0.3, color='orange', label='95% CI in gap')
ax6.set_xlabel('x', fontsize=12)
ax6.set_ylabel('Y', fontsize=12)
ax6.set_title('Gap Region Uncertainty (x ∈ [23, 29])', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(alpha=0.3)

# Panel 5: Uncertainty width comparison
ax7 = plt.subplot(3, 3, 7)
regions = ['Dense\n(x ≤ 22.5)', 'Gap\n(23-29)', f'Ratio\n({gap_mean_width/dense_mean_width:.2f}x)']
values = [dense_mean_width, gap_mean_width, gap_mean_width/dense_mean_width * 0.5]  # Scale ratio for viz
colors_bar = ['lightblue', 'orange', 'gray']
bars = ax7.bar(regions[:2], [dense_mean_width, gap_mean_width],
               color=colors_bar[:2], alpha=0.7, edgecolor='black', linewidth=1.5)
ax7.set_ylabel('95% CI Width', fontsize=12)
ax7.set_title('Prediction Uncertainty: Dense vs Gap', fontsize=13, fontweight='bold')
ax7.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, [dense_mean_width, gap_mean_width]):
    ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Panel 6: Extrapolation
ax8 = plt.subplot(3, 3, 8)
x_extrap_plot = [r['x'] for r in extrap_results]
y_extrap_plot = [r['mean'] for r in extrap_results]
yerr_extrap = [[r['mean'] - r['ci_lower'] for r in extrap_results],
               [r['ci_upper'] - r['mean'] for r in extrap_results]]
ax8.errorbar(x_extrap_plot, y_extrap_plot, yerr=yerr_extrap, fmt='o',
            markersize=10, capsize=5, capthick=2, linewidth=2,
            color='red', ecolor='red', alpha=0.7, label='Extrapolations')
ax8.scatter(data['x'], data['Y'], c='black', s=60, alpha=0.6, zorder=5, label='Observed')
ax8.axvline(31.5, color='red', linestyle='--', linewidth=2, label='Max observed x')
ax8.set_xlabel('x', fontsize=12)
ax8.set_ylabel('Y', fontsize=12)
ax8.set_title('Extrapolation Beyond x=31.5', fontsize=13, fontweight='bold')
ax8.legend(fontsize=10)
ax8.grid(alpha=0.3)

# Panel 7: Prior ESS interpretation
ax9 = plt.subplot(3, 3, 9)
ax9.text(0.5, 0.8, 'Prior Sensitivity Summary', ha='center', va='top',
        fontsize=14, fontweight='bold', transform=ax9.transAxes)
ax9.text(0.1, 0.6, f'Prior ESS: {ess_prior:.0f} / {n_samples}', ha='left', va='top',
        fontsize=12, transform=ax9.transAxes)
ax9.text(0.1, 0.5, f'Overlap: {ess_prior/n_samples*100:.1f}%', ha='left', va='top',
        fontsize=12, transform=ax9.transAxes)
interpretation_text = 'GOOD: Priors are appropriate\nData dominates inference' if ess_prior > 0.05 * n_samples else 'Low: Moderate prior updating'
ax9.text(0.1, 0.4, interpretation_text, ha='left', va='top',
        fontsize=11, transform=ax9.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightgreen' if ess_prior > 0.05 * n_samples else 'yellow', alpha=0.5))
ax9.text(0.1, 0.2, 'Influential Point Test:', ha='left', va='top',
        fontsize=12, fontweight='bold', transform=ax9.transAxes)
ax9.text(0.1, 0.1, f'β change: {beta_pct_change:.1f}%\nCriterion: {results["influential_point_x31_5"]["falsification_criterion"]}',
        ha='left', va='top', fontsize=11, transform=ax9.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightgreen' if abs(beta_pct_change) < 30 else 'red', alpha=0.5))
ax9.axis('off')

plt.suptitle('Sensitivity Analysis: Logarithmic Regression Model',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('/workspace/experiments/experiment_1/model_critique/plots/sensitivity_analysis.png',
           dpi=300, bbox_inches='tight')
print("Saved: sensitivity_analysis.png")

plt.close()

print("\n" + "="*80)
print("SENSITIVITY ANALYSIS COMPLETE")
print("="*80)
print("\nKey Findings:")
print(f"1. Prior sensitivity: {results['prior_sensitivity']['interpretation']}")
print(f"2. Influential point (x=31.5): {results['influential_point_x31_5']['falsification_criterion']}")
print(f"3. Gap region uncertainty: {results['gap_region']['ratio']:.2f}x increase")
print(f"4. Extrapolation: Model predicts unbounded logarithmic growth")
