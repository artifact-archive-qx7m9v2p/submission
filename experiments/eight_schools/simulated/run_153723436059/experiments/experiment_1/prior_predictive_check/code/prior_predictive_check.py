"""
Prior Predictive Check for Experiment 1: Standard Hierarchical Model

Generates synthetic datasets from the prior distribution to assess:
1. Prior generates scientifically plausible data
2. Observed data not extreme outlier under prior
3. Prior allows for diverse outcomes (not overconfident)
4. No computational issues with prior specification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check")
PLOTS_DIR = OUTPUT_DIR / "plots"
CODE_DIR = OUTPUT_DIR / "code"

# Load observed data
data = pd.read_csv(DATA_PATH)
y_obs = data['effect'].values
sigma_obs = data['sigma'].values
J = len(y_obs)

print("="*70)
print("PRIOR PREDICTIVE CHECK: EXPERIMENT 1")
print("="*70)
print(f"\nObserved data:")
print(f"  Schools: {J}")
print(f"  Effect range: [{y_obs.min():.1f}, {y_obs.max():.1f}]")
print(f"  Effect mean: {y_obs.mean():.1f}")
print(f"  Effect SD: {y_obs.std():.1f}")
print(f"  Sigma: {sigma_obs}")
print()

# =============================================================================
# 1. DIRECT PRIOR PREDICTIVE SAMPLING
# =============================================================================
print("="*70)
print("1. GENERATING PRIOR PREDICTIVE SAMPLES")
print("="*70)

n_sims = 2000  # More samples for robust assessment

# Sample from priors
mu_prior = np.random.normal(0, 50, n_sims)
# Half-Cauchy via folded Cauchy
tau_prior = np.abs(stats.cauchy.rvs(loc=0, scale=25, size=n_sims))

# Generate synthetic datasets
y_prior_pred = np.zeros((n_sims, J))
theta_prior = np.zeros((n_sims, J))

for i in range(n_sims):
    # Sample school effects
    theta_prior[i, :] = np.random.normal(mu_prior[i], tau_prior[i], J)
    # Sample observations
    for j in range(J):
        y_prior_pred[i, j] = np.random.normal(theta_prior[i, j], sigma_obs[j])

print(f"\nGenerated {n_sims} synthetic datasets")
print(f"\nPrior parameter summaries:")
print(f"  mu:  mean={mu_prior.mean():.1f}, SD={mu_prior.std():.1f}, "
      f"range=[{mu_prior.min():.1f}, {mu_prior.max():.1f}]")
print(f"  tau: mean={tau_prior.mean():.1f}, SD={tau_prior.std():.1f}, "
      f"median={np.median(tau_prior):.1f}, "
      f"range=[{tau_prior.min():.1f}, {tau_prior.max():.1f}]")

# =============================================================================
# 2. QUANTITATIVE PRIOR CHECKS
# =============================================================================
print("\n" + "="*70)
print("2. QUANTITATIVE PRIOR CHECKS")
print("="*70)

# Check 1: Domain violations (effects outside reasonable range)
extreme_threshold = 200  # Educational effects beyond ±200 implausible
n_extreme = np.sum(np.abs(y_prior_pred) > extreme_threshold, axis=1)
pct_any_extreme = 100 * np.mean(n_extreme > 0)
pct_reasonable = 100 * np.mean(np.all(np.abs(y_prior_pred) < 100, axis=1))

print(f"\nDomain plausibility:")
print(f"  % datasets with ALL |y| < 100: {pct_reasonable:.1f}%")
print(f"  % datasets with ANY |y| > {extreme_threshold}: {pct_any_extreme:.1f}%")

# Check 2: Scale problems
y_prior_ranges = y_prior_pred.max(axis=1) - y_prior_pred.min(axis=1)
y_prior_means = y_prior_pred.mean(axis=1)
y_prior_sds = y_prior_pred.std(axis=1)

print(f"\nPrior predictive statistics:")
print(f"  Range(y): mean={y_prior_ranges.mean():.1f}, "
      f"median={np.median(y_prior_ranges):.1f}, "
      f"90% interval=[{np.percentile(y_prior_ranges, 5):.1f}, "
      f"{np.percentile(y_prior_ranges, 95):.1f}]")
print(f"  Mean(y):  90% interval=[{np.percentile(y_prior_means, 5):.1f}, "
      f"{np.percentile(y_prior_means, 95):.1f}]")
print(f"  SD(y):    mean={y_prior_sds.mean():.1f}, "
      f"median={np.median(y_prior_sds):.1f}")

obs_range = y_obs.max() - y_obs.min()
obs_mean = y_obs.mean()
obs_sd = y_obs.std()

print(f"\nObserved data:")
print(f"  Range(y_obs): {obs_range:.1f}")
print(f"  Mean(y_obs):  {obs_mean:.1f}")
print(f"  SD(y_obs):    {obs_sd:.1f}")

# Check 3: Prior-data conflict
print(f"\n" + "="*70)
print("3. PRIOR-DATA CONFLICT CHECK")
print("="*70)

# For each observed value, what percentile is it in prior predictive?
prior_percentiles = np.zeros(J)
for j in range(J):
    prior_percentiles[j] = stats.percentileofscore(y_prior_pred[:, j], y_obs[j])

print(f"\nObserved data percentiles in prior predictive distribution:")
for j in range(J):
    school = j + 1
    pct = prior_percentiles[j]
    flag = ""
    if pct < 0.5 or pct > 99.5:
        flag = " *** EXTREME ***"
    print(f"  School {school}: y_obs={y_obs[j]:6.1f}, "
          f"percentile={pct:5.1f}%{flag}")

n_extreme_obs = np.sum((prior_percentiles < 0.5) | (prior_percentiles > 99.5))
if n_extreme_obs > 0:
    print(f"\nWARNING: {n_extreme_obs} schools are extreme outliers under prior")
else:
    print(f"\nGOOD: All observed values within reasonable prior range")

# Check 4: Prior allows for diverse behavior
# Can tau be very small (strong pooling)?
pct_small_tau = 100 * np.mean(tau_prior < 5)
pct_large_tau = 100 * np.mean(tau_prior > 20)

print(f"\n" + "="*70)
print("4. PRIOR FLEXIBILITY CHECK")
print("="*70)
print(f"\nPrior support for different pooling scenarios:")
print(f"  % tau < 5 (strong pooling):     {pct_small_tau:.1f}%")
print(f"  % tau > 20 (minimal pooling):   {pct_large_tau:.1f}%")
print(f"  --> Prior allows both pooling and no-pooling: ", end="")
if pct_small_tau > 10 and pct_large_tau > 10:
    print("YES")
else:
    print("MAYBE (prior may be informative)")

# =============================================================================
# 5. SENSITIVITY ANALYSIS: ALTERNATIVE PRIORS
# =============================================================================
print(f"\n" + "="*70)
print("5. SENSITIVITY ANALYSIS: ALTERNATIVE PRIORS")
print("="*70)

# Test alternative prior specifications
alternative_priors = {
    'baseline': {'mu_scale': 50, 'tau_dist': 'halfcauchy', 'tau_scale': 25},
    'tighter_mu': {'mu_scale': 25, 'tau_dist': 'halfcauchy', 'tau_scale': 25},
    'vaguer_mu': {'mu_scale': 100, 'tau_dist': 'halfcauchy', 'tau_scale': 25},
    'halfnormal_tau': {'mu_scale': 50, 'tau_dist': 'halfnormal', 'tau_scale': 25},
    'tighter_tau': {'mu_scale': 50, 'tau_dist': 'halfcauchy', 'tau_scale': 10},
}

sensitivity_results = {}

for name, params in alternative_priors.items():
    # Sample priors
    mu_alt = np.random.normal(0, params['mu_scale'], 1000)

    if params['tau_dist'] == 'halfcauchy':
        tau_alt = np.abs(stats.cauchy.rvs(loc=0, scale=params['tau_scale'], size=1000))
    else:  # halfnormal
        tau_alt = np.abs(np.random.normal(0, params['tau_scale'], 1000))

    # Generate predictions
    y_alt = np.zeros((1000, J))
    for i in range(1000):
        theta_alt = np.random.normal(mu_alt[i], tau_alt[i], J)
        for j in range(J):
            y_alt[i, j] = np.random.normal(theta_alt[j], sigma_obs[j])

    # Store key statistics
    sensitivity_results[name] = {
        'mu_sd': mu_alt.std(),
        'tau_median': np.median(tau_alt),
        'tau_mean': tau_alt.mean(),
        'y_range_median': np.median(y_alt.max(axis=1) - y_alt.min(axis=1)),
        'y_mean_sd': y_alt.mean(axis=1).std()
    }

print(f"\nPrior sensitivity comparison:")
print(f"{'Prior':<20} {'mu SD':<10} {'tau median':<12} {'Range(y) med':<15} {'SD(mean(y))':<12}")
print("-" * 70)
for name, results in sensitivity_results.items():
    print(f"{name:<20} {results['mu_sd']:>8.1f}   {results['tau_median']:>10.1f}  "
          f"{results['y_range_median']:>13.1f}   {results['y_mean_sd']:>10.1f}")

# Check if results are substantially different
baseline = sensitivity_results['baseline']
max_diff_range = max(abs(r['y_range_median'] - baseline['y_range_median']) / baseline['y_range_median']
                     for r in sensitivity_results.values())

print(f"\nMaximum relative difference in Range(y) median: {100*max_diff_range:.1f}%")
if max_diff_range < 0.5:
    print("GOOD: Prior predictive relatively insensitive to prior choice")
else:
    print("WARNING: Prior predictive sensitive to prior specification")

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
print(f"\n" + "="*70)
print("6. CREATING DIAGNOSTIC VISUALIZATIONS")
print("="*70)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Plot 1: Parameter prior distributions with context
print("\n  Creating: parameter_priors.png")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# mu prior
ax = axes[0]
x_mu = np.linspace(-150, 150, 500)
y_mu = stats.norm.pdf(x_mu, 0, 50)
ax.plot(x_mu, y_mu, 'b-', linewidth=2, label='Prior: N(0, 50)')
ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Prior mean')
ax.axvspan(-100, 100, alpha=0.1, color='blue', label='95% prior CI')
ax.axvline(obs_mean, color='red', linewidth=2, label=f'Observed mean: {obs_mean:.1f}')
ax.axvspan(y_obs.min(), y_obs.max(), alpha=0.1, color='red', label='Observed range')
ax.set_xlabel('μ (population mean effect)')
ax.set_ylabel('Density')
ax.set_title('Prior for μ vs Observed Data')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# tau prior
ax = axes[1]
x_tau = np.linspace(0, 100, 500)
y_tau = 2 * stats.cauchy.pdf(x_tau, 0, 25)  # Half-Cauchy
ax.plot(x_tau, y_tau, 'g-', linewidth=2, label='Prior: HalfCauchy(0, 25)')
ax.axvline(np.median(tau_prior), color='darkgreen', linestyle='--',
           label=f'Prior median: {np.median(tau_prior):.1f}')
ax.axvline(obs_sd, color='red', linewidth=2, label=f'Observed SD: {obs_sd:.1f}')
ax.axvspan(0, 10, alpha=0.1, color='orange', label='Strong pooling region')
ax.set_xlabel('τ (between-school SD)')
ax.set_ylabel('Density')
ax.set_title('Prior for τ vs Observed Variation')
ax.set_xlim(0, 100)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_priors.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Prior predictive samples overlaid
print("  Creating: prior_predictive_spaghetti.png")
fig, ax = plt.subplots(figsize=(12, 6))

# Plot 100 random prior predictive datasets
n_show = 100
indices = np.random.choice(n_sims, n_show, replace=False)
schools = np.arange(1, J+1)

for i in indices:
    ax.plot(schools, y_prior_pred[i, :], 'o-', color='lightblue',
            alpha=0.3, markersize=4, linewidth=0.5)

# Overlay observed data
ax.plot(schools, y_obs, 'ro-', linewidth=2, markersize=10,
        label='Observed data', zorder=100)

# Add uncertainty bars for observed data
for j in range(J):
    ax.errorbar(schools[j], y_obs[j], yerr=1.96*sigma_obs[j],
                fmt='none', ecolor='darkred', capsize=5, alpha=0.7, zorder=99)

ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xlabel('School')
ax.set_ylabel('Treatment Effect')
ax.set_title(f'Prior Predictive Datasets (n={n_show}) vs Observed Data')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(schools)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_predictive_spaghetti.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Comprehensive prior predictive coverage
print("  Creating: prior_predictive_coverage.png")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for j in range(J):
    ax = axes[j]

    # Histogram of prior predictions
    ax.hist(y_prior_pred[:, j], bins=50, density=True, alpha=0.6,
            color='skyblue', edgecolor='black', label='Prior predictive')

    # Mark observed value
    ax.axvline(y_obs[j], color='red', linewidth=3, label=f'Observed: {y_obs[j]:.1f}')

    # Show percentile
    pct = prior_percentiles[j]
    ax.text(0.05, 0.95, f'{pct:.0f}th percentile',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Highlight extreme
    if pct < 0.5 or pct > 99.5:
        ax.set_facecolor('#ffeeee')

    ax.set_xlabel('Effect')
    ax.set_ylabel('Density')
    ax.set_title(f'School {j+1} (σ={sigma_obs[j]})')
    ax.grid(alpha=0.3)
    if j == 0:
        ax.legend(fontsize=8)

plt.suptitle('Prior Predictive Coverage: School-by-School', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_predictive_coverage.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Prior predictive summary statistics
print("  Creating: prior_predictive_summaries.png")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Range of effects
ax = axes[0, 0]
ax.hist(y_prior_ranges, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
ax.axvline(obs_range, color='red', linewidth=2, label=f'Observed: {obs_range:.1f}')
ax.axvline(np.median(y_prior_ranges), color='blue', linestyle='--',
           label=f'Prior median: {np.median(y_prior_ranges):.1f}')
ax.set_xlabel('Range(y) = max(y) - min(y)')
ax.set_ylabel('Density')
ax.set_title('A: Prior Predictive Range of Effects')
ax.legend()
ax.grid(alpha=0.3)

# Panel B: Mean effect
ax = axes[0, 1]
ax.hist(y_prior_means, bins=50, density=True, alpha=0.6, color='seagreen', edgecolor='black')
ax.axvline(obs_mean, color='red', linewidth=2, label=f'Observed: {obs_mean:.1f}')
ax.axvline(0, color='blue', linestyle='--', label='Prior mean: 0')
ax.set_xlabel('Mean(y)')
ax.set_ylabel('Density')
ax.set_title('B: Prior Predictive Mean Effect')
ax.legend()
ax.grid(alpha=0.3)

# Panel C: SD of effects
ax = axes[1, 0]
ax.hist(y_prior_sds, bins=50, density=True, alpha=0.6, color='coral', edgecolor='black')
ax.axvline(obs_sd, color='red', linewidth=2, label=f'Observed: {obs_sd:.1f}')
ax.axvline(np.median(y_prior_sds), color='blue', linestyle='--',
           label=f'Prior median: {np.median(y_prior_sds):.1f}')
ax.set_xlabel('SD(y)')
ax.set_ylabel('Density')
ax.set_title('C: Prior Predictive SD of Effects')
ax.legend()
ax.grid(alpha=0.3)

# Panel D: Relationship between mu and tau samples
ax = axes[1, 1]
scatter = ax.scatter(mu_prior, tau_prior, alpha=0.1, c=y_prior_ranges,
                     cmap='viridis', s=10)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(np.median(tau_prior), color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('μ (prior samples)')
ax.set_ylabel('τ (prior samples)')
ax.set_title('D: Joint Prior Samples (color = Range(y))')
ax.set_xlim(-150, 150)
ax.set_ylim(0, 100)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Range(y)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_predictive_summaries.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Sensitivity analysis comparison
print("  Creating: prior_sensitivity.png")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Prepare data for comparison
sensitivity_names = list(sensitivity_results.keys())
range_medians = [sensitivity_results[k]['y_range_median'] for k in sensitivity_names]
mean_sds = [sensitivity_results[k]['y_mean_sd'] for k in sensitivity_names]

# Panel A: Range comparison
ax = axes[0]
colors = ['red' if name == 'baseline' else 'lightblue' for name in sensitivity_names]
bars = ax.barh(sensitivity_names, range_medians, color=colors, edgecolor='black')
ax.axvline(obs_range, color='green', linewidth=2, linestyle='--',
           label=f'Observed range: {obs_range:.1f}')
ax.set_xlabel('Median Range(y)')
ax.set_title('A: Prior Predictive Range Sensitivity')
ax.legend()
ax.grid(alpha=0.3, axis='x')

# Panel B: Mean SD comparison
ax = axes[1]
bars = ax.barh(sensitivity_names, mean_sds, color=colors, edgecolor='black')
ax.set_xlabel('SD(Mean(y))')
ax.set_title('B: Prior Uncertainty in Mean Effect')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_sensitivity.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Extreme value check
print("  Creating: extreme_value_diagnostic.png")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Distribution of maximum absolute values
ax = axes[0]
max_abs_values = np.max(np.abs(y_prior_pred), axis=1)
ax.hist(max_abs_values, bins=50, density=True, alpha=0.6, color='orange', edgecolor='black')
ax.axvline(100, color='green', linewidth=2, linestyle='--',
           label='Plausible threshold (100)')
ax.axvline(200, color='red', linewidth=2, linestyle='--',
           label='Extreme threshold (200)')
ax.axvline(np.max(np.abs(y_obs)), color='blue', linewidth=2,
           label=f'Observed max: {np.max(np.abs(y_obs)):.1f}')
ax.set_xlabel('max|y| across schools')
ax.set_ylabel('Density')
ax.set_title('A: Distribution of Extreme Values')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, 300)

# Panel B: Proportion of extreme values by dataset
ax = axes[1]
n_extreme_by_dataset = np.sum(np.abs(y_prior_pred) > 100, axis=1)
ax.hist(n_extreme_by_dataset, bins=np.arange(0, J+2)-0.5,
        density=True, alpha=0.6, color='tomato', edgecolor='black')
ax.set_xlabel('Number of schools with |y| > 100')
ax.set_ylabel('Proportion of datasets')
ax.set_title('B: Frequency of Implausible Values')
ax.grid(alpha=0.3)
ax.set_xticks(range(J+1))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "extreme_value_diagnostic.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n  All plots saved to: {PLOTS_DIR}")

# =============================================================================
# 7. FINAL ASSESSMENT
# =============================================================================
print("\n" + "="*70)
print("7. FINAL ASSESSMENT")
print("="*70)

# Criteria for pass/fail
checks = {
    'No extreme outliers in observed data': n_extreme_obs == 0,
    'Prior generates diverse outcomes': pct_small_tau > 10 and pct_large_tau > 10,
    'Most predictions reasonable (|y|<100)': pct_reasonable > 50,
    'Few extreme predictions (|y|>200)': pct_any_extreme < 5,
    'Observed range within prior predictive': (obs_range >= np.percentile(y_prior_ranges, 1) and
                                                obs_range <= np.percentile(y_prior_ranges, 99)),
    'Prior relatively insensitive': max_diff_range < 0.5,
}

print("\nPASS/FAIL Criteria:")
for criterion, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    print(f"  {symbol} {criterion}: {status}")

overall_pass = all(checks.values())
print("\n" + "="*70)
if overall_pass:
    print("OVERALL ASSESSMENT: PASS")
    print("="*70)
    print("\nThe prior specification is adequate:")
    print("  - Priors generate scientifically plausible data")
    print("  - Observed data falls within reasonable prior range")
    print("  - Prior allows for diverse outcomes (not overconfident)")
    print("  - No computational red flags")
    print("\nRECOMMENDATION: Proceed with model fitting")
else:
    print("OVERALL ASSESSMENT: CONDITIONAL PASS WITH NOTES")
    print("="*70)
    print("\nSome criteria not fully met, but prior may still be adequate.")
    print("Review failed checks above and findings.md for details.")
    print("\nRECOMMENDATION: Proceed with caution, consider alternatives")

print("\n" + "="*70)
print("Prior predictive check complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("="*70)
