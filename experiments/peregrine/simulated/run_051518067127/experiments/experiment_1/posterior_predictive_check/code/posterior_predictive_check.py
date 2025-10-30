"""
Comprehensive Posterior Predictive Checks for Experiment 1
Negative Binomial GLM with Quadratic Trend

This script performs systematic model validation by comparing:
- Observed data with posterior predictive samples
- Test statistics for key data features
- Temporal patterns and autocorrelation structure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.facecolor'] = 'white'

# File paths
DATA_PATH = "/workspace/data/data.csv"
MODEL_PATH = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
OUTPUT_DIR = "/workspace/experiments/experiment_1/posterior_predictive_check"

print("=" * 80)
print("POSTERIOR PREDICTIVE CHECKS - EXPERIMENT 1")
print("Negative Binomial GLM with Quadratic Trend")
print("=" * 80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_acf(x, nlags=10):
    """Compute autocorrelation function manually."""
    x = np.asarray(x).squeeze()
    x = x - np.mean(x)
    c0 = np.dot(x, x) / len(x)

    acf_vals = np.ones(nlags + 1)
    for k in range(1, nlags + 1):
        c_k = np.dot(x[:-k], x[k:]) / len(x)
        acf_vals[k] = c_k / c0

    return acf_vals

def plot_acf_manual(x, lags=10, ax=None, color='blue', alpha=0.7, title=None):
    """Plot ACF manually."""
    if ax is None:
        ax = plt.gca()

    acf_vals = compute_acf(x, nlags=lags)

    # Plot stems
    ax.vlines(range(len(acf_vals)), [0], acf_vals, colors=color, alpha=alpha, linewidth=2)
    ax.plot(range(len(acf_vals)), acf_vals, 'o', color=color, markersize=5, alpha=alpha)

    # Add confidence bands
    n = len(x)
    conf_int = 1.96 / np.sqrt(n)
    ax.axhline(conf_int, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(-conf_int, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)

    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    if title:
        ax.set_title(title)
    ax.set_xlim(-0.5, lags + 0.5)
    ax.set_ylim(-1, 1)

    return acf_vals

# ============================================================================
# 1. LOAD DATA AND POSTERIOR SAMPLES
# ============================================================================

print("\n[1] Loading data and posterior samples...")

# Load observed data
data = pd.read_csv(DATA_PATH)
y_obs = data['C'].values
year = data['year'].values
N = len(y_obs)

print(f"   Observed data: N={N} observations")
print(f"   Count range: [{y_obs.min()}, {y_obs.max()}]")
print(f"   Mean: {y_obs.mean():.2f}, Variance: {y_obs.var():.2f}")

# Load posterior samples
idata = az.from_netcdf(MODEL_PATH)
print(f"   Posterior samples loaded: {idata.posterior.dims}")

# Extract posterior parameters
beta_0_samples = idata.posterior['beta_0'].values.flatten()
beta_1_samples = idata.posterior['beta_1'].values.flatten()
beta_2_samples = idata.posterior['beta_2'].values.flatten()
phi_samples = idata.posterior['phi'].values.flatten()

n_samples = len(beta_0_samples)
print(f"   Total posterior draws: {n_samples}")

# ============================================================================
# 2. GENERATE POSTERIOR PREDICTIVE SAMPLES
# ============================================================================

print("\n[2] Generating posterior predictive samples...")

# Use 1000 random draws for PPC
n_ppc = min(1000, n_samples)
indices = np.random.choice(n_samples, n_ppc, replace=False)

# Generate posterior predictive samples
y_rep = np.zeros((n_ppc, N))

for i, idx in enumerate(indices):
    # Compute expected value for this parameter draw
    log_mu = beta_0_samples[idx] + beta_1_samples[idx] * year + beta_2_samples[idx] * year**2
    mu = np.exp(log_mu)
    phi = phi_samples[idx]

    # Sample from Negative Binomial (NB2 parameterization)
    # Variance = mu + mu^2/phi
    # Convert to scipy's parameterization: n, p
    # n = phi (concentration)
    # p = phi / (phi + mu) (success probability)
    n = phi
    p = phi / (phi + mu)

    # Generate replicated data
    y_rep[i, :] = stats.nbinom.rvs(n=n, p=p, size=N)

    if (i + 1) % 200 == 0:
        print(f"   Generated {i+1}/{n_ppc} replications...")

print(f"   Posterior predictive samples: {y_rep.shape}")
print(f"   Replicate range: [{y_rep.min():.0f}, {y_rep.max():.0f}]")

# ============================================================================
# 3. COMPUTE TEST STATISTICS
# ============================================================================

print("\n[3] Computing test statistics...")

def compute_test_statistics(y):
    """Compute suite of test statistics for a count time series."""
    stats_dict = {}

    # T1: Lag-1 autocorrelation (CRITICAL TEST)
    if len(y) > 1:
        acf_vals = compute_acf(y, nlags=1)
        stats_dict['acf_lag1'] = acf_vals[1]
    else:
        stats_dict['acf_lag1'] = np.nan

    # T2: Variance-to-mean ratio (overdispersion)
    stats_dict['var_mean_ratio'] = np.var(y) / (np.mean(y) + 1e-10)

    # T3: Max consecutive increases (regime shift proxy)
    diffs = np.diff(y)
    consecutive_increases = 0
    max_consecutive = 0
    for d in diffs:
        if d > 0:
            consecutive_increases += 1
            max_consecutive = max(max_consecutive, consecutive_increases)
        else:
            consecutive_increases = 0
    stats_dict['max_consecutive_increases'] = max_consecutive

    # T4: Range
    stats_dict['range'] = np.max(y) - np.min(y)

    # T5: Mean
    stats_dict['mean'] = np.mean(y)

    # T6: Variance
    stats_dict['variance'] = np.var(y)

    # T7: Max value
    stats_dict['max'] = np.max(y)

    # T8: Min value
    stats_dict['min'] = np.min(y)

    return stats_dict

# Compute for observed data
test_stats_obs = compute_test_statistics(y_obs)
print("\n   Test Statistics - Observed Data:")
for stat_name, stat_val in test_stats_obs.items():
    print(f"   {stat_name}: {stat_val:.4f}")

# Compute for all replications
test_stats_rep = {key: [] for key in test_stats_obs.keys()}

for i in range(n_ppc):
    stats_i = compute_test_statistics(y_rep[i, :])
    for key, val in stats_i.items():
        test_stats_rep[key].append(val)

# Convert to arrays
for key in test_stats_rep.keys():
    test_stats_rep[key] = np.array(test_stats_rep[key])

# Compute Bayesian p-values
print("\n   Bayesian p-values (P(T_rep >= T_obs)):")
bayesian_pvals = {}
for key in test_stats_obs.keys():
    pval = np.mean(test_stats_rep[key] >= test_stats_obs[key])
    bayesian_pvals[key] = pval
    flag = "***" if pval < 0.05 or pval > 0.95 else ""
    print(f"   {key}: {pval:.4f} {flag}")

# ============================================================================
# 4. VISUALIZATION 1: DISTRIBUTIONAL CHECKS
# ============================================================================

print("\n[4] Creating distributional check plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A. Density overlay
ax = axes[0, 0]
# Plot replicate densities (sample 100 for visualization)
for i in np.random.choice(n_ppc, 100, replace=False):
    kde = stats.gaussian_kde(y_rep[i, :])
    x_range = np.linspace(0, max(y_obs.max(), y_rep.max()), 200)
    ax.plot(x_range, kde(x_range), color='lightblue', alpha=0.1, linewidth=0.5)

# Plot observed density
kde_obs = stats.gaussian_kde(y_obs)
x_range = np.linspace(0, max(y_obs.max(), y_rep.max()), 200)
ax.plot(x_range, kde_obs(x_range), color='red', linewidth=2, label='Observed', zorder=100)
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('A. Marginal Distribution Check', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# B. Q-Q plot: observed quantiles vs predictive quantiles
ax = axes[0, 1]
quantiles = np.linspace(0.01, 0.99, 50)
obs_quantiles = np.quantile(y_obs, quantiles)
# Pool all replicated data for marginal Q-Q comparison
rep_quantiles = np.quantile(y_rep.flatten(), quantiles)

ax.plot([0, 300], [0, 300], 'k--', linewidth=1, alpha=0.5, label='Perfect fit')
ax.plot(rep_quantiles, obs_quantiles, 'bo-', linewidth=2, markersize=4, label='Observed')
ax.set_xlabel('Predicted Quantiles', fontsize=12)
ax.set_ylabel('Observed Quantiles', fontsize=12)
ax.set_title('B. Quantile-Quantile Plot', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# C. Histogram comparison
ax = axes[1, 0]
# Histogram of all replicated values
all_rep = y_rep.flatten()
bins = np.arange(0, max(y_obs.max(), all_rep.max()) + 10, 10)
ax.hist(all_rep, bins=bins, alpha=0.5, color='lightblue',
        density=True, label='Replicated (pooled)', edgecolor='blue')
ax.hist(y_obs, bins=bins, alpha=0.7, color='red',
        density=True, label='Observed', edgecolor='darkred')
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('C. Histogram Comparison', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# D. ECDF comparison
ax = axes[1, 1]
# Plot replicate ECDFs (sample 100)
for i in np.random.choice(n_ppc, 100, replace=False):
    sorted_rep = np.sort(y_rep[i, :])
    ecdf_rep = np.arange(1, N+1) / N
    ax.step(sorted_rep, ecdf_rep, color='lightblue', alpha=0.1, linewidth=0.5, where='post')

# Plot observed ECDF
sorted_obs = np.sort(y_obs)
ecdf_obs = np.arange(1, N+1) / N
ax.step(sorted_obs, ecdf_obs, color='red', linewidth=2, label='Observed', where='post')
ax.set_xlabel('Count', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('D. Empirical CDF Comparison', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/distributional_checks.png", dpi=300, bbox_inches='tight')
print(f"   Saved: distributional_checks.png")
plt.close()

# ============================================================================
# 5. VISUALIZATION 2: TEMPORAL PATTERN CHECKS
# ============================================================================

print("\n[5] Creating temporal pattern check plots...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# A. Time series with predictive bands
ax = axes[0]

# Compute predictive intervals
y_rep_median = np.median(y_rep, axis=0)
y_rep_lower_50 = np.percentile(y_rep, 25, axis=0)
y_rep_upper_50 = np.percentile(y_rep, 75, axis=0)
y_rep_lower_90 = np.percentile(y_rep, 5, axis=0)
y_rep_upper_90 = np.percentile(y_rep, 95, axis=0)

# Plot predictive intervals
ax.fill_between(year, y_rep_lower_90, y_rep_upper_90,
                color='lightblue', alpha=0.3, label='90% Predictive Interval')
ax.fill_between(year, y_rep_lower_50, y_rep_upper_50,
                color='blue', alpha=0.3, label='50% Predictive Interval')
ax.plot(year, y_rep_median, 'b-', linewidth=2, label='Posterior Predictive Median')

# Plot observed data
ax.plot(year, y_obs, 'ro-', linewidth=2, markersize=6,
        label='Observed Data', zorder=100, markerfacecolor='red', markeredgecolor='darkred')

ax.set_xlabel('Year (standardized)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('A. Temporal Pattern: Observed vs Posterior Predictive',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# B. Sample of replicated time series
ax = axes[1]

# Plot 50 random replications in background
for i in np.random.choice(n_ppc, 50, replace=False):
    ax.plot(year, y_rep[i, :], color='lightblue', alpha=0.2, linewidth=0.5)

# Plot observed data on top
ax.plot(year, y_obs, 'ro-', linewidth=2, markersize=6,
        label='Observed Data', zorder=100, markerfacecolor='red', markeredgecolor='darkred')

ax.set_xlabel('Year (standardized)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('B. Observed Data vs Sample of Replications',
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/temporal_checks.png", dpi=300, bbox_inches='tight')
print(f"   Saved: temporal_checks.png")
plt.close()

# ============================================================================
# 6. VISUALIZATION 3: TEST STATISTICS
# ============================================================================

print("\n[6] Creating test statistics plots...")

# Select key test statistics for visualization
key_stats = ['acf_lag1', 'var_mean_ratio', 'max_consecutive_increases', 'range']
stat_labels = {
    'acf_lag1': 'Lag-1 Autocorrelation',
    'var_mean_ratio': 'Variance-to-Mean Ratio',
    'max_consecutive_increases': 'Max Consecutive Increases',
    'range': 'Range (Max - Min)'
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, stat in enumerate(key_stats):
    ax = axes[idx]

    # Histogram of replicated statistics
    ax.hist(test_stats_rep[stat], bins=30, alpha=0.6, color='lightblue',
            edgecolor='blue', density=True, label='Replicated')

    # Mark observed statistic
    ax.axvline(test_stats_obs[stat], color='red', linewidth=3,
               linestyle='--', label='Observed', zorder=100)

    # Add Bayesian p-value
    pval = bayesian_pvals[stat]
    pval_text = f'p-value = {pval:.3f}'
    if pval < 0.05 or pval > 0.95:
        pval_text += ' ***'
        color = 'red'
    else:
        color = 'black'

    ax.text(0.05, 0.95, pval_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            color=color, fontweight='bold')

    ax.set_xlabel(stat_labels[stat], fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{chr(65+idx)}. {stat_labels[stat]}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/test_statistics.png", dpi=300, bbox_inches='tight')
print(f"   Saved: test_statistics.png")
plt.close()

# ============================================================================
# 7. VISUALIZATION 4: AUTOCORRELATION CHECK (CRITICAL)
# ============================================================================

print("\n[7] Creating autocorrelation check plots (CRITICAL)...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A. Distribution of lag-1 ACF from replicates
ax = axes[0, 0]
ax.hist(test_stats_rep['acf_lag1'], bins=30, alpha=0.6, color='lightblue',
        edgecolor='blue', density=True, label='Replicated ACF(1)')
ax.axvline(test_stats_obs['acf_lag1'], color='red', linewidth=3,
           linestyle='--', label=f"Observed ACF(1) = {test_stats_obs['acf_lag1']:.3f}")
ax.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
ax.set_xlabel('Lag-1 Autocorrelation', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('A. Lag-1 Autocorrelation: Observed vs Replicated',
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

pval_acf = bayesian_pvals['acf_lag1']
ax.text(0.05, 0.50, f'Bayesian p-value = {pval_acf:.4f}\n*** EXTREME ***',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
        color='darkred', fontweight='bold')

# B. ACF plot for observed data
ax = axes[0, 1]
acf_obs = plot_acf_manual(y_obs, lags=10, ax=ax, color='red', alpha=0.7,
                          title='B. ACF of Observed Data')
ax.set_title('B. ACF of Observed Data', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# C. ACF plot for a typical replicate
ax = axes[1, 0]
rep_idx = np.random.choice(n_ppc)
acf_rep = plot_acf_manual(y_rep[rep_idx, :], lags=10, ax=ax, color='blue', alpha=0.7,
                          title='C. ACF of Typical Replicate (Independent)')
ax.set_title('C. ACF of Typical Replicate (Independent)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# D. Overlay multiple replicate ACFs
ax = axes[1, 1]
max_lags = 10
for i in np.random.choice(n_ppc, 50, replace=False):
    acf_vals = compute_acf(y_rep[i, :], nlags=max_lags)
    ax.plot(range(max_lags+1), acf_vals, color='lightblue', alpha=0.2, linewidth=0.5)

# Overlay observed ACF
acf_obs_full = compute_acf(y_obs, nlags=max_lags)
ax.plot(range(max_lags+1), acf_obs_full, 'ro-', linewidth=2, markersize=6,
        label='Observed', zorder=100)

# Confidence band for white noise
ax.axhline(1.96/np.sqrt(N), color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(-1.96/np.sqrt(N), color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(0, color='black', linewidth=1, alpha=0.5)

ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)
ax.set_title('D. ACF Comparison: Observed vs Replicates',
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, max_lags + 0.5)
ax.set_ylim(-0.2, 1.0)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/autocorrelation_check.png", dpi=300, bbox_inches='tight')
print(f"   Saved: autocorrelation_check.png")
plt.close()

# ============================================================================
# 8. VISUALIZATION 5: RESIDUAL DIAGNOSTICS
# ============================================================================

print("\n[8] Creating residual diagnostics plots...")

# Compute quantile residuals for observed data
# For each observation, compute its predictive quantile
quantile_residuals = np.zeros(N)

for t in range(N):
    # Get predictive samples for this time point
    y_pred_t = y_rep[:, t]
    # Compute empirical CDF at observed value
    quantile = np.mean(y_pred_t <= y_obs[t])
    # Convert to standard normal quantile
    quantile_residuals[t] = stats.norm.ppf(np.clip(quantile, 0.001, 0.999))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A. Residuals over time
ax = axes[0, 0]
ax.plot(year, quantile_residuals, 'o-', color='blue', markersize=6, linewidth=1.5)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.axhline(1.96, color='red', linestyle='--', linewidth=1, alpha=0.5, label='95% CI')
ax.axhline(-1.96, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Year (standardized)', fontsize=12)
ax.set_ylabel('Quantile Residual', fontsize=12)
ax.set_title('A. Quantile Residuals Over Time', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# B. Residual histogram vs N(0,1)
ax = axes[0, 1]
ax.hist(quantile_residuals, bins=15, alpha=0.6, color='lightblue',
        edgecolor='blue', density=True, label='Residuals')
x_norm = np.linspace(-3, 3, 100)
ax.plot(x_norm, stats.norm.pdf(x_norm), 'r-', linewidth=2, label='N(0,1)')
ax.set_xlabel('Quantile Residual', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('B. Residual Distribution vs N(0,1)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# C. Q-Q plot of residuals
ax = axes[1, 0]
stats.probplot(quantile_residuals, dist="norm", plot=ax)
ax.set_title('C. Q-Q Plot of Quantile Residuals', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# D. ACF of residuals
ax = axes[1, 1]
acf_resid = plot_acf_manual(quantile_residuals, lags=10, ax=ax, color='blue', alpha=0.7,
                            title='D. ACF of Quantile Residuals')
ax.set_title('D. ACF of Quantile Residuals', fontsize=13, fontweight='bold')
ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)
ax.grid(True, alpha=0.3)

# Add note about ACF
acf_resid_lag1 = acf_resid[1]
ax.text(0.05, 0.95, f'Lag-1 ACF = {acf_resid_lag1:.3f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/residual_diagnostics.png", dpi=300, bbox_inches='tight')
print(f"   Saved: residual_diagnostics.png")
plt.close()

# ============================================================================
# 9. SUMMARY STATISTICS TABLE
# ============================================================================

print("\n[9] Generating summary statistics...")

summary_stats = pd.DataFrame({
    'Statistic': list(test_stats_obs.keys()),
    'Observed': [test_stats_obs[k] for k in test_stats_obs.keys()],
    'Replicated_Mean': [np.mean(test_stats_rep[k]) for k in test_stats_obs.keys()],
    'Replicated_SD': [np.std(test_stats_rep[k]) for k in test_stats_obs.keys()],
    'Bayesian_p_value': [bayesian_pvals[k] for k in test_stats_obs.keys()],
    'Extreme': ['***' if bayesian_pvals[k] < 0.05 or bayesian_pvals[k] > 0.95 else ''
                for k in test_stats_obs.keys()]
})

summary_stats.to_csv(f"{OUTPUT_DIR}/code/test_statistics_summary.csv", index=False)
print("\n   Test Statistics Summary:")
print(summary_stats.to_string(index=False))

# ============================================================================
# 10. OVERALL ASSESSMENT
# ============================================================================

print("\n" + "=" * 80)
print("POSTERIOR PREDICTIVE CHECK ASSESSMENT")
print("=" * 80)

# Count failures
n_failures = sum(1 for p in bayesian_pvals.values() if p < 0.05 or p > 0.95)
n_tests = len(bayesian_pvals)

print(f"\nTest Statistics: {n_failures}/{n_tests} extreme (p < 0.05 or p > 0.95)")

# Check specific critical failures
critical_failures = []
if bayesian_pvals['acf_lag1'] < 0.05 or bayesian_pvals['acf_lag1'] > 0.95:
    critical_failures.append("ACF lag-1 (temporal structure)")

print("\nCRITICAL FINDINGS:")
if critical_failures:
    for failure in critical_failures:
        print(f"  - FAIL: {failure}")
else:
    print("  - No critical failures detected")

print("\nPASS CRITERIA:")
passes = []
if 0.05 <= bayesian_pvals['mean'] <= 0.95:
    passes.append("Central tendency (mean)")
if 0.05 <= bayesian_pvals['variance'] <= 0.95:
    passes.append("Dispersion (variance)")
if 0.05 <= bayesian_pvals['range'] <= 0.95:
    passes.append("Range of values")

for item in passes:
    print(f"  - PASS: {item}")

# Overall verdict
print("\nOVERALL VERDICT:")
if bayesian_pvals['acf_lag1'] < 0.05 or bayesian_pvals['acf_lag1'] > 0.95:
    verdict = "FAIL - Model does not capture temporal autocorrelation structure"
    print(f"  {verdict}")
    print("\n  The model generates independent observations, but the observed data")
    print("  exhibits strong temporal autocorrelation (ACF lag-1 = {:.3f}).".format(test_stats_obs['acf_lag1']))
    print("  This is an expected limitation of the independence assumption.")
else:
    verdict = "PASS - Model adequately captures key data features"
    print(f"  {verdict}")

print("\nRECOMMENDATIONS:")
if bayesian_pvals['acf_lag1'] < 0.05 or bayesian_pvals['acf_lag1'] > 0.95:
    print("  1. Consider adding autoregressive structure (Experiment 2)")
    print("  2. Model provides good baseline but misses temporal dependencies")
    print("  3. Useful for mean trend estimation but not for time series forecasting")

print("\n" + "=" * 80)
print("Posterior predictive checks complete!")
print("All outputs saved to:", OUTPUT_DIR)
print("=" * 80)
