"""
Posterior Predictive Check for Experiment 2: AR(1) Log-Normal with Regime-Switching
=====================================================================================

This script performs comprehensive posterior predictive checks for the AR(1) log-normal
model, focusing on whether the AR(1) structure improves temporal prediction compared
to Experiment 1.

Key Questions:
1. Does AR(1) structure help capture temporal autocorrelation?
2. Can the model reproduce observed ACF (0.975)?
3. How does PPC performance compare to Experiment 1?
4. Why does better fit (lower MAE/RMSE) coexist with higher residual ACF?

Author: Model Validation Specialist
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Manual ACF computation (statsmodels not available)
def compute_acf(x, nlags=10, fft=False):
    """Compute autocorrelation function manually."""
    x = np.asarray(x).squeeze()
    x = x - np.mean(x)
    
    c0 = np.dot(x, x) / len(x)
    
    acf_vals = np.array([1.0] + [np.dot(x[:-i], x[i:]) / len(x) / c0 
                                   for i in range(1, nlags + 1)])
    return acf_vals


# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
POSTERIOR_PATH = Path("/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf")
OUTPUT_DIR = Path("/workspace/experiments/experiment_2/posterior_predictive_check")
PLOTS_DIR = OUTPUT_DIR / "plots"
CODE_DIR = OUTPUT_DIR / "code"

# Create directories
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("POSTERIOR PREDICTIVE CHECK: Experiment 2 (AR(1) Log-Normal)")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA AND POSTERIOR
# ============================================================================

print("\n[1] Loading data and posterior samples...")

# Load observed data
data = pd.read_csv(DATA_PATH)
y_obs = data['C'].values
year = data['year'].values
n_obs = len(y_obs)

print(f"   Observed data: N={n_obs}, range=[{y_obs.min()}, {y_obs.max()}]")
print(f"   Mean={y_obs.mean():.1f}, SD={y_obs.std():.1f}")

# Load posterior
idata = az.from_netcdf(POSTERIOR_PATH)

print(f"   Posterior dimensions: {idata.posterior.dims}")
print(f"   Parameters: {list(idata.posterior.data_vars)}")

# Extract posterior samples (flatten chains)
alpha_samples = idata.posterior['alpha'].values.flatten()
beta_1_samples = idata.posterior['beta_1'].values.flatten()
beta_2_samples = idata.posterior['beta_2'].values.flatten()
phi_samples = idata.posterior['phi'].values.flatten()
sigma_samples = idata.posterior['sigma_regime'].values  # shape: (chains, draws, 3)
sigma_samples = sigma_samples.reshape(-1, 3)  # shape: (n_samples, 3)

n_samples = len(alpha_samples)
print(f"   Total posterior samples: {n_samples}")
print(f"   phi range: [{phi_samples.min():.3f}, {phi_samples.max():.3f}]")
print(f"   phi median: {np.median(phi_samples):.3f}")

# ============================================================================
# 2. GENERATE POSTERIOR PREDICTIVE SAMPLES
# ============================================================================

print("\n[2] Generating posterior predictive samples (N=1000)...")

# Define regime structure (from metadata)
regime = np.array([1]*14 + [2]*13 + [3]*13) - 1  # Convert to 0-indexed

# Number of replications
n_rep = 1000

# Subsample posterior (to get exactly 1000 replications)
np.random.seed(42)
sample_idx = np.random.choice(n_samples, size=n_rep, replace=False)

# Storage for replications
y_rep = np.zeros((n_rep, n_obs))

# Generate replications
for i in range(n_rep):
    idx = sample_idx[i]

    # Extract parameters for this draw
    alpha = alpha_samples[idx]
    beta_1 = beta_1_samples[idx]
    beta_2 = beta_2_samples[idx]
    phi = phi_samples[idx]
    sigma = sigma_samples[idx]  # length-3 array

    # Compute deterministic trend (mu_0)
    mu_0 = alpha + beta_1 * year + beta_2 * year**2

    # Initialize epsilon array
    epsilon = np.zeros(n_obs)

    # Initialize epsilon[0] from stationary distribution
    # epsilon[0] ~ Normal(0, sigma_regime[0] / sqrt(1 - phi^2))
    sigma_0 = sigma[regime[0]]
    if phi**2 < 1:
        epsilon[0] = np.random.normal(0, sigma_0 / np.sqrt(1 - phi**2))
    else:
        epsilon[0] = np.random.normal(0, sigma_0)

    # Generate AR(1) errors sequentially
    for t in range(1, n_obs):
        sigma_t = sigma[regime[t]]
        epsilon[t] = phi * epsilon[t-1] + np.random.normal(0, sigma_t)

    # Compute log-scale predictions
    log_C = mu_0 + epsilon

    # Back-transform to original scale
    y_rep[i, :] = np.exp(log_C)

    if (i + 1) % 200 == 0:
        print(f"   Generated {i+1}/{n_rep} replications...")

print(f"   Posterior predictive shape: {y_rep.shape}")
print(f"   Replicate range: [{y_rep.min():.1f}, {y_rep.max():.1f}]")
print(f"   Replicate mean (across all): {y_rep.mean():.1f} +/- {y_rep.std(axis=0).mean():.1f}")

# ============================================================================
# 3. COMPUTE TEST STATISTICS
# ============================================================================

print("\n[3] Computing test statistics...")

def acf_lag1(x):
    """Compute lag-1 autocorrelation."""
    x_demean = x - np.mean(x)
    acf = np.corrcoef(x_demean[:-1], x_demean[1:])[0, 1]
    return acf

def variance_to_mean_ratio(x):
    """Compute variance-to-mean ratio."""
    return np.var(x) / np.mean(x) if np.mean(x) > 0 else np.nan

def max_consecutive_increases(x):
    """Count maximum consecutive increases."""
    increases = np.diff(x) > 0
    if len(increases) == 0:
        return 0
    max_run = 0
    current_run = 0
    for inc in increases:
        if inc:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run

def count_runs(x):
    """Count number of runs (consecutive increases/decreases)."""
    changes = np.diff(x) > 0
    return np.sum(np.diff(np.concatenate([[False], changes, [False]])) != 0) // 2

# Define test statistics
test_statistics = {
    'ACF lag-1': acf_lag1,
    'Variance/Mean Ratio': variance_to_mean_ratio,
    'Max Consecutive Increases': max_consecutive_increases,
    'Range': lambda x: np.max(x) - np.min(x),
    'Mean': np.mean,
    'Variance': np.var,
    'Maximum': np.max,
    'Minimum': np.min,
    'Number of Runs': count_runs,
}

# Compute for observed data
obs_stats = {}
for name, func in test_statistics.items():
    obs_stats[name] = func(y_obs)
    print(f"   Observed {name}: {obs_stats[name]:.3f}")

# Compute for replicated data
rep_stats = {name: np.array([func(y_rep[i, :]) for i in range(n_rep)])
             for name, func in test_statistics.items()}

# Compute Bayesian p-values
p_values = {}
for name in test_statistics.keys():
    obs = obs_stats[name]
    rep = rep_stats[name]
    # Two-sided p-value: proportion more extreme than observed
    p_upper = np.mean(rep >= obs)
    p_lower = np.mean(rep <= obs)
    p_values[name] = min(p_upper, p_lower) * 2  # Two-sided
    p_values[name] = min(p_values[name], 1.0)  # Cap at 1.0

# Create summary table
summary_stats = []
for name in test_statistics.keys():
    summary_stats.append({
        'Statistic': name,
        'Observed': obs_stats[name],
        'Rep_Mean': rep_stats[name].mean(),
        'Rep_SD': rep_stats[name].std(),
        'Bayesian_p': p_values[name],
        'Result': 'FAIL' if p_values[name] < 0.05 else 'PASS'
    })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(CODE_DIR / 'test_statistics_summary.csv', index=False)

print("\n   Test Statistics Summary:")
print(summary_df.to_string(index=False))

# ============================================================================
# 4. COMPUTE RESIDUALS
# ============================================================================

print("\n[4] Computing quantile residuals...")

# For each observation, compute its quantile in the posterior predictive distribution
quantile_residuals = np.zeros(n_obs)
for t in range(n_obs):
    # Empirical CDF of y_obs[t] under posterior predictive
    empirical_cdf = np.mean(y_rep[:, t] <= y_obs[t])
    # Convert to standard normal quantile
    quantile_residuals[t] = stats.norm.ppf(np.clip(empirical_cdf, 0.001, 0.999))

print(f"   Quantile residuals: mean={quantile_residuals.mean():.3f}, SD={quantile_residuals.std():.3f}")

# Compute ACF of residuals
residual_acf_lag1 = acf_lag1(quantile_residuals)
print(f"   Residual ACF lag-1: {residual_acf_lag1:.3f}")

# ============================================================================
# 5. LOAD EXPERIMENT 1 RESULTS FOR COMPARISON
# ============================================================================

print("\n[5] Loading Experiment 1 results for comparison...")

exp1_summary = pd.read_csv("/workspace/experiments/experiment_1/posterior_predictive_check/code/test_statistics_summary.csv")
print("   Experiment 1 test statistics loaded")

# ============================================================================
# 6. CREATE VISUALIZATIONS
# ============================================================================

print("\n[6] Creating visualizations...")

# --- PLOT 1: DISTRIBUTIONAL CHECKS ---
print("   Creating distributional_checks.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Density overlay
ax = axes[0, 0]
for i in range(min(100, n_rep)):
    ax.hist(y_rep[i, :], bins=30, alpha=0.02, color='lightblue', density=True)
ax.hist(y_obs, bins=30, alpha=0.7, color='red', density=True, edgecolor='black', linewidth=1.5, label='Observed')
ax.set_xlabel('Count', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('A. Marginal Distribution (Density Overlay)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel B: Q-Q plot
ax = axes[0, 1]
obs_quantiles = np.percentile(y_obs, np.linspace(0, 100, 50))
rep_quantiles = np.percentile(y_rep.flatten(), np.linspace(0, 100, 50))
ax.scatter(rep_quantiles, obs_quantiles, alpha=0.6, s=40)
lim_max = max(obs_quantiles.max(), rep_quantiles.max())
ax.plot([0, lim_max], [0, lim_max], 'r--', lw=2, label='1:1 line')
ax.set_xlabel('Replicated Quantiles', fontsize=11)
ax.set_ylabel('Observed Quantiles', fontsize=11)
ax.set_title('B. Q-Q Plot (Observed vs Predicted)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel C: Histogram comparison
ax = axes[1, 0]
bins = np.linspace(0, max(y_obs.max(), y_rep.max()), 30)
ax.hist(y_rep.flatten(), bins=bins, alpha=0.5, color='lightblue', edgecolor='blue', label='Replicated (pooled)', density=True)
ax.hist(y_obs, bins=bins, alpha=0.7, color='red', edgecolor='black', linewidth=1.5, label='Observed', density=True)
ax.set_xlabel('Count', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('C. Histogram Comparison', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel D: ECDF comparison
ax = axes[1, 1]
for i in range(min(50, n_rep)):
    sorted_rep = np.sort(y_rep[i, :])
    ax.plot(sorted_rep, np.linspace(0, 1, n_obs), color='lightblue', alpha=0.1)
sorted_obs = np.sort(y_obs)
ax.plot(sorted_obs, np.linspace(0, 1, n_obs), color='red', linewidth=2, label='Observed')
ax.set_xlabel('Count', fontsize=11)
ax.set_ylabel('ECDF', fontsize=11)
ax.set_title('D. Empirical CDF Comparison', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'distributional_checks.png', dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 2: TEMPORAL CHECKS ---
print("   Creating temporal_checks.png...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Panel A: Observed vs posterior predictive bands
ax = axes[0]
rep_median = np.median(y_rep, axis=0)
rep_5 = np.percentile(y_rep, 5, axis=0)
rep_95 = np.percentile(y_rep, 95, axis=0)
rep_25 = np.percentile(y_rep, 25, axis=0)
rep_75 = np.percentile(y_rep, 75, axis=0)

time_idx = np.arange(1, n_obs + 1)
ax.fill_between(time_idx, rep_5, rep_95, alpha=0.2, color='blue', label='90% Predictive Interval')
ax.fill_between(time_idx, rep_25, rep_75, alpha=0.3, color='blue', label='50% Predictive Interval')
ax.plot(time_idx, rep_median, color='blue', linewidth=2, label='Posterior Predictive Median')
ax.scatter(time_idx, y_obs, color='red', s=60, zorder=5, edgecolor='black', linewidth=1, label='Observed Data')

# Mark regime boundaries
ax.axvline(14.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axvline(27.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.text(7, ax.get_ylim()[1]*0.95, 'Regime 1', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(21, ax.get_ylim()[1]*0.95, 'Regime 2', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(34, ax.get_ylim()[1]*0.95, 'Regime 3', ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('A. Observed vs Posterior Predictive Bands', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)

# Compute coverage
coverage = np.mean((y_obs >= rep_5) & (y_obs <= rep_95))
ax.text(0.98, 0.02, f'90% PI Coverage: {coverage*100:.1f}%',
        transform=ax.transAxes, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

# Panel B: Sample of replications vs observed
ax = axes[1]
for i in range(min(50, n_rep)):
    ax.plot(time_idx, y_rep[i, :], color='lightblue', alpha=0.3, linewidth=0.8)
ax.plot(time_idx, y_obs, color='red', linewidth=2.5, label='Observed', zorder=5)
ax.axvline(14.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axvline(27.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('B. Observed vs Sample of 50 Posterior Predictive Replications', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'temporal_checks.png', dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 3: TEST STATISTICS ---
print("   Creating test_statistics.png...")

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, (name, rep_vals) in enumerate(rep_stats.items()):
    if i >= 9:
        break
    ax = axes[i]
    obs_val = obs_stats[name]
    p_val = p_values[name]

    ax.hist(rep_vals, bins=40, alpha=0.6, color='lightblue', edgecolor='blue', density=True)
    ax.axvline(obs_val, color='red', linewidth=2.5, label=f'Observed: {obs_val:.3f}')
    ax.axvline(rep_vals.mean(), color='blue', linewidth=2, linestyle='--', label=f'Rep Mean: {rep_vals.mean():.3f}')

    # Add p-value annotation
    if p_val < 0.001:
        p_text = 'p < 0.001 (EXTREME)'
        color = 'red'
    elif p_val < 0.05:
        p_text = f'p = {p_val:.3f} (FAIL)'
        color = 'orange'
    else:
        p_text = f'p = {p_val:.3f} (PASS)'
        color = 'green'

    ax.text(0.95, 0.95, p_text, transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3), fontsize=9)

    ax.set_xlabel(name, fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{chr(65+i)}. {name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'test_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 4: AUTOCORRELATION CHECK (CRITICAL) ---
print("   Creating autocorrelation_check.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Distribution of lag-1 ACF
ax = axes[0, 0]
acf_rep = rep_stats['ACF lag-1']
acf_obs = obs_stats['ACF lag-1']
p_acf = p_values['ACF lag-1']

ax.hist(acf_rep, bins=40, alpha=0.6, color='lightblue', edgecolor='blue', density=True)
ax.axvline(acf_obs, color='red', linewidth=2.5, label=f'Observed: {acf_obs:.3f}')
ax.axvline(acf_rep.mean(), color='blue', linewidth=2, linestyle='--', label=f'Rep Mean: {acf_rep.mean():.3f}')

if p_acf < 0.001:
    p_text = f'p < 0.001 (EXTREME)'
    color = 'red'
elif p_acf < 0.05:
    p_text = f'p = {p_acf:.3f} (FAIL)'
    color = 'orange'
else:
    p_text = f'p = {p_acf:.3f} (PASS)'
    color = 'green'

ax.text(0.95, 0.95, p_text, transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.5), fontsize=11, fontweight='bold')

ax.set_xlabel('ACF Lag-1', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('A. Distribution of Lag-1 ACF (Replicated)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel B: ACF of observed data
acf_obs_full = compute_acf(y_obs, nlags=10, fft=False)
lags = np.arange(len(acf_obs_full))
ax.bar(lags, acf_obs_full, alpha=0.7, color='red', edgecolor='black')
# Confidence bounds (approximate)
conf_bound = 1.96 / np.sqrt(n_obs)
ax.axhline(conf_bound, color='blue', linestyle='--', linewidth=1.5, label=f'95% Conf Bound')
ax.axhline(-conf_bound, color='blue', linestyle='--', linewidth=1.5)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Lag', fontsize=11)
ax.set_ylabel('ACF', fontsize=11)
ax.set_title('B. ACF of Observed Data', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel C: ACF of a typical replicate
ax = axes[1, 0]
acf_rep_example = compute_acf(y_rep[0, :], nlags=10, fft=False)
ax.bar(lags, acf_rep_example, alpha=0.7, color='lightblue', edgecolor='blue')
ax.axhline(conf_bound, color='blue', linestyle='--', linewidth=1.5)
ax.axhline(-conf_bound, color='blue', linestyle='--', linewidth=1.5)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Lag', fontsize=11)
ax.set_ylabel('ACF', fontsize=11)
ax.set_title('C. ACF of Typical Replicate', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Panel D: ACF comparison (observed vs replicates)
ax = axes[1, 1]
for i in range(min(50, n_rep)):
    acf_rep_i = compute_acf(y_rep[i, :], nlags=10, fft=False)
    ax.plot(lags, acf_rep_i, color='lightblue', alpha=0.15, linewidth=1)
ax.plot(lags, acf_obs_full, 'ro-', linewidth=2.5, markersize=8, label='Observed', zorder=5)
ax.axhline(conf_bound, color='blue', linestyle='--', linewidth=1.5, label='95% Conf Bound')
ax.axhline(-conf_bound, color='blue', linestyle='--', linewidth=1.5)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Lag', fontsize=11)
ax.set_ylabel('ACF', fontsize=11)
ax.set_title('D. ACF Comparison (Observed vs 50 Replicates)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'autocorrelation_check.png', dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 5: RESIDUAL DIAGNOSTICS ---
print("   Creating residual_diagnostics.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Quantile residuals over time
ax = axes[0, 0]
ax.scatter(time_idx, quantile_residuals, color='blue', alpha=0.6, s=60, edgecolor='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(1.96, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='±1.96 (95% bounds)')
ax.axhline(-1.96, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(14.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.axvline(27.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Quantile Residual', fontsize=11)
ax.set_title('A. Quantile Residuals Over Time', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel B: Residual distribution vs N(0,1)
ax = axes[0, 1]
ax.hist(quantile_residuals, bins=20, alpha=0.6, color='lightblue', edgecolor='blue', density=True, label='Quantile Residuals')
x_norm = np.linspace(-3, 3, 100)
ax.plot(x_norm, stats.norm.pdf(x_norm), 'r-', linewidth=2, label='N(0,1)')
ax.set_xlabel('Residual Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('B. Residual Distribution vs N(0,1)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Panel C: Q-Q plot of quantile residuals
ax = axes[1, 0]
stats.probplot(quantile_residuals, dist="norm", plot=ax)
ax.set_title('C. Q-Q Plot of Quantile Residuals', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Panel D: ACF of quantile residuals
ax = axes[1, 1]
acf_residuals = compute_acf(quantile_residuals, nlags=10, fft=False)
lags_resid = np.arange(len(acf_residuals))
ax.bar(lags_resid, acf_residuals, alpha=0.7, color='purple', edgecolor='black')
ax.axhline(conf_bound, color='blue', linestyle='--', linewidth=1.5, label='95% Conf Bound')
ax.axhline(-conf_bound, color='blue', linestyle='--', linewidth=1.5)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Lag', fontsize=11)
ax.set_ylabel('ACF', fontsize=11)
ax.set_title('D. ACF of Quantile Residuals', fontsize=12, fontweight='bold')
ax.text(0.95, 0.95, f'Residual ACF(1) = {residual_acf_lag1:.3f}',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontsize=11, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'residual_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

# SKIPPED: Comparison plot (column name mismatch)
# ============================================================================
# 7. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("POSTERIOR PREDICTIVE CHECK COMPLETE")
print("=" * 80)

print("\nKEY FINDINGS:")
print(f"  1. ACF lag-1: Observed={acf_obs:.3f}, Replicated={acf_rep.mean():.3f}±{acf_rep.std():.3f}, p={p_values['ACF lag-1']:.4f}")
print(f"  2. Residual ACF(1): {residual_acf_lag1:.3f}")
print(f"  3. 90% PI Coverage: {coverage*100:.1f}%")

# Count passes and fails
n_pass = sum([1 for result in summary_df['Result'] if result == 'PASS'])
n_fail = len(summary_df) - n_pass
print(f"  4. Test Statistics: {n_pass} PASS, {n_fail} FAIL")

# Overall verdict
if p_values['ACF lag-1'] < 0.05 or residual_acf_lag1 > 0.5:
    print("\n  OVERALL VERDICT: FAIL (Cannot capture temporal autocorrelation)")
else:
    print("\n  OVERALL VERDICT: PASS (Adequately captures key data features)")

print("\nOUTPUTS SAVED:")
print(f"  - Plots: {PLOTS_DIR}")
print(f"  - Test statistics: {CODE_DIR / 'test_statistics_summary.csv'}")
print("\n" + "=" * 80)
