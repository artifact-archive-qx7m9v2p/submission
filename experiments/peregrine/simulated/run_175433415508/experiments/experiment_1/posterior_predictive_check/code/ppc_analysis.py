"""
Posterior Predictive Check for Experiment 1: NB-Linear Model
==============================================================

This script performs comprehensive posterior predictive checks to assess
whether the Negative Binomial Linear model can adequately reproduce
key features of the observed data.

Model: C_t ~ NegativeBinomial(μ_t, φ)
       log(μ_t) = β₀ + β₁×year_t
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
from scipy.special import loggamma
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = '/workspace/data/data.csv'
IDATA_PATH = '/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf'
OUTPUT_DIR = '/workspace/experiments/experiment_1/posterior_predictive_check/plots/'

print("="*70)
print("POSTERIOR PREDICTIVE CHECK: NB-LINEAR MODEL")
print("="*70)

# Load data
df = pd.read_csv(DATA_PATH)
y_obs = df['C'].values
year = df['year'].values
n_obs = len(y_obs)

print(f"\nObserved data:")
print(f"  n = {n_obs}")
print(f"  Range: {y_obs.min()} - {y_obs.max()}")
print(f"  Mean: {y_obs.mean():.1f}")
print(f"  Variance: {y_obs.var():.1f}")

# Load posterior samples
print(f"\nLoading InferenceData from: {IDATA_PATH}")
idata = az.from_netcdf(IDATA_PATH)

print("\nInferenceData structure:")
print(idata)

# Extract posterior samples (note: variable names are beta_0, beta_1)
posterior = idata.posterior
beta0 = posterior['beta_0'].values.flatten()
beta1 = posterior['beta_1'].values.flatten()
phi = posterior['phi'].values.flatten()

n_samples = len(beta0)
print(f"\nPosterior samples: {n_samples}")
print(f"  β₀: {beta0.mean():.3f} ± {beta0.std():.3f}")
print(f"  β₁: {beta1.mean():.3f} ± {beta1.std():.3f}")
print(f"  φ:  {phi.mean():.2f} ± {phi.std():.2f}")

# Generate posterior predictive samples
print("\n" + "="*70)
print("GENERATING POSTERIOR PREDICTIVE DATA")
print("="*70)

n_ppc_samples = min(1000, n_samples)
indices = np.random.choice(n_samples, n_ppc_samples, replace=False)

y_rep = np.zeros((n_ppc_samples, n_obs))

print(f"\nGenerating {n_ppc_samples} replicated datasets...")

for i, idx in enumerate(indices):
    if (i+1) % 200 == 0:
        print(f"  Generated {i+1}/{n_ppc_samples}...")

    # Compute mean for each observation
    log_mu = beta0[idx] + beta1[idx] * year
    mu = np.exp(log_mu)

    # NegativeBinomial parameterization: mean=mu, overdispersion=phi
    # Convert to (n, p) parameterization for numpy
    # mean = n*p/(1-p) = mu
    # var = n*p/(1-p)^2 = mu + mu^2/phi
    # This gives: n = phi, p = mu/(mu + phi)
    p = mu / (mu + phi[idx])
    n = phi[idx]

    # Generate negative binomial samples
    y_rep[i, :] = np.random.negative_binomial(n, 1-p)

print(f"Completed! Generated {n_ppc_samples} × {n_obs} replicated observations")

# Compute test statistics
print("\n" + "="*70)
print("TEST STATISTICS & BAYESIAN P-VALUES")
print("="*70)

def compute_test_stat(y, stat_func):
    """Compute test statistic for a dataset"""
    return stat_func(y)

# Define test statistics
test_stats = {
    'Mean': lambda y: np.mean(y),
    'Variance': lambda y: np.var(y),
    'Min': lambda y: np.min(y),
    'Max': lambda y: np.max(y),
    'Q10': lambda y: np.percentile(y, 10),
    'Q90': lambda y: np.percentile(y, 90),
    'Skewness': lambda y: stats.skew(y),
    'Kurtosis': lambda y: stats.kurtosis(y),
    'CV': lambda y: np.std(y) / np.mean(y),
    'N_zeros': lambda y: np.sum(y == 0),
    'Prop_extreme': lambda y: np.mean(y > 200),
}

results = []

for name, func in test_stats.items():
    # Observed statistic
    t_obs = func(y_obs)

    # Replicated statistics
    t_rep = np.array([func(y_rep[i, :]) for i in range(n_ppc_samples)])

    # Bayesian p-value
    p_value = np.mean(t_rep >= t_obs)

    results.append({
        'Statistic': name,
        'Observed': t_obs,
        'Rep_Mean': np.mean(t_rep),
        'Rep_SD': np.std(t_rep),
        'p_value': p_value
    })

    status = "✓" if 0.05 <= p_value <= 0.95 else "⚠"
    print(f"{status} {name:15s}: obs={t_obs:8.2f}  "
          f"rep={np.mean(t_rep):8.2f}±{np.std(t_rep):6.2f}  "
          f"p={p_value:.3f}")

results_df = pd.DataFrame(results)

# Compute posterior predictive intervals
print("\n" + "="*70)
print("POSTERIOR PREDICTIVE INTERVALS")
print("="*70)

y_rep_mean = np.mean(y_rep, axis=0)
y_rep_q05 = np.percentile(y_rep, 5, axis=0)
y_rep_q25 = np.percentile(y_rep, 25, axis=0)
y_rep_q50 = np.percentile(y_rep, 50, axis=0)
y_rep_q75 = np.percentile(y_rep, 75, axis=0)
y_rep_q95 = np.percentile(y_rep, 95, axis=0)

# Check coverage
in_50 = np.sum((y_obs >= y_rep_q25) & (y_obs <= y_rep_q75))
in_90 = np.sum((y_obs >= y_rep_q05) & (y_obs <= y_rep_q95))

print(f"Coverage:")
print(f"  50% interval: {in_50}/{n_obs} = {100*in_50/n_obs:.1f}% (expect 50%)")
print(f"  90% interval: {in_90}/{n_obs} = {100*in_90/n_obs:.1f}% (expect 90%)")

# Compute randomized quantile residuals
print("\n" + "="*70)
print("RANDOMIZED QUANTILE RESIDUALS")
print("="*70)

# For each observation, compute empirical CDF from posterior predictive
residuals = np.zeros(n_obs)
for i in range(n_obs):
    # Empirical CDF at observed value
    cdf_val = np.mean(y_rep[:, i] <= y_obs[i])
    # Transform to standard normal
    residuals[i] = stats.norm.ppf(np.clip(cdf_val, 0.001, 0.999))

print(f"Residual summary:")
print(f"  Mean: {np.mean(residuals):.3f} (expect 0)")
print(f"  SD: {np.std(residuals):.3f} (expect 1)")
print(f"  Min: {np.min(residuals):.3f}")
print(f"  Max: {np.max(residuals):.3f}")

# Compute ACF of residuals
print("\n" + "="*70)
print("AUTOCORRELATION OF RESIDUALS")
print("="*70)

def compute_acf(x, max_lag=10):
    """Compute autocorrelation function"""
    acf = np.correlate(x - x.mean(), x - x.mean(), mode='full')
    acf = acf[len(acf)//2:] / acf[len(acf)//2]
    return acf[:max_lag+1]

acf_residuals = compute_acf(residuals, max_lag=10)
print("Residual ACF:")
for lag in range(min(6, len(acf_residuals))):
    print(f"  Lag {lag}: {acf_residuals[lag]:.3f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# 1. PPC Overview - Distribution comparison
print("\n1. PPC Overview (observed vs replicated distributions)...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A. Histogram overlay
ax = axes[0, 0]
ax.hist(y_obs, bins=20, alpha=0.6, color='black', edgecolor='black',
        label='Observed', density=True)
for i in range(min(100, n_ppc_samples)):
    ax.hist(y_rep[i, :], bins=20, alpha=0.01, color='blue', density=True)
ax.hist(y_rep[0, :], bins=20, alpha=0.01, color='blue',
        label='Replicated (100 draws)', density=True)
ax.set_xlabel('Count (C)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('A. Distribution Comparison', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# B. Empirical CDF
ax = axes[0, 1]
ax.plot(np.sort(y_obs), np.linspace(0, 1, n_obs),
        'k-', linewidth=2, label='Observed')
for i in range(min(50, n_ppc_samples)):
    ax.plot(np.sort(y_rep[i, :]), np.linspace(0, 1, n_obs),
            'b-', alpha=0.05)
ax.set_xlabel('Count (C)', fontsize=12)
ax.set_ylabel('Cumulative Probability', fontsize=12)
ax.set_title('B. Empirical CDF', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# C. Q-Q plot
ax = axes[1, 0]
# Use median of replicates as expected
y_rep_sorted = np.sort(y_rep, axis=1)
y_rep_median = np.median(y_rep_sorted, axis=0)
ax.scatter(y_rep_median, np.sort(y_obs), alpha=0.6, s=50)
min_val = min(y_rep_median.min(), y_obs.min())
max_val = max(y_rep_median.max(), y_obs.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
ax.set_xlabel('Predicted Quantiles (Median of Reps)', fontsize=12)
ax.set_ylabel('Observed Quantiles', fontsize=12)
ax.set_title('C. Q-Q Plot', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# D. Density comparison
ax = axes[1, 1]
from scipy.stats import gaussian_kde
kde_obs = gaussian_kde(y_obs)
x_range = np.linspace(y_obs.min()-10, y_obs.max()+10, 200)
ax.plot(x_range, kde_obs(x_range), 'k-', linewidth=2, label='Observed')
# Plot several replicated KDEs
for i in range(min(20, n_ppc_samples)):
    kde_rep = gaussian_kde(y_rep[i, :])
    ax.plot(x_range, kde_rep(x_range), 'b-', alpha=0.1)
ax.set_xlabel('Count (C)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('D. Kernel Density Estimate', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'ppc_overview.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}ppc_overview.png")
plt.close()

# 2. Time series with predictive intervals
print("\n2. Time series with posterior predictive bands...")
fig, ax = plt.subplots(figsize=(14, 6))

# Plot predictive intervals
ax.fill_between(year, y_rep_q05, y_rep_q95, alpha=0.2, color='blue',
                label='90% Predictive Interval')
ax.fill_between(year, y_rep_q25, y_rep_q75, alpha=0.3, color='blue',
                label='50% Predictive Interval')
ax.plot(year, y_rep_q50, 'b-', linewidth=2, label='Median Prediction')

# Plot observed data
ax.plot(year, y_obs, 'ko-', linewidth=2, markersize=6,
        label='Observed Data', zorder=10)

# Mark points outside 90% interval
outside_90 = (y_obs < y_rep_q05) | (y_obs > y_rep_q95)
if np.any(outside_90):
    ax.plot(year[outside_90], y_obs[outside_90], 'ro', markersize=10,
            label=f'Outside 90% PI (n={np.sum(outside_90)})', zorder=11)

ax.set_xlabel('Year (standardized)', fontsize=13)
ax.set_ylabel('Count (C)', fontsize=13)
ax.set_title('Posterior Predictive Check: Observed Data vs Predictions',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'ppc_timeseries.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}ppc_timeseries.png")
plt.close()

# 3. Rootogram
print("\n3. Rootogram (count distribution fit)...")
fig, ax = plt.subplots(figsize=(14, 6))

# Compute count frequencies
count_range = np.arange(0, y_obs.max() + 1)
obs_counts = np.array([np.sum(y_obs == c) for c in count_range])
exp_counts = np.array([np.mean(np.sum(y_rep == c, axis=1)) for c in count_range])

# Hanging rootogram (subtract expected from observed on sqrt scale)
obs_sqrt = np.sqrt(obs_counts)
exp_sqrt = np.sqrt(exp_counts)
hanging = obs_sqrt - exp_sqrt

# Plot
width = 1.0
ax.bar(count_range, hanging, width=width, alpha=0.6, color='steelblue',
       edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Count Value', fontsize=13)
ax.set_ylabel('Sqrt(Observed) - Sqrt(Expected)', fontsize=13)
ax.set_title('Hanging Rootogram: Observed vs Expected Count Frequencies',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add text annotation
textstr = 'Values near zero indicate good fit\nNegative bars: fewer observed than expected\nPositive bars: more observed than expected'
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'rootogram.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}rootogram.png")
plt.close()

# 4. Residual diagnostics
print("\n4. Residual diagnostics...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A. Residuals vs fitted
ax = axes[0, 0]
ax.scatter(y_rep_mean, residuals, alpha=0.6, s=50)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.axhline(y=2, color='orange', linestyle=':', linewidth=1)
ax.axhline(y=-2, color='orange', linestyle=':', linewidth=1)
ax.set_xlabel('Fitted Values (Mean)', fontsize=12)
ax.set_ylabel('Quantile Residuals', fontsize=12)
ax.set_title('A. Residuals vs Fitted Values', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# B. Residuals vs time
ax = axes[0, 1]
ax.scatter(year, residuals, alpha=0.6, s=50)
ax.plot(year, residuals, 'b-', alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.axhline(y=2, color='orange', linestyle=':', linewidth=1)
ax.axhline(y=-2, color='orange', linestyle=':', linewidth=1)
ax.set_xlabel('Year (standardized)', fontsize=12)
ax.set_ylabel('Quantile Residuals', fontsize=12)
ax.set_title('B. Residuals vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# C. Q-Q plot of residuals
ax = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('C. Normal Q-Q Plot of Residuals', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# D. Residual histogram
ax = axes[1, 1]
ax.hist(residuals, bins=20, alpha=0.6, color='steelblue',
        edgecolor='black', density=True)
x_norm = np.linspace(-3, 3, 100)
ax.plot(x_norm, stats.norm.pdf(x_norm), 'r-', linewidth=2,
        label='N(0,1)')
ax.set_xlabel('Quantile Residuals', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('D. Residual Distribution', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'residual_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}residual_diagnostics.png")
plt.close()

# 5. Test statistics Bayesian p-values
print("\n5. Test statistics with Bayesian p-values...")
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes = axes.flatten()

for idx, (name, func) in enumerate(test_stats.items()):
    if idx >= len(axes):
        break

    ax = axes[idx]

    # Observed statistic
    t_obs = func(y_obs)

    # Replicated statistics
    t_rep = np.array([func(y_rep[i, :]) for i in range(n_ppc_samples)])

    # Bayesian p-value
    p_value = np.mean(t_rep >= t_obs)

    # Plot histogram of replicated statistics
    ax.hist(t_rep, bins=30, alpha=0.6, color='steelblue',
            edgecolor='black', density=True)

    # Mark observed value
    ax.axvline(x=t_obs, color='red', linestyle='--', linewidth=2.5,
               label=f'Observed: {t_obs:.2f}')

    # Add p-value
    color = 'green' if 0.05 <= p_value <= 0.95 else 'orange'
    ax.set_title(f'{name}\np-value = {p_value:.3f}',
                fontsize=11, fontweight='bold', color=color)
    ax.set_xlabel('Test Statistic', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(test_stats), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Bayesian P-Values for Test Statistics\n(Green: 0.05 < p < 0.95, Orange: outside)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'test_statistics.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}test_statistics.png")
plt.close()

# 6. Autocorrelation check
print("\n6. Autocorrelation check of residuals...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# A. ACF plot
ax = axes[0]
max_lag = 15
acf_vals = compute_acf(residuals, max_lag=max_lag)
lags = np.arange(max_lag + 1)

# Plot ACF bars
ax.bar(lags, acf_vals, width=0.3, alpha=0.6, color='steelblue',
       edgecolor='black')

# Add confidence bands (approximate)
conf_band = 1.96 / np.sqrt(n_obs)
ax.axhline(y=conf_band, color='blue', linestyle='--', linewidth=1,
           label=f'95% Conf. Band (±{conf_band:.3f})')
ax.axhline(y=-conf_band, color='blue', linestyle='--', linewidth=1)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

ax.set_xlabel('Lag', fontsize=12)
ax.set_ylabel('Autocorrelation', fontsize=12)
ax.set_title('A. ACF of Quantile Residuals', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(lags)

# B. Lag-1 scatter plot
ax = axes[1]
ax.scatter(residuals[:-1], residuals[1:], alpha=0.6, s=50)
ax.set_xlabel('Residual at t', fontsize=12)
ax.set_ylabel('Residual at t+1', fontsize=12)
ax.set_title(f'B. Lag-1 Scatter (ρ = {acf_vals[1]:.3f})',
             fontsize=13, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'autocorrelation_check.png', dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR}autocorrelation_check.png")
plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*70)
print("POSTERIOR PREDICTIVE CHECK SUMMARY")
print("="*70)

print("\n1. TEST STATISTICS (Bayesian p-values):")
print(results_df.to_string(index=False))

n_failed = np.sum((results_df['p_value'] < 0.05) | (results_df['p_value'] > 0.95))
print(f"\n   Failed checks (p < 0.05 or p > 0.95): {n_failed}/{len(results_df)}")

print("\n2. PREDICTIVE INTERVAL COVERAGE:")
print(f"   50% interval: {100*in_50/n_obs:.1f}% (expect 50%)")
print(f"   90% interval: {100*in_90/n_obs:.1f}% (expect 90%)")

print("\n3. RESIDUAL AUTOCORRELATION:")
print(f"   Lag-1 ACF: {acf_vals[1]:.3f}")
print(f"   Lag-2 ACF: {acf_vals[2]:.3f}")
print(f"   Lag-3 ACF: {acf_vals[3]:.3f}")
print(f"   → EXPECTED to be high (no temporal structure in model)")

print("\n4. KEY FINDINGS:")
if n_failed == 0:
    print("   ✓ All test statistics have acceptable p-values")
else:
    print(f"   ⚠ {n_failed} test statistics show extreme p-values")

if in_90/n_obs >= 0.85:
    print("   ✓ Good predictive interval coverage")
else:
    print(f"   ⚠ Predictive interval coverage lower than expected")

if acf_vals[1] > 0.5:
    print(f"   ⚠ High residual autocorrelation (ρ₁={acf_vals[1]:.3f})")
    print("   → Expected! Model lacks temporal correlation structure")
    print("   → Justifies AR(1) extension in Experiment 2")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nGenerated plots:")
print("  - ppc_overview.png: Distribution comparisons")
print("  - ppc_timeseries.png: Time series with predictive bands")
print("  - rootogram.png: Count frequency fit")
print("  - residual_diagnostics.png: Residual patterns")
print("  - test_statistics.png: Bayesian p-values")
print("  - autocorrelation_check.png: Temporal correlation in residuals")
