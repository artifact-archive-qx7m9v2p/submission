"""
Posterior Predictive Check Analysis for Fixed-Effect Normal Model
Experiment 1: Meta-Analysis with Known Measurement Uncertainties
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = "/workspace/data/data.csv"
IDATA_PATH = "/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf"
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_predictive_check")
PLOTS_DIR = OUTPUT_DIR / "plots"

# Load data
print("Loading data and posterior samples...")
data = pd.read_csv(DATA_PATH)
y_obs = data['y'].values
sigma = data['sigma'].values
n_obs = len(y_obs)

# Load InferenceData
idata = az.from_netcdf(IDATA_PATH)
print(f"Loaded InferenceData with groups: {list(idata.groups())}")

# Extract posterior samples of theta
theta_samples = idata.posterior['theta'].values.flatten()
n_samples = len(theta_samples)
print(f"Posterior samples: {n_samples}")
print(f"Theta posterior mean: {theta_samples.mean():.2f} ± {theta_samples.std():.2f}")

# ============================================================================
# 1. GENERATE POSTERIOR PREDICTIVE SAMPLES
# ============================================================================
print("\n" + "="*70)
print("1. GENERATING POSTERIOR PREDICTIVE SAMPLES")
print("="*70)

# For each posterior sample of theta, generate y_rep for all observations
# y_rep[i] ~ Normal(theta, sigma[i]^2)
np.random.seed(42)
y_rep = np.zeros((n_samples, n_obs))

for s in range(n_samples):
    for i in range(n_obs):
        y_rep[s, i] = np.random.normal(theta_samples[s], sigma[i])

print(f"Generated y_rep shape: {y_rep.shape}")
print(f"y_rep mean: {y_rep.mean():.2f}")

# ============================================================================
# 2. OBSERVATION-LEVEL PPC (8-panel plot)
# ============================================================================
print("\n" + "="*70)
print("2. OBSERVATION-LEVEL POSTERIOR PREDICTIVE CHECK")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(n_obs):
    ax = axes[i]

    # Plot posterior predictive distribution
    ax.hist(y_rep[:, i], bins=50, density=True, alpha=0.6,
            color='steelblue', edgecolor='black', linewidth=0.5)

    # Mark observed value
    ax.axvline(y_obs[i], color='red', linewidth=2, linestyle='--',
               label=f'Observed: {y_obs[i]}')

    # Compute posterior predictive intervals
    pp_mean = y_rep[:, i].mean()
    pp_std = y_rep[:, i].std()
    pp_50 = np.percentile(y_rep[:, i], [25, 75])
    pp_95 = np.percentile(y_rep[:, i], [2.5, 97.5])

    # Check if observed falls within intervals
    in_50 = pp_50[0] <= y_obs[i] <= pp_50[1]
    in_95 = pp_95[0] <= y_obs[i] <= pp_95[1]

    # Compute p-value: P(y_rep >= y_obs)
    p_val = (y_rep[:, i] >= y_obs[i]).mean()

    ax.set_title(f'Obs {i+1}: y={y_obs[i]}, σ={sigma[i]}\n'
                 f'p={p_val:.3f}, 95%CI: [{pp_95[0]:.1f}, {pp_95[1]:.1f}]',
                 fontsize=10)
    ax.set_xlabel('y_rep')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "observation_level_ppc.png", dpi=300, bbox_inches='tight')
print(f"Saved: observation_level_ppc.png")
plt.close()

# Print detailed observation-level results
print("\nObservation-level PPC Results:")
print("-" * 80)
print(f"{'Obs':<5} {'y_obs':<8} {'σ':<8} {'E[y_rep]':<10} {'SD[y_rep]':<10} "
      f"{'50%CI':<20} {'95%CI':<20} {'p-value':<10}")
print("-" * 80)

obs_results = []
for i in range(n_obs):
    pp_mean = y_rep[:, i].mean()
    pp_std = y_rep[:, i].std()
    pp_50 = np.percentile(y_rep[:, i], [25, 75])
    pp_95 = np.percentile(y_rep[:, i], [2.5, 97.5])
    p_val = (y_rep[:, i] >= y_obs[i]).mean()

    in_50 = pp_50[0] <= y_obs[i] <= pp_50[1]
    in_95 = pp_95[0] <= y_obs[i] <= pp_95[1]

    print(f"{i+1:<5} {y_obs[i]:<8} {sigma[i]:<8} {pp_mean:<10.2f} {pp_std:<10.2f} "
          f"[{pp_50[0]:6.1f},{pp_50[1]:6.1f}] {'✓' if in_50 else '✗':<3} "
          f"[{pp_95[0]:6.1f},{pp_95[1]:6.1f}] {'✓' if in_95 else '✗':<3} "
          f"{p_val:<10.3f}")

    obs_results.append({
        'obs': i+1,
        'y_obs': y_obs[i],
        'sigma': sigma[i],
        'pp_mean': pp_mean,
        'pp_std': pp_std,
        'in_50': in_50,
        'in_95': in_95,
        'p_val': p_val
    })

# ============================================================================
# 3. OVERLAY POSTERIOR PREDICTIVE DISTRIBUTIONS
# ============================================================================
print("\n" + "="*70)
print("3. OVERLAY OF ALL POSTERIOR PREDICTIVE DISTRIBUTIONS")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.tab10(np.linspace(0, 1, n_obs))

for i in range(n_obs):
    # Plot KDE of posterior predictive
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(y_rep[:, i])
    x_range = np.linspace(y_rep[:, i].min(), y_rep[:, i].max(), 200)
    ax.plot(x_range, kde(x_range), color=colors[i], alpha=0.7,
            linewidth=2, label=f'Obs {i+1}')

    # Mark observed value
    ax.scatter(y_obs[i], 0, color=colors[i], s=100, marker='o',
               edgecolors='black', linewidths=2, zorder=10)

ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Posterior Predictive Distributions (colored) vs Observed Values (dots)',
             fontsize=14)
ax.legend(ncol=2, fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "overlay_posterior_predictive.png", dpi=300, bbox_inches='tight')
print(f"Saved: overlay_posterior_predictive.png")
plt.close()

# ============================================================================
# 4. AGGREGATE TEST STATISTICS
# ============================================================================
print("\n" + "="*70)
print("4. AGGREGATE TEST STATISTICS")
print("="*70)

# Define test statistics
test_stats = {
    'Mean': lambda x: np.mean(x),
    'SD': lambda x: np.std(x, ddof=1),
    'Min': lambda x: np.min(x),
    'Max': lambda x: np.max(x),
    'Range': lambda x: np.max(x) - np.min(x),
    'Median': lambda x: np.median(x)
}

# Compute for observed data
T_obs = {name: func(y_obs) for name, func in test_stats.items()}

# Compute for each posterior predictive replicate
T_rep = {name: np.array([func(y_rep[s, :]) for s in range(n_samples)])
         for name, func in test_stats.items()}

# Compute posterior p-values
p_values = {name: (T_rep[name] >= T_obs[name]).mean()
            for name in test_stats.keys()}

# Plot test statistics
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, T_obs_val) in enumerate(T_obs.items()):
    ax = axes[idx]

    # Histogram of T(y_rep)
    ax.hist(T_rep[name], bins=50, density=True, alpha=0.6,
            color='steelblue', edgecolor='black', linewidth=0.5)

    # Mark T(y_obs)
    ax.axvline(T_obs_val, color='red', linewidth=2, linestyle='--',
               label=f'T(y_obs) = {T_obs_val:.2f}')

    # Add p-value
    p_val = p_values[name]
    ax.text(0.05, 0.95, f'p-value = {p_val:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(f'T(y_rep): {name}', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Test Statistic: {name}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "test_statistics.png", dpi=300, bbox_inches='tight')
print(f"Saved: test_statistics.png")
plt.close()

# Print results
print("\nTest Statistics Results:")
print("-" * 60)
print(f"{'Statistic':<15} {'T(y_obs)':<12} {'E[T(y_rep)]':<12} {'p-value':<10} {'Assessment'}")
print("-" * 60)

for name in test_stats.keys():
    assessment = "GOOD" if 0.1 <= p_values[name] <= 0.9 else "FLAGGED"
    if p_values[name] < 0.05 or p_values[name] > 0.95:
        assessment = "EXTREME"

    print(f"{name:<15} {T_obs[name]:<12.2f} {T_rep[name].mean():<12.2f} "
          f"{p_values[name]:<10.3f} {assessment}")

# ============================================================================
# 5. LOO-PIT ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("5. LOO-PIT (PROBABILITY INTEGRAL TRANSFORM) ANALYSIS")
print("="*70)

try:
    # Compute LOO-PIT using ArviZ
    # First, we need to add posterior_predictive to idata if not present
    if 'posterior_predictive' not in idata.groups():
        print("Adding posterior_predictive to InferenceData...")

        # Create posterior predictive group
        # Reshape y_rep to match ArviZ expected format
        # Shape: (n_chains, n_draws, n_obs)
        n_chains = idata.posterior.dims['chain']
        n_draws = idata.posterior.dims['draw']

        y_rep_reshaped = y_rep[:n_chains * n_draws, :].reshape(n_chains, n_draws, n_obs)

        import xarray as xr
        posterior_predictive = xr.Dataset(
            {
                'y': xr.DataArray(
                    y_rep_reshaped,
                    dims=['chain', 'draw', 'obs_id'],
                    coords={
                        'chain': idata.posterior.coords['chain'],
                        'draw': idata.posterior.coords['draw'],
                        'obs_id': np.arange(n_obs)
                    }
                )
            }
        )

        # Add observed data group
        observed_data = xr.Dataset(
            {
                'y': xr.DataArray(
                    y_obs,
                    dims=['obs_id'],
                    coords={'obs_id': np.arange(n_obs)}
                )
            }
        )

        idata.add_groups(posterior_predictive=posterior_predictive, observed_data=observed_data)

    # Compute LOO-PIT
    loo_pit = az.loo_pit(idata=idata, y='y')

    # Plot LOO-PIT
    fig, ax = plt.subplots(figsize=(10, 6))
    az.plot_loo_pit(idata=idata, y='y', legend=True, ax=ax)
    ax.set_title('LOO-PIT Uniformity Check\n(Should be uniform on [0,1] if well-calibrated)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loo_pit.png", dpi=300, bbox_inches='tight')
    print(f"Saved: loo_pit.png")
    plt.close()

    # Perform uniformity tests
    ks_stat, ks_pval = stats.kstest(loo_pit.values.flatten(), 'uniform')

    print(f"\nLOO-PIT Uniformity Test (Kolmogorov-Smirnov):")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  p-value: {ks_pval:.4f}")
    print(f"  Assessment: {'GOOD (uniform)' if ks_pval > 0.05 else 'POOR (non-uniform)'}")

    loo_pit_results = {
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'pit_values': loo_pit.values.flatten()
    }

except Exception as e:
    print(f"LOO-PIT computation failed: {e}")
    print("Continuing with other diagnostics...")
    loo_pit_results = None

# ============================================================================
# 6. RESIDUAL ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("6. RESIDUAL ANALYSIS")
print("="*70)

# Use posterior mean of theta for residual computation
theta_hat = theta_samples.mean()

# Standardized residuals: (y_i - theta_hat) / sigma_i
residuals = (y_obs - theta_hat) / sigma

print(f"Theta posterior mean: {theta_hat:.2f}")
print(f"\nStandardized Residuals:")
print("-" * 40)
for i in range(n_obs):
    print(f"Obs {i+1}: {residuals[i]:6.2f} {'(|r| > 2)' if abs(residuals[i]) > 2 else ''}")

# 4-panel residual plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Residuals vs Study Index
ax = axes[0, 0]
ax.scatter(np.arange(1, n_obs+1), residuals, s=100, alpha=0.7, edgecolors='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5, label='|r| = 2')
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5)
ax.set_xlabel('Observation Index', fontsize=11)
ax.set_ylabel('Standardized Residual', fontsize=11)
ax.set_title('Residuals vs Study Index', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Residuals vs Sigma
ax = axes[0, 1]
ax.scatter(sigma, residuals, s=100, alpha=0.7, edgecolors='black')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.axhline(2, color='orange', linestyle=':', linewidth=1.5)
ax.axhline(-2, color='orange', linestyle=':', linewidth=1.5)
ax.set_xlabel('Measurement Uncertainty (σ)', fontsize=11)
ax.set_ylabel('Standardized Residual', fontsize=11)
ax.set_title('Residuals vs Measurement Uncertainty', fontsize=12)
ax.grid(True, alpha=0.3)

# Panel 3: Q-Q Plot
ax = axes[1, 0]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Normal Distribution)', fontsize=12)
ax.grid(True, alpha=0.3)

# Panel 4: Histogram
ax = axes[1, 1]
ax.hist(residuals, bins=8, density=True, alpha=0.6, color='steelblue',
        edgecolor='black', linewidth=1.5)
x_range = np.linspace(-3, 3, 100)
ax.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'r-', linewidth=2,
        label='N(0,1)')
ax.set_xlabel('Standardized Residual', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Residual Distribution vs N(0,1)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "residual_analysis.png", dpi=300, bbox_inches='tight')
print(f"Saved: residual_analysis.png")
plt.close()

# Normality tests
shapiro_stat, shapiro_pval = stats.shapiro(residuals)
anderson_result = stats.anderson(residuals, dist='norm')

print(f"\nNormality Tests on Residuals:")
print(f"  Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_pval:.4f}")
print(f"  Anderson-Darling: A²={anderson_result.statistic:.4f}")
print(f"    Critical values (15%, 10%, 5%, 2.5%, 1%): {anderson_result.critical_values}")
print(f"  Assessment: {'Normal' if shapiro_pval > 0.05 else 'Non-normal'}")

# ============================================================================
# 7. COVERAGE ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("7. COVERAGE ANALYSIS")
print("="*70)

coverage_levels = [50, 90, 95]
coverage_results = {}

for level in coverage_levels:
    lower = (100 - level) / 2
    upper = 100 - lower

    intervals = np.percentile(y_rep, [lower, upper], axis=0)
    in_interval = (y_obs >= intervals[0]) & (y_obs <= intervals[1])
    empirical_coverage = in_interval.mean()

    coverage_results[level] = {
        'intervals': intervals,
        'in_interval': in_interval,
        'empirical': empirical_coverage,
        'nominal': level / 100
    }

    print(f"\n{level}% Posterior Predictive Intervals:")
    print(f"  Nominal coverage: {level}%")
    print(f"  Empirical coverage: {empirical_coverage*100:.1f}% ({in_interval.sum()}/{n_obs})")
    print(f"  Assessment: {'GOOD' if abs(empirical_coverage - level/100) < 0.15 else 'POOR'}")

# Coverage plot (forest plot style)
fig, ax = plt.subplots(figsize=(12, 8))

y_pos = np.arange(n_obs)
colors_coverage = ['lightcoral', 'skyblue', 'lightgreen']

for idx, level in enumerate([95, 90, 50]):
    intervals = coverage_results[level]['intervals']
    in_interval = coverage_results[level]['in_interval']

    for i in range(n_obs):
        color = 'green' if in_interval[i] else 'red'
        offset = (idx - 1) * 0.2
        ax.plot([intervals[0, i], intervals[1, i]], [y_pos[i] + offset, y_pos[i] + offset],
                linewidth=2, color=colors_coverage[idx], alpha=0.7,
                label=f'{level}% PI' if i == 0 else '')

# Plot observed values
ax.scatter(y_obs, y_pos, s=150, color='black', marker='o', zorder=10,
           edgecolors='white', linewidths=2, label='Observed')

ax.set_yticks(y_pos)
ax.set_yticklabels([f'Obs {i+1}' for i in range(n_obs)])
ax.set_xlabel('Value', fontsize=12)
ax.set_title('Posterior Predictive Intervals vs Observed Values', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_intervals.png", dpi=300, bbox_inches='tight')
print(f"Saved: coverage_intervals.png")
plt.close()

# ============================================================================
# 8. DISCREPANCY MEASURES
# ============================================================================
print("\n" + "="*70)
print("8. DISCREPANCY MEASURES")
print("="*70)

# Mean absolute error for each observation
mae = np.abs(y_rep - y_obs[np.newaxis, :]).mean(axis=0)

# RMSE
rmse = np.sqrt(((y_obs - theta_hat)**2).mean())

# Standardized RMSE (relative to measurement uncertainties)
standardized_errors = (y_obs - theta_hat) / sigma
standardized_rmse = np.sqrt((standardized_errors**2).mean())

print(f"\nGlobal Discrepancy Measures:")
print(f"  RMSE: {rmse:.2f}")
print(f"  Standardized RMSE: {standardized_rmse:.2f}")
print(f"  (Standardized RMSE ≈ 1 indicates model explains variation well)")

print(f"\nObservation-Level Mean Absolute Error:")
print("-" * 40)
for i in range(n_obs):
    print(f"  Obs {i+1}: MAE = {mae[i]:.2f}")

# Identify worst-fit observations
worst_idx = np.argsort(mae)[::-1][:3]
print(f"\nWorst-Fit Observations (by MAE):")
for rank, idx in enumerate(worst_idx, 1):
    print(f"  {rank}. Obs {idx+1}: y={y_obs[idx]}, σ={sigma[idx]}, MAE={mae[idx]:.2f}, "
          f"residual={residuals[idx]:.2f}")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("9. SAVING RESULTS")
print("="*70)

results = {
    'obs_results': obs_results,
    'test_stats': {name: {'T_obs': T_obs[name],
                          'T_rep_mean': T_rep[name].mean(),
                          'p_value': p_values[name]}
                   for name in test_stats.keys()},
    'residuals': residuals,
    'coverage': coverage_results,
    'discrepancy': {
        'rmse': rmse,
        'standardized_rmse': standardized_rmse,
        'mae': mae
    },
    'normality_tests': {
        'shapiro': {'stat': shapiro_stat, 'pval': shapiro_pval},
        'anderson': {'stat': anderson_result.statistic,
                     'critical_values': anderson_result.critical_values.tolist()}
    }
}

if loo_pit_results is not None:
    results['loo_pit'] = {
        'ks_stat': loo_pit_results['ks_stat'],
        'ks_pval': loo_pit_results['ks_pval']
    }

# Save to numpy
np.save(OUTPUT_DIR / "code" / "ppc_results.npy", results, allow_pickle=True)
print(f"Saved: ppc_results.npy")

print("\n" + "="*70)
print("POSTERIOR PREDICTIVE CHECK ANALYSIS COMPLETE")
print("="*70)
print(f"\nAll plots saved to: {PLOTS_DIR}")
print(f"Results saved to: {OUTPUT_DIR / 'code' / 'ppc_results.npy'}")
