"""
Prior Predictive Checks for Beta-Binomial Model (Experiment 1)
Pure NumPy implementation (no Stan dependency)

This script:
1. Samples from priors (μ ~ Beta(2,25), φ ~ Gamma(2,2))
2. Generates synthetic datasets from prior predictive distribution
3. Compares prior predictions to observed data
4. Assesses whether priors are scientifically plausible
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# File paths
DATA_PATH = "/workspace/data/data.csv"
OUTPUT_DIR = "/workspace/experiments/experiment_1/prior_predictive_check/plots/"

def beta_binomial_sample(n, alpha, beta):
    """
    Sample from Beta-Binomial distribution.

    First sample p ~ Beta(alpha, beta), then sample r ~ Binomial(n, p).
    """
    p = np.random.beta(alpha, beta)
    r = np.random.binomial(n, p)
    return r

# Load observed data
print("=" * 80)
print("PRIOR PREDICTIVE CHECKS: BETA-BINOMIAL MODEL")
print("=" * 80)

data = pd.read_csv(DATA_PATH)
n_obs = data['n'].values
r_obs = data['r'].values
prop_obs = r_obs / n_obs
N = len(data)

print(f"\nObserved Data Summary:")
print(f"  Number of trials: {N}")
print(f"  Total attempts: {n_obs.sum()}")
print(f"  Total successes: {r_obs.sum()}")
print(f"  Pooled proportion: {r_obs.sum() / n_obs.sum():.4f}")
print(f"  Proportion range: [{prop_obs.min():.4f}, {prop_obs.max():.4f}]")
print(f"  SD of proportions: {prop_obs.std():.4f}")
print(f"  Extreme values: {r_obs.min()}/{n_obs[r_obs.argmin()]} and {r_obs.max()}/{n_obs[r_obs.argmax()]}")

# Compute observed variance inflation (overdispersion indicator)
pooled_p = r_obs.sum() / n_obs.sum()
expected_var_binomial = pooled_p * (1 - pooled_p)
observed_var = prop_obs.var()
obs_variance_inflation = observed_var / expected_var_binomial if expected_var_binomial > 0 else np.inf

print(f"\nObserved Overdispersion:")
print(f"  Variance inflation factor: {obs_variance_inflation:.3f}")
print(f"  (>1 indicates overdispersion)")

# Generate prior predictive samples
print("\n" + "=" * 80)
print("GENERATING PRIOR PREDICTIVE SAMPLES")
print("=" * 80)

n_samples = 2000
print(f"Generating {n_samples} prior predictive datasets...")

# Initialize storage
mu_prior = np.zeros(n_samples)
phi_prior = np.zeros(n_samples)
alpha_prior = np.zeros(n_samples)
beta_prior = np.zeros(n_samples)
r_prior = np.zeros((n_samples, N), dtype=int)
prop_prior = np.zeros((n_samples, N))
total_successes_prior = np.zeros(n_samples)
mean_proportion_prior = np.zeros(n_samples)
sd_proportion_prior = np.zeros(n_samples)
variance_inflation_prior = np.zeros(n_samples)

# Sample from priors and generate data
for i in range(n_samples):
    # Sample hyperparameters from priors
    mu = np.random.beta(2, 25)
    phi = np.random.gamma(2, 1/2)  # shape=2, scale=1/2 (equivalent to rate=2)

    # Transform to shape parameters
    alpha = mu * phi
    beta = (1 - mu) * phi

    # Store parameter samples
    mu_prior[i] = mu
    phi_prior[i] = phi
    alpha_prior[i] = alpha
    beta_prior[i] = beta

    # Generate data for each trial
    for j in range(N):
        r = beta_binomial_sample(n_obs[j], alpha, beta)
        r_prior[i, j] = r
        prop_prior[i, j] = r / n_obs[j]

    # Compute summary statistics
    total_successes_prior[i] = r_prior[i].sum()
    mean_proportion_prior[i] = prop_prior[i].mean()
    sd_proportion_prior[i] = prop_prior[i].std()

    # Compute variance inflation
    if mu * (1 - mu) > 0:
        variance_inflation_prior[i] = (sd_proportion_prior[i] ** 2) / (mu * (1 - mu))
    else:
        variance_inflation_prior[i] = np.nan

    if (i + 1) % 500 == 0:
        print(f"  Generated {i+1}/{n_samples} datasets...")

print("Prior predictive sampling complete!")

print(f"\nPrior Predictive Summary Statistics:")
print(f"  μ (mean probability):")
print(f"    Prior mean: {mu_prior.mean():.4f}")
print(f"    Prior 95% CI: [{np.percentile(mu_prior, 2.5):.4f}, {np.percentile(mu_prior, 97.5):.4f}]")
print(f"\n  φ (concentration):")
print(f"    Prior mean: {phi_prior.mean():.4f}")
print(f"    Prior 95% CI: [{np.percentile(phi_prior, 2.5):.4f}, {np.percentile(phi_prior, 97.5):.4f}]")
print(f"\n  Total successes:")
print(f"    Prior predictive mean: {total_successes_prior.mean():.1f}")
print(f"    Prior predictive 95% CI: [{np.percentile(total_successes_prior, 2.5):.0f}, {np.percentile(total_successes_prior, 97.5):.0f}]")
print(f"    Observed: {r_obs.sum()}")
print(f"\n  Variance inflation factor:")
valid_var_infl = variance_inflation_prior[~np.isnan(variance_inflation_prior)]
print(f"    Prior predictive mean: {valid_var_infl.mean():.3f}")
print(f"    Prior predictive 95% CI: [{np.percentile(valid_var_infl, 2.5):.3f}, {np.percentile(valid_var_infl, 97.5):.3f}]")
print(f"    Observed: {obs_variance_inflation:.3f}")

# Check coverage: What percentile is the observed data at?
print("\n" + "=" * 80)
print("PRIOR PREDICTIVE COVERAGE CHECKS")
print("=" * 80)

def compute_percentile(observed, prior_samples):
    """Compute what percentile the observed value is at in prior predictive"""
    return (prior_samples < observed).mean() * 100

total_succ_pct = compute_percentile(r_obs.sum(), total_successes_prior)
var_infl_pct = compute_percentile(obs_variance_inflation, valid_var_infl)

print(f"\nObserved value percentiles in prior predictive distribution:")
print(f"  Total successes: {total_succ_pct:.1f}th percentile")
print(f"  Variance inflation: {var_infl_pct:.1f}th percentile")

# Check if extreme values are plausible
print("\n" + "=" * 80)
print("EXTREME VALUE CHECKS")
print("=" * 80)

# For the trial with 0/47, what's the distribution of predicted values?
trial_0_idx = 0  # First trial has 0/47
trial_8_idx = 7  # Eighth trial has 31/215

print(f"\nTrial 1 (observed: {r_obs[trial_0_idx]}/{n_obs[trial_0_idx]} = {prop_obs[trial_0_idx]:.4f}):")
print(f"  Prior predictive: r ~ [{r_prior[:, trial_0_idx].min()}, {r_prior[:, trial_0_idx].max()}]")
print(f"  Proportion of samples with r=0: {(r_prior[:, trial_0_idx] == 0).mean():.3f}")
print(f"  Observed r=0 percentile: {compute_percentile(r_obs[trial_0_idx], r_prior[:, trial_0_idx]):.1f}th")

print(f"\nTrial 8 (observed: {r_obs[trial_8_idx]}/{n_obs[trial_8_idx]} = {prop_obs[trial_8_idx]:.4f}):")
print(f"  Prior predictive: r ~ [{r_prior[:, trial_8_idx].min()}, {r_prior[:, trial_8_idx].max()}]")
print(f"  Prior predictive mean: {r_prior[:, trial_8_idx].mean():.1f}")
print(f"  Observed r={r_obs[trial_8_idx]} percentile: {compute_percentile(r_obs[trial_8_idx], r_prior[:, trial_8_idx]):.1f}th")

# Check for computational issues
print("\n" + "=" * 80)
print("COMPUTATIONAL HEALTH CHECKS")
print("=" * 80)

print(f"\nParameter ranges:")
print(f"  α range: [{alpha_prior.min():.4f}, {alpha_prior.max():.4f}]")
print(f"  β range: [{beta_prior.min():.4f}, {beta_prior.max():.4f}]")
print(f"  φ range: [{phi_prior.min():.4f}, {phi_prior.max():.4f}]")

# Check for extreme values
extreme_phi = (phi_prior < 0.01) | (phi_prior > 100)
extreme_mu = (mu_prior < 0.001) | (mu_prior > 0.999)

print(f"\nPotential computational issues:")
print(f"  Samples with φ < 0.01 or φ > 100: {extreme_phi.sum()} ({extreme_phi.mean()*100:.1f}%)")
print(f"  Samples with μ < 0.001 or μ > 0.999: {extreme_mu.sum()} ({extreme_mu.mean()*100:.1f}%)")

if extreme_phi.sum() > 0 or extreme_mu.sum() > 0:
    print("  WARNING: Some extreme parameter values detected")
else:
    print("  All parameter values in reasonable range")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# PLOT 1: Parameter Plausibility - Prior samples for μ and φ
print("\nCreating: parameter_plausibility.png")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top left: μ prior distribution
ax = axes[0, 0]
ax.hist(mu_prior, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
# Overlay theoretical Beta(2, 25) density
x_mu = np.linspace(0, 0.4, 200)
theoretical_mu = stats.beta.pdf(x_mu, 2, 25)
ax.plot(x_mu, theoretical_mu, 'r-', linewidth=2, label='Beta(2, 25)')
ax.axvline(pooled_p, color='green', linestyle='--', linewidth=2, label=f'Observed pooled p={pooled_p:.3f}')
ax.axvline(mu_prior.mean(), color='blue', linestyle=':', linewidth=2, label=f'Prior mean={mu_prior.mean():.3f}')
ax.set_xlabel('μ (Mean Success Probability)')
ax.set_ylabel('Density')
ax.set_title('Prior Distribution for μ')
ax.legend()
ax.grid(True, alpha=0.3)

# Top right: φ prior distribution
ax = axes[0, 1]
ax.hist(phi_prior, bins=50, density=True, alpha=0.6, color='coral', edgecolor='black')
# Overlay theoretical Gamma(2, 2) density
x_phi = np.linspace(0, np.percentile(phi_prior, 99), 200)
theoretical_phi = stats.gamma.pdf(x_phi, a=2, scale=1/2)  # scale = 1/rate
ax.plot(x_phi, theoretical_phi, 'r-', linewidth=2, label='Gamma(2, 2)')
ax.axvline(phi_prior.mean(), color='blue', linestyle=':', linewidth=2, label=f'Prior mean={phi_prior.mean():.3f}')
ax.set_xlabel('φ (Concentration Parameter)')
ax.set_ylabel('Density')
ax.set_title('Prior Distribution for φ')
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom left: Joint distribution of μ and φ
ax = axes[1, 0]
h = ax.hist2d(mu_prior, phi_prior, bins=50, cmap='YlOrRd', cmin=1)
ax.axhline(1.0, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='E[φ]=1')
ax.axvline(pooled_p, color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Obs p={pooled_p:.3f}')
ax.set_xlabel('μ (Mean Success Probability)')
ax.set_ylabel('φ (Concentration Parameter)')
ax.set_title('Joint Prior Distribution')
plt.colorbar(h[3], ax=ax, label='Count')
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom right: Transformed parameters (α, β)
ax = axes[1, 1]
ax.scatter(alpha_prior, beta_prior, alpha=0.1, s=5, color='purple')
ax.set_xlabel('α = μ·φ')
ax.set_ylabel('β = (1-μ)·φ')
ax.set_title('Shape Parameters (α, β)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'parameter_plausibility.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}parameter_plausibility.png")
plt.close()

# PLOT 2: Prior Predictive Coverage - Total Successes and Overdispersion
print("\nCreating: prior_predictive_coverage.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Total successes distribution
ax = axes[0]
ax.hist(total_successes_prior, bins=50, density=True, alpha=0.6, color='steelblue', edgecolor='black')
ax.axvline(r_obs.sum(), color='red', linestyle='--', linewidth=2.5,
           label=f'Observed = {r_obs.sum()}\n({total_succ_pct:.1f}th percentile)')
ax.axvline(total_successes_prior.mean(), color='blue', linestyle=':', linewidth=2,
           label=f'Prior pred mean = {total_successes_prior.mean():.0f}')
# Add 95% CI region
ci_low, ci_high = np.percentile(total_successes_prior, [2.5, 97.5])
ax.axvspan(ci_low, ci_high, alpha=0.2, color='blue', label=f'95% CI [{ci_low:.0f}, {ci_high:.0f}]')
ax.set_xlabel('Total Successes Across All Trials')
ax.set_ylabel('Density')
ax.set_title('Prior Predictive: Total Successes')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Variance inflation factor (overdispersion)
ax = axes[1]
# Remove extreme outliers for better visualization
variance_inflation_plot = valid_var_infl[valid_var_infl < 50]
ax.hist(variance_inflation_plot, bins=50, density=True, alpha=0.6, color='coral', edgecolor='black')
ax.axvline(obs_variance_inflation, color='red', linestyle='--', linewidth=2.5,
           label=f'Observed = {obs_variance_inflation:.2f}\n({var_infl_pct:.1f}th percentile)')
ax.axvline(valid_var_infl.mean(), color='blue', linestyle=':', linewidth=2,
           label=f'Prior pred mean = {valid_var_infl.mean():.2f}')
# Add 95% CI
ci_low, ci_high = np.percentile(valid_var_infl, [2.5, 97.5])
ax.axvspan(ci_low, ci_high, alpha=0.2, color='blue', label=f'95% CI [{ci_low:.1f}, {ci_high:.1f}]')
ax.set_xlabel('Variance Inflation Factor')
ax.set_ylabel('Density')
ax.set_title('Prior Predictive: Overdispersion')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'prior_predictive_coverage.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}prior_predictive_coverage.png")
plt.close()

# PLOT 3: Trial-Level Proportion Comparison
print("\nCreating: trial_level_diagnostics.png")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(N):
    ax = axes[i]

    # Histogram of prior predictive proportions for this trial
    ax.hist(prop_prior[:, i], bins=30, density=True, alpha=0.6,
            color='steelblue', edgecolor='black')

    # Mark observed proportion
    ax.axvline(prop_obs[i], color='red', linestyle='--', linewidth=2,
               label=f'Obs={prop_obs[i]:.3f}')

    # Compute percentile
    pct = compute_percentile(prop_obs[i], prop_prior[:, i])

    ax.set_xlabel('Proportion')
    ax.set_ylabel('Density')
    ax.set_title(f'Trial {i+1}: {r_obs[i]}/{n_obs[i]}\n({pct:.1f}th %ile)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'trial_level_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}trial_level_diagnostics.png")
plt.close()

# PLOT 4: Extreme Values Diagnostic
print("\nCreating: extreme_values_diagnostic.png")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Trial 1 (0/47)
ax = axes[0]
counts, bins, patches = ax.hist(r_prior[:, trial_0_idx], bins=range(0, max(20, r_prior[:, trial_0_idx].max()+2)),
                                 density=True, alpha=0.6, color='steelblue', edgecolor='black')
ax.axvline(r_obs[trial_0_idx], color='red', linestyle='--', linewidth=2.5,
           label=f'Observed r={r_obs[trial_0_idx]}')

# Highlight r=0 bar
if r_obs[trial_0_idx] < len(patches):
    patches[int(r_obs[trial_0_idx])].set_facecolor('red')
    patches[int(r_obs[trial_0_idx])].set_alpha(0.8)

pct = compute_percentile(r_obs[trial_0_idx], r_prior[:, trial_0_idx])
prob_zero = (r_prior[:, trial_0_idx] == 0).mean()

ax.set_xlabel('Number of Successes')
ax.set_ylabel('Density')
ax.set_title(f'Trial 1: n={n_obs[trial_0_idx]}, r={r_obs[trial_0_idx]}\n' +
             f'P(r=0 | prior) = {prob_zero:.3f}, {pct:.1f}th percentile')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: Trial 8 (31/215)
ax = axes[1]
ax.hist(r_prior[:, trial_8_idx], bins=50, density=True, alpha=0.6,
        color='coral', edgecolor='black')
ax.axvline(r_obs[trial_8_idx], color='red', linestyle='--', linewidth=2.5,
           label=f'Observed r={r_obs[trial_8_idx]}')
ax.axvline(r_prior[:, trial_8_idx].mean(), color='blue', linestyle=':', linewidth=2,
           label=f'Prior pred mean={r_prior[:, trial_8_idx].mean():.1f}')

pct = compute_percentile(r_obs[trial_8_idx], r_prior[:, trial_8_idx])
ci_low, ci_high = np.percentile(r_prior[:, trial_8_idx], [2.5, 97.5])

ax.axvspan(ci_low, ci_high, alpha=0.2, color='blue', label=f'95% CI [{ci_low:.0f}, {ci_high:.0f}]')
ax.set_xlabel('Number of Successes')
ax.set_ylabel('Density')
ax.set_title(f'Trial 8: n={n_obs[trial_8_idx]}, r={r_obs[trial_8_idx]}\n' +
             f'{pct:.1f}th percentile')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'extreme_values_diagnostic.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}extreme_values_diagnostic.png")
plt.close()

# PLOT 5: Prior-Data Compatibility Overview
print("\nCreating: prior_data_compatibility.png")
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Top row: Distribution of all proportions
ax1 = fig.add_subplot(gs[0, :])
# Flatten all prior predictive proportions
all_prop_prior = prop_prior.flatten()
ax1.hist(all_prop_prior, bins=100, density=True, alpha=0.4, color='steelblue',
         edgecolor='black', label='Prior Predictive (all trials)')
# Overlay observed proportions as rug plot
for i, p in enumerate(prop_obs):
    ax1.axvline(p, color='red', alpha=0.6, linewidth=2, ymax=0.1)
ax1.scatter(prop_obs, [0.05]*len(prop_obs), color='red', s=100, zorder=5,
           marker='v', label='Observed proportions', edgecolor='black', linewidth=1)
ax1.set_xlabel('Success Proportion')
ax1.set_ylabel('Density')
ax1.set_title('Prior Predictive vs Observed Proportions (All Trials)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.05, 0.5)

# Middle left: Mean proportion across trials
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(mean_proportion_prior, bins=50, density=True, alpha=0.6,
         color='steelblue', edgecolor='black')
ax2.axvline(prop_obs.mean(), color='red', linestyle='--', linewidth=2.5,
           label=f'Observed={prop_obs.mean():.4f}')
pct = compute_percentile(prop_obs.mean(), mean_proportion_prior)
ax2.set_xlabel('Mean Proportion Across Trials')
ax2.set_ylabel('Density')
ax2.set_title(f'Prior Predictive: Mean Proportion\n({pct:.1f}th percentile)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Middle right: SD of proportions
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(sd_proportion_prior, bins=50, density=True, alpha=0.6,
         color='coral', edgecolor='black')
ax3.axvline(prop_obs.std(), color='red', linestyle='--', linewidth=2.5,
           label=f'Observed={prop_obs.std():.4f}')
pct = compute_percentile(prop_obs.std(), sd_proportion_prior)
ax3.set_xlabel('SD of Proportions Across Trials')
ax3.set_ylabel('Density')
ax3.set_title(f'Prior Predictive: Proportion Variability\n({pct:.1f}th percentile)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Bottom left: Q-Q plot
ax4 = fig.add_subplot(gs[2, 0])
# For each observed trial, compute its percentile in the prior predictive
percentiles = [compute_percentile(r_obs[i], r_prior[:, i]) for i in range(N)]
expected_percentiles = np.linspace(100/(N+1), 100*N/(N+1), N)
ax4.scatter(expected_percentiles, sorted(percentiles), s=100, alpha=0.7, color='steelblue', edgecolor='black')
ax4.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect calibration')
ax4.fill_between([0, 100], [5, 5], [95, 95], alpha=0.2, color='green', label='95% region')
ax4.set_xlabel('Expected Percentile (Uniform)')
ax4.set_ylabel('Observed Percentile in Prior Predictive')
ax4.set_title('Calibration: Prior Predictive Coverage')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 100)
ax4.set_ylim(0, 100)

# Bottom right: Sample size vs residual (observed - prior pred mean)
ax5 = fig.add_subplot(gs[2, 1])
prior_pred_means = r_prior.mean(axis=0)
residuals = r_obs - prior_pred_means
ax5.scatter(n_obs, residuals, s=100, alpha=0.7, color='purple', edgecolor='black')
ax5.axhline(0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Sample Size (n)')
ax5.set_ylabel('Residual: Observed - Prior Pred Mean')
ax5.set_title('Prior Predictive Residuals vs Sample Size')
ax5.grid(True, alpha=0.3)

# Annotate extreme points
extreme_trials = np.argsort(np.abs(residuals))[-2:]
for idx in extreme_trials:
    ax5.annotate(f'Trial {idx+1}\n{r_obs[idx]}/{n_obs[idx]}',
                xy=(n_obs[idx], residuals[idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.savefig(OUTPUT_DIR + 'prior_data_compatibility.png', dpi=300, bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR}prior_data_compatibility.png")
plt.close()

# ============================================================================
# FINAL ASSESSMENT
# ============================================================================

print("\n" + "=" * 80)
print("PRIOR PREDICTIVE CHECK ASSESSMENT")
print("=" * 80)

# Criteria for PASS/FAIL
issues = []
warnings_list = []

# Check 1: Observed data coverage
if total_succ_pct < 1 or total_succ_pct > 99:
    issues.append(f"Total successes at extreme percentile ({total_succ_pct:.1f}th)")
elif total_succ_pct < 5 or total_succ_pct > 95:
    warnings_list.append(f"Total successes near boundary ({total_succ_pct:.1f}th percentile)")

# Check 2: Overdispersion coverage
if var_infl_pct < 1 or var_infl_pct > 99:
    issues.append(f"Variance inflation at extreme percentile ({var_infl_pct:.1f}th)")
elif var_infl_pct < 5 or var_infl_pct > 95:
    warnings_list.append(f"Variance inflation near boundary ({var_infl_pct:.1f}th percentile)")

# Check 3: Trial-level extremes
extreme_trial_percentiles = [compute_percentile(r_obs[i], r_prior[:, i]) for i in range(N)]
extreme_trials_count = sum(1 for p in extreme_trial_percentiles if p < 1 or p > 99)
marginal_trials_count = sum(1 for p in extreme_trial_percentiles if (p < 5 or p > 95) and not (p < 1 or p > 99))

if extreme_trials_count > 1:
    issues.append(f"{extreme_trials_count} trials at extreme percentiles (<1 or >99)")
elif extreme_trials_count == 1:
    warnings_list.append(f"1 trial at extreme percentile")

if marginal_trials_count > 3:
    warnings_list.append(f"{marginal_trials_count} trials at marginal percentiles (1-5 or 95-99)")

# Check 4: Computational issues
if extreme_phi.sum() > n_samples * 0.05:  # More than 5% of samples
    issues.append(f"Too many extreme φ values ({extreme_phi.sum()} samples)")

# Check 5: Scientific plausibility
# Check if priors allow impossible proportions
if (prop_prior < 0).any() or (prop_prior > 1).any():
    issues.append("Prior predictive generates impossible proportions")

# Make decision
print("\nChecklist:")
print(f"  [{'✓' if total_succ_pct >= 1 and total_succ_pct <= 99 else '✗'}] Total successes within reasonable range (1-99th percentile)")
print(f"  [{'✓' if var_infl_pct >= 1 and var_infl_pct <= 99 else '✗'}] Overdispersion within reasonable range (1-99th percentile)")
print(f"  [{'✓' if extreme_trials_count <= 1 else '✗'}] At most 1 trial at extreme percentile")
print(f"  [{'✓' if extreme_phi.sum() <= n_samples * 0.05 else '✗'}] No excessive computational issues")
print(f"  [{'✓' if not ((prop_prior < 0).any() or (prop_prior > 1).any()) else '✗'}] All proportions in valid range")

if len(issues) == 0:
    decision = "PASS"
    print(f"\n{'='*80}")
    print(f"DECISION: {decision}")
    print(f"{'='*80}")
    print("\nThe priors are well-specified and generate scientifically plausible data.")
    print("The observed data falls comfortably within the prior predictive distribution.")
    print("Ready to proceed with model fitting.")
    if len(warnings_list) > 0:
        print("\nMinor warnings (non-blocking):")
        for w in warnings_list:
            print(f"  - {w}")
else:
    decision = "FAIL"
    print(f"\n{'='*80}")
    print(f"DECISION: {decision}")
    print(f"{'='*80}")
    print("\nCritical issues detected:")
    for issue in issues:
        print(f"  - {issue}")
    if len(warnings_list) > 0:
        print("\nAdditional warnings:")
        for w in warnings_list:
            print(f"  - {w}")
    print("\nRECOMMENDATIONS:")
    print("The model should be revised before fitting.")

# Save summary statistics to file for the findings report
summary_stats = {
    'decision': decision,
    'issues': issues,
    'warnings': warnings_list,
    'total_successes_percentile': float(total_succ_pct),
    'variance_inflation_percentile': float(var_infl_pct),
    'extreme_trials': int(extreme_trials_count),
    'marginal_trials': int(marginal_trials_count),
    'mu_prior_mean': float(mu_prior.mean()),
    'mu_prior_ci': [float(np.percentile(mu_prior, 2.5)), float(np.percentile(mu_prior, 97.5))],
    'phi_prior_mean': float(phi_prior.mean()),
    'phi_prior_ci': [float(np.percentile(phi_prior, 2.5)), float(np.percentile(phi_prior, 97.5))],
    'total_succ_prior_mean': float(total_successes_prior.mean()),
    'total_succ_prior_ci': [float(np.percentile(total_successes_prior, 2.5)), float(np.percentile(total_successes_prior, 97.5))],
    'total_succ_obs': int(r_obs.sum()),
    'var_infl_prior_mean': float(valid_var_infl.mean()),
    'var_infl_prior_ci': [float(np.percentile(valid_var_infl, 2.5)), float(np.percentile(valid_var_infl, 97.5))],
    'var_infl_obs': float(obs_variance_inflation)
}

import json
with open('/workspace/experiments/experiment_1/prior_predictive_check/summary_stats.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print("\n" + "=" * 80)
print("PRIOR PREDICTIVE CHECK COMPLETE")
print("=" * 80)
print(f"\nOutputs saved to: /workspace/experiments/experiment_1/prior_predictive_check/")
print(f"  - Plots: {OUTPUT_DIR}")
print(f"  - Summary: summary_stats.json")
print(f"\nFinal Decision: {decision}")
