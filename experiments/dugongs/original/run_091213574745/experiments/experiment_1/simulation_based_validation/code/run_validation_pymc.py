"""
Simulation-Based Validation for Experiment 1: Logarithmic Model

Test if MCMC can recover known parameters when truth is known.
This is a critical safety check before fitting real data.

Using PyMC for robust MCMC sampling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

# Setup paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = BASE_DIR / "code"
PLOTS_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

# Ensure plots directory exists
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("SIMULATION-BASED VALIDATION: Parameter Recovery Test")
print("=" * 80)

# ============================================================================
# STEP 1: Load real x values and set true parameters
# ============================================================================

print("\n[1] Loading real x values and setting true parameters...")

# Load real data to get x values
real_data = pd.read_csv(DATA_PATH)
x_values = real_data['x'].values
N = len(x_values)

print(f"   - Sample size: N = {N}")
print(f"   - x range: [{x_values.min():.1f}, {x_values.max():.1f}]")

# Set TRUE parameters (close to EDA estimates for realism)
beta_0_true = 2.3
beta_1_true = 0.29
sigma_true = 0.09

true_params = {
    'beta_0': beta_0_true,
    'beta_1': beta_1_true,
    'sigma': sigma_true
}

print(f"\n   TRUE PARAMETERS:")
print(f"   - β₀ (intercept):     {beta_0_true}")
print(f"   - β₁ (log slope):     {beta_1_true}")
print(f"   - σ (residual SD):    {sigma_true}")

# ============================================================================
# STEP 2: Generate synthetic data from the model
# ============================================================================

print("\n[2] Generating synthetic data from known model...")

# Generate synthetic Y values
mu_true = beta_0_true + beta_1_true * np.log(x_values)
Y_synthetic = np.random.normal(mu_true, sigma_true)

print(f"   - Generated {N} synthetic observations")
print(f"   - Y range: [{Y_synthetic.min():.3f}, {Y_synthetic.max():.3f}]")
print(f"   - Mean Y: {Y_synthetic.mean():.3f}")
print(f"   - SD Y: {Y_synthetic.std():.3f}")

# Save synthetic data for inspection
synthetic_data = pd.DataFrame({
    'x': x_values,
    'Y_synthetic': Y_synthetic,
    'mu_true': mu_true
})
synthetic_data.to_csv(CODE_DIR / "synthetic_data.csv", index=False)
print(f"   - Saved to: {CODE_DIR / 'synthetic_data.csv'}")

# ============================================================================
# STEP 3: Fit model to synthetic data using PyMC
# ============================================================================

print("\n[3] Fitting model to synthetic data with MCMC...")

# Build PyMC model
with pm.Model() as model:
    # Priors
    beta_0 = pm.Normal('beta_0', mu=2.3, sigma=0.3)
    beta_1 = pm.Normal('beta_1', mu=0.29, sigma=0.15)
    sigma = pm.Exponential('sigma', lam=10)

    # Expected value
    mu = beta_0 + beta_1 * pm.math.log(x_values)

    # Likelihood
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y_synthetic)

    # Sample from posterior
    print("   - Running MCMC (4 chains, 2000 iterations each)...")
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        random_seed=42,
        return_inferencedata=True,
        progressbar=True
    )

    # Sample posterior predictive
    print("   - Sampling posterior predictive...")
    posterior_predictive = pm.sample_posterior_predictive(
        trace,
        random_seed=42,
        progressbar=False
    )

print("   - MCMC complete!")

# ============================================================================
# STEP 4: Extract posterior samples and compute diagnostics
# ============================================================================

print("\n[4] Extracting posterior samples and computing diagnostics...")

# Extract summary statistics
summary = az.summary(
    trace,
    var_names=['beta_0', 'beta_1', 'sigma'],
    kind='stats'
)
print("\n" + "=" * 80)
print("POSTERIOR SUMMARY")
print("=" * 80)
print(summary)

# ============================================================================
# STEP 5: Check parameter recovery
# ============================================================================

print("\n" + "=" * 80)
print("PARAMETER RECOVERY ASSESSMENT")
print("=" * 80)

recovery_results = {}

for param_name, true_value in true_params.items():
    samples = trace.posterior[param_name].values.flatten()

    mean_est = samples.mean()
    median_est = np.median(samples)
    sd_est = samples.std()

    # 95% credible interval
    ci_lower = np.percentile(samples, 2.5)
    ci_upper = np.percentile(samples, 97.5)

    # Coverage: is true value in 95% CI?
    in_ci = ci_lower <= true_value <= ci_upper

    # Standardized error: |mean - true| / SD
    z_score = abs(mean_est - true_value) / sd_est

    # Bias
    bias = mean_est - true_value
    relative_bias = bias / true_value * 100

    recovery_results[param_name] = {
        'true': true_value,
        'mean': mean_est,
        'median': median_est,
        'sd': sd_est,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'in_ci': in_ci,
        'z_score': z_score,
        'bias': bias,
        'relative_bias_pct': relative_bias
    }

    print(f"\n{param_name}:")
    print(f"   True value:       {true_value:.4f}")
    print(f"   Posterior mean:   {mean_est:.4f}")
    print(f"   Posterior median: {median_est:.4f}")
    print(f"   Posterior SD:     {sd_est:.4f}")
    print(f"   95% CI:           [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"   In CI?            {in_ci} {'✓' if in_ci else '✗ FAIL'}")
    print(f"   Bias:             {bias:.4f} ({relative_bias:.2f}%)")
    print(f"   |z-score|:        {z_score:.2f} {'✓' if z_score < 2 else '✗ WARNING'}")

# ============================================================================
# STEP 6: Convergence diagnostics
# ============================================================================

print("\n" + "=" * 80)
print("CONVERGENCE DIAGNOSTICS")
print("=" * 80)

# Get detailed diagnostics from ArviZ
convergence_summary = az.summary(
    trace,
    var_names=['beta_0', 'beta_1', 'sigma'],
    kind='diagnostics'
)
print("\nDetailed Convergence Metrics:")
print(convergence_summary)

# Extract key diagnostics
convergence_results = {}
for param in ['beta_0', 'beta_1', 'sigma']:
    rhat = convergence_summary.loc[param, 'r_hat']
    ess_bulk = convergence_summary.loc[param, 'ess_bulk']
    ess_tail = convergence_summary.loc[param, 'ess_tail']

    convergence_results[param] = {
        'r_hat': rhat,
        'ess_bulk': ess_bulk,
        'ess_tail': ess_tail,
        'r_hat_pass': rhat < 1.01,
        'ess_bulk_pass': ess_bulk > 400,
        'ess_tail_pass': ess_tail > 400
    }

    print(f"\n{param}:")
    print(f"   R̂:        {rhat:.4f} {'✓' if rhat < 1.01 else '✗ FAIL'}")
    print(f"   ESS bulk: {ess_bulk:.0f} {'✓' if ess_bulk > 400 else '✗ FAIL'}")
    print(f"   ESS tail: {ess_tail:.0f} {'✓' if ess_tail > 400 else '✗ FAIL'}")

# Check for divergences
divergence_info = trace.sample_stats.diverging.sum().item()
total_samples = len(trace.posterior.chain) * len(trace.posterior.draw)
divergence_rate = divergence_info / total_samples * 100

print(f"\nDivergences: {divergence_info} / {total_samples} ({divergence_rate:.2f}%)")
print(f"   Status: {'✓' if divergence_rate < 1.0 else '✗ WARNING' if divergence_rate < 5.0 else '✗ FAIL'}")

# ============================================================================
# STEP 7: Visualizations
# ============================================================================

print("\n[5] Creating diagnostic visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# -------------------------------------------------------
# Plot 1: Parameter Recovery (True vs Posterior)
# -------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

param_labels = {
    'beta_0': r'$\beta_0$ (Intercept)',
    'beta_1': r'$\beta_1$ (Log Slope)',
    'sigma': r'$\sigma$ (Residual SD)'
}

for idx, (param_name, true_value) in enumerate(true_params.items()):
    ax = axes[idx]
    samples = trace.posterior[param_name].values.flatten()

    # Histogram of posterior
    ax.hist(samples, bins=50, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', linewidth=0.5)

    # KDE
    from scipy import stats
    kde = stats.gaussian_kde(samples)
    x_range = np.linspace(samples.min(), samples.max(), 200)
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='Posterior')

    # True value
    ax.axvline(true_value, color='red', linestyle='--', linewidth=2.5,
               label='True Value', zorder=10)

    # 95% CI
    ci_lower = recovery_results[param_name]['ci_lower']
    ci_upper = recovery_results[param_name]['ci_upper']
    ax.axvline(ci_lower, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(ci_upper, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    # Posterior mean
    mean_est = recovery_results[param_name]['mean']
    ax.axvline(mean_est, color='blue', linestyle='-', linewidth=2, alpha=0.7,
               label='Posterior Mean')

    ax.set_xlabel(param_labels[param_name], fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9, loc='best')

    # Color code title based on recovery success
    title_color = 'green' if recovery_results[param_name]['in_ci'] else 'red'
    recovery_status = '✓ RECOVERED' if recovery_results[param_name]['in_ci'] else '✗ FAILED'
    ax.set_title(f"{recovery_status}", fontsize=11, fontweight='bold', color=title_color)

plt.suptitle('Parameter Recovery Assessment', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_recovery.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'parameter_recovery.png'}")
plt.close()

# -------------------------------------------------------
# Plot 2: Prior vs Posterior
# -------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Prior specifications
priors = {
    'beta_0': ('normal', [2.3, 0.3]),
    'beta_1': ('normal', [0.29, 0.15]),
    'sigma': ('exponential', [10])  # rate parameter
}

for idx, (param_name, (dist_type, params)) in enumerate(priors.items()):
    ax = axes[idx]
    samples = trace.posterior[param_name].values.flatten()

    # Posterior
    ax.hist(samples, bins=50, density=True, alpha=0.5, color='blue',
            label='Posterior', edgecolor='black', linewidth=0.5)

    # Prior
    if dist_type == 'normal':
        x_range = np.linspace(params[0] - 4*params[1], params[0] + 4*params[1], 200)
        prior_density = stats.norm.pdf(x_range, params[0], params[1])
        ax.plot(x_range, prior_density, 'r--', linewidth=2, label='Prior')
    elif dist_type == 'exponential':
        x_range = np.linspace(0, 0.5, 200)
        prior_density = stats.expon.pdf(x_range, scale=1/params[0])
        ax.plot(x_range, prior_density, 'r--', linewidth=2, label='Prior')

    # True value
    true_value = true_params[param_name]
    ax.axvline(true_value, color='green', linestyle='-.', linewidth=2.5,
               label='True Value', zorder=10)

    ax.set_xlabel(param_labels[param_name], fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10, loc='best')
    ax.set_title('Prior vs Posterior Learning', fontsize=11, fontweight='bold')

plt.suptitle('Bayesian Learning: Prior Updated by Data', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "prior_posterior_comparison.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'prior_posterior_comparison.png'}")
plt.close()

# -------------------------------------------------------
# Plot 3: Trace Plots (Convergence Visual Check)
# -------------------------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for idx, param_name in enumerate(['beta_0', 'beta_1', 'sigma']):
    # Trace plot
    ax_trace = axes[idx, 0]
    for chain in range(4):
        chain_samples = trace.posterior[param_name].sel(chain=chain).values
        ax_trace.plot(chain_samples, alpha=0.6, linewidth=0.5, label=f'Chain {chain}')

    ax_trace.axhline(true_params[param_name], color='red', linestyle='--',
                     linewidth=2, label='True Value')
    ax_trace.set_ylabel(param_labels[param_name], fontsize=11, fontweight='bold')
    ax_trace.set_xlabel('Iteration', fontsize=10)
    ax_trace.legend(fontsize=8, loc='best', ncol=5)
    ax_trace.set_title('Trace Plot', fontsize=10, fontweight='bold')

    # Density plot by chain
    ax_density = axes[idx, 1]
    for chain in range(4):
        chain_samples = trace.posterior[param_name].sel(chain=chain).values
        ax_density.hist(chain_samples, bins=30, alpha=0.4,
                       density=True, label=f'Chain {chain}')

    ax_density.axvline(true_params[param_name], color='red', linestyle='--',
                      linewidth=2, label='True Value')
    ax_density.set_xlabel(param_labels[param_name], fontsize=11, fontweight='bold')
    ax_density.set_ylabel('Density', fontsize=10)
    ax_density.legend(fontsize=8, loc='best')
    ax_density.set_title('Posterior by Chain', fontsize=10, fontweight='bold')

plt.suptitle('MCMC Convergence Diagnostics', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_diagnostics.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'convergence_diagnostics.png'}")
plt.close()

# -------------------------------------------------------
# Plot 4: Pairs Plot (Posterior Correlations)
# -------------------------------------------------------

print("   - Creating pairs plot (this may take a moment)...")
az.plot_pair(
    trace,
    var_names=['beta_0', 'beta_1', 'sigma'],
    kind='kde',
    marginals=True,
    figsize=(10, 10),
    divergences=True
)
plt.suptitle('Posterior Correlation Structure', fontsize=14, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / "posterior_pairs.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'posterior_pairs.png'}")
plt.close()

# -------------------------------------------------------
# Plot 5: Fit to Synthetic Data
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

# Get posterior samples for plotting
beta_0_samples = trace.posterior['beta_0'].values.flatten()[:100]
beta_1_samples = trace.posterior['beta_1'].values.flatten()[:100]

# Create smooth x range for plotting
x_plot = np.linspace(x_values.min(), x_values.max(), 200)

# Plot posterior predictive draws
for i in range(100):  # Show 100 draws
    mu_draw = beta_0_samples[i] + beta_1_samples[i] * np.log(x_plot)
    ax.plot(x_plot, mu_draw, 'b-', alpha=0.02, linewidth=0.5)

# Plot true mean function
mu_true_plot = beta_0_true + beta_1_true * np.log(x_plot)
ax.plot(x_plot, mu_true_plot, 'r--', linewidth=3, label='True Mean Function', zorder=10)

# Plot synthetic data
ax.scatter(x_values, Y_synthetic, c='black', s=80, alpha=0.7,
          edgecolors='white', linewidth=1.5, label='Synthetic Data', zorder=5)

# Plot posterior mean function
mu_post_mean = (recovery_results['beta_0']['mean'] +
                recovery_results['beta_1']['mean'] * np.log(x_plot))
ax.plot(x_plot, mu_post_mean, 'b-', linewidth=3, label='Posterior Mean Function', zorder=8)

ax.set_xlabel('x', fontsize=13, fontweight='bold')
ax.set_ylabel('Y', fontsize=13, fontweight='bold')
ax.set_title('Model Fit to Synthetic Data\n(Blue: Posterior Draws, Red: True Function)',
            fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "synthetic_data_fit.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'synthetic_data_fit.png'}")
plt.close()

# ============================================================================
# STEP 8: Overall Assessment and Decision
# ============================================================================

print("\n" + "=" * 80)
print("OVERALL VALIDATION ASSESSMENT")
print("=" * 80)

# Check all criteria
all_in_ci = all(recovery_results[p]['in_ci'] for p in true_params.keys())
all_z_scores_ok = all(recovery_results[p]['z_score'] < 2 for p in true_params.keys())
all_rhat_ok = all(convergence_results[p]['r_hat_pass'] for p in convergence_results.keys())
all_ess_ok = all(
    convergence_results[p]['ess_bulk_pass'] and convergence_results[p]['ess_tail_pass']
    for p in convergence_results.keys()
)
divergences_ok = divergence_rate < 1.0

# Overall decision
validation_pass = (all_in_ci and all_z_scores_ok and all_rhat_ok and
                  all_ess_ok and divergences_ok)

print(f"\nParameter Recovery Tests:")
print(f"   - All true values in 95% CIs:  {all_in_ci} {'✓' if all_in_ci else '✗'}")
print(f"   - All |z-scores| < 2:         {all_z_scores_ok} {'✓' if all_z_scores_ok else '✗'}")

print(f"\nConvergence Tests:")
print(f"   - All R̂ < 1.01:               {all_rhat_ok} {'✓' if all_rhat_ok else '✗'}")
print(f"   - All ESS > 400:              {all_ess_ok} {'✓' if all_ess_ok else '✗'}")
print(f"   - Divergences < 1%:           {divergences_ok} {'✓' if divergences_ok else '✗'} ({divergence_rate:.2f}%)")

print(f"\n" + "=" * 80)
if validation_pass:
    print("VALIDATION RESULT: PASS")
    print("=" * 80)
    print("\nThe model successfully recovered all true parameters with proper")
    print("uncertainty quantification and no computational issues.")
    print("\nRECOMMENDATION: PROCEED TO REAL DATA FITTING")
else:
    print("VALIDATION RESULT: FAIL")
    print("=" * 80)
    print("\nThe model failed to recover known parameters. Issues detected:")
    if not all_in_ci:
        print("   - True values outside 95% credible intervals")
    if not all_z_scores_ok:
        print("   - Large standardized errors (|z| > 2)")
    if not all_rhat_ok:
        print("   - Poor convergence (R̂ > 1.01)")
    if not all_ess_ok:
        print("   - Low effective sample size (ESS < 400)")
    if not divergences_ok:
        print("   - High divergence rate (> 1%)")
    print("\nRECOMMENDATION: DO NOT PROCEED - REVISE MODEL OR SAMPLING STRATEGY")

# Save results to JSON
results_dict = {
    'validation_pass': validation_pass,
    'true_parameters': true_params,
    'recovery_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                            for kk, vv in v.items()}
                        for k, v in recovery_results.items()},
    'convergence_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                               for kk, vv in v.items()}
                           for k, v in convergence_results.items()},
    'divergence_rate_pct': float(divergence_rate),
    'total_samples': int(total_samples),
    'num_divergences': int(divergence_info)
}

with open(CODE_DIR / "validation_results.json", 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\nResults saved to: {CODE_DIR / 'validation_results.json'}")
print(f"All plots saved to: {PLOTS_DIR}")

print("\n" + "=" * 80)
print("SIMULATION-BASED VALIDATION COMPLETE")
print("=" * 80)
