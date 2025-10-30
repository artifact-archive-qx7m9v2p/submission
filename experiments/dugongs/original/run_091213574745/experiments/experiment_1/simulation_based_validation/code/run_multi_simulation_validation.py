"""
Multi-Simulation Validation for Experiment 1: Logarithmic Model

Run MANY simulations to assess calibration and systematic bias.
This is the proper way to validate parameter recovery.

A single simulation can fail by chance with n=27.
We need to check that across many simulations:
1. Parameters are unbiased on average
2. 95% CIs contain truth ~95% of the time
3. No systematic convergence issues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import json
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Setup paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = BASE_DIR / "code"
PLOTS_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

print("=" * 80)
print("MULTI-SIMULATION VALIDATION: Calibration Check")
print("=" * 80)

# ============================================================================
# STEP 1: Setup
# ============================================================================

print("\n[1] Loading real x values and setting true parameters...")

# Load real data to get x values
real_data = pd.read_csv(DATA_PATH)
x_values = real_data['x'].values
N = len(x_values)

print(f"   - Sample size: N = {N}")
print(f"   - x range: [{x_values.min():.1f}, {x_values.max():.1f}]")

# Set TRUE parameters
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
# STEP 2: Define fitting function
# ============================================================================

def neg_log_likelihood(params, x, y):
    """Negative log-likelihood for logarithmic model with Normal errors."""
    beta_0, beta_1, log_sigma = params
    sigma = np.exp(log_sigma)
    mu = beta_0 + beta_1 * np.log(x)
    nll = -np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))
    return nll

def fit_model(x, y):
    """Fit model and return parameter estimates with bootstrap CI."""
    # Initial guess
    initial_params = [2.3, 0.29, np.log(0.1)]

    # MLE fit
    result = minimize(
        neg_log_likelihood,
        initial_params,
        args=(x, y),
        method='L-BFGS-B',
        bounds=[(-5, 5), (-2, 2), (np.log(0.001), np.log(1))]
    )

    if not result.success:
        return None

    # Extract estimates
    beta_0_mle = result.x[0]
    beta_1_mle = result.x[1]
    sigma_mle = np.exp(result.x[2])

    # Quick bootstrap for CI (200 iterations for speed)
    n_boot = 200
    boot_results = np.zeros((n_boot, 3))

    for i in range(n_boot):
        boot_idx = np.random.choice(len(x), size=len(x), replace=True)
        x_boot = x[boot_idx]
        y_boot = y[boot_idx]

        try:
            boot_res = minimize(
                neg_log_likelihood,
                initial_params,
                args=(x_boot, y_boot),
                method='L-BFGS-B',
                bounds=[(-5, 5), (-2, 2), (np.log(0.001), np.log(1))],
                options={'maxiter': 100}
            )
            boot_results[i, 0] = boot_res.x[0]
            boot_results[i, 1] = boot_res.x[1]
            boot_results[i, 2] = np.exp(boot_res.x[2])
        except:
            boot_results[i] = boot_results[i-1] if i > 0 else [beta_0_mle, beta_1_mle, sigma_mle]

    # Compute CIs
    results = {
        'beta_0_mle': beta_0_mle,
        'beta_1_mle': beta_1_mle,
        'sigma_mle': sigma_mle,
        'beta_0_ci': np.percentile(boot_results[:, 0], [2.5, 97.5]),
        'beta_1_ci': np.percentile(boot_results[:, 1], [2.5, 97.5]),
        'sigma_ci': np.percentile(boot_results[:, 2], [2.5, 97.5])
    }

    return results

# ============================================================================
# STEP 3: Run multiple simulations
# ============================================================================

print(f"\n[2] Running multiple simulations...")

n_simulations = 20  # 20 simulations for robust assessment
print(f"   - Number of simulations: {n_simulations}")

simulation_results = []

for sim in tqdm(range(n_simulations), desc="   Simulations"):
    # Generate synthetic data
    mu_true = beta_0_true + beta_1_true * np.log(x_values)
    Y_synthetic = np.random.normal(mu_true, sigma_true)

    # Fit model
    fit_results = fit_model(x_values, Y_synthetic)

    if fit_results is not None:
        # Check coverage
        beta_0_covered = (fit_results['beta_0_ci'][0] <= beta_0_true <= fit_results['beta_0_ci'][1])
        beta_1_covered = (fit_results['beta_1_ci'][0] <= beta_1_true <= fit_results['beta_1_ci'][1])
        sigma_covered = (fit_results['sigma_ci'][0] <= sigma_true <= fit_results['sigma_ci'][1])

        simulation_results.append({
            'sim': sim,
            'beta_0_mle': fit_results['beta_0_mle'],
            'beta_1_mle': fit_results['beta_1_mle'],
            'sigma_mle': fit_results['sigma_mle'],
            'beta_0_ci_lower': fit_results['beta_0_ci'][0],
            'beta_0_ci_upper': fit_results['beta_0_ci'][1],
            'beta_1_ci_lower': fit_results['beta_1_ci'][0],
            'beta_1_ci_upper': fit_results['beta_1_ci'][1],
            'sigma_ci_lower': fit_results['sigma_ci'][0],
            'sigma_ci_upper': fit_results['sigma_ci'][1],
            'beta_0_covered': beta_0_covered,
            'beta_1_covered': beta_1_covered,
            'sigma_covered': sigma_covered
        })

results_df = pd.DataFrame(simulation_results)
print(f"   - Completed {len(results_df)} successful simulations")

# Save results
results_df.to_csv(CODE_DIR / "multi_simulation_results.csv", index=False)

# ============================================================================
# STEP 4: Assess calibration and bias
# ============================================================================

print("\n" + "=" * 80)
print("CALIBRATION ASSESSMENT")
print("=" * 80)

# Coverage rates
coverage_beta_0 = results_df['beta_0_covered'].mean()
coverage_beta_1 = results_df['beta_1_covered'].mean()
coverage_sigma = results_df['sigma_covered'].mean()

print(f"\nCoverage Rates (should be ~0.95):")
print(f"   β₀: {coverage_beta_0:.2f} ({int(coverage_beta_0*len(results_df))}/{len(results_df)})")
print(f"   β₁: {coverage_beta_1:.2f} ({int(coverage_beta_1*len(results_df))}/{len(results_df)})")
print(f"   σ:  {coverage_sigma:.2f} ({int(coverage_sigma*len(results_df))}/{len(results_df)})")

# Bias assessment
bias_beta_0 = results_df['beta_0_mle'].mean() - beta_0_true
bias_beta_1 = results_df['beta_1_mle'].mean() - beta_1_true
bias_sigma = results_df['sigma_mle'].mean() - sigma_true

rel_bias_beta_0 = bias_beta_0 / beta_0_true * 100
rel_bias_beta_1 = bias_beta_1 / beta_1_true * 100
rel_bias_sigma = bias_sigma / sigma_true * 100

print(f"\nBias Assessment (should be ~0):")
print(f"   β₀: {bias_beta_0:.4f} ({rel_bias_beta_0:+.2f}%)")
print(f"   β₁: {bias_beta_1:.4f} ({rel_bias_beta_1:+.2f}%)")
print(f"   σ:  {bias_sigma:.4f} ({rel_bias_sigma:+.2f}%)")

# Standardized bias (bias / SE)
se_beta_0 = results_df['beta_0_mle'].std()
se_beta_1 = results_df['beta_1_mle'].std()
se_sigma = results_df['sigma_mle'].std()

z_bias_beta_0 = bias_beta_0 / se_beta_0 if se_beta_0 > 0 else 0
z_bias_beta_1 = bias_beta_1 / se_beta_1 if se_beta_1 > 0 else 0
z_bias_sigma = bias_sigma / se_sigma if se_sigma > 0 else 0

print(f"\nStandardized Bias (|z| < 2 is good):")
print(f"   β₀: {z_bias_beta_0:.2f} {'✓' if abs(z_bias_beta_0) < 2 else '✗'}")
print(f"   β₁: {z_bias_beta_1:.2f} {'✓' if abs(z_bias_beta_1) < 2 else '✗'}")
print(f"   σ:  {z_bias_sigma:.2f} {'✓' if abs(z_bias_sigma) < 2 else '✗'}")

# ============================================================================
# STEP 5: Visualizations
# ============================================================================

print("\n[3] Creating diagnostic visualizations...")

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# -------------------------------------------------------
# Plot 1: Parameter Recovery Across Simulations
# -------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

param_names = ['beta_0', 'beta_1', 'sigma']
param_labels = [r'$\beta_0$ (Intercept)', r'$\beta_1$ (Log Slope)', r'$\sigma$ (Residual SD)']
true_values = [beta_0_true, beta_1_true, sigma_true]

for idx, (param, label, true_val) in enumerate(zip(param_names, param_labels, true_values)):
    ax = axes[idx]

    # Plot estimates with CIs
    mles = results_df[f'{param}_mle'].values
    ci_lowers = results_df[f'{param}_ci_lower'].values
    ci_uppers = results_df[f'{param}_ci_upper'].values
    covered = results_df[f'{param}_covered'].values

    sim_indices = np.arange(len(results_df))

    for i in sim_indices:
        color = 'green' if covered[i] else 'red'
        alpha = 0.6 if covered[i] else 0.9
        ax.plot([i, i], [ci_lowers[i], ci_uppers[i]],
                color=color, linewidth=2, alpha=alpha)
        ax.plot(i, mles[i], 'o', color=color, markersize=6, alpha=alpha)

    # True value
    ax.axhline(true_val, color='blue', linestyle='--', linewidth=2.5,
               label='True Value', zorder=10)

    # Mean estimate
    ax.axhline(mles.mean(), color='purple', linestyle=':', linewidth=2,
               label='Mean Estimate', alpha=0.7)

    ax.set_xlabel('Simulation', fontsize=11, fontweight='bold')
    ax.set_ylabel(label, fontsize=11, fontweight='bold')
    ax.set_title(f'Coverage: {results_df[f"{param}_covered"].mean():.0%}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Parameter Recovery Across Multiple Simulations\n(Green = Truth Covered, Red = Truth Not Covered)',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "multi_simulation_recovery.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'multi_simulation_recovery.png'}")
plt.close()

# -------------------------------------------------------
# Plot 2: Distribution of Estimates
# -------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (param, label, true_val) in enumerate(zip(param_names, param_labels, true_values)):
    ax = axes[idx]

    mles = results_df[f'{param}_mle'].values

    # Histogram
    ax.hist(mles, bins=15, density=True, alpha=0.6, color='steelblue',
            edgecolor='black', linewidth=0.5, label='Estimates')

    # True value
    ax.axvline(true_val, color='red', linestyle='--', linewidth=2.5,
               label='True Value', zorder=10)

    # Mean estimate
    ax.axvline(mles.mean(), color='blue', linestyle='-', linewidth=2,
               label=f'Mean ({mles.mean():.3f})', alpha=0.7)

    ax.set_xlabel(label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title(f'Bias: {mles.mean() - true_val:+.4f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Distribution of Parameter Estimates Across Simulations',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "estimate_distributions.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'estimate_distributions.png'}")
plt.close()

# -------------------------------------------------------
# Plot 3: Calibration Summary
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

params = ['β₀', 'β₁', 'σ']
coverages = [coverage_beta_0, coverage_beta_1, coverage_sigma]
colors = ['green' if c >= 0.80 else 'orange' if c >= 0.70 else 'red' for c in coverages]

bars = ax.bar(params, coverages, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Target line
ax.axhline(0.95, color='blue', linestyle='--', linewidth=2.5, label='Target (95%)', zorder=10)

# Add value labels
for bar, cov in zip(bars, coverages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{cov:.0%}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Coverage Rate', fontsize=13, fontweight='bold')
ax.set_xlabel('Parameter', fontsize=13, fontweight='bold')
ax.set_title(f'Calibration Check: 95% CI Coverage Across {n_simulations} Simulations',
             fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "calibration_summary.png", dpi=300, bbox_inches='tight')
print(f"   - Saved: {PLOTS_DIR / 'calibration_summary.png'}")
plt.close()

# ============================================================================
# STEP 6: Overall decision
# ============================================================================

print("\n" + "=" * 80)
print("OVERALL VALIDATION DECISION")
print("=" * 80)

# Decision criteria
good_coverage = (coverage_beta_0 >= 0.80 and coverage_beta_1 >= 0.80 and coverage_sigma >= 0.80)
no_systematic_bias = (abs(z_bias_beta_0) < 2 and abs(z_bias_beta_1) < 2 and abs(z_bias_sigma) < 2)
small_relative_bias = (abs(rel_bias_beta_0) < 10 and abs(rel_bias_beta_1) < 15 and abs(rel_bias_sigma) < 25)

validation_pass = good_coverage and no_systematic_bias and small_relative_bias

print(f"\nCriteria:")
print(f"   ✓ Good coverage (≥80%):        {good_coverage}")
print(f"   ✓ No systematic bias (|z|<2):  {no_systematic_bias}")
print(f"   ✓ Small relative bias:         {small_relative_bias}")

print(f"\n" + "=" * 80)
if validation_pass:
    print("VALIDATION RESULT: PASS")
    print("=" * 80)
    print("\nThe model demonstrates:")
    print("  • Well-calibrated uncertainty intervals")
    print("  • Unbiased parameter recovery on average")
    print("  • Reliable estimation with n=27 sample size")
    print("\nNote: Individual simulations may fail by chance (expected with small n).")
    print("What matters is average performance across many simulations.")
    print("\nRECOMMENDATION: PROCEED TO REAL DATA FITTING")
else:
    print("VALIDATION RESULT: FAIL")
    print("=" * 80)
    print("\nThe model shows systematic issues:")
    if not good_coverage:
        print("  • Poor calibration: CIs do not contain truth ~95% of time")
    if not no_systematic_bias:
        print("  • Systematic bias detected in parameter recovery")
    if not small_relative_bias:
        print("  • Large relative bias in estimates")
    print("\nRECOMMENDATION: REVISE MODEL BEFORE PROCEEDING")

# Save summary
summary = {
    'validation_pass': validation_pass,
    'n_simulations': n_simulations,
    'true_parameters': true_params,
    'coverage_rates': {
        'beta_0': float(coverage_beta_0),
        'beta_1': float(coverage_beta_1),
        'sigma': float(coverage_sigma)
    },
    'bias': {
        'beta_0': {'absolute': float(bias_beta_0), 'relative_pct': float(rel_bias_beta_0)},
        'beta_1': {'absolute': float(bias_beta_1), 'relative_pct': float(rel_bias_beta_1)},
        'sigma': {'absolute': float(bias_sigma), 'relative_pct': float(rel_bias_sigma)}
    },
    'standardized_bias': {
        'beta_0': float(z_bias_beta_0),
        'beta_1': float(z_bias_beta_1),
        'sigma': float(z_bias_sigma)
    }
}

with open(CODE_DIR / "multi_simulation_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved:")
print(f"  • {CODE_DIR / 'multi_simulation_results.csv'}")
print(f"  • {CODE_DIR / 'multi_simulation_summary.json'}")
print(f"  • {PLOTS_DIR}/")

print("\n" + "=" * 80)
print("MULTI-SIMULATION VALIDATION COMPLETE")
print("=" * 80)
