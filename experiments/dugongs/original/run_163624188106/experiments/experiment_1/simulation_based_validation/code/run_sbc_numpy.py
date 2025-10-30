"""
Simulation-Based Calibration for Log-Log Linear Model (NumPy implementation)

This script performs SBC using numerical optimization and bootstrapping:
1. Generates N_sim synthetic datasets with known parameters
2. Fits the model using maximum likelihood estimation
3. Uses bootstrap for uncertainty quantification
4. Checks parameter recovery and calibration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import norm, chi2
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SIM = 100  # Number of simulations
N_BOOT = 500  # Bootstrap samples for uncertainty
TRUE_ALPHA = 0.6
TRUE_BETA = 0.13
TRUE_SIGMA = 0.05

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = BASE_DIR / "code"
PLOTS_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

# Load real data to get x values
data_df = pd.read_csv(DATA_PATH)
x_values = data_df['x'].values
N = len(x_values)

print("="*80)
print("SIMULATION-BASED CALIBRATION: Log-Log Linear Model")
print("="*80)
print(f"\nConfiguration:")
print(f"  N observations: {N}")
print(f"  N simulations: {N_SIM}")
print(f"  Bootstrap samples: {N_BOOT}")
print(f"  True parameters:")
print(f"    alpha = {TRUE_ALPHA}")
print(f"    beta = {TRUE_BETA}")
print(f"    sigma = {TRUE_SIGMA}")
print(f"\n  x range: [{x_values.min():.1f}, {x_values.max():.1f}]")

# Transform data
log_x = np.log(x_values)


def neg_log_likelihood(params, log_x, log_y):
    """Negative log-likelihood for the log-log linear model."""
    alpha, beta, log_sigma = params
    sigma = np.exp(log_sigma)  # Ensure sigma > 0

    mu = alpha + beta * log_x
    residuals = log_y - mu

    # Normal log-likelihood
    nll = 0.5 * N * np.log(2 * np.pi) + N * np.log(sigma) + \
          0.5 * np.sum(residuals**2) / sigma**2

    return nll


def fit_model(log_x, log_y):
    """Fit the log-log linear model using MLE."""
    # Initial guess using OLS
    A = np.column_stack([np.ones(len(log_x)), log_x])
    theta_ols = np.linalg.lstsq(A, log_y, rcond=None)[0]
    residuals = log_y - (theta_ols[0] + theta_ols[1] * log_x)
    sigma_ols = np.std(residuals)

    initial_params = [theta_ols[0], theta_ols[1], np.log(sigma_ols)]

    # Optimize
    result = minimize(
        neg_log_likelihood,
        initial_params,
        args=(log_x, log_y),
        method='L-BFGS-B',
        bounds=[(-5, 5), (-1, 1), (np.log(0.001), np.log(1.0))]
    )

    if result.success:
        alpha_hat, beta_hat, log_sigma_hat = result.x
        sigma_hat = np.exp(log_sigma_hat)
        return alpha_hat, beta_hat, sigma_hat
    else:
        return None


def bootstrap_ci(log_x, log_y, n_boot=N_BOOT, alpha_level=0.05):
    """Compute bootstrap confidence intervals."""
    estimates = []

    for _ in range(n_boot):
        # Resample with replacement
        indices = np.random.choice(len(log_y), size=len(log_y), replace=True)
        log_x_boot = log_x[indices]
        log_y_boot = log_y[indices]

        result = fit_model(log_x_boot, log_y_boot)
        if result is not None:
            estimates.append(result)

    estimates = np.array(estimates)

    # Compute percentiles
    lower = np.percentile(estimates, 100 * alpha_level / 2, axis=0)
    upper = np.percentile(estimates, 100 * (1 - alpha_level / 2), axis=0)

    return lower, upper, estimates


print("\n" + "="*80)
print("RUNNING SIMULATIONS")
print("="*80)

# Storage for results
results = {
    'alpha_mean': [],
    'beta_mean': [],
    'sigma_mean': [],
    'alpha_lower': [],
    'alpha_upper': [],
    'beta_lower': [],
    'beta_upper': [],
    'sigma_lower': [],
    'sigma_upper': [],
    'alpha_rank': [],
    'beta_rank': [],
    'sigma_rank': [],
    'converged': [],
    'sim_id': []
}

# Run simulations
for sim_id in range(N_SIM):
    if (sim_id + 1) % 10 == 0:
        print(f"Simulation {sim_id + 1}/{N_SIM}...")

    # Generate synthetic data with known parameters
    log_mu = TRUE_ALPHA + TRUE_BETA * log_x
    log_Y_sim = np.random.normal(log_mu, TRUE_SIGMA)

    # Fit model
    try:
        result = fit_model(log_x, log_Y_sim)
        if result is None:
            continue

        alpha_hat, beta_hat, sigma_hat = result

        # Bootstrap for uncertainty
        lower, upper, boot_samples = bootstrap_ci(log_x, log_Y_sim)

        # Compute ranks using bootstrap samples
        alpha_rank = np.sum(boot_samples[:, 0] < TRUE_ALPHA)
        beta_rank = np.sum(boot_samples[:, 1] < TRUE_BETA)
        sigma_rank = np.sum(boot_samples[:, 2] < TRUE_SIGMA)

        # Store results
        results['alpha_mean'].append(alpha_hat)
        results['beta_mean'].append(beta_hat)
        results['sigma_mean'].append(sigma_hat)
        results['alpha_lower'].append(lower[0])
        results['alpha_upper'].append(upper[0])
        results['beta_lower'].append(lower[1])
        results['beta_upper'].append(upper[1])
        results['sigma_lower'].append(lower[2])
        results['sigma_upper'].append(upper[2])
        results['alpha_rank'].append(alpha_rank)
        results['beta_rank'].append(beta_rank)
        results['sigma_rank'].append(sigma_rank)
        results['converged'].append(True)
        results['sim_id'].append(sim_id)

    except Exception as e:
        print(f"  ERROR in simulation {sim_id}: {e}")
        continue

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv(CODE_DIR / "sbc_results.csv", index=False)

print(f"\nCompleted {len(results_df)}/{N_SIM} simulations successfully")

# Compute summary statistics
print("\n" + "="*80)
print("CALIBRATION METRICS")
print("="*80)

# Coverage rates (should be ~95%)
alpha_coverage = np.mean((results_df['alpha_lower'] <= TRUE_ALPHA) &
                         (results_df['alpha_upper'] >= TRUE_ALPHA))
beta_coverage = np.mean((results_df['beta_lower'] <= TRUE_BETA) &
                        (results_df['beta_upper'] >= TRUE_BETA))
sigma_coverage = np.mean((results_df['sigma_lower'] <= TRUE_SIGMA) &
                         (results_df['sigma_upper'] >= TRUE_SIGMA))

print(f"\n95% Credible Interval Coverage (target: 0.95):")
print(f"  alpha: {alpha_coverage:.3f}")
print(f"  beta:  {beta_coverage:.3f}")
print(f"  sigma: {sigma_coverage:.3f}")

# Bias (posterior means should be close to true values)
alpha_bias = np.mean(results_df['alpha_mean']) - TRUE_ALPHA
beta_bias = np.mean(results_df['beta_mean']) - TRUE_BETA
sigma_bias = np.mean(results_df['sigma_mean']) - TRUE_SIGMA

alpha_rmse = np.sqrt(np.mean((results_df['alpha_mean'] - TRUE_ALPHA)**2))
beta_rmse = np.sqrt(np.mean((results_df['beta_mean'] - TRUE_BETA)**2))
sigma_rmse = np.sqrt(np.mean((results_df['sigma_mean'] - TRUE_SIGMA)**2))

print(f"\nParameter Estimate Bias:")
print(f"  alpha: {alpha_bias:+.4f} (RMSE: {alpha_rmse:.4f})")
print(f"  beta:  {beta_bias:+.4f} (RMSE: {beta_rmse:.4f})")
print(f"  sigma: {sigma_bias:+.4f} (RMSE: {sigma_rmse:.4f})")

print(f"\nRelative Bias (%):")
print(f"  alpha: {100*alpha_bias/TRUE_ALPHA:+.2f}%")
print(f"  beta:  {100*beta_bias/TRUE_BETA:+.2f}%")
print(f"  sigma: {100*sigma_bias/TRUE_SIGMA:+.2f}%")

# Convergence statistics
convergence_rate = np.mean(results_df['converged'])
print(f"\nConvergence:")
print(f"  Success rate: {convergence_rate:.3f}")

# Rank statistics (should be approximately uniform)
print(f"\nRank Statistics Uniformity:")
bins = 20
expected_per_bin = len(results_df) / bins

for param_name, ranks in [('alpha', results_df['alpha_rank']),
                           ('beta', results_df['beta_rank']),
                           ('sigma', results_df['sigma_rank'])]:
    hist, _ = np.histogram(ranks, bins=bins, range=(0, N_BOOT))
    chi2_stat = np.sum((hist - expected_per_bin)**2 / expected_per_bin)
    # For chi-square with 19 df, critical value at p=0.05 is ~30.14
    uniform = chi2_stat < 30.14
    print(f"  {param_name}: chi2 = {chi2_stat:.2f} ({'UNIFORM' if uniform else 'NON-UNIFORM'})")

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Figure 1: Rank histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
params = [
    ('alpha', results_df['alpha_rank'], TRUE_ALPHA, r'$\alpha$'),
    ('beta', results_df['beta_rank'], TRUE_BETA, r'$\beta$'),
    ('sigma', results_df['sigma_rank'], TRUE_SIGMA, r'$\sigma$')
]

for ax, (param_name, ranks, true_val, label) in zip(axes, params):
    ax.hist(ranks, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axhline(len(results_df)/20, color='red', linestyle='--',
               label='Expected (uniform)', linewidth=2)
    ax.set_xlabel('Rank statistic', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{label} Rank Histogram (true = {true_val})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('SBC Rank Histograms: Should be Uniform if Well-Calibrated',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_histograms.png", dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'rank_histograms.png'}")
plt.close()

# Figure 2: Parameter recovery (estimate vs true value)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
params = [
    ('alpha_mean', TRUE_ALPHA, r'$\alpha$'),
    ('beta_mean', TRUE_BETA, r'$\beta$'),
    ('sigma_mean', TRUE_SIGMA, r'$\sigma$')
]

for ax, (col, true_val, label) in zip(axes, params):
    ax.scatter(range(len(results_df)), results_df[col],
               alpha=0.5, s=30, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.axhline(true_val, color='red', linestyle='--', linewidth=2.5,
               label=f'True value = {true_val}')
    mean_est = results_df[col].mean()
    ax.axhline(mean_est, color='green', linestyle=':', linewidth=2,
               label=f'Mean estimate = {mean_est:.4f}')
    ax.set_xlabel('Simulation ID', fontsize=11)
    ax.set_ylabel(f'MLE of {label}', fontsize=11)
    ax.set_title(f'{label} Recovery', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Parameter Recovery: Estimates Should Scatter Around True Value',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "parameter_recovery.png", dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'parameter_recovery.png'}")
plt.close()

# Figure 3: Coverage plots with credible intervals
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
params = [
    ('alpha', TRUE_ALPHA, r'$\alpha$'),
    ('beta', TRUE_BETA, r'$\beta$'),
    ('sigma', TRUE_SIGMA, r'$\sigma$')
]

for ax, (param, true_val, label) in zip(axes, params):
    lower_col = f'{param}_lower'
    upper_col = f'{param}_upper'
    mean_col = f'{param}_mean'

    # Color intervals by whether they contain true value
    contains_true = ((results_df[lower_col] <= true_val) &
                     (results_df[upper_col] >= true_val))

    # Plot intervals (sample every 2nd to avoid clutter)
    sample_indices = range(0, len(results_df), 2)
    for i in sample_indices:
        color = 'green' if contains_true.iloc[i] else 'red'
        alpha_val = 0.3 if contains_true.iloc[i] else 0.6
        ax.plot([i, i], [results_df[lower_col].iloc[i], results_df[upper_col].iloc[i]],
                color=color, alpha=alpha_val, linewidth=1.5)
        # Plot point estimate
        ax.plot(i, results_df[mean_col].iloc[i], 'o', color=color,
                markersize=3, alpha=0.5)

    ax.axhline(true_val, color='blue', linestyle='--', linewidth=2.5,
               label=f'True value = {true_val}')
    ax.set_xlabel('Simulation ID (every 2nd shown)', fontsize=11)
    ax.set_ylabel(f'95% CI for {label}', fontsize=11)
    coverage_pct = contains_true.mean()
    color_status = 'green' if 0.90 <= coverage_pct <= 0.98 else 'red'
    ax.set_title(f'{label} Coverage: {coverage_pct:.1%}',
                 fontsize=12, fontweight='bold', color=color_status)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Coverage Assessment: 95% CIs Should Contain True Value ~95% of Time',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_intervals.png", dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'coverage_intervals.png'}")
plt.close()

# Figure 4: Bias and RMSE visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
params = [
    ('alpha_mean', TRUE_ALPHA, r'$\alpha$', alpha_bias, alpha_rmse),
    ('beta_mean', TRUE_BETA, r'$\beta$', beta_bias, beta_rmse),
    ('sigma_mean', TRUE_SIGMA, r'$\sigma$', sigma_bias, sigma_rmse)
]

for ax, (col, true_val, label, bias, rmse) in zip(axes, params):
    # Distribution of estimates
    ax.hist(results_df[col], bins=30, alpha=0.6, color='steelblue',
            edgecolor='black', label='Estimates')

    # True value line
    ax.axvline(true_val, color='red', linestyle='--', linewidth=2.5,
               label=f'True = {true_val}')

    # Mean estimate line
    mean_est = results_df[col].mean()
    ax.axvline(mean_est, color='green', linestyle=':', linewidth=2,
               label=f'Mean = {mean_est:.4f}')

    ax.set_xlabel(f'{label} estimate', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    bias_pct = 100 * bias / true_val
    color_status = 'green' if abs(bias_pct) < 10 else 'red'
    ax.set_title(f'{label}: Bias = {bias:+.4f} ({bias_pct:+.1f}%)',
                 fontsize=12, fontweight='bold', color=color_status)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Bias Assessment: Distribution of Estimates Should Center on True Value',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "bias_assessment.png", dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'bias_assessment.png'}")
plt.close()

# Figure 5: Combined diagnostic summary
# Redefine params for comprehensive plot
params_ranks = [
    ("alpha", results_df["alpha_rank"], TRUE_ALPHA, r"$\alpha$"),
    ("beta", results_df["beta_rank"], TRUE_BETA, r"$\beta$"),
    ("sigma", results_df["sigma_rank"], TRUE_SIGMA, r"$\sigma$")
]

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Rank histograms
for i, (param_name, ranks, true_val, label) in enumerate(params_ranks):
    ax = fig.add_subplot(gs[0, i])
    ax.hist(ranks, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axhline(len(results_df)/20, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{label} Ranks', fontsize=11, fontweight='bold')
    ax.set_xlabel('Rank', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.grid(alpha=0.3)

# Row 2: Parameter recovery scatter
params_recovery = [
    ('alpha_mean', TRUE_ALPHA, r'$\alpha$'),
    ('beta_mean', TRUE_BETA, r'$\beta$'),
    ('sigma_mean', TRUE_SIGMA, r'$\sigma$')
]
for i, (col, true_val, label) in enumerate(params_recovery):
    ax = fig.add_subplot(gs[1, i])
    ax.scatter(range(len(results_df)), results_df[col],
               alpha=0.5, s=20, color='steelblue')
    ax.axhline(true_val, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{label} Recovery', fontsize=11, fontweight='bold')
    ax.set_xlabel('Sim ID', fontsize=9)
    ax.set_ylabel(f'{label} estimate', fontsize=9)
    ax.grid(alpha=0.3)

# Row 3: Coverage intervals (subset)
params_coverage = [
    ('alpha', TRUE_ALPHA, r'$\alpha$'),
    ('beta', TRUE_BETA, r'$\beta$'),
    ('sigma', TRUE_SIGMA, r'$\sigma$')
]
for i, (param, true_val, label) in enumerate(params_coverage):
    ax = fig.add_subplot(gs[2, i])
    lower_col = f'{param}_lower'
    upper_col = f'{param}_upper'

    # Show first 30 simulations
    subset = results_df.head(30)
    contains_true = ((subset[lower_col] <= true_val) &
                     (subset[upper_col] >= true_val))

    for j in range(len(subset)):
        color = 'green' if contains_true.iloc[j] else 'red'
        ax.plot([j, j], [subset[lower_col].iloc[j], subset[upper_col].iloc[j]],
                color=color, alpha=0.4, linewidth=2)

    ax.axhline(true_val, color='blue', linestyle='--', linewidth=2)
    ax.set_title(f'{label} Coverage', fontsize=11, fontweight='bold')
    ax.set_xlabel('Sim ID (first 30)', fontsize=9)
    ax.set_ylabel(f'{label} 95% CI', fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('Simulation-Based Calibration: Comprehensive Diagnostics',
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig(PLOTS_DIR / "comprehensive_diagnostics.png", dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'comprehensive_diagnostics.png'}")
plt.close()

print("\n" + "="*80)
print("PASS/FAIL DECISION")
print("="*80)

# Decision criteria
pass_criteria = {
    'Coverage in [0.90, 0.98]': all([0.90 <= cov <= 0.98 for cov in
                                      [alpha_coverage, beta_coverage, sigma_coverage]]),
    'Relative bias < 10%': all([abs(bias) < 0.10 for bias in
                                 [alpha_bias/TRUE_ALPHA, beta_bias/TRUE_BETA,
                                  sigma_bias/TRUE_SIGMA]]),
    'Convergence rate > 95%': convergence_rate > 0.95,
}

# Check rank uniformity
rank_checks = []
for ranks in [results_df['alpha_rank'], results_df['beta_rank'], results_df['sigma_rank']]:
    hist, _ = np.histogram(ranks, bins=20, range=(0, N_BOOT))
    chi2_stat = np.sum((hist - expected_per_bin)**2 / expected_per_bin)
    rank_checks.append(chi2_stat < 30.14)

pass_criteria['Ranks approximately uniform'] = all(rank_checks)

print("\nCriteria Assessment:")
for criterion, passed in pass_criteria.items():
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    print(f"  [{status}] {criterion}")

overall_pass = all(pass_criteria.values())
print(f"\n{'='*80}")
if overall_pass:
    print(f"OVERALL DECISION: PASS")
    print(f"{'='*80}")
    print("\nThe model successfully recovers known parameters from synthetic data.")
    print("Calibration is appropriate. PROCEED to fitting real data.")
else:
    print(f"OVERALL DECISION: FAIL")
    print(f"{'='*80}")
    print("\nWARNING: The model shows calibration issues.")
    print("Review the specific failures above before proceeding.")

print("\n" + "="*80)
print("DETAILED METRICS")
print("="*80)

print(f"\nCoverage Rates (Target: 0.95):")
print(f"  alpha: {alpha_coverage:.3f} ({'PASS' if 0.90 <= alpha_coverage <= 0.98 else 'FAIL'})")
print(f"  beta:  {beta_coverage:.3f} ({'PASS' if 0.90 <= beta_coverage <= 0.98 else 'FAIL'})")
print(f"  sigma: {sigma_coverage:.3f} ({'PASS' if 0.90 <= sigma_coverage <= 0.98 else 'FAIL'})")

print(f"\nRelative Bias (Target: < 10%):")
alpha_rel_bias = 100*alpha_bias/TRUE_ALPHA
beta_rel_bias = 100*beta_bias/TRUE_BETA
sigma_rel_bias = 100*sigma_bias/TRUE_SIGMA
print(f"  alpha: {alpha_rel_bias:+.2f}% ({'PASS' if abs(alpha_rel_bias) < 10 else 'FAIL'})")
print(f"  beta:  {beta_rel_bias:+.2f}% ({'PASS' if abs(beta_rel_bias) < 10 else 'FAIL'})")
print(f"  sigma: {sigma_rel_bias:+.2f}% ({'PASS' if abs(sigma_rel_bias) < 10 else 'FAIL'})")

print(f"\nRMSE (Absolute Error):")
print(f"  alpha: {alpha_rmse:.4f}")
print(f"  beta:  {beta_rmse:.4f}")
print(f"  sigma: {sigma_rmse:.4f}")

# Save results to file
print(f"\nResults saved to: {CODE_DIR / 'sbc_results.csv'}")
print("SBC analysis complete!")
