"""
Simulation-Based Calibration for Log-Log Linear Model

This script:
1. Generates N_sim synthetic datasets with known parameters
2. Fits the model to each synthetic dataset
3. Checks if posteriors recover the true parameters
4. Assesses calibration via coverage rates and rank statistics
"""

import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SIM = 100  # Number of simulations
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
print(f"  True parameters:")
print(f"    alpha = {TRUE_ALPHA}")
print(f"    beta = {TRUE_BETA}")
print(f"    sigma = {TRUE_SIGMA}")
print(f"\n  x range: [{x_values.min():.1f}, {x_values.max():.1f}]")

# Compile Stan model
print("\n" + "="*80)
print("COMPILING STAN MODEL")
print("="*80)
stan_file = CODE_DIR / "model.stan"
model = cmdstanpy.CmdStanModel(stan_file=stan_file)
print(f"Model compiled successfully: {stan_file}")

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
    'rhat_max': [],
    'ess_min': [],
    'converged': [],
    'sim_id': []
}

# Run simulations
print("\n" + "="*80)
print("RUNNING SIMULATIONS")
print("="*80)

log_x = np.log(x_values)

for sim_id in range(N_SIM):
    if (sim_id + 1) % 10 == 0:
        print(f"Simulation {sim_id + 1}/{N_SIM}...")

    # Generate synthetic data with known parameters
    log_mu = TRUE_ALPHA + TRUE_BETA * log_x
    log_Y_sim = np.random.normal(log_mu, TRUE_SIGMA)
    Y_sim = np.exp(log_Y_sim)

    # Prepare data for Stan
    stan_data = {
        'N': N,
        'x': x_values.tolist(),
        'Y': Y_sim.tolist()
    }

    # Fit model
    try:
        fit = model.sample(
            data=stan_data,
            chains=4,
            iter_warmup=1000,
            iter_sampling=1000,
            show_console=False,
            refresh=0,
            seed=42 + sim_id
        )

        # Extract samples
        alpha_samples = fit.stan_variable('alpha')
        beta_samples = fit.stan_variable('beta')
        sigma_samples = fit.stan_variable('sigma')

        # Compute posterior summaries
        alpha_mean = np.mean(alpha_samples)
        beta_mean = np.mean(beta_samples)
        sigma_mean = np.mean(sigma_samples)

        alpha_lower, alpha_upper = np.percentile(alpha_samples, [2.5, 97.5])
        beta_lower, beta_upper = np.percentile(beta_samples, [2.5, 97.5])
        sigma_lower, sigma_upper = np.percentile(sigma_samples, [2.5, 97.5])

        # Compute ranks (for SBC rank statistics)
        # Rank = number of posterior samples less than true value
        alpha_rank = np.sum(alpha_samples < TRUE_ALPHA)
        beta_rank = np.sum(beta_samples < TRUE_BETA)
        sigma_rank = np.sum(sigma_samples < TRUE_SIGMA)

        # Convergence diagnostics
        summary_df = fit.summary()
        rhat_max = summary_df['R_hat'].max()
        ess_min = summary_df['N_Eff'].min()
        converged = (rhat_max < 1.01) and (ess_min > 400)

        # Store results
        results['alpha_mean'].append(alpha_mean)
        results['beta_mean'].append(beta_mean)
        results['sigma_mean'].append(sigma_mean)
        results['alpha_lower'].append(alpha_lower)
        results['alpha_upper'].append(alpha_upper)
        results['beta_lower'].append(beta_lower)
        results['beta_upper'].append(beta_upper)
        results['sigma_lower'].append(sigma_lower)
        results['sigma_upper'].append(sigma_upper)
        results['alpha_rank'].append(alpha_rank)
        results['beta_rank'].append(beta_rank)
        results['sigma_rank'].append(sigma_rank)
        results['rhat_max'].append(rhat_max)
        results['ess_min'].append(ess_min)
        results['converged'].append(converged)
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

print(f"\nPosterior Mean Bias:")
print(f"  alpha: {alpha_bias:+.4f} (RMSE: {alpha_rmse:.4f})")
print(f"  beta:  {beta_bias:+.4f} (RMSE: {beta_rmse:.4f})")
print(f"  sigma: {sigma_bias:+.4f} (RMSE: {sigma_rmse:.4f})")

print(f"\nRelative Bias (%):")
print(f"  alpha: {100*alpha_bias/TRUE_ALPHA:+.2f}%")
print(f"  beta:  {100*beta_bias/TRUE_BETA:+.2f}%")
print(f"  sigma: {100*sigma_bias/TRUE_SIGMA:+.2f}%")

# Convergence statistics
convergence_rate = np.mean(results_df['converged'])
print(f"\nConvergence Diagnostics:")
print(f"  Convergence rate: {convergence_rate:.3f}")
print(f"  Mean max R-hat: {results_df['rhat_max'].mean():.4f}")
print(f"  Mean min ESS: {results_df['ess_min'].mean():.0f}")

failed_convergence = results_df[~results_df['converged']]
if len(failed_convergence) > 0:
    print(f"  WARNING: {len(failed_convergence)} simulations failed convergence")

# Rank statistics (should be approximately uniform)
print(f"\nRank Statistics Uniformity:")
total_samples = 4000  # 4 chains * 1000 samples
bins = 20
expected_per_bin = len(results_df) / bins

for param_name, ranks in [('alpha', results_df['alpha_rank']),
                           ('beta', results_df['beta_rank']),
                           ('sigma', results_df['sigma_rank'])]:
    hist, _ = np.histogram(ranks, bins=bins, range=(0, total_samples))
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
               label='Expected (uniform)')
    ax.set_xlabel('Rank statistic', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{label} (true = {true_val})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_histograms.png", dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'rank_histograms.png'}")
plt.close()

# Figure 2: Parameter recovery (posterior mean vs true value)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
params = [
    ('alpha_mean', TRUE_ALPHA, r'$\alpha$'),
    ('beta_mean', TRUE_BETA, r'$\beta$'),
    ('sigma_mean', TRUE_SIGMA, r'$\sigma$')
]

for ax, (col, true_val, label) in zip(axes, params):
    ax.scatter(range(len(results_df)), results_df[col],
               alpha=0.5, s=30, color='steelblue')
    ax.axhline(true_val, color='red', linestyle='--', linewidth=2,
               label=f'True value = {true_val}')
    ax.set_xlabel('Simulation ID', fontsize=11)
    ax.set_ylabel(f'Posterior mean of {label}', fontsize=11)
    ax.set_title(f'{label} Recovery', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

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

    # Color intervals by whether they contain true value
    contains_true = ((results_df[lower_col] <= true_val) &
                     (results_df[upper_col] >= true_val))

    # Plot intervals (sample every 2nd to avoid clutter)
    sample_indices = range(0, len(results_df), 2)
    for i in sample_indices:
        color = 'green' if contains_true.iloc[i] else 'red'
        ax.plot([i, i], [results_df[lower_col].iloc[i], results_df[upper_col].iloc[i]],
                color=color, alpha=0.3, linewidth=1)

    ax.axhline(true_val, color='blue', linestyle='--', linewidth=2,
               label=f'True value = {true_val}')
    ax.set_xlabel('Simulation ID (every 2nd)', fontsize=11)
    ax.set_ylabel(f'95% CI for {label}', fontsize=11)
    ax.set_title(f'{label} Coverage: {contains_true.mean():.2%}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_intervals.png", dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'coverage_intervals.png'}")
plt.close()

# Figure 4: Shrinkage plot (posterior mean vs true value with identity line)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
params = [
    ('alpha_mean', TRUE_ALPHA, r'$\alpha$', 0.05),
    ('beta_mean', TRUE_BETA, r'$\beta$', 0.02),
    ('sigma_mean', TRUE_SIGMA, r'$\sigma$', 0.01)
]

for ax, (col, true_val, label, margin) in zip(axes, params):
    # Scatter plot
    ax.scatter(true_val, results_df[col], alpha=0.5, s=30, color='steelblue')

    # Identity line
    plot_min = true_val - margin
    plot_max = true_val + margin
    ax.plot([plot_min, plot_max], [plot_min, plot_max],
            'r--', linewidth=2, label='Perfect recovery')

    # Mean recovery line
    mean_est = results_df[col].mean()
    ax.axhline(mean_est, color='green', linestyle=':', linewidth=2,
               label=f'Mean estimate = {mean_est:.4f}')

    ax.set_xlabel(f'True {label}', fontsize=11)
    ax.set_ylabel(f'Posterior mean of {label}', fontsize=11)
    ax.set_title(f'{label} Bias: {mean_est - true_val:+.4f}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "shrinkage_plot.png", dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'shrinkage_plot.png'}")
plt.close()

# Figure 5: Convergence diagnostics
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# R-hat distribution
axes[0].hist(results_df['rhat_max'], bins=30, edgecolor='black',
             alpha=0.7, color='steelblue')
axes[0].axvline(1.01, color='red', linestyle='--', linewidth=2,
                label='Threshold = 1.01')
axes[0].set_xlabel('Max R-hat', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('R-hat Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# ESS distribution
axes[1].hist(results_df['ess_min'], bins=30, edgecolor='black',
             alpha=0.7, color='steelblue')
axes[1].axvline(400, color='red', linestyle='--', linewidth=2,
                label='Threshold = 400')
axes[1].set_xlabel('Min ESS', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('ESS Distribution', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_diagnostics.png", dpi=150, bbox_inches='tight')
print(f"Saved: {PLOTS_DIR / 'convergence_diagnostics.png'}")
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
    'Ranks approximately uniform': all([hist.max() / expected_per_bin < 2.0
                                        for hist, _ in
                                        [np.histogram(results_df['alpha_rank'], bins=20),
                                         np.histogram(results_df['beta_rank'], bins=20),
                                         np.histogram(results_df['sigma_rank'], bins=20)]])
}

print("\nCriteria Assessment:")
for criterion, passed in pass_criteria.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {criterion}")

overall_pass = all(pass_criteria.values())
print(f"\n{'='*80}")
print(f"OVERALL DECISION: {'PASS' if overall_pass else 'FAIL'}")
print(f"{'='*80}")

if overall_pass:
    print("\nThe model successfully recovers known parameters from synthetic data.")
    print("Calibration is appropriate. PROCEED to fitting real data.")
else:
    print("\nWARNING: The model shows calibration issues.")
    print("Review the specific failures above before proceeding.")

# Save summary
summary_text = f"""# Simulation-Based Calibration Summary

## Overall Decision: {'PASS' if overall_pass else 'FAIL'}

## Coverage Rates (Target: 0.95)
- alpha: {alpha_coverage:.3f}
- beta: {beta_coverage:.3f}
- sigma: {sigma_coverage:.3f}

## Bias Statistics
- alpha: {alpha_bias:+.4f} ({100*alpha_bias/TRUE_ALPHA:+.2f}%)
- beta: {beta_bias:+.4f} ({100*beta_bias/TRUE_BETA:+.2f}%)
- sigma: {sigma_bias:+.4f} ({100*sigma_bias/TRUE_SIGMA:+.2f}%)

## Convergence
- Convergence rate: {convergence_rate:.3f}
- Mean max R-hat: {results_df['rhat_max'].mean():.4f}
- Mean min ESS: {results_df['ess_min'].mean():.0f}

## Criteria Assessment
"""

for criterion, passed in pass_criteria.items():
    summary_text += f"- [{'PASS' if passed else 'FAIL'}] {criterion}\n"

with open(CODE_DIR / "sbc_summary.txt", "w") as f:
    f.write(summary_text)

print(f"\nSummary saved to: {CODE_DIR / 'sbc_summary.txt'}")
print(f"Results saved to: {CODE_DIR / 'sbc_results.csv'}")
print("\nSBC analysis complete!")
