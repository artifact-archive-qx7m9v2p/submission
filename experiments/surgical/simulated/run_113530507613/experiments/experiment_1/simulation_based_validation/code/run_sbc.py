"""
Simulation-Based Calibration for Hierarchical Logit-Normal Model

This script performs SBC to validate that the model can recover known parameters
from simulated data, which is a critical prerequisite before fitting real data.
"""

import numpy as np
import pandas as pd
import cmdstanpy
from pathlib import Path
import json
import time
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
CODE_DIR = BASE_DIR / "code"
PLOTS_DIR = BASE_DIR / "plots"
DATA_PATH = Path("/workspace/data/data.csv")

# SBC Configuration
N_SIMS = 100  # Number of SBC simulations
N_CHAINS = 4
N_WARMUP = 1000
N_SAMPLES = 1000
ADAPT_DELTA = 0.95

print("=" * 80)
print("SIMULATION-BASED CALIBRATION")
print("=" * 80)
print(f"Number of simulations: {N_SIMS}")
print(f"Chains: {N_CHAINS}, Warmup: {N_WARMUP}, Samples: {N_SAMPLES}")
print(f"Adapt delta: {ADAPT_DELTA}")
print()

# Load data structure (we use the same n_trials as observed data)
data = pd.read_csv(DATA_PATH)
n_trials = data['n_trials'].values
J = len(n_trials)

print(f"Data structure: J = {J} groups")
print(f"n_trials: {n_trials}")
print()

# Compile Stan model
print("Compiling Stan model...")
model_path = CODE_DIR / "hierarchical_logit_normal.stan"
model = cmdstanpy.CmdStanModel(stan_file=str(model_path))
print("Model compiled successfully!")
print()

# Prior parameters (from metadata.md)
MU_MEAN = -2.6
MU_SD = 1.0
TAU_SD = 0.5  # half-normal

def simulate_from_prior():
    """Simulate one dataset from the prior predictive distribution"""
    # Draw hyperparameters from priors
    mu = np.random.normal(MU_MEAN, MU_SD)
    tau = np.abs(np.random.normal(0, TAU_SD))  # half-normal

    # Draw group-level parameters (non-centered)
    theta_raw = np.random.normal(0, 1, size=J)
    theta = mu + tau * theta_raw

    # Generate data
    p = 1 / (1 + np.exp(-theta))  # inv_logit
    r = np.random.binomial(n_trials, p)

    return {
        'mu': mu,
        'tau': tau,
        'theta_raw': theta_raw,
        'theta': theta,
        'p': p,
        'r': r
    }

def compute_rank_statistic(true_value, posterior_samples):
    """
    Compute rank of true value within posterior samples.
    Should be Uniform(0, S) if calibrated, where S = number of samples.
    """
    rank = np.sum(posterior_samples < true_value)
    return rank

# Storage for SBC results
sbc_results = {
    'mu': {'ranks': [], 'coverage_90': [], 'bias': [], 'posterior_sd': []},
    'tau': {'ranks': [], 'coverage_90': [], 'bias': [], 'posterior_sd': []},
    'theta': {j: {'ranks': [], 'coverage_90': [], 'bias': [], 'posterior_sd': []}
              for j in range(J)},
}

# Storage for computational diagnostics
diagnostics = {
    'divergences': [],
    'max_rhat': [],
    'min_ess_bulk': [],
    'min_ess_tail': [],
    'runtime': [],
    'failed_fits': []
}

print("Starting SBC simulations...")
print("=" * 80)

start_time_total = time.time()

for sim in range(N_SIMS):
    sim_start_time = time.time()

    print(f"\nSimulation {sim + 1}/{N_SIMS}")
    print("-" * 40)

    # Step 1: Simulate data from prior
    sim_data = simulate_from_prior()

    # Step 2: Fit model to simulated data
    stan_data = {
        'J': J,
        'n': n_trials.tolist(),
        'r': sim_data['r'].tolist()
    }

    try:
        fit = model.sample(
            data=stan_data,
            chains=N_CHAINS,
            iter_warmup=N_WARMUP,
            iter_sampling=N_SAMPLES,
            adapt_delta=ADAPT_DELTA,
            show_progress=False,
            show_console=False
        )

        # Extract diagnostics
        diag = fit.diagnose()
        divergences = fit.divergences

        # Get summary for Rhat and ESS
        summary = fit.summary()
        max_rhat = summary['R_hat'].max()
        min_ess_bulk = summary['N_Eff'].min() if 'N_Eff' in summary.columns else summary['ess_bulk'].min()
        min_ess_tail = summary['ess_tail'].min() if 'ess_tail' in summary.columns else min_ess_bulk

        # Extract posterior samples
        draws = fit.draws_pd()

        # Step 3: Compute rank statistics
        # For mu
        mu_samples = draws['mu'].values
        mu_rank = compute_rank_statistic(sim_data['mu'], mu_samples)
        mu_ci_90 = np.percentile(mu_samples, [5, 95])
        mu_coverage = (sim_data['mu'] >= mu_ci_90[0]) and (sim_data['mu'] <= mu_ci_90[1])
        mu_bias = np.mean(mu_samples) - sim_data['mu']

        sbc_results['mu']['ranks'].append(mu_rank)
        sbc_results['mu']['coverage_90'].append(mu_coverage)
        sbc_results['mu']['bias'].append(mu_bias)
        sbc_results['mu']['posterior_sd'].append(np.std(mu_samples))

        # For tau
        tau_samples = draws['tau'].values
        tau_rank = compute_rank_statistic(sim_data['tau'], tau_samples)
        tau_ci_90 = np.percentile(tau_samples, [5, 95])
        tau_coverage = (sim_data['tau'] >= tau_ci_90[0]) and (sim_data['tau'] <= tau_ci_90[1])
        tau_bias = np.mean(tau_samples) - sim_data['tau']

        sbc_results['tau']['ranks'].append(tau_rank)
        sbc_results['tau']['coverage_90'].append(tau_coverage)
        sbc_results['tau']['bias'].append(tau_bias)
        sbc_results['tau']['posterior_sd'].append(np.std(tau_samples))

        # For theta (all groups)
        for j in range(J):
            theta_samples = draws[f'theta[{j+1}]'].values
            theta_rank = compute_rank_statistic(sim_data['theta'][j], theta_samples)
            theta_ci_90 = np.percentile(theta_samples, [5, 95])
            theta_coverage = (sim_data['theta'][j] >= theta_ci_90[0]) and (sim_data['theta'][j] <= theta_ci_90[1])
            theta_bias = np.mean(theta_samples) - sim_data['theta'][j]

            sbc_results['theta'][j]['ranks'].append(theta_rank)
            sbc_results['theta'][j]['coverage_90'].append(theta_coverage)
            sbc_results['theta'][j]['bias'].append(theta_bias)
            sbc_results['theta'][j]['posterior_sd'].append(np.std(theta_samples))

        # Store diagnostics
        diagnostics['divergences'].append(divergences)
        diagnostics['max_rhat'].append(max_rhat)
        diagnostics['min_ess_bulk'].append(min_ess_bulk)
        diagnostics['min_ess_tail'].append(min_ess_tail)

        sim_runtime = time.time() - sim_start_time
        diagnostics['runtime'].append(sim_runtime)

        print(f"  Divergences: {divergences}, max Rhat: {max_rhat:.4f}, min ESS: {min_ess_bulk:.0f}")
        print(f"  Runtime: {sim_runtime:.1f}s")

    except Exception as e:
        print(f"  ERROR: Fit failed - {str(e)}")
        diagnostics['failed_fits'].append(sim)
        continue

total_runtime = time.time() - start_time_total

print()
print("=" * 80)
print("SBC COMPLETE")
print("=" * 80)
print(f"Total runtime: {total_runtime/60:.1f} minutes")
print(f"Failed fits: {len(diagnostics['failed_fits'])}/{N_SIMS}")
print()

# Save results
results_dict = {
    'sbc_results': {
        'mu': {k: [float(v) for v in vals] for k, vals in sbc_results['mu'].items()},
        'tau': {k: [float(v) for v in vals] for k, vals in sbc_results['tau'].items()},
        'theta': {
            j: {k: [float(v) for v in vals] for k, vals in sbc_results['theta'][j].items()}
            for j in range(J)
        }
    },
    'diagnostics': {
        'divergences': [int(d) for d in diagnostics['divergences']],
        'max_rhat': [float(r) for r in diagnostics['max_rhat']],
        'min_ess_bulk': [float(e) for e in diagnostics['min_ess_bulk']],
        'min_ess_tail': [float(e) for e in diagnostics['min_ess_tail']],
        'runtime': [float(r) for r in diagnostics['runtime']],
        'failed_fits': diagnostics['failed_fits']
    },
    'config': {
        'n_sims': N_SIMS,
        'n_chains': N_CHAINS,
        'n_warmup': N_WARMUP,
        'n_samples': N_SAMPLES,
        'adapt_delta': ADAPT_DELTA,
        'total_runtime': float(total_runtime),
        'J': J
    }
}

with open(CODE_DIR / "sbc_results.json", 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"Results saved to: {CODE_DIR / 'sbc_results.json'}")
print()

# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

print("=" * 80)
print("SBC DIAGNOSTICS")
print("=" * 80)
print()

# 1. Rank Statistics Analysis
print("1. RANK STATISTICS")
print("-" * 40)

n_posterior_samples = N_CHAINS * N_SAMPLES

def test_uniformity(ranks, n_bins=20):
    """Test if ranks are uniformly distributed using chi-square test"""
    observed, _ = np.histogram(ranks, bins=n_bins, range=(0, n_posterior_samples))
    expected = len(ranks) / n_bins
    chi2, p_value = stats.chisquare(observed, expected)
    return chi2, p_value

# Test mu
mu_chi2, mu_p = test_uniformity(sbc_results['mu']['ranks'])
print(f"mu:")
print(f"  Chi-square test: χ² = {mu_chi2:.2f}, p = {mu_p:.4f}")
print(f"  {'PASS' if mu_p > 0.05 else 'FAIL'} (p > 0.05 indicates uniform ranks)")

# Test tau
tau_chi2, tau_p = test_uniformity(sbc_results['tau']['ranks'])
print(f"\ntau:")
print(f"  Chi-square test: χ² = {tau_chi2:.2f}, p = {tau_p:.4f}")
print(f"  {'PASS' if tau_p > 0.05 else 'FAIL'} (p > 0.05 indicates uniform ranks)")

# Test theta (aggregate across groups)
all_theta_ranks = []
for j in range(J):
    all_theta_ranks.extend(sbc_results['theta'][j]['ranks'])
theta_chi2, theta_p = test_uniformity(all_theta_ranks)
print(f"\ntheta (all groups):")
print(f"  Chi-square test: χ² = {theta_chi2:.2f}, p = {theta_p:.4f}")
print(f"  {'PASS' if theta_p > 0.05 else 'FAIL'} (p > 0.05 indicates uniform ranks)")

print()

# 2. Coverage Analysis
print("2. COVERAGE (90% Credible Intervals)")
print("-" * 40)

mu_coverage_rate = np.mean(sbc_results['mu']['coverage_90'])
print(f"mu: {mu_coverage_rate*100:.1f}% (target: 90%)")

tau_coverage_rate = np.mean(sbc_results['tau']['coverage_90'])
print(f"tau: {tau_coverage_rate*100:.1f}% (target: 90%)")

theta_coverage_rates = [np.mean(sbc_results['theta'][j]['coverage_90']) for j in range(J)]
theta_coverage_mean = np.mean(theta_coverage_rates)
print(f"theta (mean across groups): {theta_coverage_mean*100:.1f}% (target: 90%)")

coverage_pass = (
    abs(mu_coverage_rate - 0.9) < 0.1 and
    abs(tau_coverage_rate - 0.9) < 0.1 and
    abs(theta_coverage_mean - 0.9) < 0.1
)
print(f"\n{'PASS' if coverage_pass else 'CONCERN'} (within 10% of nominal 90%)")

print()

# 3. Bias Analysis
print("3. BIAS (Posterior Mean - True Value)")
print("-" * 40)

mu_mean_bias = np.mean(sbc_results['mu']['bias'])
mu_sd_bias = np.std(sbc_results['mu']['bias'])
print(f"mu: mean bias = {mu_mean_bias:.4f} ± {mu_sd_bias:.4f}")

tau_mean_bias = np.mean(sbc_results['tau']['bias'])
tau_sd_bias = np.std(sbc_results['tau']['bias'])
print(f"tau: mean bias = {tau_mean_bias:.4f} ± {tau_sd_bias:.4f}")

theta_biases = []
for j in range(J):
    theta_biases.extend(sbc_results['theta'][j]['bias'])
theta_mean_bias = np.mean(theta_biases)
theta_sd_bias = np.std(theta_biases)
print(f"theta (all groups): mean bias = {theta_mean_bias:.4f} ± {theta_sd_bias:.4f}")

# Check if bias is significantly different from zero (t-test)
mu_t, mu_t_p = stats.ttest_1samp(sbc_results['mu']['bias'], 0)
tau_t, tau_t_p = stats.ttest_1samp(sbc_results['tau']['bias'], 0)
theta_t, theta_t_p = stats.ttest_1samp(theta_biases, 0)

print(f"\nSignificance test (H0: bias = 0):")
print(f"  mu: t = {mu_t:.2f}, p = {mu_t_p:.4f} {'(significant bias)' if mu_t_p < 0.05 else '(no significant bias)'}")
print(f"  tau: t = {tau_t:.2f}, p = {tau_t_p:.4f} {'(significant bias)' if tau_t_p < 0.05 else '(no significant bias)'}")
print(f"  theta: t = {theta_t:.2f}, p = {theta_t_p:.4f} {'(significant bias)' if theta_t_p < 0.05 else '(no significant bias)'}")

bias_pass = mu_t_p > 0.05 and tau_t_p > 0.05 and theta_t_p > 0.05
print(f"\n{'PASS' if bias_pass else 'FAIL'} (no significant bias)")

print()

# 4. Computational Health
print("4. COMPUTATIONAL HEALTH")
print("-" * 40)

total_divergences = sum(diagnostics['divergences'])
pct_divergences = total_divergences / (len(diagnostics['divergences']) * N_CHAINS * N_SAMPLES) * 100
print(f"Total divergences: {total_divergences} ({pct_divergences:.2f}% of post-warmup draws)")

max_rhat_overall = max(diagnostics['max_rhat'])
print(f"Max Rhat across all fits: {max_rhat_overall:.4f}")

min_ess_overall = min(diagnostics['min_ess_bulk'])
print(f"Min ESS across all fits: {min_ess_overall:.0f}")

mean_runtime = np.mean(diagnostics['runtime'])
print(f"Mean runtime per fit: {mean_runtime:.1f}s")

computational_pass = (
    pct_divergences < 1.0 and
    max_rhat_overall < 1.01 and
    min_ess_overall > 400
)
print(f"\n{'PASS' if computational_pass else 'CONCERN'} (divergences < 1%, Rhat < 1.01, ESS > 400)")

print()

# ============================================================================
# FINAL DECISION
# ============================================================================

print("=" * 80)
print("FINAL SBC DECISION")
print("=" * 80)
print()

rank_pass = mu_p > 0.05 and tau_p > 0.05 and theta_p > 0.05
overall_pass = rank_pass and coverage_pass and bias_pass and computational_pass

if overall_pass:
    decision = "PASS"
    print("✓ PASS: Model successfully recovers known parameters")
    print()
    print("All criteria met:")
    print("  - Rank statistics are uniform (proper calibration)")
    print("  - 90% CIs contain true values ~90% of the time")
    print("  - No systematic bias in parameter recovery")
    print("  - Computation is stable (minimal divergences, good Rhat/ESS)")
    print()
    print("RECOMMENDATION: Proceed to fitting real data (Stage 3)")
elif not rank_pass or not bias_pass:
    decision = "FAIL"
    print("✗ FAIL: Model cannot reliably recover known parameters")
    print()
    print("Critical issues detected:")
    if not rank_pass:
        print("  - Rank statistics are non-uniform (calibration failure)")
    if not bias_pass:
        print("  - Systematic bias in parameter recovery")
    print()
    print("RECOMMENDATION: Do NOT proceed to real data. Investigate:")
    if not rank_pass:
        print("  - Possible prior-likelihood conflict")
        print("  - Model misspecification")
    if not bias_pass:
        print("  - Bias suggests identifiability issues or wrong parameterization")
else:
    decision = "CONCERN"
    print("⚠ CONCERN: Model passes statistical tests but has issues")
    print()
    print("Issues detected:")
    if not coverage_pass:
        print("  - Coverage deviates from nominal 90%")
    if not computational_pass:
        print("  - Computational problems (divergences, Rhat, ESS)")
    print()
    print("RECOMMENDATION: Proceed cautiously. May need to:")
    print("  - Increase adapt_delta")
    print("  - Increase warmup iterations")
    print("  - Check for funnel geometry (though we use non-centered)")

print()
print("=" * 80)

# Save decision
decision_dict = {
    'decision': decision,
    'rank_pass': rank_pass,
    'coverage_pass': coverage_pass,
    'bias_pass': bias_pass,
    'computational_pass': computational_pass,
    'mu_p_value': float(mu_p),
    'tau_p_value': float(tau_p),
    'theta_p_value': float(theta_p),
    'mu_coverage': float(mu_coverage_rate),
    'tau_coverage': float(tau_coverage_rate),
    'theta_coverage': float(theta_coverage_mean),
    'mu_bias': float(mu_mean_bias),
    'tau_bias': float(tau_mean_bias),
    'theta_bias': float(theta_mean_bias),
    'divergences_pct': float(pct_divergences),
    'max_rhat': float(max_rhat_overall),
    'min_ess': float(min_ess_overall)
}

with open(CODE_DIR / "sbc_decision.json", 'w') as f:
    json.dump(decision_dict, f, indent=2)

print(f"Decision saved to: {CODE_DIR / 'sbc_decision.json'}")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Generating visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Plot 1: Rank statistics histograms with uniformity reference
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# mu ranks
ax = axes[0, 0]
ax.hist(sbc_results['mu']['ranks'], bins=20, alpha=0.7, edgecolor='black', density=True)
ax.axhline(1/n_posterior_samples, color='red', linestyle='--', linewidth=2, label='Uniform')
ax.set_xlabel('Rank')
ax.set_ylabel('Density')
ax.set_title(f'mu: Rank Statistics (p = {mu_p:.3f})')
ax.legend()

# tau ranks
ax = axes[0, 1]
ax.hist(sbc_results['tau']['ranks'], bins=20, alpha=0.7, edgecolor='black', density=True)
ax.axhline(1/n_posterior_samples, color='red', linestyle='--', linewidth=2, label='Uniform')
ax.set_xlabel('Rank')
ax.set_ylabel('Density')
ax.set_title(f'tau: Rank Statistics (p = {tau_p:.3f})')
ax.legend()

# theta ranks (all groups aggregated)
ax = axes[1, 0]
ax.hist(all_theta_ranks, bins=20, alpha=0.7, edgecolor='black', density=True)
ax.axhline(1/n_posterior_samples, color='red', linestyle='--', linewidth=2, label='Uniform')
ax.set_xlabel('Rank')
ax.set_ylabel('Density')
ax.set_title(f'theta (all groups): Rank Statistics (p = {theta_p:.3f})')
ax.legend()

# ECDF comparison
ax = axes[1, 1]
uniform_expected = np.linspace(0, 1, 100)
mu_ecdf = np.sort(np.array(sbc_results['mu']['ranks']) / n_posterior_samples)
tau_ecdf = np.sort(np.array(sbc_results['tau']['ranks']) / n_posterior_samples)
theta_ecdf = np.sort(np.array(all_theta_ranks) / n_posterior_samples)

ax.plot(uniform_expected, uniform_expected, 'k--', linewidth=2, label='Uniform (ideal)')
ax.plot(np.linspace(0, 1, len(mu_ecdf)), mu_ecdf, label='mu', alpha=0.7)
ax.plot(np.linspace(0, 1, len(tau_ecdf)), tau_ecdf, label='tau', alpha=0.7)
ax.plot(np.linspace(0, 1, len(theta_ecdf)), theta_ecdf, label='theta', alpha=0.7)
ax.set_xlabel('Expected Quantile')
ax.set_ylabel('Observed Quantile')
ax.set_title('ECDF: Rank Uniformity Check')
ax.legend()
ax.axis('equal')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_statistics.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: {PLOTS_DIR / 'rank_statistics.png'}")

# Plot 2: Coverage and bias
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Coverage by parameter
ax = axes[0, 0]
coverage_data = {
    'mu': mu_coverage_rate,
    'tau': tau_coverage_rate,
    'theta\n(mean)': theta_coverage_mean
}
bars = ax.bar(coverage_data.keys(), [v * 100 for v in coverage_data.values()],
               alpha=0.7, edgecolor='black')
ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target (90%)')
ax.set_ylabel('Coverage (%)')
ax.set_title('90% Credible Interval Coverage')
ax.set_ylim([0, 100])
ax.legend()
# Color bars based on whether they're within acceptable range
for i, (bar, val) in enumerate(zip(bars, coverage_data.values())):
    if abs(val - 0.9) < 0.1:
        bar.set_color('green')
    else:
        bar.set_color('orange')

# Bias distribution for mu
ax = axes[0, 1]
ax.hist(sbc_results['mu']['bias'], bins=20, alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No bias')
ax.set_xlabel('Posterior Mean - True Value')
ax.set_ylabel('Frequency')
ax.set_title(f'mu: Bias Distribution (mean = {mu_mean_bias:.4f})')
ax.legend()

# Bias distribution for tau
ax = axes[1, 0]
ax.hist(sbc_results['tau']['bias'], bins=20, alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No bias')
ax.set_xlabel('Posterior Mean - True Value')
ax.set_ylabel('Frequency')
ax.set_title(f'tau: Bias Distribution (mean = {tau_mean_bias:.4f})')
ax.legend()

# Bias distribution for theta
ax = axes[1, 1]
ax.hist(theta_biases, bins=30, alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No bias')
ax.set_xlabel('Posterior Mean - True Value')
ax.set_ylabel('Frequency')
ax.set_title(f'theta: Bias Distribution (mean = {theta_mean_bias:.4f})')
ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "coverage_and_bias.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: {PLOTS_DIR / 'coverage_and_bias.png'}")

# Plot 3: Computational diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Divergences over simulations
ax = axes[0, 0]
ax.plot(diagnostics['divergences'], 'o-', alpha=0.5)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Simulation')
ax.set_ylabel('Number of Divergences')
ax.set_title(f'Divergences per Fit (Total: {total_divergences})')

# Rhat over simulations
ax = axes[0, 1]
ax.plot(diagnostics['max_rhat'], 'o-', alpha=0.5)
ax.axhline(1.01, color='red', linestyle='--', linewidth=2, label='Threshold (1.01)')
ax.set_xlabel('Simulation')
ax.set_ylabel('Max Rhat')
ax.set_title(f'Max Rhat per Fit (Overall Max: {max_rhat_overall:.4f})')
ax.legend()

# ESS over simulations
ax = axes[1, 0]
ax.plot(diagnostics['min_ess_bulk'], 'o-', alpha=0.5)
ax.axhline(400, color='red', linestyle='--', linewidth=2, label='Threshold (400)')
ax.set_xlabel('Simulation')
ax.set_ylabel('Min ESS (bulk)')
ax.set_title(f'Min ESS per Fit (Overall Min: {min_ess_overall:.0f})')
ax.legend()

# Runtime distribution
ax = axes[1, 1]
ax.hist(diagnostics['runtime'], bins=20, alpha=0.7, edgecolor='black')
ax.set_xlabel('Runtime (seconds)')
ax.set_ylabel('Frequency')
ax.set_title(f'Runtime Distribution (Mean: {mean_runtime:.1f}s)')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "computational_diagnostics.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: {PLOTS_DIR / 'computational_diagnostics.png'}")

# Plot 4: Shrinkage check (posterior SD vs prior SD)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# mu shrinkage
ax = axes[0]
mu_prior_sd = MU_SD
mu_posterior_sds = sbc_results['mu']['posterior_sd']
ax.hist(mu_posterior_sds, bins=20, alpha=0.7, edgecolor='black', label='Posterior SD')
ax.axvline(mu_prior_sd, color='red', linestyle='--', linewidth=2, label='Prior SD')
ax.set_xlabel('Standard Deviation')
ax.set_ylabel('Frequency')
ax.set_title(f'mu: Posterior vs Prior SD\n(Mean posterior SD: {np.mean(mu_posterior_sds):.3f})')
ax.legend()

# tau shrinkage
ax = axes[1]
tau_prior_sd = TAU_SD
tau_posterior_sds = sbc_results['tau']['posterior_sd']
ax.hist(tau_posterior_sds, bins=20, alpha=0.7, edgecolor='black', label='Posterior SD')
ax.axvline(tau_prior_sd, color='red', linestyle='--', linewidth=2, label='Prior SD')
ax.set_xlabel('Standard Deviation')
ax.set_ylabel('Frequency')
ax.set_title(f'tau: Posterior vs Prior SD\n(Mean posterior SD: {np.mean(tau_posterior_sds):.3f})')
ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "shrinkage_check.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: {PLOTS_DIR / 'shrinkage_check.png'}")

# Plot 5: Group-level theta coverage
fig, ax = plt.subplots(figsize=(10, 6))

theta_coverage_by_group = [np.mean(sbc_results['theta'][j]['coverage_90']) * 100 for j in range(J)]
bars = ax.bar(range(1, J+1), theta_coverage_by_group, alpha=0.7, edgecolor='black')
ax.axhline(90, color='red', linestyle='--', linewidth=2, label='Target (90%)')
ax.set_xlabel('Group')
ax.set_ylabel('Coverage (%)')
ax.set_title('90% Credible Interval Coverage by Group')
ax.set_ylim([0, 100])
ax.legend()

# Color bars based on coverage
for bar, cov in zip(bars, theta_coverage_by_group):
    if abs(cov - 90) < 10:
        bar.set_color('green')
    else:
        bar.set_color('orange')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "theta_coverage_by_group.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: {PLOTS_DIR / 'theta_coverage_by_group.png'}")

print()
print("=" * 80)
print("SBC COMPLETE - All results saved")
print("=" * 80)
