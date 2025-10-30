"""
Simulation-Based Calibration (SBC) Validation for Beta-Binomial Hierarchical Model

This script validates parameter recovery through:
1. Full SBC with 50 simulations
2. Focused scenario tests (low, moderate, high overdispersion, zero-events)
3. Computational diagnostics
4. Coverage and calibration assessment
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Output directories
BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Actual data structure (n values from data.csv)
N_GROUPS = 12
N_VALUES = np.array([47, 148, 119, 810, 211, 196, 148, 215, 207, 97, 256, 360])

# Prior specifications (validated)
PRIOR_MU_ALPHA = 2.0
PRIOR_MU_BETA = 18.0
PRIOR_KAPPA_ALPHA = 1.5
PRIOR_KAPPA_BETA = 0.5

print("="*80)
print("SIMULATION-BASED CALIBRATION VALIDATION")
print("Beta-Binomial Hierarchical Model - Experiment 1")
print("="*80)
print(f"\nData Structure: {N_GROUPS} groups")
print(f"Sample sizes: {N_VALUES}")
print(f"Total observations: {N_VALUES.sum()}")
print(f"\nPriors:")
print(f"  μ ~ Beta({PRIOR_MU_ALPHA}, {PRIOR_MU_BETA}) [E[μ] = {PRIOR_MU_ALPHA/(PRIOR_MU_ALPHA+PRIOR_MU_BETA):.3f}]")
print(f"  κ ~ Gamma({PRIOR_KAPPA_ALPHA}, {PRIOR_KAPPA_BETA}) [E[κ] = {PRIOR_KAPPA_ALPHA/PRIOR_KAPPA_BETA:.3f}]")

# ============================================================================
# PART 1: SIMULATION-BASED CALIBRATION (SBC)
# ============================================================================

def simulate_data_from_prior(n_values, prior_mu_alpha, prior_mu_beta,
                             prior_kappa_alpha, prior_kappa_beta):
    """Generate synthetic data by drawing from priors."""
    # Draw hyperparameters from priors
    mu_true = np.random.beta(prior_mu_alpha, prior_mu_beta)
    kappa_true = np.random.gamma(prior_kappa_alpha, 1.0/prior_kappa_beta)

    # Convert to alpha, beta
    alpha_true = mu_true * kappa_true
    beta_true = (1 - mu_true) * kappa_true

    # Draw group-level probabilities
    p_true = np.random.beta(alpha_true, beta_true, size=len(n_values))

    # Generate binomial data
    r_sim = np.random.binomial(n_values, p_true)

    return {
        'mu_true': mu_true,
        'kappa_true': kappa_true,
        'phi_true': 1.0 / (1.0 + kappa_true),  # overdispersion
        'alpha_true': alpha_true,
        'beta_true': beta_true,
        'p_true': p_true,
        'r_sim': r_sim
    }


def fit_model_to_simulated_data(r_sim, n_values, tune=500, draws=500, chains=2):
    """Fit the Beta-Binomial hierarchical model to simulated data."""
    with pm.Model() as model:
        # Hyperpriors
        mu = pm.Beta('mu', alpha=PRIOR_MU_ALPHA, beta=PRIOR_MU_BETA)
        kappa = pm.Gamma('kappa', alpha=PRIOR_KAPPA_ALPHA, beta=PRIOR_KAPPA_BETA)

        # Transformed parameters
        alpha = pm.Deterministic('alpha', mu * kappa)
        beta = pm.Deterministic('beta', (1 - mu) * kappa)
        phi = pm.Deterministic('phi', 1.0 / (1.0 + kappa))

        # Group-level probabilities
        p = pm.Beta('p', alpha=alpha, beta=beta, shape=len(n_values))

        # Likelihood
        r_obs = pm.Binomial('r_obs', n=n_values, p=p, observed=r_sim)

        # Sample
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=False,
            random_seed=np.random.randint(0, 10000)
        )

    return trace


def compute_rank_statistic(true_value, posterior_samples):
    """Compute SBC rank statistic: how many posterior samples < true value."""
    return np.sum(posterior_samples < true_value)


def run_sbc_simulation(sim_idx, n_values):
    """Run a single SBC iteration."""
    print(f"\n  Simulation {sim_idx + 1}...", end=" ", flush=True)

    try:
        # Generate synthetic data
        sim_data = simulate_data_from_prior(
            n_values, PRIOR_MU_ALPHA, PRIOR_MU_BETA,
            PRIOR_KAPPA_ALPHA, PRIOR_KAPPA_BETA
        )

        # Fit model
        trace = fit_model_to_simulated_data(sim_data['r_sim'], n_values)

        # Extract posterior samples
        mu_post = trace.posterior['mu'].values.flatten()
        kappa_post = trace.posterior['kappa'].values.flatten()
        phi_post = trace.posterior['phi'].values.flatten()

        # Compute ranks
        rank_mu = compute_rank_statistic(sim_data['mu_true'], mu_post)
        rank_kappa = compute_rank_statistic(sim_data['kappa_true'], kappa_post)
        rank_phi = compute_rank_statistic(sim_data['phi_true'], phi_post)

        # Compute posterior summaries
        mu_mean = mu_post.mean()
        mu_std = mu_post.std()
        mu_q05 = np.percentile(mu_post, 5)
        mu_q95 = np.percentile(mu_post, 95)

        kappa_mean = kappa_post.mean()
        kappa_std = kappa_post.std()
        kappa_q05 = np.percentile(kappa_post, 5)
        kappa_q95 = np.percentile(kappa_post, 95)

        phi_mean = phi_post.mean()
        phi_std = phi_post.std()
        phi_q05 = np.percentile(phi_post, 5)
        phi_q95 = np.percentile(phi_post, 95)

        # Check containment in 90% intervals
        mu_contained = (sim_data['mu_true'] >= mu_q05) and (sim_data['mu_true'] <= mu_q95)
        kappa_contained = (sim_data['kappa_true'] >= kappa_q05) and (sim_data['kappa_true'] <= kappa_q95)
        phi_contained = (sim_data['phi_true'] >= phi_q05) and (sim_data['phi_true'] <= phi_q95)

        # Compute bias
        mu_bias = mu_mean - sim_data['mu_true']
        kappa_bias = kappa_mean - sim_data['kappa_true']
        phi_bias = phi_mean - sim_data['phi_true']

        # Convergence diagnostics
        summary = az.summary(trace, var_names=['mu', 'kappa', 'phi'])
        rhat_max = summary['r_hat'].max()
        ess_min = summary['ess_bulk'].min()

        # Divergences
        n_divergences = trace.sample_stats['diverging'].sum().item()
        total_samples = len(mu_post)
        divergence_rate = n_divergences / total_samples

        print(f"✓ (Rhat={rhat_max:.3f}, ESS={ess_min:.0f}, div={n_divergences})")

        return {
            'sim_idx': sim_idx,
            'mu_true': sim_data['mu_true'],
            'kappa_true': sim_data['kappa_true'],
            'phi_true': sim_data['phi_true'],
            'mu_mean': mu_mean,
            'kappa_mean': kappa_mean,
            'phi_mean': phi_mean,
            'mu_std': mu_std,
            'kappa_std': kappa_std,
            'phi_std': phi_std,
            'mu_q05': mu_q05,
            'mu_q95': mu_q95,
            'kappa_q05': kappa_q05,
            'kappa_q95': kappa_q95,
            'phi_q05': phi_q05,
            'phi_q95': phi_q95,
            'mu_contained': mu_contained,
            'kappa_contained': kappa_contained,
            'phi_contained': phi_contained,
            'mu_bias': mu_bias,
            'kappa_bias': kappa_bias,
            'phi_bias': phi_bias,
            'rank_mu': rank_mu,
            'rank_kappa': rank_kappa,
            'rank_phi': rank_phi,
            'rhat_max': rhat_max,
            'ess_min': ess_min,
            'n_divergences': n_divergences,
            'divergence_rate': divergence_rate,
            'success': True
        }

    except Exception as e:
        print(f"✗ FAILED: {str(e)}")
        return {
            'sim_idx': sim_idx,
            'success': False,
            'error': str(e)
        }


print("\n" + "="*80)
print("PART 1: FULL SBC WITH 50 SIMULATIONS")
print("="*80)

N_SBC_SIMS = 50
sbc_results = []

for i in range(N_SBC_SIMS):
    result = run_sbc_simulation(i, N_VALUES)
    sbc_results.append(result)

# Convert to DataFrame
sbc_df = pd.DataFrame([r for r in sbc_results if r['success']])
n_success = len(sbc_df)
n_failed = N_SBC_SIMS - n_success

print(f"\n\nSBC Complete: {n_success}/{N_SBC_SIMS} simulations successful ({n_failed} failed)")

# Save results
sbc_df.to_csv(BASE_DIR / "sbc_results.csv", index=False)
print(f"Results saved to: {BASE_DIR / 'sbc_results.csv'}")

# ============================================================================
# PART 2: SBC ANALYSIS AND DIAGNOSTICS
# ============================================================================

print("\n" + "="*80)
print("SBC DIAGNOSTICS")
print("="*80)

# Coverage statistics
mu_coverage = sbc_df['mu_contained'].mean()
kappa_coverage = sbc_df['kappa_contained'].mean()
phi_coverage = sbc_df['phi_contained'].mean()

print(f"\n1. COVERAGE (90% Credible Intervals):")
print(f"   μ:     {mu_coverage:.1%} ({sbc_df['mu_contained'].sum()}/{n_success})")
print(f"   κ:     {kappa_coverage:.1%} ({sbc_df['kappa_contained'].sum()}/{n_success})")
print(f"   φ:     {phi_coverage:.1%} ({sbc_df['phi_contained'].sum()}/{n_success})")

# Bias analysis
print(f"\n2. BIAS (Mean - True):")
print(f"   μ:     {sbc_df['mu_bias'].mean():+.4f} ± {sbc_df['mu_bias'].std():.4f}")
print(f"   κ:     {sbc_df['kappa_bias'].mean():+.4f} ± {sbc_df['kappa_bias'].std():.4f}")
print(f"   φ:     {sbc_df['phi_bias'].mean():+.4f} ± {sbc_df['phi_bias'].std():.4f}")

# Relative bias
mu_rel_bias = (sbc_df['mu_bias'] / sbc_df['mu_true']).mean()
kappa_rel_bias = (sbc_df['kappa_bias'] / sbc_df['kappa_true']).mean()
phi_rel_bias = (sbc_df['phi_bias'] / sbc_df['phi_true']).mean()

print(f"\n3. RELATIVE BIAS:")
print(f"   μ:     {mu_rel_bias:+.2%}")
print(f"   κ:     {kappa_rel_bias:+.2%}")
print(f"   φ:     {phi_rel_bias:+.2%}")

# Rank uniformity test (Kolmogorov-Smirnov)
n_samples = 1000  # draws * chains
ks_mu = stats.kstest(sbc_df['rank_mu'], stats.uniform(0, n_samples).cdf)
ks_kappa = stats.kstest(sbc_df['rank_kappa'], stats.uniform(0, n_samples).cdf)
ks_phi = stats.kstest(sbc_df['rank_phi'], stats.uniform(0, n_samples).cdf)

print(f"\n4. RANK UNIFORMITY (Kolmogorov-Smirnov Test):")
print(f"   μ:     KS={ks_mu.statistic:.3f}, p={ks_mu.pvalue:.3f}")
print(f"   κ:     KS={ks_kappa.statistic:.3f}, p={ks_kappa.pvalue:.3f}")
print(f"   φ:     KS={ks_phi.statistic:.3f}, p={ks_phi.pvalue:.3f}")

# Computational diagnostics
print(f"\n5. COMPUTATIONAL DIAGNOSTICS:")
print(f"   Max Rhat:          {sbc_df['rhat_max'].max():.4f} (median: {sbc_df['rhat_max'].median():.4f})")
print(f"   Min ESS:           {sbc_df['ess_min'].min():.0f} (median: {sbc_df['ess_min'].median():.0f})")
print(f"   Divergence rate:   {sbc_df['divergence_rate'].mean():.2%} (max: {sbc_df['divergence_rate'].max():.2%})")

n_converged = (sbc_df['rhat_max'] < 1.01).sum()
print(f"   Converged (Rhat<1.01): {n_converged}/{n_success} ({n_converged/n_success:.1%})")

# ============================================================================
# PART 3: VISUALIZATION - SBC DIAGNOSTICS
# ============================================================================

print("\n" + "="*80)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*80)

# Plot 1: SBC Rank Histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
n_bins = 20

for ax, param, ranks, ks_test in zip(
    axes,
    ['μ', 'κ', 'φ'],
    [sbc_df['rank_mu'], sbc_df['rank_kappa'], sbc_df['rank_phi']],
    [ks_mu, ks_kappa, ks_phi]
):
    ax.hist(ranks, bins=n_bins, edgecolor='black', alpha=0.7)
    ax.axhline(n_success / n_bins, color='red', linestyle='--', label='Expected (uniform)')
    ax.set_xlabel('Rank', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{param}: Rank Histogram\nKS p={ks_test.pvalue:.3f}', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
rank_plot = PLOTS_DIR / "sbc_rank_histograms.png"
plt.savefig(rank_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ Rank histograms: {rank_plot}")
plt.close()

# Plot 2: Parameter Recovery Scatter Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

params_data = [
    ('μ', sbc_df['mu_true'], sbc_df['mu_mean'], sbc_df['mu_q05'], sbc_df['mu_q95']),
    ('κ', sbc_df['kappa_true'], sbc_df['kappa_mean'], sbc_df['kappa_q05'], sbc_df['kappa_q95']),
    ('φ', sbc_df['phi_true'], sbc_df['phi_mean'], sbc_df['phi_q05'], sbc_df['phi_q95'])
]

for ax, (param, true_vals, mean_vals, q05_vals, q95_vals) in zip(axes, params_data):
    # Error bars for 90% CI
    ax.errorbar(true_vals, mean_vals,
                yerr=[mean_vals - q05_vals, q95_vals - mean_vals],
                fmt='o', alpha=0.6, capsize=3, markersize=4)

    # Perfect recovery line
    lim_min = min(true_vals.min(), mean_vals.min())
    lim_max = max(true_vals.max(), mean_vals.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', label='Perfect recovery')

    ax.set_xlabel(f'True {param}', fontsize=11)
    ax.set_ylabel(f'Posterior Mean {param}', fontsize=11)
    ax.set_title(f'{param} Recovery', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
recovery_plot = PLOTS_DIR / "parameter_recovery.png"
plt.savefig(recovery_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ Parameter recovery: {recovery_plot}")
plt.close()

# Plot 3: Coverage Assessment
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (param, true_vals, q05_vals, q95_vals, contained) in zip(
    axes,
    [
        ('μ', sbc_df['mu_true'], sbc_df['mu_q05'], sbc_df['mu_q95'], sbc_df['mu_contained']),
        ('κ', sbc_df['kappa_true'], sbc_df['kappa_q05'], sbc_df['kappa_q95'], sbc_df['kappa_contained']),
        ('φ', sbc_df['phi_true'], sbc_df['phi_q05'], sbc_df['phi_q95'], sbc_df['phi_contained'])
    ]
):
    # Sort by true value for clearer visualization
    sorted_idx = np.argsort(true_vals)

    colors = ['green' if c else 'red' for c in contained.iloc[sorted_idx]]

    for i, idx in enumerate(sorted_idx):
        ax.plot([i, i], [q05_vals.iloc[idx], q95_vals.iloc[idx]],
                color=colors[i], alpha=0.6, linewidth=2)
        ax.scatter(i, true_vals.iloc[idx], color='black', s=20, zorder=3)

    coverage = contained.mean()
    ax.set_xlabel('Simulation (sorted by true value)', fontsize=11)
    ax.set_ylabel(f'{param} Value', fontsize=11)
    ax.set_title(f'{param}: 90% CI Coverage = {coverage:.1%}', fontsize=12)
    ax.grid(alpha=0.3)

plt.tight_layout()
coverage_plot = PLOTS_DIR / "coverage_assessment.png"
plt.savefig(coverage_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ Coverage assessment: {coverage_plot}")
plt.close()

# Plot 4: Bias Distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, param, bias in zip(
    axes,
    ['μ', 'κ', 'φ'],
    [sbc_df['mu_bias'], sbc_df['kappa_bias'], sbc_df['phi_bias']]
):
    ax.hist(bias, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='No bias')
    ax.axvline(bias.mean(), color='blue', linestyle='-', label=f'Mean: {bias.mean():.4f}')
    ax.set_xlabel(f'Bias (Posterior Mean - True {param})', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{param} Bias Distribution', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
bias_plot = PLOTS_DIR / "bias_distribution.png"
plt.savefig(bias_plot, dpi=300, bbox_inches='tight')
print(f"  ✓ Bias distribution: {bias_plot}")
plt.close()

# ============================================================================
# PART 4: FOCUSED SCENARIO TESTS
# ============================================================================

print("\n" + "="*80)
print("PART 2: FOCUSED SCENARIO TESTS")
print("="*80)

def run_focused_scenario(scenario_name, mu_true, kappa_true, n_reps=5):
    """Run focused recovery test for specific parameter values."""
    print(f"\n{scenario_name}")
    print(f"  True values: μ={mu_true:.3f}, κ={kappa_true:.3f}, φ={1/(1+kappa_true):.3f}")

    results = []

    for rep in range(n_reps):
        print(f"    Rep {rep+1}/{n_reps}...", end=" ", flush=True)

        try:
            # Generate data with fixed hyperparameters
            alpha_true = mu_true * kappa_true
            beta_true = (1 - mu_true) * kappa_true
            p_true = np.random.beta(alpha_true, beta_true, size=len(N_VALUES))
            r_sim = np.random.binomial(N_VALUES, p_true)

            # Fit model
            trace = fit_model_to_simulated_data(r_sim, N_VALUES, tune=500, draws=500, chains=2)

            # Extract posteriors
            mu_post = trace.posterior['mu'].values.flatten()
            kappa_post = trace.posterior['kappa'].values.flatten()
            phi_post = trace.posterior['phi'].values.flatten()

            mu_mean = mu_post.mean()
            kappa_mean = kappa_post.mean()
            phi_mean = phi_post.mean()

            # Relative errors
            mu_rel_error = abs(mu_mean - mu_true) / mu_true
            kappa_rel_error = abs(kappa_mean - kappa_true) / kappa_true
            phi_true = 1 / (1 + kappa_true)
            phi_rel_error = abs(phi_mean - phi_true) / phi_true

            # Diagnostics
            summary = az.summary(trace, var_names=['mu', 'kappa'])
            rhat_max = summary['r_hat'].max()
            ess_min = summary['ess_bulk'].min()
            n_div = trace.sample_stats['diverging'].sum().item()

            print(f"✓ μ_err={mu_rel_error:.1%}, κ_err={kappa_rel_error:.1%}, Rhat={rhat_max:.3f}")

            results.append({
                'scenario': scenario_name,
                'rep': rep,
                'mu_true': mu_true,
                'kappa_true': kappa_true,
                'phi_true': phi_true,
                'mu_mean': mu_mean,
                'kappa_mean': kappa_mean,
                'phi_mean': phi_mean,
                'mu_rel_error': mu_rel_error,
                'kappa_rel_error': kappa_rel_error,
                'phi_rel_error': phi_rel_error,
                'rhat_max': rhat_max,
                'ess_min': ess_min,
                'n_divergences': n_div,
                'success': True
            })

        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
            results.append({
                'scenario': scenario_name,
                'rep': rep,
                'success': False,
                'error': str(e)
            })

    return results

# Define scenarios
scenarios = [
    ("Scenario A: Low Overdispersion", 0.08, 5.0),
    ("Scenario B: Moderate Overdispersion", 0.08, 1.0),
    ("Scenario C: High Overdispersion (matches data)", 0.08, 0.3),
]

all_scenario_results = []

for scenario_name, mu, kappa in scenarios:
    results = run_focused_scenario(scenario_name, mu, kappa, n_reps=5)
    all_scenario_results.extend(results)

# Save scenario results
scenario_df = pd.DataFrame([r for r in all_scenario_results if r['success']])
scenario_df.to_csv(BASE_DIR / "scenario_results.csv", index=False)
print(f"\nScenario results saved to: {BASE_DIR / 'scenario_results.csv'}")

# Scenario summary
print("\n" + "-"*80)
print("SCENARIO SUMMARY")
print("-"*80)

for scenario_name in scenario_df['scenario'].unique():
    df_scen = scenario_df[scenario_df['scenario'] == scenario_name]
    print(f"\n{scenario_name}:")
    print(f"  Success rate: {len(df_scen)}/5")
    print(f"  μ rel. error: {df_scen['mu_rel_error'].mean():.1%} ± {df_scen['mu_rel_error'].std():.1%}")
    print(f"  κ rel. error: {df_scen['kappa_rel_error'].mean():.1%} ± {df_scen['kappa_rel_error'].std():.1%}")
    print(f"  φ rel. error: {df_scen['phi_rel_error'].mean():.1%} ± {df_scen['phi_rel_error'].std():.1%}")
    print(f"  Max Rhat: {df_scen['rhat_max'].max():.3f}")
    print(f"  Converged: {(df_scen['rhat_max'] < 1.01).sum()}/5")

# Plot scenario recovery
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, param in zip(axes, ['mu', 'kappa', 'phi']):
    for i, scenario_name in enumerate(scenario_df['scenario'].unique()):
        df_scen = scenario_df[scenario_df['scenario'] == scenario_name]
        true_val = df_scen[f'{param}_true'].iloc[0]
        recovered = df_scen[f'{param}_mean'].values

        x_pos = i + np.random.normal(0, 0.05, size=len(recovered))
        ax.scatter(x_pos, recovered, alpha=0.6, s=50, label=scenario_name if i==0 else "")
        ax.scatter(i, true_val, color='red', marker='*', s=200, zorder=5,
                   label='True value' if i==0 else "")

    ax.set_xticks(range(len(scenario_df['scenario'].unique())))
    ax.set_xticklabels(['Low\nκ=5', 'Moderate\nκ=1', 'High\nκ=0.3'], fontsize=10)
    ax.set_ylabel(f'{param.capitalize()} Value', fontsize=11)
    ax.set_title(f'{param.capitalize()} Recovery by Scenario', fontsize=12)
    if param == 'mu':
        ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
scenario_plot = PLOTS_DIR / "scenario_recovery.png"
plt.savefig(scenario_plot, dpi=300, bbox_inches='tight')
print(f"\n  ✓ Scenario recovery: {scenario_plot}")
plt.close()

# ============================================================================
# PART 5: FINAL DECISION
# ============================================================================

print("\n" + "="*80)
print("FINAL ASSESSMENT: PASS/FAIL CRITERIA")
print("="*80)

criteria = {
    'coverage_mu': mu_coverage >= 0.85,
    'coverage_kappa': kappa_coverage >= 0.85,
    'coverage_phi': phi_coverage >= 0.85,
    'rank_uniform_mu': ks_mu.pvalue > 0.05,
    'rank_uniform_kappa': ks_kappa.pvalue > 0.05,
    'rank_uniform_phi': ks_phi.pvalue > 0.05,
    'convergence': (sbc_df['rhat_max'] < 1.01).sum() / n_success >= 0.80,
    'divergences': sbc_df['divergence_rate'].mean() < 0.02,
    'scenario_recovery': (scenario_df['mu_rel_error'] < 0.20).all() and
                         (scenario_df['kappa_rel_error'] < 0.20).all(),
    'no_failures': n_failed <= N_SBC_SIMS * 0.20
}

print("\nCriteria Assessment:")
print(f"  1. μ coverage ≥ 85%:           {'PASS' if criteria['coverage_mu'] else 'FAIL'} ({mu_coverage:.1%})")
print(f"  2. κ coverage ≥ 85%:           {'PASS' if criteria['coverage_kappa'] else 'FAIL'} ({kappa_coverage:.1%})")
print(f"  3. φ coverage ≥ 85%:           {'PASS' if criteria['coverage_phi'] else 'FAIL'} ({phi_coverage:.1%})")
print(f"  4. μ ranks uniform (p>0.05):   {'PASS' if criteria['rank_uniform_mu'] else 'FAIL'} (p={ks_mu.pvalue:.3f})")
print(f"  5. κ ranks uniform (p>0.05):   {'PASS' if criteria['rank_uniform_kappa'] else 'FAIL'} (p={ks_kappa.pvalue:.3f})")
print(f"  6. φ ranks uniform (p>0.05):   {'PASS' if criteria['rank_uniform_phi'] else 'FAIL'} (p={ks_phi.pvalue:.3f})")
print(f"  7. Convergence ≥ 80%:          {'PASS' if criteria['convergence'] else 'FAIL'} ({(sbc_df['rhat_max'] < 1.01).sum() / n_success:.1%})")
print(f"  8. Divergences < 2%:           {'PASS' if criteria['divergences'] else 'FAIL'} ({sbc_df['divergence_rate'].mean():.2%})")
print(f"  9. Scenario recovery < 20%:    {'PASS' if criteria['scenario_recovery'] else 'FAIL'}")
print(f" 10. Failures ≤ 20%:            {'PASS' if criteria['no_failures'] else 'FAIL'} ({n_failed}/{N_SBC_SIMS})")

overall_pass = all(criteria.values())

print("\n" + "="*80)
if overall_pass:
    print("OVERALL DECISION: ✓ PASS")
    print("Model demonstrates excellent parameter recovery and calibration.")
    print("Proceed to fit real data.")
else:
    print("OVERALL DECISION: ✗ FAIL")
    print("Model shows recovery or calibration issues.")
    print("DO NOT proceed to real data - diagnose and fix issues first.")
print("="*80)

# Save decision
with open(BASE_DIR / "decision.txt", 'w') as f:
    f.write("PASS" if overall_pass else "FAIL")

print(f"\nAll outputs saved to: {BASE_DIR}")
print("\nValidation complete!")
