"""
Simulation-Based Calibration for Experiment 2: Random Effects Logistic Regression
FIXED VERSION with robust posterior extraction

This script validates parameter recovery through:
1. 50 simulations from prior (SBC coverage and calibration)
2. 15 focused scenarios (low/moderate/high heterogeneity)
3. Comparison to Experiment 1 failure points

Model:
  r_i | θ_i, n_i ~ Binomial(n_i, logit^{-1}(θ_i))
  θ_i = μ + τ * z_i
  z_i ~ Normal(0, 1)
  μ ~ Normal(-2.51, 1)
  τ ~ HalfNormal(1)
"""

import sys
sys.path.insert(0, '/tmp/agent-home/.local/lib/python3.13/site-packages')

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import expit, logit
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Read actual data to get group sample sizes
data = pd.read_csv('/workspace/data/data.csv')
n_groups = len(data)
n_obs = data['n'].values  # [47, 148, 119, ..., 360]

print("="*80)
print("SIMULATION-BASED CALIBRATION: Experiment 2 Random Effects Logistic Regression")
print("="*80)
print(f"\nData structure: {n_groups} groups")
print(f"Sample sizes: {n_obs}")
print(f"Total observations: {n_obs.sum()}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_synthetic_data(n_obs, mu_true, tau_true, seed=None):
    """Generate synthetic data from the hierarchical logistic model."""
    if seed is not None:
        np.random.seed(seed)

    n_groups = len(n_obs)
    z_true = np.random.normal(0, 1, size=n_groups)
    theta_true = mu_true + tau_true * z_true
    p_true = expit(theta_true)
    r = np.random.binomial(n_obs, p_true)

    return r, mu_true, tau_true, z_true, theta_true


def fit_model(r, n_obs, n_samples=500, n_chains=2, seed=None):
    """Fit the random effects logistic regression model."""
    n_groups = len(r)

    with pm.Model() as model:
        # Priors
        mu = pm.Normal('mu', mu=logit(0.075), sigma=1.0)
        tau = pm.HalfNormal('tau', sigma=1.0)

        # Non-centered parameterization
        z = pm.Normal('z', mu=0, sigma=1, shape=n_groups)
        theta = pm.Deterministic('theta', mu + tau * z)

        # Likelihood
        p = pm.Deterministic('p', pm.math.invlogit(theta))
        y = pm.Binomial('y', n=n_obs, p=p, observed=r)

        # Sample
        idata = pm.sample(
            n_samples,
            chains=n_chains,
            tune=500,
            random_seed=seed,
            return_inferencedata=True,
            target_accept=0.95,
            idata_kwargs={'log_likelihood': True}
        )

    # Check convergence
    rhat = az.rhat(idata)
    max_rhat_mu = float(rhat['mu'].values)
    max_rhat_tau = float(rhat['tau'].values)
    max_rhat = max(max_rhat_mu, max_rhat_tau)

    success = max_rhat < 1.01

    return idata, success


def extract_posterior_samples(idata, var_name):
    """Robustly extract posterior samples as a 1D array."""
    # Get the posterior group
    posterior = idata.posterior[var_name]
    # Stack chain and draw dimensions
    samples = posterior.values.flatten()
    return samples


def compute_rank_statistic(true_value, posterior_samples):
    """Compute SBC rank statistic."""
    rank = np.sum(posterior_samples < true_value)
    return rank


def check_coverage(true_value, posterior_samples, level=0.9):
    """Check if true value falls within credible interval."""
    lower = np.percentile(posterior_samples, (1 - level) / 2 * 100)
    upper = np.percentile(posterior_samples, (1 + level) / 2 * 100)
    return lower <= true_value <= upper


def relative_error(estimate, true_value):
    """Compute relative error as percentage."""
    return np.abs(estimate - true_value) / np.abs(true_value) * 100


# ============================================================================
# PART 1: FULL SBC (50 SIMULATIONS FROM PRIOR)
# ============================================================================

print("\n" + "="*80)
print("PART 1: SIMULATION-BASED CALIBRATION (50 simulations)")
print("="*80)

n_sbc_sims = 50
sbc_results = []

for i in range(n_sbc_sims):
    print(f"\nSBC Simulation {i+1}/{n_sbc_sims}")

    # Draw true parameters from prior
    mu_true = np.random.normal(logit(0.075), 1.0)
    tau_true = np.abs(np.random.normal(0, 1.0))  # HalfNormal

    # Generate synthetic data
    r_sim, mu_true, tau_true, z_true, theta_true = generate_synthetic_data(
        n_obs, mu_true, tau_true, seed=100+i
    )

    print(f"  True μ = {mu_true:.3f}, τ = {tau_true:.3f}")
    print(f"  Generated {len(r_sim)} groups, total events = {r_sim.sum()}/{n_obs.sum()}")

    # Fit model
    try:
        idata, converged = fit_model(r_sim, n_obs, n_samples=500, n_chains=2, seed=200+i)

        # Extract posterior samples - FIXED
        mu_post = extract_posterior_samples(idata, 'mu')
        tau_post = extract_posterior_samples(idata, 'tau')

        # Compute diagnostics
        rhat_mu = float(az.rhat(idata)['mu'].values)
        rhat_tau = float(az.rhat(idata)['tau'].values)

        # Compute rank statistics
        rank_mu = compute_rank_statistic(mu_true, mu_post)
        rank_tau = compute_rank_statistic(tau_true, tau_post)

        # Check coverage
        coverage_mu_90 = check_coverage(mu_true, mu_post, level=0.9)
        coverage_tau_90 = check_coverage(tau_true, tau_post, level=0.9)

        # Compute bias
        mu_mean = np.mean(mu_post)
        tau_mean = np.mean(tau_post)
        bias_mu = mu_mean - mu_true
        bias_tau = tau_mean - tau_true
        rel_error_mu = relative_error(mu_mean, mu_true)
        rel_error_tau = relative_error(tau_mean, tau_true)

        # Check for divergences
        n_divergences = 0
        if hasattr(idata.sample_stats, 'diverging'):
            n_divergences = int(idata.sample_stats.diverging.sum())

        print(f"  Converged: {converged} (R̂_μ={rhat_mu:.4f}, R̂_τ={rhat_tau:.4f})")
        print(f"  Coverage: μ={coverage_mu_90}, τ={coverage_tau_90}")
        print(f"  Rel. Error: μ={rel_error_mu:.1f}%, τ={rel_error_tau:.1f}%")
        print(f"  Divergences: {n_divergences}")

        sbc_results.append({
            'sim': i+1,
            'mu_true': mu_true,
            'tau_true': tau_true,
            'mu_mean': mu_mean,
            'tau_mean': tau_mean,
            'bias_mu': bias_mu,
            'bias_tau': bias_tau,
            'rel_error_mu': rel_error_mu,
            'rel_error_tau': rel_error_tau,
            'rank_mu': rank_mu,
            'rank_tau': rank_tau,
            'coverage_mu_90': coverage_mu_90,
            'coverage_tau_90': coverage_tau_90,
            'rhat_mu': rhat_mu,
            'rhat_tau': rhat_tau,
            'converged': converged,
            'n_divergences': n_divergences,
            'success': converged and n_divergences < 10
        })

    except Exception as e:
        print(f"  FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sbc_results.append({
            'sim': i+1,
            'mu_true': mu_true,
            'tau_true': tau_true,
            'mu_mean': np.nan,
            'tau_mean': np.nan,
            'bias_mu': np.nan,
            'bias_tau': np.nan,
            'rel_error_mu': np.nan,
            'rel_error_tau': np.nan,
            'rank_mu': np.nan,
            'rank_tau': np.nan,
            'coverage_mu_90': False,
            'coverage_tau_90': False,
            'rhat_mu': np.nan,
            'rhat_tau': np.nan,
            'converged': False,
            'n_divergences': np.nan,
            'success': False
        })

sbc_df = pd.DataFrame(sbc_results)
sbc_df.to_csv('/workspace/experiments/experiment_2/simulation_based_validation/sbc_results.csv', index=False)

print("\n" + "="*80)
print("SBC SUMMARY STATISTICS")
print("="*80)

successful_sims = sbc_df['success'].sum()
convergence_rate = sbc_df['converged'].sum() / n_sbc_sims * 100

print(f"\nConvergence Rate: {convergence_rate:.1f}% ({sbc_df['converged'].sum()}/{n_sbc_sims})")
print(f"Successful Simulations: {successful_sims}/{n_sbc_sims}")

if successful_sims > 0:
    valid_df = sbc_df[sbc_df['success']]

    print(f"\n90% COVERAGE (among successful simulations):")
    coverage_mu = valid_df['coverage_mu_90'].mean() * 100
    coverage_tau = valid_df['coverage_tau_90'].mean() * 100
    print(f"  μ: {coverage_mu:.1f}% ({valid_df['coverage_mu_90'].sum()}/{len(valid_df)})")
    print(f"  τ: {coverage_tau:.1f}% ({valid_df['coverage_tau_90'].sum()}/{len(valid_df)})")

    print(f"\nBIAS (among successful simulations):")
    print(f"  μ: mean = {valid_df['bias_mu'].mean():.4f}, median = {valid_df['bias_mu'].median():.4f}")
    print(f"  τ: mean = {valid_df['bias_tau'].mean():.4f}, median = {valid_df['bias_tau'].median():.4f}")

    print(f"\nRELATIVE ERROR (among successful simulations):")
    print(f"  μ: mean = {valid_df['rel_error_mu'].mean():.1f}%, median = {valid_df['rel_error_mu'].median():.1f}%")
    print(f"  τ: mean = {valid_df['rel_error_tau'].mean():.1f}%, median = {valid_df['rel_error_tau'].median():.1f}%")

    print(f"\nDIVERGENCES:")
    print(f"  Mean: {valid_df['n_divergences'].mean():.1f}")
    print(f"  Median: {valid_df['n_divergences'].median():.1f}")
    print(f"  Max: {valid_df['n_divergences'].max():.0f}")


# ============================================================================
# PART 2: FOCUSED SCENARIOS (15 SIMULATIONS)
# ============================================================================

print("\n" + "="*80)
print("PART 2: FOCUSED SCENARIOS (Low/Moderate/High Heterogeneity)")
print("="*80)

scenarios = [
    {'name': 'Low Heterogeneity', 'tau': 0.3, 'n_reps': 5},
    {'name': 'Moderate Heterogeneity', 'tau': 0.7, 'n_reps': 5},
    {'name': 'High Heterogeneity', 'tau': 1.2, 'n_reps': 5}  # Matches our data ICC=0.66
]

scenario_results = []

for scenario in scenarios:
    print(f"\n{scenario['name']} (τ = {scenario['tau']})")
    print("-" * 40)

    for rep in range(scenario['n_reps']):
        print(f"\n  Rep {rep+1}/{scenario['n_reps']}")

        # Use prior mean for μ, fixed τ
        mu_true = logit(0.075)  # ≈ -2.51
        tau_true = scenario['tau']

        # Generate synthetic data
        r_sim, mu_true, tau_true, z_true, theta_true = generate_synthetic_data(
            n_obs, mu_true, tau_true, seed=300 + len(scenario_results)
        )

        print(f"    True μ = {mu_true:.3f}, τ = {tau_true:.3f}")

        # Fit model
        try:
            idata, converged = fit_model(r_sim, n_obs, n_samples=500, n_chains=2,
                                        seed=400 + len(scenario_results))

            # Extract posterior samples - FIXED
            mu_post = extract_posterior_samples(idata, 'mu')
            tau_post = extract_posterior_samples(idata, 'tau')

            # Compute diagnostics
            rhat_mu = float(az.rhat(idata)['mu'].values)
            rhat_tau = float(az.rhat(idata)['tau'].values)

            mu_mean = np.mean(mu_post)
            tau_mean = np.mean(tau_post)
            rel_error_mu = relative_error(mu_mean, mu_true)
            rel_error_tau = relative_error(tau_mean, tau_true)

            coverage_mu_90 = check_coverage(mu_true, mu_post, level=0.9)
            coverage_tau_90 = check_coverage(tau_true, tau_post, level=0.9)

            n_divergences = 0
            if hasattr(idata.sample_stats, 'diverging'):
                n_divergences = int(idata.sample_stats.diverging.sum())

            print(f"    Converged: {converged} (R̂_μ={rhat_mu:.4f}, R̂_τ={rhat_tau:.4f})")
            print(f"    Coverage: μ={coverage_mu_90}, τ={coverage_tau_90}")
            print(f"    Rel. Error: μ={rel_error_mu:.1f}%, τ={rel_error_tau:.1f}%")

            scenario_results.append({
                'scenario': scenario['name'],
                'tau_target': scenario['tau'],
                'rep': rep+1,
                'mu_true': mu_true,
                'tau_true': tau_true,
                'mu_mean': mu_mean,
                'tau_mean': tau_mean,
                'rel_error_mu': rel_error_mu,
                'rel_error_tau': rel_error_tau,
                'coverage_mu_90': coverage_mu_90,
                'coverage_tau_90': coverage_tau_90,
                'rhat_mu': rhat_mu,
                'rhat_tau': rhat_tau,
                'converged': converged,
                'n_divergences': n_divergences,
                'success': converged and rel_error_mu < 30 and rel_error_tau < 30
            })

        except Exception as e:
            print(f"    FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            scenario_results.append({
                'scenario': scenario['name'],
                'tau_target': scenario['tau'],
                'rep': rep+1,
                'mu_true': mu_true,
                'tau_true': tau_true,
                'mu_mean': np.nan,
                'tau_mean': np.nan,
                'rel_error_mu': np.nan,
                'rel_error_tau': np.nan,
                'coverage_mu_90': False,
                'coverage_tau_90': False,
                'rhat_mu': np.nan,
                'rhat_tau': np.nan,
                'converged': False,
                'n_divergences': np.nan,
                'success': False
            })

scenario_df = pd.DataFrame(scenario_results)
scenario_df.to_csv('/workspace/experiments/experiment_2/simulation_based_validation/scenario_results.csv', index=False)

print("\n" + "="*80)
print("SCENARIO SUMMARY")
print("="*80)

for scenario in scenarios:
    scenario_subset = scenario_df[scenario_df['scenario'] == scenario['name']]
    convergence = scenario_subset['converged'].sum() / len(scenario_subset) * 100
    success_rate = scenario_subset['success'].sum() / len(scenario_subset) * 100

    print(f"\n{scenario['name']} (τ = {scenario['tau']}):")
    print(f"  Convergence: {convergence:.0f}% ({scenario_subset['converged'].sum()}/{len(scenario_subset)})")
    print(f"  Success Rate: {success_rate:.0f}% ({scenario_subset['success'].sum()}/{len(scenario_subset)})")

    valid = scenario_subset[scenario_subset['converged']]
    if len(valid) > 0:
        print(f"  Mean Rel. Error: μ={valid['rel_error_mu'].mean():.1f}%, τ={valid['rel_error_tau'].mean():.1f}%")
        print(f"  Coverage: μ={valid['coverage_mu_90'].mean()*100:.0f}%, τ={valid['coverage_tau_90'].mean()*100:.0f}%")


# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*80)

# Filter to successful simulations for plotting
valid_sbc = sbc_df[sbc_df['success']]

if len(valid_sbc) > 10:  # Need at least 10 successful simulations
    # ========================================================================
    # PLOT 1: SBC Rank Histograms (Key Calibration Check)
    # ========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Expected number of ranks per bin for uniform distribution
    n_bins = 20
    n_samples_per_chain = 500
    n_chains = 2
    n_total_samples = n_samples_per_chain * n_chains
    expected_per_bin = len(valid_sbc) / n_bins

    # μ ranks
    axes[0].hist(valid_sbc['rank_mu'], bins=n_bins, edgecolor='black', alpha=0.7)
    axes[0].axhline(expected_per_bin, color='red', linestyle='--',
                    label=f'Expected (uniform): {expected_per_bin:.1f}')
    axes[0].set_xlabel('Rank Statistic')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('SBC Rank Histogram: μ (Intercept)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # τ ranks
    axes[1].hist(valid_sbc['rank_tau'], bins=n_bins, edgecolor='black', alpha=0.7)
    axes[1].axhline(expected_per_bin, color='red', linestyle='--',
                    label=f'Expected (uniform): {expected_per_bin:.1f}')
    axes[1].set_xlabel('Rank Statistic')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('SBC Rank Histogram: τ (Between-Group SD)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/sbc_rank_histograms.png',
                dpi=300, bbox_inches='tight')
    print("  Saved: sbc_rank_histograms.png")
    plt.close()

    # Kolmogorov-Smirnov test for uniformity
    ks_mu = stats.kstest(valid_sbc['rank_mu'] / n_total_samples, 'uniform')
    ks_tau = stats.kstest(valid_sbc['rank_tau'] / n_total_samples, 'uniform')
    print(f"\nKS Test for Uniformity:")
    print(f"  μ: D={ks_mu.statistic:.4f}, p={ks_mu.pvalue:.4f}")
    print(f"  τ: D={ks_tau.statistic:.4f}, p={ks_tau.pvalue:.4f}")

    # ========================================================================
    # PLOT 2: Parameter Recovery (True vs Estimated)
    # ========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # μ recovery
    axes[0].scatter(valid_sbc['mu_true'], valid_sbc['mu_mean'], alpha=0.6, s=50)
    mu_min = min(valid_sbc['mu_true'].min(), valid_sbc['mu_mean'].min())
    mu_max = max(valid_sbc['mu_true'].max(), valid_sbc['mu_mean'].max())
    axes[0].plot([mu_min, mu_max], [mu_min, mu_max], 'r--', label='Perfect Recovery')
    axes[0].set_xlabel('True μ')
    axes[0].set_ylabel('Estimated μ (Posterior Mean)')
    axes[0].set_title('Parameter Recovery: μ (Intercept)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # τ recovery
    axes[1].scatter(valid_sbc['tau_true'], valid_sbc['tau_mean'], alpha=0.6, s=50)
    tau_min = min(valid_sbc['tau_true'].min(), valid_sbc['tau_mean'].min())
    tau_max = max(valid_sbc['tau_true'].max(), valid_sbc['tau_mean'].max())
    axes[1].plot([tau_min, tau_max], [tau_min, tau_max], 'r--', label='Perfect Recovery')
    axes[1].set_xlabel('True τ')
    axes[1].set_ylabel('Estimated τ (Posterior Mean)')
    axes[1].set_title('Parameter Recovery: τ (Between-Group SD)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/parameter_recovery.png',
                dpi=300, bbox_inches='tight')
    print("  Saved: parameter_recovery.png")
    plt.close()

    # ========================================================================
    # PLOT 3: Relative Error Distribution
    # ========================================================================

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(valid_sbc['rel_error_mu'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(30, color='red', linestyle='--', label='30% Threshold')
    axes[0].set_xlabel('Relative Error (%)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Relative Error Distribution: μ')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(valid_sbc['rel_error_tau'], bins=20, edgecolor='black', alpha=0.7)
    axes[1].axvline(30, color='red', linestyle='--', label='30% Threshold')
    axes[1].set_xlabel('Relative Error (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Relative Error Distribution: τ')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/relative_error_distribution.png',
                dpi=300, bbox_inches='tight')
    print("  Saved: relative_error_distribution.png")
    plt.close()

    # ========================================================================
    # PLOT 4: Scenario Comparison (Focused Tests)
    # ========================================================================

    scenario_valid = scenario_df[scenario_df['converged']]

    if len(scenario_valid) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Recovery by scenario
        for scenario_name in scenario_df['scenario'].unique():
            subset = scenario_valid[scenario_valid['scenario'] == scenario_name]
            if len(subset) > 0:
                axes[0, 0].scatter(subset['mu_true'], subset['mu_mean'],
                                  label=scenario_name, alpha=0.7, s=100)

        axes[0, 0].plot([scenario_valid['mu_true'].min(), scenario_valid['mu_true'].max()],
                       [scenario_valid['mu_true'].min(), scenario_valid['mu_true'].max()],
                       'k--', label='Perfect Recovery')
        axes[0, 0].set_xlabel('True μ')
        axes[0, 0].set_ylabel('Estimated μ')
        axes[0, 0].set_title('μ Recovery by Scenario')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # τ recovery by scenario
        for scenario_name in scenario_df['scenario'].unique():
            subset = scenario_valid[scenario_valid['scenario'] == scenario_name]
            if len(subset) > 0:
                axes[0, 1].scatter(subset['tau_true'], subset['tau_mean'],
                                  label=scenario_name, alpha=0.7, s=100)

        axes[0, 1].plot([scenario_valid['tau_true'].min(), scenario_valid['tau_true'].max()],
                       [scenario_valid['tau_true'].min(), scenario_valid['tau_true'].max()],
                       'k--', label='Perfect Recovery')
        axes[0, 1].set_xlabel('True τ')
        axes[0, 1].set_ylabel('Estimated τ')
        axes[0, 1].set_title('τ Recovery by Scenario')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Relative error by scenario
        scenario_summary = scenario_valid.groupby('scenario').agg({
            'rel_error_mu': 'mean',
            'rel_error_tau': 'mean'
        }).reset_index()

        x_pos = np.arange(len(scenario_summary))
        width = 0.35

        axes[1, 0].bar(x_pos - width/2, scenario_summary['rel_error_mu'],
                      width, label='μ', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, scenario_summary['rel_error_tau'],
                      width, label='τ', alpha=0.7)
        axes[1, 0].axhline(30, color='red', linestyle='--', label='30% Threshold')
        axes[1, 0].set_xlabel('Scenario')
        axes[1, 0].set_ylabel('Mean Relative Error (%)')
        axes[1, 0].set_title('Recovery Error by Scenario')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(scenario_summary['scenario'], rotation=15, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Coverage by scenario
        coverage_summary = scenario_valid.groupby('scenario').agg({
            'coverage_mu_90': 'mean',
            'coverage_tau_90': 'mean'
        }).reset_index()

        axes[1, 1].bar(x_pos - width/2, coverage_summary['coverage_mu_90'] * 100,
                      width, label='μ', alpha=0.7)
        axes[1, 1].bar(x_pos + width/2, coverage_summary['coverage_tau_90'] * 100,
                      width, label='τ', alpha=0.7)
        axes[1, 1].axhline(90, color='red', linestyle='--', label='90% Target')
        axes[1, 1].set_xlabel('Scenario')
        axes[1, 1].set_ylabel('Coverage (%)')
        axes[1, 1].set_title('90% Interval Coverage by Scenario')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(coverage_summary['scenario'], rotation=15, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/scenario_comparison.png',
                    dpi=300, bbox_inches='tight')
        print("  Saved: scenario_comparison.png")
        plt.close()
else:
    print(f"  WARNING: Only {len(valid_sbc)} successful simulations. Need at least 10 for reliable plots.")


# ============================================================================
# FINAL ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("FINAL ASSESSMENT")
print("="*80)

# Compute key metrics
convergence_rate = sbc_df['converged'].sum() / len(sbc_df) * 100
success_rate = sbc_df['success'].sum() / len(sbc_df) * 100

if len(valid_sbc) > 0:
    coverage_mu = valid_sbc['coverage_mu_90'].mean() * 100
    coverage_tau = valid_sbc['coverage_tau_90'].mean() * 100
    mean_rel_error_mu = valid_sbc['rel_error_mu'].mean()
    mean_rel_error_tau = valid_sbc['rel_error_tau'].mean()
    mean_divergences = valid_sbc['n_divergences'].mean()
    pct_divergence_rate = mean_divergences / (500 * 2) * 100

    # KS test
    if len(valid_sbc) > 10:
        ks_mu = stats.kstest(valid_sbc['rank_mu'] / (500*2), 'uniform')
        ks_tau = stats.kstest(valid_sbc['rank_tau'] / (500*2), 'uniform')
        ks_mu_pass = ks_mu.pvalue > 0.05
        ks_tau_pass = ks_tau.pvalue > 0.05
    else:
        ks_mu_pass = False
        ks_tau_pass = False
        ks_mu = type('obj', (object,), {'pvalue': np.nan})()
        ks_tau = type('obj', (object,), {'pvalue': np.nan})()
else:
    coverage_mu = 0
    coverage_tau = 0
    mean_rel_error_mu = np.nan
    mean_rel_error_tau = np.nan
    mean_divergences = np.nan
    pct_divergence_rate = np.nan
    ks_mu_pass = False
    ks_tau_pass = False
    ks_mu = type('obj', (object,), {'pvalue': np.nan})()
    ks_tau = type('obj', (object,), {'pvalue': np.nan})()

# High heterogeneity scenario (matches our data)
high_het = scenario_df[scenario_df['scenario'] == 'High Heterogeneity']
high_het_valid = high_het[high_het['converged']]
if len(high_het_valid) > 0:
    high_het_convergence = high_het['converged'].sum() / len(high_het) * 100
    high_het_error_mu = high_het_valid['rel_error_mu'].mean()
    high_het_error_tau = high_het_valid['rel_error_tau'].mean()
    high_het_pass = (high_het_convergence >= 80 and
                     high_het_error_mu < 30 and
                     high_het_error_tau < 30)
else:
    high_het_convergence = 0
    high_het_error_mu = np.nan
    high_het_error_tau = np.nan
    high_het_pass = False

# Overall PASS/FAIL criteria
pass_criteria = {
    'convergence': convergence_rate >= 80,
    'coverage_mu': coverage_mu >= 85,
    'coverage_tau': coverage_tau >= 85,
    'calibration_mu': ks_mu_pass,
    'calibration_tau': ks_tau_pass,
    'high_het': high_het_pass,
    'divergences': pct_divergence_rate < 1.0 if not np.isnan(pct_divergence_rate) else False
}

overall_pass = all(pass_criteria.values())

print("\nCRITERIA CHECKLIST:")
print(f"  {'PASS' if pass_criteria['convergence'] else 'FAIL'} Convergence >= 80%: {convergence_rate:.1f}%")
print(f"  {'PASS' if pass_criteria['coverage_mu'] else 'FAIL'} Coverage μ >= 85%: {coverage_mu:.1f}%")
print(f"  {'PASS' if pass_criteria['coverage_tau'] else 'FAIL'} Coverage τ >= 85%: {coverage_tau:.1f}%")
print(f"  {'PASS' if pass_criteria['calibration_mu'] else 'FAIL'} Calibration μ (KS p > 0.05): {ks_mu.pvalue:.4f}")
print(f"  {'PASS' if pass_criteria['calibration_tau'] else 'FAIL'} Calibration τ (KS p > 0.05): {ks_tau.pvalue:.4f}")
print(f"  {'PASS' if pass_criteria['high_het'] else 'FAIL'} High heterogeneity scenario: Conv={high_het_convergence:.0f}%, Error μ={high_het_error_mu:.1f}%, τ={high_het_error_tau:.1f}%")
print(f"  {'PASS' if pass_criteria['divergences'] else 'FAIL'} Divergence rate < 1%: {pct_divergence_rate:.3f}%")

print(f"\n{'='*80}")
print(f"RESULT: {'PASS' if overall_pass else 'FAIL'}")
print(f"{'='*80}")

if overall_pass:
    print("\nThe random effects logistic regression model successfully recovers known parameters.")
    print("Model is ready for fitting to real data.")
else:
    print("\nParameter recovery issues detected. Review diagnostics before proceeding.")

# Save summary
with open('/workspace/experiments/experiment_2/simulation_based_validation/assessment_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SIMULATION-BASED CALIBRATION ASSESSMENT\n")
    f.write("Experiment 2: Random Effects Logistic Regression\n")
    f.write("="*80 + "\n\n")

    f.write(f"Overall Result: {'PASS' if overall_pass else 'FAIL'}\n\n")

    f.write("Key Metrics:\n")
    f.write(f"  Convergence Rate: {convergence_rate:.1f}% (Target: >=80%)\n")
    f.write(f"  Coverage μ: {coverage_mu:.1f}% (Target: >=85%)\n")
    f.write(f"  Coverage τ: {coverage_tau:.1f}% (Target: >=85%)\n")
    f.write(f"  Mean Rel. Error μ: {mean_rel_error_mu:.1f}%\n")
    f.write(f"  Mean Rel. Error τ: {mean_rel_error_tau:.1f}%\n")
    f.write(f"  KS Test μ: p={ks_mu.pvalue:.4f} (Target: >0.05)\n")
    f.write(f"  KS Test τ: p={ks_tau.pvalue:.4f} (Target: >0.05)\n")
    f.write(f"  Divergence Rate: {pct_divergence_rate:.3f}% (Target: <1%)\n\n")

    f.write("High Heterogeneity Scenario (τ=1.2, matches ICC=0.66):\n")
    f.write(f"  Convergence: {high_het_convergence:.0f}%\n")
    f.write(f"  Rel. Error μ: {high_het_error_mu:.1f}%\n")
    f.write(f"  Rel. Error τ: {high_het_error_tau:.1f}%\n")

print("\nSaved: assessment_summary.txt")
print("\nSBC validation complete!")
