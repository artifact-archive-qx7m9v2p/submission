"""
Simulation-Based Calibration for Experiment 2: Random Effects Logistic Regression
QUICK VERSION (20 SBC simulations + 9 scenario tests)

This is a reduced but still meaningful validation with:
- 20 simulations from prior (enough for calibration check)
- 3 scenarios x 3 reps each = 9 (enough for targeted testing)
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
n_obs = data['n'].values

print("="*80)
print("SIMULATION-BASED CALIBRATION: Experiment 2 (Quick Version)")
print("="*80)
print(f"\nData structure: {n_groups} groups, total N = {n_obs.sum()}")


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
        mu = pm.Normal('mu', mu=logit(0.075), sigma=1.0)
        tau = pm.HalfNormal('tau', sigma=1.0)
        z = pm.Normal('z', mu=0, sigma=1, shape=n_groups)
        theta = pm.Deterministic('theta', mu + tau * z)
        p = pm.Deterministic('p', pm.math.invlogit(theta))
        y = pm.Binomial('y', n=n_obs, p=p, observed=r)
        idata = pm.sample(n_samples, chains=n_chains, tune=500, random_seed=seed,
                         return_inferencedata=True, target_accept=0.95,
                         idata_kwargs={'log_likelihood': True})

    rhat = az.rhat(idata)
    max_rhat = max(float(rhat['mu'].values), float(rhat['tau'].values))
    success = max_rhat < 1.01
    return idata, success


def extract_posterior_samples(idata, var_name):
    """Robustly extract posterior samples as a 1D array."""
    return idata.posterior[var_name].values.flatten()


def compute_rank_statistic(true_value, posterior_samples):
    """Compute SBC rank statistic."""
    return np.sum(posterior_samples < true_value)


def check_coverage(true_value, posterior_samples, level=0.9):
    """Check if true value falls within credible interval."""
    lower = np.percentile(posterior_samples, (1 - level) / 2 * 100)
    upper = np.percentile(posterior_samples, (1 + level) / 2 * 100)
    return lower <= true_value <= upper


def relative_error(estimate, true_value):
    """Compute relative error as percentage."""
    return np.abs(estimate - true_value) / np.abs(true_value) * 100


# ============================================================================
# PART 1: SBC (20 SIMULATIONS)
# ============================================================================

print("\n" + "="*80)
print("PART 1: SIMULATION-BASED CALIBRATION (20 simulations)")
print("="*80)

n_sbc_sims = 20
sbc_results = []

for i in range(n_sbc_sims):
    print(f"\nSBC Simulation {i+1}/{n_sbc_sims}")

    mu_true = np.random.normal(logit(0.075), 1.0)
    tau_true = np.abs(np.random.normal(0, 1.0))

    r_sim, mu_true, tau_true, z_true, theta_true = generate_synthetic_data(
        n_obs, mu_true, tau_true, seed=100+i)

    print(f"  True μ={mu_true:.3f}, τ={tau_true:.3f}, events={r_sim.sum()}/{n_obs.sum()}")

    try:
        idata, converged = fit_model(r_sim, n_obs, n_samples=500, n_chains=2, seed=200+i)

        mu_post = extract_posterior_samples(idata, 'mu')
        tau_post = extract_posterior_samples(idata, 'tau')

        rhat_mu = float(az.rhat(idata)['mu'].values)
        rhat_tau = float(az.rhat(idata)['tau'].values)

        rank_mu = compute_rank_statistic(mu_true, mu_post)
        rank_tau = compute_rank_statistic(tau_true, tau_post)

        coverage_mu_90 = check_coverage(mu_true, mu_post, level=0.9)
        coverage_tau_90 = check_coverage(tau_true, tau_post, level=0.9)

        mu_mean = np.mean(mu_post)
        tau_mean = np.mean(tau_post)
        bias_mu = mu_mean - mu_true
        bias_tau = tau_mean - tau_true
        rel_error_mu = relative_error(mu_mean, mu_true)
        rel_error_tau = relative_error(tau_mean, tau_true)

        n_divergences = int(idata.sample_stats.diverging.sum()) if hasattr(idata.sample_stats, 'diverging') else 0

        print(f"  Converged={converged} (R̂_μ={rhat_mu:.4f}, R̂_τ={rhat_tau:.4f})")
        print(f"  Coverage: μ={coverage_mu_90}, τ={coverage_tau_90}")
        print(f"  Rel. Error: μ={rel_error_mu:.1f}%, τ={rel_error_tau:.1f}%")

        sbc_results.append({
            'sim': i+1, 'mu_true': mu_true, 'tau_true': tau_true,
            'mu_mean': mu_mean, 'tau_mean': tau_mean,
            'bias_mu': bias_mu, 'bias_tau': bias_tau,
            'rel_error_mu': rel_error_mu, 'rel_error_tau': rel_error_tau,
            'rank_mu': rank_mu, 'rank_tau': rank_tau,
            'coverage_mu_90': coverage_mu_90, 'coverage_tau_90': coverage_tau_90,
            'rhat_mu': rhat_mu, 'rhat_tau': rhat_tau,
            'converged': converged, 'n_divergences': n_divergences,
            'success': converged and n_divergences < 10
        })

    except Exception as e:
        print(f"  FAILED: {str(e)}")
        sbc_results.append({
            'sim': i+1, 'mu_true': mu_true, 'tau_true': tau_true,
            'mu_mean': np.nan, 'tau_mean': np.nan, 'bias_mu': np.nan, 'bias_tau': np.nan,
            'rel_error_mu': np.nan, 'rel_error_tau': np.nan,
            'rank_mu': np.nan, 'rank_tau': np.nan,
            'coverage_mu_90': False, 'coverage_tau_90': False,
            'rhat_mu': np.nan, 'rhat_tau': np.nan,
            'converged': False, 'n_divergences': np.nan, 'success': False
        })

sbc_df = pd.DataFrame(sbc_results)
sbc_df.to_csv('/workspace/experiments/experiment_2/simulation_based_validation/sbc_results.csv', index=False)

print("\n" + "="*80)
print("SBC SUMMARY")
print("="*80)

successful_sims = sbc_df['success'].sum()
convergence_rate = sbc_df['converged'].sum() / n_sbc_sims * 100

print(f"\nConvergence: {convergence_rate:.1f}% ({sbc_df['converged'].sum()}/{n_sbc_sims})")
print(f"Successful: {successful_sims}/{n_sbc_sims}")

if successful_sims > 0:
    valid_df = sbc_df[sbc_df['success']]
    print(f"\n90% Coverage:")
    print(f"  μ: {valid_df['coverage_mu_90'].mean()*100:.1f}% ({valid_df['coverage_mu_90'].sum()}/{len(valid_df)})")
    print(f"  τ: {valid_df['coverage_tau_90'].mean()*100:.1f}% ({valid_df['coverage_tau_90'].sum()}/{len(valid_df)})")
    print(f"\nRelative Error:")
    print(f"  μ: mean={valid_df['rel_error_mu'].mean():.1f}%, median={valid_df['rel_error_mu'].median():.1f}%")
    print(f"  τ: mean={valid_df['rel_error_tau'].mean():.1f}%, median={valid_df['rel_error_tau'].median():.1f}%")
    print(f"\nDivergences: mean={valid_df['n_divergences'].mean():.1f}, max={valid_df['n_divergences'].max():.0f}")


# ============================================================================
# PART 2: FOCUSED SCENARIOS (9 SIMULATIONS)
# ============================================================================

print("\n" + "="*80)
print("PART 2: FOCUSED SCENARIOS (3 scenarios x 3 reps)")
print("="*80)

scenarios = [
    {'name': 'Low Heterogeneity', 'tau': 0.3, 'n_reps': 3},
    {'name': 'Moderate Heterogeneity', 'tau': 0.7, 'n_reps': 3},
    {'name': 'High Heterogeneity', 'tau': 1.2, 'n_reps': 3}
]

scenario_results = []

for scenario in scenarios:
    print(f"\n{scenario['name']} (τ={scenario['tau']})")
    print("-" * 40)

    for rep in range(scenario['n_reps']):
        print(f"  Rep {rep+1}/{scenario['n_reps']}")

        mu_true = logit(0.075)
        tau_true = scenario['tau']

        r_sim, mu_true, tau_true, z_true, theta_true = generate_synthetic_data(
            n_obs, mu_true, tau_true, seed=300 + len(scenario_results))

        try:
            idata, converged = fit_model(r_sim, n_obs, n_samples=500, n_chains=2,
                                        seed=400 + len(scenario_results))

            mu_post = extract_posterior_samples(idata, 'mu')
            tau_post = extract_posterior_samples(idata, 'tau')

            rhat_mu = float(az.rhat(idata)['mu'].values)
            rhat_tau = float(az.rhat(idata)['tau'].values)

            mu_mean = np.mean(mu_post)
            tau_mean = np.mean(tau_post)
            rel_error_mu = relative_error(mu_mean, mu_true)
            rel_error_tau = relative_error(tau_mean, tau_true)

            coverage_mu_90 = check_coverage(mu_true, mu_post, level=0.9)
            coverage_tau_90 = check_coverage(tau_true, tau_post, level=0.9)

            n_divergences = int(idata.sample_stats.diverging.sum()) if hasattr(idata.sample_stats, 'diverging') else 0

            print(f"    Conv={converged}, Rel.Err: μ={rel_error_mu:.1f}%, τ={rel_error_tau:.1f}%")

            scenario_results.append({
                'scenario': scenario['name'], 'tau_target': scenario['tau'], 'rep': rep+1,
                'mu_true': mu_true, 'tau_true': tau_true,
                'mu_mean': mu_mean, 'tau_mean': tau_mean,
                'rel_error_mu': rel_error_mu, 'rel_error_tau': rel_error_tau,
                'coverage_mu_90': coverage_mu_90, 'coverage_tau_90': coverage_tau_90,
                'rhat_mu': rhat_mu, 'rhat_tau': rhat_tau,
                'converged': converged, 'n_divergences': n_divergences,
                'success': converged and rel_error_mu < 30 and rel_error_tau < 30
            })

        except Exception as e:
            print(f"    FAILED: {str(e)}")
            scenario_results.append({
                'scenario': scenario['name'], 'tau_target': scenario['tau'], 'rep': rep+1,
                'mu_true': mu_true, 'tau_true': tau_true,
                'mu_mean': np.nan, 'tau_mean': np.nan,
                'rel_error_mu': np.nan, 'rel_error_tau': np.nan,
                'coverage_mu_90': False, 'coverage_tau_90': False,
                'rhat_mu': np.nan, 'rhat_tau': np.nan,
                'converged': False, 'n_divergences': np.nan, 'success': False
            })

scenario_df = pd.DataFrame(scenario_results)
scenario_df.to_csv('/workspace/experiments/experiment_2/simulation_based_validation/scenario_results.csv', index=False)

print("\n" + "="*80)
print("SCENARIO SUMMARY")
print("="*80)

for scenario in scenarios:
    subset = scenario_df[scenario_df['scenario'] == scenario['name']]
    conv_rate = subset['converged'].sum() / len(subset) * 100
    success_rate = subset['success'].sum() / len(subset) * 100

    print(f"\n{scenario['name']} (τ={scenario['tau']}):")
    print(f"  Convergence: {conv_rate:.0f}% ({subset['converged'].sum()}/{len(subset)})")
    print(f"  Success: {success_rate:.0f}% ({subset['success'].sum()}/{len(subset)})")

    valid = subset[subset['converged']]
    if len(valid) > 0:
        print(f"  Rel. Error: μ={valid['rel_error_mu'].mean():.1f}%, τ={valid['rel_error_tau'].mean():.1f}%")


# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

valid_sbc = sbc_df[sbc_df['success']]

if len(valid_sbc) >= 10:
    # Plot 1: Rank histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_bins = 20
    expected = len(valid_sbc) / n_bins

    axes[0].hist(valid_sbc['rank_mu'], bins=n_bins, edgecolor='black', alpha=0.7)
    axes[0].axhline(expected, color='red', linestyle='--', label=f'Expected: {expected:.1f}')
    axes[0].set_xlabel('Rank')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('SBC Ranks: μ')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(valid_sbc['rank_tau'], bins=n_bins, edgecolor='black', alpha=0.7)
    axes[1].axhline(expected, color='red', linestyle='--', label=f'Expected: {expected:.1f}')
    axes[1].set_xlabel('Rank')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('SBC Ranks: τ')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/sbc_rank_histograms.png', dpi=300)
    print("  Saved: sbc_rank_histograms.png")
    plt.close()

    # KS tests
    ks_mu = stats.kstest(valid_sbc['rank_mu'] / 1000, 'uniform')
    ks_tau = stats.kstest(valid_sbc['rank_tau'] / 1000, 'uniform')
    print(f"  KS test μ: p={ks_mu.pvalue:.4f}")
    print(f"  KS test τ: p={ks_tau.pvalue:.4f}")

    # Plot 2: Parameter recovery
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(valid_sbc['mu_true'], valid_sbc['mu_mean'], alpha=0.6, s=50)
    lims = [valid_sbc['mu_true'].min(), valid_sbc['mu_true'].max()]
    axes[0].plot(lims, lims, 'r--', label='Perfect Recovery')
    axes[0].set_xlabel('True μ')
    axes[0].set_ylabel('Estimated μ')
    axes[0].set_title('Parameter Recovery: μ')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(valid_sbc['tau_true'], valid_sbc['tau_mean'], alpha=0.6, s=50)
    lims = [0, valid_sbc['tau_true'].max()]
    axes[1].plot(lims, lims, 'r--', label='Perfect Recovery')
    axes[1].set_xlabel('True τ')
    axes[1].set_ylabel('Estimated τ')
    axes[1].set_title('Parameter Recovery: τ')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/parameter_recovery.png', dpi=300)
    print("  Saved: parameter_recovery.png")
    plt.close()

    # Plot 3: Scenario comparison
    scenario_valid = scenario_df[scenario_df['converged']]
    if len(scenario_valid) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Recovery errors
        summary = scenario_valid.groupby('scenario').agg({'rel_error_mu': 'mean', 'rel_error_tau': 'mean'}).reset_index()
        x = np.arange(len(summary))
        width = 0.35

        axes[0].bar(x - width/2, summary['rel_error_mu'], width, label='μ', alpha=0.7)
        axes[0].bar(x + width/2, summary['rel_error_tau'], width, label='τ', alpha=0.7)
        axes[0].axhline(30, color='red', linestyle='--', label='30% Threshold')
        axes[0].set_ylabel('Mean Relative Error (%)')
        axes[0].set_xlabel('Scenario')
        axes[0].set_title('Recovery Error by Scenario')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(summary['scenario'], rotation=15, ha='right')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Coverage
        cov_summary = scenario_valid.groupby('scenario').agg({'coverage_mu_90': 'mean', 'coverage_tau_90': 'mean'}).reset_index()

        axes[1].bar(x - width/2, cov_summary['coverage_mu_90']*100, width, label='μ', alpha=0.7)
        axes[1].bar(x + width/2, cov_summary['coverage_tau_90']*100, width, label='τ', alpha=0.7)
        axes[1].axhline(90, color='red', linestyle='--', label='90% Target')
        axes[1].set_ylabel('Coverage (%)')
        axes[1].set_xlabel('Scenario')
        axes[1].set_title('90% Interval Coverage')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cov_summary['scenario'], rotation=15, ha='right')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('/workspace/experiments/experiment_2/simulation_based_validation/plots/scenario_comparison.png', dpi=300)
        print("  Saved: scenario_comparison.png")
        plt.close()
else:
    print(f"  WARNING: Only {len(valid_sbc)} successful sims. Skipping plots.")


# ============================================================================
# FINAL ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("FINAL ASSESSMENT")
print("="*80)

convergence_rate = sbc_df['converged'].sum() / len(sbc_df) * 100

if len(valid_sbc) >= 10:
    coverage_mu = valid_sbc['coverage_mu_90'].mean() * 100
    coverage_tau = valid_sbc['coverage_tau_90'].mean() * 100
    mean_rel_error_mu = valid_sbc['rel_error_mu'].mean()
    mean_rel_error_tau = valid_sbc['rel_error_tau'].mean()
    mean_divergences = valid_sbc['n_divergences'].mean()
    pct_divergence_rate = mean_divergences / 1000 * 100

    ks_mu_pass = ks_mu.pvalue > 0.05
    ks_tau_pass = ks_tau.pvalue > 0.05
else:
    coverage_mu = 0
    coverage_tau = 0
    mean_rel_error_mu = np.nan
    mean_rel_error_tau = np.nan
    pct_divergence_rate = np.nan
    ks_mu_pass = False
    ks_tau_pass = False
    ks_mu = type('obj', (object,), {'pvalue': np.nan})()
    ks_tau = type('obj', (object,), {'pvalue': np.nan})()

# High heterogeneity scenario
high_het = scenario_df[scenario_df['scenario'] == 'High Heterogeneity']
high_het_valid = high_het[high_het['converged']]
if len(high_het_valid) > 0:
    high_het_conv = high_het['converged'].sum() / len(high_het) * 100
    high_het_err_mu = high_het_valid['rel_error_mu'].mean()
    high_het_err_tau = high_het_valid['rel_error_tau'].mean()
    high_het_pass = (high_het_conv >= 80 and high_het_err_mu < 30 and high_het_err_tau < 30)
else:
    high_het_conv = 0
    high_het_err_mu = np.nan
    high_het_err_tau = np.nan
    high_het_pass = False

# PASS/FAIL criteria
criteria = {
    'convergence': convergence_rate >= 80,
    'coverage_mu': coverage_mu >= 85,
    'coverage_tau': coverage_tau >= 85,
    'calibration_mu': ks_mu_pass,
    'calibration_tau': ks_tau_pass,
    'high_het': high_het_pass,
    'divergences': pct_divergence_rate < 1.0 if not np.isnan(pct_divergence_rate) else False
}

overall_pass = all(criteria.values())

print("\nCRITERIA:")
print(f"  {'PASS' if criteria['convergence'] else 'FAIL'} Convergence >= 80%: {convergence_rate:.1f}%")
print(f"  {'PASS' if criteria['coverage_mu'] else 'FAIL'} Coverage μ >= 85%: {coverage_mu:.1f}%")
print(f"  {'PASS' if criteria['coverage_tau'] else 'FAIL'} Coverage τ >= 85%: {coverage_tau:.1f}%")
print(f"  {'PASS' if criteria['calibration_mu'] else 'FAIL'} Calibration μ (KS p>0.05): {ks_mu.pvalue:.4f}")
print(f"  {'PASS' if criteria['calibration_tau'] else 'FAIL'} Calibration τ (KS p>0.05): {ks_tau.pvalue:.4f}")
print(f"  {'PASS' if criteria['high_het'] else 'FAIL'} High heterogeneity: Conv={high_het_conv:.0f}%, Err μ={high_het_err_mu:.1f}%, τ={high_het_err_tau:.1f}%")
print(f"  {'PASS' if criteria['divergences'] else 'FAIL'} Divergences < 1%: {pct_divergence_rate:.3f}%")

print(f"\n{'='*80}")
print(f"OVERALL RESULT: {'PASS' if overall_pass else 'FAIL'}")
print(f"{'='*80}")

if overall_pass:
    print("\nModel successfully recovers known parameters. Ready for real data.")
else:
    print("\nParameter recovery issues detected. Review diagnostics.")

# Save assessment
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
    f.write(f"  KS Test μ: p={ks_mu.pvalue:.4f}\n")
    f.write(f"  KS Test τ: p={ks_tau.pvalue:.4f}\n")
    f.write(f"  Divergence Rate: {pct_divergence_rate:.3f}%\n\n")

    f.write("High Heterogeneity Scenario (τ=1.2, ICC=0.66):\n")
    f.write(f"  Convergence: {high_het_conv:.0f}%\n")
    f.write(f"  Rel. Error μ: {high_het_err_mu:.1f}%\n")
    f.write(f"  Rel. Error τ: {high_het_err_tau:.1f}%\n")

print("\nSaved: assessment_summary.txt")
print("\nValidation complete!")
