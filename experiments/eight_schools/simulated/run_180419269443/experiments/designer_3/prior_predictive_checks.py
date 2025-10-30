"""
Prior Predictive Checks for Bayesian Meta-Analysis Models
Designer 3 - Model Validation

Purpose: Validate priors BEFORE fitting to data
Principle: If prior predictive distributions don't include observed data,
           priors are either too informative or misspecified.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_data(path='/workspace/data/data.csv'):
    """Load observed meta-analysis data"""
    data = pd.read_csv(path)
    return data['y'].values, data['sigma'].values


def weighted_mean(y, sigma):
    """Compute inverse-variance weighted mean"""
    weights = 1 / sigma**2
    return np.sum(weights * y) / np.sum(weights)


def i_squared(y, sigma):
    """Compute I² heterogeneity statistic"""
    J = len(y)
    weights = 1 / sigma**2
    pooled = weighted_mean(y, sigma)
    Q = np.sum(weights * (y - pooled)**2)
    I2 = max(0, (Q - (J - 1)) / Q)
    return I2 * 100  # Return as percentage


# ============================================================
# PRIOR PREDICTIVE SIMULATION
# ============================================================

def simulate_prior_predictive_model1(n_sims=1000, J=8, sigma_obs=None):
    """
    Prior predictive simulation for Model 1 (Weakly Informative)

    Priors:
        mu ~ Normal(0, 25)
        tau ~ Half-Normal(0, 10)

    Parameters
    ----------
    n_sims : int
        Number of prior predictive datasets to simulate
    J : int
        Number of studies
    sigma_obs : array-like
        Observed standard errors (use actual values from data)

    Returns
    -------
    results : dict
        Dictionary with simulated datasets and summary statistics
    """
    # Use observed sigmas if provided
    if sigma_obs is None:
        sigma_obs = np.array([15, 10, 16, 11, 9, 11, 10, 18])  # From data

    # Storage
    mu_sims = np.zeros(n_sims)
    tau_sims = np.zeros(n_sims)
    y_sims = np.zeros((n_sims, J))
    pooled_means = np.zeros(n_sims)
    i2_sims = np.zeros(n_sims)
    ranges = np.zeros(n_sims)

    for s in range(n_sims):
        # Sample from priors
        mu = np.random.normal(0, 25)
        tau = np.abs(np.random.normal(0, 10))  # Half-normal

        # Generate study-level effects
        theta = np.random.normal(mu, tau, J)

        # Generate observed effects
        y = np.random.normal(theta, sigma_obs)

        # Store
        mu_sims[s] = mu
        tau_sims[s] = tau
        y_sims[s, :] = y
        pooled_means[s] = weighted_mean(y, sigma_obs)
        i2_sims[s] = i_squared(y, sigma_obs)
        ranges[s] = y.max() - y.min()

    results = {
        'mu': mu_sims,
        'tau': tau_sims,
        'y': y_sims,
        'pooled_mean': pooled_means,
        'i_squared': i2_sims,
        'range': ranges,
        'min': y_sims.min(axis=1),
        'max': y_sims.max(axis=1),
        'sd': y_sims.std(axis=1)
    }

    return results


def simulate_prior_predictive_model3a(n_sims=1000, J=8, sigma_obs=None):
    """
    Prior predictive simulation for Model 3a (Skeptical)

    Priors:
        mu ~ Normal(0, 10)  # Tighter, centered at null
        tau ~ Half-Normal(0, 5)  # Expect low heterogeneity
    """
    if sigma_obs is None:
        sigma_obs = np.array([15, 10, 16, 11, 9, 11, 10, 18])

    mu_sims = np.zeros(n_sims)
    tau_sims = np.zeros(n_sims)
    y_sims = np.zeros((n_sims, J))
    pooled_means = np.zeros(n_sims)
    i2_sims = np.zeros(n_sims)

    for s in range(n_sims):
        mu = np.random.normal(0, 10)  # Skeptical: tight around null
        tau = np.abs(np.random.normal(0, 5))
        theta = np.random.normal(mu, tau, J)
        y = np.random.normal(theta, sigma_obs)

        mu_sims[s] = mu
        tau_sims[s] = tau
        y_sims[s, :] = y
        pooled_means[s] = weighted_mean(y, sigma_obs)
        i2_sims[s] = i_squared(y, sigma_obs)

    return {
        'mu': mu_sims,
        'tau': tau_sims,
        'y': y_sims,
        'pooled_mean': pooled_means,
        'i_squared': i2_sims
    }


def simulate_prior_predictive_model3b(n_sims=1000, J=8, sigma_obs=None):
    """
    Prior predictive simulation for Model 3b (Enthusiastic)

    Priors:
        mu ~ Normal(15, 15)  # Optimistic: large positive effect
        tau ~ Half-Cauchy(0, 10)  # Allow high heterogeneity
    """
    if sigma_obs is None:
        sigma_obs = np.array([15, 10, 16, 11, 9, 11, 10, 18])

    mu_sims = np.zeros(n_sims)
    tau_sims = np.zeros(n_sims)
    y_sims = np.zeros((n_sims, J))
    pooled_means = np.zeros(n_sims)
    i2_sims = np.zeros(n_sims)

    for s in range(n_sims):
        mu = np.random.normal(15, 15)  # Enthusiastic
        tau = stats.halfcauchy.rvs(scale=10)  # Heavy-tailed
        theta = np.random.normal(mu, tau, J)
        y = np.random.normal(theta, sigma_obs)

        mu_sims[s] = mu
        tau_sims[s] = tau
        y_sims[s, :] = y
        pooled_means[s] = weighted_mean(y, sigma_obs)
        i2_sims[s] = i_squared(y, sigma_obs)

    return {
        'mu': mu_sims,
        'tau': tau_sims,
        'y': y_sims,
        'pooled_mean': pooled_means,
        'i_squared': i2_sims
    }


# ============================================================
# DIAGNOSTIC CHECKS
# ============================================================

def check_prior_includes_observed(prior_sims, observed_value, stat_name='statistic'):
    """
    Check if observed value is within prior predictive distribution

    Parameters
    ----------
    prior_sims : array
        Prior predictive samples for a statistic
    observed_value : float
        Observed value from data
    stat_name : str
        Name of statistic for printing

    Returns
    -------
    result : dict
        Diagnostic results
    """
    # Compute percentile of observed value in prior predictive
    percentile = stats.percentileofscore(prior_sims, observed_value)

    # Check if in middle 90% (5th-95th percentile)
    in_middle_90 = 5 < percentile < 95
    in_middle_50 = 25 < percentile < 75

    # Compute prior predictive intervals
    prior_50 = np.percentile(prior_sims, [25, 75])
    prior_90 = np.percentile(prior_sims, [5, 95])
    prior_95 = np.percentile(prior_sims, [2.5, 97.5])

    result = {
        'statistic': stat_name,
        'observed': observed_value,
        'percentile': percentile,
        'in_middle_50': in_middle_50,
        'in_middle_90': in_middle_90,
        'prior_50_CI': prior_50,
        'prior_90_CI': prior_90,
        'prior_95_CI': prior_95,
        'status': 'PASS' if in_middle_90 else 'FAIL'
    }

    return result


def print_check_result(result):
    """Pretty print diagnostic check result"""
    print(f"\n{result['statistic'].upper()}")
    print(f"  Observed value: {result['observed']:.2f}")
    print(f"  Prior predictive percentile: {result['percentile']:.1f}%")
    print(f"  Prior 50% CI: [{result['prior_50_CI'][0]:.2f}, {result['prior_50_CI'][1]:.2f}]")
    print(f"  Prior 90% CI: [{result['prior_90_CI'][0]:.2f}, {result['prior_90_CI'][1]:.2f}]")
    print(f"  Prior 95% CI: [{result['prior_95_CI'][0]:.2f}, {result['prior_95_CI'][1]:.2f}]")
    print(f"  In middle 50%: {'YES' if result['in_middle_50'] else 'NO'}")
    print(f"  In middle 90%: {'YES' if result['in_middle_90'] else 'NO'}")
    print(f"  STATUS: {result['status']}")


# ============================================================
# VISUALIZATION
# ============================================================

def plot_prior_predictive_diagnostics(prior_sims, y_obs, sigma_obs,
                                       model_name='Model 1',
                                       save_path=None):
    """
    Create comprehensive prior predictive diagnostic plots

    Parameters
    ----------
    prior_sims : dict
        Results from simulate_prior_predictive_*
    y_obs : array
        Observed effect sizes
    sigma_obs : array
        Observed standard errors
    model_name : str
        Name of model for title
    save_path : str
        Path to save figure (if None, display only)
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Prior Predictive Checks - {model_name}',
                 fontsize=16, fontweight='bold')

    # Compute observed statistics
    obs_mean = weighted_mean(y_obs, sigma_obs)
    obs_i2 = i_squared(y_obs, sigma_obs)
    obs_range = y_obs.max() - y_obs.min()
    obs_sd = y_obs.std()

    # 1. Prior on mu
    ax = axes[0, 0]
    ax.hist(prior_sims['mu'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(obs_mean, color='red', linewidth=2, label=f'Observed: {obs_mean:.2f}')
    ax.set_xlabel('mu (mean effect)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Distribution: mu')
    ax.legend()

    # 2. Prior on tau
    ax = axes[0, 1]
    ax.hist(prior_sims['tau'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('tau (between-study SD)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Distribution: tau')

    # 3. Prior predictive: pooled mean
    ax = axes[0, 2]
    ax.hist(prior_sims['pooled_mean'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(obs_mean, color='red', linewidth=2, label=f'Observed: {obs_mean:.2f}')
    ax.set_xlabel('Pooled mean')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Predictive: Pooled Mean')
    percentile = stats.percentileofscore(prior_sims['pooled_mean'], obs_mean)
    ax.text(0.05, 0.95, f'Percentile: {percentile:.1f}%',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()

    # 4. Prior predictive: I²
    ax = axes[1, 0]
    ax.hist(prior_sims['i_squared'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(obs_i2, color='red', linewidth=2, label=f'Observed: {obs_i2:.1f}%')
    ax.set_xlabel('I² (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Predictive: I² Statistic')
    percentile = stats.percentileofscore(prior_sims['i_squared'], obs_i2)
    ax.text(0.95, 0.95, f'Percentile: {percentile:.1f}%',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()

    # 5. Prior predictive: range
    ax = axes[1, 1]
    ax.hist(prior_sims['range'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(obs_range, color='red', linewidth=2, label=f'Observed: {obs_range:.2f}')
    ax.set_xlabel('Range (max - min)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Predictive: Range')
    ax.legend()

    # 6. Prior predictive: SD
    ax = axes[1, 2]
    ax.hist(prior_sims['sd'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(obs_sd, color='red', linewidth=2, label=f'Observed: {obs_sd:.2f}')
    ax.set_xlabel('Standard deviation')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Predictive: SD')
    ax.legend()

    # 7. Prior predictive: min
    ax = axes[2, 0]
    ax.hist(prior_sims['min'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(y_obs.min(), color='red', linewidth=2, label=f'Observed: {y_obs.min():.2f}')
    ax.set_xlabel('Minimum effect')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Predictive: Minimum')
    ax.legend()

    # 8. Prior predictive: max
    ax = axes[2, 1]
    ax.hist(prior_sims['max'], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(y_obs.max(), color='red', linewidth=2, label=f'Observed: {y_obs.max():.2f}')
    ax.set_xlabel('Maximum effect')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Predictive: Maximum')
    ax.legend()

    # 9. Overall assessment
    ax = axes[2, 2]
    ax.axis('off')

    # Check all statistics
    checks = [
        check_prior_includes_observed(prior_sims['pooled_mean'], obs_mean, 'Pooled mean'),
        check_prior_includes_observed(prior_sims['i_squared'], obs_i2, 'I²'),
        check_prior_includes_observed(prior_sims['range'], obs_range, 'Range'),
        check_prior_includes_observed(prior_sims['sd'], obs_sd, 'SD'),
        check_prior_includes_observed(prior_sims['min'], y_obs.min(), 'Min'),
        check_prior_includes_observed(prior_sims['max'], y_obs.max(), 'Max')
    ]

    n_pass = sum([c['status'] == 'PASS' for c in checks])
    n_total = len(checks)

    assessment_text = f"Overall Assessment\n{'='*20}\n\n"
    assessment_text += f"Checks passed: {n_pass}/{n_total}\n\n"

    for c in checks:
        status_symbol = '✓' if c['status'] == 'PASS' else '✗'
        in_50 = '✓' if c['in_middle_50'] else '✗'
        assessment_text += f"{status_symbol} {c['statistic']}: {c['percentile']:.0f}% (50%: {in_50})\n"

    overall_status = 'PASS' if n_pass >= 5 else 'FAIL'
    assessment_text += f"\nOVERALL: {overall_status}"

    ax.text(0.1, 0.9, assessment_text, transform=ax.transAxes,
            verticalalignment='top', fontfamily='monospace', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen' if overall_status == 'PASS' else 'lightcoral', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    return fig, checks


def plot_model_comparison(sims_dict, y_obs, sigma_obs, save_path=None):
    """
    Compare prior predictive distributions across models

    Parameters
    ----------
    sims_dict : dict
        Dictionary with keys = model names, values = prior_sims results
    y_obs : array
        Observed effects
    sigma_obs : array
        Observed SEs
    save_path : str
        Path to save figure
    """
    obs_mean = weighted_mean(y_obs, sigma_obs)
    obs_i2 = i_squared(y_obs, sigma_obs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Prior Predictive Comparison Across Models', fontsize=16, fontweight='bold')

    colors = ['blue', 'green', 'orange']
    model_names = list(sims_dict.keys())

    # 1. Pooled mean
    ax = axes[0, 0]
    for i, name in enumerate(model_names):
        sims = sims_dict[name]
        ax.hist(sims['pooled_mean'], bins=50, alpha=0.5, label=name, color=colors[i])
    ax.axvline(obs_mean, color='red', linewidth=2, linestyle='--', label='Observed')
    ax.set_xlabel('Pooled mean')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Predictive: Pooled Mean')
    ax.legend()

    # 2. I²
    ax = axes[0, 1]
    for i, name in enumerate(model_names):
        sims = sims_dict[name]
        ax.hist(sims['i_squared'], bins=50, alpha=0.5, label=name, color=colors[i])
    ax.axvline(obs_i2, color='red', linewidth=2, linestyle='--', label='Observed')
    ax.set_xlabel('I² (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Predictive: I²')
    ax.legend()

    # 3. Prior on mu
    ax = axes[1, 0]
    for i, name in enumerate(model_names):
        sims = sims_dict[name]
        ax.hist(sims['mu'], bins=50, alpha=0.5, label=name, color=colors[i])
    ax.axvline(obs_mean, color='red', linewidth=2, linestyle='--', label='Observed mean')
    ax.set_xlabel('mu (prior)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Distribution: mu')
    ax.legend()

    # 4. Prior on tau
    ax = axes[1, 1]
    for i, name in enumerate(model_names):
        sims = sims_dict[name]
        ax.hist(sims['tau'], bins=50, alpha=0.5, label=name, color=colors[i])
    ax.set_xlabel('tau (prior)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prior Distribution: tau')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    return fig


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_all_prior_checks(save_dir='/workspace/experiments/designer_3/visualizations'):
    """
    Run prior predictive checks for all models

    Parameters
    ----------
    save_dir : str
        Directory to save diagnostic plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("="*70)
    print("PRIOR PREDICTIVE CHECKS")
    print("="*70)

    # Load observed data
    y_obs, sigma_obs = load_data()
    print(f"\nObserved data:")
    print(f"  J = {len(y_obs)} studies")
    print(f"  Pooled mean = {weighted_mean(y_obs, sigma_obs):.2f}")
    print(f"  I² = {i_squared(y_obs, sigma_obs):.1f}%")
    print(f"  Range = [{y_obs.min():.2f}, {y_obs.max():.2f}]")

    # Simulate prior predictive for each model
    print("\n" + "="*70)
    print("SIMULATING PRIOR PREDICTIVE DISTRIBUTIONS...")
    print("="*70)

    print("\nModel 1 (Weakly Informative)...")
    sims1 = simulate_prior_predictive_model1(n_sims=10000, sigma_obs=sigma_obs)

    print("Model 3a (Skeptical)...")
    sims3a = simulate_prior_predictive_model3a(n_sims=10000, sigma_obs=sigma_obs)

    print("Model 3b (Enthusiastic)...")
    sims3b = simulate_prior_predictive_model3b(n_sims=10000, sigma_obs=sigma_obs)

    # Run diagnostics for each model
    print("\n" + "="*70)
    print("MODEL 1: WEAKLY INFORMATIVE")
    print("="*70)
    fig1, checks1 = plot_prior_predictive_diagnostics(
        sims1, y_obs, sigma_obs,
        model_name='Model 1 (Weakly Informative)',
        save_path=f'{save_dir}/prior_checks_model1.png'
    )
    for c in checks1:
        print_check_result(c)

    print("\n" + "="*70)
    print("MODEL 3A: SKEPTICAL")
    print("="*70)
    fig3a, checks3a = plot_prior_predictive_diagnostics(
        sims3a, y_obs, sigma_obs,
        model_name='Model 3a (Skeptical)',
        save_path=f'{save_dir}/prior_checks_model3a.png'
    )
    for c in checks3a:
        print_check_result(c)

    print("\n" + "="*70)
    print("MODEL 3B: ENTHUSIASTIC")
    print("="*70)
    fig3b, checks3b = plot_prior_predictive_diagnostics(
        sims3b, y_obs, sigma_obs,
        model_name='Model 3b (Enthusiastic)',
        save_path=f'{save_dir}/prior_checks_model3b.png'
    )
    for c in checks3b:
        print_check_result(c)

    # Compare models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    sims_dict = {
        'Model 1 (Weakly Informative)': sims1,
        'Model 3a (Skeptical)': sims3a,
        'Model 3b (Enthusiastic)': sims3b
    }
    fig_comp = plot_model_comparison(
        sims_dict, y_obs, sigma_obs,
        save_path=f'{save_dir}/prior_comparison.png'
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    all_checks = {
        'Model 1': checks1,
        'Model 3a': checks3a,
        'Model 3b': checks3b
    }

    for model_name, checks in all_checks.items():
        n_pass = sum([c['status'] == 'PASS' for c in checks])
        overall = 'PASS' if n_pass >= 5 else 'FAIL'
        print(f"\n{model_name}: {n_pass}/6 checks passed - {overall}")

    print("\n" + "="*70)
    print("Prior predictive checks complete!")
    print(f"Visualizations saved to: {save_dir}")
    print("="*70)

    return {
        'sims': sims_dict,
        'checks': all_checks,
        'figures': {
            'model1': fig1,
            'model3a': fig3a,
            'model3b': fig3b,
            'comparison': fig_comp
        }
    }


if __name__ == '__main__':
    # Run all prior predictive checks
    results = run_all_prior_checks()

    # Show plots (comment out if running headless)
    # plt.show()
