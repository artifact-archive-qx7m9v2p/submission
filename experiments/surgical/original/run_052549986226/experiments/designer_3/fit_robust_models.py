"""
Fit robust Bayesian models for binomial data with outliers
Designer 3: Student-t, Horseshoe, and Mixture models

Usage:
    python fit_robust_models.py --model student_t
    python fit_robust_models.py --model horseshoe
    python fit_robust_models.py --model mixture
    python fit_robust_models.py --model all
"""

import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
MODEL_DIR = Path("/workspace/experiments/designer_3")
OUTPUT_DIR = MODEL_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load and prepare data for Stan"""
    df = pd.read_csv(DATA_PATH)

    stan_data = {
        'N': len(df),
        'n_trials': df['n_trials'].values.astype(int),
        'r': df['r_successes'].values.astype(int)
    }

    return df, stan_data

def calculate_tau0(N=12, p0=3, sigma_y=1.0):
    """
    Calculate horseshoe prior scale parameter

    Parameters:
    - N: Total number of groups
    - p0: Expected number of non-zero effects
    - sigma_y: Expected effect size on logit scale

    Returns:
    - tau0: Prior scale for global shrinkage
    """
    tau0 = (p0 / (N - p0)) * (sigma_y / np.sqrt(N))
    return tau0

def fit_student_t_model(stan_data, chains=4, iter_sampling=2000, iter_warmup=1000):
    """Fit Student-t hierarchical model"""
    print("\n" + "="*70)
    print("FITTING STUDENT-T HIERARCHICAL MODEL")
    print("="*70)

    model_path = MODEL_DIR / "student_t_hierarchical.stan"
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    print(f"\nModel: {model_path}")
    print(f"Chains: {chains}, Iterations: {iter_sampling}, Warmup: {iter_warmup}")

    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_sampling=iter_sampling,
        iter_warmup=iter_warmup,
        adapt_delta=0.95,
        max_treedepth=12,
        show_progress=True
    )

    print("\n" + "-"*70)
    print("SAMPLING DIAGNOSTICS")
    print("-"*70)
    print(fit.diagnose())

    return fit

def fit_horseshoe_model(stan_data, chains=4, iter_sampling=3000, iter_warmup=1500):
    """Fit Horseshoe prior model"""
    print("\n" + "="*70)
    print("FITTING HORSESHOE PRIOR MODEL")
    print("="*70)

    # Add tau0 to data
    tau0 = calculate_tau0(N=stan_data['N'], p0=3, sigma_y=1.0)
    stan_data_hs = stan_data.copy()
    stan_data_hs['tau0'] = tau0

    print(f"\nHorseshoe prior scale (tau0): {tau0:.4f}")
    print(f"  Based on: p0={3} expected outliers out of N={stan_data['N']} groups")

    model_path = MODEL_DIR / "horseshoe_hierarchical.stan"
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    print(f"\nModel: {model_path}")
    print(f"Chains: {chains}, Iterations: {iter_sampling}, Warmup: {iter_warmup}")

    fit = model.sample(
        data=stan_data_hs,
        chains=chains,
        iter_sampling=iter_sampling,
        iter_warmup=iter_warmup,
        adapt_delta=0.95,
        max_treedepth=12,
        show_progress=True
    )

    print("\n" + "-"*70)
    print("SAMPLING DIAGNOSTICS")
    print("-"*70)
    print(fit.diagnose())

    return fit

def fit_mixture_model(stan_data, chains=4, iter_sampling=4000, iter_warmup=2000):
    """Fit finite mixture model"""
    print("\n" + "="*70)
    print("FITTING MIXTURE MODEL (2 COMPONENTS)")
    print("="*70)

    # Add K (number of components) to data
    stan_data_mix = stan_data.copy()
    stan_data_mix['K'] = 2

    model_path = MODEL_DIR / "mixture_hierarchical.stan"
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))

    print(f"\nModel: {model_path}")
    print(f"Chains: {chains}, Iterations: {iter_sampling}, Warmup: {iter_warmup}")
    print("  Note: Longer iterations due to marginalizing over cluster assignments")

    # Initialize chains with plausible cluster means
    init = [{'mu': np.array([-3.0, -2.0])} for _ in range(chains)]

    fit = model.sample(
        data=stan_data_mix,
        chains=chains,
        iter_sampling=iter_sampling,
        iter_warmup=iter_warmup,
        adapt_delta=0.95,
        max_treedepth=12,
        inits=init,
        show_progress=True
    )

    print("\n" + "-"*70)
    print("SAMPLING DIAGNOSTICS")
    print("-"*70)
    print(fit.diagnose())

    return fit

def summarize_model(fit, model_name, df):
    """Print comprehensive model summary"""
    print("\n" + "="*70)
    print(f"SUMMARY: {model_name.upper()}")
    print("="*70)

    # Convert to InferenceData for ArviZ
    idata = az.from_cmdstanpy(fit)

    # Key parameters summary
    if model_name == 'student_t':
        params_of_interest = ['mu', 'sigma', 'nu', 'phi_posterior', 'is_outlier']
    elif model_name == 'horseshoe':
        params_of_interest = ['mu', 'tau', 'n_active', 'phi_posterior']
    elif model_name == 'mixture':
        params_of_interest = ['mu', 'sigma', 'pi', 'n_cluster1', 'n_cluster2',
                            'cluster_separation', 'phi_posterior']

    print("\n" + "-"*70)
    print("KEY PARAMETERS")
    print("-"*70)

    summary = az.summary(idata, var_names=params_of_interest)
    print(summary.to_string())

    # Group-specific parameters
    print("\n" + "-"*70)
    print("GROUP-SPECIFIC ESTIMATES")
    print("-"*70)

    if model_name in ['student_t', 'horseshoe']:
        alpha_summary = az.summary(idata, var_names=['alpha'])
        p_summary = az.summary(idata, var_names=['p'])

        group_df = pd.DataFrame({
            'group': range(1, 13),
            'n_trials': df['n_trials'].values,
            'r_observed': df['r_successes'].values,
            'rate_observed': df['success_rate'].values,
            'alpha_mean': alpha_summary['mean'].values,
            'alpha_sd': alpha_summary['sd'].values,
            'p_mean': p_summary['mean'].values,
            'p_2.5%': p_summary['hdi_3%'].values,
            'p_97.5%': p_summary['hdi_97%'].values
        })

        if model_name == 'horseshoe':
            lambda_summary = az.summary(idata, var_names=['lambda'])
            group_df['lambda_mean'] = lambda_summary['mean'].values

    elif model_name == 'mixture':
        prob_cluster = fit.stan_variable('prob_cluster')  # Shape: (n_draws, N, K)
        cluster_prob_mean = prob_cluster.mean(axis=0)  # Shape: (N, K)

        group_df = pd.DataFrame({
            'group': range(1, 13),
            'n_trials': df['n_trials'].values,
            'r_observed': df['r_successes'].values,
            'rate_observed': df['success_rate'].values,
            'P(cluster=1)': cluster_prob_mean[:, 0],
            'P(cluster=2)': cluster_prob_mean[:, 1]
        })

    print(group_df.to_string(index=False))

    # Falsification checks
    print("\n" + "-"*70)
    print("FALSIFICATION CRITERIA")
    print("-"*70)

    if model_name == 'student_t':
        nu_samples = fit.stan_variable('nu')
        nu_mean = nu_samples.mean()
        nu_q025 = np.percentile(nu_samples, 2.5)
        nu_q975 = np.percentile(nu_samples, 97.5)

        print(f"\nDegrees of freedom (nu):")
        print(f"  Mean: {nu_mean:.2f}")
        print(f"  95% CI: [{nu_q025:.2f}, {nu_q975:.2f}]")

        if nu_mean > 50:
            print("  ⚠️  WARNING: nu > 50, data may not need heavy tails (use Normal)")
        elif nu_mean < 5:
            print("  ⚠️  WARNING: nu < 5, very heavy tails (consider mixture model)")
        else:
            print("  ✓ nu in reasonable range [5, 50], Student-t appropriate")

    elif model_name == 'horseshoe':
        tau_samples = fit.stan_variable('tau')
        lambda_samples = fit.stan_variable('lambda')
        n_active_samples = fit.stan_variable('n_active')

        tau_mean = tau_samples.mean()
        n_active_mean = n_active_samples.mean()
        lambda_mean = lambda_samples.mean(axis=0)

        print(f"\nGlobal shrinkage (tau):")
        print(f"  Mean: {tau_mean:.4f}")
        print(f"  Prior expectation (tau0): {calculate_tau0():.4f}")

        print(f"\nNumber of active groups (lambda > 0.5):")
        print(f"  Mean: {n_active_mean:.1f}")
        print(f"  Expected: ~3")

        if np.allclose(lambda_mean, lambda_mean.mean(), rtol=0.3):
            print("  ⚠️  WARNING: All lambda similar, no sparsity detected")
        else:
            print("  ✓ Sparsity detected, some groups differ from others")

    elif model_name == 'mixture':
        pi_samples = fit.stan_variable('pi')
        mu_samples = fit.stan_variable('mu')

        pi_mean = pi_samples.mean(axis=0)
        mu_mean = mu_samples.mean(axis=0)
        sep_samples = fit.stan_variable('cluster_separation')
        sep_mean = sep_samples.mean()

        print(f"\nMixing proportions (pi):")
        print(f"  P(cluster 1): {pi_mean[0]:.3f}")
        print(f"  P(cluster 2): {pi_mean[1]:.3f}")

        print(f"\nCluster means (logit scale):")
        print(f"  mu[1]: {mu_mean[0]:.3f} -> {stats.expit(mu_mean[0]):.3f} on prob scale")
        print(f"  mu[2]: {mu_mean[1]:.3f} -> {stats.expit(mu_mean[1]):.3f} on prob scale")
        print(f"  Separation: {sep_mean:.3f}")

        if pi_mean[0] < 0.1 or pi_mean[0] > 0.9:
            print("  ⚠️  WARNING: Mixing proportion extreme, no mixture needed")
        elif sep_mean < 0.5:
            print("  ⚠️  WARNING: Cluster separation small, unclear if mixture needed")
        else:
            print("  ✓ Mixture model justified")

    # Posterior predictive checks
    print("\n" + "-"*70)
    print("POSTERIOR PREDICTIVE CHECKS")
    print("-"*70)

    phi_samples = fit.stan_variable('phi_posterior')
    phi_mean = phi_samples.mean()
    phi_q025 = np.percentile(phi_samples, 2.5)
    phi_q975 = np.percentile(phi_samples, 97.5)

    print(f"\nOverdispersion parameter (phi):")
    print(f"  Posterior mean: {phi_mean:.2f}")
    print(f"  95% CI: [{phi_q025:.2f}, {phi_q975:.2f}]")
    print(f"  Observed (from EDA): ~3.5-5.1")

    if 3.0 <= phi_mean <= 6.0:
        print("  ✓ Model reproduces observed overdispersion")
    else:
        print("  ⚠️  WARNING: Model may not capture overdispersion correctly")

    return idata, group_df

def compute_loo(fit, model_name):
    """Compute LOO-CV for model comparison"""
    print("\n" + "-"*70)
    print(f"LOO-CV: {model_name.upper()}")
    print("-"*70)

    idata = az.from_cmdstanpy(fit)
    loo = az.loo(idata)

    print(f"\nLOO-ELPD: {loo.loo:.2f}")
    print(f"LOO-SE: {loo.loo_se:.2f}")
    print(f"P-LOO: {loo.p_loo:.2f}")

    # Check Pareto k diagnostics
    k_values = loo.pareto_k.values
    print(f"\nPareto k diagnostics:")
    print(f"  k < 0.5 (good): {np.sum(k_values < 0.5)}/12 groups")
    print(f"  k > 0.7 (bad): {np.sum(k_values > 0.7)}/12 groups")

    if np.any(k_values > 0.7):
        bad_groups = np.where(k_values > 0.7)[0] + 1
        print(f"  ⚠️  Groups with high k: {bad_groups}")

    return loo

def plot_diagnostics(fit, model_name, df):
    """Create diagnostic plots"""
    print(f"\nCreating diagnostic plots for {model_name}...")

    idata = az.from_cmdstanpy(fit)

    # Create figure directory
    fig_dir = OUTPUT_DIR / f"{model_name}_diagnostics"
    fig_dir.mkdir(exist_ok=True)

    # 1. Trace plots for key parameters
    if model_name == 'student_t':
        var_names = ['mu', 'sigma', 'nu']
    elif model_name == 'horseshoe':
        var_names = ['mu', 'tau']
    elif model_name == 'mixture':
        var_names = ['mu', 'sigma', 'pi']

    az.plot_trace(idata, var_names=var_names)
    plt.tight_layout()
    plt.savefig(fig_dir / "trace_plots.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Posterior distributions
    az.plot_posterior(idata, var_names=var_names)
    plt.tight_layout()
    plt.savefig(fig_dir / "posterior_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Posterior predictive check
    r_rep = fit.stan_variable('r_rep')
    r_obs = df['r_successes'].values

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(12):
        ax = axes[i]
        ax.hist(r_rep[:, i], bins=30, alpha=0.6, color='blue', density=True, label='Posterior pred.')
        ax.axvline(r_obs[i], color='red', linewidth=2, label='Observed')
        ax.set_title(f"Group {i+1} (n={df['n_trials'].iloc[i]})")
        ax.set_xlabel("Successes")
        if i == 0:
            ax.legend()

    plt.suptitle(f"{model_name.upper()}: Posterior Predictive Checks", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(fig_dir / "posterior_predictive.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Model-specific plots
    if model_name == 'student_t':
        # Plot alpha with outlier flags
        alpha_samples = fit.stan_variable('alpha')
        is_outlier = fit.stan_variable('is_outlier').mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        positions = range(1, 13)
        parts = ax.violinplot([alpha_samples[:, i] for i in range(12)],
                               positions=positions, widths=0.7, showmeans=True)

        # Color outliers differently
        for i, pc in enumerate(parts['bodies']):
            if is_outlier[i] > 0.5:
                pc.set_facecolor('red')
                pc.set_alpha(0.6)
            else:
                pc.set_facecolor('blue')
                pc.set_alpha(0.4)

        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(2, color='red', linestyle='--', alpha=0.3, label='Outlier threshold')
        ax.axhline(-2, color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel("Group")
        ax.set_ylabel("Random effect (alpha)")
        ax.set_title("Student-t Model: Group Random Effects")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "random_effects.png", dpi=150, bbox_inches='tight')
        plt.close()

    elif model_name == 'horseshoe':
        # Plot lambda (shrinkage factors)
        lambda_samples = fit.stan_variable('lambda')

        fig, ax = plt.subplots(figsize=(10, 6))
        positions = range(1, 13)
        parts = ax.violinplot([lambda_samples[:, i] for i in range(12)],
                               positions=positions, widths=0.7, showmeans=True)

        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Active threshold')
        ax.set_xlabel("Group")
        ax.set_ylabel("Local shrinkage (lambda)")
        ax.set_title("Horseshoe Model: Local Shrinkage Parameters")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "shrinkage_factors.png", dpi=150, bbox_inches='tight')
        plt.close()

    elif model_name == 'mixture':
        # Plot cluster assignments
        prob_cluster = fit.stan_variable('prob_cluster')
        cluster_prob_mean = prob_cluster.mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(1, 13)
        width = 0.35

        ax.bar(x - width/2, cluster_prob_mean[:, 0], width, label='Cluster 1 (low rate)', alpha=0.7)
        ax.bar(x + width/2, cluster_prob_mean[:, 1], width, label='Cluster 2 (high rate)', alpha=0.7)

        ax.set_xlabel("Group")
        ax.set_ylabel("Posterior probability")
        ax.set_title("Mixture Model: Cluster Assignment Probabilities")
        ax.legend()
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(fig_dir / "cluster_assignments.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved to: {fig_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Fit robust Bayesian models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['student_t', 'horseshoe', 'mixture', 'all'],
                       help='Which model to fit')
    args = parser.parse_args()

    # Load data
    df, stan_data = load_data()

    print("="*70)
    print("ROBUST BAYESIAN MODELS FOR BINOMIAL DATA WITH OUTLIERS")
    print("="*70)
    print(f"\nDataset: {len(df)} groups")
    print(f"Total trials: {stan_data['n_trials'].sum()}")
    print(f"Total successes: {stan_data['r'].sum()}")
    print(f"Overall rate: {stan_data['r'].sum() / stan_data['n_trials'].sum():.4f}")

    results = {}

    # Fit models
    if args.model in ['student_t', 'all']:
        fit = fit_student_t_model(stan_data)
        idata, group_df = summarize_model(fit, 'student_t', df)
        loo = compute_loo(fit, 'student_t')
        plot_diagnostics(fit, 'student_t', df)
        results['student_t'] = {'fit': fit, 'idata': idata, 'loo': loo, 'group_df': group_df}

        # Save results
        fit.save_csvfiles(dir=str(OUTPUT_DIR / "student_t_samples"))
        group_df.to_csv(OUTPUT_DIR / "student_t_group_estimates.csv", index=False)

    if args.model in ['horseshoe', 'all']:
        fit = fit_horseshoe_model(stan_data)
        idata, group_df = summarize_model(fit, 'horseshoe', df)
        loo = compute_loo(fit, 'horseshoe')
        plot_diagnostics(fit, 'horseshoe', df)
        results['horseshoe'] = {'fit': fit, 'idata': idata, 'loo': loo, 'group_df': group_df}

        # Save results
        fit.save_csvfiles(dir=str(OUTPUT_DIR / "horseshoe_samples"))
        group_df.to_csv(OUTPUT_DIR / "horseshoe_group_estimates.csv", index=False)

    if args.model in ['mixture', 'all']:
        fit = fit_mixture_model(stan_data)
        idata, group_df = summarize_model(fit, 'mixture', df)
        loo = compute_loo(fit, 'mixture')
        plot_diagnostics(fit, 'mixture', df)
        results['mixture'] = {'fit': fit, 'idata': idata, 'loo': loo, 'group_df': group_df}

        # Save results
        fit.save_csvfiles(dir=str(OUTPUT_DIR / "mixture_samples"))
        group_df.to_csv(OUTPUT_DIR / "mixture_group_estimates.csv", index=False)

    # Model comparison
    if len(results) > 1:
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'LOO-ELPD': [results[m]['loo'].loo for m in results.keys()],
            'SE': [results[m]['loo'].loo_se for m in results.keys()],
            'P-LOO': [results[m]['loo'].p_loo for m in results.keys()]
        })

        comparison_df = comparison_df.sort_values('LOO-ELPD', ascending=False)
        print("\n" + comparison_df.to_string(index=False))

        best_model = comparison_df.iloc[0]['Model']
        print(f"\n✓ Best model by LOO-CV: {best_model.upper()}")

        # Compare best to others
        for i in range(1, len(comparison_df)):
            model = comparison_df.iloc[i]['Model']
            delta_loo = comparison_df.iloc[0]['LOO-ELPD'] - comparison_df.iloc[i]['LOO-ELPD']
            se_diff = np.sqrt(comparison_df.iloc[0]['SE']**2 + comparison_df.iloc[i]['SE']**2)

            if delta_loo < 2:
                print(f"  {model}: ΔLOO = {delta_loo:.1f} (models equivalent)")
            elif delta_loo < 10:
                print(f"  {model}: ΔLOO = {delta_loo:.1f} (some evidence for {best_model})")
            else:
                print(f"  {model}: ΔLOO = {delta_loo:.1f} (strong evidence for {best_model})")

        comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
