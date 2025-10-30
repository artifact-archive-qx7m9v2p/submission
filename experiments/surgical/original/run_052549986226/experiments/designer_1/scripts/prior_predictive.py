#!/usr/bin/env python3
"""
Prior predictive checks for all three models
Designer 1 - Verify priors are sensible before fitting

Usage:
    python prior_predictive.py
"""

import cmdstanpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
BASE_DIR = Path('/workspace/experiments/designer_1')
DATA_PATH = Path('/workspace/data/data.csv')
STAN_DIR = BASE_DIR / 'stan_models'
RESULTS_DIR = BASE_DIR / 'results'
PLOTS_DIR = RESULTS_DIR / 'prior_predictive_plots'
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

def create_prior_predictive_stan(model_name):
    """Create modified Stan file for prior predictive sampling"""

    if model_name == 'a':
        code = """
        data {
          int<lower=1> N;
          array[N] int<lower=0> n_trials;
        }

        generated quantities {
          real<lower=0> alpha = gamma_rng(2, 0.5);
          real<lower=0> beta = gamma_rng(2, 0.1);

          real mean_p = alpha / (alpha + beta);
          real var_p = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1));
          real phi = 1 + var_p / (mean_p * (1 - mean_p));

          array[N] real p_prior = rep_array(0.0, N);
          array[N] int r_prior = rep_array(0, N);

          for (i in 1:N) {
            p_prior[i] = beta_rng(alpha, beta);
            r_prior[i] = binomial_rng(n_trials[i], p_prior[i]);
          }
        }
        """
    elif model_name == 'b':
        code = """
        data {
          int<lower=1> N;
          array[N] int<lower=0> n_trials;
        }

        generated quantities {
          real<lower=0, upper=1> mu = beta_rng(2, 18);
          real<lower=0> kappa = gamma_rng(2, 0.1);

          real alpha = mu * kappa;
          real beta_val = (1 - mu) * kappa;
          real var_p = (mu * (1 - mu)) / (kappa + 1);
          real phi = 1 + 1 / kappa;

          array[N] real p_prior = rep_array(0.0, N);
          array[N] int r_prior = rep_array(0, N);

          for (i in 1:N) {
            p_prior[i] = beta_rng(alpha, beta_val);
            r_prior[i] = binomial_rng(n_trials[i], p_prior[i]);
          }
        }
        """
    else:  # model_name == 'c'
        code = """
        data {
          int<lower=1> N;
          array[N] int<lower=0> n_trials;
        }

        generated quantities {
          real<lower=0, upper=1> pi = beta_rng(2, 2);
          real<lower=0, upper=1> mu1 = beta_rng(3, 50);
          real<lower=0, upper=1> mu2 = beta_rng(5, 20);
          real<lower=0> kappa1 = gamma_rng(2, 0.1);
          real<lower=0> kappa2 = gamma_rng(2, 0.1);

          real alpha1 = mu1 * kappa1;
          real beta1 = (1 - mu1) * kappa1;
          real alpha2 = mu2 * kappa2;
          real beta2 = (1 - mu2) * kappa2;

          real mean_p = pi * mu1 + (1 - pi) * mu2;

          array[N] real p_prior = rep_array(0.0, N);
          array[N] int r_prior = rep_array(0, N);

          for (i in 1:N) {
            int component = bernoulli_rng(pi) + 1;
            if (component == 1) {
              p_prior[i] = beta_rng(alpha1, beta1);
            } else {
              p_prior[i] = beta_rng(alpha2, beta2);
            }
            r_prior[i] = binomial_rng(n_trials[i], p_prior[i]);
          }
        }
        """

    # Write to file
    prior_file = STAN_DIR / f'prior_predictive_{model_name}.stan'
    with open(prior_file, 'w') as f:
        f.write(code)

    return prior_file

def sample_prior_predictive(model_name, n_trials, n_draws=1000):
    """Sample from prior predictive distribution"""
    print(f"\nSampling prior predictive for Model {model_name.upper()}...")

    # Create prior predictive Stan file
    stan_file = create_prior_predictive_stan(model_name)

    # Compile model
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))

    # Sample
    data = {
        'N': len(n_trials),
        'n_trials': n_trials.tolist()
    }

    fit = model.sample(
        data=data,
        chains=1,
        iter_sampling=n_draws,
        fixed_param=True,
        show_console=False
    )

    return fit

def plot_prior_distributions(fit_a, fit_b, fit_c, data):
    """Create comprehensive prior predictive check plots"""

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Prior Predictive Distributions - All Models', fontsize=16, y=0.995)

    observed_mean = data['success_rate'].mean()
    observed_std = data['success_rate'].std()

    # Model A
    alpha_a = fit_a.stan_variable('alpha')
    beta_a = fit_a.stan_variable('beta')
    mean_p_a = fit_a.stan_variable('mean_p')
    phi_a = fit_a.stan_variable('phi')

    axes[0, 0].hist(alpha_a, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Model A: α prior')
    axes[0, 0].set_xlabel('α')
    axes[0, 0].axvline(alpha_a.mean(), color='red', linestyle='--', label=f'Mean: {alpha_a.mean():.1f}')
    axes[0, 0].legend()

    axes[0, 1].hist(beta_a, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Model A: β prior')
    axes[0, 1].set_xlabel('β')
    axes[0, 1].axvline(beta_a.mean(), color='red', linestyle='--', label=f'Mean: {beta_a.mean():.1f}')
    axes[0, 1].legend()

    axes[0, 2].hist(mean_p_a, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 2].set_title('Model A: Mean success rate')
    axes[0, 2].set_xlabel('Mean p')
    axes[0, 2].axvline(observed_mean, color='red', linestyle='--', label=f'Observed: {observed_mean:.3f}')
    axes[0, 2].axvline(mean_p_a.mean(), color='blue', linestyle='--', label=f'Prior mean: {mean_p_a.mean():.3f}')
    axes[0, 2].legend()

    # Model B
    mu_b = fit_b.stan_variable('mu')
    kappa_b = fit_b.stan_variable('kappa')
    phi_b = fit_b.stan_variable('phi')

    axes[1, 0].hist(mu_b, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Model B: μ prior')
    axes[1, 0].set_xlabel('μ (mean success rate)')
    axes[1, 0].axvline(observed_mean, color='red', linestyle='--', label=f'Observed: {observed_mean:.3f}')
    axes[1, 0].axvline(mu_b.mean(), color='blue', linestyle='--', label=f'Prior mean: {mu_b.mean():.3f}')
    axes[1, 0].legend()

    axes[1, 1].hist(kappa_b, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Model B: κ prior')
    axes[1, 1].set_xlabel('κ (concentration)')
    axes[1, 1].axvline(kappa_b.mean(), color='red', linestyle='--', label=f'Mean: {kappa_b.mean():.1f}')
    axes[1, 1].legend()

    axes[1, 2].hist(phi_b, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1, 2].set_title('Model B: Overdispersion φ')
    axes[1, 2].set_xlabel('φ')
    axes[1, 2].axvline(3.5, color='red', linestyle='--', label='Observed: ~3.5')
    axes[1, 2].axvline(phi_b.mean(), color='blue', linestyle='--', label=f'Prior mean: {phi_b.mean():.1f}')
    axes[1, 2].set_xlim(0, 50)
    axes[1, 2].legend()

    # Model C
    pi_c = fit_c.stan_variable('pi')
    mu1_c = fit_c.stan_variable('mu1')
    mu2_c = fit_c.stan_variable('mu2')
    mean_p_c = fit_c.stan_variable('mean_p')

    axes[2, 0].hist(pi_c, bins=50, alpha=0.7, edgecolor='black')
    axes[2, 0].set_title('Model C: π (mixing proportion)')
    axes[2, 0].set_xlabel('π')
    axes[2, 0].axvline(pi_c.mean(), color='red', linestyle='--', label=f'Mean: {pi_c.mean():.2f}')
    axes[2, 0].legend()

    axes[2, 1].hist(mu1_c, bins=50, alpha=0.5, edgecolor='black', label='μ₁ (low)', color='blue')
    axes[2, 1].hist(mu2_c, bins=50, alpha=0.5, edgecolor='black', label='μ₂ (high)', color='orange')
    axes[2, 1].set_title('Model C: Component means')
    axes[2, 1].set_xlabel('Component mean')
    axes[2, 1].legend()

    axes[2, 2].hist(mean_p_c, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[2, 2].set_title('Model C: Overall mean')
    axes[2, 2].set_xlabel('Mean p (weighted)')
    axes[2, 2].axvline(observed_mean, color='red', linestyle='--', label=f'Observed: {observed_mean:.3f}')
    axes[2, 2].axvline(mean_p_c.mean(), color='blue', linestyle='--', label=f'Prior mean: {mean_p_c.mean():.3f}')
    axes[2, 2].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'prior_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'prior_distributions.png'}")
    plt.close()

def plot_prior_predictive_data(fit_a, fit_b, fit_c, data):
    """Plot prior predictive success rates vs observed"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Prior Predictive Success Rates vs Observed', fontsize=16)

    # Get prior predictive samples
    p_prior_a = fit_a.stan_variable('p_prior')
    p_prior_b = fit_b.stan_variable('p_prior')
    p_prior_c = fit_c.stan_variable('p_prior')

    observed = data['success_rate'].values
    groups = data['group'].values

    # Model A
    for i in range(len(groups)):
        axes[0].violinplot([p_prior_a[:, i]], positions=[i], widths=0.7,
                           showmeans=False, showmedians=True)
    axes[0].scatter(range(len(groups)), observed, color='red', s=100,
                    marker='*', label='Observed', zorder=5)
    axes[0].set_title('Model A')
    axes[0].set_xlabel('Group')
    axes[0].set_ylabel('Success Rate')
    axes[0].legend()
    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels(groups)

    # Model B
    for i in range(len(groups)):
        axes[1].violinplot([p_prior_b[:, i]], positions=[i], widths=0.7,
                           showmeans=False, showmedians=True)
    axes[1].scatter(range(len(groups)), observed, color='red', s=100,
                    marker='*', label='Observed', zorder=5)
    axes[1].set_title('Model B')
    axes[1].set_xlabel('Group')
    axes[1].set_ylabel('Success Rate')
    axes[1].legend()
    axes[1].set_xticks(range(len(groups)))
    axes[1].set_xticklabels(groups)

    # Model C
    for i in range(len(groups)):
        axes[2].violinplot([p_prior_c[:, i]], positions=[i], widths=0.7,
                           showmeans=False, showmedians=True)
    axes[2].scatter(range(len(groups)), observed, color='red', s=100,
                    marker='*', label='Observed', zorder=5)
    axes[2].set_title('Model C (Mixture)')
    axes[2].set_xlabel('Group')
    axes[2].set_ylabel('Success Rate')
    axes[2].legend()
    axes[2].set_xticks(range(len(groups)))
    axes[2].set_xticklabels(groups)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'prior_predictive_vs_observed.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'prior_predictive_vs_observed.png'}")
    plt.close()

def summarize_priors(fit_a, fit_b, fit_c):
    """Print summary statistics for prior distributions"""

    print("\n" + "="*60)
    print("PRIOR SUMMARY STATISTICS")
    print("="*60)

    # Model A
    print("\nModel A (α, β parameterization):")
    alpha_a = fit_a.stan_variable('alpha')
    beta_a = fit_a.stan_variable('beta')
    mean_p_a = fit_a.stan_variable('mean_p')
    phi_a = fit_a.stan_variable('phi')

    print(f"  α: mean={alpha_a.mean():.2f}, sd={alpha_a.std():.2f}, "
          f"95% CI=[{np.percentile(alpha_a, 2.5):.2f}, {np.percentile(alpha_a, 97.5):.2f}]")
    print(f"  β: mean={beta_a.mean():.2f}, sd={beta_a.std():.2f}, "
          f"95% CI=[{np.percentile(beta_a, 2.5):.2f}, {np.percentile(beta_a, 97.5):.2f}]")
    print(f"  Mean p: {mean_p_a.mean():.3f}, 95% CI=[{np.percentile(mean_p_a, 2.5):.3f}, "
          f"{np.percentile(mean_p_a, 97.5):.3f}]")
    print(f"  φ: {phi_a.mean():.2f}, 95% CI=[{np.percentile(phi_a, 2.5):.2f}, "
          f"{np.percentile(phi_a, 97.5):.2f}]")

    # Model B
    print("\nModel B (μ, κ parameterization):")
    mu_b = fit_b.stan_variable('mu')
    kappa_b = fit_b.stan_variable('kappa')
    phi_b = fit_b.stan_variable('phi')

    print(f"  μ: mean={mu_b.mean():.3f}, sd={mu_b.std():.3f}, "
          f"95% CI=[{np.percentile(mu_b, 2.5):.3f}, {np.percentile(mu_b, 97.5):.3f}]")
    print(f"  κ: mean={kappa_b.mean():.2f}, sd={kappa_b.std():.2f}, "
          f"95% CI=[{np.percentile(kappa_b, 2.5):.2f}, {np.percentile(kappa_b, 97.5):.2f}]")
    print(f"  φ: {phi_b.mean():.2f}, 95% CI=[{np.percentile(phi_b, 2.5):.2f}, "
          f"{np.percentile(phi_b, 97.5):.2f}]")

    # Model C
    print("\nModel C (Mixture):")
    pi_c = fit_c.stan_variable('pi')
    mu1_c = fit_c.stan_variable('mu1')
    mu2_c = fit_c.stan_variable('mu2')
    mean_p_c = fit_c.stan_variable('mean_p')

    print(f"  π: mean={pi_c.mean():.3f}, 95% CI=[{np.percentile(pi_c, 2.5):.3f}, "
          f"{np.percentile(pi_c, 97.5):.3f}]")
    print(f"  μ₁: mean={mu1_c.mean():.3f}, 95% CI=[{np.percentile(mu1_c, 2.5):.3f}, "
          f"{np.percentile(mu1_c, 97.5):.3f}]")
    print(f"  μ₂: mean={mu2_c.mean():.3f}, 95% CI=[{np.percentile(mu2_c, 2.5):.3f}, "
          f"{np.percentile(mu2_c, 97.5):.3f}]")
    print(f"  Overall mean p: {mean_p_c.mean():.3f}, 95% CI=[{np.percentile(mean_p_c, 2.5):.3f}, "
          f"{np.percentile(mean_p_c, 97.5):.3f}]")

    # Comparison to observed
    print("\n" + "="*60)
    print("COMPARISON TO OBSERVED DATA")
    print("="*60)
    print(f"Observed mean success rate: 0.076")
    print(f"Observed overdispersion φ: ~3.5")
    print(f"\nPrior compatibility:")
    print(f"  Model A: P(0.05 < mean_p < 0.10) = {np.mean((mean_p_a > 0.05) & (mean_p_a < 0.10)):.2%}")
    print(f"  Model B: P(0.05 < μ < 0.10) = {np.mean((mu_b > 0.05) & (mu_b < 0.10)):.2%}")
    print(f"  Model C: P(0.05 < mean_p < 0.10) = {np.mean((mean_p_c > 0.05) & (mean_p_c < 0.10)):.2%}")
    print(f"\n  Model A: P(2 < φ < 5) = {np.mean((phi_a > 2) & (phi_a < 5)):.2%}")
    print(f"  Model B: P(2 < φ < 5) = {np.mean((phi_b > 2) & (phi_b < 5)):.2%}")

def main():
    print("="*60)
    print("PRIOR PREDICTIVE CHECKS")
    print("="*60)

    # Load data
    data = pd.read_csv(DATA_PATH)
    n_trials = data['n_trials'].values

    # Sample from priors
    fit_a = sample_prior_predictive('a', n_trials, n_draws=2000)
    fit_b = sample_prior_predictive('b', n_trials, n_draws=2000)
    fit_c = sample_prior_predictive('c', n_trials, n_draws=2000)

    # Create plots
    plot_prior_distributions(fit_a, fit_b, fit_c, data)
    plot_prior_predictive_data(fit_a, fit_b, fit_c, data)

    # Print summaries
    summarize_priors(fit_a, fit_b, fit_c)

    print(f"\n{'='*60}")
    print(f"Prior predictive checks complete!")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
