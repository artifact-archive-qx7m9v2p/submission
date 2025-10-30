"""
Fit Complete Pooling Model (Experiment 2) - Analytic/Sampling Solution
========================================================================

Model: y_i ~ Normal(mu, sigma_i) with common mu
This has an analytic posterior that we can sample from directly.

Posterior for mu is Normal(mu_post, sigma_post) where:
  precision_post = 1/sigma_prior^2 + sum(1/sigma_i^2)
  mu_post = (mu_prior/sigma_prior^2 + sum(y_i/sigma_i^2)) / precision_post
  sigma_post = 1/sqrt(precision_post)
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import os
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
os.makedirs('/workspace/experiments/experiment_2/posterior_inference/diagnostics', exist_ok=True)
os.makedirs('/workspace/experiments/experiment_2/posterior_inference/plots', exist_ok=True)

# Load data
data_df = pd.read_csv('/workspace/data/data.csv')
y = data_df['y'].values
sigma = data_df['sigma'].values
N = len(y)

print("="*60)
print("COMPLETE POOLING MODEL - EXPERIMENT 2")
print("="*60)
print(f"\nData Summary:")
print(f"  Studies: {N}")
print(f"  Observed y: {y}")
print(f"  Known sigma: {sigma}")

# Prior parameters
mu_prior = 0
sigma_prior = 50

print("\n" + "="*60)
print("MODEL SPECIFICATION")
print("="*60)
print("\nLikelihood: y_i ~ Normal(mu, sigma_i)")
print(f"Prior: mu ~ Normal({mu_prior}, {sigma_prior})")

# Compute analytic posterior
print("\n" + "="*60)
print("ANALYTIC POSTERIOR COMPUTATION")
print("="*60)

# Posterior parameters (conjugate normal-normal)
precision_prior = 1 / sigma_prior**2
precision_likelihood = np.sum(1 / sigma**2)
precision_post = precision_prior + precision_likelihood

sigma_post = 1 / np.sqrt(precision_post)
mu_post = (mu_prior * precision_prior + np.sum(y / sigma**2)) / precision_post

print(f"\nPosterior: mu ~ Normal({mu_post:.4f}, {sigma_post:.4f})")
print(f"\n  Posterior mean: {mu_post:.2f}")
print(f"  Posterior SD: {sigma_post:.2f}")
print(f"  95% CI: [{mu_post - 1.96*sigma_post:.2f}, {mu_post + 1.96*sigma_post:.2f}]")

# Generate MCMC-like samples from the posterior for compatibility with ArviZ
print("\n" + "="*60)
print("SAMPLING FROM POSTERIOR")
print("="*60)
print("Generating 4 chains × 1000 samples from analytic posterior")

n_chains = 4
n_samples = 1000
n_total = n_chains * n_samples

# Sample mu from posterior
mu_samples = np.random.normal(mu_post, sigma_post, size=(n_chains, n_samples))

# Generate posterior predictive samples
y_rep_samples = np.zeros((n_chains, n_samples, N))
log_lik_samples = np.zeros((n_chains, n_samples, N))

for chain in range(n_chains):
    for sample in range(n_samples):
        mu_val = mu_samples[chain, sample]
        # Posterior predictive
        y_rep_samples[chain, sample, :] = np.random.normal(mu_val, sigma)
        # Log likelihood
        log_lik_samples[chain, sample, :] = stats.norm.logpdf(y, mu_val, sigma)

print(f"\nGenerated {n_total} posterior samples")
print(f"  mu samples shape: {mu_samples.shape}")
print(f"  y_rep samples shape: {y_rep_samples.shape}")
print(f"  log_lik samples shape: {log_lik_samples.shape}")

# Create ArviZ InferenceData object
print("\n" + "="*60)
print("CREATING ARVIZ INFERENCEDATA")
print("="*60)

idata = az.from_dict(
    posterior={
        'mu': mu_samples
    },
    posterior_predictive={
        'y_rep': y_rep_samples
    },
    log_likelihood={
        'log_lik': log_lik_samples
    },
    observed_data={
        'y': y,
        'sigma': sigma
    },
    coords={
        'study': np.arange(N)
    },
    dims={
        'y_rep': ['study'],
        'log_lik': ['study']
    }
)

# Save InferenceData
idata.to_netcdf('/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf')
print("\nSaved InferenceData to: posterior_inference/diagnostics/posterior_inference.netcdf")

# Convergence diagnostics
print("\n" + "="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)

summary = az.summary(idata, var_names=['mu'], round_to=4)
print("\nPosterior Summary:")
print(summary)

# Save summary
summary.to_csv('/workspace/experiments/experiment_2/posterior_inference/diagnostics/convergence_summary.csv')

# Detailed diagnostics
print("\nDetailed Diagnostics:")
print(f"  R-hat: {summary.loc['mu', 'r_hat']:.4f}")
print(f"  ESS bulk: {summary.loc['mu', 'ess_bulk']:.0f}")
print(f"  ESS tail: {summary.loc['mu', 'ess_tail']:.0f}")
print(f"  MCSE: {summary.loc['mu', 'mcse_mean']:.4f}")
print(f"  MCSE/SD ratio: {summary.loc['mu', 'mcse_mean'] / summary.loc['mu', 'sd']:.4f}")

print("\nNote: Samples are from analytic posterior (independent draws),")
print("      so R-hat should be ~1.00 and ESS should be high.")

# Posterior results
mu_mean = summary.loc['mu', 'mean']
mu_sd = summary.loc['mu', 'sd']
mu_hdi_low = summary.loc['mu', 'hdi_3%']
mu_hdi_high = summary.loc['mu', 'hdi_97%']

print("\n" + "="*60)
print("POSTERIOR INFERENCE")
print("="*60)
print(f"\nCommon Effect (mu):")
print(f"  Posterior mean: {mu_mean:.2f}")
print(f"  Posterior SD: {mu_sd:.2f}")
print(f"  95% HDI: [{mu_hdi_low:.2f}, {mu_hdi_high:.2f}]")

# Compare with Experiment 1
print("\n" + "="*60)
print("COMPARISON WITH EXPERIMENT 1 (HIERARCHICAL)")
print("="*60)
print("\nExperiment 1 results (from context):")
print("  mu = 9.87 ± 4.89")
print("\nExperiment 2 results (complete pooling):")
print(f"  mu = {mu_mean:.2f} ± {mu_sd:.2f}")
print("\nDifference:")
print(f"  Mean shift: {mu_mean - 9.87:.2f}")
print(f"  Uncertainty reduction: {4.89 - mu_sd:.2f}")
print("\nInterpretation:")
if abs(mu_mean - 9.87) < 2:
    print("  - Posterior means are similar (< 2 units)")
else:
    print("  - Posterior means differ substantially")
if mu_sd < 4.89:
    print("  - Complete pooling has narrower uncertainty (as expected)")
else:
    print("  - Unexpected: complete pooling does not reduce uncertainty")

# Compute residuals
mu_median = np.median(mu_samples)

residuals = (y - mu_median) / sigma
print("\n" + "="*60)
print("RESIDUALS (standardized)")
print("="*60)
for i in range(N):
    status = ""
    if abs(residuals[i]) > 2:
        status = " [LARGE]"
    if abs(residuals[i]) > 3:
        status = " [EXTREME]"
    print(f"  Study {i+1}: {residuals[i]:6.2f}{status}")

print("\nResidual statistics:")
print(f"  Mean: {np.mean(residuals):.2f}")
print(f"  SD: {np.std(residuals):.2f}")
print(f"  Max |residual|: {np.max(np.abs(residuals)):.2f}")

if np.any(np.abs(residuals) > 2):
    print("\nWarning: Large residuals suggest complete pooling may be inadequate")
    print("         (studies may have different true effects)")

print("\n" + "="*60)
print("DIAGNOSTIC PLOTS")
print("="*60)

# Create diagnostic plots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Plot 1: Trace plot
ax1 = fig.add_subplot(gs[0, 0])
for chain in range(n_chains):
    ax1.plot(mu_samples[chain, :], alpha=0.7, label=f'Chain {chain+1}')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('mu')
ax1.set_title('Trace Plot', fontweight='bold')
ax1.legend()

# Plot 2: Rank plot
ax2 = fig.add_subplot(gs[0, 1])
az.plot_rank(idata, var_names=['mu'], ax=ax2)
ax2.set_title('Rank Plot', fontweight='bold')

# Plot 3: Autocorrelation (should be flat for independent samples)
ax3 = fig.add_subplot(gs[0, 2])
az.plot_autocorr(idata, var_names=['mu'], combined=True, ax=ax3, max_lag=50)
ax3.set_title('Autocorrelation', fontweight='bold')

# Plot 4: Posterior distribution
ax4 = fig.add_subplot(gs[1, 0])
az.plot_posterior(idata, var_names=['mu'], hdi_prob=0.95, ax=ax4)
ax4.set_title('Posterior Distribution', fontweight='bold')

# Plot 5: Forest plot with chains
ax5 = fig.add_subplot(gs[1, 1])
az.plot_forest(idata, var_names=['mu'], hdi_prob=0.95, combined=False, ax=ax5)
ax5.set_title('Forest Plot (by chain)', fontweight='bold')

# Plot 6: Posterior density overlay
ax6 = fig.add_subplot(gs[1, 2])
for chain in range(n_chains):
    ax6.hist(mu_samples[chain, :], bins=30, alpha=0.3, density=True, label=f'Chain {chain+1}')
# Add analytic posterior
x_range = np.linspace(mu_post - 4*sigma_post, mu_post + 4*sigma_post, 200)
ax6.plot(x_range, stats.norm.pdf(x_range, mu_post, sigma_post), 'k-', linewidth=2, label='Analytic')
ax6.set_xlabel('mu')
ax6.set_ylabel('Density')
ax6.set_title('Posterior Density (all chains)', fontweight='bold')
ax6.legend()

plt.suptitle('Convergence Diagnostics - Complete Pooling Model',
             fontsize=16, fontweight='bold', y=1.00)
plt.savefig('/workspace/experiments/experiment_2/posterior_inference/plots/convergence_diagnostics.png',
            dpi=150, bbox_inches='tight')
print("\nSaved: plots/convergence_diagnostics.png")
plt.close()

# Additional plot: posterior density with comparison to Exp 1
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Exp 2 posterior
x_range = np.linspace(mu_post - 4*sigma_post, mu_post + 4*sigma_post, 200)
exp2_density = stats.norm.pdf(x_range, mu_post, sigma_post)
ax.plot(x_range, exp2_density, 'b-', linewidth=3, label='Exp 2 (Complete Pooling)', alpha=0.8)
ax.fill_between(x_range, exp2_density, alpha=0.3)

# Add 95% HDI
hdi_low = mu_post - 1.96*sigma_post
hdi_high = mu_post + 1.96*sigma_post
ax.axvline(hdi_low, color='b', linestyle='--', alpha=0.5, linewidth=1)
ax.axvline(hdi_high, color='b', linestyle='--', alpha=0.5, linewidth=1)

# Add Exp 1 reference (approximate normal)
exp1_mu = 9.87
exp1_sd = 4.89
x_range_exp1 = np.linspace(exp1_mu - 4*exp1_sd, exp1_mu + 4*exp1_sd, 200)
exp1_density = stats.norm.pdf(x_range_exp1, exp1_mu, exp1_sd)
ax.plot(x_range_exp1, exp1_density, 'r--', linewidth=3, label='Exp 1 (Hierarchical)', alpha=0.8)
ax.fill_between(x_range_exp1, exp1_density, alpha=0.2, color='red')

# Add Exp 1 HDI
exp1_hdi_low = exp1_mu - 1.96*exp1_sd
exp1_hdi_high = exp1_mu + 1.96*exp1_sd
ax.axvline(exp1_hdi_low, color='r', linestyle=':', alpha=0.5, linewidth=1)
ax.axvline(exp1_hdi_high, color='r', linestyle=':', alpha=0.5, linewidth=1)

ax.legend(fontsize=12)
ax.set_title('Complete Pooling (Exp 2) vs Hierarchical (Exp 1)', fontsize=14, fontweight='bold')
ax.set_xlabel('mu (common effect)', fontsize=12)
ax.set_ylabel('Posterior Density', fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/posterior_inference/plots/posterior_comparison.png',
            dpi=150, bbox_inches='tight')
print("Saved: plots/posterior_comparison.png")
plt.close()

# Plot residuals
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Standardized residuals
axes[0].bar(range(1, N+1), residuals, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[0].axhline(2, color='red', linestyle='--', alpha=0.5, label='±2 SD')
axes[0].axhline(-2, color='red', linestyle='--', alpha=0.5)
axes[0].axhline(3, color='darkred', linestyle='--', alpha=0.5, label='±3 SD')
axes[0].axhline(-3, color='darkred', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Study', fontsize=12)
axes[0].set_ylabel('Standardized Residual', fontsize=12)
axes[0].set_title('Standardized Residuals', fontweight='bold', fontsize=13)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (Normal)', fontweight='bold', fontsize=13)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/posterior_inference/plots/residual_diagnostics.png',
            dpi=150, bbox_inches='tight')
print("Saved: plots/residual_diagnostics.png")
plt.close()

print("\n" + "="*60)
print("MODEL FITTING COMPLETE")
print("="*60)
print("\nNext steps:")
print("1. Review convergence diagnostics above")
print("2. Run posterior predictive checks")
print("3. Perform LOO comparison with Experiment 1")
