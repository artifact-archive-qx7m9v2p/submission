"""
Fit Complete Pooling Model (Experiment 2) - PyMC Implementation
================================================================

Model: y_i ~ Normal(mu, sigma_i) with common mu
This tests the hypothesis that tau = 0 (complete homogeneity)
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os

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

# Build PyMC model
print("\n" + "="*60)
print("MODEL SPECIFICATION")
print("="*60)
print("\nLikelihood: y_i ~ Normal(mu, sigma_i)")
print("Prior: mu ~ Normal(0, 50)")

with pm.Model() as model:
    # Prior
    mu = pm.Normal('mu', mu=0, sigma=50)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Posterior predictive
    y_rep = pm.Normal('y_rep', mu=mu, sigma=sigma, shape=N)

print("\nModel structure:")
print(model)

# Fit model
print("\n" + "="*60)
print("MODEL FITTING")
print("="*60)
print("\nStrategy: 4 chains × 2000 iterations (1000 tune, 1000 sample)")
print("Starting MCMC sampling...")

with model:
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={'log_likelihood': True}
    )

    # Sample posterior predictive
    print("\nSampling posterior predictive...")
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=42))

print("\n" + "="*60)
print("SAMPLING COMPLETE")
print("="*60)

# Add observed data
idata.observed_data['sigma'] = ('study', sigma)

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

# Check for divergences
n_divergences = idata.sample_stats['diverging'].sum().item()
print(f"\nDivergent transitions: {n_divergences}")

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
print("  (Complete pooling typically has narrower CI)")

# Compute residuals
mu_samples = idata.posterior['mu'].values.flatten()
mu_median = np.median(mu_samples)

residuals = (y - mu_median) / sigma
print("\n" + "="*60)
print("RESIDUALS (standardized)")
print("="*60)
for i in range(N):
    print(f"  Study {i+1}: {residuals[i]:.2f}")

print("\nResidual statistics:")
print(f"  Mean: {np.mean(residuals):.2f}")
print(f"  SD: {np.std(residuals):.2f}")
print(f"  Max |residual|: {np.max(np.abs(residuals)):.2f}")

print("\n" + "="*60)
print("DIAGNOSTIC PLOTS")
print("="*60)

# Create diagnostic plots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Plot 1: Trace plot
ax1 = fig.add_subplot(gs[0, 0])
az.plot_trace(idata, var_names=['mu'], axes=[[ax1, None]], compact=True)
ax1.set_title('Trace Plot', fontweight='bold')

# Plot 2: Rank plot
ax2 = fig.add_subplot(gs[0, 1])
az.plot_rank(idata, var_names=['mu'], ax=ax2)
ax2.set_title('Rank Plot', fontweight='bold')

# Plot 3: Autocorrelation
ax3 = fig.add_subplot(gs[0, 2])
az.plot_autocorr(idata, var_names=['mu'], combined=True, ax=ax3)
ax3.set_title('Autocorrelation', fontweight='bold')

# Plot 4: Posterior distribution
ax4 = fig.add_subplot(gs[1, 0])
az.plot_posterior(idata, var_names=['mu'], hdi_prob=0.95, ax=ax4)
ax4.set_title('Posterior Distribution', fontweight='bold')

# Plot 5: Forest plot
ax5 = fig.add_subplot(gs[1, 1])
az.plot_forest(idata, var_names=['mu'], hdi_prob=0.95, combined=True, ax=ax5)
ax5.set_title('Forest Plot', fontweight='bold')

# Plot 6: Energy plot
ax6 = fig.add_subplot(gs[1, 2])
az.plot_energy(idata, ax=ax6)
ax6.set_title('Energy Plot', fontweight='bold')

plt.suptitle('Convergence Diagnostics - Complete Pooling Model',
             fontsize=16, fontweight='bold', y=1.00)
plt.savefig('/workspace/experiments/experiment_2/posterior_inference/plots/convergence_diagnostics.png',
            dpi=150, bbox_inches='tight')
print("\nSaved: plots/convergence_diagnostics.png")
plt.close()

# Additional plot: posterior density with comparison to Exp 1
fig, ax = plt.subplots(figsize=(10, 6))

# Plot posterior
az.plot_posterior(idata, var_names=['mu'], hdi_prob=0.95, ax=ax, color='steelblue')

# Add Exp 1 reference (approximate normal)
exp1_mu = 9.87
exp1_sd = 4.89
x_range = np.linspace(exp1_mu - 3*exp1_sd, exp1_mu + 3*exp1_sd, 100)
exp1_density = 1/(exp1_sd * np.sqrt(2*np.pi)) * np.exp(-0.5*((x_range - exp1_mu)/exp1_sd)**2)

# Normalize to match scale
ylim = ax.get_ylim()
exp1_density_scaled = exp1_density / exp1_density.max() * ylim[1] * 0.8

ax.plot(x_range, exp1_density_scaled,
        'r--', alpha=0.5, linewidth=2, label='Exp 1 (hierarchical)')

ax.legend()
ax.set_title('Complete Pooling (Exp 2) vs Hierarchical (Exp 1)', fontsize=14, fontweight='bold')
ax.set_xlabel('mu (common effect)', fontsize=12)

plt.tight_layout()
plt.savefig('/workspace/experiments/experiment_2/posterior_inference/plots/posterior_comparison.png',
            dpi=150, bbox_inches='tight')
print("Saved: plots/posterior_comparison.png")
plt.close()

print("\n" + "="*60)
print("MODEL FITTING COMPLETE")
print("="*60)
print("\nNext steps:")
print("1. Review convergence diagnostics above")
print("2. Run posterior predictive checks")
print("3. Perform LOO comparison with Experiment 1")
