"""
Fit Change-Point Segmented Regression Model
Experiment 2: Test if change point at x≈7 is real

Model:
    Y_i ~ StudentT(ν, μ_i, σ)
    μ_i = α + β₁·x_i                  if x_i ≤ τ
    μ_i = α + β₁·τ + β₂·(x_i - τ)    if x_i > τ

Priors:
    α ~ Normal(1.8, 0.3)
    β₁ ~ Normal(0.15, 0.1)
    β₂ ~ Normal(0.02, 0.05)
    τ ~ Uniform(5, 12)
    ν ~ Gamma(2, 0.1)
    σ ~ HalfNormal(0.15)
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Set random seed
np.random.seed(42)

# Load data
data_path = Path("/workspace/data/data.csv")
df = pd.read_csv(data_path)
x_obs = df['x'].values
y_obs = df['Y'].values
N = len(x_obs)

print("="*60)
print("CHANGE-POINT SEGMENTED REGRESSION")
print("="*60)
print(f"\nData: N = {N} observations")
print(f"x range: [{x_obs.min():.1f}, {x_obs.max():.1f}]")
print(f"Y range: [{y_obs.min():.2f}, {y_obs.max():.2f}]")

# Build model
print("\n" + "="*60)
print("MODEL SPECIFICATION")
print("="*60)

with pm.Model() as changepoint_model:
    # Data
    x = pm.Data('x', x_obs)
    y = pm.Data('Y_obs', y_obs)

    # Priors
    alpha = pm.Normal('alpha', mu=1.8, sigma=0.3)
    beta_1 = pm.Normal('beta_1', mu=0.15, sigma=0.1)
    beta_2 = pm.Normal('beta_2', mu=0.02, sigma=0.05)
    tau = pm.Uniform('tau', lower=5, upper=12)
    nu = pm.Gamma('nu', alpha=2, beta=0.1)
    sigma = pm.HalfNormal('sigma', sigma=0.15)

    # Piecewise linear mean
    # Use pm.math.switch for conditional logic
    mu = pm.Deterministic('mu',
                          pm.math.switch(
                              x <= tau,
                              alpha + beta_1 * x,
                              alpha + beta_1 * tau + beta_2 * (x - tau)
                          ))

    # Likelihood
    Y = pm.StudentT('Y', nu=nu, mu=mu, sigma=sigma, observed=y)

print("\nModel built successfully")
print("Parameters: α, β₁, β₂, τ, ν, σ")
print("Likelihood: Y ~ StudentT(ν, μ, σ)")
print("Mean: μ = piecewise linear with change point at τ")

# Sample from posterior - Start conservatively
print("\n" + "="*60)
print("SAMPLING STRATEGY: CONSERVATIVE START")
print("="*60)
print("\nInitial probe: 4 chains × 500 iterations (1500 warmup)")
print("Target accept: 0.95 (change-point models need higher adapt_delta)")

start_time = time.time()

with changepoint_model:
    # Conservative sampling for change-point model
    trace = pm.sample(
        draws=500,
        tune=1500,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={
            'log_likelihood': True  # Critical for LOO-CV
        }
    )

sampling_time = time.time() - start_time
print(f"\nSampling completed in {sampling_time:.1f} seconds")
print(f"Total posterior samples: {len(trace.posterior.chain) * len(trace.posterior.draw)}")

# Check convergence
print("\n" + "="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)

summary = az.summary(trace, var_names=['alpha', 'beta_1', 'beta_2', 'tau', 'nu', 'sigma'])
print("\nParameter Summary:")
print(summary)

# Check for issues
divergences = trace.sample_stats['diverging'].values.sum()
n_samples = len(trace.posterior.chain) * len(trace.posterior.draw)
pct_divergences = 100 * divergences / n_samples

print(f"\n\nDivergent transitions: {divergences} ({pct_divergences:.2f}%)")

# Assess convergence
rhat_max = summary['r_hat'].max()
ess_min = summary['ess_bulk'].min()

print("\nConvergence Assessment:")
print(f"  Max R-hat: {rhat_max:.4f} (threshold: < 1.02)")
print(f"  Min ESS_bulk: {ess_min:.0f} (threshold: > 200 for τ, > 400 for others)")

converged = (rhat_max < 1.02) and (ess_min > 200) and (pct_divergences < 10)

if converged:
    print("\n✓ CONVERGENCE ACHIEVED")
else:
    print("\n⚠ CONVERGENCE ISSUES DETECTED")
    if rhat_max >= 1.02:
        print(f"  - R-hat too high: {rhat_max:.4f}")
    if ess_min < 200:
        print(f"  - ESS too low: {ess_min:.0f}")
    if pct_divergences >= 10:
        print(f"  - Too many divergences: {pct_divergences:.2f}%")

# Save results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

output_dir = Path("/workspace/experiments/experiment_2/posterior_inference")

# Save InferenceData with log_likelihood
idata_path = output_dir / "diagnostics" / "posterior_inference.netcdf"
trace.to_netcdf(idata_path)
print(f"\n✓ Saved InferenceData: {idata_path}")
print(f"  Size: {idata_path.stat().st_size / 1024 / 1024:.1f} MB")

# Verify log_likelihood is present
print(f"  Groups: {list(trace.groups())}")
if 'log_likelihood' in trace.groups():
    print(f"  log_likelihood shape: {trace.log_likelihood['Y'].shape}")
else:
    print("  ⚠ WARNING: log_likelihood not in InferenceData!")

# Save summary
summary_path = output_dir / "diagnostics" / "parameter_summary.csv"
summary.to_csv(summary_path)
print(f"\n✓ Saved parameter summary: {summary_path}")

# Create basic diagnostic plots
print("\n" + "="*60)
print("CREATING DIAGNOSTIC PLOTS")
print("="*60)

# Plot 1: Trace plots
fig, axes = plt.subplots(6, 2, figsize=(14, 18))
az.plot_trace(trace, var_names=['alpha', 'beta_1', 'beta_2', 'tau', 'nu', 'sigma'],
              axes=axes, compact=False)
plt.tight_layout()
trace_plot_path = output_dir / "plots" / "trace_plots.png"
plt.savefig(trace_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved trace plots: {trace_plot_path}")

# Plot 2: Rank plots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
az.plot_rank(trace, var_names=['alpha', 'beta_1', 'beta_2', 'tau', 'nu', 'sigma'],
             ax=axes)
plt.tight_layout()
rank_plot_path = output_dir / "plots" / "rank_plots.png"
plt.savefig(rank_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved rank plots: {rank_plot_path}")

# Plot 3: Posterior distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, var in enumerate(['alpha', 'beta_1', 'beta_2', 'tau', 'nu', 'sigma']):
    az.plot_posterior(trace, var_names=[var], ax=axes[i], hdi_prob=0.95)
plt.tight_layout()
posterior_plot_path = output_dir / "plots" / "posterior_distributions.png"
plt.savefig(posterior_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved posterior distributions: {posterior_plot_path}")

# Plot 4: Model fit
print("\nGenerating model fit visualization...")

# Get posterior samples for prediction
alpha_post = trace.posterior['alpha'].values.flatten()
beta_1_post = trace.posterior['beta_1'].values.flatten()
beta_2_post = trace.posterior['beta_2'].values.flatten()
tau_post = trace.posterior['tau'].values.flatten()

# Create prediction grid
x_pred = np.linspace(0.5, 35, 500)

# Compute posterior predictive for each sample
n_samples = len(alpha_post)
y_pred_samples = []

for i in range(min(500, n_samples)):  # Use 500 samples for speed
    mu_pred = np.where(
        x_pred <= tau_post[i],
        alpha_post[i] + beta_1_post[i] * x_pred,
        alpha_post[i] + beta_1_post[i] * tau_post[i] + beta_2_post[i] * (x_pred - tau_post[i])
    )
    y_pred_samples.append(mu_pred)

y_pred_samples = np.array(y_pred_samples)

# Compute percentiles
y_pred_median = np.median(y_pred_samples, axis=0)
y_pred_5 = np.percentile(y_pred_samples, 5, axis=0)
y_pred_95 = np.percentile(y_pred_samples, 95, axis=0)

fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(x_pred, y_pred_5, y_pred_95, alpha=0.3, color='steelblue', label='90% CI')
ax.plot(x_pred, y_pred_median, color='darkblue', linewidth=2, label='Posterior median')
ax.scatter(x_obs, y_obs, color='red', s=80, alpha=0.8, label='Observed data', zorder=10, edgecolors='black')

# Mark median change point
tau_median = np.median(tau_post)
tau_5 = np.percentile(tau_post, 5)
tau_95 = np.percentile(tau_post, 95)
ax.axvline(tau_median, color='orange', linestyle='--', linewidth=2,
           label=f'Change point τ = {tau_median:.1f} [{tau_5:.1f}, {tau_95:.1f}]')

ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('Y', fontsize=13)
ax.set_title('Change-Point Model Fit to Data', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fit_plot_path = output_dir / "plots" / "model_fit.png"
plt.savefig(fit_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved model fit: {fit_plot_path}")

print("\n" + "="*60)
print("FITTING COMPLETE")
print("="*60)
print(f"\nStatus: {'SUCCESS' if converged else 'NEEDS ATTENTION'}")
print(f"\nFiles generated:")
print(f"  - {idata_path}")
print(f"  - {summary_path}")
print(f"  - {trace_plot_path}")
print(f"  - {rank_plot_path}")
print(f"  - {posterior_plot_path}")
print(f"  - {fit_plot_path}")

if converged:
    print("\n✓ Model ready for LOO-CV comparison with Model 1")
else:
    print("\n⚠ Review diagnostics before proceeding to model comparison")
