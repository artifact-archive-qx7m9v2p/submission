"""Create diagnostic plots for skeptical prior model"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150

# Paths
exp_dir = Path('/workspace/experiments/experiment_4/experiment_4a_skeptical/posterior_inference')
diag_dir = exp_dir / 'diagnostics'
plot_dir = exp_dir / 'plots'

# Load InferenceData
idata = az.from_netcdf(diag_dir / 'posterior_inference.netcdf')

print("Creating diagnostic plots for Model 4a (Skeptical)...")

# 1. Trace plot for mu
fig = plt.figure(figsize=(14, 5))
az.plot_trace(idata, var_names=['mu'], compact=False, figsize=(14, 5))
plt.suptitle('Model 4a (Skeptical): mu Convergence Diagnostics', y=1.02)
plt.tight_layout()
plt.savefig(plot_dir / 'trace_mu.png', dpi=150, bbox_inches='tight')
print(f"  Saved: trace_mu.png")
plt.close()

# 2. Rank plot
fig, ax = plt.subplots(figsize=(8, 5))
az.plot_rank(idata, var_names=['mu'], ax=ax)
plt.title('Model 4a (Skeptical): Rank Plot for mu')
plt.tight_layout()
plt.savefig(plot_dir / 'rank_mu.png', dpi=150, bbox_inches='tight')
print(f"  Saved: rank_mu.png")
plt.close()

# 3. Prior vs Posterior overlay
mu_samples = idata.posterior['mu'].values.flatten()
tau_samples = idata.posterior['tau'].values.flatten()

mu_mean = mu_samples.mean()
mu_sd = mu_samples.std()
mu_ci = np.percentile(mu_samples, [2.5, 97.5])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# mu
x_mu = np.linspace(-15, 25, 300)
prior_mu = stats.norm.pdf(x_mu, 0, 10)
axes[0].plot(x_mu, prior_mu, 'r--', label='Prior: N(0, 10)', linewidth=2)
axes[0].hist(mu_samples, bins=50, density=True, alpha=0.6, label='Posterior', color='steelblue')
axes[0].axvline(0, color='r', linestyle=':', alpha=0.5, label='Prior mean')
axes[0].axvline(mu_mean, color='steelblue', linestyle=':', alpha=0.8, label='Posterior mean')
axes[0].set_xlabel('mu (population mean effect)')
axes[0].set_ylabel('Density')
axes[0].set_title('Skeptical Prior vs Posterior: mu')
axes[0].legend()
axes[0].grid(alpha=0.3)

# tau
x_tau = np.linspace(0, 8, 300)
prior_tau = stats.halfnorm.pdf(x_tau, scale=5)
axes[1].plot(x_tau, prior_tau, 'r--', label='Prior: Half-Normal(0, 5)', linewidth=2)
axes[1].hist(tau_samples, bins=50, density=True, alpha=0.6, label='Posterior', color='steelblue')
axes[1].set_xlabel('tau (between-study SD)')
axes[1].set_ylabel('Density')
axes[1].set_title('Skeptical Prior vs Posterior: tau')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir / 'prior_posterior_overlay.png', dpi=150, bbox_inches='tight')
print(f"  Saved: prior_posterior_overlay.png")
plt.close()

# 4. Forest plot for theta
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_forest(idata, var_names=['theta'], combined=True, ax=ax, colors='steelblue')
ax.set_title('Study-Specific Effects (Skeptical Prior Model)')
ax.set_xlabel('Treatment Effect')
plt.tight_layout()
plt.savefig(plot_dir / 'forest_plot.png', dpi=150, bbox_inches='tight')
print(f"  Saved: forest_plot.png")
plt.close()

print("\nAll plots created successfully!")
print(f"\nSummary:")
print(f"  mu: {mu_mean:.2f} ± {mu_sd:.2f}")
print(f"  95% CI: [{mu_ci[0]:.2f}, {mu_ci[1]:.2f}]")
