"""
Create diagnostic plots for Bayesian Log-Log Linear Model posterior inference
"""

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path

# Paths
DATA_PATH = "/workspace/data/data.csv"
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/posterior_inference")
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Load data
df = pd.read_csv(DATA_PATH)
log_Y = np.log(df['Y'].values)
log_x = np.log(df['x'].values)

# Load InferenceData
print("Loading InferenceData...")
idata = az.from_netcdf(DIAGNOSTICS_DIR / "posterior_inference.netcdf")

print(f"InferenceData groups: {list(idata.groups())}")
print(f"Posterior shape: {idata.posterior.dims}")

# 1. Trace plots for convergence assessment
print("\n1. Creating trace plots...")
axes = az.plot_trace(
    idata,
    var_names=["alpha", "beta", "sigma"],
    compact=True,
    figsize=(12, 8)
)
plt.suptitle("Trace Plots: Convergence Diagnostics", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "trace_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'trace_plots.png'}")

# 2. Rank plots for chain mixing
print("2. Creating rank plots...")
az.plot_rank(
    idata,
    var_names=["alpha", "beta", "sigma"],
    figsize=(12, 4)
)
plt.suptitle("Rank Plots: Chain Mixing Assessment", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'rank_plots.png'}")

# 3. Posterior distributions vs priors
print("3. Creating posterior vs prior plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Extract posterior samples
alpha_post = idata.posterior.alpha.values.flatten()
beta_post = idata.posterior.beta.values.flatten()
sigma_post = idata.posterior.sigma.values.flatten()

# Alpha
axes[0].hist(alpha_post, bins=50, density=True, alpha=0.6, label='Posterior', color='steelblue')
alpha_prior = np.linspace(alpha_post.min()-0.1, alpha_post.max()+0.1, 200)
axes[0].plot(alpha_prior, np.exp(-0.5*((alpha_prior-0.6)/0.3)**2)/(0.3*np.sqrt(2*np.pi)),
             'r--', lw=2, label='Prior N(0.6, 0.3)')
axes[0].set_xlabel('alpha (log-scale intercept)')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_title('Alpha: Posterior vs Prior')
axes[0].grid(alpha=0.3)

# Beta
axes[1].hist(beta_post, bins=50, density=True, alpha=0.6, label='Posterior', color='steelblue')
beta_prior = np.linspace(beta_post.min()-0.02, beta_post.max()+0.02, 200)
axes[1].plot(beta_prior, np.exp(-0.5*((beta_prior-0.13)/0.1)**2)/(0.1*np.sqrt(2*np.pi)),
             'r--', lw=2, label='Prior N(0.13, 0.1)')
axes[1].set_xlabel('beta (power law exponent)')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].set_title('Beta: Posterior vs Prior')
axes[1].grid(alpha=0.3)

# Sigma
axes[2].hist(sigma_post, bins=50, density=True, alpha=0.6, label='Posterior', color='steelblue')
sigma_prior = np.linspace(0, sigma_post.max()+0.02, 200)
# Half-normal prior
axes[2].plot(sigma_prior, 2*np.exp(-0.5*(sigma_prior/0.1)**2)/(0.1*np.sqrt(2*np.pi)),
             'r--', lw=2, label='Prior HalfN(0.1)')
axes[2].set_xlabel('sigma (log-scale residual SD)')
axes[2].set_ylabel('Density')
axes[2].legend()
axes[2].set_title('Sigma: Posterior vs Prior')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_vs_prior.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'posterior_vs_prior.png'}")

# 4. Pairs plot (parameter correlations)
print("4. Creating pairs plot...")
az.plot_pair(
    idata,
    var_names=["alpha", "beta", "sigma"],
    kind='hexbin',
    marginals=True,
    figsize=(10, 10)
)
plt.suptitle("Pairs Plot: Parameter Correlations", y=0.995, fontsize=14)
plt.savefig(PLOTS_DIR / "pairs_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'pairs_plot.png'}")

# 5. Fitted line with credible intervals
print("5. Creating fitted line plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Get posterior samples for predictions
alpha_samples = idata.posterior.alpha.values.reshape(-1)
beta_samples = idata.posterior.beta.values.reshape(-1)

# Create grid for predictions
x_grid = np.linspace(df['x'].min(), df['x'].max(), 100)
log_x_grid = np.log(x_grid)

# Compute predictions for each posterior sample (subsample for efficiency)
n_samples = 1000
idx = np.random.choice(len(alpha_samples), n_samples, replace=False)
log_y_preds = alpha_samples[idx, np.newaxis] + beta_samples[idx, np.newaxis] * log_x_grid

# Compute quantiles
log_y_mean = np.mean(log_y_preds, axis=0)
log_y_lower = np.percentile(log_y_preds, 2.5, axis=0)
log_y_upper = np.percentile(log_y_preds, 97.5, axis=0)

# Plot in log scale
axes[0].scatter(log_x, log_Y, alpha=0.6, s=50, label='Observed data', zorder=3)
axes[0].plot(log_x_grid, log_y_mean, 'r-', lw=2, label='Posterior mean', zorder=2)
axes[0].fill_between(log_x_grid, log_y_lower, log_y_upper, alpha=0.3, color='red', label='95% CI', zorder=1)
axes[0].set_xlabel('log(x)')
axes[0].set_ylabel('log(Y)')
axes[0].set_title('Log-Log Scale')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot in original scale
y_preds = np.exp(log_y_preds)
y_mean = np.mean(y_preds, axis=0)
y_lower = np.percentile(y_preds, 2.5, axis=0)
y_upper = np.percentile(y_preds, 97.5, axis=0)

axes[1].scatter(df['x'], df['Y'], alpha=0.6, s=50, label='Observed data', zorder=3)
axes[1].plot(x_grid, y_mean, 'r-', lw=2, label='Posterior mean', zorder=2)
axes[1].fill_between(x_grid, y_lower, y_upper, alpha=0.3, color='red', label='95% CI', zorder=1)
axes[1].set_xlabel('x')
axes[1].set_ylabel('Y')
axes[1].set_title('Original Scale')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("Model Fit: Observations vs Posterior Predictions", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fitted_line.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'fitted_line.png'}")

# 6. Residual plots
print("6. Creating residual plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Compute residuals using posterior mean
y_obs_pred = idata.posterior_predictive.y_obs.mean(dim=["chain", "draw"]).values
residuals = log_Y - y_obs_pred

# Residuals vs fitted
axes[0].scatter(y_obs_pred, residuals, alpha=0.6, s=50)
axes[0].axhline(0, color='red', linestyle='--', lw=2)
axes[0].set_xlabel('Fitted values (log scale)')
axes[0].set_ylabel('Residuals (log scale)')
axes[0].set_title('Residuals vs Fitted')
axes[0].grid(alpha=0.3)

# Residuals vs predictor
axes[1].scatter(log_x, residuals, alpha=0.6, s=50)
axes[1].axhline(0, color='red', linestyle='--', lw=2)
axes[1].set_xlabel('log(x)')
axes[1].set_ylabel('Residuals (log scale)')
axes[1].set_title('Residuals vs Predictor')
axes[1].grid(alpha=0.3)

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title('Q-Q Plot')
axes[2].grid(alpha=0.3)

plt.suptitle("Residual Diagnostics (Log Scale)", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "residual_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'residual_plots.png'}")

# 7. LOO-PIT plot (calibration check)
print("7. Creating LOO-PIT plot...")
az.plot_loo_pit(idata, y="y_obs", figsize=(10, 4))
plt.suptitle("LOO-PIT: Posterior Predictive Calibration", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "loo_pit.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'loo_pit.png'}")

# 8. Forest plot (parameter estimates)
print("8. Creating forest plot...")
az.plot_forest(
    idata,
    var_names=["alpha", "beta", "sigma"],
    combined=True,
    figsize=(8, 4),
    r_hat=True,
    ess=True
)
plt.suptitle("Parameter Estimates with Convergence Diagnostics", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "forest_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'forest_plot.png'}")

# 9. Energy plot (HMC diagnostic)
print("9. Creating energy plot...")
az.plot_energy(idata, figsize=(8, 5))
plt.suptitle("Energy Plot: HMC Transition Diagnostic", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "energy_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {PLOTS_DIR / 'energy_plot.png'}")

print("\nAll diagnostic plots created successfully!")
print(f"Location: {PLOTS_DIR}")
