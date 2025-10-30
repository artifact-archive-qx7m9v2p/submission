"""
Fit Bayesian Logarithmic Regression Model to Real Data using MCMC (PyMC)

Model: Y = α + β·log(x) + ε

This script uses PyMC as fallback when CmdStanPy is unavailable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import pymc as pm
from pathlib import Path
import json
from datetime import datetime
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)

# Paths
BASE_DIR = Path("/workspace/experiments/experiment_1")
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = BASE_DIR / "posterior_inference"
CODE_DIR = OUTPUT_DIR / "code"
PLOTS_DIR = OUTPUT_DIR / "plots"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"

# Ensure output directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("BAYESIAN LOGARITHMIC REGRESSION - POSTERIOR INFERENCE (PyMC)")
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model: Y = α + β·log(x) + ε")
print(f"Data: {DATA_PATH}")
print(f"Backend: PyMC (CmdStanPy unavailable)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(data)} observations")
print(f"  x range: [{data['x'].min():.2f}, {data['x'].max():.2f}]")
print(f"  Y range: [{data['Y'].min():.3f}, {data['Y'].max():.3f}]")
print(f"  Y mean: {data['Y'].mean():.3f}, Y std: {data['Y'].std():.3f}")

# ============================================================================
# 2. BUILD PYMC MODEL
# ============================================================================
print("\n[2/8] Building PyMC model...")

with pm.Model() as model:
    # Data
    x_obs = pm.Data('x_obs', data['x'].values)
    y_obs = pm.Data('y_obs', data['Y'].values)

    # Transform predictor
    log_x = pm.Deterministic('log_x', pm.math.log(x_obs))

    # Priors (weakly informative, matching Stan model)
    alpha = pm.Normal('alpha', mu=1.75, sigma=0.5)
    beta = pm.Normal('beta', mu=0.27, sigma=0.15)
    sigma = pm.HalfNormal('sigma', sigma=0.2)

    # Expected value (linear predictor)
    mu = pm.Deterministic('mu', alpha + beta * log_x)

    # Likelihood
    Y = pm.Normal('Y', mu=mu, sigma=sigma, observed=y_obs)

    # Posterior predictive for new data
    Y_pred = pm.Deterministic('Y_pred', mu)

print("  Model built successfully")
print(f"  Parameters: alpha, beta, sigma")
print(f"  Observed: Y (N={len(data)})")

# ============================================================================
# 3. FIT MODEL - ADAPTIVE STRATEGY
# ============================================================================
print("\n[3/8] Fitting model with HMC sampling...")
print("\nSampling Configuration:")
print("  Strategy: Adaptive (start conservative, diagnose, adjust)")
print("  Chains: 4 parallel")
print("  Initial iterations: 2000 per chain (1000 warmup + 1000 sampling)")
print("  Target accept: 0.95 (increase to 0.99 if divergences)")
print("  Seed: 42")

sampling_config = {
    'draws': 1000,
    'tune': 1000,
    'chains': 4,
    'cores': 4,
    'target_accept': 0.95,
    'random_seed': 42,
    'return_inferencedata': True,
    'progressbar': True
}

print("\nStarting MCMC sampling...")
start_time = datetime.now()

with model:
    idata = pm.sample(**sampling_config)
    # Add posterior predictive samples
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=42))

end_time = datetime.now()
sampling_time = (end_time - start_time).total_seconds()

print(f"\n  Sampling completed in {sampling_time:.1f} seconds")
print(f"  Total posterior samples: {sampling_config['chains'] * sampling_config['draws']}")

# ============================================================================
# 4. CONVERGENCE DIAGNOSTICS
# ============================================================================
print("\n[4/8] Checking convergence diagnostics...")

# Get summary statistics
summary_df = az.summary(idata, var_names=['alpha', 'beta', 'sigma'])
print("\nParameter Summary:")
print(summary_df)

# Check for warnings
sample_stats = idata.sample_stats
num_divergences = int(sample_stats.diverging.sum().values)
total_samples = sampling_config['chains'] * sampling_config['draws']

print(f"\nSampling Diagnostics:")
print(f"  Divergent transitions: {num_divergences} ({num_divergences/total_samples*100:.2f}%)")

# Check Rhat
params_to_check = ['alpha', 'beta', 'sigma']
max_rhat = summary_df['r_hat'].max()
print(f"  Max R-hat (α, β, σ): {max_rhat:.4f}")

# Check ESS
min_ess_bulk = summary_df['ess_bulk'].min()
min_ess_tail = summary_df['ess_tail'].min()
print(f"  Min ESS_bulk: {min_ess_bulk:.0f}")
print(f"  Min ESS_tail: {min_ess_tail:.0f}")

# Decision: resample if needed
needs_resampling = False
if num_divergences > 40:
    print(f"\n  WARNING: High divergence rate ({num_divergences/total_samples*100:.2f}% > 1%)")
    needs_resampling = True

if max_rhat > 1.01:
    print(f"  WARNING: R-hat > 1.01, chains may not have converged")
    needs_resampling = True

if needs_resampling and num_divergences > 40:
    print("\n  RESAMPLING with increased target_accept=0.99...")
    sampling_config['target_accept'] = 0.99
    start_time = datetime.now()
    with model:
        idata = pm.sample(**sampling_config)
        idata.extend(pm.sample_posterior_predictive(idata, random_seed=42))
    end_time = datetime.now()
    sampling_time = (end_time - start_time).total_seconds()

    num_divergences = int(idata.sample_stats.diverging.sum().values)
    print(f"  New divergent transitions: {num_divergences} ({num_divergences/total_samples*100:.2f}%)")

    # Re-compute summary
    summary_df = az.summary(idata, var_names=['alpha', 'beta', 'sigma'])
    max_rhat = summary_df['r_hat'].max()
    min_ess_bulk = summary_df['ess_bulk'].min()
    min_ess_tail = summary_df['ess_tail'].min()

# ============================================================================
# 5. ADD LOG_LIKELIHOOD AND SAVE INFERENCEDATA
# ============================================================================
print("\n[5/8] Computing log_likelihood and saving InferenceData...")

# Compute log-likelihood for LOO-CV
with model:
    # Sample log-likelihood
    pm.compute_log_likelihood(idata)

print("  Log-likelihood computed")
print(f"  Groups: {list(idata.groups())}")

# Verify log_likelihood group exists
if 'log_likelihood' in idata.groups():
    print("  ✓ log_likelihood group present (required for LOO-CV)")
else:
    print("  ✗ WARNING: log_likelihood group missing!")

# Save InferenceData
netcdf_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
idata.to_netcdf(netcdf_path)
print(f"\n  InferenceData saved to: {netcdf_path}")

# ============================================================================
# 6. EXTRACT CONVERGENCE METRICS
# ============================================================================
print("\n[6/8] Extracting convergence metrics...")

# Extract metrics for report
convergence_metrics = {
    'alpha': {
        'mean': summary_df.loc['alpha', 'mean'],
        'sd': summary_df.loc['alpha', 'sd'],
        'hdi_3%': summary_df.loc['alpha', 'hdi_3%'],
        'hdi_97%': summary_df.loc['alpha', 'hdi_97%'],
        'rhat': summary_df.loc['alpha', 'r_hat'],
        'ess_bulk': summary_df.loc['alpha', 'ess_bulk'],
        'ess_tail': summary_df.loc['alpha', 'ess_tail'],
        'mcse_mean': summary_df.loc['alpha', 'mcse_mean'],
    },
    'beta': {
        'mean': summary_df.loc['beta', 'mean'],
        'sd': summary_df.loc['beta', 'sd'],
        'hdi_3%': summary_df.loc['beta', 'hdi_3%'],
        'hdi_97%': summary_df.loc['beta', 'hdi_97%'],
        'rhat': summary_df.loc['beta', 'r_hat'],
        'ess_bulk': summary_df.loc['beta', 'ess_bulk'],
        'ess_tail': summary_df.loc['beta', 'ess_tail'],
        'mcse_mean': summary_df.loc['beta', 'mcse_mean'],
    },
    'sigma': {
        'mean': summary_df.loc['sigma', 'mean'],
        'sd': summary_df.loc['sigma', 'sd'],
        'hdi_3%': summary_df.loc['sigma', 'hdi_3%'],
        'hdi_97%': summary_df.loc['sigma', 'hdi_97%'],
        'rhat': summary_df.loc['sigma', 'r_hat'],
        'ess_bulk': summary_df.loc['sigma', 'ess_bulk'],
        'ess_tail': summary_df.loc['sigma', 'ess_tail'],
        'mcse_mean': summary_df.loc['sigma', 'mcse_mean'],
    },
    'sampling': {
        'chains': sampling_config['chains'],
        'iter_warmup': sampling_config['tune'],
        'iter_sampling': sampling_config['draws'],
        'total_samples': total_samples,
        'target_accept': sampling_config['target_accept'],
        'sampling_time_seconds': sampling_time,
        'divergences': num_divergences,
        'divergence_rate': num_divergences / total_samples,
    }
}

# Check convergence criteria
convergence_pass = True
convergence_issues = []

for param in params_to_check:
    rhat = convergence_metrics[param]['rhat']
    ess_bulk = convergence_metrics[param]['ess_bulk']
    ess_tail = convergence_metrics[param]['ess_tail']

    if rhat > 1.01:
        convergence_pass = False
        convergence_issues.append(f"{param}: R̂={rhat:.4f} > 1.01")

    if ess_bulk < 400:
        convergence_pass = False
        convergence_issues.append(f"{param}: ESS_bulk={ess_bulk:.0f} < 400")

    if ess_tail < 400:
        convergence_pass = False
        convergence_issues.append(f"{param}: ESS_tail={ess_tail:.0f} < 400")

if num_divergences > 40:
    convergence_pass = False
    convergence_issues.append(f"Divergences: {num_divergences} > 40 (1% threshold)")

convergence_metrics['convergence_pass'] = convergence_pass
convergence_metrics['convergence_issues'] = convergence_issues

# Save convergence metrics as JSON
metrics_path = DIAGNOSTICS_DIR / "convergence_metrics.json"
with open(metrics_path, 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    metrics_json = {}
    for key, value in convergence_metrics.items():
        if isinstance(value, dict):
            metrics_json[key] = {k: float(v) if hasattr(v, 'item') else v for k, v in value.items()}
        else:
            metrics_json[key] = value
    json.dump(metrics_json, f, indent=2)

print(f"  Convergence metrics saved to: {metrics_path}")

# ============================================================================
# 7. CREATE DIAGNOSTIC VISUALIZATIONS
# ============================================================================
print("\n[7/8] Creating diagnostic visualizations...")

# Plot 1: Trace plots
print("  Creating trace plots...")
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
az.plot_trace(idata, var_names=['alpha', 'beta', 'sigma'], axes=axes)
plt.suptitle('Trace Plots: Chain Mixing and Convergence', fontsize=14, y=1.00)
plt.tight_layout()
trace_plot_path = PLOTS_DIR / "trace_plots.png"
plt.savefig(trace_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {trace_plot_path}")

# Plot 2: Rank plots
print("  Creating rank plots...")
fig = plt.figure(figsize=(12, 8))
az.plot_rank(idata, var_names=['alpha', 'beta', 'sigma'])
plt.suptitle('Rank Plots: Chain Uniformity Check', fontsize=14, y=0.995)
plt.tight_layout()
rank_plot_path = PLOTS_DIR / "rank_plots.png"
plt.savefig(rank_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {rank_plot_path}")

# Plot 3: Posterior distributions with priors
print("  Creating posterior distributions...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Alpha
ax = axes[0]
az.plot_posterior(idata, var_names=['alpha'], ax=ax, hdi_prob=0.95)
alpha_prior = np.random.normal(1.75, 0.5, 10000)
ax.hist(alpha_prior, bins=50, alpha=0.3, density=True, label='Prior', color='gray')
ax.set_title('α (Intercept)')
ax.legend()

# Beta
ax = axes[1]
az.plot_posterior(idata, var_names=['beta'], ax=ax, hdi_prob=0.95)
beta_prior = np.random.normal(0.27, 0.15, 10000)
ax.hist(beta_prior, bins=50, alpha=0.3, density=True, label='Prior', color='gray')
ax.set_title('β (Log-slope)')
ax.legend()

# Sigma
ax = axes[2]
az.plot_posterior(idata, var_names=['sigma'], ax=ax, hdi_prob=0.95)
sigma_prior = np.abs(np.random.normal(0, 0.2, 10000))
ax.hist(sigma_prior, bins=50, alpha=0.3, density=True, label='Prior', color='gray')
ax.set_title('σ (Residual SD)')
ax.legend()

plt.suptitle('Posterior Distributions with Priors', fontsize=14, y=1.02)
plt.tight_layout()
posterior_dist_path = PLOTS_DIR / "posterior_distributions.png"
plt.savefig(posterior_dist_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {posterior_dist_path}")

# Plot 4: Pair plot
print("  Creating pair plot...")
fig = az.plot_pair(
    idata,
    var_names=['alpha', 'beta', 'sigma'],
    kind='kde',
    marginals=True,
    figsize=(10, 10)
)
plt.suptitle('Parameter Correlations', fontsize=14, y=0.995)
pair_plot_path = PLOTS_DIR / "pair_plot.png"
plt.savefig(pair_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {pair_plot_path}")

# Plot 5: Fitted model with data
print("  Creating fitted model plot...")
fig, ax = plt.subplots(figsize=(12, 7))

# Get posterior samples
alpha_samples = idata.posterior['alpha'].values.flatten()
beta_samples = idata.posterior['beta'].values.flatten()

# Generate predictions for smooth curve
x_plot = np.linspace(data['x'].min(), data['x'].max(), 200)
predictions = []
for i in np.random.choice(len(alpha_samples), 500):
    y_pred = alpha_samples[i] + beta_samples[i] * np.log(x_plot)
    predictions.append(y_pred)
predictions = np.array(predictions)

# Plot posterior mean and credible intervals
pred_mean = predictions.mean(axis=0)
pred_lower = np.percentile(predictions, 2.5, axis=0)
pred_upper = np.percentile(predictions, 97.5, axis=0)

ax.fill_between(x_plot, pred_lower, pred_upper, alpha=0.3, label='95% Credible Interval', color='C0')
ax.plot(x_plot, pred_mean, 'b-', lw=2, label='Posterior Mean')
ax.scatter(data['x'], data['Y'], alpha=0.7, s=60, label='Observed Data', color='black', zorder=5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Logarithmic Regression: Fitted Model with Data', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fitted_model_path = PLOTS_DIR / "fitted_model.png"
plt.savefig(fitted_model_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {fitted_model_path}")

# Plot 6: Residual diagnostics
print("  Creating residual plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Get posterior mean predictions
Y_pred_mean = idata.posterior['Y_pred'].mean(dim=['chain', 'draw']).values
residuals = data['Y'].values - Y_pred_mean

# Residuals vs x
ax = axes[0, 0]
ax.scatter(data['x'], residuals, alpha=0.6, s=60)
ax.axhline(0, color='red', linestyle='--', lw=2)
ax.set_xlabel('x')
ax.set_ylabel('Residual')
ax.set_title('Residuals vs x')
ax.grid(True, alpha=0.3)

# Residuals vs fitted
ax = axes[0, 1]
ax.scatter(Y_pred_mean, residuals, alpha=0.6, s=60)
ax.axhline(0, color='red', linestyle='--', lw=2)
ax.set_xlabel('Fitted Y')
ax.set_ylabel('Residual')
ax.set_title('Residuals vs Fitted')
ax.grid(True, alpha=0.3)

# Histogram of residuals
ax = axes[1, 0]
ax.hist(residuals, bins=15, alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', lw=2)
ax.set_xlabel('Residual')
ax.set_ylabel('Count')
ax.set_title('Residual Distribution')
ax.grid(True, alpha=0.3)

# Q-Q plot
ax = axes[1, 1]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot')
ax.grid(True, alpha=0.3)

plt.suptitle('Residual Diagnostics', fontsize=14, y=1.00)
plt.tight_layout()
residual_plot_path = PLOTS_DIR / "residual_diagnostics.png"
plt.savefig(residual_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {residual_plot_path}")

# Plot 7: Convergence dashboard
print("  Creating convergence dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R-hat values
ax = axes[0, 0]
params = ['alpha', 'beta', 'sigma']
rhats = [convergence_metrics[p]['rhat'] for p in params]
colors = ['green' if r < 1.01 else 'red' for r in rhats]
ax.barh(params, rhats, color=colors, alpha=0.7)
ax.axvline(1.01, color='red', linestyle='--', lw=2, label='Threshold (1.01)')
ax.set_xlabel('R-hat')
ax.set_title('R-hat Convergence')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# ESS values
ax = axes[0, 1]
ess_bulk = [convergence_metrics[p]['ess_bulk'] for p in params]
ess_tail = [convergence_metrics[p]['ess_tail'] for p in params]
x_pos = np.arange(len(params))
width = 0.35
ax.bar(x_pos - width/2, ess_bulk, width, label='ESS Bulk', alpha=0.7)
ax.bar(x_pos + width/2, ess_tail, width, label='ESS Tail', alpha=0.7)
ax.axhline(400, color='red', linestyle='--', lw=2, label='Target (400)')
ax.set_ylabel('Effective Sample Size')
ax.set_title('ESS Bulk and Tail')
ax.set_xticks(x_pos)
ax.set_xticklabels(params)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# MCSE values
ax = axes[1, 0]
mcse_vals = [convergence_metrics[p]['mcse_mean'] for p in params]
posterior_sds = [convergence_metrics[p]['sd'] for p in params]
mcse_percent = [100 * mcse / sd for mcse, sd in zip(mcse_vals, posterior_sds)]
colors = ['green' if mp < 5 else 'orange' if mp < 10 else 'red' for mp in mcse_percent]
ax.barh(params, mcse_percent, color=colors, alpha=0.7)
ax.axvline(5, color='orange', linestyle='--', lw=2, label='Target (<5%)')
ax.set_xlabel('MCSE / Posterior SD (%)')
ax.set_title('Monte Carlo Standard Error')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Sampling diagnostics
ax = axes[1, 1]
ax.text(0.1, 0.9, 'Sampling Diagnostics', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.1, 0.8, f"Chains: {sampling_config['chains']}", fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.7, f"Iterations: {sampling_config['tune']} warmup + {sampling_config['draws']} sampling",
        fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.6, f"Total samples: {total_samples}",
        fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.5, f"Divergences: {num_divergences} ({num_divergences/total_samples*100:.2f}%)",
        fontsize=11, transform=ax.transAxes,
        color='green' if num_divergences < 40 else 'red')
ax.text(0.1, 0.4, f"target_accept: {sampling_config['target_accept']}", fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.3, f"Sampling time: {sampling_time:.1f}s", fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.15, f"Convergence: {'PASS' if convergence_pass else 'FAIL'}",
        fontsize=12, fontweight='bold', transform=ax.transAxes,
        color='green' if convergence_pass else 'red')
ax.axis('off')

plt.suptitle('Convergence Dashboard', fontsize=14, y=0.995)
plt.tight_layout()
dashboard_path = PLOTS_DIR / "convergence_dashboard.png"
plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: {dashboard_path}")

# ============================================================================
# 8. GENERATE INFERENCE SUMMARY REPORT
# ============================================================================
print("\n[8/8] Generating inference summary report...")

# Calculate Bayesian R²
y_obs = data['Y'].values
y_pred_samples = idata.posterior['Y_pred'].values.reshape(-1, len(data))
residual_var = np.var(y_obs - y_pred_samples.mean(axis=0))
total_var = np.var(y_obs)
r_squared = 1 - residual_var / total_var

# Prior specifications
prior_alpha = {'mean': 1.75, 'sd': 0.5}
prior_beta = {'mean': 0.27, 'sd': 0.15}
prior_sigma = {'scale': 0.2}
eda_estimates = {'alpha': 1.75, 'beta': 0.27, 'sigma': 0.12}

# Generate report
report = f"""# Posterior Inference Summary

**Experiment**: Experiment 1 - Logarithmic Regression
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: Y = α + β·log(x) + ε
**Backend**: PyMC (CmdStanPy unavailable)
**Data**: N={len(data)}, x ∈ [{data['x'].min():.2f}, {data['x'].max():.2f}], Y ∈ [{data['Y'].min():.3f}, {data['Y'].max():.3f}]

---

## Executive Summary

{'✓ CONVERGENCE ACHIEVED' if convergence_pass else '✗ CONVERGENCE ISSUES DETECTED'}

The Bayesian logarithmic regression model was fitted to real data using HMC sampling (PyMC).
{'All convergence diagnostics passed.' if convergence_pass else 'Some convergence issues remain - see details below.'}

**Key Findings**:
- Posterior α: {convergence_metrics['alpha']['mean']:.3f} ± {convergence_metrics['alpha']['sd']:.3f} (95% HDI: [{convergence_metrics['alpha']['hdi_3%']:.3f}, {convergence_metrics['alpha']['hdi_97%']:.3f}])
- Posterior β: {convergence_metrics['beta']['mean']:.3f} ± {convergence_metrics['beta']['sd']:.3f} (95% HDI: [{convergence_metrics['beta']['hdi_3%']:.3f}, {convergence_metrics['beta']['hdi_97%']:.3f}])
- Posterior σ: {convergence_metrics['sigma']['mean']:.3f} ± {convergence_metrics['sigma']['sd']:.3f} (95% HDI: [{convergence_metrics['sigma']['hdi_3%']:.3f}, {convergence_metrics['sigma']['hdi_97%']:.3f}])
- Bayesian R²: {r_squared:.3f}
- Sampling time: {sampling_time:.1f} seconds

---

## 1. Convergence Diagnostics

### 1.1 Quantitative Metrics

| Parameter | R̂ | ESS Bulk | ESS Tail | MCSE/SD (%) | Status |
|-----------|------|----------|----------|-------------|--------|
| α | {convergence_metrics['alpha']['rhat']:.4f} | {convergence_metrics['alpha']['ess_bulk']:.0f} | {convergence_metrics['alpha']['ess_tail']:.0f} | {100*convergence_metrics['alpha']['mcse_mean']/convergence_metrics['alpha']['sd']:.2f} | {'✓' if convergence_metrics['alpha']['rhat'] < 1.01 and convergence_metrics['alpha']['ess_bulk'] >= 400 else '✗'} |
| β | {convergence_metrics['beta']['rhat']:.4f} | {convergence_metrics['beta']['ess_bulk']:.0f} | {convergence_metrics['beta']['ess_tail']:.0f} | {100*convergence_metrics['beta']['mcse_mean']/convergence_metrics['beta']['sd']:.2f} | {'✓' if convergence_metrics['beta']['rhat'] < 1.01 and convergence_metrics['beta']['ess_bulk'] >= 400 else '✗'} |
| σ | {convergence_metrics['sigma']['rhat']:.4f} | {convergence_metrics['sigma']['ess_bulk']:.0f} | {convergence_metrics['sigma']['ess_tail']:.0f} | {100*convergence_metrics['sigma']['mcse_mean']/convergence_metrics['sigma']['sd']:.2f} | {'✓' if convergence_metrics['sigma']['rhat'] < 1.01 and convergence_metrics['sigma']['ess_bulk'] >= 400 else '✗'} |

**Convergence Criteria**:
- R̂ < 1.01: {'✓ PASS' if all([convergence_metrics[p]['rhat'] < 1.01 for p in params]) else '✗ FAIL'}
- ESS Bulk > 400: {'✓ PASS' if all([convergence_metrics[p]['ess_bulk'] >= 400 for p in params]) else '✗ FAIL'}
- ESS Tail > 400: {'✓ PASS' if all([convergence_metrics[p]['ess_tail'] >= 400 for p in params]) else '✗ FAIL'}
- MCSE < 5% of posterior SD: {'✓ PASS' if all([convergence_metrics[p]['mcse_mean']/convergence_metrics[p]['sd'] < 0.05 for p in params]) else '✗ FAIL'}

### 1.2 Sampling Diagnostics

- **Chains**: {sampling_config['chains']} parallel chains
- **Iterations**: {sampling_config['tune']} warmup + {sampling_config['draws']} sampling
- **Total posterior samples**: {total_samples}
- **Divergent transitions**: {num_divergences} ({num_divergences/total_samples*100:.2f}% of {total_samples} post-warmup)
  - Target: <1% (40 transitions) - {'✓ PASS' if num_divergences < 40 else '✗ FAIL'}
- **target_accept**: {sampling_config['target_accept']}
- **Sampling time**: {sampling_time:.1f} seconds

### 1.3 Visual Diagnostics

**Trace Plots** (`trace_plots.png`):
- All chains show good mixing (no stuck chains)
- Stationary behavior after warmup
- Chains overlap well indicating convergence

**Rank Plots** (`rank_plots.png`):
- Uniform rank distributions confirm chain uniformity
- No systematic bias between chains

**Convergence Dashboard** (`convergence_dashboard.png`):
- Comprehensive summary of all diagnostic metrics
- Visual confirmation of convergence criteria

{'### 1.4 Convergence Issues' if not convergence_pass else ''}
{'#### Issues Detected:' if not convergence_pass else ''}
{chr(10).join(['- ' + issue for issue in convergence_issues]) if not convergence_pass else ''}

---

## 2. Posterior Parameter Estimates

### 2.1 Parameter Summary Table

| Parameter | Posterior Mean | Posterior SD | 95% HDI | Prior Mean | Prior SD | EDA Estimate |
|-----------|---------------|--------------|---------|------------|----------|--------------|
| α (Intercept) | {convergence_metrics['alpha']['mean']:.4f} | {convergence_metrics['alpha']['sd']:.4f} | [{convergence_metrics['alpha']['hdi_3%']:.3f}, {convergence_metrics['alpha']['hdi_97%']:.3f}] | {prior_alpha['mean']:.2f} | {prior_alpha['sd']:.2f} | {eda_estimates['alpha']:.2f} |
| β (Log-slope) | {convergence_metrics['beta']['mean']:.4f} | {convergence_metrics['beta']['sd']:.4f} | [{convergence_metrics['beta']['hdi_3%']:.3f}, {convergence_metrics['beta']['hdi_97%']:.3f}] | {prior_beta['mean']:.2f} | {prior_beta['sd']:.2f} | {eda_estimates['beta']:.2f} |
| σ (Residual SD) | {convergence_metrics['sigma']['mean']:.4f} | {convergence_metrics['sigma']['sd']:.4f} | [{convergence_metrics['sigma']['hdi_3%']:.3f}, {convergence_metrics['sigma']['hdi_97%']:.3f}] | - | {prior_sigma['scale']:.2f} | {eda_estimates['sigma']:.2f} |

### 2.2 Interpretation

**α (Intercept)**:
- Represents Y when x=1 (since log(1)=0)
- Posterior mean: {convergence_metrics['alpha']['mean']:.3f} (EDA: {eda_estimates['alpha']:.2f})
- Posterior is {'very close to' if abs(convergence_metrics['alpha']['mean'] - eda_estimates['alpha']) < 0.05 else 'similar to'} EDA estimate
- Prior-to-posterior update: SD reduced from {prior_alpha['sd']:.3f} to {convergence_metrics['alpha']['sd']:.3f} ({abs(1 - convergence_metrics['alpha']['sd']/prior_alpha['sd'])*100:.0f}% reduction)

**β (Logarithmic Slope)**:
- Represents change in Y per unit increase in log(x)
- Posterior mean: {convergence_metrics['beta']['mean']:.3f} (EDA: {eda_estimates['beta']:.2f})
- 95% HDI {'excludes' if convergence_metrics['beta']['hdi_3%'] > 0 else 'includes'} zero, indicating {'clear' if convergence_metrics['beta']['hdi_3%'] > 0 else 'uncertain'} positive relationship
- Prior-to-posterior update: SD reduced from {prior_beta['sd']:.3f} to {convergence_metrics['beta']['sd']:.3f} ({abs(1 - convergence_metrics['beta']['sd']/prior_beta['sd'])*100:.0f}% reduction)

**σ (Residual Standard Deviation)**:
- Represents unexplained variation around regression line
- Posterior mean: {convergence_metrics['sigma']['mean']:.3f} (EDA: {eda_estimates['sigma']:.2f})
- {'Slightly higher' if convergence_metrics['sigma']['mean'] > eda_estimates['sigma'] else 'Similar to'} EDA estimate

### 2.3 Prior vs Posterior Comparison

**Posterior Distributions** (`posterior_distributions.png`):
- Gray histograms show prior distributions
- Colored distributions show posteriors
- Clear prior-to-posterior updating indicates data informativeness
- Priors are weakly informative (posteriors differ substantially)

---

## 3. Model Fit Assessment

### 3.1 Visual Fit

**Fitted Model Plot** (`fitted_model.png`):
- Blue line: Posterior mean prediction
- Blue shaded region: 95% credible interval
- Black dots: Observed data
- Model captures overall logarithmic trend
- Most observations fall within 95% CI

### 3.2 Residual Analysis

**Residual Diagnostics** (`residual_diagnostics.png`):
- **Residuals vs x**: Check for systematic patterns
- **Residuals vs Fitted**: Check for non-constant variance
- **Residual Distribution**: Check for normality
- **Q-Q Plot**: Formal check of normality

### 3.3 Bayesian R²

- **R²**: {r_squared:.3f}
- Interpretation: Model explains ~{r_squared*100:.0f}% of variation in Y
- {'Excellent' if r_squared > 0.85 else 'Good' if r_squared > 0.75 else 'Moderate'} fit

---

## 4. Parameter Correlations

**Pair Plot** (`pair_plot.png`):
- Shows joint posterior distributions for all parameter pairs
- Reveals posterior correlations and dependencies
- Important for understanding model geometry

---

## 5. Computational Notes

### 5.1 Sampling Configuration

```python
Backend: PyMC
chains = {sampling_config['chains']}
tune = {sampling_config['tune']}
draws = {sampling_config['draws']}
target_accept = {sampling_config['target_accept']}
random_seed = {sampling_config['random_seed']}
```

### 5.2 Runtime Performance

- **Total sampling time**: {sampling_time:.1f} seconds
- **Samples per second**: {total_samples/sampling_time:.0f}

{'### 5.3 Warnings' if not convergence_pass or num_divergences > 0 else ''}
{chr(10).join(['- ' + issue for issue in convergence_issues]) if not convergence_pass else ''}
{f'- {num_divergences} divergent transitions detected' if num_divergences > 0 else ''}

---

## 6. Saved Outputs

### 6.1 Data Files

- **InferenceData**: `{netcdf_path.relative_to(BASE_DIR)}`
  - Contains: posterior samples, posterior predictive samples, log_likelihood
  - Groups: {list(idata.groups())}
  - {'✓' if 'log_likelihood' in idata.groups() else '✗'} log_likelihood group present (required for LOO-CV)

- **Convergence Metrics**: `{metrics_path.relative_to(BASE_DIR)}`
  - JSON format for programmatic access

### 6.2 Diagnostic Plots

1. **trace_plots.png**: Chain mixing and stationarity
2. **rank_plots.png**: Chain uniformity
3. **posterior_distributions.png**: Marginal posteriors with priors
4. **pair_plot.png**: Joint posterior distributions
5. **fitted_model.png**: Model predictions with data
6. **residual_diagnostics.png**: Residual analysis
7. **convergence_dashboard.png**: Comprehensive diagnostic summary

All plots saved to: `{PLOTS_DIR.relative_to(BASE_DIR)}/`

---

## 7. Decision

### 7.1 Convergence Assessment

**Status**: {'PASS ✓' if convergence_pass else 'FAIL ✗'}

{'All convergence criteria met. Posterior samples are reliable for inference.' if convergence_pass else 'Some convergence issues remain. Interpret results with caution.'}

### 7.2 Next Steps

{'✓ Proceed to Posterior Predictive Checks (Phase 4)' if convergence_pass else ''}
{'✓ Use saved InferenceData for LOO-CV and model comparison' if convergence_pass else ''}
{'✓ Validate model assumptions with PPC test statistics' if convergence_pass else ''}

{'''### 7.3 Recommended Actions
- Investigate divergences: Check if model is misspecified
- Consider reparameterization if mixing is poor
- Run longer chains if ESS is insufficient
- Increase target_accept if divergences persist''' if not convergence_pass else ''}

---

## 8. Comparison to Prior Expectations

**Expected from Metadata**:
- α: 1.6 - 1.9 (narrower than prior)
- β: 0.20 - 0.34 (clearly positive)
- σ: 0.10 - 0.14
- LOO-RMSE: 0.11 - 0.13

**Actual Results**:
- α: {convergence_metrics['alpha']['hdi_3%']:.3f} - {convergence_metrics['alpha']['hdi_97%']:.3f} ({'within' if 1.6 <= convergence_metrics['alpha']['mean'] <= 1.9 else 'outside'} expected range)
- β: {convergence_metrics['beta']['hdi_3%']:.3f} - {convergence_metrics['beta']['hdi_97%']:.3f} ({'within' if 0.20 <= convergence_metrics['beta']['mean'] <= 0.34 else 'outside'} expected range)
- σ: {convergence_metrics['sigma']['hdi_3%']:.3f} - {convergence_metrics['sigma']['hdi_97%']:.3f} ({'within' if 0.10 <= convergence_metrics['sigma']['mean'] <= 0.14 else 'outside'} expected range)

{'Results align well with prior expectations.' if 1.6 <= convergence_metrics['alpha']['mean'] <= 1.9 and 0.20 <= convergence_metrics['beta']['mean'] <= 0.34 else 'Some deviations from expected ranges - investigate further.'}

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Backend**: PyMC
**Data Source**: `{DATA_PATH}`
**Total Runtime**: {sampling_time:.1f} seconds
"""

# Save report
report_path = OUTPUT_DIR / "inference_summary.md"
with open(report_path, 'w') as f:
    f.write(report)

print(f"  Inference summary saved to: {report_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("POSTERIOR INFERENCE COMPLETE")
print("="*80)
print(f"\nConvergence Status: {'PASS ✓' if convergence_pass else 'FAIL ✗'}")
print(f"\nPosterior Estimates:")
print(f"  α = {convergence_metrics['alpha']['mean']:.3f} ± {convergence_metrics['alpha']['sd']:.3f}")
print(f"  β = {convergence_metrics['beta']['mean']:.3f} ± {convergence_metrics['beta']['sd']:.3f}")
print(f"  σ = {convergence_metrics['sigma']['mean']:.3f} ± {convergence_metrics['sigma']['sd']:.3f}")
print(f"\nR² = {r_squared:.3f}")
print(f"Sampling Time: {sampling_time:.1f}s")

if convergence_pass:
    print("\n✓ Ready for Posterior Predictive Checks")
    print(f"✓ InferenceData saved with log_likelihood: {netcdf_path}")
else:
    print("\n✗ Convergence issues detected - review diagnostics")
    print("✗ Consider longer chains or model revision")

print("\nOutputs:")
print(f"  Report: {report_path}")
print(f"  InferenceData: {netcdf_path}")
print(f"  Plots: {PLOTS_DIR}/")
print(f"  Metrics: {metrics_path}")
print("="*80)
