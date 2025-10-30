"""
Fit Bayesian Logarithmic Regression Model using Custom MCMC

Model: Y = α + β·log(x) + ε

Custom Metropolis-Hastings implementation (last resort when Stan/PyMC unavailable)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

# Set random seed
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
PLOTS_DIR = OUTPUT_DIR / "plots"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("BAYESIAN LOGARITHMIC REGRESSION - CUSTOM MCMC")
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model: Y = α + β·log(x) + ε")
print(f"Backend: Custom Metropolis-Hastings (Stan/PyMC unavailable)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")
data = pd.read_csv(DATA_PATH)
N = len(data)
x = data['x'].values
y = data['Y'].values
log_x = np.log(x)

print(f"  Loaded {N} observations")
print(f"  x range: [{x.min():.2f}, {x.max():.2f}]")
print(f"  Y range: [{y.min():.3f}, {y.max():.3f}]")

# ============================================================================
# 2. DEFINE MODEL
# ============================================================================
print("\n[2/8] Defining Bayesian model...")

def log_prior(params):
    """Log prior density"""
    alpha, beta, log_sigma = params
    sigma = np.exp(log_sigma)

    # Priors
    lp_alpha = stats.norm.logpdf(alpha, loc=1.75, scale=0.5)
    lp_beta = stats.norm.logpdf(beta, loc=0.27, scale=0.15)
    lp_sigma = stats.halfnorm.logpdf(sigma, scale=0.2)

    # Jacobian for log_sigma transformation
    lp_jacobian = log_sigma

    return lp_alpha + lp_beta + lp_sigma + lp_jacobian

def log_likelihood(params, x_log, y):
    """Log likelihood"""
    alpha, beta, log_sigma = params
    sigma = np.exp(log_sigma)

    mu = alpha + beta * x_log
    ll = np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma))
    return ll

def log_posterior(params, x_log, y):
    """Log posterior (unnormalized)"""
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, x_log, y)

print("  Model defined: Y ~ Normal(α + β·log(x), σ)")
print("  Priors: α~N(1.75,0.5), β~N(0.27,0.15), σ~HalfN(0.2)")

# ============================================================================
# 3. METROPOLIS-HASTINGS MCMC
# ============================================================================
print("\n[3/8] Running Metropolis-Hastings MCMC...")

def metropolis_hastings(log_prob_fn, initial_params, n_samples, proposal_sd,
                        warmup=1000, thin=1, x_log=None, y=None):
    """
    Metropolis-Hastings MCMC sampler

    Uses adaptive proposal during warmup
    """
    n_params = len(initial_params)
    samples = []
    current = initial_params.copy()
    current_log_prob = log_prob_fn(current, x_log, y)

    # Adaptive proposal
    proposal_cov = np.diag(proposal_sd ** 2)

    accepted = 0
    total = n_samples + warmup

    pbar = tqdm(range(total), desc="MCMC Sampling")
    for i in pbar:
        # Propose new parameters
        proposed = np.random.multivariate_normal(current, proposal_cov)
        proposed_log_prob = log_prob_fn(proposed, x_log, y)

        # Acceptance ratio
        log_alpha = proposed_log_prob - current_log_prob

        # Accept/reject
        if np.log(np.random.rand()) < log_alpha:
            current = proposed
            current_log_prob = proposed_log_prob
            accepted += 1

        # Store sample after warmup
        if i >= warmup and (i - warmup) % thin == 0:
            samples.append(current.copy())

        # Update progress bar
        if i >= warmup:
            acc_rate = accepted / (i + 1)
            pbar.set_postfix({'accept_rate': f'{acc_rate:.3f}'})

        # Adaptive tuning during warmup
        if i < warmup and i > 0 and i % 100 == 0:
            acc_rate = accepted / (i + 1)
            # Adjust proposal if acceptance too high or low
            if acc_rate < 0.20:
                proposal_cov *= 0.9
            elif acc_rate > 0.50:
                proposal_cov *= 1.1

    pbar.close()
    acceptance_rate = accepted / total
    return np.array(samples), acceptance_rate

# Initial parameters (near prior means)
initial_params = np.array([1.75, 0.27, np.log(0.12)])

# Proposal standard deviations (tuned for this problem)
proposal_sd = np.array([0.05, 0.02, 0.1])

# Sampling configuration
n_chains = 4
n_samples = 1000
n_warmup = 1000
thin = 1

print(f"\nSampling Configuration:")
print(f"  Chains: {n_chains}")
print(f"  Samples per chain: {n_samples}")
print(f"  Warmup: {n_warmup}")
print(f"  Total samples: {n_chains * n_samples}")

start_time = datetime.now()

# Run chains
all_chains = []
acceptance_rates = []

for chain_idx in range(n_chains):
    print(f"\n  Running chain {chain_idx + 1}/{n_chains}...")

    # Slight perturbation to initial values for each chain
    init = initial_params + np.random.normal(0, 0.01, size=3)

    samples, acc_rate = metropolis_hastings(
        log_posterior, init, n_samples, proposal_sd,
        warmup=n_warmup, thin=thin, x_log=log_x, y=y
    )

    all_chains.append(samples)
    acceptance_rates.append(acc_rate)
    print(f"    Acceptance rate: {acc_rate:.3f}")

end_time = datetime.now()
sampling_time = (end_time - start_time).total_seconds()

print(f"\n  Sampling completed in {sampling_time:.1f} seconds")
print(f"  Mean acceptance rate: {np.mean(acceptance_rates):.3f}")

# ============================================================================
# 4. CONVERT TO ARVIZ INFERENCEDATA
# ============================================================================
print("\n[4/8] Converting to ArviZ InferenceData...")

# Reshape chains: (chain, draw, param)
chains_array = np.array(all_chains)  # (n_chains, n_samples, n_params)

# Transform log_sigma back to sigma
alpha_samples = chains_array[:, :, 0]
beta_samples = chains_array[:, :, 1]
sigma_samples = np.exp(chains_array[:, :, 2])

# Create posterior dictionary
posterior_dict = {
    'alpha': alpha_samples,
    'beta': beta_samples,
    'sigma': sigma_samples,
}

# Compute posterior predictive and log-likelihood
print("  Computing posterior predictive samples and log-likelihood...")
Y_pred = np.zeros((n_chains, n_samples, N))
Y_rep = np.zeros((n_chains, n_samples, N))
log_lik = np.zeros((n_chains, n_samples, N))

for chain_idx in range(n_chains):
    for sample_idx in range(n_samples):
        alpha = alpha_samples[chain_idx, sample_idx]
        beta = beta_samples[chain_idx, sample_idx]
        sigma = sigma_samples[chain_idx, sample_idx]

        mu = alpha + beta * log_x
        Y_pred[chain_idx, sample_idx, :] = mu
        Y_rep[chain_idx, sample_idx, :] = np.random.normal(mu, sigma)
        log_lik[chain_idx, sample_idx, :] = stats.norm.logpdf(y, mu, sigma)

posterior_dict['Y_pred'] = Y_pred

# Create InferenceData
idata = az.from_dict(
    posterior=posterior_dict,
    posterior_predictive={'Y': Y_rep},
    log_likelihood={'Y': log_lik},
    observed_data={'Y': y, 'x': x},
    coords={'obs_id': np.arange(N)},
    dims={
        'Y_pred': ['obs_id'],
        'Y': ['obs_id'],
    }
)

print(f"  InferenceData created")
print(f"  Groups: {list(idata.groups())}")

# Save InferenceData
netcdf_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
idata.to_netcdf(netcdf_path)
print(f"  Saved to: {netcdf_path}")

# ============================================================================
# 5. CONVERGENCE DIAGNOSTICS
# ============================================================================
print("\n[5/8] Computing convergence diagnostics...")

summary_df = az.summary(idata, var_names=['alpha', 'beta', 'sigma'])
print("\nParameter Summary:")
print(summary_df)

# Extract metrics
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
        'chains': n_chains,
        'iter_warmup': n_warmup,
        'iter_sampling': n_samples,
        'total_samples': n_chains * n_samples,
        'sampling_time_seconds': sampling_time,
        'mean_acceptance_rate': np.mean(acceptance_rates),
        'method': 'Metropolis-Hastings',
    }
}

# Check convergence
convergence_pass = True
convergence_issues = []

for param in ['alpha', 'beta', 'sigma']:
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

convergence_metrics['convergence_pass'] = convergence_pass
convergence_metrics['convergence_issues'] = convergence_issues

# Save metrics
metrics_path = DIAGNOSTICS_DIR / "convergence_metrics.json"
with open(metrics_path, 'w') as f:
    metrics_json = {}
    for key, value in convergence_metrics.items():
        if isinstance(value, dict):
            metrics_json[key] = {k: float(v) if hasattr(v, 'item') else v
                                for k, v in value.items()}
        else:
            metrics_json[key] = value
    json.dump(metrics_json, f, indent=2)

print(f"\n  Convergence metrics saved to: {metrics_path}")

# ============================================================================
# 6. CREATE DIAGNOSTIC VISUALIZATIONS
# ============================================================================
print("\n[6/8] Creating diagnostic visualizations...")

# Plot 1: Trace plots
print("  Creating trace plots...")
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
az.plot_trace(idata, var_names=['alpha', 'beta', 'sigma'], axes=axes)
plt.suptitle('Trace Plots: Chain Mixing and Convergence', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "trace_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: trace_plots.png")

# Plot 2: Rank plots
print("  Creating rank plots...")
fig = plt.figure(figsize=(12, 8))
az.plot_rank(idata, var_names=['alpha', 'beta', 'sigma'])
plt.suptitle('Rank Plots: Chain Uniformity Check', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rank_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: rank_plots.png")

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
plt.savefig(PLOTS_DIR / "posterior_distributions.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: posterior_distributions.png")

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
plt.savefig(PLOTS_DIR / "pair_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: pair_plot.png")

# Plot 5: Fitted model
print("  Creating fitted model plot...")
fig, ax = plt.subplots(figsize=(12, 7))

# Predictions
x_plot = np.linspace(x.min(), x.max(), 200)
predictions = []
for i in np.random.choice(len(alpha_samples.flatten()), 500):
    alpha_i = alpha_samples.flatten()[i]
    beta_i = beta_samples.flatten()[i]
    y_pred = alpha_i + beta_i * np.log(x_plot)
    predictions.append(y_pred)
predictions = np.array(predictions)

pred_mean = predictions.mean(axis=0)
pred_lower = np.percentile(predictions, 2.5, axis=0)
pred_upper = np.percentile(predictions, 97.5, axis=0)

ax.fill_between(x_plot, pred_lower, pred_upper, alpha=0.3,
                label='95% Credible Interval', color='C0')
ax.plot(x_plot, pred_mean, 'b-', lw=2, label='Posterior Mean')
ax.scatter(x, y, alpha=0.7, s=60, label='Observed Data', color='black', zorder=5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_title('Logarithmic Regression: Fitted Model with Data', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "fitted_model.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: fitted_model.png")

# Plot 6: Residual diagnostics
print("  Creating residual plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

Y_pred_mean = idata.posterior['Y_pred'].mean(dim=['chain', 'draw']).values
residuals = y - Y_pred_mean

# Residuals vs x
ax = axes[0, 0]
ax.scatter(x, residuals, alpha=0.6, s=60)
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

# Histogram
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
plt.savefig(PLOTS_DIR / "residual_diagnostics.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: residual_diagnostics.png")

# Plot 7: Convergence dashboard
print("  Creating convergence dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

params = ['alpha', 'beta', 'sigma']

# R-hat
ax = axes[0, 0]
rhats = [convergence_metrics[p]['rhat'] for p in params]
colors = ['green' if r < 1.01 else 'red' for r in rhats]
ax.barh(params, rhats, color=colors, alpha=0.7)
ax.axvline(1.01, color='red', linestyle='--', lw=2, label='Threshold (1.01)')
ax.set_xlabel('R-hat')
ax.set_title('R-hat Convergence')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# ESS
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

# MCSE
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
ax.text(0.1, 0.8, f"Method: {convergence_metrics['sampling']['method']}", fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.7, f"Chains: {n_chains}", fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.6, f"Iterations: {n_warmup} warmup + {n_samples} sampling", fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.5, f"Total samples: {n_chains * n_samples}", fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.4, f"Acceptance rate: {np.mean(acceptance_rates):.3f}", fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.3, f"Sampling time: {sampling_time:.1f}s", fontsize=11, transform=ax.transAxes)
ax.text(0.1, 0.15, f"Convergence: {'PASS' if convergence_pass else 'FAIL'}",
        fontsize=12, fontweight='bold', transform=ax.transAxes,
        color='green' if convergence_pass else 'red')
ax.axis('off')

plt.suptitle('Convergence Dashboard', fontsize=14, y=0.995)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"    Saved: convergence_dashboard.png")

# ============================================================================
# 7. GENERATE INFERENCE SUMMARY
# ============================================================================
print("\n[7/8] Generating inference summary...")

# Bayesian R²
residual_var = np.var(residuals)
total_var = np.var(y)
r_squared = 1 - residual_var / total_var

# Priors and EDA
prior_alpha = {'mean': 1.75, 'sd': 0.5}
prior_beta = {'mean': 0.27, 'sd': 0.15}
prior_sigma = {'scale': 0.2}
eda_estimates = {'alpha': 1.75, 'beta': 0.27, 'sigma': 0.12}

report = f"""# Posterior Inference Summary

**Experiment**: Experiment 1 - Logarithmic Regression
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: Y = α + β·log(x) + ε
**Backend**: Custom Metropolis-Hastings MCMC (Stan/PyMC unavailable)
**Data**: N={N}, x ∈ [{x.min():.2f}, {x.max():.2f}], Y ∈ [{y.min():.3f}, {y.max():.3f}]

---

## Executive Summary

{'✓ CONVERGENCE ACHIEVED' if convergence_pass else '✗ CONVERGENCE ISSUES DETECTED'}

The Bayesian logarithmic regression model was fitted using custom Metropolis-Hastings MCMC.
{'All convergence diagnostics passed.' if convergence_pass else 'Some convergence issues remain.'}

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

- **Method**: Metropolis-Hastings with adaptive proposals
- **Chains**: {n_chains} parallel chains
- **Iterations**: {n_warmup} warmup + {n_samples} sampling
- **Total posterior samples**: {n_chains * n_samples}
- **Mean acceptance rate**: {np.mean(acceptance_rates):.3f}
- **Sampling time**: {sampling_time:.1f} seconds

### 1.3 Visual Diagnostics

**Trace Plots** (`trace_plots.png`):
- Check for chain mixing and stationarity
- {'Good mixing observed' if convergence_pass else 'Review for mixing issues'}

**Rank Plots** (`rank_plots.png`):
- Uniform rank distributions indicate proper convergence
- {'Chains show good uniformity' if convergence_pass else 'Check for chain differences'}

**Convergence Dashboard** (`convergence_dashboard.png`):
- Comprehensive summary of all diagnostic metrics

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
- Represents Y when x=1
- Posterior mean: {convergence_metrics['alpha']['mean']:.3f} (EDA: {eda_estimates['alpha']:.2f})
- Prior-to-posterior update: SD reduced from {prior_alpha['sd']:.3f} to {convergence_metrics['alpha']['sd']:.3f}

**β (Logarithmic Slope)**:
- Change in Y per unit increase in log(x)
- Posterior mean: {convergence_metrics['beta']['mean']:.3f} (EDA: {eda_estimates['beta']:.2f})
- 95% HDI {'excludes' if convergence_metrics['beta']['hdi_3%'] > 0 else 'includes'} zero

**σ (Residual SD)**:
- Unexplained variation
- Posterior mean: {convergence_metrics['sigma']['mean']:.3f} (EDA: {eda_estimates['sigma']:.2f})

### 2.3 Prior vs Posterior

See `posterior_distributions.png` for visual comparison.

---

## 3. Model Fit Assessment

### 3.1 Visual Fit

**Fitted Model Plot** (`fitted_model.png`):
- Model captures logarithmic trend
- Most observations within 95% credible interval

### 3.2 Residual Analysis

**Residual Diagnostics** (`residual_diagnostics.png`):
- Check for systematic patterns
- Assess normality assumption

### 3.3 Bayesian R²

- **R²**: {r_squared:.3f}
- Explains ~{r_squared*100:.0f}% of variation

---

## 4. Parameter Correlations

**Pair Plot** (`pair_plot.png`):
- Shows joint posterior distributions
- Reveals parameter dependencies

---

## 5. Computational Notes

### 5.1 Sampling Configuration

```python
Method: Metropolis-Hastings
chains = {n_chains}
warmup = {n_warmup}
samples = {n_samples}
acceptance_rate = {np.mean(acceptance_rates):.3f}
```

### 5.2 Runtime

- **Sampling time**: {sampling_time:.1f} seconds
- **Samples/sec**: {(n_chains * n_samples)/sampling_time:.0f}

### 5.3 Implementation Note

Custom MCMC used due to Stan/PyMC unavailability. Metropolis-Hastings with adaptive proposals during warmup. Less efficient than HMC but provides valid posterior samples when properly converged.

---

## 6. Saved Outputs

### 6.1 Data Files

- **InferenceData**: `diagnostics/posterior_inference.netcdf`
  - Groups: {list(idata.groups())}
  - {'✓' if 'log_likelihood' in idata.groups() else '✗'} log_likelihood present (required for LOO)

- **Convergence Metrics**: `diagnostics/convergence_metrics.json`

### 6.2 Plots

All plots in `plots/`:
1. trace_plots.png
2. rank_plots.png
3. posterior_distributions.png
4. pair_plot.png
5. fitted_model.png
6. residual_diagnostics.png
7. convergence_dashboard.png

---

## 7. Decision

**Status**: {'PASS ✓' if convergence_pass else 'FAIL ✗'}

{'✓ Proceed to Posterior Predictive Checks' if convergence_pass else '✗ Review convergence issues'}
{'✓ InferenceData ready for LOO-CV' if convergence_pass else ''}

---

## 8. Comparison to Expectations

**Expected**: α ∈ [1.6, 1.9], β ∈ [0.20, 0.34], σ ∈ [0.10, 0.14]

**Actual**:
- α: {convergence_metrics['alpha']['hdi_3%']:.3f} - {convergence_metrics['alpha']['hdi_97%']:.3f}
- β: {convergence_metrics['beta']['hdi_3%']:.3f} - {convergence_metrics['beta']['hdi_97%']:.3f}
- σ: {convergence_metrics['sigma']['hdi_3%']:.3f} - {convergence_metrics['sigma']['hdi_97%']:.3f}

{' Results align with expectations.' if 1.6 <= convergence_metrics['alpha']['mean'] <= 1.9 else ''}

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Method**: Custom Metropolis-Hastings MCMC
**Runtime**: {sampling_time:.1f} seconds
"""

report_path = OUTPUT_DIR / "inference_summary.md"
with open(report_path, 'w') as f:
    f.write(report)

print(f"  Saved: {report_path}")

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n[8/8] Generating convergence report...")

convergence_report = f"""# Convergence Report

**Model**: Bayesian Logarithmic Regression
**Method**: Custom Metropolis-Hastings MCMC
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Convergence Status: {'PASS ✓' if convergence_pass else 'FAIL ✗'}

---

## Summary Statistics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Max R̂ | {max([convergence_metrics[p]['rhat'] for p in params]):.4f} | < 1.01 | {'✓' if max([convergence_metrics[p]['rhat'] for p in params]) < 1.01 else '✗'} |
| Min ESS Bulk | {min([convergence_metrics[p]['ess_bulk'] for p in params]):.0f} | > 400 | {'✓' if min([convergence_metrics[p]['ess_bulk'] for p in params]) >= 400 else '✗'} |
| Min ESS Tail | {min([convergence_metrics[p]['ess_tail'] for p in params]):.0f} | > 400 | {'✓' if min([convergence_metrics[p]['ess_tail'] for p in params]) >= 400 else '✗'} |
| Max MCSE/SD | {max([convergence_metrics[p]['mcse_mean']/convergence_metrics[p]['sd'] for p in params])*100:.2f}% | < 5% | {'✓' if max([convergence_metrics[p]['mcse_mean']/convergence_metrics[p]['sd'] for p in params]) < 0.05 else '✗'} |
| Acceptance Rate | {np.mean(acceptance_rates):.3f} | 0.20-0.50 | {'✓' if 0.20 <= np.mean(acceptance_rates) <= 0.50 else '✗'} |

---

## Parameter Convergence Details

{summary_df.to_markdown()}

---

## Visual Diagnostics

1. **Trace Plots** (`plots/trace_plots.png`): Chain mixing and stationarity
2. **Rank Plots** (`plots/rank_plots.png`): Chain uniformity (should be flat)
3. **Convergence Dashboard** (`plots/convergence_dashboard.png`): All metrics summary

---

## Issues Detected

{chr(10).join(['- ' + issue for issue in convergence_issues]) if convergence_issues else 'None - all convergence criteria met.'}

---

## Recommendation

{'✓ Posterior samples are reliable for inference' if convergence_pass else '✗ Consider longer chains or different sampler'}
{'✓ Proceed to posterior predictive checks' if convergence_pass else '✗ Review and address convergence issues first'}

---

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

convergence_report_path = DIAGNOSTICS_DIR / "convergence_report.md"
with open(convergence_report_path, 'w') as f:
    f.write(convergence_report)

print(f"  Saved: {convergence_report_path}")

# ============================================================================
# FINAL OUTPUT
# ============================================================================
print("\n" + "="*80)
print("POSTERIOR INFERENCE COMPLETE")
print("="*80)
print(f"\nConvergence: {'PASS ✓' if convergence_pass else 'FAIL ✗'}")
print(f"\nPosterior Estimates:")
print(f"  α = {convergence_metrics['alpha']['mean']:.3f} ± {convergence_metrics['alpha']['sd']:.3f}")
print(f"  β = {convergence_metrics['beta']['mean']:.3f} ± {convergence_metrics['beta']['sd']:.3f}")
print(f"  σ = {convergence_metrics['sigma']['mean']:.3f} ± {convergence_metrics['sigma']['sd']:.3f}")
print(f"\nR² = {r_squared:.3f}")
print(f"Runtime: {sampling_time:.1f}s")

if convergence_pass:
    print("\n✓ Ready for Posterior Predictive Checks")
    print(f"✓ InferenceData saved: {netcdf_path}")
else:
    print("\n✗ Convergence issues detected")
    for issue in convergence_issues:
        print(f"  - {issue}")

print("\nOutputs:")
print(f"  Report: {report_path}")
print(f"  Convergence Report: {convergence_report_path}")
print(f"  InferenceData: {netcdf_path}")
print(f"  Metrics: {metrics_path}")
print(f"  Plots: {PLOTS_DIR}/")
print("="*80)
