"""
Fit Negative Binomial GLM with Quadratic Trend to Real Data
Experiment 1: Posterior Inference

Model:
  C_t ~ NegativeBinomial2(mu_t, phi)
  log(mu_t) = beta_0 + beta_1 * year_t + beta_2 * year_t^2

Priors:
  beta_0 ~ Normal(4.5, 1.0)
  beta_1 ~ Normal(0.9, 0.5)
  beta_2 ~ Normal(0, 0.3)
  phi ~ Gamma(2, 0.1)
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Paths
DATA_PATH = "/workspace/data/data.csv"
OUTPUT_DIR = "/workspace/experiments/experiment_1/posterior_inference"
DIAGNOSTICS_DIR = f"{OUTPUT_DIR}/diagnostics"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"

print("=" * 80)
print("EXPERIMENT 1: NEGATIVE BINOMIAL GLM WITH QUADRATIC TREND")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"   - Observations: {len(data)}")
print(f"   - Year range: [{data['year'].min():.2f}, {data['year'].max():.2f}]")
print(f"   - Count range: [{data['C'].min()}, {data['C'].max()}]")
print(f"   - Mean count: {data['C'].mean():.2f}")
print(f"   - Variance: {data['C'].var():.2f}")
print(f"   - Variance/Mean ratio: {data['C'].var() / data['C'].mean():.2f}")

# Extract variables
N = len(data)
year = data['year'].values
C = data['C'].values.astype(int)

# ============================================================================
# 2. BUILD PYMC MODEL
# ============================================================================
print("\n[2] Building PyMC model...")
with pm.Model() as model:
    # Priors
    beta_0 = pm.Normal('beta_0', mu=4.5, sigma=1.0)
    beta_1 = pm.Normal('beta_1', mu=0.9, sigma=0.5)
    beta_2 = pm.Normal('beta_2', mu=0.0, sigma=0.3)
    phi = pm.Gamma('phi', alpha=2, beta=0.1)

    # Linear predictor on log scale
    log_mu = beta_0 + beta_1 * year + beta_2 * year**2

    # Expected count on natural scale
    mu = pm.math.exp(log_mu)

    # Likelihood: NegativeBinomial
    # PyMC uses alpha (dispersion) parameterization: Var = mu + mu^2/alpha
    # This matches Stan's neg_binomial_2(mu, phi) where phi = alpha
    C_obs = pm.NegativeBinomial('C_obs', mu=mu, alpha=phi, observed=C)

print("   Model structure:")
print(f"   - Parameters: beta_0, beta_1, beta_2, phi")
print(f"   - Likelihood: NegativeBinomial(mu=exp(beta_0 + beta_1*year + beta_2*year^2), alpha=phi)")
print(f"   - Observations: {N}")

# ============================================================================
# 3. SAMPLE FROM POSTERIOR
# ============================================================================
print("\n[3] Sampling from posterior...")
print("   Strategy: 4 chains, 2000 iterations (1000 warmup)")
print("   Target: R-hat < 1.01, ESS > 400")

with model:
    # Sample with tuning
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}  # CRITICAL: compute log_likelihood
    )

print("\n   Sampling complete!")

# ============================================================================
# 4. CONVERGENCE DIAGNOSTICS
# ============================================================================
print("\n[4] Convergence diagnostics...")

# Get summary statistics
summary = az.summary(trace, var_names=['beta_0', 'beta_1', 'beta_2', 'phi'])
print("\n" + "=" * 80)
print("PARAMETER SUMMARY")
print("=" * 80)
print(summary.to_string())

# Check convergence criteria
rhat_max = summary['r_hat'].max()
ess_bulk_min = summary['ess_bulk'].min()
ess_tail_min = summary['ess_tail'].min()

print("\n" + "=" * 80)
print("CONVERGENCE ASSESSMENT")
print("=" * 80)
print(f"Max R-hat: {rhat_max:.4f} (threshold: < 1.01)")
print(f"Min ESS bulk: {ess_bulk_min:.0f} (threshold: > 400)")
print(f"Min ESS tail: {ess_tail_min:.0f} (threshold: > 400)")

# Check for divergences
n_divergent = trace.sample_stats.diverging.sum().values
print(f"Divergent transitions: {n_divergent} (threshold: 0)")

# Energy diagnostics
try:
    bfmi = az.bfmi(trace)
    print(f"Bayesian Fraction of Missing Information (mean): {bfmi.mean():.4f}")
    print(f"   (BFMI < 0.2 indicates problems)")
except:
    print("BFMI calculation not available")
    bfmi = np.array([1.0])

# Overall convergence decision
convergence_pass = (
    rhat_max < 1.01 and
    ess_bulk_min > 400 and
    ess_tail_min > 400 and
    n_divergent == 0
)

if convergence_pass:
    print("\n*** CONVERGENCE: PASS ***")
else:
    print("\n*** CONVERGENCE: ISSUES DETECTED ***")
    if rhat_max >= 1.01:
        print(f"   - R-hat too high ({rhat_max:.4f})")
    if ess_bulk_min <= 400:
        print(f"   - ESS bulk too low ({ess_bulk_min:.0f})")
    if ess_tail_min <= 400:
        print(f"   - ESS tail too low ({ess_tail_min:.0f})")
    if n_divergent > 0:
        print(f"   - {n_divergent} divergent transitions")

# Save convergence summary
convergence_report = f"""CONVERGENCE DIAGNOSTICS
{"=" * 80}

Sampling Configuration:
- Chains: 4
- Iterations per chain: 2000 (1000 warmup + 1000 sampling)
- Total draws: 4000
- Target accept: 0.95

Convergence Metrics:
- Max R-hat: {rhat_max:.4f} (threshold: < 1.01)
- Min ESS bulk: {ess_bulk_min:.0f} (threshold: > 400)
- Min ESS tail: {ess_tail_min:.0f} (threshold: > 400)
- Divergent transitions: {n_divergent} (threshold: 0)
- Mean BFMI: {bfmi.mean():.4f} (threshold: > 0.2)

Overall Assessment: {"PASS" if convergence_pass else "FAIL"}

Parameter-Level Diagnostics:
{summary.to_string()}
"""

with open(f"{DIAGNOSTICS_DIR}/convergence_summary.txt", 'w') as f:
    f.write(convergence_report)

print(f"\n   Saved: {DIAGNOSTICS_DIR}/convergence_summary.txt")

# Save parameter summary as CSV
summary.to_csv(f"{DIAGNOSTICS_DIR}/parameter_summary.csv")
print(f"   Saved: {DIAGNOSTICS_DIR}/parameter_summary.csv")

# ============================================================================
# 5. SAVE INFERENCE DATA WITH LOG_LIKELIHOOD
# ============================================================================
print("\n[5] Saving InferenceData with log_likelihood...")

# Verify log_likelihood is present
if 'log_likelihood' in trace.groups():
    log_lik_shape = trace.log_likelihood['C_obs'].shape
    print(f"   - log_likelihood shape: {log_lik_shape}")
    print(f"   - Expected: (4 chains, 1000 draws, {N} observations)")

    # Save to netcdf
    trace.to_netcdf(f"{DIAGNOSTICS_DIR}/posterior_inference.netcdf")
    print(f"   Saved: {DIAGNOSTICS_DIR}/posterior_inference.netcdf")
else:
    print("   ERROR: log_likelihood group not found!")
    print(f"   Available groups: {trace.groups()}")

# ============================================================================
# 6. PARAMETER INFERENCE
# ============================================================================
print("\n[6] Parameter inference...")

# Extract posterior samples
posterior = trace.posterior

beta_0_samples = posterior['beta_0'].values.flatten()
beta_1_samples = posterior['beta_1'].values.flatten()
beta_2_samples = posterior['beta_2'].values.flatten()
phi_samples = posterior['phi'].values.flatten()

# Compute credible intervals
def credible_interval(samples, prob):
    return np.percentile(samples, [(1-prob)/2*100, (1+prob)/2*100])

print("\n" + "=" * 80)
print("POSTERIOR PARAMETER ESTIMATES")
print("=" * 80)

for param_name, samples in [('beta_0', beta_0_samples),
                             ('beta_1', beta_1_samples),
                             ('beta_2', beta_2_samples),
                             ('phi', phi_samples)]:
    print(f"\n{param_name}:")
    print(f"  Mean: {samples.mean():.4f}")
    print(f"  Median: {np.median(samples):.4f}")
    print(f"  SD: {samples.std():.4f}")
    print(f"  50% CI: [{credible_interval(samples, 0.50)[0]:.4f}, {credible_interval(samples, 0.50)[1]:.4f}]")
    print(f"  90% CI: [{credible_interval(samples, 0.90)[0]:.4f}, {credible_interval(samples, 0.90)[1]:.4f}]")
    print(f"  95% CI: [{credible_interval(samples, 0.95)[0]:.4f}, {credible_interval(samples, 0.95)[1]:.4f}]")

# Interpretation
print("\n" + "=" * 80)
print("PARAMETER INTERPRETATION")
print("=" * 80)
print(f"beta_0 (intercept): Baseline log-count at year=0 is {beta_0_samples.mean():.4f}")
print(f"   -> exp({beta_0_samples.mean():.4f}) = {np.exp(beta_0_samples.mean()):.1f} cases at year=0")
print(f"\nbeta_1 (linear trend): Linear growth rate on log scale is {beta_1_samples.mean():.4f}")
print(f"   -> Positive value indicates exponential growth")
print(f"\nbeta_2 (quadratic trend): Acceleration/deceleration is {beta_2_samples.mean():.4f}")
if beta_2_samples.mean() > 0:
    print(f"   -> Positive value indicates accelerating growth")
elif beta_2_samples.mean() < 0:
    print(f"   -> Negative value indicates decelerating growth")
else:
    print(f"   -> Near-zero indicates approximately linear growth on log scale")
print(f"\nphi (dispersion): Overdispersion parameter is {phi_samples.mean():.4f}")
print(f"   -> Variance = mu + mu^2/phi")
print(f"   -> Higher phi = less overdispersion")

# ============================================================================
# 7. MODEL FIT ASSESSMENT
# ============================================================================
print("\n[7] Model fit assessment...")

# Generate posterior predictive mean for each observation
# Use posterior mean of parameters
beta_0_mean = beta_0_samples.mean()
beta_1_mean = beta_1_samples.mean()
beta_2_mean = beta_2_samples.mean()

log_mu_pred = beta_0_mean + beta_1_mean * year + beta_2_mean * year**2
mu_pred = np.exp(log_mu_pred)

# Compute residuals
residuals = C - mu_pred

# Fit metrics
mae = np.abs(residuals).mean()
rmse = np.sqrt((residuals**2).mean())

# Bayesian R-squared
# R^2 = Var(fitted) / Var(observed)
var_fitted = mu_pred.var()
var_observed = C.var()
r2_bayesian = var_fitted / var_observed

print(f"\nIn-sample fit metrics:")
print(f"  MAE: {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  Bayesian R²: {r2_bayesian:.4f}")
print(f"\nResidual statistics:")
print(f"  Mean: {residuals.mean():.2f}")
print(f"  SD: {residuals.std():.2f}")
print(f"  Min: {residuals.min():.2f}")
print(f"  Max: {residuals.max():.2f}")

# Check for residual autocorrelation
residual_acf_lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
print(f"  ACF lag-1: {residual_acf_lag1:.4f}")

# ============================================================================
# 8. DIAGNOSTIC PLOTS
# ============================================================================
print("\n[8] Creating diagnostic plots...")

# 8.1 Trace plots
print("   - Trace plots...")
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
az.plot_trace(trace, var_names=['beta_0', 'beta_1', 'beta_2', 'phi'], axes=axes)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/trace_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"     Saved: {PLOTS_DIR}/trace_plots.png")

# 8.2 Posterior distributions
print("   - Posterior distributions...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
az.plot_posterior(trace, var_names=['beta_0', 'beta_1', 'beta_2', 'phi'],
                  kind='hist', hdi_prob=0.95, axes=axes.flatten())
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/posterior_distributions.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"     Saved: {PLOTS_DIR}/posterior_distributions.png")

# 8.3 Rank plots for convergence
print("   - Rank plots...")
az.plot_rank(trace, var_names=['beta_0', 'beta_1', 'beta_2', 'phi'])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/rank_plots.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"     Saved: {PLOTS_DIR}/rank_plots.png")

# 8.4 Fitted trend
print("   - Fitted trend with credible intervals...")

# Generate posterior predictive samples for visualization
n_samples = 500
sample_indices = np.random.choice(len(beta_0_samples), size=n_samples, replace=False)

# Create year grid for smooth curve
year_grid = np.linspace(year.min(), year.max(), 200)
mu_samples = np.zeros((n_samples, len(year_grid)))

for i, idx in enumerate(sample_indices):
    log_mu_i = beta_0_samples[idx] + beta_1_samples[idx] * year_grid + beta_2_samples[idx] * year_grid**2
    mu_samples[i, :] = np.exp(log_mu_i)

# Compute credible intervals
mu_median = np.median(mu_samples, axis=0)
mu_50_lower = np.percentile(mu_samples, 25, axis=0)
mu_50_upper = np.percentile(mu_samples, 75, axis=0)
mu_95_lower = np.percentile(mu_samples, 2.5, axis=0)
mu_95_upper = np.percentile(mu_samples, 97.5, axis=0)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(year, C, alpha=0.6, s=50, label='Observed data', zorder=3)
ax.plot(year_grid, mu_median, 'r-', linewidth=2, label='Posterior median', zorder=4)
ax.fill_between(year_grid, mu_50_lower, mu_50_upper, alpha=0.3, color='red',
                label='50% CI', zorder=1)
ax.fill_between(year_grid, mu_95_lower, mu_95_upper, alpha=0.2, color='red',
                label='95% CI', zorder=0)
ax.set_xlabel('Year (standardized)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Fitted Trend with Posterior Credible Intervals', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/fitted_trend.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"     Saved: {PLOTS_DIR}/fitted_trend.png")

# 8.5 Residual diagnostics
print("   - Residual diagnostics...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals over time
axes[0, 0].scatter(year, residuals, alpha=0.6)
axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Year (standardized)')
axes[0, 0].set_ylabel('Residual (Observed - Predicted)')
axes[0, 0].set_title('Residuals over Time')
axes[0, 0].grid(True, alpha=0.3)

# Residuals vs fitted
axes[0, 1].scatter(mu_pred, residuals, alpha=0.6)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Fitted values')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].set_title('Residuals vs Fitted')
axes[0, 1].grid(True, alpha=0.3)

# ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=20, ax=axes[1, 0], alpha=0.05)
axes[1, 0].set_title(f'Residual ACF (lag-1 = {residual_acf_lag1:.3f})')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelation')

# QQ plot
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot of Residuals')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/residual_diagnostics.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"     Saved: {PLOTS_DIR}/residual_diagnostics.png")

# ============================================================================
# 9. ISSUES FOR POSTERIOR PREDICTIVE CHECK
# ============================================================================
print("\n[9] Issues identified for PPC...")

issues = []

if residual_acf_lag1 > 0.5:
    issues.append(f"Strong residual autocorrelation (ACF lag-1 = {residual_acf_lag1:.3f})")
    print(f"   - Strong residual autocorrelation detected (ACF lag-1 = {residual_acf_lag1:.3f})")

# Check for heteroscedasticity (correlation between |residuals| and fitted values)
heterosced_corr = np.corrcoef(mu_pred, np.abs(residuals))[0, 1]
if abs(heterosced_corr) > 0.3:
    issues.append(f"Heteroscedasticity detected (corr = {heterosced_corr:.3f})")
    print(f"   - Heteroscedasticity detected (|residual| vs fitted corr = {heterosced_corr:.3f})")

# Check for outliers (residuals > 3 SD)
outliers = np.abs(residuals) > 3 * residuals.std()
n_outliers = outliers.sum()
if n_outliers > 0:
    issues.append(f"{n_outliers} outliers (|residual| > 3 SD)")
    print(f"   - {n_outliers} outliers detected (|residual| > 3 SD)")

# Check for systematic bias (mean residual significantly different from 0)
mean_residual = residuals.mean()
if abs(mean_residual) > 5:
    issues.append(f"Systematic bias (mean residual = {mean_residual:.2f})")
    print(f"   - Systematic bias detected (mean residual = {mean_residual:.2f})")

if not issues:
    print("   - No major issues detected in residuals")
else:
    print(f"\n   Total issues: {len(issues)}")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\nConvergence: {'PASS' if convergence_pass else 'FAIL'}")
print(f"In-sample R²: {r2_bayesian:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"Residual ACF lag-1: {residual_acf_lag1:.4f}")

decision = "PROCEED TO PPC" if convergence_pass else "SKIP TO CRITIQUE"
print(f"\nDecision: {decision}")

if issues:
    print(f"\nIssues for PPC investigation:")
    for issue in issues:
        print(f"  - {issue}")

print("\n" + "=" * 80)
print("OUTPUTS SAVED TO:")
print("=" * 80)
print(f"  Code: {OUTPUT_DIR}/code/fit_model_pymc.py")
print(f"  Diagnostics: {DIAGNOSTICS_DIR}/")
print(f"    - posterior_inference.netcdf")
print(f"    - convergence_summary.txt")
print(f"    - parameter_summary.csv")
print(f"  Plots: {PLOTS_DIR}/")
print(f"    - trace_plots.png")
print(f"    - posterior_distributions.png")
print(f"    - rank_plots.png")
print(f"    - fitted_trend.png")
print(f"    - residual_diagnostics.png")
print("=" * 80)
