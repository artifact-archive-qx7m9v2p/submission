"""
Comprehensive Model Assessment for Experiment 1: Fixed Changepoint Negative Binomial

This script conducts complete assessment of the ACCEPTED model, including:
1. LOO Cross-Validation with Pareto k diagnostics
2. Calibration assessment (via coverage analysis)
3. Absolute predictive metrics (RMSE, MAE, MAPE, R²)
4. Temporal structure assessment by regime
5. Uncertainty quantification and coverage
6. Scientific validity assessment
7. Adequacy determination

Author: Model Assessment Agent
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("COMPREHENSIVE MODEL ASSESSMENT - EXPERIMENT 1")
print("="*80)
print()

# ==============================================================================
# 1. LOAD DATA AND MODEL
# ==============================================================================

print("1. Loading data and posterior inference...")
print("-"*80)

# Load data
data = pd.read_csv('/workspace/data/data.csv')
C_obs = data['C'].values
year = data['year'].values
N = len(C_obs)

# Changepoint
tau = 17

# Load posterior
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')

print(f"✓ Loaded {N} observations")
print(f"✓ Loaded posterior with {idata.posterior.dims['draw']} draws × {idata.posterior.dims['chain']} chains")
print(f"✓ Available groups: {list(idata.groups())}")
print()

# Extract posterior samples (keeping chain/draw dimensions first)
beta_0_full = idata.posterior['beta_0'].values  # shape: (chains, draws)
beta_1_full = idata.posterior['beta_1'].values
beta_2_full = idata.posterior['beta_2'].values
alpha_full = idata.posterior['alpha'].values

# Flattened versions for summary statistics
beta_0 = beta_0_full.flatten()
beta_1 = beta_1_full.flatten()
beta_2 = beta_2_full.flatten()
alpha = alpha_full.flatten()

print(f"✓ Extracted parameters: beta_0, beta_1, beta_2, alpha")

# Reconstruct mu (log mean) from model equation
# log(μ_t) = β_0 + β_1 × year_t + β_2 × I(t > tau) × (year_t - year_tau)
print("✓ Reconstructing mu from model parameters...")

year_tau = year[tau]  # year value at changepoint

# Create post-break indicator
post_break = np.arange(N) >= tau  # Boolean array

# Compute mu for each posterior sample
# Shape will be (chains, draws, N)
n_chains = beta_0_full.shape[0]
n_draws = beta_0_full.shape[1]

mu_samples = np.zeros((n_chains, n_draws, N))
for i in range(N):
    # Base: beta_0 + beta_1 * year
    log_mu = beta_0_full + beta_1_full * year[i]

    # Add changepoint effect if post-break
    if post_break[i]:
        log_mu += beta_2_full * (year[i] - year_tau)

    mu_samples[:, :, i] = np.exp(log_mu)

print(f"✓ Reconstructed mu with shape {mu_samples.shape}")
print()

# ==============================================================================
# 2. LOO CROSS-VALIDATION ASSESSMENT
# ==============================================================================

print("2. Computing LOO Cross-Validation...")
print("-"*80)

# Compute LOO
loo = az.loo(idata, pointwise=True)

# Extract metrics
elpd_loo = loo.elpd_loo
se_elpd = loo.se
p_loo = loo.p_loo
pareto_k = loo.pareto_k.values if hasattr(loo.pareto_k, 'values') else loo.pareto_k

# Pareto k diagnostics
k_good = np.sum(pareto_k < 0.5)
k_ok = np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))
k_bad = np.sum((pareto_k >= 0.7) & (pareto_k < 1.0))
k_very_bad = np.sum(pareto_k >= 1.0)

print(f"ELPD_loo: {elpd_loo:.2f} ± {se_elpd:.2f}")
print(f"p_loo (effective parameters): {p_loo:.2f}")
print()
print("Pareto k diagnostics:")
print(f"  k < 0.5 (good):           {k_good}/{N} ({k_good/N*100:.1f}%)")
print(f"  0.5 ≤ k < 0.7 (ok):       {k_ok}/{N} ({k_ok/N*100:.1f}%)")
print(f"  0.7 ≤ k < 1.0 (bad):      {k_bad}/{N} ({k_bad/N*100:.1f}%)")
print(f"  k ≥ 1.0 (very bad):       {k_very_bad}/{N} ({k_very_bad/N*100:.1f}%)")
print(f"  Max k: {np.max(pareto_k):.3f}")
print(f"  Mean k: {np.mean(pareto_k):.3f}")
print()

# Interpret results
if k_bad + k_very_bad == 0:
    loo_status = "EXCELLENT"
    loo_interpretation = "All observations have reliable LOO estimates. Model generalizes well."
elif k_bad + k_very_bad < 0.1 * N:
    loo_status = "GOOD"
    loo_interpretation = f"{k_bad + k_very_bad} observations have problematic k values, but less than 10%."
else:
    loo_status = "CONCERNING"
    loo_interpretation = f"{k_bad + k_very_bad} observations ({(k_bad+k_very_bad)/N*100:.1f}%) have unreliable LOO estimates."

print(f"LOO Status: {loo_status}")
print(f"Interpretation: {loo_interpretation}")
print()

# p_loo interpretation
actual_params = 4  # beta_0, beta_1, beta_2, alpha
print(f"p_loo ({p_loo:.2f}) vs actual parameters ({actual_params}):")
if p_loo < actual_params * 0.8:
    print("  Model is well-regularized by priors (p_loo < actual params)")
elif p_loo > actual_params * 1.2:
    print("  Model may be overfitting (p_loo > actual params)")
else:
    print("  p_loo consistent with model complexity")
print()

# Save LOO results
loo_df = pd.DataFrame({
    'observation': np.arange(1, N+1),
    'year': year,
    'C_obs': C_obs,
    'pareto_k': pareto_k,
})
loo_df.to_csv('/workspace/experiments/model_assessment/results/loo_results.csv', index=False)
print("✓ Saved LOO results to results/loo_results.csv")

# Save LOO summary
with open('/workspace/experiments/model_assessment/results/loo_summary.txt', 'w') as f:
    f.write("LOO Cross-Validation Summary\n")
    f.write("="*80 + "\n\n")
    f.write(f"ELPD_loo: {elpd_loo:.2f} ± {se_elpd:.2f}\n")
    f.write(f"p_loo: {p_loo:.2f}\n")
    f.write(f"Actual parameters: {actual_params}\n\n")
    f.write("Pareto k diagnostics:\n")
    f.write(f"  k < 0.5 (good):       {k_good}/{N} ({k_good/N*100:.1f}%)\n")
    f.write(f"  0.5 ≤ k < 0.7 (ok):   {k_ok}/{N} ({k_ok/N*100:.1f}%)\n")
    f.write(f"  0.7 ≤ k < 1.0 (bad):  {k_bad}/{N} ({k_bad/N*100:.1f}%)\n")
    f.write(f"  k ≥ 1.0 (very bad):   {k_very_bad}/{N} ({k_very_bad/N*100:.1f}%)\n")
    f.write(f"  Max k: {np.max(pareto_k):.3f}\n")
    f.write(f"  Mean k: {np.mean(pareto_k):.3f}\n\n")
    f.write(f"Status: {loo_status}\n")
    f.write(f"Interpretation: {loo_interpretation}\n")
print("✓ Saved LOO summary to results/loo_summary.txt")
print()

# ==============================================================================
# 3. CALIBRATION ASSESSMENT (Coverage-based)
# ==============================================================================

print("3. Assessing calibration via coverage analysis...")
print("-"*80)
print("Note: LOO-PIT requires posterior_predictive samples (not available).")
print("Using coverage-based calibration assessment instead.")
print()

# ==============================================================================
# 4. ABSOLUTE PREDICTIVE METRICS
# ==============================================================================

print("4. Computing absolute predictive metrics...")
print("-"*80)

# Posterior predictive mean for each observation
mu_mean = mu_samples.mean(axis=(0, 1))  # Average over chains and draws

# Compute metrics
residuals = C_obs - mu_mean
rmse = np.sqrt(np.mean(residuals**2))
mae = np.mean(np.abs(residuals))
mape = np.mean(np.abs(residuals / C_obs)) * 100

# R-squared
ss_res = np.sum(residuals**2)
ss_tot = np.sum((C_obs - C_obs.mean())**2)
r2 = 1 - ss_res / ss_tot

print(f"Predictive Performance Metrics:")
print(f"  RMSE:  {rmse:.2f}")
print(f"  MAE:   {mae:.2f}")
print(f"  MAPE:  {mape:.2f}%")
print(f"  R²:    {r2:.4f} ({r2*100:.2f}% variance explained)")
print()

# Context from data
print(f"Context (for interpretation):")
print(f"  Observed C range: [{C_obs.min()}, {C_obs.max()}]")
print(f"  Observed C mean: {C_obs.mean():.1f} ± {C_obs.std():.1f}")
print(f"  RMSE as % of mean: {rmse/C_obs.mean()*100:.1f}%")
print(f"  MAE as % of mean: {mae/C_obs.mean()*100:.1f}%")
print()

# ==============================================================================
# 5. TEMPORAL STRUCTURE ASSESSMENT BY REGIME
# ==============================================================================

print("5. Temporal structure assessment by regime...")
print("-"*80)

# Split by regime (changepoint at t=17, so indices 0-16 are pre-break, 17-39 are post-break)
pre_break_idx = np.arange(tau)
post_break_idx = np.arange(tau, N)

# Pre-break metrics
pre_residuals = residuals[pre_break_idx]
pre_rmse = np.sqrt(np.mean(pre_residuals**2))
pre_mae = np.mean(np.abs(pre_residuals))
pre_r2 = 1 - np.sum(pre_residuals**2) / np.sum((C_obs[pre_break_idx] - C_obs[pre_break_idx].mean())**2)

# Post-break metrics
post_residuals = residuals[post_break_idx]
post_rmse = np.sqrt(np.mean(post_residuals**2))
post_mae = np.mean(np.abs(post_residuals))
post_r2 = 1 - np.sum(post_residuals**2) / np.sum((C_obs[post_break_idx] - C_obs[post_break_idx].mean())**2)

print("Pre-break regime (observations 1-17):")
print(f"  RMSE: {pre_rmse:.2f}")
print(f"  MAE:  {pre_mae:.2f}")
print(f"  R²:   {pre_r2:.4f}")
print(f"  Mean obs: {C_obs[pre_break_idx].mean():.1f}")
print()

print("Post-break regime (observations 18-40):")
print(f"  RMSE: {post_rmse:.2f}")
print(f"  MAE:  {post_mae:.2f}")
print(f"  R²:   {post_r2:.4f}")
print(f"  Mean obs: {C_obs[post_break_idx].mean():.1f}")
print()

# Relative performance
print("Regime comparison:")
if pre_rmse < post_rmse:
    print(f"  Pre-break has lower RMSE (better fit)")
    print(f"  RMSE ratio (post/pre): {post_rmse/pre_rmse:.2f}x")
else:
    print(f"  Post-break has lower RMSE (better fit)")
    print(f"  RMSE ratio (pre/post): {pre_rmse/post_rmse:.2f}x")
print()

# ==============================================================================
# 6. UNCERTAINTY QUANTIFICATION AND COVERAGE
# ==============================================================================

print("6. Uncertainty quantification and coverage...")
print("-"*80)

# Compute credible intervals for mu
mu_5 = np.percentile(mu_samples, 5, axis=(0, 1))
mu_95 = np.percentile(mu_samples, 95, axis=(0, 1))

# Coverage: how many observations fall within 90% CI?
in_interval = (C_obs >= mu_5) & (C_obs <= mu_95)
coverage = np.mean(in_interval) * 100

print(f"90% Credible Interval Coverage:")
print(f"  Observations in interval: {np.sum(in_interval)}/{N}")
print(f"  Coverage: {coverage:.1f}%")
print()

if coverage < 85:
    coverage_status = "UNDER-COVERAGE (model over-confident)"
elif coverage > 95:
    coverage_status = "OVER-COVERAGE (model under-confident)"
else:
    coverage_status = "APPROPRIATE (near nominal 90%)"

print(f"Coverage status: {coverage_status}")
print()

# Interval widths
interval_widths = mu_95 - mu_5
print(f"Credible interval widths:")
print(f"  Mean width: {interval_widths.mean():.2f}")
print(f"  Min width: {interval_widths.min():.2f}")
print(f"  Max width: {interval_widths.max():.2f}")
print(f"  Width as % of mean prediction: {interval_widths.mean()/mu_mean.mean()*100:.1f}%")
print()

# Width by regime
pre_widths = interval_widths[pre_break_idx]
post_widths = interval_widths[post_break_idx]
print(f"Interval widths by regime:")
print(f"  Pre-break mean width: {pre_widths.mean():.2f}")
print(f"  Post-break mean width: {post_widths.mean():.2f}")
print(f"  Ratio (post/pre): {post_widths.mean()/pre_widths.mean():.2f}x")
print()

# ==============================================================================
# 7. SCIENTIFIC VALIDITY ASSESSMENT
# ==============================================================================

print("7. Scientific validity assessment...")
print("-"*80)

# Primary research question
print("PRIMARY RESEARCH QUESTION:")
print("  Is there a structural break at observation 17?")
print()

# Statistical evidence
prob_positive = np.mean(beta_2 > 0)
print(f"Evidence for structural break (β₂ > 0):")
print(f"  P(β₂ > 0): {prob_positive*100:.2f}%")
print(f"  β₂ mean: {beta_2.mean():.3f}")
print(f"  β₂ 95% HDI: [{np.percentile(beta_2, 2.5):.3f}, {np.percentile(beta_2, 97.5):.3f}]")
print()

if prob_positive > 0.99:
    evidence_level = "CONCLUSIVE (>99%)"
elif prob_positive > 0.95:
    evidence_level = "STRONG (>95%)"
elif prob_positive > 0.90:
    evidence_level = "MODERATE (>90%)"
else:
    evidence_level = "WEAK (<90%)"

print(f"Evidence level: {evidence_level}")
print()

# Effect size
beta_1_mean = beta_1.mean()
total_post_slope = beta_1 + beta_2
acceleration_ratio = total_post_slope / beta_1
acc_mean = acceleration_ratio.mean()
acc_5 = np.percentile(acceleration_ratio, 5)
acc_95 = np.percentile(acceleration_ratio, 95)

print(f"Effect size (acceleration in growth rate):")
print(f"  Pre-break slope (β₁): {beta_1_mean:.3f}")
print(f"  Post-break slope (β₁ + β₂): {total_post_slope.mean():.3f}")
print(f"  Acceleration ratio: {acc_mean:.2f}x (90% CI: [{acc_5:.2f}, {acc_95:.2f}])")
print(f"  Percentage increase: {(acc_mean-1)*100:.1f}%")
print()

# Parameter interpretability
print("Parameter interpretability:")
print(f"  β₀ (intercept): {beta_0.mean():.3f} - log-rate at year=0")
print(f"  β₁ (pre-break slope): {beta_1_mean:.3f} - exponential growth rate")
print(f"  β₂ (additional slope): {beta_2.mean():.3f} - regime change magnitude")
print(f"  α (dispersion): {alpha.mean():.3f} - overdispersion parameter")
print("  All parameters have clear scientific meaning ✓")
print()

# ==============================================================================
# 8. SAVE COMPREHENSIVE METRICS
# ==============================================================================

print("8. Saving comprehensive metrics...")
print("-"*80)

metrics_dict = {
    # LOO metrics
    'elpd_loo': float(elpd_loo),
    'se_elpd': float(se_elpd),
    'p_loo': float(p_loo),
    'pareto_k_max': float(np.max(pareto_k)),
    'pareto_k_mean': float(np.mean(pareto_k)),
    'pareto_k_good_pct': float(k_good/N*100),
    'pareto_k_bad_pct': float((k_bad + k_very_bad)/N*100),

    # Absolute metrics
    'rmse': float(rmse),
    'mae': float(mae),
    'mape': float(mape),
    'r_squared': float(r2),

    # By regime
    'pre_break_rmse': float(pre_rmse),
    'pre_break_mae': float(pre_mae),
    'pre_break_r2': float(pre_r2),
    'post_break_rmse': float(post_rmse),
    'post_break_mae': float(post_mae),
    'post_break_r2': float(post_r2),

    # Coverage
    'coverage_90pct': float(coverage),
    'mean_interval_width': float(interval_widths.mean()),
    'pre_break_width': float(pre_widths.mean()),
    'post_break_width': float(post_widths.mean()),

    # Scientific validity
    'prob_beta2_positive': float(prob_positive),
    'beta_2_mean': float(beta_2.mean()),
    'acceleration_ratio': float(acc_mean),
    'pct_increase': float((acc_mean-1)*100)
}

metrics_df = pd.DataFrame([metrics_dict])
metrics_df.to_csv('/workspace/experiments/model_assessment/results/assessment_metrics.csv', index=False)
print("✓ Saved assessment metrics to results/assessment_metrics.csv")
print()

# ==============================================================================
# 9. CREATE VISUALIZATIONS
# ==============================================================================

print("9. Creating visualizations...")
print("-"*80)

# -------------------------
# Plot 1: LOO Diagnostics
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Pareto k values over time
ax = axes[0, 0]
ax.plot(np.arange(1, N+1), pareto_k, 'o-', alpha=0.7, markersize=6)
ax.axhline(0.5, color='orange', linestyle='--', label='k=0.5 (concern threshold)')
ax.axhline(0.7, color='red', linestyle='--', label='k=0.7 (bad threshold)')
ax.axvline(tau, color='purple', linestyle=':', alpha=0.5, label='Changepoint')
ax.set_xlabel('Observation')
ax.set_ylabel('Pareto k')
ax.set_title('LOO Pareto k Diagnostics')
ax.legend()
ax.grid(True, alpha=0.3)

# Pareto k histogram
ax = axes[0, 1]
ax.hist(pareto_k, bins=20, alpha=0.7, edgecolor='black')
ax.axvline(0.5, color='orange', linestyle='--', linewidth=2)
ax.axvline(0.7, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Pareto k')
ax.set_ylabel('Count')
ax.set_title(f'Pareto k Distribution (max={np.max(pareto_k):.3f})')
ax.grid(True, alpha=0.3)

# Scatter of k vs observation index colored by regime
ax = axes[1, 0]
colors = ['blue' if i < tau else 'orange' for i in range(N)]
ax.scatter(np.arange(1, N+1), pareto_k, c=colors, alpha=0.7, s=60, edgecolors='black', linewidths=1)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=2)
ax.axhline(0.7, color='red', linestyle='--', linewidth=2)
ax.axvline(tau, color='purple', linestyle=':', alpha=0.5)
ax.set_xlabel('Observation')
ax.set_ylabel('Pareto k')
ax.set_title('Pareto k by Regime (Blue=Pre, Orange=Post)')
ax.grid(True, alpha=0.3)

# Summary text
ax = axes[1, 1]
ax.axis('off')
summary_text = f"""LOO Cross-Validation Summary

ELPD_loo: {elpd_loo:.2f} ± {se_elpd:.2f}
p_loo: {p_loo:.2f}

Pareto k diagnostics:
  Good (k < 0.5): {k_good}/{N} ({k_good/N*100:.1f}%)
  OK (0.5 ≤ k < 0.7): {k_ok}/{N} ({k_ok/N*100:.1f}%)
  Bad (k ≥ 0.7): {k_bad + k_very_bad}/{N} ({(k_bad + k_very_bad)/N*100:.1f}%)

Max k: {np.max(pareto_k):.3f}
Mean k: {np.mean(pareto_k):.3f}

Status: {loo_status}
"""
ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', va='center')

plt.tight_layout()
plt.savefig('/workspace/experiments/model_assessment/plots/loo_diagnostics.png', dpi=300, bbox_inches='tight')
print("✓ Saved plots/loo_diagnostics.png")
plt.close()

# -------------------------
# Plot 2: Predictive Performance
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Observed vs Predicted
ax = axes[0, 0]
ax.scatter(mu_mean, C_obs, alpha=0.7, s=60, edgecolors='black', linewidths=1)
min_val = min(mu_mean.min(), C_obs.min())
max_val = max(mu_mean.max(), C_obs.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
ax.set_xlabel('Predicted (posterior mean)')
ax.set_ylabel('Observed')
ax.set_title(f'Observed vs Predicted (R² = {r2:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Fitted time series
ax = axes[0, 1]
ax.plot(np.arange(1, N+1), C_obs, 'ko', label='Observed', markersize=6, alpha=0.7)
ax.plot(np.arange(1, N+1), mu_mean, 'b-', label='Posterior mean', linewidth=2)
ax.fill_between(np.arange(1, N+1), mu_5, mu_95, alpha=0.3, label='90% CI')
ax.axvline(tau, color='purple', linestyle=':', alpha=0.5, label='Changepoint')
ax.set_xlabel('Observation')
ax.set_ylabel('C')
ax.set_title('Model Fit Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals vs Time
ax = axes[1, 0]
ax.scatter(np.arange(1, N+1), residuals, alpha=0.7, s=60, c=np.arange(N), cmap='viridis')
ax.axhline(0, color='red', linestyle='--')
ax.axvline(tau, color='purple', linestyle=':', alpha=0.5, label='Changepoint')
ax.set_xlabel('Observation')
ax.set_ylabel('Residual (Observed - Predicted)')
ax.set_title('Residuals Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals vs Fitted
ax = axes[1, 1]
ax.scatter(mu_mean, residuals, alpha=0.7, s=60, edgecolors='black', linewidths=1)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Predicted (posterior mean)')
ax.set_ylabel('Residual')
ax.set_title('Residuals vs Fitted')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/model_assessment/plots/predictive_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved plots/predictive_performance.png")
plt.close()

# -------------------------
# Plot 3: Residuals Temporal Analysis
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals by regime
ax = axes[0, 0]
ax.boxplot([pre_residuals, post_residuals], labels=['Pre-break', 'Post-break'])
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.set_ylabel('Residuals')
ax.set_title('Residuals by Regime')
ax.grid(True, alpha=0.3)

# Absolute residuals over time
ax = axes[0, 1]
ax.plot(np.arange(1, N+1), np.abs(residuals), 'o-', alpha=0.7)
ax.axvline(tau, color='purple', linestyle=':', alpha=0.5, label='Changepoint')
ax.set_xlabel('Observation')
ax.set_ylabel('|Residual|')
ax.set_title('Absolute Residuals Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Q-Q plot
ax = axes[1, 0]
standardized_residuals = residuals / np.std(residuals)
stats.probplot(standardized_residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot (Standardized Residuals)')
ax.grid(True, alpha=0.3)

# ACF plot (from existing data if available)
ax = axes[1, 1]
# Compute ACF manually
max_lag = 10
acf_values = [1.0]  # ACF at lag 0 is always 1
for lag in range(1, max_lag + 1):
    if lag < len(residuals):
        acf = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
        acf_values.append(acf)
    else:
        acf_values.append(0)

ax.stem(range(len(acf_values)), acf_values, basefmt=' ')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.axhline(1.96/np.sqrt(N), color='blue', linestyle='--', label='95% CI')
ax.axhline(-1.96/np.sqrt(N), color='blue', linestyle='--')
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_title(f'Residual ACF (ACF(1) = {acf_values[1]:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/experiments/model_assessment/plots/residuals_temporal.png', dpi=300, bbox_inches='tight')
print("✓ Saved plots/residuals_temporal.png")
plt.close()

# -------------------------
# Plot 4: Regime Comparison
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Pre-break fit
ax = axes[0, 0]
ax.plot(np.arange(1, tau+1), C_obs[pre_break_idx], 'ko', label='Observed', markersize=8)
ax.plot(np.arange(1, tau+1), mu_mean[pre_break_idx], 'b-', label='Predicted', linewidth=2)
ax.fill_between(np.arange(1, tau+1), mu_5[pre_break_idx], mu_95[pre_break_idx], alpha=0.3)
ax.set_xlabel('Observation')
ax.set_ylabel('C')
ax.set_title(f'Pre-Break Fit (obs 1-{tau})')
ax.legend()
ax.grid(True, alpha=0.3)

# Post-break fit
ax = axes[0, 1]
ax.plot(np.arange(tau+1, N+1), C_obs[post_break_idx], 'ko', label='Observed', markersize=8)
ax.plot(np.arange(tau+1, N+1), mu_mean[post_break_idx], 'b-', label='Predicted', linewidth=2)
ax.fill_between(np.arange(tau+1, N+1), mu_5[post_break_idx], mu_95[post_break_idx], alpha=0.3)
ax.set_xlabel('Observation')
ax.set_ylabel('C')
ax.set_title(f'Post-Break Fit (obs {tau+1}-{N})')
ax.legend()
ax.grid(True, alpha=0.3)

# Metrics comparison
ax = axes[0, 2]
metrics_comp = pd.DataFrame({
    'Pre-break': [pre_rmse, pre_mae, pre_r2],
    'Post-break': [post_rmse, post_mae, post_r2]
}, index=['RMSE', 'MAE', 'R²'])
metrics_comp.plot(kind='bar', ax=ax, rot=0)
ax.set_ylabel('Value')
ax.set_title('Predictive Metrics by Regime')
ax.legend()
ax.grid(True, alpha=0.3)

# Pre-break residuals
ax = axes[1, 0]
ax.scatter(np.arange(1, tau+1), pre_residuals, alpha=0.7, s=60, c='blue')
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Observation')
ax.set_ylabel('Residual')
ax.set_title('Pre-Break Residuals')
ax.grid(True, alpha=0.3)

# Post-break residuals
ax = axes[1, 1]
ax.scatter(np.arange(tau+1, N+1), post_residuals, alpha=0.7, s=60, c='orange')
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Observation')
ax.set_ylabel('Residual')
ax.set_title('Post-Break Residuals')
ax.grid(True, alpha=0.3)

# Summary text
ax = axes[1, 2]
ax.axis('off')
regime_text = f"""Regime Performance Summary

Pre-Break (obs 1-{tau}):
  RMSE: {pre_rmse:.2f}
  MAE:  {pre_mae:.2f}
  R²:   {pre_r2:.3f}
  Mean C: {C_obs[pre_break_idx].mean():.1f}

Post-Break (obs {tau+1}-{N}):
  RMSE: {post_rmse:.2f}
  MAE:  {post_mae:.2f}
  R²:   {post_r2:.3f}
  Mean C: {C_obs[post_break_idx].mean():.1f}

Overall:
  RMSE: {rmse:.2f}
  MAE:  {mae:.2f}
  R²:   {r2:.3f}
"""
ax.text(0.1, 0.5, regime_text, fontsize=11, family='monospace', va='center')

plt.tight_layout()
plt.savefig('/workspace/experiments/model_assessment/plots/regime_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved plots/regime_comparison.png")
plt.close()

# -------------------------
# Plot 5: Uncertainty Assessment
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Interval widths over time
ax = axes[0, 0]
ax.plot(np.arange(1, N+1), interval_widths, 'o-', alpha=0.7, markersize=6)
ax.axvline(tau, color='purple', linestyle=':', alpha=0.5, label='Changepoint')
ax.set_xlabel('Observation')
ax.set_ylabel('90% CI Width')
ax.set_title('Uncertainty (CI Width) Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Interval widths vs predicted values
ax = axes[0, 1]
ax.scatter(mu_mean, interval_widths, alpha=0.7, s=60, edgecolors='black', linewidths=1)
ax.set_xlabel('Predicted Value')
ax.set_ylabel('90% CI Width')
ax.set_title('Uncertainty vs Prediction Level')
ax.grid(True, alpha=0.3)

# Coverage indicator
ax = axes[1, 0]
colors = ['green' if x else 'red' for x in in_interval]
ax.scatter(np.arange(1, N+1), C_obs, c=colors, s=80, alpha=0.7, edgecolors='black', linewidths=1)
ax.plot(np.arange(1, N+1), mu_mean, 'b-', linewidth=2, alpha=0.5)
ax.fill_between(np.arange(1, N+1), mu_5, mu_95, alpha=0.2)
ax.axvline(tau, color='purple', linestyle=':', alpha=0.5)
ax.set_xlabel('Observation')
ax.set_ylabel('C')
ax.set_title(f'Coverage Visualization (Green = In CI, Red = Outside)')
ax.grid(True, alpha=0.3)

# Summary
ax = axes[1, 1]
ax.axis('off')
uncertainty_text = f"""Uncertainty Quantification

Coverage:
  90% CI coverage: {coverage:.1f}%
  Target: 90%
  In interval: {np.sum(in_interval)}/{N}
  Status: {coverage_status}

Interval Widths:
  Mean: {interval_widths.mean():.2f}
  Min: {interval_widths.min():.2f}
  Max: {interval_widths.max():.2f}
  CV: {np.std(interval_widths)/np.mean(interval_widths):.2f}

By Regime:
  Pre-break: {pre_widths.mean():.2f}
  Post-break: {post_widths.mean():.2f}
  Ratio (post/pre): {post_widths.mean()/pre_widths.mean():.2f}x
"""
ax.text(0.1, 0.5, uncertainty_text, fontsize=11, family='monospace', va='center')

plt.tight_layout()
plt.savefig('/workspace/experiments/model_assessment/plots/uncertainty_assessment.png', dpi=300, bbox_inches='tight')
print("✓ Saved plots/uncertainty_assessment.png")
plt.close()

print()
print("="*80)
print("ASSESSMENT COMPLETE")
print("="*80)
print()
print(f"Results saved to: /workspace/experiments/model_assessment/")
print(f"  - results/: LOO results, assessment metrics, summary")
print(f"  - plots/: 5 comprehensive visualization files")
print()
print("Key findings:")
print(f"  LOO Status: {loo_status}")
print(f"  Coverage: {coverage:.1f}% (target: 90%)")
print(f"  R²: {r2:.3f}")
print(f"  Evidence for structural break: {prob_positive*100:.2f}%")
print(f"  Acceleration ratio: {acc_mean:.2f}x")
print(f"  Residual ACF(1): {acf_values[1]:.3f}")
print()
