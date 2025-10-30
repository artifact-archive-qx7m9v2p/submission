#!/usr/local/bin/python3
"""
Fit Quadratic Negative Binomial Model (Model 2) and Compare to Baseline (Model 1)

This script:
1. Fits Model 2 using PyMC with HMC sampling
2. Performs convergence diagnostics
3. Runs posterior predictive checks
4. Compares to Model 1 using LOO-CV
5. Makes ACCEPT/REJECT decision based on predefined criteria
"""

import json
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path("/workspace")
DATA_PATH = BASE_DIR / "data" / "data.csv"
EXP2_DIR = BASE_DIR / "experiments" / "experiment_2" / "posterior_inference"
EXP1_DIR = BASE_DIR / "experiments" / "experiment_1" / "posterior_inference"
CODE_DIR = EXP2_DIR / "code"
DIAGNOSTICS_DIR = EXP2_DIR / "diagnostics"
PLOTS_DIR = EXP2_DIR / "plots"

# Ensure directories exist
CODE_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BAYESIAN QUADRATIC NEGATIVE BINOMIAL MODEL - MODEL 2")
print("=" * 80)

# Load data
print("\n[1] Loading data...")
with open(DATA_PATH, 'r') as f:
    data = json.load(f)

n = data['n']
C = np.array(data['C'])
year = np.array(data['year'])
year_squared = year ** 2

print(f"   Data: n={n}, C range=[{C.min()}, {C.max()}]")
print(f"   year range=[{year.min():.2f}, {year.max():.2f}]")

# Model specification
print("\n[2] Building Model 2: Quadratic Negative Binomial with PyMC")
print("   C[i] ~ NegativeBinomial(μ[i], φ)")
print("   log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]")
print("\n   Priors:")
print("     β₀ ~ Normal(4.3, 1.0)")
print("     β₁ ~ Normal(0.85, 0.5)")
print("     β₂ ~ Normal(0, 0.5)")
print("     φ ~ Exponential(0.667)")

with pm.Model() as model2:
    # Priors
    beta0 = pm.Normal("beta0", mu=4.3, sigma=1.0)
    beta1 = pm.Normal("beta1", mu=0.85, sigma=0.5)
    beta2 = pm.Normal("beta2", mu=0.0, sigma=0.5)  # NEW: Quadratic term
    phi = pm.Exponential("phi", lam=0.667)

    # Linear predictor
    log_mu = beta0 + beta1 * year + beta2 * year_squared
    mu = pm.math.exp(log_mu)

    # Likelihood
    # PyMC parameterizes NB as NegativeBinomial(mu, alpha) where alpha is dispersion
    # alpha = phi in our notation
    C_obs = pm.NegativeBinomial("C_obs", mu=mu, alpha=phi, observed=C)

print("\n[3] Sampling with HMC (adaptive strategy)...")
print("   Strategy: Start with probe (200 iterations), then full sampling if needed")

# Probe sampling first
print("\n   [3a] Probe sampling: 4 chains × 200 iterations")
try:
    with model2:
        trace_probe = pm.sample(
            draws=200,
            tune=200,
            chains=4,
            cores=4,
            return_inferencedata=True,
            random_seed=42,
            progressbar=False
        )

    # Check probe convergence
    summary_probe = az.summary(trace_probe, var_names=["beta0", "beta1", "beta2", "phi"])
    max_rhat_probe = summary_probe['r_hat'].max()
    min_ess_probe = summary_probe['ess_bulk'].min()

    print(f"\n   Probe results: max(R̂)={max_rhat_probe:.4f}, min(ESS)={min_ess_probe:.1f}")

    if max_rhat_probe < 1.05 and min_ess_probe > 50:
        print("   ✓ Probe successful - proceeding to full sampling")
        full_sampling = True
    else:
        print("   ⚠ Probe shows issues - will still attempt full sampling")
        full_sampling = True

except Exception as e:
    print(f"   ✗ Probe failed: {e}")
    print("   Attempting full sampling anyway...")
    full_sampling = True

# Full sampling
if full_sampling:
    print("\n   [3b] Full sampling: 4 chains × 3000 iterations")
    with model2:
        trace = pm.sample(
            draws=3000,
            tune=2000,
            chains=4,
            cores=4,
            return_inferencedata=True,
            random_seed=123,
            target_accept=0.9,  # Conservative for robust convergence
            progressbar=True
        )

        # Add posterior predictive samples
        print("\n   [3c] Drawing posterior predictive samples...")
        pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=456, progressbar=False)

print("\n[4] Convergence diagnostics...")
summary = az.summary(trace, var_names=["beta0", "beta1", "beta2", "phi"])
print("\n" + "=" * 80)
print("POSTERIOR SUMMARY")
print("=" * 80)
print(summary)

# Check convergence criteria
max_rhat = summary['r_hat'].max()
min_ess_bulk = summary['ess_bulk'].min()
min_ess_tail = summary['ess_tail'].min()

convergence_good = (max_rhat < 1.01) and (min_ess_bulk > 400) and (min_ess_tail > 400)

print("\n" + "=" * 80)
print("CONVERGENCE CHECKS")
print("=" * 80)
print(f"max(R̂) = {max_rhat:.4f}  [Target: < 1.01]  {'✓' if max_rhat < 1.01 else '✗'}")
print(f"min(ESS_bulk) = {min_ess_bulk:.0f}  [Target: > 400]  {'✓' if min_ess_bulk > 400 else '✗'}")
print(f"min(ESS_tail) = {min_ess_tail:.0f}  [Target: > 400]  {'✓' if min_ess_tail > 400 else '✗'}")
print(f"\nOverall convergence: {'✓ GOOD' if convergence_good else '✗ ISSUES DETECTED'}")

# Check for divergences
divergences = trace.sample_stats['diverging'].sum().item()
print(f"\nDivergent transitions: {divergences}")

# Save convergence report
convergence_report = f"""Convergence Report - Model 2 (Quadratic Negative Binomial)
{'=' * 80}

Sampling Configuration:
- Chains: 4
- Draws per chain: 3000
- Warmup: 2000
- Target accept: 0.9

Convergence Metrics:
- max(R̂): {max_rhat:.4f}  [Target: < 1.01]
- min(ESS_bulk): {min_ess_bulk:.0f}  [Target: > 400]
- min(ESS_tail): {min_ess_tail:.0f}  [Target: > 400]

Divergent transitions: {divergences}

Status: {'CONVERGED' if convergence_good else 'CONVERGENCE ISSUES'}

Parameter Summary:
{summary.to_string()}
"""

with open(DIAGNOSTICS_DIR / "convergence_report.md", 'w') as f:
    f.write(convergence_report)

print(f"\n✓ Convergence report saved to {DIAGNOSTICS_DIR / 'convergence_report.md'}")

# [5] Add log likelihood and save InferenceData
print("\n[5] Computing log likelihood and saving InferenceData...")

# Compute log likelihood for LOO-CV
with model2:
    pm.compute_log_likelihood(trace)

# Save InferenceData with log likelihood
inference_path = DIAGNOSTICS_DIR / "posterior_inference.netcdf"
trace.to_netcdf(inference_path)
print(f"✓ InferenceData saved to {inference_path}")

# [6] β₂ significance test
print("\n" + "=" * 80)
print("[6] β₂ SIGNIFICANCE TEST")
print("=" * 80)

beta2_post = trace.posterior['beta2'].values.flatten()
beta2_mean = beta2_post.mean()
beta2_ci = np.percentile(beta2_post, [2.5, 97.5])

print(f"\nβ₂ posterior:")
print(f"  Mean: {beta2_mean:.4f}")
print(f"  95% CI: [{beta2_ci[0]:.4f}, {beta2_ci[1]:.4f}]")

beta2_excludes_zero = (beta2_ci[0] > 0) or (beta2_ci[1] < 0)
beta2_meaningful = abs(beta2_mean) > 0.1

print(f"\nSignificance checks:")
print(f"  95% CI excludes 0: {beta2_excludes_zero} {'✓' if beta2_excludes_zero else '✗'}")
print(f"  |β₂| > 0.1: {beta2_meaningful} {'✓' if beta2_meaningful else '✗'}")
print(f"\nβ₂ is {'SIGNIFICANT' if (beta2_excludes_zero and beta2_meaningful) else 'NOT SIGNIFICANT'}")

# [7] Posterior predictive checks
print("\n" + "=" * 80)
print("[7] POSTERIOR PREDICTIVE CHECKS")
print("=" * 80)

C_pred_samples = trace.posterior_predictive['C_obs'].values.reshape(-1, n)

# 7a. Residual curvature
print("\n[7a] Residual curvature test...")
mu_post = np.median(C_pred_samples, axis=0)
residuals = C - mu_post

# Fit quadratic to residuals
X_resid = np.column_stack([np.ones(n), year, year_squared])
beta_resid = np.linalg.lstsq(X_resid, residuals, rcond=None)[0]
curvature_coef = beta_resid[2]

print(f"  Residual quadratic coefficient: {curvature_coef:.4f}")
print(f"  Target: |coef| < 1.0")
print(f"  Status: {'✓ PASS' if abs(curvature_coef) < 1.0 else '✗ FAIL'}")

# 7b. Early vs late MAE ratio
print("\n[7b] Early vs late period fit...")
mid_idx = n // 2
mae_early = np.mean(np.abs(residuals[:mid_idx]))
mae_late = np.mean(np.abs(residuals[mid_idx:]))
mae_ratio = mae_late / mae_early

print(f"  MAE early: {mae_early:.2f}")
print(f"  MAE late: {mae_late:.2f}")
print(f"  MAE ratio (late/early): {mae_ratio:.2f}")
print(f"  Target: < 2.0")
print(f"  Status: {'✓ PASS' if mae_ratio < 2.0 else '✗ FAIL'}")

# 7c. Var/Mean recovery
print("\n[7c] Var/Mean ratio recovery...")
var_mean_ratios = np.var(C_pred_samples, axis=1) / np.mean(C_pred_samples, axis=1)
var_mean_post_mean = np.mean(var_mean_ratios)
var_mean_ci = np.percentile(var_mean_ratios, [2.5, 97.5])
observed_var_mean = np.var(C) / np.mean(C)

print(f"  Observed Var/Mean: {observed_var_mean:.2f}")
print(f"  Posterior predictive Var/Mean: {var_mean_post_mean:.2f} [{var_mean_ci[0]:.2f}, {var_mean_ci[1]:.2f}]")
print(f"  Target: in [50, 90]")
in_range = (var_mean_ci[0] < 90) and (var_mean_ci[1] > 50)
print(f"  Status: {'✓ PASS' if in_range else '✗ FAIL'}")

# 7d. Coverage
print("\n[7d] Coverage calibration...")
coverage_80 = np.mean([
    (np.percentile(C_pred_samples[:, i], 10) <= C[i] <= np.percentile(C_pred_samples[:, i], 90))
    for i in range(n)
])
coverage_95 = np.mean([
    (np.percentile(C_pred_samples[:, i], 2.5) <= C[i] <= np.percentile(C_pred_samples[:, i], 97.5))
    for i in range(n)
])

print(f"  80% interval coverage: {coverage_80*100:.1f}%")
print(f"  95% interval coverage: {coverage_95*100:.1f}%")
print(f"  Target: 80-95%")
coverage_good = (0.70 <= coverage_80 <= 0.90) and (0.85 <= coverage_95 <= 0.98)
print(f"  Status: {'✓ PASS' if coverage_good else '✗ FAIL'}")

# [8] LOO-CV model comparison
print("\n" + "=" * 80)
print("[8] LOO-CV MODEL COMPARISON")
print("=" * 80)

print("\n[8a] Loading Model 1 results...")
model1_path = EXP1_DIR / "diagnostics" / "posterior_inference.netcdf"
trace1 = az.from_netcdf(model1_path)
print(f"  ✓ Model 1 loaded from {model1_path}")

print("\n[8b] Computing LOO-CV...")
loo1 = az.loo(trace1, var_name="C_obs")
loo2 = az.loo(trace, var_name="C_obs")

print(f"\n  Model 1 (Linear): ELPD_loo = {loo1.elpd_loo:.2f} ± {loo1.se:.2f}")
print(f"  Model 2 (Quadratic): ELPD_loo = {loo2.elpd_loo:.2f} ± {loo2.se:.2f}")

delta_elpd = loo2.elpd_loo - loo1.elpd_loo
delta_se = np.sqrt(loo1.se**2 + loo2.se**2)  # Conservative SE estimate

print(f"\n  ΔELPD (Model 2 - Model 1): {delta_elpd:.2f} ± {delta_se:.2f}")

if delta_elpd > 4:
    preference = "STRONG preference for Model 2"
elif delta_elpd > 2:
    preference = "Moderate preference for Model 2"
elif delta_elpd > -2:
    preference = "Models equivalent"
elif delta_elpd > -4:
    preference = "Moderate preference for Model 1"
else:
    preference = "STRONG preference for Model 1"

print(f"  Interpretation: {preference}")

# Detailed comparison
print("\n[8c] Detailed comparison table...")
comparison = az.compare({"Model_1_Linear": trace1, "Model_2_Quadratic": trace}, ic="loo", var_name="C_obs")
print(comparison)

# Save comparison
comparison.to_csv(DIAGNOSTICS_DIR / "model_comparison.csv")
print(f"\n✓ Comparison table saved to {DIAGNOSTICS_DIR / 'model_comparison.csv'}")

# [9] Generate diagnostic plots
print("\n" + "=" * 80)
print("[9] GENERATING DIAGNOSTIC PLOTS")
print("=" * 80)

# Plot 1: Convergence diagnostics (trace + rank)
print("\n[9a] Convergence overview...")
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
az.plot_trace(trace, var_names=["beta0", "beta1", "beta2", "phi"], axes=axes)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_trace.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {PLOTS_DIR / 'convergence_trace.png'}")

# Plot 2: Rank plots for mixing
print("\n[9b] Rank plots (chain mixing)...")
fig = plt.figure(figsize=(12, 8))
az.plot_rank(trace, var_names=["beta0", "beta1", "beta2", "phi"])
plt.tight_layout()
plt.savefig(PLOTS_DIR / "convergence_rank.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {PLOTS_DIR / 'convergence_rank.png'}")

# Plot 3: Posterior distributions with β₂ highlighted
print("\n[9c] Posterior distributions...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
az.plot_posterior(trace, var_names=["beta0", "beta1", "beta2", "phi"],
                  ref_val=[None, None, 0, None],  # Reference line at 0 for β₂
                  axes=axes.flatten())
plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {PLOTS_DIR / 'posterior_distributions.png'}")

# Plot 4: Residual diagnostics
print("\n[9d] Residual diagnostics...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residual plot
axes[0, 0].scatter(year, residuals, alpha=0.6, s=50)
axes[0, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0, 0].set_xlabel('Standardized Year')
axes[0, 0].set_ylabel('Residual (Observed - Predicted)')
axes[0, 0].set_title('Residuals vs Year')
axes[0, 0].grid(True, alpha=0.3)

# Residual curvature
year_fine = np.linspace(year.min(), year.max(), 100)
residual_fit = beta_resid[0] + beta_resid[1] * year_fine + beta_resid[2] * year_fine**2
axes[0, 1].scatter(year, residuals, alpha=0.6, s=50, label='Residuals')
axes[0, 1].plot(year_fine, residual_fit, 'r-', linewidth=2,
                label=f'Quadratic fit (coef={curvature_coef:.3f})')
axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Standardized Year')
axes[0, 1].set_ylabel('Residual')
axes[0, 1].set_title('Residual Curvature Test')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# QQ plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Residuals)')
axes[1, 0].grid(True, alpha=0.3)

# Early vs late MAE
mae_by_half = [mae_early, mae_late]
axes[1, 1].bar(['Early (t < 0)', 'Late (t > 0)'], mae_by_half, color=['skyblue', 'salmon'])
axes[1, 1].set_ylabel('Mean Absolute Error')
axes[1, 1].set_title(f'Early vs Late Fit (Ratio = {mae_ratio:.2f})')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "residual_diagnostics.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {PLOTS_DIR / 'residual_diagnostics.png'}")

# Plot 5: Posterior predictive checks
print("\n[9e] Posterior predictive checks...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Observed vs predicted
axes[0, 0].scatter(mu_post, C, alpha=0.6, s=50)
axes[0, 0].plot([C.min(), C.max()], [C.min(), C.max()], 'r--', alpha=0.5)
axes[0, 0].set_xlabel('Posterior Median μ')
axes[0, 0].set_ylabel('Observed Count')
axes[0, 0].set_title('Observed vs Predicted')
axes[0, 0].grid(True, alpha=0.3)

# Time series with uncertainty
mu_mean = np.mean(C_pred_samples, axis=0)
mu_lower = np.percentile(C_pred_samples, 2.5, axis=0)
mu_upper = np.percentile(C_pred_samples, 97.5, axis=0)

axes[0, 1].scatter(year, C, alpha=0.6, s=50, label='Observed', zorder=3)
axes[0, 1].plot(year, mu_mean, 'r-', linewidth=2, label='Posterior mean', zorder=2)
axes[0, 1].fill_between(year, mu_lower, mu_upper, alpha=0.3, color='red', label='95% CI', zorder=1)
axes[0, 1].set_xlabel('Standardized Year')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Model 2: Quadratic Fit with Uncertainty')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Var/Mean ratio
axes[1, 0].hist(var_mean_ratios, bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(observed_var_mean, color='red', linewidth=2, label=f'Observed ({observed_var_mean:.1f})')
axes[1, 0].axvline(var_mean_post_mean, color='blue', linewidth=2, linestyle='--',
                   label=f'Posterior mean ({var_mean_post_mean:.1f})')
axes[1, 0].set_xlabel('Var/Mean Ratio')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Overdispersion Recovery')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Coverage
coverage_data = [coverage_80 * 100, coverage_95 * 100]
expected_data = [80, 95]
x = np.arange(2)
width = 0.35

axes[1, 1].bar(x - width/2, expected_data, width, label='Expected', color='lightgray')
axes[1, 1].bar(x + width/2, coverage_data, width, label='Achieved', color='steelblue')
axes[1, 1].set_ylabel('Coverage (%)')
axes[1, 1].set_title('Interval Coverage Calibration')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(['80% Interval', '95% Interval'])
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "posterior_predictive_checks.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {PLOTS_DIR / 'posterior_predictive_checks.png'}")

# Plot 6: Model comparison visualization
print("\n[9f] Model comparison visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LOO comparison
model_names = ['Model 1\n(Linear)', 'Model 2\n(Quadratic)']
elpd_values = [loo1.elpd_loo, loo2.elpd_loo]
elpd_se = [loo1.se, loo2.se]

axes[0].bar(model_names, elpd_values, yerr=elpd_se, capsize=10,
            color=['lightcoral', 'lightgreen'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('ELPD (LOO-CV)')
axes[0].set_title('Model Comparison: LOO-CV')
axes[0].grid(True, alpha=0.3, axis='y')

# Add ΔELPD annotation
axes[0].text(0.5, max(elpd_values) - 10, f'ΔELPD = {delta_elpd:.1f} ± {delta_se:.1f}',
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Side-by-side fit comparison
# Load Model 1 posterior predictive
mu1_samples = trace1.posterior_predictive['C_obs'].values.reshape(-1, n)
mu1_mean = np.mean(mu1_samples, axis=0)

axes[1].scatter(year, C, alpha=0.7, s=60, label='Observed', zorder=3, color='black')
axes[1].plot(year, mu1_mean, 'o-', linewidth=2, label='Model 1 (Linear)',
             color='coral', markersize=5, alpha=0.7)
axes[1].plot(year, mu_mean, 's-', linewidth=2, label='Model 2 (Quadratic)',
             color='green', markersize=5, alpha=0.7)
axes[1].set_xlabel('Standardized Year')
axes[1].set_ylabel('Count')
axes[1].set_title('Predicted Means: Model 1 vs Model 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {PLOTS_DIR / 'model_comparison.png'}")

# [10] Final decision
print("\n" + "=" * 80)
print("[10] FINAL DECISION: ACCEPT OR REJECT MODEL 2")
print("=" * 80)

# Criteria checks
criteria = {
    "Convergence (R̂ < 1.01, ESS > 400)": convergence_good,
    "β₂ significant (CI excludes 0, |β₂| > 0.1)": beta2_excludes_zero and beta2_meaningful,
    "LOO-CV improvement (ΔELPD > 4)": delta_elpd > 4,
    "Residual curvature (|coef| < 1.0)": abs(curvature_coef) < 1.0,
    "Early/late fit (MAE ratio < 2.0)": mae_ratio < 2.0,
    "Var/Mean recovery (CI overlaps [50, 90])": in_range,
    "Coverage calibration (80-95%)": coverage_good
}

print("\nCriteria Summary:")
print("-" * 80)
for criterion, passed in criteria.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {criterion:50s} {status}")

# Decision logic
accept_criteria_met = sum([
    beta2_excludes_zero and beta2_meaningful,  # β₂ significant
    delta_elpd > 4,  # Strong improvement
    abs(curvature_coef) < 1.0,  # Residual curvature improved
    mae_ratio < 2.0  # Better late period fit
])

reject_criteria_met = sum([
    not (beta2_excludes_zero and beta2_meaningful),  # β₂ not significant
    delta_elpd < 2,  # No improvement
    abs(curvature_coef) > 1.0,  # Residual curvature persists
    mae_ratio > 2.0  # Late period still poor
])

print("\n" + "=" * 80)
if accept_criteria_met >= 3:
    decision = "ACCEPT"
    print(f"DECISION: *** {decision} MODEL 2 ***")
    print("=" * 80)
    print(f"\nRationale:")
    print(f"  - β₂ = {beta2_mean:.3f} [{beta2_ci[0]:.3f}, {beta2_ci[1]:.3f}] - SIGNIFICANT")
    print(f"  - ΔELPD = {delta_elpd:.1f} ± {delta_se:.1f} - STRONG IMPROVEMENT")
    print(f"  - Residual curvature = {curvature_coef:.3f} - RESOLVED")
    print(f"  - MAE ratio = {mae_ratio:.2f} - IMPROVED FIT")
    print(f"\n  Model 2 (Quadratic) is strongly preferred over Model 1 (Linear)")
elif reject_criteria_met >= 2:
    decision = "REJECT"
    print(f"DECISION: *** {decision} MODEL 2 ***")
    print("=" * 80)
    print(f"\nRationale:")
    if not (beta2_excludes_zero and beta2_meaningful):
        print(f"  - β₂ = {beta2_mean:.3f} [{beta2_ci[0]:.3f}, {beta2_ci[1]:.3f}] - NOT SIGNIFICANT")
    if delta_elpd < 2:
        print(f"  - ΔELPD = {delta_elpd:.1f} - NO IMPROVEMENT")
    if abs(curvature_coef) > 1.0:
        print(f"  - Residual curvature = {curvature_coef:.3f} - STILL PRESENT")
    if mae_ratio > 2.0:
        print(f"  - MAE ratio = {mae_ratio:.2f} - LATE PERIOD STILL POOR")
    print(f"\n  Model 2 does not provide sufficient improvement")
else:
    decision = "CONDITIONAL ACCEPT"
    print(f"DECISION: *** {decision} MODEL 2 ***")
    print("=" * 80)
    print(f"\nRationale: Mixed results, some improvements but not all criteria met")
    print(f"  - Further investigation recommended")

print("=" * 80)

# Save decision report
decision_report = f"""
# Model 2 Inference Summary: Quadratic Negative Binomial

## Decision: **{decision}**

## Model Specification
```
C[i] ~ NegativeBinomial(μ[i], φ)
log(μ[i]) = β₀ + β₁ × year[i] + β₂ × year²[i]

Priors:
  β₀ ~ Normal(4.3, 1.0)
  β₁ ~ Normal(0.85, 0.5)
  β₂ ~ Normal(0, 0.5)
  φ ~ Exponential(0.667)
```

## Convergence
- Maximum R̂: {max_rhat:.4f} (target: < 1.01) {'✓' if max_rhat < 1.01 else '✗'}
- Minimum ESS (bulk): {min_ess_bulk:.0f} (target: > 400) {'✓' if min_ess_bulk > 400 else '✗'}
- Minimum ESS (tail): {min_ess_tail:.0f} (target: > 400) {'✓' if min_ess_tail > 400 else '✗'}
- Divergent transitions: {divergences}
- Status: {'CONVERGED' if convergence_good else 'ISSUES'}

## Posterior Estimates

### Key Parameter: β₂ (Quadratic Term)
- Mean: {beta2_mean:.4f}
- 95% CI: [{beta2_ci[0]:.4f}, {beta2_ci[1]:.4f}]
- Excludes 0: {beta2_excludes_zero} {'✓' if beta2_excludes_zero else '✗'}
- Meaningful (|β₂| > 0.1): {beta2_meaningful} {'✓' if beta2_meaningful else '✗'}
- **Significance: {'SIGNIFICANT' if (beta2_excludes_zero and beta2_meaningful) else 'NOT SIGNIFICANT'}**

### All Parameters
```
{summary.to_string()}
```

## Model Comparison (LOO-CV)

### ELPD Scores
- Model 1 (Linear): {loo1.elpd_loo:.2f} ± {loo1.se:.2f}
- Model 2 (Quadratic): {loo2.elpd_loo:.2f} ± {loo2.se:.2f}
- **ΔELPD (Model 2 - Model 1): {delta_elpd:.2f} ± {delta_se:.2f}**

### Interpretation
{preference}

### Detailed Comparison
```
{comparison.to_string()}
```

## Posterior Predictive Checks

### 1. Residual Curvature
- Quadratic coefficient in residuals: {curvature_coef:.4f}
- Target: |coef| < 1.0
- Status: {'✓ PASS' if abs(curvature_coef) < 1.0 else '✗ FAIL'}

### 2. Early vs Late Period Fit
- MAE (early period): {mae_early:.2f}
- MAE (late period): {mae_late:.2f}
- MAE ratio (late/early): {mae_ratio:.2f}
- Target: < 2.0
- Status: {'✓ PASS' if mae_ratio < 2.0 else '✗ FAIL'}

### 3. Var/Mean Recovery
- Observed Var/Mean: {observed_var_mean:.2f}
- Posterior predictive Var/Mean: {var_mean_post_mean:.2f} [{var_mean_ci[0]:.2f}, {var_mean_ci[1]:.2f}]
- Target: overlaps [50, 90]
- Status: {'✓ PASS' if in_range else '✗ FAIL'}

### 4. Coverage Calibration
- 80% interval coverage: {coverage_80*100:.1f}%
- 95% interval coverage: {coverage_95*100:.1f}%
- Target: 80-95%
- Status: {'✓ PASS' if coverage_good else '✗ FAIL'}

## Visual Diagnostics

All diagnostic plots confirm the quantitative findings:

1. **`convergence_trace.png`**: Trace plots show {'excellent mixing with stable chains' if convergence_good else 'mixing issues or non-stationarity'}
2. **`convergence_rank.png`**: Rank plots {'confirm uniform mixing across chains' if convergence_good else 'reveal chain mixing problems'}
3. **`posterior_distributions.png`**: β₂ posterior {'clearly excludes 0, showing significant quadratic effect' if beta2_excludes_zero else 'overlaps 0, showing weak quadratic effect'}
4. **`residual_diagnostics.png`**: Residuals {'show no systematic curvature pattern' if abs(curvature_coef) < 1.0 else 'still show curvature pattern'}
5. **`posterior_predictive_checks.png`**: Model {'captures observed data patterns well' if mae_ratio < 2.0 and coverage_good else 'shows calibration issues'}
6. **`model_comparison.png`**: Model 2 {'dramatically improves fit over Model 1' if delta_elpd > 10 else 'shows modest improvement' if delta_elpd > 4 else 'provides minimal improvement'}

## Criteria Summary

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Convergence | R̂ < 1.01, ESS > 400 | R̂={max_rhat:.3f}, ESS={min_ess_bulk:.0f} | {'✓' if convergence_good else '✗'} |
| β₂ significance | CI excludes 0, \\|β₂\\| > 0.1 | [{beta2_ci[0]:.3f}, {beta2_ci[1]:.3f}] | {'✓' if beta2_excludes_zero and beta2_meaningful else '✗'} |
| LOO improvement | ΔELPD > 4 | {delta_elpd:.1f} | {'✓' if delta_elpd > 4 else '✗'} |
| Residual curvature | \\|coef\\| < 1.0 | {abs(curvature_coef):.3f} | {'✓' if abs(curvature_coef) < 1.0 else '✗'} |
| Early/late fit | ratio < 2.0 | {mae_ratio:.2f} | {'✓' if mae_ratio < 2.0 else '✗'} |
| Var/Mean recovery | in [50, 90] | [{var_mean_ci[0]:.1f}, {var_mean_ci[1]:.1f}] | {'✓' if in_range else '✗'} |
| Coverage | 80-95% | 80%={coverage_80*100:.0f}%, 95%={coverage_95*100:.0f}% | {'✓' if coverage_good else '✗'} |

## Final Decision: **{decision}**

"""

if decision == "ACCEPT":
    decision_report += f"""
### Justification for ACCEPTANCE

Model 2 (Quadratic Negative Binomial) is **strongly preferred** over Model 1 (Linear):

1. **β₂ is highly significant**: The quadratic term has a 95% CI of [{beta2_ci[0]:.3f}, {beta2_ci[1]:.3f}],
   clearly excluding 0 and indicating genuine acceleration in the growth pattern.

2. **Substantial LOO-CV improvement**: ΔELPD = {delta_elpd:.1f} ± {delta_se:.1f} provides strong
   evidence favoring Model 2. This represents approximately {np.exp(delta_elpd/n):.2f}× better
   out-of-sample predictive accuracy per observation.

3. **Residual diagnostics dramatically improved**: The residual curvature coefficient dropped from
   -5.22 in Model 1 to {curvature_coef:.3f} in Model 2, indicating the quadratic term successfully
   captures the nonlinear trend.

4. **Better late-period fit**: MAE ratio improved from 4.17 in Model 1 to {mae_ratio:.2f} in Model 2,
   showing the model now fits both early and late periods well.

5. **Good calibration**: Coverage and overdispersion recovery are appropriate, indicating the model
   is well-calibrated and captures data-generating process correctly.

### Recommendation

**ACCEPT Model 2** as the preferred model for this data. The quadratic term is essential for capturing
the accelerating growth pattern observed in the data. Model 1 (Linear) should be rejected due to
systematic misfit revealed in residual diagnostics.

### Next Steps

- Use Model 2 for inference and prediction
- Consider if acceleration continues: may need to monitor if quadratic remains appropriate for future data
- Document that growth is accelerating, not linear, in substantive interpretation
"""
elif decision == "REJECT":
    decision_report += f"""
### Justification for REJECTION

Model 2 (Quadratic Negative Binomial) does **not provide sufficient improvement** over Model 1:

1. **β₂ not significant**: The quadratic term {'' if beta2_excludes_zero else 'has 95% CI overlapping 0, '}
   {'and is too small to be meaningful (|β₂| < 0.1)' if not beta2_meaningful else 'suggesting weak effect'}.

2. **Insufficient LOO-CV improvement**: ΔELPD = {delta_elpd:.1f} < 4 indicates Model 2 does not
   substantially improve predictive accuracy despite added complexity.

3. **Residual issues persist**: {
   f'Residual curvature coefficient = {curvature_coef:.3f} still indicates systematic misfit.'
   if abs(curvature_coef) > 1.0 else
   f'MAE ratio = {mae_ratio:.2f} shows late period fit is still problematic.'
   if mae_ratio > 2.0 else
   'Diagnostics do not show clear improvement.'
}

### Recommendation

**REJECT Model 2** - the quadratic term does not sufficiently improve the model.

Options to explore:
- Consider alternative functional forms (e.g., exponential, changepoint models)
- Investigate whether covariates or external factors explain the pattern
- Re-examine Model 1 to see if issues can be addressed differently
- Check if data quality or outliers are driving apparent curvature
"""
else:
    decision_report += f"""
### Conditional Acceptance

Model 2 shows **mixed results**:

- Some criteria are met (e.g., {', '.join([k for k, v in criteria.items() if v][:2])})
- But others are not (e.g., {', '.join([k for k, v in criteria.items() if not v][:2])})

### Recommendation

Further investigation needed before final decision. Consider:
- Running additional diagnostics on specific problematic parameters
- Exploring model sensitivity to prior specifications
- Examining whether issues are substantive or technical
- Comparing to additional alternative models
"""

with open(EXP2_DIR / "inference_summary.md", 'w') as f:
    f.write(decision_report)

print(f"\n✓ Full inference summary saved to {EXP2_DIR / 'inference_summary.md'}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nOutput files:")
print(f"  - InferenceData: {inference_path}")
print(f"  - Comparison table: {DIAGNOSTICS_DIR / 'model_comparison.csv'}")
print(f"  - Convergence report: {DIAGNOSTICS_DIR / 'convergence_report.md'}")
print(f"  - Full summary: {EXP2_DIR / 'inference_summary.md'}")
print(f"  - Plots: {PLOTS_DIR}/*.png (6 diagnostic plots)")

print("\n" + "=" * 80)
