"""
Comprehensive Model Comparison: Fixed-Effect vs Random-Effects Meta-Analysis
=============================================================================

Compares Model 1 (Fixed-Effect) and Model 2 (Random-Effects Hierarchical)
using LOO-CV, calibration checks, and predictive metrics.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = '/workspace/data/data.csv'
MODEL1_PATH = "/workspace/experiments/model_comparison/idata1_with_predictions.netcdf"
MODEL2_PATH = "/workspace/experiments/model_comparison/idata2_with_predictions.netcdf"
OUTPUT_DIR = '/workspace/experiments/model_comparison'
PLOTS_DIR = f'{OUTPUT_DIR}/plots'

print("="*80)
print("COMPREHENSIVE BAYESIAN MODEL COMPARISON")
print("="*80)
print()

# ============================================================================
# 1. LOAD DATA AND MODELS
# ============================================================================

print("1. Loading data and fitted models...")
print("-" * 80)

# Load data
data = pd.read_csv(DATA_PATH)
y_obs = data['y'].values
sigma = data['sigma'].values
n_studies = len(y_obs)

print(f"   Studies: {n_studies}")
print(f"   Observations: {y_obs}")
print(f"   Standard errors: {sigma}")
print()

# Load InferenceData
idata1 = az.from_netcdf(MODEL1_PATH)
idata2 = az.from_netcdf(MODEL2_PATH)

print("   Model 1 (Fixed-Effect): Loaded")
print(f"   - Posterior shape: {idata1.posterior.dims}")
print(f"   - Groups: {list(idata1.groups())}")

print("   Model 2 (Random-Effects): Loaded")
print(f"   - Posterior shape: {idata2.posterior.dims}")
print(f"   - Groups: {list(idata2.groups())}")
print()

# Verify log_likelihood exists
assert 'log_likelihood' in idata1.groups(), "Model 1 missing log_likelihood!"
assert 'log_likelihood' in idata2.groups(), "Model 2 missing log_likelihood!"

# ============================================================================
# 2. LOO-CV COMPARISON (PRIMARY ANALYSIS)
# ============================================================================

print("2. LOO-CV Model Comparison")
print("-" * 80)

# Compute LOO for each model
loo1 = az.loo(idata1, pointwise=True)
loo2 = az.loo(idata2, pointwise=True)

print("   Model 1 (Fixed-Effect):")
print(f"   - ELPD LOO: {loo1.elpd_loo:.2f} ± {loo1.se:.2f}")
print(f"   - p_LOO: {loo1.p_loo:.2f}")
pareto_k1 = loo1.pareto_k.values if hasattr(loo1.pareto_k, 'values') else loo1.pareto_k
print(f"   - Pareto k > 0.7: {np.sum(pareto_k1 > 0.7)}/{n_studies}")
print(f"   - Pareto k > 0.5: {np.sum(pareto_k1 > 0.5)}/{n_studies}")
print()

print("   Model 2 (Random-Effects):")
print(f"   - ELPD LOO: {loo2.elpd_loo:.2f} ± {loo2.se:.2f}")
print(f"   - p_LOO: {loo2.p_loo:.2f}")
pareto_k2 = loo2.pareto_k.values if hasattr(loo2.pareto_k, 'values') else loo2.pareto_k
print(f"   - Pareto k > 0.7: {np.sum(pareto_k2 > 0.7)}/{n_studies}")
print(f"   - Pareto k > 0.5: {np.sum(pareto_k2 > 0.5)}/{n_studies}")
print()

# Compare models
comparison = az.compare({'Model 1 (Fixed)': idata1, 'Model 2 (Random)': idata2})
print("   Model Comparison Table:")
print(comparison)
print()

# Extract comparison metrics
delta_elpd = comparison.iloc[1]['elpd_loo'] - comparison.iloc[0]['elpd_loo']
delta_se = comparison.iloc[1]['dse']
ratio = abs(delta_elpd / delta_se) if delta_se > 0 else 0

print(f"   ΔELPD (Model 2 - Model 1): {delta_elpd:.2f} ± {delta_se:.2f}")
print(f"   Ratio |ΔELPD/SE|: {ratio:.2f}")
print()

if ratio > 2:
    print(f"   *** MODELS ARE DISTINGUISHABLE (|ΔELPD/SE| > 2) ***")
    if delta_elpd > 0:
        print(f"   --> Model 1 (Fixed) is preferred")
    else:
        print(f"   --> Model 2 (Random) is preferred")
else:
    print(f"   *** NO SUBSTANTIAL DIFFERENCE (|ΔELPD/SE| < 2) ***")
    print(f"   --> Prefer simpler model (Model 1) by parsimony")
print()

# Save comparison table
comparison.to_csv(f'{OUTPUT_DIR}/loo_comparison_table.csv')

# ============================================================================
# 3. CALIBRATION ASSESSMENT
# ============================================================================

print("3. Calibration Assessment")
print("-" * 80)

# A. LOO-PIT (Probability Integral Transform)
print("   A. LOO-PIT Analysis")

try:
    loo_pit1 = az.loo_pit(idata1, y='y_obs')
    loo_pit2 = az.loo_pit(idata2, y='y_obs')

    # Test uniformity using Kolmogorov-Smirnov test
    ks_stat1, ks_pval1 = stats.kstest(loo_pit1.values.flatten(), 'uniform')
    ks_stat2, ks_pval2 = stats.kstest(loo_pit2.values.flatten(), 'uniform')

    print(f"   Model 1: KS stat = {ks_stat1:.3f}, p-value = {ks_pval1:.3f}")
    print(f"   Model 2: KS stat = {ks_stat2:.3f}, p-value = {ks_pval2:.3f}")
except Exception as e:
    print(f"   Warning: LOO-PIT computation failed: {e}")
    print(f"   Skipping LOO-PIT analysis")
    ks_stat1, ks_pval1 = np.nan, np.nan
    ks_stat2, ks_pval2 = np.nan, np.nan

print()

# B. Coverage Analysis
print("   B. Posterior Predictive Coverage")

def compute_coverage(idata, y_obs, intervals=[0.5, 0.9, 0.95]):
    """Compute empirical coverage of posterior predictive intervals."""

    # Get posterior predictive samples
    y_pred = idata.posterior_predictive['y_obs'].values.reshape(-1, len(y_obs))

    results = {}
    for interval in intervals:
        lower = (1 - interval) / 2 * 100
        upper = (1 + interval) / 2 * 100

        ci_lower = np.percentile(y_pred, lower, axis=0)
        ci_upper = np.percentile(y_pred, upper, axis=0)

        in_interval = (y_obs >= ci_lower) & (y_obs <= ci_upper)
        empirical_coverage = np.mean(in_interval)

        # Average interval width
        avg_width = np.mean(ci_upper - ci_lower)

        results[interval] = {
            'nominal': interval,
            'empirical': empirical_coverage,
            'width': avg_width,
            'in_interval': in_interval
        }

        print(f"      {int(interval*100)}% interval: Empirical coverage = {empirical_coverage:.1%}, Avg width = {avg_width:.1f}")

    return results

coverage1 = compute_coverage(idata1, y_obs)
print()
coverage2 = compute_coverage(idata2, y_obs)
print()

# C. Sharpness comparison
print("   C. Sharpness (Average 95% CI width)")
width1 = coverage1[0.95]['width']
width2 = coverage2[0.95]['width']
print(f"   Model 1: {width1:.2f}")
print(f"   Model 2: {width2:.2f}")
print(f"   Difference: {width2 - width1:.2f} ({(width2/width1 - 1)*100:.1f}% wider for Model 2)")
print()

# ============================================================================
# 4. PREDICTIVE METRICS
# ============================================================================

print("4. Predictive Performance Metrics")
print("-" * 80)

# A. Point Predictions
def get_predictions(idata, y_obs):
    """Extract posterior predictive means and residuals."""
    y_pred = idata.posterior_predictive['y_obs'].values.reshape(-1, len(y_obs))
    y_pred_mean = y_pred.mean(axis=0)
    y_pred_std = y_pred.std(axis=0)
    residuals = y_obs - y_pred_mean
    return y_pred_mean, y_pred_std, residuals

y_pred1, y_std1, resid1 = get_predictions(idata1, y_obs)
y_pred2, y_std2, resid2 = get_predictions(idata2, y_obs)

print("   A. Point Predictions")
pred_df = pd.DataFrame({
    'Study': range(1, n_studies+1),
    'y_obs': y_obs,
    'sigma': sigma,
    'Model1_pred': y_pred1,
    'Model1_std': y_std1,
    'Model2_pred': y_pred2,
    'Model2_std': y_std2,
    'Model1_resid': resid1,
    'Model2_resid': resid2
})
print(pred_df.to_string(index=False))
pred_df.to_csv(f'{OUTPUT_DIR}/predictions_comparison.csv', index=False)
print()

# B. RMSE and MAE
print("   B. Error Metrics")

def compute_metrics(y_obs, y_pred, sigma):
    """Compute RMSE, MAE, and standardized versions."""
    resid = y_obs - y_pred
    rmse = np.sqrt(np.mean(resid**2))
    mae = np.mean(np.abs(resid))

    # Standardized by sigma
    std_resid = resid / sigma
    std_rmse = np.sqrt(np.mean(std_resid**2))
    std_mae = np.mean(np.abs(std_resid))

    return {
        'RMSE': rmse,
        'MAE': mae,
        'Standardized RMSE': std_rmse,
        'Standardized MAE': std_mae
    }

metrics1 = compute_metrics(y_obs, y_pred1, sigma)
metrics2 = compute_metrics(y_obs, y_pred2, sigma)

metrics_df = pd.DataFrame({
    'Metric': list(metrics1.keys()),
    'Model 1 (Fixed)': list(metrics1.values()),
    'Model 2 (Random)': list(metrics2.values())
})
metrics_df['Difference'] = metrics_df['Model 2 (Random)'] - metrics_df['Model 1 (Fixed)']

print(metrics_df.to_string(index=False))
metrics_df.to_csv(f'{OUTPUT_DIR}/predictive_metrics.csv', index=False)
print()

# ============================================================================
# 5. PARAMETER COMPARISON
# ============================================================================

print("5. Parameter Comparison")
print("-" * 80)

# Model 1: theta
theta1 = idata1.posterior['theta'].values.flatten()
theta1_mean = theta1.mean()
theta1_std = theta1.std()
theta1_hdi = az.hdi(idata1, var_names=['theta'], hdi_prob=0.95)['theta'].values

print("   Model 1 (Fixed-Effect):")
print(f"   - θ = {theta1_mean:.2f} ± {theta1_std:.2f}")
print(f"   - 95% HDI: [{theta1_hdi[0]:.2f}, {theta1_hdi[1]:.2f}]")
print()

# Model 2: mu and tau
mu2 = idata2.posterior['mu'].values.flatten()
mu2_mean = mu2.mean()
mu2_std = mu2.std()
mu2_hdi = az.hdi(idata2, var_names=['mu'], hdi_prob=0.95)['mu'].values

tau2 = idata2.posterior['tau'].values.flatten()
tau2_mean = tau2.mean()
tau2_std = tau2.std()
tau2_hdi = az.hdi(idata2, var_names=['tau'], hdi_prob=0.95)['tau'].values

print("   Model 2 (Random-Effects):")
print(f"   - μ = {mu2_mean:.2f} ± {mu2_std:.2f}")
print(f"   - 95% HDI: [{mu2_hdi[0]:.2f}, {mu2_hdi[1]:.2f}]")
print(f"   - τ = {tau2_mean:.2f} ± {tau2_std:.2f}")
print(f"   - 95% HDI: [{tau2_hdi[0]:.2f}, {tau2_hdi[1]:.2f}]")
print()

# Compare point estimates
print("   Comparison of Overall Effect:")
print(f"   - Difference (μ - θ): {mu2_mean - theta1_mean:.2f}")
print(f"   - Relative difference: {(mu2_mean / theta1_mean - 1) * 100:.1f}%")
print(f"   - Uncertainty ratio (SD_μ / SD_θ): {mu2_std / theta1_std:.2f}")
print()

# Study-specific estimates (Model 2)
print("   Model 2 Study-Specific Estimates (θ_i):")
theta_i_mean = idata2.posterior['theta'].values.mean(axis=(0,1))
theta_i_std = idata2.posterior['theta'].values.std(axis=(0,1))

for i in range(n_studies):
    shrinkage = (y_obs[i] - theta_i_mean[i]) / (y_obs[i] - mu2_mean) if abs(y_obs[i] - mu2_mean) > 0.01 else 0
    print(f"   - Study {i+1}: θ_{i+1} = {theta_i_mean[i]:6.2f} ± {theta_i_std[i]:.2f} (shrinkage: {(1-shrinkage)*100:5.1f}%)")
print()

# ============================================================================
# 6. PARSIMONY ANALYSIS
# ============================================================================

print("6. Parsimony Analysis: Complexity vs Fit Trade-off")
print("-" * 80)

print("   Model Complexity:")
print(f"   - Model 1: 1 parameter (θ)")
print(f"   - Model 2: 10 parameters (μ, τ, θ_1...θ_8)")
print()

print("   Effective Complexity (from LOO):")
print(f"   - Model 1: p_LOO = {loo1.p_loo:.2f}")
print(f"   - Model 2: p_LOO = {loo2.p_loo:.2f}")
print()

print("   Interpretation:")
if loo2.p_loo < 10:
    print(f"   - Model 2 has effective complexity of {loo2.p_loo:.1f} (< 10 actual parameters)")
    print(f"   - This indicates strong shrinkage/regularization")

complexity_gain = loo2.p_loo - loo1.p_loo
performance_gain = delta_elpd

print(f"   - Additional complexity: {complexity_gain:.2f} effective parameters")
print(f"   - Performance gain: {performance_gain:.2f} ELPD units")
print()

if abs(performance_gain) < 2 * delta_se:
    print("   *** Complexity NOT justified: No meaningful performance gain ***")
else:
    print("   *** Complexity justified: Meaningful performance improvement ***")
print()

# ============================================================================
# 7. SENSITIVITY AND ROBUSTNESS
# ============================================================================

print("7. Sensitivity and Robustness Analysis")
print("-" * 80)

# A. Influential Observations
print("   A. Influential Observations")

influence_df = pd.DataFrame({
    'Study': range(1, n_studies+1),
    'y_obs': y_obs,
    'sigma': sigma,
    'Model1_k': pareto_k1,
    'Model2_k': pareto_k2
})

print(influence_df.to_string(index=False))
influence_df.to_csv(f'{OUTPUT_DIR}/influence_diagnostics.csv', index=False)
print()

# Identify problematic observations
problematic1 = np.where(pareto_k1 > 0.7)[0]
problematic2 = np.where(pareto_k2 > 0.7)[0]

if len(problematic1) > 0:
    print(f"   Model 1: {len(problematic1)} studies with k > 0.7: {list(problematic1 + 1)}")
else:
    print("   Model 1: No problematic observations (all k < 0.7)")

if len(problematic2) > 0:
    print(f"   Model 2: {len(problematic2)} studies with k > 0.7: {list(problematic2 + 1)}")
else:
    print("   Model 2: No problematic observations (all k < 0.7)")
print()

# B. Model Agreement
print("   B. Model Agreement")

pred_corr = np.corrcoef(y_pred1, y_pred2)[0, 1]
print(f"   - Correlation of predictions: {pred_corr:.3f}")

pred_diff = y_pred2 - y_pred1
max_diff_idx = np.argmax(np.abs(pred_diff))

print(f"   - Max prediction difference: {pred_diff[max_diff_idx]:.2f} (Study {max_diff_idx+1})")
print(f"   - Mean absolute difference: {np.mean(np.abs(pred_diff)):.2f}")
print()

# Where do models disagree most?
print("   Largest Prediction Differences:")
diff_ranking = np.argsort(np.abs(pred_diff))[::-1]
for idx in diff_ranking[:3]:
    print(f"   - Study {idx+1}: Model 2 - Model 1 = {pred_diff[idx]:+.2f}")

print()

# ============================================================================
# 8. SAVE NUMERICAL RESULTS
# ============================================================================

print("8. Saving Numerical Results")
print("-" * 80)

results = {
    'loo_comparison': {
        'model1_elpd': float(loo1.elpd_loo),
        'model1_se': float(loo1.se),
        'model1_p_loo': float(loo1.p_loo),
        'model2_elpd': float(loo2.elpd_loo),
        'model2_se': float(loo2.se),
        'model2_p_loo': float(loo2.p_loo),
        'delta_elpd': float(delta_elpd),
        'delta_se': float(delta_se),
        'ratio': float(ratio)
    },
    'calibration': {
        'model1_ks_stat': float(ks_stat1) if not np.isnan(ks_stat1) else None,
        'model1_ks_pval': float(ks_pval1) if not np.isnan(ks_pval1) else None,
        'model2_ks_stat': float(ks_stat2) if not np.isnan(ks_stat2) else None,
        'model2_ks_pval': float(ks_pval2) if not np.isnan(ks_pval2) else None
    },
    'parameters': {
        'model1_theta': float(theta1_mean),
        'model1_theta_std': float(theta1_std),
        'model2_mu': float(mu2_mean),
        'model2_mu_std': float(mu2_std),
        'model2_tau': float(tau2_mean),
        'model2_tau_std': float(tau2_std)
    }
}

with open(f'{OUTPUT_DIR}/comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"   - Saved to: {OUTPUT_DIR}/comparison_results.json")
print()

print("="*80)
print("NUMERICAL ANALYSIS COMPLETE")
print("="*80)
print()
print("Next: Run visualization script to generate comparison plots")
