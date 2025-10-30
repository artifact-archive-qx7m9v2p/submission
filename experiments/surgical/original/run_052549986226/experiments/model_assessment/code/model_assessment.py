"""
Comprehensive Model Assessment for Beta-Binomial Model
======================================================

Computes:
- LOO diagnostics (ELPD, Pareto k summaries)
- Calibration metrics (PIT uniformity, interval coverage)
- Absolute predictive metrics (RMSE, MAE, interval width)
- Group-level performance stratified by sample size

Author: Model Assessment Specialist
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("MODEL ASSESSMENT: Beta-Binomial (Reparameterized)")
print("=" * 80)

# Load observed data
data = pd.read_csv('/workspace/data/data.csv')
print(f"\n✓ Loaded observed data: {len(data)} groups")

# Load InferenceData
idata = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
print(f"✓ Loaded InferenceData with {idata.posterior.dims['draw'] * idata.posterior.dims['chain']} samples")

# Load group posterior summaries
group_summary = pd.read_csv('/workspace/experiments/experiment_1/posterior_inference/results/group_posterior_summary.csv')
print(f"✓ Loaded group posterior summaries")

# Load LOO summary
loo_summary = pd.read_csv('/workspace/experiments/experiment_1/posterior_predictive_check/results/loo_summary.csv')
print(f"✓ Loaded LOO Pareto k values")

# ============================================================================
# 1. LOO DIAGNOSTICS
# ============================================================================

print("\n" + "=" * 80)
print("1. LOO CROSS-VALIDATION DIAGNOSTICS")
print("=" * 80)

# Compute LOO
loo = az.loo(idata, pointwise=True)

print(f"\nLOO Results:")
print(f"  ELPD_LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"  p_LOO (effective parameters): {loo.p_loo:.2f}")
print(f"  LOO Information Criterion: {-2 * loo.elpd_loo:.2f}")

# Pareto k diagnostics
pareto_k = loo.pareto_k.values
print(f"\nPareto k Diagnostics:")
print(f"  Mean k: {np.mean(pareto_k):.4f}")
print(f"  Max k: {np.max(pareto_k):.4f} (Group {np.argmax(pareto_k) + 1})")
print(f"  Min k: {np.min(pareto_k):.4f} (Group {np.argmin(pareto_k) + 1})")

k_categories = {
    'k < 0.5 (good)': np.sum(pareto_k < 0.5),
    '0.5 ≤ k < 0.7 (ok)': np.sum((pareto_k >= 0.5) & (pareto_k < 0.7)),
    'k ≥ 0.7 (problematic)': np.sum(pareto_k >= 0.7)
}

print(f"\nPareto k Categories:")
for category, count in k_categories.items():
    print(f"  {category}: {count}/{len(pareto_k)} groups")

# Group-level LOO ELPD
loo_pointwise = loo.elpd_loo.values if hasattr(loo.elpd_loo, 'values') else loo.elpd_loo
if hasattr(loo_pointwise, '__len__') and len(loo_pointwise) > 1:
    print(f"\nGroup-level ELPD:")
    print(f"  Best (highest ELPD): Group {np.argmax(loo_pointwise) + 1} (ELPD = {np.max(loo_pointwise):.2f})")
    print(f"  Worst (lowest ELPD): Group {np.argmin(loo_pointwise) + 1} (ELPD = {np.min(loo_pointwise):.2f})")

# Save LOO results
loo_results = pd.DataFrame({
    'metric': ['elpd_loo', 'se_elpd', 'p_loo', 'looic', 'mean_pareto_k', 'max_pareto_k', 'n_high_k'],
    'value': [
        loo.elpd_loo,
        loo.se,
        loo.p_loo,
        -2 * loo.elpd_loo,
        np.mean(pareto_k),
        np.max(pareto_k),
        np.sum(pareto_k >= 0.7)
    ]
})
loo_results.to_csv('/workspace/experiments/model_assessment/results/loo_diagnostics.csv', index=False)
print(f"\n✓ Saved LOO diagnostics to results/loo_diagnostics.csv")

# ============================================================================
# 2. CALIBRATION ASSESSMENT
# ============================================================================

print("\n" + "=" * 80)
print("2. CALIBRATION ASSESSMENT")
print("=" * 80)

# Extract posterior predictive samples
if 'posterior_predictive' in idata.groups():
    y_rep = idata.posterior_predictive['y_rep'].values.reshape(-1, len(data))
    print(f"\n✓ Extracted {y_rep.shape[0]} posterior predictive samples")
else:
    # Generate posterior predictive if not available
    print("\n⚠ Posterior predictive not in InferenceData, generating...")
    mu = idata.posterior['mu'].values.flatten()
    kappa = idata.posterior['kappa'].values.flatten()

    y_rep = np.zeros((len(mu), len(data)))
    for s in range(len(mu)):
        a = mu[s] * kappa[s]
        b = (1 - mu[s]) * kappa[s]
        for i in range(len(data)):
            p_i = np.random.beta(a, b)
            y_rep[s, i] = np.random.binomial(data.n_trials.iloc[i], p_i)

    print(f"✓ Generated {y_rep.shape[0]} posterior predictive samples")

# Compute coverage at multiple credible levels
credible_levels = [0.50, 0.80, 0.90, 0.95]
coverage_results = []

print(f"\nEmpirical Coverage at Nominal Levels:")
for level in credible_levels:
    alpha = 1 - level
    lower = np.percentile(y_rep, 100 * alpha / 2, axis=0)
    upper = np.percentile(y_rep, 100 * (1 - alpha / 2), axis=0)

    in_interval = (data.r_successes.values >= lower) & (data.r_successes.values <= upper)
    empirical_coverage = np.mean(in_interval)

    coverage_results.append({
        'nominal_level': level,
        'empirical_coverage': empirical_coverage,
        'n_in_interval': np.sum(in_interval),
        'n_total': len(data),
        'calibration_error': empirical_coverage - level
    })

    print(f"  {int(level*100)}% interval: {empirical_coverage:.3f} ({np.sum(in_interval)}/{len(data)} groups)")

coverage_df = pd.DataFrame(coverage_results)
coverage_df.to_csv('/workspace/experiments/model_assessment/results/coverage_analysis.csv', index=False)
print(f"\n✓ Saved coverage analysis to results/coverage_analysis.csv")

# LOO-PIT calibration (use existing if available)
try:
    pit_values = pd.read_csv('/workspace/experiments/experiment_1/posterior_predictive_check/results/pit_values.csv')
    pit = pit_values['pit'].values
    print(f"\n✓ Loaded LOO-PIT values from PPC")
except:
    print(f"\n⚠ Could not load PIT values, will compute in calibration plot")
    pit = None

if pit is not None:
    # KS test for uniformity
    ks_stat, ks_pval = stats.kstest(pit, 'uniform')
    print(f"\nLOO-PIT Uniformity Test:")
    print(f"  Kolmogorov-Smirnov D: {ks_stat:.4f}")
    print(f"  p-value: {ks_pval:.4f}")
    print(f"  Interpretation: {'Well-calibrated' if ks_pval > 0.05 else 'Possible calibration issue'}")

# ============================================================================
# 3. ABSOLUTE PREDICTIVE METRICS
# ============================================================================

print("\n" + "=" * 80)
print("3. ABSOLUTE PREDICTIVE METRICS")
print("=" * 80)

# Use posterior mean as point prediction
predicted = group_summary['posterior_mean'].values * data['n_trials'].values
observed = data['r_successes'].values

# Compute metrics
rmse = np.sqrt(np.mean((observed - predicted) ** 2))
mae = np.mean(np.abs(observed - predicted))

# Rate-based metrics (more interpretable)
predicted_rates = group_summary['posterior_mean'].values
observed_rates = data['success_rate'].values

rmse_rates = np.sqrt(np.mean((observed_rates - predicted_rates) ** 2))
mae_rates = np.mean(np.abs(observed_rates - predicted_rates))

print(f"\nPoint Prediction Metrics (counts):")
print(f"  RMSE: {rmse:.3f} successes")
print(f"  MAE: {mae:.3f} successes")

print(f"\nPoint Prediction Metrics (rates):")
print(f"  RMSE: {rmse_rates:.4f} ({rmse_rates*100:.2f}%)")
print(f"  MAE: {mae_rates:.4f} ({mae_rates*100:.2f}%)")

# Interval metrics (90% credible intervals)
ci_90_lower = np.percentile(y_rep, 5, axis=0)
ci_90_upper = np.percentile(y_rep, 95, axis=0)
interval_widths = ci_90_upper - ci_90_lower
avg_interval_width = np.mean(interval_widths)

# Convert to rates
interval_widths_rates = interval_widths / data['n_trials'].values
avg_interval_width_rates = np.mean(interval_widths_rates)

print(f"\nInterval Metrics (90% CIs):")
print(f"  Average width (counts): {avg_interval_width:.2f} successes")
print(f"  Average width (rates): {avg_interval_width_rates:.4f} ({avg_interval_width_rates*100:.2f}%)")

# Coverage already computed above
coverage_90 = coverage_df[coverage_df['nominal_level'] == 0.90]['empirical_coverage'].values[0]
print(f"  Empirical coverage: {coverage_90:.3f} (target: 0.90)")

# Log score (from LOO)
log_score = loo.elpd_loo
print(f"\nLog Score (ELPD_LOO): {log_score:.2f} ± {loo.se:.2f}")

# Save absolute metrics
abs_metrics = pd.DataFrame({
    'metric': ['rmse_counts', 'mae_counts', 'rmse_rates', 'mae_rates',
               'avg_ci_width_counts', 'avg_ci_width_rates', 'coverage_90', 'elpd_loo'],
    'value': [rmse, mae, rmse_rates, mae_rates,
              avg_interval_width, avg_interval_width_rates, coverage_90, log_score]
})
abs_metrics.to_csv('/workspace/experiments/model_assessment/results/absolute_metrics.csv', index=False)
print(f"\n✓ Saved absolute metrics to results/absolute_metrics.csv")

# ============================================================================
# 4. METRICS BY GROUP SIZE
# ============================================================================

print("\n" + "=" * 80)
print("4. PERFORMANCE STRATIFIED BY GROUP SIZE")
print("=" * 80)

# Create sample size categories
n_trials = data['n_trials'].values
size_categories = []
for n in n_trials:
    if n < 100:
        size_categories.append('Small (n<100)')
    elif n < 200:
        size_categories.append('Medium (100≤n<200)')
    else:
        size_categories.append('Large (n≥200)')

data['size_category'] = size_categories

# Compute metrics by size
size_metrics = []
for cat in ['Small (n<100)', 'Medium (100≤n<200)', 'Large (n≥200)']:
    mask = np.array(size_categories) == cat
    if np.sum(mask) > 0:
        rmse_cat = np.sqrt(np.mean((observed_rates[mask] - predicted_rates[mask]) ** 2))
        mae_cat = np.mean(np.abs(observed_rates[mask] - predicted_rates[mask]))
        mean_width = np.mean(interval_widths_rates[mask])

        size_metrics.append({
            'size_category': cat,
            'n_groups': np.sum(mask),
            'rmse_rates': rmse_cat,
            'mae_rates': mae_cat,
            'avg_ci_width': mean_width
        })

        print(f"\n{cat} ({np.sum(mask)} groups):")
        print(f"  RMSE: {rmse_cat:.4f} ({rmse_cat*100:.2f}%)")
        print(f"  MAE: {mae_cat:.4f} ({mae_cat*100:.2f}%)")
        print(f"  Avg CI width: {mean_width:.4f} ({mean_width*100:.2f}%)")

size_metrics_df = pd.DataFrame(size_metrics)
size_metrics_df.to_csv('/workspace/experiments/model_assessment/results/metrics_by_size.csv', index=False)
print(f"\n✓ Saved size-stratified metrics to results/metrics_by_size.csv")

# ============================================================================
# 5. GROUP-LEVEL DETAILED RESULTS
# ============================================================================

# Combine all group-level information
group_results = data[['group', 'n_trials', 'r_successes', 'success_rate']].copy()
group_results['predicted_rate'] = predicted_rates
group_results['abs_error'] = np.abs(observed_rates - predicted_rates)
group_results['squared_error'] = (observed_rates - predicted_rates) ** 2
group_results['ci_90_lower'] = ci_90_lower / data['n_trials'].values
group_results['ci_90_upper'] = ci_90_upper / data['n_trials'].values
group_results['ci_90_width'] = interval_widths_rates
group_results['in_ci_90'] = (observed >= ci_90_lower) & (observed <= ci_90_upper)
group_results['pareto_k'] = pareto_k
group_results['size_category'] = size_categories
group_results['shrinkage_pct'] = group_summary['shrinkage_pct'].values

group_results.to_csv('/workspace/experiments/model_assessment/results/group_level_metrics.csv', index=False)
print(f"\n✓ Saved group-level metrics to results/group_level_metrics.csv")

print("\n" + "=" * 80)
print("✓ ALL METRICS COMPUTED SUCCESSFULLY")
print("=" * 80)
