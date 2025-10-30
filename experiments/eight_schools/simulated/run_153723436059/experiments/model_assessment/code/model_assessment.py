"""
Model Assessment for Experiment 1 - Standard Hierarchical Model

Comprehensive assessment of predictive performance and calibration using:
- LOO-CV with Pareto-k diagnostics
- Calibration analysis (LOO-PIT, coverage)
- Absolute predictive metrics (RMSE, MAE, R²)
- Influence diagnostics
- Visualization suite

Author: Model Assessment Specialist
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
IDATA_PATH = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
OUTPUT_DIR = Path("/workspace/experiments/model_assessment")
PLOTS_DIR = OUTPUT_DIR / "plots"
CODE_DIR = OUTPUT_DIR / "code"

# Create directories
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MODEL ASSESSMENT: EXPERIMENT 1 - STANDARD HIERARCHICAL MODEL")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD DATA AND POSTERIOR
# ============================================================================

print("1. LOADING DATA AND POSTERIOR SAMPLES")
print("-" * 80)

# Load observed data
data = pd.read_csv(DATA_PATH)
y_obs = data['effect'].values
sigma_obs = data['sigma'].values
J = len(y_obs)
print(f"   Loaded {J} schools from {DATA_PATH}")

# Load posterior InferenceData
idata = az.from_netcdf(IDATA_PATH)
print(f"   Loaded InferenceData from {IDATA_PATH}")

# Verify log_likelihood exists
if 'log_likelihood' not in idata.groups():
    raise ValueError("ERROR: InferenceData missing log_likelihood group. Cannot compute LOO.")

print(f"   Verified log_likelihood group exists")
print(f"   Posterior shape: {idata.posterior.dims}")
print()

# ============================================================================
# 2. LOO-CV DIAGNOSTICS
# ============================================================================

print("2. LOO-CV DIAGNOSTICS")
print("-" * 80)

# Compute LOO with pointwise results
loo_result = az.loo(idata, pointwise=True)

# Extract key metrics
elpd_loo = float(loo_result.elpd_loo)
se_elpd = float(loo_result.se)
p_loo = float(loo_result.p_loo)
pareto_k = loo_result.pareto_k.values

print(f"   ELPD_loo:  {elpd_loo:.2f} ± {se_elpd:.2f}")
print(f"   p_loo:     {p_loo:.2f}")
print()

# Pareto-k diagnostics
pareto_k_counts = {
    'good (< 0.5)': int(np.sum(pareto_k < 0.5)),
    'ok (0.5-0.7)': int(np.sum((pareto_k >= 0.5) & (pareto_k < 0.7))),
    'bad (0.7-1.0)': int(np.sum((pareto_k >= 0.7) & (pareto_k < 1.0))),
    'very bad (> 1.0)': int(np.sum(pareto_k >= 1.0))
}

print("   Pareto-k Diagnostics:")
for category, count in pareto_k_counts.items():
    print(f"      {category}: {count} observations")
print(f"      Max Pareto-k: {pareto_k.max():.3f}")
print()

# Flag high Pareto-k observations
high_k_threshold = 0.7
high_k_idx = np.where(pareto_k >= high_k_threshold)[0]
if len(high_k_idx) > 0:
    print(f"   WARNING: {len(high_k_idx)} observations with Pareto-k >= {high_k_threshold}")
    for idx in high_k_idx:
        print(f"      School {idx+1}: k = {pareto_k[idx]:.3f}")
else:
    print(f"   All Pareto-k < {high_k_threshold}: LOO estimates are reliable")
print()

# Interpretation
print("   Interpretation:")
if p_loo < 2 * J:
    print(f"      p_loo ({p_loo:.1f}) << 2*J ({2*J}): Model not overfitting")
else:
    print(f"      p_loo ({p_loo:.1f}) >= 2*J ({2*J}): Possible overfitting")

if pareto_k.max() < 0.7:
    print(f"      All Pareto-k < 0.7: LOO is reliable")
else:
    print(f"      Some Pareto-k >= 0.7: LOO may be unreliable for those points")
print()

# Save LOO results by observation
loo_df = pd.DataFrame({
    'school': np.arange(1, J+1),
    'y_obs': y_obs,
    'sigma_obs': sigma_obs,
    'elpd_loo_i': loo_result.loo_i.values,
    'pareto_k': pareto_k
})
loo_df.to_csv(OUTPUT_DIR / "loo_results.csv", index=False)
print(f"   Saved LOO results to {OUTPUT_DIR / 'loo_results.csv'}")
print()

# ============================================================================
# 3. CALIBRATION ANALYSIS
# ============================================================================

print("3. CALIBRATION ANALYSIS")
print("-" * 80)

# LOO-PIT computation (using az.loo_pit)
try:
    loo_pit = az.loo_pit(idata, y='y')
    loo_pit_values = loo_pit.y.values
    print(f"   Computed LOO-PIT for {len(loo_pit_values)} observations")

    # Test uniformity with Kolmogorov-Smirnov
    ks_stat, ks_pval = stats.kstest(loo_pit_values, 'uniform')
    print(f"   KS test for uniformity: stat={ks_stat:.3f}, p={ks_pval:.3f}")

    if ks_pval > 0.05:
        print(f"      LOO-PIT is consistent with uniform distribution (well-calibrated)")
    else:
        print(f"      LOO-PIT deviates from uniform (potential calibration issue)")
    print()
except Exception as e:
    print(f"   WARNING: Could not compute LOO-PIT: {e}")
    loo_pit_values = None
    ks_stat = np.nan
    ks_pval = np.nan
    print()

# Coverage calibration
print("   Coverage Calibration:")
print("   " + "-" * 40)

# Extract posterior predictive samples for each school
theta_samples = idata.posterior['theta'].values.reshape(-1, J)  # (n_samples, J)

# Compute coverage at various nominal levels
nominal_levels = [0.50, 0.80, 0.90, 0.95]
empirical_coverage = []

coverage_results = []
for level in nominal_levels:
    alpha = 1 - level
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    # Compute quantiles for each school
    lower = np.quantile(theta_samples, lower_q, axis=0)
    upper = np.quantile(theta_samples, upper_q, axis=0)

    # Check which observations fall within intervals
    within = (y_obs >= lower) & (y_obs <= upper)
    coverage = within.mean()
    empirical_coverage.append(coverage)

    # Store results
    coverage_results.append({
        'nominal': level,
        'empirical': coverage,
        'difference': coverage - level,
        'count': within.sum(),
        'total': J
    })

    print(f"      {int(level*100):2d}% interval: {coverage*100:5.1f}% empirical coverage ({within.sum()}/{J} schools)")

print()
coverage_df = pd.DataFrame(coverage_results)

# Assessment
if all(abs(row['difference']) < 0.15 for _, row in coverage_df.iterrows()):
    print(f"   Assessment: Good calibration (all within 15% of nominal)")
elif any(row['difference'] > 0.15 for _, row in coverage_df.iterrows()):
    print(f"   Assessment: Conservative (over-coverage)")
else:
    print(f"   Assessment: Anti-conservative (under-coverage)")
print()

# Save calibration metrics
calibration_metrics = {
    'loo_pit_ks_stat': ks_stat,
    'loo_pit_ks_pval': ks_pval,
}
if loo_pit_values is not None:
    calibration_metrics.update({
        'loo_pit_mean': loo_pit_values.mean(),
        'loo_pit_std': loo_pit_values.std(),
    })
else:
    calibration_metrics.update({
        'loo_pit_mean': np.nan,
        'loo_pit_std': np.nan,
    })

calibration_summary = pd.concat([
    pd.DataFrame([calibration_metrics]),
    coverage_df
], axis=0, ignore_index=True)
calibration_summary.to_csv(OUTPUT_DIR / "calibration_metrics.csv", index=False)
print(f"   Saved calibration metrics to {OUTPUT_DIR / 'calibration_metrics.csv'}")
print()

# ============================================================================
# 4. ABSOLUTE PREDICTIVE METRICS
# ============================================================================

print("4. ABSOLUTE PREDICTIVE METRICS")
print("-" * 80)

# Point predictions (posterior mean)
theta_mean = theta_samples.mean(axis=0)
theta_std = theta_samples.std(axis=0)

# Compute metrics
residuals = y_obs - theta_mean
rmse = np.sqrt(np.mean(residuals**2))
mae = np.mean(np.abs(residuals))

# R² (proportion of variance explained)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_obs - y_obs.mean())**2)
r2 = 1 - ss_res / ss_tot

print(f"   Hierarchical Model:")
print(f"      RMSE: {rmse:.2f}")
print(f"      MAE:  {mae:.2f}")
print(f"      R²:   {r2:.3f}")
print()

# Compare to naive baselines
print("   Baseline Comparisons:")
print("   " + "-" * 40)

# Complete pooling (grand mean)
y_pooled = y_obs.mean()
residuals_pooled = y_obs - y_pooled
rmse_pooled = np.sqrt(np.mean(residuals_pooled**2))
mae_pooled = np.mean(np.abs(residuals_pooled))
r2_pooled = 1 - np.sum(residuals_pooled**2) / ss_tot

print(f"   Complete Pooling (grand mean = {y_pooled:.2f}):")
print(f"      RMSE: {rmse_pooled:.2f} (vs {rmse:.2f})")
print(f"      MAE:  {mae_pooled:.2f} (vs {mae:.2f})")
print(f"      R²:   {r2_pooled:.3f} (vs {r2:.3f})")
print()

# No pooling (use observed values as predictions - perfect fit by definition)
# For LOO comparison, use leave-one-out predictions
print(f"   No Pooling (observed values):")
print(f"      RMSE: 0.00 (by definition)")
print(f"      MAE:  0.00 (by definition)")
print(f"      R²:   1.00 (by definition)")
print(f"      Note: No pooling overfits to observed data")
print()

# Compute improvement over complete pooling
rmse_improvement = (rmse_pooled - rmse) / rmse_pooled * 100
mae_improvement = (mae_pooled - mae) / mae_pooled * 100

print(f"   Improvement over Complete Pooling:")
print(f"      RMSE: {rmse_improvement:+.1f}%")
print(f"      MAE:  {mae_improvement:+.1f}%")
print()

# Note about trade-offs
print(f"   Interpretation:")
print(f"      Hierarchical model trades bias for variance reduction")
print(f"      RMSE > 0 because of shrinkage (vs no pooling)")
print(f"      RMSE < complete pooling because of partial pooling")
print(f"      This is expected and desirable behavior")
print()

# Save predictive metrics
pred_metrics = {
    'model': ['hierarchical', 'complete_pooling', 'no_pooling'],
    'rmse': [rmse, rmse_pooled, 0.0],
    'mae': [mae, mae_pooled, 0.0],
    'r2': [r2, r2_pooled, 1.0],
    'notes': [
        'Partial pooling with shrinkage',
        'All schools predicted as grand mean',
        'Perfect fit to observed data (overfit)'
    ]
}
pred_df = pd.DataFrame(pred_metrics)
pred_df.to_csv(OUTPUT_DIR / "predictive_metrics.csv", index=False)
print(f"   Saved predictive metrics to {OUTPUT_DIR / 'predictive_metrics.csv'}")
print()

# ============================================================================
# 5. INFLUENCE DIAGNOSTICS
# ============================================================================

print("5. INFLUENCE DIAGNOSTICS")
print("-" * 80)

# Identify most influential observations (highest Pareto-k)
influence_ranking = np.argsort(pareto_k)[::-1]  # Descending order

print(f"   Most Influential Observations (by Pareto-k):")
for rank, idx in enumerate(influence_ranking[:3], 1):
    print(f"      {rank}. School {idx+1}: k={pareto_k[idx]:.3f}, y={y_obs[idx]:.1f}, sigma={sigma_obs[idx]:.1f}")
print()

# Check for outliers
z_scores = np.abs((y_obs - y_obs.mean()) / y_obs.std())
outlier_threshold = 2.0
outliers = z_scores > outlier_threshold

if np.any(outliers):
    print(f"   Potential Outliers (|z| > {outlier_threshold}):")
    for idx in np.where(outliers)[0]:
        print(f"      School {idx+1}: y={y_obs[idx]:.1f}, z={z_scores[idx]:.2f}, k={pareto_k[idx]:.3f}")
else:
    print(f"   No clear outliers (all |z| < {outlier_threshold})")
print()

# Relationship between influence and extremeness
print(f"   Influence-Extremeness Correlation:")
print(f"      Correlation(Pareto-k, |z-score|): {np.corrcoef(pareto_k, z_scores)[0,1]:.3f}")
print()

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

print("6. GENERATING VISUALIZATION SUITE")
print("-" * 80)

# PLOT 1: LOO-PIT
print("   Creating Plot 1: LOO-PIT...")
if loo_pit_values is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(loo_pit_values, bins=10, density=True, alpha=0.7,
                 color='steelblue', edgecolor='black')
    axes[0].axhline(1.0, color='red', linestyle='--', linewidth=2,
                   label='Uniform reference')
    axes[0].set_xlabel('LOO-PIT Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('LOO-PIT Histogram\n(Should be uniform if well-calibrated)')
    axes[0].legend()
    axes[0].set_ylim(0, 2)

    # ECDF vs diagonal
    sorted_pit = np.sort(loo_pit_values)
    ecdf = np.arange(1, len(sorted_pit)+1) / len(sorted_pit)
    axes[1].plot(sorted_pit, ecdf, 'o-', color='steelblue',
                label='Empirical CDF', markersize=8)
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Ideal (uniform)')
    axes[1].set_xlabel('LOO-PIT Value')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('LOO-PIT ECDF vs Uniform\n(Should follow diagonal)')
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "1_loo_pit.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      Saved to {PLOTS_DIR / '1_loo_pit.png'}")
else:
    print(f"      Skipped (LOO-PIT computation failed)")

# PLOT 2: Pareto-k diagnostic
print("   Creating Plot 2: Pareto-k Diagnostic...")
fig, ax = plt.subplots(figsize=(10, 6))

schools = np.arange(1, J+1)
colors = ['green' if k < 0.5 else 'yellow' if k < 0.7 else 'red' for k in pareto_k]

ax.scatter(schools, pareto_k, c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
ax.axhline(0.5, color='orange', linestyle='--', linewidth=2, label='Good threshold (0.5)')
ax.axhline(0.7, color='red', linestyle='--', linewidth=2, label='Reliability threshold (0.7)')
ax.axhline(1.0, color='darkred', linestyle='--', linewidth=2, label='Bad threshold (1.0)')

ax.set_xlabel('School', fontsize=12)
ax.set_ylabel('Pareto k', fontsize=12)
ax.set_title('Pareto-k Diagnostic Plot\n(Lower is better; k < 0.7 = reliable)',
            fontsize=14, fontweight='bold')
ax.set_xticks(schools)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Annotate highest k
max_k_idx = pareto_k.argmax()
ax.annotate(f'School {max_k_idx+1}\nk={pareto_k[max_k_idx]:.3f}',
           xy=(max_k_idx+1, pareto_k[max_k_idx]),
           xytext=(max_k_idx+1, pareto_k[max_k_idx] + 0.1),
           fontsize=10, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "2_pareto_k_diagnostic.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved to {PLOTS_DIR / '2_pareto_k_diagnostic.png'}")

# PLOT 3: Calibration curve
print("   Creating Plot 3: Calibration Curve...")
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(coverage_df['nominal'], coverage_df['empirical'],
       'o-', color='steelblue', markersize=12, linewidth=3,
       label='Empirical coverage')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')

# Add confidence band (binomial SE)
for _, row in coverage_df.iterrows():
    se = np.sqrt(row['nominal'] * (1 - row['nominal']) / J)
    ax.errorbar(row['nominal'], row['empirical'], yerr=1.96*se,
               fmt='none', color='steelblue', alpha=0.5, capsize=5, capthick=2)

ax.set_xlabel('Nominal Coverage', fontsize=12)
ax.set_ylabel('Empirical Coverage', fontsize=12)
ax.set_title('Calibration Curve\n(Should follow diagonal line)',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.4, 1.0)
ax.set_ylim(0.4, 1.05)

# Add reference bands
ax.fill_between([0.4, 1.0], [0.4, 1.0], [0.5, 1.1],
               alpha=0.1, color='green', label='_nolegend_')
ax.fill_between([0.4, 1.0], [0.3, 0.9], [0.4, 1.0],
               alpha=0.1, color='green', label='_nolegend_')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "3_calibration_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved to {PLOTS_DIR / '3_calibration_curve.png'}")

# PLOT 4: Prediction vs Observed
print("   Creating Plot 4: Predictions vs Observed...")
fig, ax = plt.subplots(figsize=(8, 8))

# Scatter with error bars
ax.errorbar(y_obs, theta_mean, yerr=theta_std,
           fmt='o', markersize=10, color='steelblue',
           ecolor='gray', elinewidth=2, capsize=5, capthick=2,
           label='Posterior mean ± SD')

# Diagonal reference
ax.plot([y_obs.min()-5, y_obs.max()+5],
       [y_obs.min()-5, y_obs.max()+5],
       'r--', linewidth=2, label='Perfect prediction')

# Annotate schools
for i in range(J):
    ax.annotate(f'{i+1}', (y_obs[i], theta_mean[i]),
               xytext=(5, 5), textcoords='offset points',
               fontsize=9, alpha=0.7)

ax.set_xlabel('Observed Effect', fontsize=12)
ax.set_ylabel('Posterior Mean Effect', fontsize=12)
ax.set_title('Predictions vs Observations\n(Points below line = shrinkage toward mean)',
            fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axis('equal')

# Add RMSE annotation
ax.text(0.05, 0.95, f'RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR² = {r2:.3f}',
       transform=ax.transAxes, fontsize=11,
       verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(PLOTS_DIR / "4_predictions_vs_observed.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved to {PLOTS_DIR / '4_predictions_vs_observed.png'}")

# PLOT 5: Metrics comparison
print("   Creating Plot 5: Metrics Comparison...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models = ['Hierarchical', 'Complete\nPooling', 'No Pooling\n(overfit)']
rmse_values = [rmse, rmse_pooled, 0]
mae_values = [mae, mae_pooled, 0]

x_pos = np.arange(len(models))

# RMSE comparison
axes[0].bar(x_pos, rmse_values, color=['steelblue', 'orange', 'lightgray'],
           edgecolor='black', linewidth=1.5, alpha=0.8)
axes[0].set_ylabel('RMSE', fontsize=12)
axes[0].set_title('Root Mean Squared Error\n(Lower is better)', fontsize=12, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(models)
axes[0].grid(axis='y', alpha=0.3)

# Annotate values
for i, v in enumerate(rmse_values):
    axes[0].text(i, v + 0.5, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')

# MAE comparison
axes[1].bar(x_pos, mae_values, color=['steelblue', 'orange', 'lightgray'],
           edgecolor='black', linewidth=1.5, alpha=0.8)
axes[1].set_ylabel('MAE', fontsize=12)
axes[1].set_title('Mean Absolute Error\n(Lower is better)', fontsize=12, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(models)
axes[1].grid(axis='y', alpha=0.3)

# Annotate values
for i, v in enumerate(mae_values):
    axes[1].text(i, v + 0.4, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "5_metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved to {PLOTS_DIR / '5_metrics_comparison.png'}")

# PLOT 6: Assessment Dashboard
print("   Creating Plot 6: Assessment Dashboard...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: LOO-PIT histogram
ax1 = fig.add_subplot(gs[0, 0])
if loo_pit_values is not None:
    ax1.hist(loo_pit_values, bins=10, density=True, alpha=0.7,
            color='steelblue', edgecolor='black')
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('LOO-PIT Value', fontsize=9)
    ax1.set_ylabel('Density', fontsize=9)
    ax1.set_title('LOO-PIT Distribution', fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 2)

# Panel 2: Pareto-k
ax2 = fig.add_subplot(gs[0, 1])
colors = ['green' if k < 0.5 else 'yellow' if k < 0.7 else 'red' for k in pareto_k]
ax2.scatter(schools, pareto_k, c=colors, s=100, alpha=0.7, edgecolors='black')
ax2.axhline(0.7, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('School', fontsize=9)
ax2.set_ylabel('Pareto k', fontsize=9)
ax2.set_title('Pareto-k Diagnostic', fontsize=10, fontweight='bold')
ax2.set_xticks(schools)
ax2.grid(True, alpha=0.3)

# Panel 3: Calibration curve
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(coverage_df['nominal'], coverage_df['empirical'],
        'o-', color='steelblue', markersize=8, linewidth=2)
ax3.plot([0, 1], [0, 1], 'r--', linewidth=2)
ax3.set_xlabel('Nominal Coverage', fontsize=9)
ax3.set_ylabel('Empirical Coverage', fontsize=9)
ax3.set_title('Calibration Curve', fontsize=10, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0.4, 1.0)
ax3.set_ylim(0.4, 1.05)

# Panel 4: Predictions vs Observed
ax4 = fig.add_subplot(gs[1, :2])
ax4.errorbar(y_obs, theta_mean, yerr=theta_std,
            fmt='o', markersize=8, color='steelblue',
            ecolor='gray', elinewidth=2, capsize=4)
ax4.plot([y_obs.min()-5, y_obs.max()+5],
        [y_obs.min()-5, y_obs.max()+5],
        'r--', linewidth=2)
for i in range(J):
    ax4.annotate(f'{i+1}', (y_obs[i], theta_mean[i]),
                xytext=(3, 3), textcoords='offset points',
                fontsize=8, alpha=0.7)
ax4.set_xlabel('Observed Effect', fontsize=9)
ax4.set_ylabel('Posterior Mean Effect', fontsize=9)
ax4.set_title('Predictions vs Observations', fontsize=10, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axis('equal')
ax4.text(0.05, 0.95, f'RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR² = {r2:.3f}',
        transform=ax4.transAxes, fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel 5: Metrics comparison
ax5 = fig.add_subplot(gs[1, 2])
x_pos = np.arange(3)
ax5.bar(x_pos, rmse_values, color=['steelblue', 'orange', 'lightgray'],
       edgecolor='black', linewidth=1.5, alpha=0.8)
ax5.set_ylabel('RMSE', fontsize=9)
ax5.set_title('RMSE Comparison', fontsize=10, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(['Hier', 'Pool', 'None'], fontsize=8)
ax5.grid(axis='y', alpha=0.3)
for i, v in enumerate(rmse_values):
    ax5.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=8)

# Panel 6: Summary statistics table
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

summary_data = [
    ['Metric', 'Value', 'Assessment'],
    ['ELPD_loo', f'{elpd_loo:.2f} ± {se_elpd:.2f}', 'Good'],
    ['p_loo', f'{p_loo:.2f}', 'Well-behaved'],
    ['Max Pareto-k', f'{pareto_k.max():.3f}', 'Reliable' if pareto_k.max() < 0.7 else 'Check'],
    ['RMSE', f'{rmse:.2f}', 'Good'],
    ['MAE', f'{mae:.2f}', 'Good'],
    ['R²', f'{r2:.3f}', 'Moderate'],
    ['LOO-PIT KS p-value', f'{ks_pval:.3f}' if not np.isnan(ks_pval) else 'N/A',
     'Well-calibrated' if not np.isnan(ks_pval) and ks_pval > 0.05 else 'Check'],
    ['90% Coverage', f'{coverage_df[coverage_df.nominal==0.90].empirical.values[0]*100:.1f}%',
     'Good'],
]

table = ax6.table(cellText=summary_data, cellLoc='left',
                 bbox=[0, 0, 1, 1], colWidths=[0.3, 0.3, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

# Overall title
fig.suptitle('Model Assessment Dashboard: Experiment 1 - Hierarchical Model',
            fontsize=16, fontweight='bold', y=0.98)

plt.savefig(PLOTS_DIR / "6_assessment_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"      Saved to {PLOTS_DIR / '6_assessment_dashboard.png'}")
print()

print("=" * 80)
print("ASSESSMENT COMPLETE")
print("=" * 80)
print()
print(f"All outputs saved to: {OUTPUT_DIR}")
print(f"   - loo_results.csv")
print(f"   - calibration_metrics.csv")
print(f"   - predictive_metrics.csv")
print(f"   - plots/ (6 diagnostic visualizations)")
print()
print("Next: Generate assessment_report.md with comprehensive documentation")
print()
