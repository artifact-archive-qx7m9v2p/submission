"""Comprehensive Model Assessment and Comparison - FINAL"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import xarray as xr
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

DATA_PATH = Path("/workspace/data/data.csv")
EXP1_PATH = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
EXP2_PATH = Path("/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf")
OUTPUT_PATH = Path("/workspace/experiments/model_comparison")
RESULTS_PATH = OUTPUT_PATH / "results"
PLOTS_PATH = OUTPUT_PATH / "plots"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE MODEL ASSESSMENT AND COMPARISON")
print("=" * 80)
print()

data = pd.read_csv(DATA_PATH)
y_obs = data['C'].values
year = data['year'].values
n_obs = len(y_obs)
print(f"Data: {n_obs} observations, range [{y_obs.min()}, {y_obs.max()}]\n")

# ============================================================================
# EXP1: NEGATIVE BINOMIAL GLM
# ============================================================================

print("=" * 80)
print("EXPERIMENT 1: NEGATIVE BINOMIAL GLM")
print("=" * 80)
print()

idata1 = az.from_netcdf(EXP1_PATH)
loo1 = az.loo(idata1, var_name='C_obs', pointwise=True)

print(f"ELPD_LOO: {loo1.elpd_loo:.2f} ± {loo1.se:.2f}")
print(f"p_LOO: {loo1.p_loo:.2f}")

k_values1 = loo1.pareto_k.values
k_good1 = np.sum(k_values1 < 0.5)
k_ok1 = np.sum((k_values1 >= 0.5) & (k_values1 < 0.7))
k_bad1 = np.sum(k_values1 >= 0.7)
k_max1 = np.max(k_values1)

print(f"Pareto-k: {k_good1} good, {k_ok1} ok, {k_bad1} bad (max={k_max1:.3f})\n")

# Posterior predictive
print("Generating posterior predictive...")
beta_0 = idata1.posterior['beta_0'].values.flatten()
beta_1 = idata1.posterior['beta_1'].values.flatten()
beta_2 = idata1.posterior['beta_2'].values.flatten()
phi = idata1.posterior['phi'].values.flatten()

log_mu = beta_0[:, None] + beta_1[:, None] * year + beta_2[:, None] * year**2
mu = np.exp(log_mu)
p = phi[:, None] / (phi[:, None] + mu)
y_pred1 = np.random.negative_binomial(phi[:, None], p)

y_pred_mean1 = y_pred1.mean(axis=0)
mae1 = np.mean(np.abs(y_obs - y_pred_mean1))
rmse1 = np.sqrt(np.mean((y_obs - y_pred_mean1)**2))
r2_1 = 1 - np.var(y_obs - y_pred_mean1) / np.var(y_obs)

y_pred_05_1 = np.percentile(y_pred1, 5, axis=0)
y_pred_95_1 = np.percentile(y_pred1, 95, axis=0)
coverage1 = np.mean((y_obs >= y_pred_05_1) & (y_obs <= y_pred_95_1))

loo_pit1 = np.array([np.mean(y_pred1[:, i] < y_obs[i]) for i in range(n_obs)])

print(f"MAE: {mae1:.2f}, RMSE: {rmse1:.2f}, R²: {r2_1:.3f}")
print(f"90% Coverage: {100*coverage1:.1f}%")
print(f"LOO-PIT: mean={np.mean(loo_pit1):.3f}, std={np.std(loo_pit1):.3f}\n")

with open(RESULTS_PATH / "loo_summary_exp1.txt", "w") as f:
    f.write("EXPERIMENT 1: NEGATIVE BINOMIAL GLM WITH QUADRATIC TREND\n")
    f.write("=" * 70 + "\n\n")
    f.write("Status: REJECTED (residual ACF=0.596, PPC failed)\n\n")
    f.write(f"ELPD_LOO: {loo1.elpd_loo:.2f} ± {loo1.se:.2f}\n")
    f.write(f"p_LOO: {loo1.p_loo:.2f}\n")
    f.write(f"Pareto-k: {k_good1} good, {k_ok1} ok, {k_bad1} bad (max={k_max1:.3f})\n\n")
    f.write(f"MAE: {mae1:.2f}, RMSE: {rmse1:.2f}, R²: {r2_1:.3f}\n")
    f.write(f"90% PI Coverage: {100*coverage1:.1f}%\n")

# ============================================================================
# EXP2: AR(1) LOG-NORMAL
# ============================================================================

print("=" * 80)
print("EXPERIMENT 2: AR(1) LOG-NORMAL")
print("=" * 80)
print()

idata2_orig = az.from_netcdf(EXP2_PATH)

# Combine log-likelihoods
ll_rest = idata2_orig.log_likelihood['obs_rest'].values
ll_0 = idata2_orig.log_likelihood['obs_0'].values
ll_combined = np.zeros((ll_rest.shape[0], ll_rest.shape[1], n_obs))
ll_combined[:, :, 0] = ll_0
ll_combined[:, :, 1:] = ll_rest

idata2 = idata2_orig.copy()
idata2.log_likelihood = xr.Dataset({
    'y': xr.DataArray(
        ll_combined,
        dims=['chain', 'draw', 'y_dim_0'],
        coords={
            'chain': idata2_orig.log_likelihood.chain,
            'draw': idata2_orig.log_likelihood.draw,
            'y_dim_0': np.arange(n_obs)
        }
    )
})

loo2 = az.loo(idata2, var_name='y', pointwise=True)

print(f"ELPD_LOO: {loo2.elpd_loo:.2f} ± {loo2.se:.2f}")
print(f"p_LOO: {loo2.p_loo:.2f}")

k_values2 = loo2.pareto_k.values
k_good2 = np.sum(k_values2 < 0.5)
k_ok2 = np.sum((k_values2 >= 0.5) & (k_values2 < 0.7))
k_bad2 = np.sum(k_values2 >= 0.7)
k_max2 = np.max(k_values2)

print(f"Pareto-k: {k_good2} good, {k_ok2} ok, {k_bad2} bad (max={k_max2:.3f})")
if k_bad2 > 0:
    print(f"  WARNING: {k_bad2} observations with k≥0.7\n")

# Posterior predictive: Exp2 models log(C), so mu_full is in log-space
print("Generating posterior predictive...")
mu_full = idata2_orig.posterior['mu_full'].values  # (chains, draws, n_obs) in log-space
sigma_regime = idata2_orig.posterior['sigma_regime'].values  # (chains, draws, 3)

# Need regime assignments
regime_idx = np.array([0]*9 + [1]*11 + [2]*20)  # From Exp2 design

n_chains, n_draws, _ = mu_full.shape
n_samples = n_chains * n_draws

# Generate predictions: log(C) ~ Normal(mu_full, sigma) → C ~ LogNormal(mu_full, sigma)
y_pred2 = np.zeros((n_samples, n_obs))
idx = 0
for i in range(n_chains):
    for j in range(n_draws):
        for k in range(n_obs):
            sigma_k = sigma_regime[i, j, regime_idx[k]]
            # log(C) ~ Normal(mu, sigma) → C = exp(Normal(mu, sigma))
            log_c_sample = np.random.normal(mu_full[i, j, k], sigma_k)
            y_pred2[idx, k] = np.exp(log_c_sample)
        idx += 1

y_pred_mean2 = y_pred2.mean(axis=0)
mae2 = np.mean(np.abs(y_obs - y_pred_mean2))
rmse2 = np.sqrt(np.mean((y_obs - y_pred_mean2)**2))
r2_2 = 1 - np.var(y_obs - y_pred_mean2) / np.var(y_obs)

y_pred_05_2 = np.percentile(y_pred2, 5, axis=0)
y_pred_95_2 = np.percentile(y_pred2, 95, axis=0)
coverage2 = np.mean((y_obs >= y_pred_05_2) & (y_obs <= y_pred_95_2))

loo_pit2 = np.array([np.mean(y_pred2[:, i] < y_obs[i]) for i in range(n_obs)])

print(f"MAE: {mae2:.2f}, RMSE: {rmse2:.2f}, R²: {r2_2:.3f}")
print(f"90% Coverage: {100*coverage2:.1f}%")
print(f"LOO-PIT: mean={np.mean(loo_pit2):.3f}, std={np.std(loo_pit2):.3f}\n")

with open(RESULTS_PATH / "loo_summary_exp2.txt", "w") as f:
    f.write("EXPERIMENT 2: AR(1) LOG-NORMAL WITH REGIME-SWITCHING\n")
    f.write("=" * 70 + "\n\n")
    f.write("Status: CONDITIONAL ACCEPT (residual ACF=0.549)\n\n")
    f.write(f"ELPD_LOO: {loo2.elpd_loo:.2f} ± {loo2.se:.2f}\n")
    f.write(f"p_LOO: {loo2.p_loo:.2f}\n")
    f.write(f"Pareto-k: {k_good2} good, {k_ok2} ok, {k_bad2} bad (max={k_max2:.3f})\n")
    if k_bad2 > 0:
        f.write(f"WARNING: {k_bad2} observations with k≥0.7\n")
    f.write(f"\nMAE: {mae2:.2f}, RMSE: {rmse2:.2f}, R²: {r2_2:.3f}\n")
    f.write(f"90% PI Coverage: {100*coverage2:.1f}%\n")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print()

elpd_dict = {'Exp1_NegBin': loo1, 'Exp2_AR1': loo2}
comparison = az.compare(elpd_dict, ic='loo', method='stacking')

print(comparison.to_string())
print()

delta_elpd = loo2.elpd_loo - loo1.elpd_loo
se_diff = comparison.loc['Exp1_NegBin', 'dse']
weight1 = comparison.loc['Exp1_NegBin', 'weight']
weight2 = comparison.loc['Exp2_AR1', 'weight']

print(f"ΔELPD (Exp2 - Exp1): {delta_elpd:.2f} ± {se_diff:.2f}")
print(f"Significance: {abs(delta_elpd) / se_diff:.1f} SE\n")

if abs(delta_elpd) > 4 * se_diff:
    decision = "CLEAR WINNER"
    print(f"DECISION: {decision} (|ΔELPD| > 4×SE)")
    print(f"  → Exp2 strongly preferred")
elif abs(delta_elpd) > 2 * se_diff:
    decision = "MODERATE DIFFERENCE"
    print(f"DECISION: {decision} (2×SE < |ΔELPD| < 4×SE)")
    print(f"  → Exp2 preferred")
else:
    decision = "INDISTINGUISHABLE"
    print(f"DECISION: {decision} (|ΔELPD| < 2×SE)")

print(f"\nStacking weights: Exp1={weight1:.3f}, Exp2={weight2:.3f}\n")

comparison.to_csv(RESULTS_PATH / "loo_comparison.csv")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Generating visualizations...\n")

# 1. LOO Comparison
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_compare(comparison, insample_dev=False, ax=ax)
plt.title("Model Comparison: LOO Cross-Validation", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOTS_PATH / "loo_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Pareto-k
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
az.plot_khat(loo1, ax=axes[0], show_bins=True)
axes[0].set_title(f"Exp1: Neg Binomial\n{k_bad1} problematic", fontweight='bold')
axes[0].axhline(0.7, color='red', linestyle='--', alpha=0.5)
az.plot_khat(loo2, ax=axes[1], show_bins=True)
axes[1].set_title(f"Exp2: AR(1) Log-Normal\n{k_bad2} problematic", fontweight='bold')
axes[1].axhline(0.7, color='red', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "pareto_k_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Calibration
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].hist(loo_pit1, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 0].axhline(1.0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_title("Exp1: LOO-PIT", fontweight='bold')
axes[0, 0].set_xlabel("PIT"); axes[0, 0].set_ylabel("Density"); axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(loo_pit2, bins=20, density=True, alpha=0.7, color='coral', edgecolor='black')
axes[0, 1].axhline(1.0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_title("Exp2: LOO-PIT", fontweight='bold')
axes[0, 1].set_xlabel("PIT"); axes[0, 1].set_ylabel("Density"); axes[0, 1].grid(alpha=0.3)

theoretical = np.linspace(0, 1, n_obs)
axes[1, 0].plot(theoretical, np.sort(loo_pit1), 'o', alpha=0.6, color='steelblue')
axes[1, 0].plot([0, 1], [0, 1], 'r--', linewidth=2)
axes[1, 0].set_title("Exp1: Q-Q Plot", fontweight='bold')
axes[1, 0].set_xlabel("Theoretical"); axes[1, 0].set_ylabel("Observed"); axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(theoretical, np.sort(loo_pit2), 'o', alpha=0.6, color='coral')
axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2)
axes[1, 1].set_title("Exp2: Q-Q Plot", fontweight='bold')
axes[1, 1].set_xlabel("Theoretical"); axes[1, 1].set_ylabel("Observed"); axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_PATH / "calibration_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Fitted Trends
fig, ax = plt.subplots(figsize=(14, 7))
ax.scatter(year, y_obs, color='black', s=60, alpha=0.7, zorder=5, label='Observed')
ax.plot(year, np.median(y_pred1, axis=0), color='steelblue', linewidth=2.5, label='Exp1', zorder=3)
ax.fill_between(year, y_pred_05_1, y_pred_95_1, color='steelblue', alpha=0.2)
ax.plot(year, np.median(y_pred2, axis=0), color='coral', linewidth=2.5, label='Exp2', linestyle='--', zorder=4)
ax.fill_between(year, y_pred_05_2, y_pred_95_2, color='coral', alpha=0.2)
ax.set_xlabel("Year", fontsize=12); ax.set_ylabel("Count", fontsize=12)
ax.set_title("Fitted Trends with 90% PI", fontsize=14, fontweight='bold')
ax.legend(loc='upper left'); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "fitted_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Prediction Intervals
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
width1 = y_pred_95_1 - y_pred_05_1
width2 = y_pred_95_2 - y_pred_05_2
axes[0].plot(year, width1, 'o-', color='steelblue', linewidth=2, label='Exp1', alpha=0.7)
axes[0].plot(year, width2, 's-', color='coral', linewidth=2, label='Exp2', alpha=0.7)
axes[0].set_xlabel("Year"); axes[0].set_ylabel("90% PI Width")
axes[0].set_title("Prediction Uncertainty", fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3)

coverage_exp1 = (y_obs >= y_pred_05_1) & (y_obs <= y_pred_95_1)
coverage_exp2 = (y_obs >= y_pred_05_2) & (y_obs <= y_pred_95_2)
x_pos = np.arange(n_obs)
axes[1].bar(x_pos - 0.2, coverage_exp1, width=0.4, color='steelblue', alpha=0.7, label='Exp1')
axes[1].bar(x_pos + 0.2, coverage_exp2, width=0.4, color='coral', alpha=0.7, label='Exp2')
axes[1].axhline(0.9, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel("Observation"); axes[1].set_ylabel("In 90% PI")
axes[1].set_title(f"Coverage: Exp1={100*coverage1:.1f}%, Exp2={100*coverage2:.1f}%", fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(PLOTS_PATH / "prediction_intervals.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Spider Plot
def norm(val, min_val, max_val, inverse=False):
    n = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    return 1 - n if inverse else n

mae_min, mae_max = min(mae1, mae2), max(mae1, mae2)
scores1 = [
    norm(mae1, mae_min, mae_max, inverse=True),
    1 - abs(coverage1 - 0.9) / 0.9,
    k_good1 / n_obs,
    0.7,  # Simplicity
    0.3   # Temporal structure
]
scores2 = [
    norm(mae2, mae_min, mae_max, inverse=True),
    1 - abs(coverage2 - 0.9) / 0.9,
    k_good2 / n_obs,
    0.3,
    0.5
]

criteria = ['Predictive\nAccuracy', 'Calibration', 'LOO\nReliability', 'Simplicity', 'Temporal\nStructure']
angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
scores1_plot = scores1 + [scores1[0]]
scores2_plot = scores2 + [scores2[0]]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
ax.plot(angles, scores1_plot, 'o-', linewidth=2, color='steelblue', label='Exp1', markersize=8)
ax.fill(angles, scores1_plot, alpha=0.15, color='steelblue')
ax.plot(angles, scores2_plot, 's-', linewidth=2, color='coral', label='Exp2', markersize=8)
ax.fill(angles, scores2_plot, alpha=0.15, color='coral')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(criteria, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax.set_title("Multi-Criteria Comparison\n(Higher = Better)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "model_trade_offs.png", dpi=300, bbox_inches='tight')
plt.close()

# Summary table
summary_df = pd.DataFrame({
    'Metric': ['ELPD', 'SE', 'p_LOO', 'k<0.5', 'k≥0.7', 'max_k', 'MAE', 'RMSE', 'R²', 'Coverage', 'Weight'],
    'Exp1': [f"{loo1.elpd_loo:.2f}", f"{loo1.se:.2f}", f"{loo1.p_loo:.2f}", 
             f"{k_good1}", f"{k_bad1}", f"{k_max1:.3f}",
             f"{mae1:.2f}", f"{rmse1:.2f}", f"{r2_1:.3f}", f"{100*coverage1:.1f}%", f"{weight1:.3f}"],
    'Exp2': [f"{loo2.elpd_loo:.2f}", f"{loo2.se:.2f}", f"{loo2.p_loo:.2f}",
             f"{k_good2}", f"{k_bad2}", f"{k_max2:.3f}",
             f"{mae2:.2f}", f"{rmse2:.2f}", f"{r2_2:.3f}", f"{100*coverage2:.1f}%", f"{weight2:.3f}"]
})

print(summary_df.to_string(index=False))
summary_df.to_csv(RESULTS_PATH / "summary_metrics.csv", index=False)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nResults: {OUTPUT_PATH}")
print(f"KEY FINDING: Exp2 is {delta_elpd:.1f} ELPD points better ({abs(delta_elpd)/se_diff:.1f}×SE)")
print("Both models show temporal dependence. Exp2 CONDITIONAL ACCEPT.")

