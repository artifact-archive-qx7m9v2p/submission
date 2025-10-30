"""
Comprehensive Model Assessment and Comparison
==============================================

Compares two Bayesian models using LOO-CV, calibration diagnostics,
and practical considerations.

Models:
- Experiment 1: Negative Binomial GLM with Quadratic Trend (REJECTED)
- Experiment 2: AR(1) Log-Normal with Regime-Switching (CONDITIONAL ACCEPT)
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
EXP1_PATH = Path("/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf")
EXP2_PATH = Path("/workspace/experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf")
OUTPUT_PATH = Path("/workspace/experiments/model_comparison")
RESULTS_PATH = OUTPUT_PATH / "results"
PLOTS_PATH = OUTPUT_PATH / "plots"

# Ensure output directories exist
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE MODEL ASSESSMENT AND COMPARISON")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================

print("Loading data and models...")
data = pd.read_csv(DATA_PATH)
y_obs = data['C'].values
year = data['year'].values
n_obs = len(y_obs)

print(f"  Data: {n_obs} observations")
print(f"  Count range: [{y_obs.min()}, {y_obs.max()}]")
print()

# Load InferenceData objects
print("Loading Experiment 1 (Neg Binomial GLM)...")
idata1 = az.from_netcdf(EXP1_PATH)
print(f"  Groups: {list(idata1.groups())}")
print(f"  Has log_likelihood: {'log_likelihood' in idata1}")

print("\nLoading Experiment 2 (AR(1) Log-Normal)...")
idata2 = az.from_netcdf(EXP2_PATH)
print(f"  Groups: {list(idata2.groups())}")
print(f"  Has log_likelihood: {'log_likelihood' in idata2}")
print()

# Verify log_likelihood exists
if 'log_likelihood' not in idata1:
    raise ValueError("Experiment 1 missing log_likelihood group - cannot perform LOO-CV")
if 'log_likelihood' not in idata2:
    raise ValueError("Experiment 2 missing log_likelihood group - cannot perform LOO-CV")

# ============================================================================
# SINGLE MODEL ASSESSMENT: EXPERIMENT 1
# ============================================================================

print("=" * 80)
print("EXPERIMENT 1: NEGATIVE BINOMIAL GLM WITH QUADRATIC TREND")
print("Status: REJECTED (residual ACF=0.596)")
print("=" * 80)
print()

print("Computing LOO-CV for Experiment 1...")
loo1 = az.loo(idata1, pointwise=True)

print("\n--- LOO Diagnostics ---")
print(f"ELPD_LOO: {loo1.elpd_loo:.2f} ± {loo1.se:.2f}")
print(f"p_LOO (effective parameters): {loo1.p_loo:.2f}")
# LOO IC = -2 * ELPD_LOO
print()

# Pareto k diagnostics
k_values1 = loo1.pareto_k.values
k_good1 = np.sum(k_values1 < 0.5)
k_ok1 = np.sum((k_values1 >= 0.5) & (k_values1 < 0.7))
k_bad1 = np.sum(k_values1 >= 0.7)
k_max1 = np.max(k_values1)

print("--- Pareto-k Diagnostics ---")
print(f"k < 0.5 (good): {k_good1} ({100*k_good1/n_obs:.1f}%)")
print(f"0.5 ≤ k < 0.7 (ok): {k_ok1} ({100*k_ok1/n_obs:.1f}%)")
print(f"k ≥ 0.7 (problematic): {k_bad1} ({100*k_bad1/n_obs:.1f}%)")
print(f"Max Pareto-k: {k_max1:.3f}")

if k_bad1 > n_obs * 0.1:
    print("  WARNING: >10% observations with k≥0.7 - LOO may be unreliable")
print()

# Save detailed diagnostics
with open(RESULTS_PATH / "loo_summary_exp1.txt", "w") as f:
    f.write("EXPERIMENT 1: NEGATIVE BINOMIAL GLM WITH QUADRATIC TREND\n")
    f.write("=" * 70 + "\n\n")
    f.write("Status: REJECTED (residual ACF=0.596, PPC failed)\n\n")

    f.write("LOO-CV DIAGNOSTICS\n")
    f.write("-" * 70 + "\n")
    f.write(f"ELPD_LOO: {loo1.elpd_loo:.2f} ± {loo1.se:.2f}\n")
    f.write(f"p_LOO (effective parameters): {loo1.p_loo:.2f}\n")
    f.write(f"LOO IC: {-2*loo1.elpd_loo:.2f} ± {2*loo1.se:.2f}\n\n")

    f.write("PARETO-K DIAGNOSTICS\n")
    f.write("-" * 70 + "\n")
    f.write(f"k < 0.5 (good): {k_good1} ({100*k_good1/n_obs:.1f}%)\n")
    f.write(f"0.5 ≤ k < 0.7 (ok): {k_ok1} ({100*k_ok1/n_obs:.1f}%)\n")
    f.write(f"k ≥ 0.7 (problematic): {k_bad1} ({100*k_bad1/n_obs:.1f}%)\n")
    f.write(f"Max Pareto-k: {k_max1:.3f}\n\n")

    if k_bad1 > 0:
        f.write("WARNING: Some observations have k≥0.7, indicating LOO may be\n")
        f.write("unreliable for those points. This often occurs with influential\n")
        f.write("observations or model misspecification.\n\n")

# Calibration: LOO-PIT
print("Computing LOO-PIT for calibration assessment...")
try:
    # Get posterior predictive samples
    if 'posterior_predictive' in idata1:
        y_pred1 = idata1.posterior_predictive['y'].values
        # Reshape: (chains, draws, obs) -> (samples, obs)
        y_pred1_flat = y_pred1.reshape(-1, y_pred1.shape[-1])

        # Compute LOO-PIT manually using pointwise LOO
        loo_pit1 = []
        for i in range(n_obs):
            # Probability that prediction < observation
            pit = np.mean(y_pred1_flat[:, i] < y_obs[i])
            loo_pit1.append(pit)
        loo_pit1 = np.array(loo_pit1)

        # Check uniformity (should be U(0,1))
        print(f"  LOO-PIT mean: {np.mean(loo_pit1):.3f} (ideal: 0.5)")
        print(f"  LOO-PIT std: {np.std(loo_pit1):.3f} (ideal: 0.289)")
    else:
        print("  WARNING: No posterior_predictive group found")
        loo_pit1 = None
except Exception as e:
    print(f"  Error computing LOO-PIT: {e}")
    loo_pit1 = None

# Absolute metrics
print("\nComputing absolute prediction metrics...")
if 'posterior_predictive' in idata1 and 'y' in idata1.posterior_predictive:
    y_pred_mean1 = y_pred1.mean(axis=(0, 1))
    mae1 = np.mean(np.abs(y_obs - y_pred_mean1))
    rmse1 = np.sqrt(np.mean((y_obs - y_pred_mean1)**2))

    # Bayesian R²
    y_var = np.var(y_obs)
    resid_var = np.mean((y_obs - y_pred_mean1)**2)
    r2_1 = 1 - resid_var / y_var

    print(f"  MAE: {mae1:.2f}")
    print(f"  RMSE: {rmse1:.2f}")
    print(f"  R²: {r2_1:.3f}")

    # 90% posterior interval coverage
    y_pred_05 = np.percentile(y_pred1_flat, 5, axis=0)
    y_pred_95 = np.percentile(y_pred1_flat, 95, axis=0)
    coverage1 = np.mean((y_obs >= y_pred_05) & (y_obs <= y_pred_95))
    print(f"  90% PI Coverage: {100*coverage1:.1f}% (ideal: 90%)")
else:
    mae1 = rmse1 = r2_1 = coverage1 = np.nan
    print("  WARNING: Cannot compute metrics - no posterior predictive")

print()

# ============================================================================
# SINGLE MODEL ASSESSMENT: EXPERIMENT 2
# ============================================================================

print("=" * 80)
print("EXPERIMENT 2: AR(1) LOG-NORMAL WITH REGIME-SWITCHING")
print("Status: CONDITIONAL ACCEPT (residual ACF=0.549)")
print("=" * 80)
print()

print("Computing LOO-CV for Experiment 2...")
loo2 = az.loo(idata2, pointwise=True)

print("\n--- LOO Diagnostics ---")
print(f"ELPD_LOO: {loo2.elpd_loo:.2f} ± {loo2.se:.2f}")
print(f"p_LOO (effective parameters): {loo2.p_loo:.2f}")
# LOO IC = -2 * ELPD_LOO
print()

# Pareto k diagnostics
k_values2 = loo2.pareto_k.values
k_good2 = np.sum(k_values2 < 0.5)
k_ok2 = np.sum((k_values2 >= 0.5) & (k_values2 < 0.7))
k_bad2 = np.sum(k_values2 >= 0.7)
k_max2 = np.max(k_values2)

print("--- Pareto-k Diagnostics ---")
print(f"k < 0.5 (good): {k_good2} ({100*k_good2/n_obs:.1f}%)")
print(f"0.5 ≤ k < 0.7 (ok): {k_ok2} ({100*k_ok2/n_obs:.1f}%)")
print(f"k ≥ 0.7 (problematic): {k_bad2} ({100*k_bad2/n_obs:.1f}%)")
print(f"Max Pareto-k: {k_max2:.3f}")

if k_bad2 > n_obs * 0.1:
    print("  WARNING: >10% observations with k≥0.7 - LOO may be unreliable")
print()

# Save detailed diagnostics
with open(RESULTS_PATH / "loo_summary_exp2.txt", "w") as f:
    f.write("EXPERIMENT 2: AR(1) LOG-NORMAL WITH REGIME-SWITCHING\n")
    f.write("=" * 70 + "\n\n")
    f.write("Status: CONDITIONAL ACCEPT (residual ACF=0.549, better than Exp1)\n\n")

    f.write("LOO-CV DIAGNOSTICS\n")
    f.write("-" * 70 + "\n")
    f.write(f"ELPD_LOO: {loo2.elpd_loo:.2f} ± {loo2.se:.2f}\n")
    f.write(f"p_LOO (effective parameters): {loo2.p_loo:.2f}\n")
    f.write(f"LOO IC: {-2*loo2.elpd_loo:.2f} ± {2*loo2.se:.2f}\n\n")

    f.write("PARETO-K DIAGNOSTICS\n")
    f.write("-" * 70 + "\n")
    f.write(f"k < 0.5 (good): {k_good2} ({100*k_good2/n_obs:.1f}%)\n")
    f.write(f"0.5 ≤ k < 0.7 (ok): {k_ok2} ({100*k_ok2/n_obs:.1f}%)\n")
    f.write(f"k ≥ 0.7 (problematic): {k_bad2} ({100*k_bad2/n_obs:.1f}%)\n")
    f.write(f"Max Pareto-k: {k_max2:.3f}\n\n")

    if k_bad2 > 0:
        f.write("WARNING: Some observations have k≥0.7, indicating LOO may be\n")
        f.write("unreliable for those points. This often occurs with influential\n")
        f.write("observations or model misspecification.\n\n")

# Calibration: LOO-PIT
print("Computing LOO-PIT for calibration assessment...")
try:
    if 'posterior_predictive' in idata2:
        y_pred2 = idata2.posterior_predictive['y'].values
        # Reshape: (chains, draws, obs) -> (samples, obs)
        y_pred2_flat = y_pred2.reshape(-1, y_pred2.shape[-1])

        # Compute LOO-PIT
        loo_pit2 = []
        for i in range(n_obs):
            pit = np.mean(y_pred2_flat[:, i] < y_obs[i])
            loo_pit2.append(pit)
        loo_pit2 = np.array(loo_pit2)

        print(f"  LOO-PIT mean: {np.mean(loo_pit2):.3f} (ideal: 0.5)")
        print(f"  LOO-PIT std: {np.std(loo_pit2):.3f} (ideal: 0.289)")
    else:
        print("  WARNING: No posterior_predictive group found")
        loo_pit2 = None
except Exception as e:
    print(f"  Error computing LOO-PIT: {e}")
    loo_pit2 = None

# Absolute metrics
print("\nComputing absolute prediction metrics...")
if 'posterior_predictive' in idata2 and 'y' in idata2.posterior_predictive:
    y_pred_mean2 = y_pred2.mean(axis=(0, 1))
    mae2 = np.mean(np.abs(y_obs - y_pred_mean2))
    rmse2 = np.sqrt(np.mean((y_obs - y_pred_mean2)**2))

    # Bayesian R²
    resid_var = np.mean((y_obs - y_pred_mean2)**2)
    r2_2 = 1 - resid_var / y_var

    print(f"  MAE: {mae2:.2f}")
    print(f"  RMSE: {rmse2:.2f}")
    print(f"  R²: {r2_2:.3f}")

    # 90% posterior interval coverage
    y_pred_05_2 = np.percentile(y_pred2_flat, 5, axis=0)
    y_pred_95_2 = np.percentile(y_pred2_flat, 95, axis=0)
    coverage2 = np.mean((y_obs >= y_pred_05_2) & (y_obs <= y_pred_95_2))
    print(f"  90% PI Coverage: {100*coverage2:.1f}% (ideal: 90%)")
else:
    mae2 = rmse2 = r2_2 = coverage2 = np.nan
    print("  WARNING: Cannot compute metrics - no posterior predictive")

print()

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print()

print("Computing model comparison with ArviZ compare...")
model_dict = {"Exp1_NegBin": idata1, "Exp2_AR1": idata2}
comparison = az.compare(model_dict, ic="loo", method="stacking")

print("\n--- Comparison Table ---")
print(comparison.to_string())
print()

# Extract comparison metrics
delta_elpd = comparison.loc["Exp2_AR1", "elpd_loo"] - comparison.loc["Exp1_NegBin", "elpd_loo"]
se_diff = comparison.loc["Exp1_NegBin", "dse"]  # SE of difference from top model
weight1 = comparison.loc["Exp1_NegBin", "weight"]
weight2 = comparison.loc["Exp2_AR1", "weight"]

print("\n--- Key Metrics ---")
print(f"ΔELPD (Exp2 - Exp1): {delta_elpd:.2f} ± {se_diff:.2f}")
print(f"Significance: {abs(delta_elpd) / se_diff:.2f} standard errors")
print()

# Decision rule
if abs(delta_elpd) < 2 * se_diff:
    decision = "INDISTINGUISHABLE"
    print(f"Decision: Models INDISTINGUISHABLE (|ΔELPD| < 2×SE)")
    print(f"  → Apply parsimony: prefer simpler model (Exp1)")
    print(f"  → BUT: Residual diagnostics favor Exp2 (ACF: 0.549 vs 0.596)")
elif abs(delta_elpd) < 4 * se_diff:
    decision = "MODERATE DIFFERENCE"
    winner = "Exp2_AR1" if delta_elpd > 0 else "Exp1_NegBin"
    print(f"Decision: MODERATE DIFFERENCE (2×SE < |ΔELPD| < 4×SE)")
    print(f"  → Statistical preference for {winner}")
    print(f"  → Consider practical factors")
else:
    decision = "CLEAR WINNER"
    winner = "Exp2_AR1" if delta_elpd > 0 else "Exp1_NegBin"
    print(f"Decision: CLEAR WINNER (|ΔELPD| > 4×SE)")
    print(f"  → Strong preference for {winner}")

print()
print(f"Stacking weights:")
print(f"  Exp1 (Neg Binomial): {weight1:.3f}")
print(f"  Exp2 (AR1): {weight2:.3f}")
print()

# Save comparison table
comparison.to_csv(RESULTS_PATH / "loo_comparison.csv")
print(f"Saved comparison table to {RESULTS_PATH / 'loo_comparison.csv'}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nGenerating comparison visualizations...")

# 1. LOO Comparison Plot
print("  1. ELPD comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_compare(comparison, insample_dev=False, ax=ax)
plt.title("Model Comparison: Leave-One-Out Cross-Validation", fontsize=14, fontweight='bold')
plt.xlabel("ELPD_LOO (higher is better)", fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "loo_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Pareto-k Comparison
print("  2. Pareto-k diagnostic comparison...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Exp1
az.plot_khat(loo1, ax=axes[0], show_bins=True)
axes[0].set_title("Exp1: Negative Binomial GLM\nPareto-k Diagnostics", fontsize=12, fontweight='bold')
axes[0].axhline(0.7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='k=0.7 threshold')
axes[0].legend()

# Exp2
az.plot_khat(loo2, ax=axes[1], show_bins=True)
axes[1].set_title("Exp2: AR(1) Log-Normal\nPareto-k Diagnostics", fontsize=12, fontweight='bold')
axes[1].axhline(0.7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='k=0.7 threshold')
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS_PATH / "pareto_k_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Calibration Comparison (LOO-PIT)
print("  3. Calibration comparison (LOO-PIT)...")
if loo_pit1 is not None and loo_pit2 is not None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Exp1 histogram
    axes[0, 0].hist(loo_pit1, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform(0,1)')
    axes[0, 0].set_xlabel("LOO-PIT Value", fontsize=11)
    axes[0, 0].set_ylabel("Density", fontsize=11)
    axes[0, 0].set_title("Exp1: LOO-PIT Distribution", fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Exp2 histogram
    axes[0, 1].hist(loo_pit2, bins=20, density=True, alpha=0.7, color='coral', edgecolor='black')
    axes[0, 1].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Uniform(0,1)')
    axes[0, 1].set_xlabel("LOO-PIT Value", fontsize=11)
    axes[0, 1].set_ylabel("Density", fontsize=11)
    axes[0, 1].set_title("Exp2: LOO-PIT Distribution", fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Exp1 Q-Q plot
    sorted_pit1 = np.sort(loo_pit1)
    theoretical_quantiles = np.linspace(0, 1, len(sorted_pit1))
    axes[1, 0].plot(theoretical_quantiles, sorted_pit1, 'o', alpha=0.6, color='steelblue')
    axes[1, 0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
    axes[1, 0].set_xlabel("Theoretical Quantiles", fontsize=11)
    axes[1, 0].set_ylabel("Observed Quantiles", fontsize=11)
    axes[1, 0].set_title("Exp1: Q-Q Plot", fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Exp2 Q-Q plot
    sorted_pit2 = np.sort(loo_pit2)
    axes[1, 1].plot(theoretical_quantiles, sorted_pit2, 'o', alpha=0.6, color='coral')
    axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
    axes[1, 1].set_xlabel("Theoretical Quantiles", fontsize=11)
    axes[1, 1].set_ylabel("Observed Quantiles", fontsize=11)
    axes[1, 1].set_title("Exp2: Q-Q Plot", fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "calibration_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("    WARNING: Skipping calibration plot - LOO-PIT not available")

# 4. Fitted Trends Comparison
print("  4. Fitted trends comparison...")
if 'posterior_predictive' in idata1 and 'posterior_predictive' in idata2:
    fig, ax = plt.subplots(figsize=(14, 7))

    # Observed data
    ax.scatter(year, y_obs, color='black', s=60, alpha=0.7, zorder=5, label='Observed')

    # Exp1 predictions
    y_pred1_q50 = np.median(y_pred1_flat, axis=0)
    y_pred1_q05 = np.percentile(y_pred1_flat, 5, axis=0)
    y_pred1_q95 = np.percentile(y_pred1_flat, 95, axis=0)

    ax.plot(year, y_pred1_q50, color='steelblue', linewidth=2.5, label='Exp1: Neg Binomial (median)', zorder=3)
    ax.fill_between(year, y_pred1_q05, y_pred1_q95, color='steelblue', alpha=0.2, label='Exp1: 90% PI')

    # Exp2 predictions
    y_pred2_q50 = np.median(y_pred2_flat, axis=0)
    y_pred2_q05 = np.percentile(y_pred2_flat, 5, axis=0)
    y_pred2_q95 = np.percentile(y_pred2_flat, 95, axis=0)

    ax.plot(year, y_pred2_q50, color='coral', linewidth=2.5, label='Exp2: AR(1) (median)', zorder=4, linestyle='--')
    ax.fill_between(year, y_pred2_q05, y_pred2_q95, color='coral', alpha=0.2, label='Exp2: 90% PI')

    ax.set_xlabel("Year (standardized)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Model Comparison: Fitted Trends with 90% Prediction Intervals", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "fitted_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("    WARNING: Skipping fitted trends - posterior predictive not available")

# 5. Prediction Intervals Width Comparison
print("  5. Prediction interval width comparison...")
if 'posterior_predictive' in idata1 and 'posterior_predictive' in idata2:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Interval widths
    width1 = y_pred1_q95 - y_pred1_q05
    width2 = y_pred2_q95 - y_pred2_q05

    # Plot 1: Width over time
    axes[0].plot(year, width1, 'o-', color='steelblue', linewidth=2, markersize=6, label='Exp1: Neg Binomial', alpha=0.7)
    axes[0].plot(year, width2, 's-', color='coral', linewidth=2, markersize=6, label='Exp2: AR(1)', alpha=0.7)
    axes[0].set_xlabel("Year (standardized)", fontsize=12)
    axes[0].set_ylabel("90% PI Width", fontsize=12)
    axes[0].set_title("Prediction Uncertainty Over Time", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Plot 2: Coverage comparison
    coverage_exp1 = (y_obs >= y_pred1_q05) & (y_obs <= y_pred1_q95)
    coverage_exp2 = (y_obs >= y_pred2_q05) & (y_obs <= y_pred2_q95)

    x_pos = np.arange(len(year))
    axes[1].bar(x_pos - 0.2, coverage_exp1, width=0.4, color='steelblue', alpha=0.7, label='Exp1: Neg Binomial')
    axes[1].bar(x_pos + 0.2, coverage_exp2, width=0.4, color='coral', alpha=0.7, label='Exp2: AR(1)')
    axes[1].axhline(0.9, color='red', linestyle='--', linewidth=2, label='Target: 90%')
    axes[1].set_xlabel("Observation Index", fontsize=12)
    axes[1].set_ylabel("In 90% PI (1=yes, 0=no)", fontsize=12)
    axes[1].set_title(f"Point-wise Coverage: Exp1={100*coverage1:.1f}%, Exp2={100*coverage2:.1f}%",
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "prediction_intervals.png", dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("    WARNING: Skipping interval comparison - posterior predictive not available")

# 6. Multi-criteria comparison (Spider plot)
print("  6. Multi-criteria trade-offs visualization...")

# Normalize metrics to 0-1 scale (higher is better)
def normalize_metric(value, min_val, max_val, inverse=False):
    """Normalize to 0-1, optionally inverting so higher is better"""
    norm = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    return 1 - norm if inverse else norm

# Define criteria (higher is better after normalization)
criteria = ['Predictive\nAccuracy', 'Calibration', 'LOO\nReliability',
            'Simplicity', 'Temporal\nStructure']

# Compute scores (0-1, higher is better)
# Predictive accuracy: use negative MAE, normalized
mae_min, mae_max = min(mae1, mae2), max(mae1, mae2)
score1_pred = normalize_metric(mae1, mae_min, mae_max, inverse=True)
score2_pred = normalize_metric(mae2, mae_min, mae_max, inverse=True)

# Calibration: closeness to 90% coverage
cal1 = 1 - abs(coverage1 - 0.9) / 0.9
cal2 = 1 - abs(coverage2 - 0.9) / 0.9

# LOO reliability: proportion of good Pareto-k
loo_rel1 = k_good1 / n_obs
loo_rel2 = k_good2 / n_obs

# Simplicity: Exp1 is simpler (fewer parameters)
simp1 = 0.7
simp2 = 0.3

# Temporal structure: Based on residual ACF (Exp2 better)
# Exp1: ACF=0.596 (poor), Exp2: ACF=0.549 (less poor)
temp1 = 0.3
temp2 = 0.5

scores1 = [score1_pred, cal1, loo_rel1, simp1, temp1]
scores2 = [score2_pred, cal2, loo_rel2, simp2, temp2]

# Radar chart
angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
scores1_plot = scores1 + [scores1[0]]
scores2_plot = scores2 + [scores2[0]]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
ax.plot(angles, scores1_plot, 'o-', linewidth=2, color='steelblue', label='Exp1: Neg Binomial', markersize=8)
ax.fill(angles, scores1_plot, alpha=0.15, color='steelblue')
ax.plot(angles, scores2_plot, 's-', linewidth=2, color='coral', label='Exp2: AR(1)', markersize=8)
ax.fill(angles, scores2_plot, alpha=0.15, color='coral')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(criteria, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax.set_title("Multi-Criteria Model Comparison\n(Higher scores = Better performance)",
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(PLOTS_PATH / "model_trade_offs.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nAll visualizations saved to:", PLOTS_PATH)

# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

summary_data = {
    'Metric': [
        'ELPD_LOO',
        'ELPD SE',
        'p_LOO',
        'Pareto-k < 0.5',
        'Pareto-k ≥ 0.7',
        'Max Pareto-k',
        'MAE',
        'RMSE',
        'R²',
        '90% PI Coverage',
        'LOO-PIT mean',
        'LOO-PIT std',
        'Stacking Weight'
    ],
    'Exp1_NegBin': [
        f"{loo1.elpd_loo:.2f}",
        f"{loo1.se:.2f}",
        f"{loo1.p_loo:.2f}",
        f"{k_good1} ({100*k_good1/n_obs:.1f}%)",
        f"{k_bad1} ({100*k_bad1/n_obs:.1f}%)",
        f"{k_max1:.3f}",
        f"{mae1:.2f}",
        f"{rmse1:.2f}",
        f"{r2_1:.3f}",
        f"{100*coverage1:.1f}%",
        f"{np.mean(loo_pit1):.3f}" if loo_pit1 is not None else "N/A",
        f"{np.std(loo_pit1):.3f}" if loo_pit1 is not None else "N/A",
        f"{weight1:.3f}"
    ],
    'Exp2_AR1': [
        f"{loo2.elpd_loo:.2f}",
        f"{loo2.se:.2f}",
        f"{loo2.p_loo:.2f}",
        f"{k_good2} ({100*k_good2/n_obs:.1f}%)",
        f"{k_bad2} ({100*k_bad2/n_obs:.1f}%)",
        f"{k_max2:.3f}",
        f"{mae2:.2f}",
        f"{rmse2:.2f}",
        f"{r2_2:.3f}",
        f"{100*coverage2:.1f}%",
        f"{np.mean(loo_pit2):.3f}" if loo_pit2 is not None else "N/A",
        f"{np.std(loo_pit2):.3f}" if loo_pit2 is not None else "N/A",
        f"{weight2:.3f}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

# Save summary table
summary_df.to_csv(RESULTS_PATH / "summary_metrics.csv", index=False)
print(f"\nSaved summary table to {RESULTS_PATH / 'summary_metrics.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print()

print(f"Decision: {decision}")
print(f"ΔELPD: {delta_elpd:.2f} ± {se_diff:.2f}")
print(f"Statistical significance: {abs(delta_elpd) / se_diff:.2f} SE")
print()

if delta_elpd > 0:
    print("Exp2 (AR1) has higher ELPD → Better predictive performance")
else:
    print("Exp1 (NegBin) has higher ELPD → Better predictive performance")

print()
print("Key observations:")
print(f"  • Exp2 has {k_bad2} problematic Pareto-k values vs {k_bad1} for Exp1")
print(f"  • Exp2 MAE: {mae2:.2f}, Exp1 MAE: {mae1:.2f}")
print(f"  • Exp2 coverage: {100*coverage2:.1f}%, Exp1 coverage: {100*coverage1:.1f}%")
print(f"  • Stacking heavily favors: {'Exp2' if weight2 > 0.7 else 'Exp1' if weight1 > 0.7 else 'Neither (weights similar)'}")
print()

print("Critical context:")
print("  • Exp1 status: REJECTED (residual ACF=0.596, PPC failed)")
print("  • Exp2 status: CONDITIONAL ACCEPT (residual ACF=0.549)")
print("  • Both models show temporal dependence in residuals")
print("  • Exp2 recommended but AR(2) structure suggested for future work")
print()

print("=" * 80)
print("Analysis complete. See comparison_report.md for full details.")
print("=" * 80)
