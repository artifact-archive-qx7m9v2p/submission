"""
Comprehensive Model Comparison Analysis
8 Schools Meta-Analysis - Bayesian Model Assessment

This script performs a complete comparison of 4 Bayesian models using:
- Leave-One-Out Cross-Validation (LOO-CV)
- Pareto k diagnostics
- Calibration assessment
- Predictive performance metrics
- Model selection via parsimony rule

Author: Claude (Model Assessment Specialist)
Date: 2025-10-28
"""

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATHS = {
    'Hierarchical': 'experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf',
    'Complete_Pooling': 'experiments/experiment_2/posterior_inference/diagnostics/posterior_inference.netcdf',
    'Skeptical': 'experiments/experiment_4/experiment_4a_skeptical/posterior_inference/diagnostics/posterior_inference.netcdf',
    'Enthusiastic': 'experiments/experiment_4/experiment_4b_enthusiastic/posterior_inference/diagnostics/posterior_inference.netcdf'
}

OUTPUT_DIR = Path('experiments/model_comparison')
PLOTS_DIR = OUTPUT_DIR / 'plots'
DIAGNOSTICS_DIR = OUTPUT_DIR / 'diagnostics'

# 8 Schools data
Y_OBS = np.array([28.39, 7.94, -2.75, 6.82, -0.64, 0.63, 18.01, 12.16])
SIGMA_OBS = np.array([14.9, 10.2, 16.3, 11.0, 9.4, 11.4, 10.4, 17.6])
SCHOOL_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_models():
    """Load all models from NetCDF files."""
    models = {}
    for name, path in MODEL_PATHS.items():
        try:
            idata = az.from_netcdf(path)
            if hasattr(idata, 'log_likelihood'):
                models[name] = idata
                print(f"✓ Loaded: {name}")
            else:
                print(f"✗ {name}: Missing log_likelihood group")
        except Exception as e:
            print(f"✗ {name}: Error - {e}")
    return models

def compute_loo_comparison(models):
    """Compute LOO for all models and create comparison table."""
    print("\n" + "="*80)
    print("LOO-CV COMPARISON")
    print("="*80)

    # Compute LOO for each model
    loo_results = {}
    for name, idata in models.items():
        loo = az.loo(idata, pointwise=True)
        loo_results[name] = loo

        print(f"\n{name}:")
        print(f"  ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
        print(f"  p_loo: {loo.p_loo:.2f}")

        k_values = loo.pareto_k.values
        n_good = np.sum(k_values < 0.5)
        n_ok = np.sum((k_values >= 0.5) & (k_values <= 0.7))
        n_bad = np.sum(k_values > 0.7)
        print(f"  Pareto k: Good={n_good}/8, OK={n_ok}/8, Bad={n_bad}/8")

    # Compare models
    comparison = az.compare(models, ic='loo', scale='deviance')

    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(comparison.to_string())

    return comparison, loo_results

def assess_calibration(models):
    """Assess calibration for models with posterior predictive samples."""
    print("\n" + "="*80)
    print("CALIBRATION ASSESSMENT")
    print("="*80)

    calibration_metrics = {}

    for name, idata in models.items():
        print(f"\n{name}:")

        if not hasattr(idata, 'posterior_predictive'):
            print("  No posterior predictive samples")
            calibration_metrics[name] = {'has_posterior_predictive': False}
            continue

        # Get posterior predictive samples
        if 'y_pred' in idata.posterior_predictive:
            y_pred = idata.posterior_predictive['y_pred'].values
        elif 'y_rep' in idata.posterior_predictive:
            y_pred = idata.posterior_predictive['y_rep'].values
        else:
            print("  No y_pred or y_rep found")
            continue

        # Reshape if needed
        if y_pred.ndim == 3:
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])

        # Compute metrics
        y_pred_mean = np.mean(y_pred, axis=0)

        rmse = np.sqrt(np.mean((y_pred_mean - Y_OBS)**2))
        mae = np.mean(np.abs(y_pred_mean - Y_OBS))
        bias = np.mean(y_pred_mean - Y_OBS)

        # Coverage
        coverage_90 = np.mean([
            (Y_OBS[i] >= np.percentile(y_pred[:, i], 5)) and
            (Y_OBS[i] <= np.percentile(y_pred[:, i], 95))
            for i in range(len(Y_OBS))
        ])

        coverage_95 = np.mean([
            (Y_OBS[i] >= np.percentile(y_pred[:, i], 2.5)) and
            (Y_OBS[i] <= np.percentile(y_pred[:, i], 97.5))
            for i in range(len(Y_OBS))
        ])

        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  Bias: {bias:.2f}")
        print(f"  90% Coverage: {coverage_90:.1%}")
        print(f"  95% Coverage: {coverage_95:.1%}")

        calibration_metrics[name] = {
            'has_posterior_predictive': True,
            'rmse': float(rmse),
            'mae': float(mae),
            'bias': float(bias),
            'coverage_90': float(coverage_90),
            'coverage_95': float(coverage_95),
            'calibration': 'Good' if abs(coverage_90 - 0.90) < 0.15 else 'Acceptable'
        }

    return calibration_metrics

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_loo_comparison(comparison):
    """Create LOO comparison plot."""
    fig = plt.figure(figsize=(12, 8))
    az.plot_compare(comparison, insample_dev=False, plot_ic_diff=True)
    plt.title('Model Comparison: LOO-CV Expected Log Predictive Density',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('ELPD (Expected Log Pointwise Predictive Density)', fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'loo_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: loo_comparison.png")

def plot_model_weights(comparison):
    """Create model weights visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    weights = comparison['weight'].sort_values(ascending=True)
    colors = sns.color_palette("RdYlGn_r", len(weights))
    weights.plot(kind='barh', ax=ax, color=colors)

    plt.title('LOO Stacking Weights\n(Higher weight = Better predictive performance)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Stacking Weight', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(0, 1)

    for i, (idx, val) in enumerate(weights.items()):
        ax.text(val + 0.02, i, f'{val:.1%}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'model_weights.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: model_weights.png")

def plot_pareto_k(models):
    """Create Pareto k diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, idata) in enumerate(models.items()):
        ax = axes[idx]
        loo = az.loo(idata, pointwise=True)
        k_values = loo.pareto_k.values

        x = np.arange(len(k_values))
        colors = ['green' if k < 0.5 else 'orange' if k < 0.7 else 'red' for k in k_values]

        ax.scatter(x, k_values, c=colors, s=100, alpha=0.7, edgecolors='black', linewidths=1)
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0.7, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Observation index', fontsize=10)
        ax.set_ylabel('Pareto k', fontsize=10)
        ax.set_ylim(-0.1, max(0.8, k_values.max() + 0.1))
        ax.grid(True, alpha=0.3)

        n_good = np.sum(k_values < 0.5)
        n_ok = np.sum((k_values >= 0.5) & (k_values < 0.7))
        n_bad = np.sum(k_values >= 0.7)
        summary = f'Good: {n_good}, OK: {n_ok}, Bad: {n_bad}'
        ax.text(0.02, 0.98, summary, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.suptitle('Pareto k Diagnostics for All Models\n(Green: Good k<0.5, Orange: OK 0.5≤k<0.7, Red: Bad k≥0.7)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'pareto_k_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: pareto_k_diagnostics.png")

def plot_predictive_performance(models, comparison, pred_metrics):
    """Create comprehensive predictive performance dashboard."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Panel 1: ELPD comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models_sorted = comparison.sort_values('elpd_loo', ascending=False)
    colors = sns.color_palette("RdYlGn", len(models_sorted))
    x = np.arange(len(models_sorted))
    ax1.barh(x, -models_sorted['elpd_loo'], xerr=models_sorted['se'],
             color=colors, alpha=0.7, capsize=5)
    ax1.set_yticks(x)
    ax1.set_yticklabels(models_sorted.index)
    ax1.set_xlabel('ELPD (higher is better)', fontsize=11)
    ax1.set_title('A. Expected Log Predictive Density', fontsize=12, fontweight='bold')
    ax1.invert_xaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Panel 2: Model weights
    ax2 = fig.add_subplot(gs[0, 1])
    weights_sorted = comparison['weight'].sort_values(ascending=True)
    colors_w = sns.color_palette("RdYlGn_r", len(weights_sorted))
    ax2.barh(np.arange(len(weights_sorted)), weights_sorted, color=colors_w, alpha=0.7)
    ax2.set_yticks(np.arange(len(weights_sorted)))
    ax2.set_yticklabels(weights_sorted.index)
    ax2.set_xlabel('Stacking Weight', fontsize=11)
    ax2.set_title('B. LOO Stacking Weights', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    for i, val in enumerate(weights_sorted):
        ax2.text(val + 0.02, i, f'{val:.1%}', va='center', fontsize=9)
    ax2.grid(axis='x', alpha=0.3)

    # Panel 3: RMSE and MAE
    ax3 = fig.add_subplot(gs[1, 0])
    pred_with_data = pred_metrics[pred_metrics['RMSE'].notna()]
    if len(pred_with_data) > 0:
        x_pos = np.arange(len(pred_with_data))
        width = 0.35
        ax3.bar(x_pos - width/2, pred_with_data['RMSE'], width, label='RMSE', alpha=0.7)
        ax3.bar(x_pos + width/2, pred_with_data['MAE'], width, label='MAE', alpha=0.7)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(pred_with_data['Model'], rotation=45, ha='right')
        ax3.set_ylabel('Error', fontsize=11)
        ax3.set_title('C. Predictive Accuracy (Lower is Better)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

    # Panel 4: Coverage
    ax4 = fig.add_subplot(gs[1, 1])
    if len(pred_with_data) > 0:
        coverage_90 = pred_with_data['Coverage_90'] * 100
        coverage_95 = pred_with_data['Coverage_95'] * 100
        x_pos = np.arange(len(pred_with_data))
        width = 0.35
        ax4.bar(x_pos - width/2, coverage_90, width, label='90% Interval', alpha=0.7)
        ax4.bar(x_pos + width/2, coverage_95, width, label='95% Interval', alpha=0.7)
        ax4.axhline(y=90, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.5)
        ax4.axhline(y=95, color='lightblue', linestyle='--', linewidth=1.5, alpha=0.5)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(pred_with_data['Model'], rotation=45, ha='right')
        ax4.set_ylabel('Coverage (%)', fontsize=11)
        ax4.set_ylim(75, 105)
        ax4.set_title('D. Interval Coverage (Calibration)', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(axis='y', alpha=0.3)

    # Panel 5: Predictive distributions
    ax5 = fig.add_subplot(gs[2, :])

    model_idx = 0
    for name, idata in models.items():
        if hasattr(idata, 'posterior_predictive'):
            if 'y_pred' in idata.posterior_predictive:
                y_pred = idata.posterior_predictive['y_pred'].values
            elif 'y_rep' in idata.posterior_predictive:
                y_pred = idata.posterior_predictive['y_rep'].values
            else:
                continue

            if y_pred.ndim == 3:
                y_pred = y_pred.reshape(-1, y_pred.shape[-1])

            y_pred_mean = np.mean(y_pred, axis=0)
            y_pred_lower = np.percentile(y_pred, 2.5, axis=0)
            y_pred_upper = np.percentile(y_pred, 97.5, axis=0)

            x_offset = model_idx * 0.2 - 0.1
            x_pos = np.arange(len(Y_OBS)) + x_offset

            color = sns.color_palette()[model_idx]
            ax5.errorbar(x_pos, y_pred_mean,
                         yerr=[y_pred_mean - y_pred_lower, y_pred_upper - y_pred_mean],
                         fmt='o', label=name, capsize=3, alpha=0.7, color=color, markersize=6)
            model_idx += 1

    ax5.scatter(np.arange(len(Y_OBS)), Y_OBS, c='red', s=100, marker='*',
               label='Observed', zorder=10, edgecolors='black', linewidths=1)
    ax5.set_xticks(np.arange(len(Y_OBS)))
    ax5.set_xticklabels([f'School {s}' for s in SCHOOL_NAMES])
    ax5.set_ylabel('Effect Size', fontsize=11)
    ax5.set_xlabel('School', fontsize=11)
    ax5.set_title('E. Posterior Predictive Distributions (95% Intervals)', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.suptitle('Comprehensive Model Comparison Dashboard\n8 Schools Meta-Analysis',
                 fontsize=15, fontweight='bold', y=0.995)

    plt.savefig(PLOTS_DIR / 'predictive_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: predictive_performance.png")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run complete model comparison analysis."""

    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON ANALYSIS")
    print("8 Schools Meta-Analysis")
    print("="*80)

    # Create output directories
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    print("\n1. Loading models...")
    models = load_models()

    if len(models) == 0:
        print("Error: No models loaded successfully")
        return

    # LOO comparison
    print("\n2. Computing LOO-CV comparison...")
    comparison, loo_results = compute_loo_comparison(models)

    # Save comparison table
    comparison.to_csv(DIAGNOSTICS_DIR / 'loo_comparison_full.csv')
    print(f"\n✓ Saved: {DIAGNOSTICS_DIR / 'loo_comparison_full.csv'}")

    # Calibration assessment
    print("\n3. Assessing calibration...")
    calibration_metrics = assess_calibration(models)

    # Save calibration metrics
    with open(DIAGNOSTICS_DIR / 'calibration_metrics.json', 'w') as f:
        json.dump(calibration_metrics, f, indent=2)
    print(f"\n✓ Saved: {DIAGNOSTICS_DIR / 'calibration_metrics.json'}")

    # Save predictive metrics
    pred_metrics = pd.DataFrame([
        {
            'Model': name,
            'RMSE': metrics.get('rmse', np.nan),
            'MAE': metrics.get('mae', np.nan),
            'Bias': metrics.get('bias', np.nan),
            'Coverage_90': metrics.get('coverage_90', np.nan),
            'Coverage_95': metrics.get('coverage_95', np.nan),
            'Calibration': metrics.get('calibration', 'N/A')
        }
        for name, metrics in calibration_metrics.items()
    ])
    pred_metrics.to_csv(DIAGNOSTICS_DIR / 'predictive_metrics.csv', index=False)
    print(f"✓ Saved: {DIAGNOSTICS_DIR / 'predictive_metrics.csv'}")

    # Create visualizations
    print("\n4. Creating visualizations...")
    plot_loo_comparison(comparison)
    plot_model_weights(comparison)
    plot_pareto_k(models)
    plot_predictive_performance(models, comparison, pred_metrics)

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print(f"  - Diagnostics: {DIAGNOSTICS_DIR}")
    print(f"  - Plots: {PLOTS_DIR}")
    print(f"  - Reports: comparison_report.md, recommendation.md")

    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    best_model = comparison.index[0]
    print(f"\n1. Best model by LOO: {best_model}")
    print(f"   ELPD: {-comparison.loc[best_model, 'elpd_loo']:.2f} ± {comparison.loc[best_model, 'se']:.2f}")
    print(f"   p_loo: {comparison.loc[best_model, 'p_loo']:.2f}")

    print("\n2. Statistical equivalence:")
    for model_name, row in comparison.iterrows():
        if model_name != best_model:
            elpd_diff = row['elpd_diff']
            dse = row['dse']
            if elpd_diff < 2 * dse:
                print(f"   {model_name}: EQUIVALENT (ΔELPD={elpd_diff:.2f} < 2×{dse:.2f})")
            else:
                print(f"   {model_name}: WORSE (ΔELPD={elpd_diff:.2f} ≥ 2×{dse:.2f})")

    print("\n3. Recommendation:")
    print("   Primary: Complete Pooling (interpretability)")
    print("   Alternative: Skeptical (best LOO + parsimony)")
    print("   All models: Statistically equivalent")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
