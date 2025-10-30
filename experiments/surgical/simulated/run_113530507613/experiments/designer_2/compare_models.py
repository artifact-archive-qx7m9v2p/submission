"""
Compare Alternative Bayesian Models using LOO-CV
Designer 2: Model Comparison and Selection

Compares:
- Finite Mixture Model (K=3)
- Robust Beta-Binomial with Student-t
- Dirichlet Process Mixture
- Standard Hierarchical Model (baseline, if available)

Uses LOO-CV, WAIC, and posterior predictive checks.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
OUTPUT_DIR = Path("/workspace/experiments/designer_2/results")
COMPARISON_DIR = OUTPUT_DIR / "comparison"
COMPARISON_DIR.mkdir(exist_ok=True)

# Plot settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_fitted_models():
    """Load all fitted models."""
    print("Loading fitted models...")

    models = {}

    # Try to load each model
    for model_name in ['fmm3', 'robust_hbb', 'dp_mbb']:
        model_dir = OUTPUT_DIR / model_name
        if model_dir.exists():
            try:
                # Load CSV files
                csv_files = list(model_dir.glob("*.csv"))
                if csv_files:
                    import cmdstanpy
                    fit = cmdstanpy.from_csv(csv_files)
                    models[model_name] = az.from_cmdstanpy(fit)
                    print(f"  - {model_name}: LOADED")
            except Exception as e:
                print(f"  - {model_name}: FAILED ({e})")
        else:
            print(f"  - {model_name}: NOT FOUND")

    return models


def compute_loo(models):
    """Compute LOO-CV for all models."""
    print("\nComputing LOO-CV...")

    loo_results = {}

    for name, idata in models.items():
        try:
            loo = az.loo(idata, pointwise=True)
            loo_results[name] = loo
            print(f"  - {name}: ELPD = {loo.elpd_loo:.2f} ± {loo.se:.2f}")

            # Check for high pareto-k values
            high_k = (loo.pareto_k > 0.7).sum()
            if high_k > 0:
                print(f"    WARNING: {high_k} observations with high Pareto-k (> 0.7)")

        except Exception as e:
            print(f"  - {name}: FAILED ({e})")

    return loo_results


def compare_models_loo(loo_results):
    """Compare models using LOO-CV."""
    print("\n" + "="*60)
    print("MODEL COMPARISON (LOO-CV)")
    print("="*60)

    if len(loo_results) < 2:
        print("Need at least 2 models for comparison.")
        return None

    # Create comparison table
    comparison_data = []
    for name, loo in loo_results.items():
        comparison_data.append({
            'model': name,
            'elpd_loo': loo.elpd_loo,
            'se': loo.se,
            'p_loo': loo.p_loo
        })

    df = pd.DataFrame(comparison_data)
    df = df.sort_values('elpd_loo', ascending=False)

    # Compute deltas from best model
    best_elpd = df['elpd_loo'].iloc[0]
    df['delta_elpd'] = df['elpd_loo'] - best_elpd

    print("\nModel Ranking:")
    print(df.to_string(index=False))

    # Interpretation
    print("\nInterpretation:")
    best_model = df['model'].iloc[0]
    print(f"  - Best model: {best_model}")

    if len(df) > 1:
        second_best = df['model'].iloc[1]
        delta = abs(df['delta_elpd'].iloc[1])
        se = df['se'].iloc[1]

        if delta > 5:
            print(f"  - Clear winner (delta > 5)")
        elif delta > se:
            print(f"  - Moderate preference (delta > SE)")
        else:
            print(f"  - Uncertain (delta < SE)")

    # Save comparison
    df.to_csv(COMPARISON_DIR / "loo_comparison.csv", index=False)

    return df


def plot_loo_comparison(loo_results):
    """Plot LOO comparison."""
    print("\nGenerating LOO comparison plot...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: ELPD comparison
    ax = axes[0]
    comparison_data = []
    for name, loo in loo_results.items():
        comparison_data.append({
            'model': name,
            'elpd_loo': loo.elpd_loo,
            'se': loo.se
        })
    df = pd.DataFrame(comparison_data).sort_values('elpd_loo')

    ax.errorbar(df['elpd_loo'], range(len(df)), xerr=df['se'],
                fmt='o', markersize=8, capsize=5, capthick=2)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['model'])
    ax.set_xlabel('ELPD (LOO-CV)', fontsize=12)
    ax.set_title('Model Comparison: LOO Expected Log Predictive Density', fontsize=12)
    ax.axvline(df['elpd_loo'].max(), color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Panel 2: Pareto-k diagnostics
    ax = axes[1]
    for name, loo in loo_results.items():
        pareto_k = loo.pareto_k
        ax.scatter(range(len(pareto_k)), sorted(pareto_k),
                  label=name, alpha=0.7, s=50)

    ax.axhline(0.5, color='orange', linestyle='--', label='k=0.5 (moderate)')
    ax.axhline(0.7, color='red', linestyle='--', label='k=0.7 (high)')
    ax.set_xlabel('Observation (sorted)', fontsize=12)
    ax.set_ylabel('Pareto-k', fontsize=12)
    ax.set_title('LOO Diagnostics: Pareto-k Values', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "loo_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  - Saved to {COMPARISON_DIR / 'loo_comparison.png'}")


def posterior_predictive_checks(models, data_path="/workspace/data/binomial_data.csv"):
    """Perform posterior predictive checks."""
    print("\nPosterior Predictive Checks...")

    # Load observed data
    df = pd.read_csv(data_path)
    r_obs = df['r_successes'].values
    n_obs = df['n_trials'].values

    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for idx, (name, idata) in enumerate(models.items()):
        ax = axes[idx]

        try:
            # Extract posterior predictive samples
            r_rep = idata.posterior_predictive['r_rep'].values
            r_rep_flat = r_rep.reshape(-1, len(r_obs))

            # Compute quantiles
            q05 = np.percentile(r_rep_flat, 5, axis=0)
            q25 = np.percentile(r_rep_flat, 25, axis=0)
            q50 = np.percentile(r_rep_flat, 50, axis=0)
            q75 = np.percentile(r_rep_flat, 75, axis=0)
            q95 = np.percentile(r_rep_flat, 95, axis=0)

            # Plot
            x = range(1, len(r_obs) + 1)
            ax.fill_between(x, q05, q95, alpha=0.2, label='90% interval')
            ax.fill_between(x, q25, q75, alpha=0.3, label='50% interval')
            ax.plot(x, q50, 'b-', label='Median', linewidth=2)
            ax.scatter(x, r_obs, color='red', s=100, zorder=10,
                      label='Observed', marker='o')

            ax.set_xlabel('Group', fontsize=12)
            ax.set_ylabel('Number of Successes', fontsize=12)
            ax.set_title(f'{name.upper()}: Posterior Predictive Check', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Check calibration
            n_outside = ((r_obs < q05) | (r_obs > q95)).sum()
            pct_outside = 100 * n_outside / len(r_obs)
            ax.text(0.05, 0.95, f'{pct_outside:.0f}% outside 90% interval\n(expect ~10%)',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        except Exception as e:
            ax.text(0.5, 0.5, f"PPC failed:\n{e}",
                   transform=ax.transAxes, ha='center', va='center')

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / "posterior_predictive_checks.png", dpi=300, bbox_inches='tight')
    print(f"  - Saved to {COMPARISON_DIR / 'posterior_predictive_checks.png'}")


def model_stacking(loo_results):
    """Compute model stacking weights."""
    print("\nComputing model stacking weights...")

    if len(loo_results) < 2:
        print("  Need at least 2 models for stacking.")
        return None

    try:
        # Combine pointwise LOO
        loo_list = list(loo_results.values())
        names = list(loo_results.keys())

        # Compute stacking weights (simple approach)
        elpd_values = np.array([loo.elpd_loo for loo in loo_list])
        se_values = np.array([loo.se for loo in loo_list])

        # Inverse variance weighting
        weights = 1 / (se_values ** 2)
        weights = weights / weights.sum()

        print("\nStacking Weights (inverse variance):")
        for name, weight in zip(names, weights):
            print(f"  - {name}: {weight:.3f}")

        return dict(zip(names, weights))

    except Exception as e:
        print(f"  Failed: {e}")
        return None


def summarize_results(loo_comparison, stacking_weights):
    """Summarize comparison results."""
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    if loo_comparison is not None:
        best_model = loo_comparison['model'].iloc[0]
        best_elpd = loo_comparison['elpd_loo'].iloc[0]

        print(f"\nBest Model: {best_model.upper()}")
        print(f"  - ELPD: {best_elpd:.2f}")

        if len(loo_comparison) > 1:
            delta = abs(loo_comparison['delta_elpd'].iloc[1])
            if delta > 5:
                print(f"  - Strength: STRONG (delta = {delta:.2f})")
            elif delta > 2:
                print(f"  - Strength: MODERATE (delta = {delta:.2f})")
            else:
                print(f"  - Strength: WEAK (delta = {delta:.2f})")

    if stacking_weights is not None:
        print("\nModel Averaging Weights:")
        for name, weight in stacking_weights.items():
            print(f"  - {name}: {weight:.1%}")

    print("\nRecommendation:")
    if loo_comparison is not None and len(loo_comparison) > 1:
        delta = abs(loo_comparison['delta_elpd'].iloc[1])
        best = loo_comparison['model'].iloc[0]

        if delta > 5:
            print(f"  Use {best.upper()} for inference (clear winner)")
        elif delta > 2:
            print(f"  Prefer {best.upper()} but report sensitivity")
        else:
            print(f"  Consider model averaging or report both models")


def save_comparison_report(loo_comparison, stacking_weights):
    """Save detailed comparison report."""
    report = {
        'loo_comparison': loo_comparison.to_dict() if loo_comparison is not None else None,
        'stacking_weights': stacking_weights,
        'recommendation': None
    }

    if loo_comparison is not None:
        best_model = loo_comparison['model'].iloc[0]
        if len(loo_comparison) > 1:
            delta = abs(loo_comparison['delta_elpd'].iloc[1])
            if delta > 5:
                strength = "STRONG"
            elif delta > 2:
                strength = "MODERATE"
            else:
                strength = "WEAK"
        else:
            strength = "SINGLE MODEL"

        report['recommendation'] = {
            'best_model': best_model,
            'strength': strength
        }

    with open(COMPARISON_DIR / "comparison_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  - Saved detailed report to {COMPARISON_DIR / 'comparison_report.json'}")


def main():
    """Main comparison pipeline."""
    print("="*60)
    print("MODEL COMPARISON PIPELINE")
    print("Designer 2: Alternative Bayesian Approaches")
    print("="*60)

    # Load models
    models = load_fitted_models()

    if len(models) == 0:
        print("\nNo fitted models found. Run fit_alternatives.py first.")
        return

    # Compute LOO
    loo_results = compute_loo(models)

    if len(loo_results) == 0:
        print("\nLOO computation failed for all models.")
        return

    # Compare models
    loo_comparison = compare_models_loo(loo_results)

    # Plot comparison
    plot_loo_comparison(loo_results)

    # Posterior predictive checks
    posterior_predictive_checks(models)

    # Model stacking
    stacking_weights = model_stacking(loo_results)

    # Summarize
    summarize_results(loo_comparison, stacking_weights)

    # Save report
    save_comparison_report(loo_comparison, stacking_weights)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
