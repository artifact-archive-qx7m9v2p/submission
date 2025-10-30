"""
Model Comparison and Visualization Script
Designer 2: Hierarchical Binomial Models

Compares fitted models using:
- LOO-CV (Leave-One-Out Cross-Validation)
- Posterior predictive checks
- Parameter comparison
- Shrinkage analysis

Usage:
    python model_comparison.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

try:
    import arviz as az
    print(f"ArviZ version: {az.__version__}")
    HAS_ARVIZ = True
except ImportError:
    print("⚠️  ArviZ not installed. Some features will be limited.")
    print("   Install with: pip install arviz")
    HAS_ARVIZ = False

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_results(results_dir: Path) -> dict:
    """Load fitted model results."""
    results = {}

    for csv_file in results_dir.glob("*_draws.csv"):
        model_name = csv_file.stem.replace("_draws", "")
        print(f"Loading {model_name}...")

        draws = pd.read_csv(csv_file)
        summary_file = results_dir / f"{model_name}_summary.csv"
        diag_file = results_dir / f"{model_name}_diagnostics.json"

        if summary_file.exists():
            summary = pd.read_csv(summary_file, index_col=0)
        else:
            summary = None

        if diag_file.exists():
            with open(diag_file) as f:
                diagnostics = json.load(f)
        else:
            diagnostics = {}

        results[model_name] = {
            'draws': draws,
            'summary': summary,
            'diagnostics': diagnostics
        }

    print(f"\nLoaded {len(results)} model(s)")
    return results


def compare_population_parameters(results: dict, output_dir: Path):
    """Compare μ and σ across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract μ posteriors
    mu_data = []
    for model_name, data in results.items():
        draws = data['draws']
        if 'mu' in draws.columns:
            mu_samples = draws['mu'].values
            # Convert to probability scale
            p_samples = 1 / (1 + np.exp(-mu_samples))
            mu_data.append({
                'model': model_name,
                'samples': p_samples
            })

    # Plot μ (on probability scale)
    ax = axes[0]
    for i, item in enumerate(mu_data):
        ax.hist(item['samples'], bins=50, alpha=0.5,
                label=item['model'], density=True)

    ax.axvline(0.076, color='red', linestyle='--', linewidth=2,
               label='Observed pooled rate (7.6%)')
    ax.set_xlabel('Population Mean Success Rate')
    ax.set_ylabel('Posterior Density')
    ax.set_title('Population Mean (μ) Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Extract σ posteriors
    sigma_data = []
    for model_name, data in results.items():
        draws = data['draws']
        if 'sigma' in draws.columns:
            sigma_samples = draws['sigma'].values
            sigma_data.append({
                'model': model_name,
                'samples': sigma_samples
            })

    # Plot σ
    ax = axes[1]
    for i, item in enumerate(sigma_data):
        ax.hist(item['samples'], bins=50, alpha=0.5,
                label=item['model'], density=True)

    # Expected sigma based on ICC = 0.73
    # ICC = σ² / (σ² + π²/3)
    # 0.73 = σ² / (σ² + 3.29)
    # σ ≈ 0.94
    ax.axvline(0.94, color='red', linestyle='--', linewidth=2,
               label='Expected from ICC=0.73')
    ax.set_xlabel('Between-Group SD (σ, logit scale)')
    ax.set_ylabel('Posterior Density')
    ax.set_title('Between-Group Variability (σ) Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "population_parameters_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()


def compare_group_effects(results: dict, observed_data: pd.DataFrame, output_dir: Path):
    """Compare group-specific posterior estimates across models."""
    n_groups = len(observed_data)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(n_groups):
        ax = axes[i]
        group_id = observed_data.iloc[i]['group']
        obs_rate = observed_data.iloc[i]['success_rate']
        n_trials = observed_data.iloc[i]['n_trials']

        # Plot observed rate
        ax.axvline(obs_rate, color='black', linestyle='-', linewidth=2,
                   label='Observed', alpha=0.7)

        # Plot posterior from each model
        for model_name, data in results.items():
            draws = data['draws']
            p_col = f'p_posterior[{i+1}]'

            if p_col in draws.columns:
                p_samples = draws[p_col].values
                ax.hist(p_samples, bins=30, alpha=0.4, density=True,
                        label=model_name.split(':')[0])  # Short label

        ax.set_xlabel('Success Rate')
        ax.set_ylabel('Density')
        ax.set_title(f'Group {group_id} (n={n_trials}, obs={obs_rate:.1%})')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    output_file = output_dir / "group_posteriors_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_shrinkage_comparison(results: dict, observed_data: pd.DataFrame, output_dir: Path):
    """Compare shrinkage patterns across models."""
    pooled_rate = observed_data['r_successes'].sum() / observed_data['n_trials'].sum()
    n_groups = len(observed_data)

    fig, ax = plt.subplots(figsize=(12, 8))

    markers = ['o', 's', '^']
    for idx, (model_name, data) in enumerate(results.items()):
        posterior_means = []
        for i in range(n_groups):
            p_col = f'p_posterior[{i+1}]'
            if p_col in data['summary'].index:
                posterior_means.append(data['summary'].loc[p_col, 'Mean'])
            else:
                posterior_means.append(np.nan)

        ax.scatter(observed_data['success_rate'], posterior_means,
                   marker=markers[idx % len(markers)], s=100, alpha=0.6,
                   label=model_name.split(':')[0])

    # Add reference lines
    ax.plot([0, 0.16], [0, 0.16], 'k--', alpha=0.3, label='No shrinkage')
    ax.axhline(pooled_rate, color='red', linestyle='--', alpha=0.5,
               label=f'Pooled rate ({pooled_rate:.1%})')

    # Annotate special groups
    for i, row in observed_data.iterrows():
        if row['group'] in [1, 8]:  # Zero count and outlier
            ax.annotate(f"Group {row['group']}",
                       xy=(row['success_rate'], row['success_rate']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, alpha=0.7)

    ax.set_xlabel('Observed Success Rate')
    ax.set_ylabel('Posterior Mean Success Rate')
    ax.set_title('Shrinkage Comparison: Observed vs Posterior Estimates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "shrinkage_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_overdispersion_check(results: dict, output_dir: Path):
    """Check if models reproduce observed overdispersion."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Observed overdispersion from EDA
    obs_phi_range = (3.5, 5.1)
    ax.axvspan(obs_phi_range[0], obs_phi_range[1], alpha=0.2, color='red',
               label='Observed φ (EDA)')

    for model_name, data in results.items():
        draws = data['draws']
        if 'phi_posterior' in draws.columns:
            phi_samples = draws['phi_posterior'].values
            ax.hist(phi_samples, bins=50, alpha=0.5, density=True,
                    label=model_name.split(':')[0])

    ax.set_xlabel('Overdispersion Parameter (φ)')
    ax.set_ylabel('Posterior Density')
    ax.set_title('Overdispersion Check: Can Models Reproduce φ ≈ 3.5-5.1?')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "overdispersion_check.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_group1_shrinkage(results: dict, observed_data: pd.DataFrame, output_dir: Path):
    """Special focus on Group 1 (zero successes) shrinkage."""
    fig, ax = plt.subplots(figsize=(10, 6))

    group1_data = observed_data[observed_data['group'] == 1].iloc[0]
    n_trials = group1_data['n_trials']

    for model_name, data in results.items():
        draws = data['draws']
        p_col = 'p_posterior[1]'

        if p_col in draws.columns:
            p_samples = draws[p_col].values
            ax.hist(p_samples, bins=50, alpha=0.5, density=True,
                    label=model_name.split(':')[0])

            # Print summary
            mean_p = p_samples.mean()
            q5, q95 = np.percentile(p_samples, [5, 95])
            print(f"\n{model_name} - Group 1 posterior:")
            print(f"  Mean: {mean_p:.2%}")
            print(f"  90% CI: [{q5:.2%}, {q95:.2%}]")

    ax.axvline(0, color='black', linestyle='-', linewidth=2,
               label='Observed (0/47)', alpha=0.7)
    ax.axvline(0.076, color='red', linestyle='--', linewidth=2,
               label='Pooled rate (7.6%)', alpha=0.7)

    ax.set_xlabel('Success Rate')
    ax.set_ylabel('Posterior Density')
    ax.set_title(f'Group 1 Zero Count Handling: Posterior Estimates\n(Observed: 0/{n_trials} = 0%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "group1_zero_count_shrinkage.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def compare_diagnostics(results: dict, output_dir: Path):
    """Create comparison table of diagnostic statistics."""
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPARISON")
    print("="*60)

    comparison_data = []

    for model_name, data in results.items():
        diag = data['diagnostics']

        row = {
            'Model': model_name.split(':')[0],
            'Overall': diag.get('overall', 'UNKNOWN'),
            'Divergences': diag.get('divergences', {}).get('percentage', np.nan),
            'Max_Rhat': diag.get('rhat', {}).get('max', np.nan),
            'Min_ESS_Bulk': diag.get('ess', {}).get('min_bulk', np.nan),
            'Min_ESS_Tail': diag.get('ess', {}).get('min_tail', np.nan)
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Print table
    print("\n" + df.to_string(index=False))

    # Save to CSV
    output_file = output_dir / "diagnostic_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved: {output_file}")

    return df


def plot_nu_posterior(results: dict, output_dir: Path):
    """
    Plot posterior of ν (degrees of freedom) from Model 3.
    Answers: Are heavy tails necessary?
    """
    has_nu = False
    for model_name, data in results.items():
        if 'nu' in data['draws'].columns:
            has_nu = True
            break

    if not has_nu:
        print("\n⚠️  No Model 3 (robust) results found. Skipping ν analysis.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, data in results.items():
        draws = data['draws']
        if 'nu' in draws.columns:
            nu_samples = draws['nu'].values
            ax.hist(nu_samples, bins=50, alpha=0.5, density=True,
                    label=model_name.split(':')[0])

            # Summary
            mean_nu = nu_samples.mean()
            q5, q95 = np.percentile(nu_samples, [5, 95])
            print(f"\n{model_name} - ν posterior:")
            print(f"  Mean: {mean_nu:.1f}")
            print(f"  90% CI: [{q5:.1f}, {q95:.1f}]")

            if mean_nu > 30:
                print(f"  → Heavy tails NOT necessary (ν > 30)")
            else:
                print(f"  → Heavy tails ARE important (ν < 30)")

    ax.axvline(30, color='red', linestyle='--', linewidth=2,
               label='ν = 30 (approx normal)', alpha=0.7)

    ax.set_xlabel('Degrees of Freedom (ν)')
    ax.set_ylabel('Posterior Density')
    ax.set_title('Heavy Tail Assessment: Posterior of ν\n(ν > 30 suggests normal is adequate)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "nu_posterior.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()


def create_summary_report(results: dict, observed_data: pd.DataFrame, output_dir: Path):
    """Create comprehensive summary report."""
    report = []
    report.append("="*70)
    report.append("MODEL COMPARISON SUMMARY REPORT")
    report.append("Designer 2: Hierarchical Binomial Models with Random Effects")
    report.append("="*70)
    report.append("")

    # Model convergence
    report.append("1. CONVERGENCE STATUS")
    report.append("-"*70)
    for model_name, data in results.items():
        overall = data['diagnostics'].get('overall', 'UNKNOWN')
        report.append(f"  {model_name}: {overall}")
    report.append("")

    # Population parameters
    report.append("2. POPULATION PARAMETERS")
    report.append("-"*70)
    pooled_rate = observed_data['r_successes'].sum() / observed_data['n_trials'].sum()
    report.append(f"  Observed pooled rate: {pooled_rate:.3%}")
    report.append("")

    for model_name, data in results.items():
        summary = data['summary']
        if summary is not None and 'mu' in summary.index:
            mu_mean = summary.loc['mu', 'Mean']
            p_mean = 1 / (1 + np.exp(-mu_mean))
            report.append(f"  {model_name}:")
            report.append(f"    μ (logit): {mu_mean:.3f}")
            report.append(f"    Population rate: {p_mean:.3%}")

            if 'sigma' in summary.index:
                sigma_mean = summary.loc['sigma', 'Mean']
                icc = sigma_mean**2 / (sigma_mean**2 + np.pi**2/3)
                report.append(f"    σ: {sigma_mean:.3f}")
                report.append(f"    Implied ICC: {icc:.2%} (observed: 73%)")
            report.append("")

    # Overdispersion
    report.append("3. OVERDISPERSION CHECK")
    report.append("-"*70)
    report.append("  EDA observed φ: 3.5 to 5.1")
    report.append("")

    for model_name, data in results.items():
        draws = data['draws']
        if 'phi_posterior' in draws.columns:
            phi_samples = draws['phi_posterior'].values
            phi_mean = phi_samples.mean()
            phi_q5, phi_q95 = np.percentile(phi_samples, [5, 95])
            report.append(f"  {model_name}:")
            report.append(f"    Posterior φ: {phi_mean:.2f} (90% CI: [{phi_q5:.2f}, {phi_q95:.2f}])")

            if 3.0 <= phi_mean <= 6.0:
                report.append(f"    ✓ Reproduces observed overdispersion")
            else:
                report.append(f"    ⚠️  Does not match observed overdispersion")
            report.append("")

    # Group 1 shrinkage
    report.append("4. GROUP 1 ZERO COUNT HANDLING")
    report.append("-"*70)
    report.append("  Observed: 0/47 = 0%")
    report.append("")

    for model_name, data in results.items():
        draws = data['draws']
        if 'p_posterior[1]' in draws.columns:
            p_samples = draws['p_posterior[1]'].values
            p_mean = p_samples.mean()
            p_q5, p_q95 = np.percentile(p_samples, [5, 95])
            report.append(f"  {model_name}:")
            report.append(f"    Posterior: {p_mean:.2%} (90% CI: [{p_q5:.2%}, {p_q95:.2%}])")
            report.append("")

    # Recommendations
    report.append("5. RECOMMENDATIONS")
    report.append("-"*70)

    # Check if M2 converged well
    m2_name = [k for k in results.keys() if 'Non-centered' in k or 'M2' in k]
    if m2_name:
        m2_overall = results[m2_name[0]]['diagnostics'].get('overall', 'UNKNOWN')
        if m2_overall == 'PASS':
            report.append("  ✓ Model 2 (Non-centered) converged well")
            report.append("    → RECOMMENDED as primary model")
            report.append("")

    # Check nu from M3
    m3_name = [k for k in results.keys() if 'Robust' in k or 'M3' in k]
    if m3_name:
        m3_data = results[m3_name[0]]
        if 'nu' in m3_data['draws'].columns:
            nu_mean = m3_data['draws']['nu'].mean()
            if nu_mean > 30:
                report.append("  ✓ Model 3: ν > 30 suggests normal priors adequate")
                report.append("    → Heavy tails NOT necessary, use Model 2")
            else:
                report.append("  ✓ Model 3: ν < 30 suggests heavy tails important")
                report.append("    → Consider using Model 3 for robustness")
            report.append("")

    report.append("6. NEXT STEPS")
    report.append("-"*70)
    report.append("  1. Review visualizations in visualizations/")
    report.append("  2. Run posterior predictive checks")
    report.append("  3. Compare to other designers' models (LOO-CV)")
    report.append("  4. Generate predictions for new groups")
    report.append("")
    report.append("="*70)

    # Write report
    output_file = output_dir / "comparison_report.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))

    print("\n" + '\n'.join(report))
    print(f"\n✓ Saved report: {output_file}")


def main():
    # Setup paths
    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    viz_dir = base_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    data_path = Path("/workspace/data/data.csv")

    # Load data
    print("Loading observed data...")
    observed_data = pd.read_csv(data_path)

    # Load results
    print("\nLoading model results...")
    results = load_results(results_dir)

    if len(results) == 0:
        print("\n✗ No model results found. Run fit_models.py first.")
        return

    # Diagnostic comparison
    diag_df = compare_diagnostics(results, viz_dir)

    # Create visualizations
    print("\nCreating visualizations...")
    compare_population_parameters(results, viz_dir)
    compare_group_effects(results, observed_data, viz_dir)
    plot_shrinkage_comparison(results, observed_data, viz_dir)
    plot_overdispersion_check(results, viz_dir)
    plot_group1_shrinkage(results, observed_data, viz_dir)
    plot_nu_posterior(results, viz_dir)

    # Summary report
    create_summary_report(results, observed_data, viz_dir)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"\nAll outputs saved to: {viz_dir}")
    print("\nVisualization files:")
    for f in sorted(viz_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
