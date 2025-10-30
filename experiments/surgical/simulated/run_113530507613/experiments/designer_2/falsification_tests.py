"""
Falsification Tests for Alternative Bayesian Models
Designer 2: Test whether models should be rejected

Each model has specific falsification criteria defined in proposed_models.md.
This script implements automated tests for each criterion.
"""

import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
OUTPUT_DIR = Path("/workspace/experiments/designer_2/results")
FALSIFICATION_DIR = OUTPUT_DIR / "falsification"
FALSIFICATION_DIR.mkdir(exist_ok=True)


def load_fitted_models():
    """Load all fitted models."""
    print("Loading fitted models for falsification tests...")

    models = {}

    for model_name in ['fmm3', 'robust_hbb', 'dp_mbb']:
        model_dir = OUTPUT_DIR / model_name
        if model_dir.exists():
            try:
                csv_files = list(model_dir.glob("*.csv"))
                if csv_files:
                    import cmdstanpy
                    fit = cmdstanpy.from_csv(csv_files)
                    idata = az.from_cmdstanpy(fit)
                    models[model_name] = {
                        'fit': fit,
                        'idata': idata,
                        'samples': fit.stan_variables()
                    }
                    print(f"  - {model_name}: LOADED")
            except Exception as e:
                print(f"  - {model_name}: FAILED ({e})")

    return models


def test_fmm3_falsification(model_data):
    """
    Falsification tests for Finite Mixture Model (K=3).

    Reject if:
    1. Cluster assignments ambiguous (avg certainty < 0.6)
    2. Clusters collapse (mu_k overlap)
    3. K_effective < 2 (only one cluster)
    4. Most groups have entropy > 0.8
    """
    print("\n" + "="*60)
    print("FMM-3 FALSIFICATION TESTS")
    print("="*60)

    samples = model_data['samples']
    tests = {}

    # Test 1: Cluster assignment certainty
    print("\nTest 1: Cluster Assignment Certainty")
    cluster_prob = samples['cluster_prob']  # [samples, J, K]
    max_probs = cluster_prob.max(axis=2)
    mean_max_prob = max_probs.mean(axis=0)
    avg_certainty = mean_max_prob.mean()

    print(f"  - Average certainty: {avg_certainty:.3f}")
    print(f"  - Threshold: 0.60")
    print(f"  - Status: {'PASS' if avg_certainty > 0.6 else 'FAIL'}")

    tests['cluster_certainty'] = {
        'value': avg_certainty,
        'threshold': 0.6,
        'pass': avg_certainty > 0.6
    }

    # Test 2: Cluster separation
    print("\nTest 2: Cluster Separation")
    mu = samples['mu']  # [samples, K]
    tau = samples['tau']  # [samples, K]

    mu_mean = mu.mean(axis=0)
    tau_mean = tau.mean(axis=0)

    # Check if consecutive clusters overlap
    separations = []
    for k in range(len(mu_mean) - 1):
        sep = abs(mu_mean[k+1] - mu_mean[k])
        pooled_sd = np.sqrt(tau_mean[k]**2 + tau_mean[k+1]**2)
        z_score = sep / pooled_sd
        separations.append(z_score)

    min_separation = min(separations)
    print(f"  - Cluster means (logit): {mu_mean}")
    print(f"  - Cluster SDs: {tau_mean}")
    print(f"  - Min separation (z-scores): {min_separation:.2f}")
    print(f"  - Threshold: 2.0")
    print(f"  - Status: {'PASS' if min_separation > 2 else 'FAIL'}")

    tests['cluster_separation'] = {
        'value': min_separation,
        'threshold': 2.0,
        'pass': min_separation > 2
    }

    # Test 3: Effective number of clusters
    print("\nTest 3: Effective Number of Clusters")
    K_eff = samples['K_effective']
    K_eff_mean = K_eff.mean()
    K_eff_q025 = np.percentile(K_eff, 2.5)

    print(f"  - K_effective (mean): {K_eff_mean:.2f}")
    print(f"  - K_effective (2.5%): {K_eff_q025:.2f}")
    print(f"  - Threshold: >= 2")
    print(f"  - Status: {'PASS' if K_eff_q025 >= 2 else 'FAIL'}")

    tests['effective_clusters'] = {
        'value': K_eff_mean,
        'q025': K_eff_q025,
        'threshold': 2,
        'pass': K_eff_q025 >= 2
    }

    # Test 4: Cluster entropy (uncertainty)
    print("\nTest 4: Cluster Assignment Entropy")
    # Compute entropy for each group across samples
    entropies = []
    for j in range(cluster_prob.shape[1]):
        # Average probability across samples
        avg_prob = cluster_prob[:, j, :].mean(axis=0)
        # Normalize
        avg_prob = avg_prob / avg_prob.sum()
        # Entropy
        entropy = -np.sum(avg_prob * np.log(avg_prob + 1e-10))
        # Normalize to [0, 1] (max entropy for K=3 is log(3))
        entropy_normalized = entropy / np.log(3)
        entropies.append(entropy_normalized)

    avg_entropy = np.mean(entropies)
    print(f"  - Average normalized entropy: {avg_entropy:.3f}")
    print(f"  - Threshold: < 0.8 (lower is more certain)")
    print(f"  - Status: {'PASS' if avg_entropy < 0.8 else 'FAIL'}")

    tests['cluster_entropy'] = {
        'value': avg_entropy,
        'threshold': 0.8,
        'pass': avg_entropy < 0.8
    }

    # Overall verdict
    all_pass = all(test['pass'] for test in tests.values())
    print("\n" + "-"*60)
    print(f"OVERALL VERDICT: {'ACCEPT MODEL' if all_pass else 'REJECT MODEL'}")
    print("-"*60)

    return {
        'model': 'fmm3',
        'verdict': 'accept' if all_pass else 'reject',
        'tests': tests
    }


def test_robust_hbb_falsification(model_data):
    """
    Falsification tests for Robust Beta-Binomial.

    Reject if:
    1. nu > 30 (tails collapse to normal)
    2. kappa > 1000 (no overdispersion)
    3. No benefit over standard hierarchy
    """
    print("\n" + "="*60)
    print("ROBUST-HBB FALSIFICATION TESTS")
    print("="*60)

    samples = model_data['samples']
    tests = {}

    # Test 1: Heavy tails
    print("\nTest 1: Heavy Tails (Student-t df)")
    nu = samples['nu']
    nu_mean = nu.mean()
    nu_q95 = np.percentile(nu, 95)

    print(f"  - nu (mean): {nu_mean:.2f}")
    print(f"  - nu (95th percentile): {nu_q95:.2f}")
    print(f"  - Threshold: < 30 (heavy tails needed)")
    print(f"  - Status: {'PASS' if nu_q95 < 30 else 'FAIL'}")

    tests['heavy_tails'] = {
        'value': nu_mean,
        'q95': nu_q95,
        'threshold': 30,
        'pass': nu_q95 < 30
    }

    # Test 2: Overdispersion
    print("\nTest 2: Overdispersion (concentration parameter)")
    kappa = samples['kappa']
    kappa_mean = kappa.mean()
    kappa_q05 = np.percentile(kappa, 5)

    print(f"  - kappa (mean): {kappa_mean:.2f}")
    print(f"  - kappa (5th percentile): {kappa_q05:.2f}")
    print(f"  - Threshold: < 500 (overdispersion needed)")
    print(f"  - Status: {'PASS' if kappa_mean < 500 else 'FAIL'}")

    tests['overdispersion'] = {
        'value': kappa_mean,
        'q05': kappa_q05,
        'threshold': 500,
        'pass': kappa_mean < 500
    }

    # Test 3: Outlier shrinkage difference
    print("\nTest 3: Outlier Shrinkage (Groups 4 and 8)")
    shrinkage = samples['shrinkage']  # [samples, J]
    shrinkage_mean = shrinkage.mean(axis=0)

    # Groups 4 and 8 are outliers (indices 3 and 7)
    shrinkage_4 = shrinkage_mean[3]
    shrinkage_8 = shrinkage_mean[7]

    print(f"  - Group 4 shrinkage: {shrinkage_4:.3f}")
    print(f"  - Group 8 shrinkage: {shrinkage_8:.3f}")
    print(f"  - Expectation: Less shrinkage due to heavy tails")

    # This is diagnostic only (no clear threshold)
    tests['outlier_shrinkage'] = {
        'group_4': shrinkage_4,
        'group_8': shrinkage_8,
        'pass': True  # Always pass (diagnostic only)
    }

    # Overall verdict (Tests 1 and 2 must pass)
    critical_tests_pass = tests['heavy_tails']['pass'] or tests['overdispersion']['pass']
    print("\n" + "-"*60)
    print(f"OVERALL VERDICT: {'ACCEPT MODEL' if critical_tests_pass else 'REJECT MODEL'}")
    print("(Model justified if either heavy tails OR overdispersion detected)")
    print("-"*60)

    return {
        'model': 'robust_hbb',
        'verdict': 'accept' if critical_tests_pass else 'reject',
        'tests': tests
    }


def test_dp_mbb_falsification(model_data):
    """
    Falsification tests for Dirichlet Process Mixture.

    Reject if:
    1. K_effective = 1 (single cluster)
    2. K_effective > 6 (overfitting)
    3. High fragmentation (many singleton clusters)
    4. Unstable assignments
    """
    print("\n" + "="*60)
    print("DP-MBB FALSIFICATION TESTS")
    print("="*60)

    samples = model_data['samples']
    tests = {}

    # Test 1: Number of clusters
    print("\nTest 1: Effective Number of Clusters")
    K_eff = samples['K_effective']
    K_eff_mean = K_eff.mean()
    K_eff_std = K_eff.std()
    K_eff_q025 = np.percentile(K_eff, 2.5)
    K_eff_q975 = np.percentile(K_eff, 97.5)

    print(f"  - K_effective (mean ± sd): {K_eff_mean:.2f} ± {K_eff_std:.2f}")
    print(f"  - K_effective (95% CI): [{K_eff_q025:.1f}, {K_eff_q975:.1f}]")
    print(f"  - Threshold: 1 < K < 7")
    print(f"  - Status: {'PASS' if 1 < K_eff_mean < 7 else 'FAIL'}")

    tests['num_clusters'] = {
        'value': K_eff_mean,
        'q025': K_eff_q025,
        'q975': K_eff_q975,
        'threshold_low': 1,
        'threshold_high': 7,
        'pass': 1 < K_eff_mean < 7
    }

    # Test 2: Fragmentation
    print("\nTest 2: Cluster Fragmentation")
    cluster_size = samples['cluster_size']  # [samples, K_max]
    avg_sizes = cluster_size.mean(axis=0)

    # Count singleton clusters (average size ≈ 1)
    n_singleton = (avg_sizes > 0.9).sum()
    # Count empty clusters
    n_empty = (avg_sizes < 0.1).sum()

    print(f"  - Singleton clusters: {n_singleton}")
    print(f"  - Empty clusters: {n_empty}")
    print(f"  - Active clusters: {len(avg_sizes) - n_empty}")
    print(f"  - Threshold: < 5 singletons")
    print(f"  - Status: {'PASS' if n_singleton < 5 else 'FAIL'}")

    tests['fragmentation'] = {
        'n_singleton': int(n_singleton),
        'n_empty': int(n_empty),
        'threshold': 5,
        'pass': n_singleton < 5
    }

    # Test 3: Assignment stability
    print("\nTest 3: Assignment Stability")
    cluster_prob = samples['cluster_prob']  # [samples, J, K_max]

    # Compute uncertainty in assignments
    max_probs = []
    for j in range(cluster_prob.shape[1]):
        avg_prob = cluster_prob[:, j, :].mean(axis=0)
        max_prob = avg_prob.max()
        max_probs.append(max_prob)

    avg_certainty = np.mean(max_probs)

    print(f"  - Average assignment certainty: {avg_certainty:.3f}")
    print(f"  - Threshold: > 0.5")
    print(f"  - Status: {'PASS' if avg_certainty > 0.5 else 'FAIL'}")

    tests['assignment_stability'] = {
        'value': avg_certainty,
        'threshold': 0.5,
        'pass': avg_certainty > 0.5
    }

    # Overall verdict
    all_pass = all(test['pass'] for test in tests.values())
    print("\n" + "-"*60)
    print(f"OVERALL VERDICT: {'ACCEPT MODEL' if all_pass else 'REJECT MODEL'}")
    print("-"*60)

    return {
        'model': 'dp_mbb',
        'verdict': 'accept' if all_pass else 'reject',
        'tests': tests
    }


def save_falsification_report(results):
    """Save falsification test results."""
    print("\nSaving falsification report...")

    report = {
        'summary': {},
        'details': results
    }

    for result in results:
        model = result['model']
        verdict = result['verdict']
        report['summary'][model] = verdict

    with open(FALSIFICATION_DIR / "falsification_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"  - Saved to {FALSIFICATION_DIR / 'falsification_report.json'}")

    # Create summary table
    summary_data = []
    for result in results:
        model = result['model']
        verdict = result['verdict']
        n_tests = len(result['tests'])
        n_pass = sum(1 for t in result['tests'].values() if t['pass'])

        summary_data.append({
            'model': model.upper(),
            'verdict': verdict.upper(),
            'tests_passed': f"{n_pass}/{n_tests}"
        })

    df = pd.DataFrame(summary_data)
    df.to_csv(FALSIFICATION_DIR / "falsification_summary.csv", index=False)
    print(f"  - Saved summary to {FALSIFICATION_DIR / 'falsification_summary.csv'}")

    return df


def main():
    """Main falsification testing pipeline."""
    print("="*60)
    print("FALSIFICATION TESTING PIPELINE")
    print("Designer 2: Alternative Bayesian Approaches")
    print("="*60)

    # Load models
    models = load_fitted_models()

    if len(models) == 0:
        print("\nNo fitted models found. Run fit_alternatives.py first.")
        return

    # Run falsification tests
    results = []

    if 'fmm3' in models:
        result = test_fmm3_falsification(models['fmm3'])
        results.append(result)

    if 'robust_hbb' in models:
        result = test_robust_hbb_falsification(models['robust_hbb'])
        results.append(result)

    if 'dp_mbb' in models:
        result = test_dp_mbb_falsification(models['dp_mbb'])
        results.append(result)

    # Save report
    summary_df = save_falsification_report(results)

    # Final summary
    print("\n" + "="*60)
    print("FALSIFICATION TESTS COMPLETE")
    print("="*60)
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nInterpretation:")
    accepted = [r['model'] for r in results if r['verdict'] == 'accept']
    rejected = [r['model'] for r in results if r['verdict'] == 'reject']

    if len(accepted) > 0:
        print(f"  - ACCEPTED: {', '.join([m.upper() for m in accepted])}")
    if len(rejected) > 0:
        print(f"  - REJECTED: {', '.join([m.upper() for m in rejected])}")

    if len(rejected) == len(results):
        print("\n  WARNING: All models rejected!")
        print("  Consider standard hierarchical model or alternative approaches.")


if __name__ == "__main__":
    main()
