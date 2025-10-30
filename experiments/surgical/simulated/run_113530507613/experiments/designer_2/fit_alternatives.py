"""
Fit Alternative Bayesian Models for Binomial Data
Designer 2: Mixture Models and Robust Approaches

This script fits three alternative model classes:
1. Finite Mixture Model (K=3)
2. Robust Beta-Binomial with Student-t
3. Dirichlet Process Mixture

All models implemented in Stan for posterior inference.
"""

import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path("/workspace/data/binomial_data.csv")
MODEL_DIR = Path("/workspace/experiments/designer_2")
OUTPUT_DIR = MODEL_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Stan compilation settings
STAN_COMPILE_OPTIONS = {
    "stanc_options": {"warn-uninitialized": True},
    "cpp_options": {"STAN_THREADS": True}
}

# Sampling settings
SAMPLING_CONFIG = {
    "chains": 4,
    "parallel_chains": 4,
    "iter_warmup": 2000,
    "iter_sampling": 2000,
    "thin": 1,
    "adapt_delta": 0.95,  # higher for complex models
    "max_treedepth": 12,
    "seed": 42
}


def load_data():
    """Load and prepare binomial data."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    data = {
        'J': len(df),
        'n': df['n_trials'].values.astype(int),
        'r': df['r_successes'].values.astype(int)
    }

    print(f"  - Groups: {data['J']}")
    print(f"  - Total trials: {data['n'].sum()}")
    print(f"  - Total successes: {data['r'].sum()}")
    print(f"  - Pooled rate: {data['r'].sum() / data['n'].sum():.4f}")

    return data, df


def compile_models():
    """Compile all Stan models."""
    models = {}

    print("\nCompiling Stan models...")

    # Model 1: Finite Mixture (K=3)
    print("  - Compiling FMM-3...")
    try:
        models['fmm3'] = cmdstanpy.CmdStanModel(
            stan_file=str(MODEL_DIR / "model_fmm3.stan"),
            **STAN_COMPILE_OPTIONS
        )
        print("    SUCCESS")
    except Exception as e:
        print(f"    FAILED: {e}")
        models['fmm3'] = None

    # Model 2: Robust Beta-Binomial
    print("  - Compiling Robust-HBB...")
    try:
        models['robust_hbb'] = cmdstanpy.CmdStanModel(
            stan_file=str(MODEL_DIR / "model_robust_hbb.stan"),
            **STAN_COMPILE_OPTIONS
        )
        print("    SUCCESS")
    except Exception as e:
        print(f"    FAILED: {e}")
        models['robust_hbb'] = None

    # Model 3: Dirichlet Process
    print("  - Compiling DP-MBB...")
    try:
        models['dp_mbb'] = cmdstanpy.CmdStanModel(
            stan_file=str(MODEL_DIR / "model_dp_mbb.stan"),
            **STAN_COMPILE_OPTIONS
        )
        print("    SUCCESS")
    except Exception as e:
        print(f"    FAILED: {e}")
        models['dp_mbb'] = None

    return models


def fit_fmm3(model, data):
    """Fit Finite Mixture Model (K=3)."""
    print("\n" + "="*60)
    print("MODEL 1: Finite Mixture Model (K=3)")
    print("="*60)

    if model is None:
        print("Model compilation failed. Skipping.")
        return None

    # Prepare data
    stan_data = {
        'J': data['J'],
        'K': 3,  # Fixed at 3 clusters
        'n': data['n'],
        'r': data['r']
    }

    print("\nFitting model...")
    try:
        fit = model.sample(data=stan_data, **SAMPLING_CONFIG)
        print("SUCCESS")

        # Quick diagnostics
        diagnostics = fit.diagnose()
        print("\nDiagnostics:")
        print(diagnostics)

        # Save fit
        fit.save_csvfiles(dir=str(OUTPUT_DIR / "fmm3"))

        return fit

    except Exception as e:
        print(f"FAILED: {e}")
        return None


def fit_robust_hbb(model, data):
    """Fit Robust Beta-Binomial with Student-t."""
    print("\n" + "="*60)
    print("MODEL 2: Robust Beta-Binomial with Student-t")
    print("="*60)

    if model is None:
        print("Model compilation failed. Skipping.")
        return None

    # Prepare data
    stan_data = {
        'J': data['J'],
        'n': data['n'],
        'r': data['r']
    }

    print("\nFitting model...")
    try:
        fit = model.sample(data=stan_data, **SAMPLING_CONFIG)
        print("SUCCESS")

        # Quick diagnostics
        diagnostics = fit.diagnose()
        print("\nDiagnostics:")
        print(diagnostics)

        # Save fit
        fit.save_csvfiles(dir=str(OUTPUT_DIR / "robust_hbb"))

        return fit

    except Exception as e:
        print(f"FAILED: {e}")
        return None


def fit_dp_mbb(model, data):
    """Fit Dirichlet Process Mixture."""
    print("\n" + "="*60)
    print("MODEL 3: Dirichlet Process Mixture")
    print("="*60)

    if model is None:
        print("Model compilation failed. Skipping.")
        return None

    # Prepare data
    stan_data = {
        'J': data['J'],
        'K_max': 10,  # Truncation level
        'n': data['n'],
        'r': data['r'],
        'a0': 2.0,    # Beta(2, 28) base distribution
        'b0': 28.0
    }

    print("\nFitting model...")
    print("Warning: This model is computationally intensive.")
    print("May take 2-4x longer than standard hierarchy.")

    try:
        fit = model.sample(data=stan_data, **SAMPLING_CONFIG)
        print("SUCCESS")

        # Quick diagnostics
        diagnostics = fit.diagnose()
        print("\nDiagnostics:")
        print(diagnostics)

        # Save fit
        fit.save_csvfiles(dir=str(OUTPUT_DIR / "dp_mbb"))

        return fit

    except Exception as e:
        print(f"FAILED: {e}")
        return None


def compute_basic_diagnostics(fit, model_name):
    """Compute basic MCMC diagnostics."""
    if fit is None:
        return None

    print(f"\n{model_name} - Basic Diagnostics:")

    # Convert to ArviZ
    try:
        idata = az.from_cmdstanpy(fit)
    except Exception as e:
        print(f"  Failed to convert to ArviZ: {e}")
        return None

    # Rhat
    rhat = az.rhat(idata)
    max_rhat = float(rhat.max())
    print(f"  - Max Rhat: {max_rhat:.4f} (should be < 1.01)")

    # ESS
    ess_bulk = az.ess(idata, method="bulk")
    min_ess_bulk = float(ess_bulk.min())
    print(f"  - Min ESS (bulk): {min_ess_bulk:.0f} (should be > 400)")

    # Divergences
    divergences = fit.divergences
    if divergences is not None:
        n_div = divergences.sum()
        pct_div = 100 * n_div / (SAMPLING_CONFIG['chains'] * SAMPLING_CONFIG['iter_sampling'])
        print(f"  - Divergences: {n_div} ({pct_div:.2f}%, should be < 1%)")

    # Summary
    summary = az.summary(idata, round_to=3)

    return {
        'max_rhat': max_rhat,
        'min_ess_bulk': min_ess_bulk,
        'divergences': divergences.sum() if divergences is not None else 0,
        'summary': summary
    }


def extract_model_specific_diagnostics(fit, model_name):
    """Extract model-specific diagnostics for falsification tests."""
    if fit is None:
        return None

    print(f"\n{model_name} - Model-Specific Diagnostics:")

    if model_name == "FMM-3":
        # Cluster diagnostics
        samples = fit.stan_variables()

        # Cluster probabilities
        cluster_prob = samples['cluster_prob']  # [samples, J, K]
        max_probs = cluster_prob.max(axis=2)    # Max probability per group

        mean_max_prob = max_probs.mean(axis=0)  # Average across samples
        avg_certainty = mean_max_prob.mean()

        print(f"  - Average cluster certainty: {avg_certainty:.3f} (should be > 0.6)")

        # Cluster separation
        mu = samples['mu']  # [samples, K]
        mu_mean = mu.mean(axis=0)
        print(f"  - Cluster means (logit): {mu_mean}")
        print(f"  - Cluster means (prob): {1/(1+np.exp(-mu_mean))}")

        # Effective clusters
        K_eff = samples['K_effective'].mean()
        print(f"  - Effective clusters: {K_eff:.1f} (should be ~3)")

        return {
            'avg_certainty': avg_certainty,
            'mu_mean': mu_mean,
            'K_effective': K_eff,
            'falsification_pass': avg_certainty > 0.6 and K_eff >= 2
        }

    elif model_name == "Robust-HBB":
        # Tail heaviness and overdispersion
        samples = fit.stan_variables()

        nu = samples['nu']
        kappa = samples['kappa']

        nu_mean = nu.mean()
        kappa_mean = kappa.mean()

        print(f"  - Degrees of freedom (nu): {nu_mean:.2f} (< 10 = heavy tails)")
        print(f"  - Concentration (kappa): {kappa_mean:.2f} (< 500 = overdispersed)")

        is_heavy_tailed = nu_mean < 30
        is_overdispersed = kappa_mean < 500

        print(f"  - Heavy tails detected: {is_heavy_tailed}")
        print(f"  - Overdispersion detected: {is_overdispersed}")

        return {
            'nu_mean': nu_mean,
            'kappa_mean': kappa_mean,
            'is_heavy_tailed': is_heavy_tailed,
            'is_overdispersed': is_overdispersed,
            'falsification_pass': is_heavy_tailed or is_overdispersed
        }

    elif model_name == "DP-MBB":
        # Cluster discovery
        samples = fit.stan_variables()

        K_eff = samples['K_effective']
        K_eff_mean = K_eff.mean()
        K_eff_std = K_eff.std()

        print(f"  - Effective clusters: {K_eff_mean:.2f} ± {K_eff_std:.2f}")

        # Cluster sizes
        cluster_size = samples['cluster_size']  # [samples, K_max]
        avg_sizes = cluster_size.mean(axis=0)

        print(f"  - Average cluster sizes: {avg_sizes[avg_sizes > 0.5]}")

        # Fragmentation check
        n_singleton_clusters = (avg_sizes > 0.9).sum()  # Clusters with ~1 group

        print(f"  - Singleton clusters: {n_singleton_clusters} (should be < 5)")

        return {
            'K_effective_mean': K_eff_mean,
            'K_effective_std': K_eff_std,
            'n_singleton': n_singleton_clusters,
            'falsification_pass': 1 < K_eff_mean < 7 and n_singleton_clusters < 5
        }

    return None


def save_results(fits, diagnostics):
    """Save fitting results and diagnostics."""
    print("\nSaving results...")

    results_summary = {
        'models_attempted': list(fits.keys()),
        'models_converged': [k for k, v in fits.items() if v is not None],
        'diagnostics': diagnostics,
        'sampling_config': SAMPLING_CONFIG
    }

    with open(OUTPUT_DIR / "fitting_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"  - Saved to {OUTPUT_DIR / 'fitting_summary.json'}")


def main():
    """Main fitting pipeline."""
    print("="*60)
    print("ALTERNATIVE BAYESIAN MODELS - FITTING PIPELINE")
    print("Designer 2: Mixture Models and Robust Approaches")
    print("="*60)

    # Load data
    data, df = load_data()

    # Compile models
    models = compile_models()

    # Fit models
    fits = {}
    diagnostics = {}

    # Model 1: FMM-3
    if models['fmm3'] is not None:
        fits['fmm3'] = fit_fmm3(models['fmm3'], data)
        if fits['fmm3'] is not None:
            diagnostics['fmm3'] = {
                'basic': compute_basic_diagnostics(fits['fmm3'], "FMM-3"),
                'specific': extract_model_specific_diagnostics(fits['fmm3'], "FMM-3")
            }

    # Model 2: Robust-HBB
    if models['robust_hbb'] is not None:
        fits['robust_hbb'] = fit_robust_hbb(models['robust_hbb'], data)
        if fits['robust_hbb'] is not None:
            diagnostics['robust_hbb'] = {
                'basic': compute_basic_diagnostics(fits['robust_hbb'], "Robust-HBB"),
                'specific': extract_model_specific_diagnostics(fits['robust_hbb'], "Robust-HBB")
            }

    # Model 3: DP-MBB (optional, most complex)
    if models['dp_mbb'] is not None:
        print("\n" + "="*60)
        print("WARNING: DP-MBB is computationally intensive")
        print("Consider skipping if FMM-3 already shows clear results")
        print("="*60)

        fits['dp_mbb'] = fit_dp_mbb(models['dp_mbb'], data)
        if fits['dp_mbb'] is not None:
            diagnostics['dp_mbb'] = {
                'basic': compute_basic_diagnostics(fits['dp_mbb'], "DP-MBB"),
                'specific': extract_model_specific_diagnostics(fits['dp_mbb'], "DP-MBB")
            }

    # Save results
    save_results(fits, diagnostics)

    # Final summary
    print("\n" + "="*60)
    print("FITTING COMPLETE")
    print("="*60)
    print("\nSummary:")
    for model_name, fit in fits.items():
        if fit is not None:
            status = "SUCCESS"
            if diagnostics[model_name]['specific']:
                passed = diagnostics[model_name]['specific'].get('falsification_pass', False)
                status += f" (Falsification: {'PASS' if passed else 'FAIL'})"
        else:
            status = "FAILED"
        print(f"  - {model_name}: {status}")

    print(f"\nNext steps:")
    print(f"  1. Run compare_models.py for LOO-CV comparison")
    print(f"  2. Run falsification_tests.py for detailed checks")
    print(f"  3. Review results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
