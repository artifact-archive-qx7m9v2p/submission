"""
Hierarchical Binomial Model Fitting Script
Designer 2: Random Effects Models

Fits three hierarchical binomial models using CmdStanPy:
1. Model 1: Centered parameterization (baseline, may have issues)
2. Model 2: Non-centered parameterization (RECOMMENDED)
3. Model 3: Robust with Student-t priors (outlier handling)

Usage:
    python fit_models.py [--model {1,2,3,all}] [--chains 4] [--iter 2000]

Requirements:
    - cmdstanpy
    - pandas
    - numpy
    - arviz (for diagnostics)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings

try:
    import cmdstanpy
    print(f"CmdStanPy version: {cmdstanpy.__version__}")
except ImportError:
    raise ImportError("CmdStanPy not installed. Install with: pip install cmdstanpy")

try:
    import arviz as az
    print(f"ArviZ version: {az.__version__}")
except ImportError:
    warnings.warn("ArviZ not installed. Install for better diagnostics: pip install arviz")
    az = None


def load_data(data_path: str) -> dict:
    """Load binomial trial data and prepare for Stan."""
    df = pd.read_csv(data_path)

    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Number of groups: {len(df)}")
    print(f"Total trials: {df['n_trials'].sum()}")
    print(f"Total successes: {df['r_successes'].sum()}")
    print(f"Pooled rate: {df['r_successes'].sum() / df['n_trials'].sum():.4f}")
    print(f"\nTrial sizes: {df['n_trials'].min()} to {df['n_trials'].max()}")
    print(f"Success counts: {df['r_successes'].min()} to {df['r_successes'].max()}")
    print(f"Success rates: {df['success_rate'].min():.4f} to {df['success_rate'].max():.4f}")

    # Check for Group 1 zero count
    zero_groups = df[df['r_successes'] == 0]
    if len(zero_groups) > 0:
        print(f"\n⚠️  WARNING: {len(zero_groups)} group(s) with zero successes:")
        for _, row in zero_groups.iterrows():
            print(f"   Group {row['group']}: 0/{row['n_trials']} trials")
        print("   Will be handled through hierarchical shrinkage (no ad-hoc correction)")

    # Prepare Stan data
    stan_data = {
        'N': len(df),
        'n_trials': df['n_trials'].tolist(),
        'r_successes': df['r_successes'].tolist()
    }

    return stan_data, df


def fit_model(model_num: int, stan_data: dict,
              chains: int = 4, iter_warmup: int = 1000,
              iter_sampling: int = 2000, adapt_delta: float = 0.9,
              max_treedepth: int = 10, seed: int = 12345) -> tuple:
    """
    Fit specified hierarchical binomial model.

    Args:
        model_num: Which model to fit (1, 2, or 3)
        stan_data: Data dictionary for Stan
        chains: Number of MCMC chains
        iter_warmup: Warmup iterations per chain
        iter_sampling: Sampling iterations per chain
        adapt_delta: Target acceptance rate (increase if divergences)
        max_treedepth: Maximum tree depth (increase if warnings)
        seed: Random seed for reproducibility

    Returns:
        (fit object, model object, model name)
    """
    models = {
        1: ("model1_centered.stan", "M1: Centered",
            "Standard centered parameterization - may have computational issues"),
        2: ("model2_noncentered.stan", "M2: Non-centered (RECOMMENDED)",
            "Non-centered parameterization - computationally efficient"),
        3: ("model3_robust.stan", "M3: Robust Student-t",
            "Heavy-tailed priors for outlier robustness")
    }

    if model_num not in models:
        raise ValueError(f"Model must be 1, 2, or 3. Got {model_num}")

    stan_file, model_name, description = models[model_num]
    model_path = Path(__file__).parent / stan_file

    print("\n" + "="*60)
    print(f"FITTING {model_name}")
    print("="*60)
    print(f"Description: {description}")
    print(f"Stan file: {stan_file}")
    print(f"\nMCMC Settings:")
    print(f"  Chains: {chains}")
    print(f"  Warmup iterations: {iter_warmup}")
    print(f"  Sampling iterations: {iter_sampling}")
    print(f"  adapt_delta: {adapt_delta}")
    print(f"  max_treedepth: {max_treedepth}")
    print(f"  seed: {seed}")

    # Compile model
    print("\nCompiling Stan model...")
    model = cmdstanpy.CmdStanModel(stan_file=str(model_path))
    print("✓ Compilation successful")

    # Fit model
    print("\nSampling from posterior...")
    print("(This may take a few minutes...)\n")

    fit = model.sample(
        data=stan_data,
        chains=chains,
        parallel_chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
        show_console=True,
        output_dir=str(Path(__file__).parent / "stan_output")
    )

    return fit, model, model_name


def check_diagnostics(fit, model_name: str) -> dict:
    """
    Run comprehensive MCMC diagnostics.

    Returns dictionary with diagnostic results and pass/fail status.
    """
    print("\n" + "="*60)
    print(f"DIAGNOSTICS: {model_name}")
    print("="*60)

    diagnostics = {}
    all_passed = True

    # 1. Divergences
    try:
        divergences = fit.divergences
        total_divergences = divergences.sum() if hasattr(divergences, 'sum') else 0
        total_iterations = fit.num_draws_sampling * fit.chains
        div_pct = (total_divergences / total_iterations) * 100 if total_iterations > 0 else 0

        diagnostics['divergences'] = {
            'count': int(total_divergences),
            'percentage': div_pct,
            'passed': div_pct < 1.0
        }

        if div_pct > 0:
            status = "✓ PASS" if div_pct < 1.0 else "✗ FAIL"
            print(f"\nDivergent transitions: {total_divergences} ({div_pct:.2f}%) {status}")
            if div_pct >= 1.0:
                print("  ⚠️  WARNING: >1% divergences. Consider:")
                print("     - Increase adapt_delta to 0.95 or 0.99")
                print("     - Try non-centered parameterization (Model 2)")
                all_passed = False
        else:
            print(f"\nDivergent transitions: 0 ✓ EXCELLENT")
    except Exception as e:
        print(f"\n⚠️  Could not check divergences: {e}")
        diagnostics['divergences'] = {'error': str(e), 'passed': False}
        all_passed = False

    # 2. Rhat
    try:
        summary_df = fit.summary()
        rhat_vals = summary_df['R_hat'].dropna()
        max_rhat = rhat_vals.max()

        diagnostics['rhat'] = {
            'max': float(max_rhat),
            'passed': max_rhat < 1.01
        }

        status = "✓ PASS" if max_rhat < 1.01 else "✗ FAIL"
        print(f"Max Rhat: {max_rhat:.4f} {status}")
        if max_rhat >= 1.01:
            print("  ⚠️  WARNING: Rhat ≥ 1.01. Chains may not have converged.")
            print("     Consider running longer chains.")
            all_passed = False

            # Show which parameters have high Rhat
            high_rhat = summary_df[summary_df['R_hat'] >= 1.01]
            if len(high_rhat) > 0:
                print(f"     Parameters with Rhat ≥ 1.01: {list(high_rhat.index)}")
    except Exception as e:
        print(f"⚠️  Could not check Rhat: {e}")
        diagnostics['rhat'] = {'error': str(e), 'passed': False}
        all_passed = False

    # 3. Effective Sample Size
    try:
        ess_bulk = summary_df['ess_bulk'].dropna()
        ess_tail = summary_df['ess_tail'].dropna()
        min_ess_bulk = ess_bulk.min()
        min_ess_tail = ess_tail.min()

        diagnostics['ess'] = {
            'min_bulk': float(min_ess_bulk),
            'min_tail': float(min_ess_tail),
            'passed': min_ess_bulk > 400 and min_ess_tail > 400
        }

        status_bulk = "✓ PASS" if min_ess_bulk > 400 else "✗ FAIL"
        status_tail = "✓ PASS" if min_ess_tail > 400 else "✗ FAIL"
        print(f"Min ESS (bulk): {min_ess_bulk:.0f} {status_bulk}")
        print(f"Min ESS (tail): {min_ess_tail:.0f} {status_tail}")

        if min_ess_bulk <= 400 or min_ess_tail <= 400:
            print("  ⚠️  WARNING: ESS < 400 for some parameters.")
            print("     Consider running longer chains or reparameterization.")
            all_passed = False
    except Exception as e:
        print(f"⚠️  Could not check ESS: {e}")
        diagnostics['ess'] = {'error': str(e), 'passed': False}
        all_passed = False

    # 4. Tree depth
    try:
        max_td = fit.max_treedepth
        hit_max = (fit.sampler_diagnostics('treedepth__') == max_td).sum()
        total_iterations = fit.num_draws_sampling * fit.chains
        td_pct = (hit_max / total_iterations) * 100

        diagnostics['treedepth'] = {
            'hit_max_count': int(hit_max),
            'percentage': float(td_pct),
            'passed': td_pct < 1.0
        }

        if hit_max > 0:
            status = "✓ OK" if td_pct < 1.0 else "⚠️  WARNING"
            print(f"Hit max treedepth: {hit_max} times ({td_pct:.2f}%) {status}")
            if td_pct >= 1.0:
                print("     Consider increasing max_treedepth to 12 or 15")
        else:
            print(f"Hit max treedepth: 0 times ✓ GOOD")
    except Exception as e:
        print(f"⚠️  Could not check treedepth: {e}")
        diagnostics['treedepth'] = {'error': str(e), 'passed': True}  # Not critical

    # Overall assessment
    print("\n" + "-"*60)
    if all_passed:
        print("✓ ALL DIAGNOSTICS PASSED - Model fit is reliable")
        diagnostics['overall'] = 'PASS'
    else:
        print("✗ SOME DIAGNOSTICS FAILED - Review warnings above")
        diagnostics['overall'] = 'FAIL'
    print("-"*60)

    return diagnostics


def summarize_posteriors(fit, model_name: str, observed_data: pd.DataFrame):
    """Print summary of key posterior quantities."""
    print("\n" + "="*60)
    print(f"POSTERIOR SUMMARY: {model_name}")
    print("="*60)

    summary_df = fit.summary()

    # Population parameters
    print("\nPopulation Parameters:")
    print("-" * 40)

    if 'mu' in summary_df.index:
        mu_mean = summary_df.loc['mu', 'Mean']
        mu_sd = summary_df.loc['mu', 'StdDev']
        mu_q5 = summary_df.loc['mu', '5%']
        mu_q95 = summary_df.loc['mu', '95%']

        # Convert to probability scale
        p_mean = 1 / (1 + np.exp(-mu_mean))
        p_q5 = 1 / (1 + np.exp(-mu_q5))
        p_q95 = 1 / (1 + np.exp(-mu_q95))

        print(f"μ (population mean, logit scale):")
        print(f"  Mean: {mu_mean:.3f} (SD: {mu_sd:.3f})")
        print(f"  90% CI: [{mu_q5:.3f}, {mu_q95:.3f}]")
        print(f"  → Probability scale: {p_mean:.3%} (90% CI: [{p_q5:.3%}, {p_q95:.3%}])")

    if 'sigma' in summary_df.index:
        sigma_mean = summary_df.loc['sigma', 'Mean']
        sigma_sd = summary_df.loc['sigma', 'StdDev']
        sigma_q5 = summary_df.loc['sigma', '5%']
        sigma_q95 = summary_df.loc['sigma', '95%']

        print(f"\nσ (between-group SD, logit scale):")
        print(f"  Mean: {sigma_mean:.3f} (SD: {sigma_sd:.3f})")
        print(f"  90% CI: [{sigma_q5:.3f}, {sigma_q95:.3f}]")

        # Calculate implied ICC
        # ICC ≈ σ² / (σ² + π²/3)
        logistic_var = np.pi**2 / 3
        icc_mean = sigma_mean**2 / (sigma_mean**2 + logistic_var)
        print(f"  → Implied ICC: {icc_mean:.2%}")
        print(f"     (EDA estimated ICC: 73%)")

    if 'nu' in summary_df.index:
        nu_mean = summary_df.loc['nu', 'Mean']
        nu_q5 = summary_df.loc['nu', '5%']
        nu_q95 = summary_df.loc['nu', '95%']

        print(f"\nν (degrees of freedom for Student-t):")
        print(f"  Mean: {nu_mean:.1f}")
        print(f"  90% CI: [{nu_q5:.1f}, {nu_q95:.1f}]")
        if nu_mean > 30:
            print(f"  → Heavy tails NOT necessary (ν > 30 ≈ normal)")
        else:
            print(f"  → Heavy tails ARE important (ν < 30)")

    # Overdispersion
    if 'phi_posterior' in summary_df.index:
        phi_mean = summary_df.loc['phi_posterior', 'Mean']
        phi_q5 = summary_df.loc['phi_posterior', '5%']
        phi_q95 = summary_df.loc['phi_posterior', '95%']

        print(f"\nOverdispersion φ:")
        print(f"  Posterior mean: {phi_mean:.2f}")
        print(f"  90% CI: [{phi_q5:.2f}, {phi_q95:.2f}]")
        print(f"  → EDA observed: φ ≈ 3.5-5.1")

        if phi_mean >= 3.0 and phi_mean <= 6.0:
            print(f"  ✓ Model reproduces observed overdispersion")
        else:
            print(f"  ⚠️  Model φ differs from observed - check fit")

    # Group-specific estimates
    print("\nGroup-Specific Success Rates:")
    print("-" * 40)
    print(f"{'Group':<6} {'Observed':<10} {'Posterior Mean':<15} {'90% CI':<20} {'Shrinkage':<10}")
    print("-" * 70)

    for i in range(len(observed_data)):
        group_id = observed_data.iloc[i]['group']
        obs_rate = observed_data.iloc[i]['success_rate']

        p_param = f'p_posterior[{i+1}]'
        if p_param in summary_df.index:
            p_mean = summary_df.loc[p_param, 'Mean']
            p_q5 = summary_df.loc[p_param, '5%']
            p_q95 = summary_df.loc[p_param, '95%']

            # Calculate shrinkage
            pooled_rate = observed_data['r_successes'].sum() / observed_data['n_trials'].sum()
            if obs_rate != pooled_rate:
                shrinkage_pct = abs((obs_rate - p_mean) / (obs_rate - pooled_rate)) * 100
            else:
                shrinkage_pct = 0

            print(f"{group_id:<6} {obs_rate:>8.1%} {p_mean:>13.1%}   "
                  f"[{p_q5:.1%}, {p_q95:.1%}]    {shrinkage_pct:>6.1f}%")

    print("\nNote: Shrinkage shows % of distance from observed to pooled rate")
    print("      Higher shrinkage for groups with small sample sizes")


def save_results(fit, model_name: str, diagnostics: dict, output_dir: Path):
    """Save fit results and diagnostics to files."""
    output_dir.mkdir(exist_ok=True)

    # Save diagnostics as JSON
    diag_file = output_dir / f"{model_name.replace(' ', '_').replace(':', '')}_diagnostics.json"
    with open(diag_file, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n✓ Saved diagnostics to: {diag_file}")

    # Save posterior summary as CSV
    summary_file = output_dir / f"{model_name.replace(' ', '_').replace(':', '')}_summary.csv"
    fit.summary().to_csv(summary_file)
    print(f"✓ Saved posterior summary to: {summary_file}")

    # Save draws as CSV (for downstream analysis)
    draws_file = output_dir / f"{model_name.replace(' ', '_').replace(':', '')}_draws.csv"
    fit.draws_pd().to_csv(draws_file, index=False)
    print(f"✓ Saved posterior draws to: {draws_file}")


def main():
    parser = argparse.ArgumentParser(description='Fit hierarchical binomial models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['1', '2', '3', 'all'],
                       help='Which model to fit (default: all)')
    parser.add_argument('--data', type=str,
                       default='/workspace/data/data.csv',
                       help='Path to data file')
    parser.add_argument('--chains', type=int, default=4,
                       help='Number of MCMC chains (default: 4)')
    parser.add_argument('--iter', type=int, default=2000,
                       help='Sampling iterations per chain (default: 2000)')
    parser.add_argument('--warmup', type=int, default=1000,
                       help='Warmup iterations per chain (default: 1000)')
    parser.add_argument('--adapt_delta', type=float, default=0.9,
                       help='Target acceptance rate (default: 0.9)')
    parser.add_argument('--seed', type=int, default=12345,
                       help='Random seed (default: 12345)')

    args = parser.parse_args()

    # Load data
    stan_data, observed_df = load_data(args.data)

    # Determine which models to fit
    if args.model == 'all':
        models_to_fit = [2, 3, 1]  # Fit M2 first (recommended), then M3, then M1
    else:
        models_to_fit = [int(args.model)]

    # Fit models
    results = {}
    for model_num in models_to_fit:
        try:
            fit, model_obj, model_name = fit_model(
                model_num, stan_data,
                chains=args.chains,
                iter_warmup=args.warmup,
                iter_sampling=args.iter,
                adapt_delta=args.adapt_delta,
                seed=args.seed
            )

            # Run diagnostics
            diagnostics = check_diagnostics(fit, model_name)

            # Summarize posteriors
            summarize_posteriors(fit, model_name, observed_df)

            # Save results
            output_dir = Path(__file__).parent / "results"
            save_results(fit, model_name, diagnostics, output_dir)

            results[model_name] = {
                'fit': fit,
                'diagnostics': diagnostics
            }

        except Exception as e:
            print(f"\n✗ ERROR fitting {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "="*60)
    print("FITTING COMPLETE")
    print("="*60)
    print(f"\nFitted {len(results)} model(s)")
    for model_name, result in results.items():
        status = result['diagnostics'].get('overall', 'UNKNOWN')
        print(f"  {model_name}: {status}")

    print("\nNext steps:")
    print("  1. Review diagnostics above")
    print("  2. Run posterior predictive checks (posterior_predictive.py)")
    print("  3. Compare models using LOO-CV")
    print("  4. Visualize results")


if __name__ == '__main__':
    main()
