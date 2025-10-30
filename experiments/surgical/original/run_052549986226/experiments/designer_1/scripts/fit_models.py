#!/usr/bin/env python3
"""
Fit all three beta-binomial models using CmdStanPy
Designer 1 - Model fitting script

Usage:
    python fit_models.py [--model MODEL_NAME]

Options:
    --model: Fit specific model (a, b, c, or 'all')
"""

import argparse
import cmdstanpy
import pandas as pd
import numpy as np
import arviz as az
import json
from pathlib import Path

# Setup paths
BASE_DIR = Path('/workspace/experiments/designer_1')
DATA_PATH = Path('/workspace/data/data.csv')
STAN_DIR = BASE_DIR / 'stan_models'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    """Load and prepare data for Stan"""
    data = pd.read_csv(DATA_PATH)

    stan_data = {
        'N': len(data),
        'n_trials': data['n_trials'].values.tolist(),
        'r_success': data['r_successes'].values.tolist()
    }

    return data, stan_data

def fit_model_a(stan_data, chains=4, iter_sampling=2000):
    """Fit Model A: Standard beta-binomial"""
    print("\n" + "="*60)
    print("Fitting Model A: Homogeneous Beta-Binomial (α, β)")
    print("="*60)

    stan_file = STAN_DIR / 'model_a_beta_binomial.stan'
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))

    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_sampling=iter_sampling,
        iter_warmup=1000,
        adapt_delta=0.95,
        max_treedepth=12,
        show_console=True
    )

    # Save fit
    fit.save_csvfiles(dir=str(RESULTS_DIR / 'model_a'))

    return fit

def fit_model_b(stan_data, chains=4, iter_sampling=2000):
    """Fit Model B: Reparameterized beta-binomial"""
    print("\n" + "="*60)
    print("Fitting Model B: Reparameterized Beta-Binomial (μ, κ)")
    print("="*60)

    stan_file = STAN_DIR / 'model_b_reparameterized.stan'
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))

    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_sampling=iter_sampling,
        iter_warmup=1000,
        adapt_delta=0.95,
        max_treedepth=12,
        show_console=True
    )

    # Save fit
    fit.save_csvfiles(dir=str(RESULTS_DIR / 'model_b'))

    return fit

def fit_model_c(stan_data, chains=8, iter_sampling=3000):
    """Fit Model C: Mixture model (more chains/iterations for harder model)"""
    print("\n" + "="*60)
    print("Fitting Model C: Two-Component Mixture")
    print("="*60)
    print("Note: Using 8 chains and 3000 iterations for mixture model")

    stan_file = STAN_DIR / 'model_c_mixture.stan'
    model = cmdstanpy.CmdStanModel(stan_file=str(stan_file))

    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_sampling=iter_sampling,
        iter_warmup=1500,
        adapt_delta=0.95,
        max_treedepth=12,
        show_console=True
    )

    # Save fit
    fit.save_csvfiles(dir=str(RESULTS_DIR / 'model_c'))

    return fit

def check_diagnostics(fit, model_name):
    """Check convergence diagnostics"""
    print(f"\n{'='*60}")
    print(f"Diagnostics for {model_name}")
    print(f"{'='*60}")

    # Get summary
    summary = fit.summary()

    # Check Rhat
    rhat_issues = summary[summary['R_hat'] > 1.01]
    if len(rhat_issues) > 0:
        print(f"⚠️  WARNING: {len(rhat_issues)} parameters have Rhat > 1.01")
        print(rhat_issues[['Mean', 'R_hat', 'N_Eff']])
    else:
        print("✅ All Rhat < 1.01")

    # Check ESS
    ess_issues = summary[summary['N_Eff'] < 400]
    if len(ess_issues) > 0:
        print(f"⚠️  WARNING: {len(ess_issues)} parameters have ESS < 400")
        print(ess_issues[['Mean', 'R_hat', 'N_Eff']])
    else:
        print("✅ All ESS > 400")

    # Check divergences
    divergences = fit.divergences
    if divergences > 0:
        print(f"⚠️  WARNING: {divergences} divergent transitions")
    else:
        print("✅ No divergent transitions")

    # Save summary
    summary.to_csv(RESULTS_DIR / f'{model_name}_summary.csv')
    print(f"\nSummary saved to {RESULTS_DIR / f'{model_name}_summary.csv'}")

    return summary

def compute_loo(fit, model_name):
    """Compute LOO-CV using ArviZ"""
    print(f"\nComputing LOO-CV for {model_name}...")

    idata = az.from_cmdstanpy(fit, log_likelihood='log_lik')
    loo = az.loo(idata)

    print(f"LOO: {loo.loo:.2f} ± {loo.loo_se:.2f}")
    print(f"pLOO: {loo.p_loo:.2f}")

    # Check Pareto k values
    pareto_k = loo.pareto_k
    bad_k = np.sum(pareto_k > 0.7)
    if bad_k > 0:
        print(f"⚠️  WARNING: {bad_k} observations have Pareto k > 0.7")
        print(f"Max Pareto k: {np.max(pareto_k):.3f}")
    else:
        print("✅ All Pareto k < 0.7")

    # Save LOO results
    with open(RESULTS_DIR / f'{model_name}_loo.json', 'w') as f:
        json.dump({
            'loo': float(loo.loo),
            'loo_se': float(loo.loo_se),
            'p_loo': float(loo.p_loo),
            'max_pareto_k': float(np.max(pareto_k)),
            'n_bad_pareto_k': int(bad_k)
        }, f, indent=2)

    return loo, idata

def main():
    parser = argparse.ArgumentParser(description='Fit beta-binomial models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['a', 'b', 'c', 'all'],
                       help='Which model to fit')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    data, stan_data = load_data()
    print(f"Data: {stan_data['N']} groups")
    print(f"Total trials: {sum(stan_data['n_trials'])}")
    print(f"Total successes: {sum(stan_data['r_success'])}")

    results = {}

    # Fit models
    if args.model in ['a', 'all']:
        fit_a = fit_model_a(stan_data)
        summary_a = check_diagnostics(fit_a, 'model_a')
        loo_a, idata_a = compute_loo(fit_a, 'model_a')
        results['model_a'] = {'fit': fit_a, 'summary': summary_a, 'loo': loo_a, 'idata': idata_a}

    if args.model in ['b', 'all']:
        fit_b = fit_model_b(stan_data)
        summary_b = check_diagnostics(fit_b, 'model_b')
        loo_b, idata_b = compute_loo(fit_b, 'model_b')
        results['model_b'] = {'fit': fit_b, 'summary': summary_b, 'loo': loo_b, 'idata': idata_b}

    if args.model in ['c', 'all']:
        fit_c = fit_model_c(stan_data)
        summary_c = check_diagnostics(fit_c, 'model_c')
        loo_c, idata_c = compute_loo(fit_c, 'model_c')
        results['model_c'] = {'fit': fit_c, 'summary': summary_c, 'loo': loo_c, 'idata': idata_c}

    # Compare models if all were fit
    if len(results) > 1:
        print("\n" + "="*60)
        print("Model Comparison (LOO-CV)")
        print("="*60)

        for model_name in results:
            loo = results[model_name]['loo']
            print(f"{model_name}: LOO = {loo.loo:.2f} ± {loo.loo_se:.2f}")

        # Compute differences
        if 'model_a' in results and 'model_b' in results:
            diff_ab = results['model_a']['loo'].loo - results['model_b']['loo'].loo
            print(f"\nΔLOO(A - B) = {diff_ab:.2f}")
            print("Note: Models A and B should be nearly identical")

        if 'model_b' in results and 'model_c' in results:
            diff_bc = results['model_b']['loo'].loo - results['model_c']['loo'].loo
            print(f"ΔLOO(B - C) = {diff_bc:.2f}")
            if diff_bc > 10:
                print("✅ Homogeneous model (B) strongly preferred")
            elif diff_bc < -10:
                print("⚠️  Mixture model (C) strongly preferred - investigate!")
            else:
                print("⚠️  Models similar - need further analysis")

    print(f"\n{'='*60}")
    print("Model fitting complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*60}")

    return results

if __name__ == '__main__':
    main()
