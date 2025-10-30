"""
Fit and Compare Alternative Bayesian Models
Designer #2 - Robust and Alternative Parameterizations

Models:
1. Robust Hierarchical (Student-t hyperpriors)
2. Beta-Binomial (direct probability scale)
3. Beta-Binomial Marginalized (efficient)
4. Finite Mixture (2-component)

Author: Model Designer #2
Date: 2025-10-30
"""

import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Paths
DATA_PATH = Path("/workspace/data/binomial_data.csv")
MODEL_DIR = Path("/workspace/experiments/designer_2")
OUTPUT_DIR = MODEL_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
data = pd.read_csv(DATA_PATH)
print("Data loaded:")
print(data)

# Prepare Stan data
stan_data = {
    'J': len(data),
    'n': data['n'].tolist(),
    'r': data['r'].tolist()
}

print(f"\nStan data: J={stan_data['J']} groups")
print(f"Total trials: {sum(stan_data['n'])}")
print(f"Total successes: {sum(stan_data['r'])}")

# Model configurations
models = {
    'robust_hierarchical': {
        'file': MODEL_DIR / 'model_robust_hierarchical.stan',
        'description': 'Robust Hierarchical (Student-t)',
        'color': '#E74C3C'
    },
    'beta_binomial': {
        'file': MODEL_DIR / 'model_beta_binomial.stan',
        'description': 'Beta-Binomial (Full)',
        'color': '#3498DB'
    },
    'beta_binomial_marginalized': {
        'file': MODEL_DIR / 'model_beta_binomial_marginalized.stan',
        'description': 'Beta-Binomial (Marginalized)',
        'color': '#9B59B6'
    },
    'mixture': {
        'file': MODEL_DIR / 'model_mixture.stan',
        'description': 'Finite Mixture (2-component)',
        'color': '#F39C12'
    }
}

# Sampling configuration
SAMPLING_CONFIG = {
    'iter_warmup': 1000,
    'iter_sampling': 2000,
    'chains': 4,
    'parallel_chains': 4,
    'adapt_delta': 0.95,
    'max_treedepth': 12,
    'show_console': False
}

# Storage for results
results = {}
convergence_diagnostics = {}
loo_comparisons = {}

# ============================================================================
# Fit Models
# ============================================================================

print("\n" + "="*80)
print("FITTING MODELS")
print("="*80)

for model_name, model_config in models.items():
    print(f"\n{'-'*80}")
    print(f"Model: {model_config['description']}")
    print(f"File: {model_config['file']}")
    print(f"{'-'*80}")

    try:
        # Compile model
        print("Compiling...")
        model = cmdstanpy.CmdStanModel(stan_file=str(model_config['file']))

        # Fit model
        print("Sampling...")
        fit = model.sample(
            data=stan_data,
            **SAMPLING_CONFIG
        )

        # Store results
        results[model_name] = {
            'fit': fit,
            'model': model,
            'config': model_config
        }

        # Convergence diagnostics
        print("\nConvergence Diagnostics:")
        summary = fit.summary()

        # R-hat check
        rhat_issues = summary[summary['R_hat'] > 1.05]
        if len(rhat_issues) > 0:
            print(f"  WARNING: {len(rhat_issues)} parameters with R_hat > 1.05")
            print(rhat_issues[['Mean', 'R_hat']])
        else:
            print("  R_hat: All parameters < 1.05 (GOOD)")

        # Divergences
        divergences = fit.num_unconstrained_params_per_chain()
        if 'divergent__' in fit.method_variables():
            n_divergences = fit.divergences
            print(f"  Divergences: {n_divergences}")
            if n_divergences > 0:
                print(f"    WARNING: Model has convergence issues")
        else:
            print("  Divergences: 0 (GOOD)")

        # ESS
        ess_bulk = summary['N_Eff'].min()
        print(f"  Min ESS (bulk): {ess_bulk:.0f}")
        if ess_bulk < 400:
            print(f"    WARNING: Low ESS, chains may not have mixed well")

        # Store diagnostics
        convergence_diagnostics[model_name] = {
            'rhat_max': summary['R_hat'].max(),
            'rhat_issues': len(rhat_issues),
            'ess_bulk_min': ess_bulk,
            'divergences': n_divergences if 'divergent__' in fit.method_variables() else 0,
            'converged': len(rhat_issues) == 0 and ess_bulk >= 400
        }

        print(f"\nModel Status: {'CONVERGED' if convergence_diagnostics[model_name]['converged'] else 'CONVERGENCE ISSUES'}")

    except Exception as e:
        print(f"\nERROR fitting {model_name}: {str(e)}")
        convergence_diagnostics[model_name] = {
            'error': str(e),
            'converged': False
        }

# ============================================================================
# LOO Comparison
# ============================================================================

print("\n" + "="*80)
print("LOO-CV COMPARISON")
print("="*80)

loo_results = {}
idata_dict = {}

for model_name, result in results.items():
    if result is None or convergence_diagnostics[model_name].get('error'):
        continue

    try:
        fit = result['fit']

        # Convert to InferenceData for arviz
        idata = az.from_cmdstanpy(
            posterior=fit,
            log_likelihood='log_lik',
            posterior_predictive='r_rep',
            observed_data={'r': stan_data['r']}
        )
        idata_dict[model_name] = idata

        # Compute LOO
        loo = az.loo(idata)
        loo_results[model_name] = loo

        print(f"\n{result['config']['description']}:")
        print(f"  ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
        print(f"  p_loo: {loo.p_loo:.2f}")

        # Check Pareto k values
        k_values = loo.pareto_k
        if hasattr(k_values, 'values'):
            k_values = k_values.values

        k_high = np.sum(k_values > 0.7)
        k_moderate = np.sum((k_values > 0.5) & (k_values <= 0.7))

        print(f"  Pareto k diagnostics:")
        print(f"    k > 0.7 (bad): {k_high} groups")
        print(f"    k ∈ (0.5, 0.7) (moderate): {k_moderate} groups")
        print(f"    k < 0.5 (good): {len(k_values) - k_high - k_moderate} groups")

        if k_high > 0:
            print(f"    WARNING: High k values indicate model misspecification")
            high_k_groups = np.where(k_values > 0.7)[0]
            print(f"    Problem groups: {high_k_groups + 1}")  # 1-indexed

    except Exception as e:
        print(f"\nERROR computing LOO for {model_name}: {str(e)}")

# Compare models
if len(loo_results) >= 2:
    print("\n" + "-"*80)
    print("Pairwise LOO Comparisons:")
    print("-"*80)

    model_names_list = list(loo_results.keys())
    for i, model1 in enumerate(model_names_list):
        for model2 in model_names_list[i+1:]:
            try:
                comparison = az.compare({
                    model1: idata_dict[model1],
                    model2: idata_dict[model2]
                })

                print(f"\n{results[model1]['config']['description']} vs {results[model2]['config']['description']}:")
                print(comparison)

                # Interpret difference
                delta_elpd = comparison.iloc[0]['elpd_loo'] - comparison.iloc[1]['elpd_loo']
                delta_se = comparison.iloc[1]['se']

                if abs(delta_elpd) < 2 * delta_se:
                    print("  Verdict: Models are equivalent (ΔELPD < 2×SE)")
                elif abs(delta_elpd) < 4:
                    print(f"  Verdict: Weak preference for {comparison.index[0]} (ΔELPD ∈ (2×SE, 4))")
                else:
                    print(f"  Verdict: Strong preference for {comparison.index[0]} (ΔELPD > 4)")

            except Exception as e:
                print(f"\nERROR comparing {model1} vs {model2}: {str(e)}")

# ============================================================================
# Posterior Summaries
# ============================================================================

print("\n" + "="*80)
print("POSTERIOR SUMMARIES")
print("="*80)

posterior_summaries = {}

for model_name, result in results.items():
    if result is None or convergence_diagnostics[model_name].get('error'):
        continue

    print(f"\n{'-'*80}")
    print(f"Model: {result['config']['description']}")
    print(f"{'-'*80}")

    try:
        fit = result['fit']
        summary = fit.summary()

        # Key parameters
        if model_name == 'robust_hierarchical':
            params = ['mu', 'tau', 'nu', 'mu_prob', 'phi']
            print("\nKey Parameters:")
            for param in params:
                if param in summary.index:
                    row = summary.loc[param]
                    print(f"  {param:12s}: {row['Mean']:7.3f} ± {row['StdDev']:6.3f}  [{row['5%']:7.3f}, {row['95%']:7.3f}]")

            # Interpretation
            nu_mean = summary.loc['nu', 'Mean']
            print(f"\n  Degrees of freedom (nu = {nu_mean:.1f}):")
            if nu_mean < 10:
                print("    Heavy tails (strong outlier accommodation)")
            elif nu_mean < 20:
                print("    Moderate tails (some outlier accommodation)")
            else:
                print("    Light tails (approaching Normal)")

        elif model_name in ['beta_binomial', 'beta_binomial_marginalized']:
            params = ['alpha', 'beta', 'mu_pop', 'sigma_pop', 'kappa', 'phi']
            print("\nKey Parameters:")
            for param in params:
                if param in summary.index:
                    row = summary.loc[param]
                    print(f"  {param:12s}: {row['Mean']:7.3f} ± {row['StdDev']:6.3f}  [{row['5%']:7.3f}, {row['95%']:7.3f}]")

            # Interpretation
            if 'kappa' in summary.index:
                kappa_mean = summary.loc['kappa', 'Mean']
                print(f"\n  Concentration (kappa = {kappa_mean:.1f}):")
                if kappa_mean < 10:
                    print("    High dispersion (weak pooling)")
                elif kappa_mean < 100:
                    print("    Moderate dispersion")
                else:
                    print("    Low dispersion (strong pooling)")

        elif model_name == 'mixture':
            params = ['pi', 'mu[1]', 'mu[2]', 'p_comp1', 'p_comp2', 'separation', 'n_comp1', 'n_comp2']
            print("\nKey Parameters:")
            for param in params:
                if param in summary.index:
                    row = summary.loc[param]
                    print(f"  {param:12s}: {row['Mean']:7.3f} ± {row['StdDev']:6.3f}  [{row['5%']:7.3f}, {row['95%']:7.3f}]")

            # Component assignments
            print("\n  Component Assignments:")
            for j in range(stan_data['J']):
                prob_comp2_param = f'prob_comp2[{j+1}]'
                if prob_comp2_param in summary.index:
                    prob = summary.loc[prob_comp2_param, 'Mean']
                    z = summary.loc[f'z[{j+1}]', 'Mean']
                    print(f"    Group {j+1:2d}: P(comp2) = {prob:.3f}, Assignment = {int(z)}")

            # Interpretation
            if 'separation' in summary.index:
                sep_mean = summary.loc['separation', 'Mean']
                print(f"\n  Component separation = {sep_mean:.3f} logit units")
                if sep_mean < 0.5:
                    print("    WARNING: Components are not well-separated")

        # Store summary
        posterior_summaries[model_name] = summary

    except Exception as e:
        print(f"\nERROR summarizing {model_name}: {str(e)}")

# ============================================================================
# Posterior Predictive Checks
# ============================================================================

print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECKS")
print("="*80)

for model_name, result in results.items():
    if result is None or convergence_diagnostics[model_name].get('error'):
        continue

    print(f"\n{'-'*80}")
    print(f"Model: {result['config']['description']}")
    print(f"{'-'*80}")

    try:
        fit = result['fit']
        r_rep = fit.stan_variable('r_rep')  # Shape: (iterations, J)

        # Compute predictive intervals
        r_obs = np.array(stan_data['r'])
        r_pred_mean = r_rep.mean(axis=0)
        r_pred_lower = np.percentile(r_rep, 2.5, axis=0)
        r_pred_upper = np.percentile(r_rep, 97.5, axis=0)

        # Coverage
        coverage = np.mean((r_obs >= r_pred_lower) & (r_obs <= r_pred_upper))
        print(f"  95% Predictive Interval Coverage: {coverage:.1%}")
        print(f"    Expected: 95%, Observed: {coverage:.1%}")
        if coverage < 0.85:
            print(f"    WARNING: Poor coverage, model may underestimate uncertainty")

        # Check extreme groups
        extreme_groups = [1, 3, 7]  # Groups 2, 4, 8 (0-indexed: 1, 3, 7)
        print(f"\n  Extreme Group Checks:")
        for j in extreme_groups:
            in_interval = (r_obs[j] >= r_pred_lower[j]) and (r_obs[j] <= r_pred_upper[j])
            status = "COVERED" if in_interval else "MISSED"
            print(f"    Group {j+1:2d}: r={r_obs[j]:3d}, pred=[{r_pred_lower[j]:.1f}, {r_pred_upper[j]:.1f}] - {status}")

        # Dispersion check
        if 'phi' in fit.stan_variables():
            phi_samples = fit.stan_variable('phi')
            phi_mean = phi_samples.mean()
            phi_lower = np.percentile(phi_samples, 2.5)
            phi_upper = np.percentile(phi_samples, 97.5)

            print(f"\n  Overdispersion Parameter (φ):")
            print(f"    Posterior: {phi_mean:.2f} [{phi_lower:.2f}, {phi_upper:.2f}]")
            print(f"    Observed: 3.59 (from EDA)")
            if phi_lower <= 3.59 <= phi_upper:
                print(f"    GOOD: Observed φ within posterior interval")
            else:
                print(f"    WARNING: Model doesn't capture observed dispersion")

    except Exception as e:
        print(f"\nERROR in predictive checks for {model_name}: {str(e)}")

# ============================================================================
# Save Results
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save convergence diagnostics
diag_df = pd.DataFrame(convergence_diagnostics).T
diag_df.to_csv(OUTPUT_DIR / 'convergence_diagnostics.csv')
print(f"Convergence diagnostics saved to {OUTPUT_DIR / 'convergence_diagnostics.csv'}")

# Save LOO comparison
if len(loo_results) >= 2:
    loo_comparison_df = az.compare({
        name: idata_dict[name] for name in loo_results.keys()
    })
    loo_comparison_df.to_csv(OUTPUT_DIR / 'loo_comparison.csv')
    print(f"LOO comparison saved to {OUTPUT_DIR / 'loo_comparison.csv'}")

# Save posterior summaries
for model_name, summary in posterior_summaries.items():
    summary.to_csv(OUTPUT_DIR / f'posterior_summary_{model_name}.csv')
print(f"Posterior summaries saved to {OUTPUT_DIR}")

# Save Stan fits (for later analysis)
for model_name, result in results.items():
    if result is not None and not convergence_diagnostics[model_name].get('error'):
        result['fit'].save_csvfiles(str(OUTPUT_DIR / f'fit_{model_name}'))
print(f"Stan fits saved to {OUTPUT_DIR}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll results saved to: {OUTPUT_DIR}")
print("\nNext steps:")
print("  1. Review convergence diagnostics")
print("  2. Compare LOO results")
print("  3. Examine posterior predictive checks")
print("  4. Choose best model or identify red flags")
print("  5. Run sensitivity analyses if needed")
