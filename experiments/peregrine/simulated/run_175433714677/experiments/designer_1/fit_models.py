#!/usr/bin/env python3
"""
Fit and compare parsimonious Bayesian models for count data.

Models:
  1. Log-Linear Negative Binomial (baseline)
  2. Quadratic Negative Binomial (if Model 1 shows curvature)

Author: Designer 1 (Parsimony Track)
Date: 2025-10-29
"""

import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
WORKSPACE = Path("/workspace")
DATA_PATH = WORKSPACE / "data.json"
MODELS_DIR = WORKSPACE / "experiments" / "designer_1"
OUTPUT_DIR = MODELS_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Sampling configuration
CHAINS = 4
WARMUP = 1000
SAMPLING = 1000
ADAPT_DELTA = 0.8
MAX_TREEDEPTH = 10
SEED = 12345


def load_data():
    """Load and prepare data for Stan."""
    print("Loading data...")
    data = pd.read_json(DATA_PATH)

    # Prepare Stan data for Model 1 (linear)
    stan_data_1 = {
        'N': len(data),
        'y': data['C'].values.astype(int),
        'year': data['year'].values
    }

    # Prepare Stan data for Model 2 (quadratic)
    stan_data_2 = {
        'N': len(data),
        'y': data['C'].values.astype(int),
        'year': data['year'].values,
        'year_sq': (data['year'].values ** 2)
    }

    print(f"  N = {stan_data_1['N']} observations")
    print(f"  Count range: [{data['C'].min()}, {data['C'].max()}]")
    print(f"  Year range: [{data['year'].min():.2f}, {data['year'].max():.2f}]")
    print(f"  Observed Var/Mean ratio: {data['C'].var() / data['C'].mean():.2f}")

    return data, stan_data_1, stan_data_2


def compile_models():
    """Compile Stan models."""
    print("\nCompiling Stan models...")

    model1_path = MODELS_DIR / "model1_linear.stan"
    model2_path = MODELS_DIR / "model2_quadratic.stan"

    print(f"  Model 1: {model1_path}")
    model1 = cmdstanpy.CmdStanModel(stan_file=str(model1_path))

    print(f"  Model 2: {model2_path}")
    model2 = cmdstanpy.CmdStanModel(stan_file=str(model2_path))

    return model1, model2


def fit_model(model, stan_data, model_name):
    """Fit a Stan model with diagnostics."""
    print(f"\nFitting {model_name}...")

    fit = model.sample(
        data=stan_data,
        chains=CHAINS,
        iter_warmup=WARMUP,
        iter_sampling=SAMPLING,
        adapt_delta=ADAPT_DELTA,
        max_treedepth=MAX_TREEDEPTH,
        seed=SEED,
        show_progress=True
    )

    # Basic diagnostics
    print(f"\n{model_name} Diagnostics:")
    print(f"  Divergences: {fit.num_divergences()}")
    print(f"  Max tree depth hits: {fit.num_max_treedepth()}")

    # Convert to ArviZ InferenceData
    idata = az.from_cmdstanpy(
        fit,
        posterior_predictive='y_rep',
        log_likelihood='log_lik'
    )

    return fit, idata


def convergence_diagnostics(idata, model_name):
    """Check convergence diagnostics."""
    print(f"\n{model_name} Convergence:")

    summary = az.summary(idata, var_names=['beta_0', 'beta_1', 'phi'])
    print(summary)

    # Check critical thresholds
    rhat_ok = (summary['r_hat'] < 1.01).all()
    ess_bulk_ok = (summary['ess_bulk'] > 400).all()
    ess_tail_ok = (summary['ess_tail'] > 400).all()

    print(f"\n  R̂ < 1.01: {'PASS' if rhat_ok else 'FAIL'}")
    print(f"  ESS_bulk > 400: {'PASS' if ess_bulk_ok else 'FAIL'}")
    print(f"  ESS_tail > 400: {'PASS' if ess_tail_ok else 'FAIL'}")

    return rhat_ok and ess_bulk_ok and ess_tail_ok


def posterior_predictive_checks(idata, data, model_name):
    """Posterior predictive checks."""
    print(f"\n{model_name} Posterior Predictive Checks:")

    y_obs = data['C'].values
    y_rep = idata.posterior_predictive['y_rep'].values

    # Reshape y_rep: (chains, draws, N) -> (chains*draws, N)
    y_rep_flat = y_rep.reshape(-1, y_rep.shape[-1])

    # Check 1: Variance-to-mean ratio
    obs_var_mean = y_obs.var() / y_obs.mean()
    pred_var_mean = np.var(y_rep_flat, axis=1) / np.mean(y_rep_flat, axis=1)
    pred_var_mean_median = np.median(pred_var_mean)
    pred_var_mean_ci = np.percentile(pred_var_mean, [2.5, 97.5])

    print(f"\n  Variance-to-Mean Ratio:")
    print(f"    Observed: {obs_var_mean:.2f}")
    print(f"    Predicted (median): {pred_var_mean_median:.2f}")
    print(f"    Predicted (95% CI): [{pred_var_mean_ci[0]:.2f}, {pred_var_mean_ci[1]:.2f}]")
    print(f"    Check: {'PASS' if pred_var_mean_ci[0] < obs_var_mean < pred_var_mean_ci[1] else 'FAIL'}")

    # Check 2: Calibration (prediction intervals)
    for coverage in [50, 80, 90]:
        lower = (100 - coverage) / 2
        upper = 100 - lower
        pred_intervals = np.percentile(y_rep_flat, [lower, upper], axis=0)

        in_interval = np.mean((y_obs >= pred_intervals[0]) & (y_obs <= pred_intervals[1]))
        print(f"\n  {coverage}% Prediction Interval:")
        print(f"    Coverage: {in_interval*100:.1f}%")
        print(f"    Check: {'PASS' if abs(in_interval - coverage/100) < 0.1 else 'WARN'}")

    # Check 3: Extreme value recovery
    obs_max = y_obs.max()
    pred_max = y_rep_flat.max(axis=1)
    pred_max_exceeds_obs = np.mean(pred_max >= obs_max)

    print(f"\n  Extreme Value Recovery:")
    print(f"    Observed max: {obs_max}")
    print(f"    Pr(pred_max >= obs_max): {pred_max_exceeds_obs:.3f}")
    print(f"    Check: {'PASS' if pred_max_exceeds_obs > 0.1 else 'WARN'}")

    return {
        'var_mean_obs': obs_var_mean,
        'var_mean_pred': pred_var_mean_median,
        'var_mean_ci': pred_var_mean_ci
    }


def model_comparison(idata_dict):
    """Compare models using LOO-CV."""
    print("\n" + "="*60)
    print("MODEL COMPARISON (LOO-CV)")
    print("="*60)

    # Compute LOO for each model
    loo_dict = {}
    for name, idata in idata_dict.items():
        print(f"\nComputing LOO for {name}...")
        loo = az.loo(idata, pointwise=True)
        loo_dict[name] = loo

        print(f"  ELPD_loo: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
        print(f"  p_loo: {loo.p_loo:.2f}")

        # Check Pareto-k diagnostic
        pareto_k = loo.pareto_k
        n_bad_k = np.sum(pareto_k > 0.7)
        print(f"  Pareto-k > 0.7: {n_bad_k}/{len(pareto_k)} ({n_bad_k/len(pareto_k)*100:.1f}%)")

        if n_bad_k > 0:
            print(f"    WARNING: {n_bad_k} observations have high Pareto-k (>0.7)")

    # Compare models
    if len(idata_dict) > 1:
        print("\n" + "-"*60)
        comp = az.compare(idata_dict)
        print("\nModel Comparison Table:")
        print(comp)

        # Interpretation
        print("\n" + "-"*60)
        print("INTERPRETATION:")
        best_model = comp.index[0]
        print(f"\nBest model: {best_model}")

        if len(comp) > 1:
            delta_elpd = comp.loc[comp.index[1], 'elpd_diff']
            se_diff = comp.loc[comp.index[1], 'dse']

            print(f"ΔELPD (vs. 2nd best): {delta_elpd:.2f} ± {se_diff:.2f}")

            if abs(delta_elpd) < 2:
                print("  → Models are statistically equivalent (ΔELPD < 2)")
                print("  → Recommendation: Choose simpler model")
            elif abs(delta_elpd) < 4:
                print("  → Weak evidence for best model (2 < ΔELPD < 4)")
                print("  → Consider interpretability and parsimony")
            else:
                print("  → Strong evidence for best model (ΔELPD > 4)")
                print("  → Recommendation: Use best model")

    return loo_dict


def plot_posterior_predictive(idata, data, model_name, output_dir):
    """Create posterior predictive check plots."""
    print(f"\nCreating plots for {model_name}...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{model_name}: Posterior Predictive Checks', fontsize=14)

    # Extract data
    y_obs = data['C'].values
    year = data['year'].values
    y_rep = idata.posterior_predictive['y_rep'].values.reshape(-1, len(y_obs))

    # Plot 1: Observed vs. Predicted (with uncertainty)
    ax = axes[0, 0]
    pred_median = np.median(y_rep, axis=0)
    pred_lower = np.percentile(y_rep, 5, axis=0)
    pred_upper = np.percentile(y_rep, 95, axis=0)

    ax.scatter(year, y_obs, color='black', s=50, alpha=0.7, label='Observed', zorder=3)
    ax.plot(year, pred_median, color='blue', linewidth=2, label='Predicted (median)', zorder=2)
    ax.fill_between(year, pred_lower, pred_upper, color='blue', alpha=0.3, label='90% PI', zorder=1)
    ax.set_xlabel('Year (standardized)')
    ax.set_ylabel('Count')
    ax.set_title('Observed vs. Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Residuals vs. Fitted
    ax = axes[0, 1]
    residuals = y_obs - pred_median
    ax.scatter(pred_median, residuals, alpha=0.6, s=50)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals (observed - predicted)')
    ax.set_title('Residuals vs. Fitted')
    ax.grid(True, alpha=0.3)

    # Plot 3: Posterior predictive distribution
    ax = axes[1, 0]
    ax.hist(y_obs, bins=20, alpha=0.5, color='black', label='Observed', density=True)
    for i in np.random.choice(len(y_rep), size=50, replace=False):
        ax.hist(y_rep[i], bins=20, alpha=0.02, color='blue', density=True)
    ax.set_xlabel('Count')
    ax.set_ylabel('Density')
    ax.set_title('Distribution: Observed vs. Replicated')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Variance-Mean relationship
    ax = axes[1, 1]
    var_mean_obs = y_obs.var() / y_obs.mean()
    var_mean_pred = np.var(y_rep, axis=1) / np.mean(y_rep, axis=1)

    ax.hist(var_mean_pred, bins=50, alpha=0.7, color='blue', label='Predicted', density=True)
    ax.axvline(var_mean_obs, color='red', linewidth=3, linestyle='--', label='Observed')
    ax.set_xlabel('Variance-to-Mean Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Overdispersion Check')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f'{model_name.lower().replace(" ", "_")}_ppc.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("PARSIMONIOUS BAYESIAN MODEL FITTING")
    print("="*60)

    # Load data
    data, stan_data_1, stan_data_2 = load_data()

    # Compile models
    model1, model2 = compile_models()

    # Fit Model 1 (always)
    fit1, idata1 = fit_model(model1, stan_data_1, "Model 1: Log-Linear")
    converged_1 = convergence_diagnostics(idata1, "Model 1")
    ppc_1 = posterior_predictive_checks(idata1, data, "Model 1")
    plot_posterior_predictive(idata1, data, "Model 1 Log-Linear", OUTPUT_DIR)

    # Decide whether to fit Model 2
    idata_dict = {"Model_1_Linear": idata1}

    # Check if we should fit Model 2
    # Criteria: Model 1 shows systematic residuals or poor calibration
    fit_model2 = False

    if not converged_1:
        print("\n" + "!"*60)
        print("WARNING: Model 1 did not converge properly!")
        print("!"*60)

    if ppc_1['var_mean_obs'] < ppc_1['var_mean_ci'][0] or ppc_1['var_mean_obs'] > ppc_1['var_mean_ci'][1]:
        print("\n" + "!"*60)
        print("Model 1 fails variance calibration - considering Model 2")
        print("!"*60)
        fit_model2 = True

    # For demonstration, always fit Model 2 to compare
    print("\n" + "-"*60)
    print("DECISION: Fitting Model 2 for comparison")
    print("-"*60)
    fit_model2 = True

    if fit_model2:
        fit2, idata2 = fit_model(model2, stan_data_2, "Model 2: Quadratic")
        converged_2 = convergence_diagnostics(idata2, "Model 2")
        ppc_2 = posterior_predictive_checks(idata2, data, "Model 2")
        plot_posterior_predictive(idata2, data, "Model 2 Quadratic", OUTPUT_DIR)

        idata_dict["Model_2_Quadratic"] = idata2

    # Model comparison
    loo_dict = model_comparison(idata_dict)

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    for name, idata in idata_dict.items():
        result_path = OUTPUT_DIR / f"{name.lower()}_idata.nc"
        idata.to_netcdf(result_path)
        print(f"  Saved: {result_path}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
