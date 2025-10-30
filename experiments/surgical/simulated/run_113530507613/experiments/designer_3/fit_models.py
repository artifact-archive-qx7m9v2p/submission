"""
Fit Bayesian Regression Models with Covariates

This script implements the three regression models proposed by Designer 3:
1. Hierarchical logistic regression with sample size covariate
2. Hierarchical logistic regression with quadratic group effect
3. Hierarchical logistic regression with random slopes

Usage:
    python fit_models.py --data <path_to_data.csv> --output <output_dir>
"""

import numpy as np
import pandas as pd
import cmdstanpy
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json


class RegressionModelFitter:
    """Fit and compare Bayesian regression models with covariates"""

    def __init__(self, data_path, output_dir, models_dir=None):
        """
        Parameters
        ----------
        data_path : str
            Path to CSV file with columns: group_id, n_trials, r_successes
        output_dir : str
            Directory to save results
        models_dir : str, optional
            Directory containing .stan files (default: same as output_dir)
        """
        self.data = pd.read_csv(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if models_dir is None:
            self.models_dir = self.output_dir
        else:
            self.models_dir = Path(models_dir)

        self.validate_data()
        self.models = {}
        self.fits = {}
        self.loo_results = {}

    def validate_data(self):
        """Validate data structure and constraints"""
        required_cols = ['group_id', 'n_trials', 'r_successes']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check binomial constraints
        assert (self.data['r_successes'] <= self.data['n_trials']).all(), \
            "r_successes must be <= n_trials"
        assert (self.data['r_successes'] >= 0).all(), "r_successes must be >= 0"
        assert (self.data['n_trials'] > 0).all(), "n_trials must be > 0"

        print(f"Data validated: {len(self.data)} groups")
        print(f"  n_trials: {self.data['n_trials'].min()} to {self.data['n_trials'].max()}")
        print(f"  r_successes: {self.data['r_successes'].min()} to {self.data['r_successes'].max()}")
        print(f"  Success rates: {self.data['r_successes'].min() / self.data['n_trials'].max():.3f} "
              f"to {self.data['r_successes'].max() / self.data['n_trials'].min():.3f}")

    def prepare_data_model1(self):
        """Prepare data for Model 1: Sample size covariate"""
        log_n = np.log(self.data['n_trials'].values)
        log_n_centered = log_n - log_n.mean()

        stan_data = {
            'J': len(self.data),
            'n': self.data['n_trials'].values.astype(int),
            'r': self.data['r_successes'].values.astype(int),
            'log_n_centered': log_n_centered
        }
        return stan_data

    def prepare_data_model2(self):
        """Prepare data for Model 2: Quadratic group effect"""
        group_id = self.data['group_id'].values
        # Scale to [-1, 1] range
        group_scaled = (group_id - group_id.mean()) / (group_id.max() - group_id.min()) * 2
        group_scaled_sq = group_scaled ** 2

        stan_data = {
            'J': len(self.data),
            'n': self.data['n_trials'].values.astype(int),
            'r': self.data['r_successes'].values.astype(int),
            'group_scaled': group_scaled,
            'group_scaled_sq': group_scaled_sq
        }
        return stan_data

    def prepare_data_model3(self):
        """Prepare data for Model 3: Random slopes"""
        # Same as Model 1
        return self.prepare_data_model1()

    def compile_model(self, model_name):
        """Compile Stan model"""
        model_path = self.models_dir / f"{model_name}.stan"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"\nCompiling {model_name}...")
        model = cmdstanpy.CmdStanModel(stan_file=str(model_path))
        self.models[model_name] = model
        print(f"  Compiled successfully")
        return model

    def fit_model(self, model_name, stan_data, chains=4, iter_sampling=1000, iter_warmup=1000):
        """Fit Stan model with MCMC"""
        if model_name not in self.models:
            self.compile_model(model_name)

        model = self.models[model_name]
        print(f"\nFitting {model_name}...")
        print(f"  Chains: {chains}, Iterations: {iter_warmup} warmup + {iter_sampling} sampling")

        try:
            fit = model.sample(
                data=stan_data,
                chains=chains,
                iter_sampling=iter_sampling,
                iter_warmup=iter_warmup,
                show_progress=True,
                adapt_delta=0.95  # Conservative for robustness
            )
            self.fits[model_name] = fit

            # Check diagnostics
            self.check_diagnostics(model_name, fit)

            return fit

        except Exception as e:
            print(f"  ERROR: Fitting failed - {e}")
            return None

    def check_diagnostics(self, model_name, fit):
        """Check MCMC diagnostics"""
        print(f"\n  Diagnostics for {model_name}:")

        # Convert to ArviZ InferenceData
        idata = az.from_cmdstanpy(fit)

        # Rhat
        rhat = az.rhat(idata)
        max_rhat = max([rhat[var].max().values for var in rhat.data_vars])
        print(f"    Max Rhat: {max_rhat:.4f} {'[GOOD]' if max_rhat < 1.01 else '[WARNING]'}")

        # ESS
        ess = az.ess(idata)
        min_ess = min([ess[var].min().values for var in ess.data_vars])
        print(f"    Min ESS: {min_ess:.0f} {'[GOOD]' if min_ess > 400 else '[WARNING]'}")

        # Divergences
        try:
            divergences = fit.method_variables()['divergent__'].sum()
            total_samples = len(fit.method_variables()['divergent__'])
            div_pct = 100 * divergences / total_samples
            print(f"    Divergences: {divergences} ({div_pct:.2f}%) "
                  f"{'[GOOD]' if div_pct < 1 else '[WARNING]'}")
        except:
            print(f"    Divergences: Unable to check")

    def compute_loo(self, model_name):
        """Compute LOO-CV for model comparison"""
        if model_name not in self.fits:
            raise ValueError(f"Model {model_name} not fitted yet")

        print(f"\nComputing LOO for {model_name}...")
        idata = az.from_cmdstanpy(self.fits[model_name])

        loo = az.loo(idata, pointwise=True)
        self.loo_results[model_name] = loo

        print(f"  ELPD LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
        print(f"  p_loo: {loo.p_loo:.2f}")

        # Check Pareto k values
        k_vals = loo.pareto_k.values
        n_high_k = (k_vals > 0.7).sum()
        if n_high_k > 0:
            print(f"  WARNING: {n_high_k} observations with Pareto k > 0.7")

        return loo

    def compare_models(self):
        """Compare models using LOO-CV"""
        if len(self.loo_results) < 2:
            print("\nNeed at least 2 models for comparison")
            return None

        print("\n" + "="*60)
        print("MODEL COMPARISON (LOO-CV)")
        print("="*60)

        # Convert to ArviZ InferenceData for comparison
        idata_dict = {}
        for model_name in self.loo_results.keys():
            idata_dict[model_name] = az.from_cmdstanpy(self.fits[model_name])

        # Compare
        comp = az.compare(idata_dict, ic='loo')
        print("\n", comp)

        # Interpretation
        print("\n" + "-"*60)
        print("INTERPRETATION:")
        print("-"*60)
        best_model = comp.index[0]
        print(f"\nBest model: {best_model}")

        if len(comp) > 1:
            second_model = comp.index[1]
            elpd_diff = comp.loc[second_model, 'elpd_diff']
            se_diff = comp.loc[second_model, 'dse']

            print(f"\nΔLOO ({second_model} vs {best_model}): {elpd_diff:.2f} ± {se_diff:.2f}")

            if abs(elpd_diff) < 2:
                print("  → Negligible difference (ΔLOO < 2)")
                print("  → Models are practically equivalent")
                print("  → Prefer simpler model")
            elif abs(elpd_diff) < 4:
                print("  → Weak evidence (2 < ΔLOO < 4)")
                print("  → Slight preference for best model")
            else:
                print("  → Substantial evidence (ΔLOO > 4)")
                print("  → Clear preference for best model")

        return comp

    def plot_predictions(self, model_name):
        """Plot observed vs predicted for a fitted model"""
        if model_name not in self.fits:
            raise ValueError(f"Model {model_name} not fitted yet")

        fit = self.fits[model_name]
        idata = az.from_cmdstanpy(fit)

        # Extract predictions
        p_samples = idata.posterior['p'].values.reshape(-1, len(self.data))
        p_mean = p_samples.mean(axis=0)
        p_lower = np.percentile(p_samples, 2.5, axis=0)
        p_upper = np.percentile(p_samples, 97.5, axis=0)

        # Observed success rates
        observed_rate = self.data['r_successes'] / self.data['n_trials']

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: Observed vs Predicted
        ax = axes[0]
        ax.errorbar(observed_rate, p_mean,
                   yerr=[p_mean - p_lower, p_upper - p_mean],
                   fmt='o', alpha=0.6, capsize=3)
        ax.plot([0, observed_rate.max()], [0, observed_rate.max()],
               'k--', alpha=0.3, label='y=x')
        ax.set_xlabel('Observed Success Rate')
        ax.set_ylabel('Predicted Success Rate')
        ax.set_title(f'{model_name}: Observed vs Predicted')
        ax.legend()
        ax.grid(alpha=0.3)

        # Panel 2: Residuals
        ax = axes[1]
        residuals = observed_rate - p_mean
        ax.scatter(p_mean, residuals, alpha=0.6)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Predicted Success Rate')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{model_name}: Residual Plot')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / f'{model_name}_predictions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved prediction plot: {plot_path}")
        plt.close()

    def plot_coefficients(self, model_name):
        """Plot coefficient posteriors"""
        if model_name not in self.fits:
            raise ValueError(f"Model {model_name} not fitted yet")

        idata = az.from_cmdstanpy(self.fits[model_name])

        # Select key parameters
        if model_name == 'model1_size_covariate':
            var_names = ['beta_0', 'beta_1', 'tau']
        elif model_name == 'model2_quadratic_group':
            var_names = ['beta_0', 'beta_1', 'beta_2', 'tau']
        elif model_name == 'model3_random_slopes':
            var_names = ['beta_0', 'beta_1', 'tau_alpha', 'tau_gamma', 'rho']
        else:
            var_names = None

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        az.plot_forest(idata, var_names=var_names, combined=True, ax=ax)
        ax.set_title(f'{model_name}: Coefficient Posteriors')
        plt.tight_layout()

        plot_path = self.output_dir / f'{model_name}_coefficients.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved coefficient plot: {plot_path}")
        plt.close()

    def extract_results(self, model_name):
        """Extract key results from fitted model"""
        if model_name not in self.fits:
            raise ValueError(f"Model {model_name} not fitted yet")

        idata = az.from_cmdstanpy(self.fits[model_name])
        summary = az.summary(idata)

        results = {
            'model_name': model_name,
            'summary_statistics': summary.to_dict(),
            'loo': {
                'elpd_loo': float(self.loo_results[model_name].elpd_loo),
                'se': float(self.loo_results[model_name].se),
                'p_loo': float(self.loo_results[model_name].p_loo)
            }
        }

        # Save to JSON
        results_path = self.output_dir / f'{model_name}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved results: {results_path}")
        return results

    def run_full_pipeline(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*60)
        print("BAYESIAN REGRESSION MODEL COMPARISON PIPELINE")
        print("="*60)

        results = {}

        # Model 1: Sample size covariate
        print("\n\n### MODEL 1: Sample Size Covariate ###")
        stan_data1 = self.prepare_data_model1()
        fit1 = self.fit_model('model1_size_covariate', stan_data1)
        if fit1 is not None:
            loo1 = self.compute_loo('model1_size_covariate')
            self.plot_predictions('model1_size_covariate')
            self.plot_coefficients('model1_size_covariate')
            results['model1'] = self.extract_results('model1_size_covariate')

        # Model 2: Quadratic group effect
        print("\n\n### MODEL 2: Quadratic Group Effect ###")
        stan_data2 = self.prepare_data_model2()
        fit2 = self.fit_model('model2_quadratic_group', stan_data2)
        if fit2 is not None:
            loo2 = self.compute_loo('model2_quadratic_group')
            self.plot_predictions('model2_quadratic_group')
            self.plot_coefficients('model2_quadratic_group')
            results['model2'] = self.extract_results('model2_quadratic_group')

        # Model 3: Random slopes (if requested)
        print("\n\n### MODEL 3: Random Slopes (WARNING: Complex Model) ###")
        response = input("Fit Model 3 (random slopes)? [y/N]: ")
        if response.lower() == 'y':
            stan_data3 = self.prepare_data_model3()
            fit3 = self.fit_model('model3_random_slopes', stan_data3,
                                 iter_warmup=1500, iter_sampling=1500)
            if fit3 is not None:
                loo3 = self.compute_loo('model3_random_slopes')
                self.plot_predictions('model3_random_slopes')
                self.plot_coefficients('model3_random_slopes')
                results['model3'] = self.extract_results('model3_random_slopes')

        # Compare models
        if len(self.loo_results) >= 2:
            comparison = self.compare_models()
            if comparison is not None:
                comp_path = self.output_dir / 'model_comparison.csv'
                comparison.to_csv(comp_path)
                print(f"\nSaved comparison: {comp_path}")

        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"\nResults saved to: {self.output_dir}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Fit Bayesian regression models with covariates'
    )
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Directory containing .stan files (default: same as output)')

    args = parser.parse_args()

    # Run pipeline
    fitter = RegressionModelFitter(
        data_path=args.data,
        output_dir=args.output,
        models_dir=args.models_dir
    )

    results = fitter.run_full_pipeline()

    return results


if __name__ == '__main__':
    main()
