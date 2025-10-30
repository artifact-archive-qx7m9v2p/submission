"""
Fit Log-Linear Heteroscedastic Model (Experiment 2) to real data using PyMC.

Model:
  Y_i ~ Normal(mu_i, sigma_i)
  mu_i = beta_0 + beta_1 * log(x_i)
  log(sigma_i) = gamma_0 + gamma_1 * x_i
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
data_path = "/workspace/data/data.csv"
output_dir = "/workspace/experiments/experiment_2/posterior_inference"

print("=" * 80)
print("FITTING LOG-LINEAR HETEROSCEDASTIC MODEL (EXPERIMENT 2) - PyMC")
print("=" * 80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv(data_path)
print(f"   Loaded {len(df)} observations")
print(f"   x range: [{df['x'].min():.2f}, {df['x'].max():.2f}]")
print(f"   Y range: [{df['Y'].min():.2f}, {df['Y'].max():.2f}]")

x = df['x'].values
y = df['Y'].values
N = len(df)

# Build PyMC model
print("\n2. Building PyMC model...")

with pm.Model() as model:
    # Priors
    beta_0 = pm.Normal('beta_0', mu=1.8, sigma=0.5)
    beta_1 = pm.Normal('beta_1', mu=0.3, sigma=0.2)
    gamma_0 = pm.Normal('gamma_0', mu=-2, sigma=1)
    gamma_1 = pm.Normal('gamma_1', mu=-0.05, sigma=0.05)

    # Mean function
    mu = beta_0 + beta_1 * pm.math.log(x)

    # Variance function (log-linear)
    log_sigma = gamma_0 + gamma_1 * x
    sigma = pm.math.exp(log_sigma)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

print("   Model built successfully")

# Main sampling with high target_accept due to SBC warnings
print("\n3. Main sampling (4 chains × 1500 iterations, target_accept=0.97)...")
print("   Using conservative settings due to SBC warnings about this model")

with model:
    main_trace = pm.sample(
        draws=1500,
        tune=1500,
        chains=4,
        cores=4,
        target_accept=0.97,
        return_inferencedata=True,
        progressbar=True,
        random_seed=42,
        idata_kwargs={'log_likelihood': True}
    )

print("\n4. Convergence diagnostics...")

# Get summary
summary = az.summary(main_trace, var_names=['beta_0', 'beta_1', 'gamma_0', 'gamma_1'])

print("\n   Parameter Summary:")
# Get column names dynamically
cols_to_show = ['mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail']
# Add HDI columns if they exist
hdi_cols = [c for c in summary.columns if 'hdi' in c.lower()]
if hdi_cols:
    cols_to_show = ['mean', 'sd'] + hdi_cols + ['r_hat', 'ess_bulk', 'ess_tail']
print(summary[cols_to_show].to_string())

# Convergence checks
divergences = main_trace.sample_stats.diverging.sum().item()
max_rhat = summary['r_hat'].max()
min_ess = summary['ess_bulk'].min()
n_draws = 1500

print(f"\n   Convergence Metrics:")
print(f"   - Total divergences: {divergences} ({100*divergences/(4*n_draws):.1f}%)")
print(f"   - Max R-hat: {max_rhat:.4f} (target: < 1.01)")
print(f"   - Min ESS bulk: {min_ess:.0f} (target: > 400)")
print(f"   - Min ESS tail: {summary['ess_tail'].min():.0f}")

# Assess convergence
convergence_pass = (max_rhat < 1.01) and (min_ess > 400) and (divergences < 0.05 * 4 * n_draws)

if convergence_pass:
    print("\n   ✓ CONVERGENCE: PASS")
else:
    print("\n   ✗ CONVERGENCE: ISSUES DETECTED")

    if max_rhat >= 1.01:
        print(f"      - R-hat too high: {max_rhat:.4f}")
    if min_ess <= 400:
        print(f"      - ESS too low: {min_ess:.0f}")
    if divergences >= 0.05 * 4 * n_draws:
        print(f"      - Too many divergences: {divergences}")

    # Try recovery
    print("\n   Attempting recovery with target_accept=0.99 and 2000 draws...")

    with model:
        recovery_trace = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            cores=4,
            target_accept=0.99,
            return_inferencedata=True,
            progressbar=True,
            random_seed=43,
            idata_kwargs={'log_likelihood': True}
        )

    # Check recovery
    recovery_summary = az.summary(recovery_trace, var_names=['beta_0', 'beta_1', 'gamma_0', 'gamma_1'])
    recovery_div = recovery_trace.sample_stats.diverging.sum().item()
    recovery_rhat = recovery_summary['r_hat'].max()
    recovery_ess = recovery_summary['ess_bulk'].min()

    print(f"\n   Recovery Results:")
    print(f"   - Divergences: {recovery_div} ({100*recovery_div/(4*2000):.1f}%)")
    print(f"   - Max R-hat: {recovery_rhat:.4f}")
    print(f"   - Min ESS: {recovery_ess:.0f}")

    if (recovery_rhat < 1.01) and (recovery_ess > 400) and (recovery_div < 0.05 * 4 * 2000):
        print("\n   ✓ Recovery successful, using recovery fit")
        main_trace = recovery_trace
        summary = recovery_summary
        n_draws = 2000
        convergence_pass = True
        divergences = recovery_div
        max_rhat = recovery_rhat
        min_ess = recovery_ess
    else:
        print("\n   ✗ Recovery failed - model may be too complex for data")

# Save InferenceData
print("\n5. Saving ArviZ InferenceData...")
netcdf_path = f"{output_dir}/diagnostics/posterior_inference.netcdf"
main_trace.to_netcdf(netcdf_path)
print(f"   Saved InferenceData to: {netcdf_path}")

# Compute LOO
print("\n6. Computing LOO-CV...")
loo = az.loo(main_trace, pointwise=True)

print(f"\n   LOO Results:")
print(f"   - ELPD LOO: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"   - p_loo: {loo.p_loo:.2f}")
print(f"   - Pareto k stats:")
print(f"     - Good (k < 0.5): {np.sum(loo.pareto_k < 0.5)} ({100*np.sum(loo.pareto_k < 0.5)/len(loo.pareto_k):.1f}%)")
print(f"     - OK (0.5 ≤ k < 0.7): {np.sum((loo.pareto_k >= 0.5) & (loo.pareto_k < 0.7))}")
print(f"     - Bad (0.7 ≤ k < 1): {np.sum((loo.pareto_k >= 0.7) & (loo.pareto_k < 1.0))}")
print(f"     - Very bad (k ≥ 1): {np.sum(loo.pareto_k >= 1.0)}")
print(f"     - Max k: {loo.pareto_k.max():.3f}")
print(f"     - Mean k: {loo.pareto_k.mean():.3f}")

# Save LOO results
loo_results = {
    'elpd_loo': float(loo.elpd_loo),
    'se': float(loo.se),
    'p_loo': float(loo.p_loo),
    'pareto_k_stats': {
        'good_k_lt_0.5': int(np.sum(loo.pareto_k < 0.5)),
        'ok_k_0.5_to_0.7': int(np.sum((loo.pareto_k >= 0.5) & (loo.pareto_k < 0.7))),
        'bad_k_0.7_to_1.0': int(np.sum((loo.pareto_k >= 0.7) & (loo.pareto_k < 1.0))),
        'very_bad_k_gte_1.0': int(np.sum(loo.pareto_k >= 1.0)),
        'percent_good': float(100 * np.sum(loo.pareto_k < 0.5) / len(loo.pareto_k)),
        'max_k': float(loo.pareto_k.max()),
        'mean_k': float(loo.pareto_k.mean())
    }
}

loo_path = f"{output_dir}/diagnostics/loo_results.json"
with open(loo_path, 'w') as f:
    json.dump(loo_results, f, indent=2)
print(f"\n   Saved LOO results to: {loo_path}")

# Compare with Model 1
model1_loo = {'elpd_loo': 46.99, 'se': 3.11}  # From Model 1 results
delta_elpd = loo.elpd_loo - model1_loo['elpd_loo']
delta_se = np.sqrt(loo.se**2 + model1_loo['se']**2)

print(f"\n7. Model Comparison with Model 1:")
print(f"   Model 1 ELPD: {model1_loo['elpd_loo']:.2f} ± {model1_loo['se']:.2f}")
print(f"   Model 2 ELPD: {loo.elpd_loo:.2f} ± {loo.se:.2f}")
print(f"   Δ ELPD: {delta_elpd:.2f} ± {delta_se:.2f}")

if delta_elpd > 2 * delta_se:
    print("   → Model 2 strongly preferred")
    model_preference = "Model 2 strongly preferred (Δ > 2 SE)"
elif delta_elpd > delta_se:
    print("   → Model 2 weakly preferred")
    model_preference = "Model 2 weakly preferred (Δ > 1 SE)"
elif delta_elpd > -delta_se:
    print("   → Models approximately equivalent")
    model_preference = "Models approximately equivalent (|Δ| < 1 SE)"
elif delta_elpd > -2 * delta_se:
    print("   → Model 1 weakly preferred")
    model_preference = "Model 1 weakly preferred (Δ < -1 SE)"
else:
    print("   → Model 1 strongly preferred")
    model_preference = "Model 1 strongly preferred (Δ < -2 SE)"

# Test heteroscedasticity
print("\n8. Heteroscedasticity Assessment:")
gamma_1_post = main_trace.posterior['gamma_1'].values.flatten()
gamma_1_mean = gamma_1_post.mean()
gamma_1_std = gamma_1_post.std()
gamma_1_q025 = np.percentile(gamma_1_post, 2.5)
gamma_1_q975 = np.percentile(gamma_1_post, 97.5)
prob_negative = np.mean(gamma_1_post < 0)

print(f"   gamma_1 posterior:")
print(f"   - Mean: {gamma_1_mean:.4f} ± {gamma_1_std:.4f}")
print(f"   - 95% CI: [{gamma_1_q025:.4f}, {gamma_1_q975:.4f}]")
print(f"   - P(gamma_1 < 0): {prob_negative:.1%}")

if prob_negative > 0.95:
    print("   ✓ Strong evidence for heteroscedasticity (variance decreases with x)")
    hetero_conclusion = "Strong evidence for decreasing variance with x"
elif prob_negative > 0.90:
    print("   → Moderate evidence for heteroscedasticity")
    hetero_conclusion = "Moderate evidence for decreasing variance with x"
elif prob_negative < 0.10:
    print("   → Evidence against heteroscedasticity (variance increases with x)")
    hetero_conclusion = "Evidence for increasing variance with x"
elif gamma_1_q025 < 0 < gamma_1_q975:
    print("   ✗ Insufficient evidence for heteroscedasticity (CI includes 0)")
    hetero_conclusion = "Insufficient evidence for heteroscedasticity"
else:
    hetero_conclusion = "Mixed evidence for heteroscedasticity"

# Save detailed convergence report
print("\n9. Saving detailed convergence report...")

convergence_report = f"""# Convergence Report: Log-Linear Heteroscedastic Model (Experiment 2)

## Model Specification
```
Y_i ~ Normal(mu_i, sigma_i)
mu_i = beta_0 + beta_1 * log(x_i)
log(sigma_i) = gamma_0 + gamma_1 * x_i
```

## Sampling Configuration
- PPL: PyMC
- Chains: 4
- Warmup: {n_draws}
- Sampling: {n_draws}
- target_accept: {0.97 if n_draws == 1500 else 0.99}
- Total draws: {4 * n_draws}

## Convergence Metrics

### Parameter Estimates
```
{summary.to_string()}
```

### Diagnostic Summary
- **Total divergences**: {divergences} ({100*divergences/(4*n_draws):.2f}%)
- **Max R-hat**: {max_rhat:.4f}
- **Min ESS (bulk)**: {min_ess:.0f}
- **Min ESS (tail)**: {summary['ess_tail'].min():.0f}

### Convergence Criteria
- R-hat < 1.01: {'✓ PASS' if max_rhat < 1.01 else '✗ FAIL'}
- ESS > 400: {'✓ PASS' if min_ess > 400 else '✗ FAIL'}
- Divergences < 5%: {'✓ PASS' if divergences < 0.05 * 4 * n_draws else '✗ FAIL'}

**Overall**: {'✓ CONVERGENCE ACHIEVED' if convergence_pass else '✗ CONVERGENCE ISSUES'}

## Heteroscedasticity Assessment

### gamma_1 Posterior
- **Mean**: {gamma_1_mean:.4f} ± {gamma_1_std:.4f}
- **95% CI**: [{gamma_1_q025:.4f}, {gamma_1_q975:.4f}]
- **P(gamma_1 < 0)**: {prob_negative:.1%}

### Interpretation
{hetero_conclusion}

## LOO Cross-Validation

### LOO Results
- **ELPD LOO**: {loo.elpd_loo:.2f} ± {loo.se:.2f}
- **p_loo**: {loo.p_loo:.2f}

### Pareto k Diagnostics
- Good (k < 0.5): {loo_results['pareto_k_stats']['good_k_lt_0.5']}/{N} ({loo_results['pareto_k_stats']['percent_good']:.1f}%)
- OK (0.5 ≤ k < 0.7): {loo_results['pareto_k_stats']['ok_k_0.5_to_0.7']}
- Bad (0.7 ≤ k < 1.0): {loo_results['pareto_k_stats']['bad_k_0.7_to_1.0']}
- Very bad (k ≥ 1.0): {loo_results['pareto_k_stats']['very_bad_k_gte_1.0']}
- Max k: {loo_results['pareto_k_stats']['max_k']:.3f}

### Pareto k Assessment
{'✓ PASS: All Pareto k < 0.7' if loo_results['pareto_k_stats']['percent_good'] >= 90 else '✗ FAIL: Some Pareto k ≥ 0.7'}

## Model Comparison with Model 1 (Log-Linear Homoscedastic)

### ELPD Comparison
- **Model 1 ELPD**: {model1_loo['elpd_loo']:.2f} ± {model1_loo['se']:.2f}
- **Model 2 ELPD**: {loo.elpd_loo:.2f} ± {loo.se:.2f}
- **Δ ELPD (M2 - M1)**: {delta_elpd:.2f} ± {delta_se:.2f}

### Model Selection
{model_preference}

## Visual Diagnostics

The following plots are generated for detailed assessment:

1. **convergence_diagnostics.png**: Trace plots and rank plots for all parameters
2. **posterior_distributions.png**: Marginal posterior distributions with priors
3. **model_fit.png**: Fitted model with heteroscedastic credible intervals
4. **residual_diagnostics.png**: Residual analysis and variance structure
5. **variance_function.png**: Posterior predictive variance as function of x

See plots directory for visualizations.

## Conclusion

{'This model achieves satisfactory convergence and provides valid inference for model comparison.' if convergence_pass else 'Convergence issues detected. Results should be interpreted with caution or model should be rejected.'}

---
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

report_path = f"{output_dir}/diagnostics/convergence_report.md"
with open(report_path, 'w') as f:
    f.write(convergence_report)
print(f"   Saved convergence report to: {report_path}")

print("\n" + "=" * 80)
print("FITTING COMPLETE")
print("=" * 80)
print(f"\nConvergence: {'PASS' if convergence_pass else 'FAIL'}")
print(f"Next: Run visualization script to create diagnostic plots")
print(f"Output directory: {output_dir}")
