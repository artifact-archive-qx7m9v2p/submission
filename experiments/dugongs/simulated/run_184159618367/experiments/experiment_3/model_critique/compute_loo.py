#!/usr/bin/env python3
"""
Compute LOO-CV diagnostics for model critique.
"""

import arviz as az
import numpy as np
import json

# Load posterior
idata = az.from_netcdf('/workspace/experiments/experiment_3/posterior_inference/diagnostics/posterior_inference.netcdf')

# Compute LOO
try:
    loo_result = az.loo(idata, pointwise=True)

    # Extract key metrics
    loo_summary = {
        'elpd_loo': float(loo_result.elpd_loo),
        'se_elpd_loo': float(loo_result.se),
        'p_loo': float(loo_result.p_loo),
        'looic': float(loo_result.loo_i.sum() * -2),
        'n_high_pareto_k': int((loo_result.pareto_k > 0.7).sum()),
        'n_very_high_pareto_k': int((loo_result.pareto_k > 1.0).sum()),
        'max_pareto_k': float(loo_result.pareto_k.max()),
        'mean_pareto_k': float(loo_result.pareto_k.mean()),
        'pareto_k_values': loo_result.pareto_k.values.tolist()
    }

    # Categorize Pareto k values
    k_good = (loo_result.pareto_k < 0.5).sum()
    k_ok = ((loo_result.pareto_k >= 0.5) & (loo_result.pareto_k < 0.7)).sum()
    k_bad = ((loo_result.pareto_k >= 0.7) & (loo_result.pareto_k < 1.0)).sum()
    k_very_bad = (loo_result.pareto_k >= 1.0).sum()

    loo_summary['pareto_k_categories'] = {
        'good (k < 0.5)': int(k_good),
        'ok (0.5 <= k < 0.7)': int(k_ok),
        'bad (0.7 <= k < 1.0)': int(k_bad),
        'very_bad (k >= 1.0)': int(k_very_bad)
    }

    # Print summary
    print("LOO-CV Summary:")
    print(f"  ELPD LOO: {loo_summary['elpd_loo']:.2f} ± {loo_summary['se_elpd_loo']:.2f}")
    print(f"  p_loo: {loo_summary['p_loo']:.2f}")
    print(f"  LOOIC: {loo_summary['looic']:.2f}")
    print(f"\nPareto k diagnostics:")
    print(f"  Good (k < 0.5): {k_good}/27")
    print(f"  OK (0.5 <= k < 0.7): {k_ok}/27")
    print(f"  Bad (0.7 <= k < 1.0): {k_bad}/27")
    print(f"  Very bad (k >= 1.0): {k_very_bad}/27")
    print(f"  Max k: {loo_summary['max_pareto_k']:.3f}")
    print(f"  Mean k: {loo_summary['mean_pareto_k']:.3f}")

    # Save results
    with open('/workspace/experiments/experiment_3/model_critique/loo_diagnostics.json', 'w') as f:
        json.dump(loo_summary, f, indent=2)

    print("\nLOO diagnostics saved to loo_diagnostics.json")

    # Create diagnostic plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    az.plot_khat(loo_result, ax=ax, show_bins=True)
    ax.set_title('Pareto k Diagnostic Values', fontsize=14, fontweight='bold')
    ax.set_xlabel('Data Point Index', fontsize=12)
    ax.set_ylabel('Pareto k', fontsize=12)
    plt.tight_layout()
    plt.savefig('/workspace/experiments/experiment_3/model_critique/pareto_k_diagnostic.png', dpi=300, bbox_inches='tight')
    print("Pareto k plot saved to pareto_k_diagnostic.png")

except Exception as e:
    print(f"Error computing LOO: {e}")
    import traceback
    traceback.print_exc()
