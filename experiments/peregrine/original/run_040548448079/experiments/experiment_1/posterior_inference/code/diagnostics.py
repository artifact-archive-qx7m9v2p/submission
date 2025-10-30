"""
Comprehensive convergence diagnostics for posterior inference.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE CONVERGENCE DIAGNOSTICS")
print("="*80)

# Load inference data
print("\n[1/4] Loading inference data...")
trace = az.from_netcdf('/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf')
print("  Loaded successfully")

# Generate detailed summary table
print("\n[2/4] Generating detailed summary statistics...")
summary = az.summary(
    trace,
    var_names=['beta_0', 'beta_1', 'beta_2', 'alpha'],
    stat_funcs={'q5': lambda x: np.percentile(x, 5),
                'q95': lambda x: np.percentile(x, 95)}
)

# Save summary table
summary_path = '/workspace/experiments/experiment_1/posterior_inference/diagnostics/summary_table.csv'
summary.to_csv(summary_path)
print(f"  Saved summary table to: {summary_path}")

# Print formatted summary
print("\n" + "="*80)
print("PARAMETER SUMMARY STATISTICS")
print("="*80)
print(summary.to_string())

# Check convergence criteria
print("\n[3/4] Checking convergence criteria...")

rhat_max = summary['r_hat'].max()
ess_bulk_min = summary['ess_bulk'].min()
ess_tail_min = summary['ess_tail'].min()
mcse_sd_max = summary['mcse_sd'].max()

# Get divergences
divergences = trace.sample_stats.diverging.sum().values
n_samples = trace.posterior.dims['draw'] * trace.posterior.dims['chain']
div_pct = 100 * divergences / n_samples

# Get energy diagnostics
energy = trace.sample_stats.energy.values.flatten()
energy_diff = np.diff(energy)
bfmi = np.var(energy_diff) / np.var(energy) if len(energy) > 1 else 0

# Convergence report
print("\n" + "="*80)
print("CONVERGENCE CRITERIA")
print("="*80)

criteria_passed = []

print("\n1. R-hat Diagnostic:")
print(f"   Max R-hat: {rhat_max:.6f}")
print(f"   Target: < 1.01")
if rhat_max < 1.01:
    print("   Status: ✓ PASS")
    criteria_passed.append(True)
else:
    print("   Status: ✗ FAIL")
    criteria_passed.append(False)

print("\n2. Effective Sample Size (ESS):")
print(f"   Min ESS bulk: {ess_bulk_min:.0f}")
print(f"   Min ESS tail: {ess_tail_min:.0f}")
print(f"   Target: > 400 for both")
if ess_bulk_min > 400 and ess_tail_min > 400:
    print("   Status: ✓ PASS")
    criteria_passed.append(True)
else:
    print("   Status: ✗ FAIL")
    criteria_passed.append(False)

print("\n3. Divergent Transitions:")
print(f"   Count: {divergences} / {n_samples} ({div_pct:.2f}%)")
print(f"   Target: < 1%")
if div_pct < 1.0:
    print("   Status: ✓ PASS")
    criteria_passed.append(True)
else:
    print("   Status: ✗ FAIL")
    criteria_passed.append(False)

print("\n4. MCSE (Monte Carlo Standard Error):")
print(f"   Max MCSE/SD: {mcse_sd_max:.4f}")
print(f"   Target: < 0.05 (5% of posterior SD)")
if mcse_sd_max < 0.05:
    print("   Status: ✓ PASS")
    criteria_passed.append(True)
else:
    print("   Status: ✗ FAIL")
    criteria_passed.append(False)

print("\n5. Energy Diagnostic (BFMI):")
print(f"   BFMI: {bfmi:.4f}")
print(f"   Target: > 0.3")
if bfmi > 0.3:
    print("   Status: ✓ PASS")
    criteria_passed.append(True)
else:
    print("   Status: ✗ FAIL (indicates pathological posterior geometry)")
    criteria_passed.append(False)

# Overall verdict
print("\n" + "="*80)
if all(criteria_passed):
    print("OVERALL: ✓ ALL CONVERGENCE CRITERIA PASSED")
else:
    print("OVERALL: ✗ CONVERGENCE ISSUES DETECTED")
print("="*80)

# Save convergence report
print("\n[4/4] Saving convergence report...")
report_path = '/workspace/experiments/experiment_1/posterior_inference/diagnostics/convergence_report.txt'

with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("CONVERGENCE DIAGNOSTICS REPORT\n")
    f.write("Experiment 1: Fixed Changepoint Negative Binomial Regression\n")
    f.write("="*80 + "\n\n")

    f.write("SAMPLING CONFIGURATION\n")
    f.write("-"*80 + "\n")
    f.write(f"Chains: {trace.posterior.dims['chain']}\n")
    f.write(f"Draws per chain: {trace.posterior.dims['draw']}\n")
    f.write(f"Total draws: {n_samples}\n")
    f.write(f"Parameters: beta_0, beta_1, beta_2, alpha\n\n")

    f.write("CONVERGENCE CRITERIA\n")
    f.write("-"*80 + "\n")
    f.write(f"1. R-hat: {rhat_max:.6f} (target: < 1.01) - {'PASS' if rhat_max < 1.01 else 'FAIL'}\n")
    f.write(f"2. ESS bulk: {ess_bulk_min:.0f} (target: > 400) - {'PASS' if ess_bulk_min > 400 else 'FAIL'}\n")
    f.write(f"3. ESS tail: {ess_tail_min:.0f} (target: > 400) - {'PASS' if ess_tail_min > 400 else 'FAIL'}\n")
    f.write(f"4. Divergences: {div_pct:.2f}% (target: < 1%) - {'PASS' if div_pct < 1 else 'FAIL'}\n")
    f.write(f"5. MCSE/SD: {mcse_sd_max:.4f} (target: < 0.05) - {'PASS' if mcse_sd_max < 0.05 else 'FAIL'}\n")
    f.write(f"6. BFMI: {bfmi:.4f} (target: > 0.3) - {'PASS' if bfmi > 0.3 else 'FAIL'}\n\n")

    f.write("PARAMETER-LEVEL DIAGNOSTICS\n")
    f.write("-"*80 + "\n")
    f.write(summary[['mean', 'sd', 'ess_bulk', 'ess_tail', 'r_hat']].to_string())
    f.write("\n\n")

    f.write("OVERALL VERDICT\n")
    f.write("-"*80 + "\n")
    if all(criteria_passed):
        f.write("✓ ALL CONVERGENCE CRITERIA MET\n")
        f.write("The posterior inference is reliable and can be used for scientific conclusions.\n")
    else:
        f.write("✗ CONVERGENCE ISSUES DETECTED\n")
        f.write("The posterior inference may not be reliable. Review diagnostics carefully.\n")

print(f"  Saved convergence report to: {report_path}")

print("\n" + "="*80)
print("DIAGNOSTICS COMPLETE")
print("="*80)
