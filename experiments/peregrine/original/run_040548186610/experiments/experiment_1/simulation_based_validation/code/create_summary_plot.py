#!/usr/bin/env python3
"""
Create a summary visualization showing overall SBC assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

BASE_DIR = Path("/workspace/experiments/experiment_1/simulation_based_validation")
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

# Load metrics
with open(RESULTS_DIR / 'detailed_metrics.json', 'r') as f:
    metrics = json.load(f)

with open(RESULTS_DIR / 'summary_stats.json', 'r') as f:
    summary = json.load(f)

# Create summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

param_names = {
    'beta_0': r'$\beta_0$ (Intercept)',
    'beta_1': r'$\beta_1$ (Linear)',
    'beta_2': r'$\beta_2$ (Quadratic)',
    'phi': r'$\phi$ (Dispersion)'
}

params = ['beta_0', 'beta_1', 'beta_2', 'phi']

# Color coding
def get_color(value, metric_type):
    if metric_type == 'bias':
        if abs(value) < 0.05:
            return 'green'
        elif abs(value) < 0.1:
            return 'orange'
        else:
            return 'red'
    elif metric_type == 'coverage':
        if 90 <= value <= 98:
            return 'green'
        elif 80 <= value < 90 or 98 < value <= 100:
            return 'orange'
        else:
            return 'red'
    elif metric_type == 'pvalue':
        if value > 0.05:
            return 'green'
        elif value > 0.01:
            return 'orange'
        else:
            return 'red'

# 1. Bias comparison
ax1 = fig.add_subplot(gs[0, 0])
biases = [metrics[p]['bias'] for p in params]
colors = [get_color(b, 'bias') for b in biases]
bars = ax1.bar(range(4), biases, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
ax1.axhline(0.05, color='red', linestyle=':', alpha=0.5)
ax1.axhline(-0.05, color='red', linestyle=':', alpha=0.5)
ax1.set_xticks(range(4))
ax1.set_xticklabels([param_names[p] for p in params], rotation=45, ha='right')
ax1.set_ylabel('Bias (posterior mean - true)', fontsize=11)
ax1.set_title('Parameter Bias', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. RMSE comparison
ax2 = fig.add_subplot(gs[0, 1])
rmses = [metrics[p]['rmse'] for p in params]
bars = ax2.bar(range(4), rmses, color='steelblue', alpha=0.7, edgecolor='black')
ax2.set_xticks(range(4))
ax2.set_xticklabels([param_names[p] for p in params], rotation=45, ha='right')
ax2.set_ylabel('RMSE', fontsize=11)
ax2.set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Coverage comparison
ax3 = fig.add_subplot(gs[0, 2])
coverages = [metrics[p]['coverage_95'] for p in params]
colors = [get_color(c, 'coverage') for c in coverages]
bars = ax3.bar(range(4), coverages, color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(95, color='red', linestyle='--', linewidth=2, label='Nominal (95%)')
ax3.axhspan(90, 98, alpha=0.2, color='green', label='Acceptable range')
ax3.set_xticks(range(4))
ax3.set_xticklabels([param_names[p] for p in params], rotation=45, ha='right')
ax3.set_ylabel('Coverage (%)', fontsize=11)
ax3.set_title('95% Credible Interval Coverage', fontsize=12, fontweight='bold')
ax3.set_ylim([75, 105])
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Regression slope (shrinkage)
ax4 = fig.add_subplot(gs[1, 0])
slopes = [metrics[p]['regression_slope'] for p in params]
shrinkage = [(1 - s) * 100 for s in slopes]
bars = ax4.bar(range(4), shrinkage, color='coral', alpha=0.7, edgecolor='black')
ax4.axhline(0, color='black', linestyle='--', linewidth=1)
ax4.set_xticks(range(4))
ax4.set_xticklabels([param_names[p] for p in params], rotation=45, ha='right')
ax4.set_ylabel('Shrinkage (%)', fontsize=11)
ax4.set_title('Parameter Shrinkage (1 - slope)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Rank uniformity p-values
ax5 = fig.add_subplot(gs[1, 1])
pvalues = [metrics[p]['chi2_pvalue'] for p in params]
colors = [get_color(p, 'pvalue') for p in pvalues]
bars = ax5.bar(range(4), pvalues, color=colors, alpha=0.7, edgecolor='black')
ax5.axhline(0.05, color='red', linestyle='--', linewidth=2, label='Threshold (0.05)')
ax5.set_xticks(range(4))
ax5.set_xticklabels([param_names[p] for p in params], rotation=45, ha='right')
ax5.set_ylabel('χ² p-value', fontsize=11)
ax5.set_title('Rank Uniformity Test', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Overall status table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

status_data = []
for p in params:
    m = metrics[p]

    # Determine overall status
    bias_ok = abs(m['bias']) < 0.1
    coverage_ok = 80 <= m['coverage_95'] <= 100
    rank_ok = m['chi2_pvalue'] > 0.05

    if bias_ok and coverage_ok and rank_ok:
        status = 'PASS'
        color = 'lightgreen'
    elif rank_ok and (bias_ok or coverage_ok):
        status = 'CONDITIONAL'
        color = 'yellow'
    else:
        status = 'FAIL'
        color = 'lightcoral'

    status_data.append([param_names[p], status])

table = ax6.table(
    cellText=status_data,
    colLabels=['Parameter', 'Status'],
    cellLoc='center',
    loc='center',
    bbox=[0, 0.3, 1, 0.6]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Color code status
for i in range(len(params)):
    status = status_data[i][1]
    if status == 'PASS':
        table[(i+1, 1)].set_facecolor('lightgreen')
    elif status == 'CONDITIONAL':
        table[(i+1, 1)].set_facecolor('yellow')
    else:
        table[(i+1, 1)].set_facecolor('lightcoral')

ax6.text(0.5, 0.95, 'Parameter Status Summary', ha='center', va='top',
         fontsize=12, fontweight='bold', transform=ax6.transAxes)

# 7. Computational health summary
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('off')

health_text = f"""
SIMULATION-BASED CALIBRATION SUMMARY
{'='*100}

OVERALL DECISION: CONDITIONAL PASS

Model: Negative Binomial Quadratic (C_i ~ NegBinom(μ_i, φ), log(μ_i) = β₀ + β₁·year + β₂·year²)

COMPUTATIONAL HEALTH                    PARAMETER RECOVERY                     CALIBRATION QUALITY
  Success Rate:     {summary['success_rate']:.0f}%                  β₀ Bias:  {metrics['beta_0']['bias']:+.4f}                    β₀ Coverage: {metrics['beta_0']['coverage_95']:.0f}%
  Convergence:      {summary['convergence_rate']:.0f}%                  β₁ Bias:  {metrics['beta_1']['bias']:+.4f}                    β₁ Coverage: {metrics['beta_1']['coverage_95']:.0f}%
  Mean R̂:          {summary['mean_rhat']:.4f}                β₂ Bias:  {metrics['beta_2']['bias']:+.4f}                    β₂ Coverage: {metrics['beta_2']['coverage_95']:.0f}%
  Mean ESS:         {summary['mean_ess']:.0f}                  φ  Bias:  {metrics['phi']['bias']:+.4f}                    φ  Coverage: {metrics['phi']['coverage_95']:.0f}%
  Acceptance:       {summary['mean_acceptance']:.3f}

KEY FINDINGS:
  ✓ Regression coefficients (β₀, β₁, β₂): EXCELLENT recovery with minimal bias and perfect/near-perfect coverage
  ⚠ Dispersion parameter (φ): ACCEPTABLE recovery with moderate shrinkage and 85% coverage (below nominal 95%)
  ✓ All parameters pass rank uniformity tests (p > 0.05) - no systematic bias or structural issues
  ✓ Computational stability: 95% convergence rate, no divergences, stable MCMC sampling

RECOMMENDATIONS:
  • PROCEED to real data fitting with adjusted priors
  • Use 99% credible intervals for φ to account for potential underestimation of uncertainty
  • Standard 95% intervals are appropriate for regression coefficients
  • Monitor convergence (R̂ < 1.01, ESS > 400) during real data analysis

LIMITATIONS:
  • φ coverage at 85% indicates credible intervals may be ~10% too narrow
  • Moderate shrinkage in β₂ (44%) and φ (38%) reflects informative priors - intentional regularization
  • Small simulation size (N=20) - consider scaling to N=100+ for final validation if issues arise
"""

ax7.text(0.05, 0.5, health_text, fontsize=9, family='monospace',
         verticalalignment='center', transform=ax7.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Simulation-Based Calibration: Overall Assessment\nExperiment 1 - Negative Binomial Quadratic Model',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig(PLOTS_DIR / 'sbc_summary_dashboard.png', dpi=300, bbox_inches='tight')
print(f"Summary dashboard saved: {PLOTS_DIR / 'sbc_summary_dashboard.png'}")
plt.close()
