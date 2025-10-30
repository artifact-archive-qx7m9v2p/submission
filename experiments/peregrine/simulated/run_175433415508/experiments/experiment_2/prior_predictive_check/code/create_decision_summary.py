"""
Create a decision summary visualization for prior predictive check
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Create summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Title
fig.suptitle('Prior Predictive Check Decision Summary: FAIL',
             fontsize=18, fontweight='bold', color='darkred', y=0.98)

# Key statistics (from the run)
stats = {
    'median_count': 112,
    'max_count': 674970346,
    'pct_above_5000': 4.72,
    'pct_above_10000': 3.22,
    'mean_acf1_epsilon': 0.766,
    'expected_acf1': 0.909,
    'rho_acf_corr': 0.3926,
    'mean_growth_rate': 207.1,
    'observed_max': 269,
    'n_obs': 40
}

# ============================================================================
# Panel 1: Pass/Fail Checklist
# ============================================================================
ax = fig.add_subplot(gs[0, :])
ax.axis('off')

checklist = [
    ('✓', 'Count range plausibility (95% < 5000)', f"{stats['pct_above_5000']:.1f}% above 5000", 'green'),
    ('✗', 'Extreme outlier control (<1% above 10k)', f"{stats['pct_above_10000']:.1f}% above 10,000", 'red'),
    ('✓', 'AR coefficient concentration (0.7-0.95)', f"Mean ρ = {stats['expected_acf1']:.3f}", 'green'),
    ('✗', 'Temporal correlation realistic', 'ACF computation failed (NaN)', 'red'),
    ('✓', 'Growth pattern plausible', f"{stats['mean_growth_rate']:.1f}% mean growth", 'green'),
    ('✓', 'Innovation scale appropriate', 'Mean σ = 0.522 < 2.0', 'green'),
    ('✗', 'AR(1) process validation', f"ρ-ACF correlation = {stats['rho_acf_corr']:.3f} < 0.95", 'red'),
]

y_start = 0.85
y_step = 0.12

ax.text(0.5, 0.95, 'VALIDATION CHECKLIST', ha='center', fontsize=14,
        fontweight='bold', transform=ax.transAxes)

for i, (symbol, test, result, color) in enumerate(checklist):
    y = y_start - i * y_step
    ax.text(0.05, y, symbol, fontsize=20, color=color, transform=ax.transAxes, fontweight='bold')
    ax.text(0.12, y, test, fontsize=11, transform=ax.transAxes, va='center')
    ax.text(0.98, y, result, fontsize=10, transform=ax.transAxes, va='center', ha='right',
           style='italic', color=color)

# Overall verdict
ax.text(0.5, 0.02, 'OVERALL: FAIL (3/7 checks failed)', ha='center', fontsize=13,
       fontweight='bold', color='darkred', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7, pad=0.5))

# ============================================================================
# Panel 2: The Problem - Scale Comparison
# ============================================================================
ax = fig.add_subplot(gs[1, 0])

categories = ['Observed\nMax', '95th\nPercentile', '99th\nPercentile', 'Maximum\nGenerated']
values = [269, 4503, 143745, 674970346]
colors_bar = ['green', 'orange', 'red', 'darkred']

bars = ax.barh(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_xlabel('Count Value', fontsize=11, fontweight='bold')
ax.set_xscale('log')
ax.set_title('Problem: Extreme Tail Behavior', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, values):
    if val < 10000:
        label = f'{val:,.0f}'
    elif val < 1000000:
        label = f'{val/1000:.0f}K'
    else:
        label = f'{val/1000000:.0f}M'
    ax.text(val, bar.get_y() + bar.get_height()/2, f'  {label}',
           va='center', fontsize=10, fontweight='bold')

ax.axvline(5000, color='orange', linestyle='--', lw=2, alpha=0.5, label='Plausible limit')
ax.axvline(10000, color='red', linestyle='--', lw=2, alpha=0.5, label='Extreme threshold')
ax.legend(fontsize=9, loc='lower right')

# ============================================================================
# Panel 3: Root Cause - Multiplicative Explosion
# ============================================================================
ax = fig.add_subplot(gs[1, 1])
ax.axis('off')

ax.text(0.5, 0.95, 'Root Cause: Multiplicative Explosion', ha='center',
       fontsize=12, fontweight='bold', transform=ax.transAxes)

explanation = """When rare parameter combinations occur:

1. β₀ from upper tail: 6.5 (3% probability)
2. β₁ from upper tail: 2.0 (2.5% probability)
3. σ from upper tail: 1.5 (8% probability)
4. ε_t reaches +3σ: 4.5

Then: η = 6.5 + 2.0×1.67 + 4.5 = 14.3
      μ = exp(14.3) = 1.6 MILLION

With 500 draws × 40 timepoints = 20,000 samples
→ ~100 extreme values expected

The exponential link amplifies tail events!
"""

ax.text(0.05, 0.75, explanation, fontsize=10, transform=ax.transAxes,
       va='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# Panel 4: AR(1) Validation Problem
# ============================================================================
ax = fig.add_subplot(gs[1, 2])
ax.axis('off')

ax.text(0.5, 0.95, 'AR(1) Process Issue', ha='center',
       fontsize=12, fontweight='bold', transform=ax.transAxes)

ar_explanation = """Expected: ρ parameter ≈ ACF(1) of ε

Observed: Correlation = 0.39 (should be >0.95)

Possible causes:
• Short time series (N=40)
• Near-unit-root instability (ρ≈0.91)
• Trend component interference
• Initialization issues

Mean ACF(1) of ε: 0.766
Expected (E[ρ]): 0.909

Discrepancy: 0.143 (16% error)

This suggests the AR(1) process is not
fully capturing the intended correlation
structure in finite samples.
"""

ax.text(0.05, 0.75, ar_explanation, fontsize=9, transform=ax.transAxes,
       va='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ============================================================================
# Panel 5: Recommended Prior Changes
# ============================================================================
ax = fig.add_subplot(gs[2, :2])
ax.axis('off')

ax.text(0.5, 0.95, 'RECOMMENDED PRIOR REVISIONS', ha='center',
       fontsize=13, fontweight='bold', transform=ax.transAxes,
       color='darkgreen')

recommendations = """
CURRENT PRIORS                                      RECOMMENDED PRIORS (Version 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

β₀ ~ Normal(4.69, 1.0)                         →    β₀ ~ Normal(4.69, 1.0)                    [Keep]
β₁ ~ Normal(1.0, 0.5)                          →    β₁ ~ TruncatedNormal(1.0, 0.5, -0.5, 2.0) [CHANGE]
φ  ~ Gamma(2, 0.1)        [mean=20]            →    φ  ~ Normal(35, 15)                       [CHANGE]
ρ  ~ Beta(20, 2)          [mean=0.909]         →    ρ  ~ Beta(20, 2)                          [Keep]
σ  ~ Exponential(2)       [mean=0.5]           →    σ  ~ Exponential(5)                       [CHANGE]

KEY CHANGES:
1. TRUNCATE β₁: Prevent extreme growth (>1000%) that creates explosive dynamics
2. INFORM φ: Use Experiment 1 posterior (35.6 ± 10.8) to constrain variance
3. TIGHTEN σ: Reduce innovation scale from E[σ]=0.5 to E[σ]=0.2 for stability

EXPECTED IMPACT:
• 99th percentile counts: 143,745 → ~5,000  (97% reduction)
• Extreme outliers (>10K): 3.22% → <0.1%    (32× reduction)
• Maintains temporal correlation flexibility
• Computational stability improved
"""

ax.text(0.02, 0.75, recommendations, fontsize=9, transform=ax.transAxes,
       va='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# ============================================================================
# Panel 6: What We Learned
# ============================================================================
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

ax.text(0.5, 0.95, 'Key Insights', ha='center',
       fontsize=12, fontweight='bold', transform=ax.transAxes)

insights = """
✓ ρ ~ Beta(20,2) is appropriate
  (motivated by EDA ACF=0.971)

✓ Median behavior is reasonable
  (50th percentile ≈ observed)

✓ AR(1) structure is sound
  (trajectories show persistence)

✗ Tail behavior is catastrophic
  (multiplicative explosion)

✗ Need tighter innovation prior
  (σ controls AR variability)

✗ Need to inform φ from Exp1
  (reduce variance uncertainty)

Prior predictive check WORKED:
Caught issues before fitting!
"""

ax.text(0.05, 0.75, insights, fontsize=9, transform=ax.transAxes,
       va='top',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# Save
output_path = Path('/workspace/experiments/experiment_2/prior_predictive_check/plots/decision_summary.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Decision summary saved to: {output_path}")
