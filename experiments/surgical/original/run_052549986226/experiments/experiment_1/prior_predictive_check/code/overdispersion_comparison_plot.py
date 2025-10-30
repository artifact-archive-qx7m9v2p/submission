"""
Create a clear visualization comparing different overdispersion measures
to clarify the metadata discrepancy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set random seed
np.random.seed(42)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Paths
DATA_PATH = Path("/workspace/data/data.csv")
OUTPUT_DIR = Path("/workspace/experiments/experiment_1/prior_predictive_check/plots")

# Load data
data = pd.read_csv(DATA_PATH)
n_trials = data['n_trials'].values
r_success = data['r_successes'].values
group_rates = r_success / n_trials
pooled_rate = r_success.sum() / n_trials.sum()

# Calculate various overdispersion measures
# 1. Quasi-likelihood (Pearson chi-square / df)
expected = n_trials * pooled_rate
pearson_resid = (r_success - expected) / np.sqrt(expected * (1 - pooled_rate))
chi_square = np.sum(pearson_resid**2)
df = len(data) - 1
quasi_disp = chi_square / df

# 2. Beta-binomial phi
var_group_rates = np.var(group_rates, ddof=1)
kappa_est = (pooled_rate * (1 - pooled_rate) / var_group_rates) - 1
phi_bb = 1 + 1/kappa_est

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Observed vs Expected under Binomial
ax = axes[0, 0]
ax.scatter(expected, r_success, s=100, alpha=0.7, edgecolor='black', linewidth=1.5)
lim_max = max(expected.max(), r_success.max()) * 1.1
ax.plot([0, lim_max], [0, lim_max], 'r--', lw=2, label='Perfect agreement')
ax.plot([0, lim_max], [0, lim_max*1.5], 'gray', ls=':', alpha=0.5, label='50% higher')
ax.plot([0, lim_max], [0, lim_max*0.5], 'gray', ls=':', alpha=0.5, label='50% lower')

for i, (exp, obs) in enumerate(zip(expected, r_success), 1):
    ax.annotate(f'{i}', (exp, obs), xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xlabel('Expected count (under binomial)', fontsize=11)
ax.set_ylabel('Observed count', fontsize=11)
ax.set_title(f'Observed vs Expected Counts\nPearson χ² = {chi_square:.2f}, df = {df}', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Pearson Residuals
ax = axes[0, 1]
x_pos = np.arange(1, len(data) + 1)
colors = ['red' if abs(r) > 2 else 'steelblue' for r in pearson_resid]
ax.bar(x_pos, pearson_resid, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(0, color='black', linewidth=1)
ax.axhline(2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='±2 threshold')
ax.axhline(-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Group ID', fontsize=11)
ax.set_ylabel('Pearson Residual', fontsize=11)
ax.set_title(f'Pearson Residuals\nQuasi-likelihood dispersion = χ²/df = {quasi_disp:.2f}', fontsize=12)
ax.set_xticks(x_pos)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Group Rate Distribution
ax = axes[1, 0]
ax.hist(group_rates, bins=15, density=True, alpha=0.7, edgecolor='black',
        label='Observed group rates')

# Overlay theoretical binomial distribution (if all groups had same rate)
rate_range = np.linspace(0, 0.25, 1000)
# Expected variance under binomial: p(1-p)/n_avg
n_avg = n_trials.mean()
binom_std = np.sqrt(pooled_rate * (1 - pooled_rate) / n_avg)
ax.plot(rate_range, stats.norm(pooled_rate, binom_std).pdf(rate_range),
        'r--', lw=2, label=f'Binomial expectation (n̄={n_avg:.0f})')

ax.axvline(pooled_rate, color='green', linestyle=':', lw=2, label=f'Pooled rate: {pooled_rate:.3f}')
ax.set_xlabel('Success rate', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Distribution of Group Rates\nObserved variance = {var_group_rates:.6f}', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Comparison of Overdispersion Measures
ax = axes[1, 1]

measures = ['Quasi-likelihood\n(χ²/df)', 'Beta-binomial\n(φ = 1+1/κ)', 'Metadata\nclaim']
values = [quasi_disp, phi_bb, 3.5]  # Using 3.5 as midpoint of claimed 3.5-5.1
colors_bar = ['orange', 'steelblue', 'gray']

bars = ax.barh(measures, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val + 0.1, i, f'{val:.2f}', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel('Overdispersion estimate', fontsize=11)
ax.set_title('Comparison of Overdispersion Measures\n(Different definitions, different values!)',
             fontsize=12, fontweight='bold')
ax.axvline(1, color='black', linestyle=':', linewidth=1, alpha=0.5, label='No overdispersion')
ax.set_xlim(0, 5)
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Add explanation text
explanation = """Key Insight:
• Quasi-likelihood (3.51) measures aggregate deviation from binomial model
• Beta-binomial φ (1.02) measures between-group heterogeneity in p_i
• Same data, different interpretations!
• For beta-binomial model, φ ≈ 1.02 is the correct prior target"""

ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "overdispersion_explained.png", dpi=300, bbox_inches='tight')
plt.close()

print("Created overdispersion_explained.png")
print(f"\nSummary:")
print(f"  Quasi-likelihood dispersion: {quasi_disp:.2f}")
print(f"  Beta-binomial phi: {phi_bb:.2f}")
print(f"  These measure different aspects of overdispersion!")
