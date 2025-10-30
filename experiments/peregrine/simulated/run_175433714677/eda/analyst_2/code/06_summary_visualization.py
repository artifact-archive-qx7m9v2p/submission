"""
Summary Visualization
Focus: Key findings in a single comprehensive figure
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import poisson, nbinom
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load data
with open('/workspace/data/data_analyst_2.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame({'C': data['C'], 'year': data['year']})
df['time_index'] = np.arange(len(df))

mean_C = df['C'].mean()
var_C = df['C'].var(ddof=1)
r_nb = 1.5493
p_nb = 0.0140

# ============================================================================
# COMPREHENSIVE SUMMARY FIGURE
# ============================================================================
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel 1: Time series with trend (top left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df['time_index'], df['C'], 'o-', color='steelblue', linewidth=2, markersize=6, alpha=0.7)
# Add rolling mean
rolling = df['C'].rolling(window=8, center=True).mean()
ax1.plot(df['time_index'], rolling, 'r-', linewidth=3, label='Rolling mean (w=8)')
ax1.set_xlabel('Time Index', fontsize=10)
ax1.set_ylabel('Count (C)', fontsize=10)
ax1.set_title('A. Count Time Series', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Distribution histogram (top middle)
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df['C'], bins=20, alpha=0.7, color='steelblue', edgecolor='black', density=True)
# Overlay distributions
x_vals = np.linspace(df['C'].min(), df['C'].max(), 200)
nb_pdf = nbinom.pmf(np.round(x_vals).astype(int), r_nb, p_nb)
ax2.plot(x_vals, nb_pdf, 'g-', linewidth=3, label=f'NegBinom(r={r_nb:.2f})', alpha=0.8)
ax2.axvline(mean_C, color='darkred', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean={mean_C:.1f}')
ax2.set_xlabel('Count (C)', fontsize=10)
ax2.set_ylabel('Density / PMF', fontsize=10)
ax2.set_title('B. Distribution with NB Fit', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Mean-Variance relationship (top right)
ax3 = fig.add_subplot(gs[0, 2])
n_segments = 8
segment_size = len(df) // n_segments
segment_means = []
segment_vars = []
for i in range(n_segments):
    start = i * segment_size
    end = start + segment_size if i < n_segments - 1 else len(df)
    segment = df['C'].iloc[start:end]
    segment_means.append(segment.mean())
    segment_vars.append(segment.var(ddof=1))
ax3.scatter(segment_means, segment_vars, s=120, alpha=0.7, color='steelblue', edgecolors='black', linewidth=2)
# Reference line
max_val = max(max(segment_means), max(segment_vars))
x_ref = np.linspace(0, max_val, 100)
ax3.plot(x_ref, x_ref, 'r--', linewidth=2, alpha=0.5, label='Var=Mean (Poisson)')
# Quadratic fit
z = np.polyfit(segment_means, segment_vars, 2)
p = np.poly1d(z)
x_fit = np.linspace(min(segment_means), max(segment_means), 100)
ax3.plot(x_fit, p(x_fit), 'g-', linewidth=2.5, alpha=0.7, label='Quadratic fit')
ax3.set_xlabel('Segment Mean', fontsize=10)
ax3.set_ylabel('Segment Variance', fontsize=10)
ax3.set_title('C. Mean-Variance Relationship', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Q-Q plot for NB (middle left)
ax4 = fig.add_subplot(gs[1, 0])
theoretical_quantiles = nbinom.ppf(np.linspace(0.01, 0.99, len(df)), r_nb, p_nb)
observed_quantiles = np.sort(df['C'])
ax4.scatter(theoretical_quantiles, observed_quantiles, alpha=0.7, s=50, color='steelblue')
min_val = min(theoretical_quantiles.min(), observed_quantiles.min())
max_val = max(theoretical_quantiles.max(), observed_quantiles.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax4.set_xlabel('Theoretical Quantiles (NB)', fontsize=10)
ax4.set_ylabel('Observed Quantiles', fontsize=10)
ax4.set_title('D. Q-Q Plot (Negative Binomial)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Panel 5: Dispersion over time (middle middle)
ax5 = fig.add_subplot(gs[1, 1])
n_groups = 4
group_size = len(df) // n_groups
vm_ratios = []
group_labels = []
for i in range(n_groups):
    start = i * group_size
    end = start + group_size if i < n_groups - 1 else len(df)
    group_data = df['C'].iloc[start:end]
    vm_ratios.append(group_data.var(ddof=1) / group_data.mean())
    group_labels.append(f'Q{i+1}')
ax5.bar(range(n_groups), vm_ratios, color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
ax5.axhline(1, color='red', linestyle='--', linewidth=2, label='Poisson (ratio=1)')
ax5.set_xlabel('Time Quartile', fontsize=10)
ax5.set_ylabel('Variance / Mean', fontsize=10)
ax5.set_title('E. Dispersion by Time Period', fontsize=11, fontweight='bold')
ax5.set_xticks(range(n_groups))
ax5.set_xticklabels(group_labels)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Panel 6: Pearson residuals (middle right)
ax6 = fig.add_subplot(gs[1, 2])
nb_mean = r_nb * (1 - p_nb) / p_nb
nb_var = r_nb * (1 - p_nb) / p_nb**2
pearson_resid = (df['C'] - nb_mean) / np.sqrt(nb_var)
ax6.scatter(df['time_index'], pearson_resid, alpha=0.7, s=50, color='steelblue')
ax6.axhline(0, color='red', linestyle='--', linewidth=2)
ax6.axhline(2, color='orange', linestyle=':', linewidth=1.5)
ax6.axhline(-2, color='orange', linestyle=':', linewidth=1.5)
ax6.set_xlabel('Time Index', fontsize=10)
ax6.set_ylabel('Pearson Residual', fontsize=10)
ax6.set_title('F. Residuals (Negative Binomial)', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Panel 7: Model comparison (bottom - spans 3 columns)
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('off')

# Create summary table
summary_text = """
KEY FINDINGS SUMMARY

DISTRIBUTION PROPERTIES:
  • Mean: 109.40  |  Variance: 7704.66  |  Variance/Mean Ratio: 70.43
  • Strong evidence of OVERDISPERSION (variance >> mean)
  • Skewness: 0.64 (right-skewed)  |  Kurtosis: -1.13 (light-tailed)

HYPOTHESIS TEST RESULTS:
  ✗ Poisson Distribution: REJECTED (χ² p < 0.001, KS p < 0.001)
  ✓ Negative Binomial: ACCEPTED (KS p = 0.261, LR test p < 0.001)
  ✓ No Zero-Inflation: No zeros observed in dataset
  ✗ Constant Dispersion: REJECTED (Levene test p = 0.010)

MODEL COMPARISON (AIC):
  Rank 1: Log-Normal (AIC = 453.99) - continuous approximation
  Rank 2: Gamma (AIC = 455.73) - continuous approximation
  Rank 3: Negative Binomial (AIC = 455.98) - discrete, RECOMMENDED
  Rank 4: Poisson (AIC = 2954.11) - completely inadequate

NEGATIVE BINOMIAL PARAMETERS (MLE):
  • r (dispersion): 1.549 (low value indicates high overdispersion)
  • p (probability): 0.0140
  • Implied mean: 109.40  |  Implied variance: 7834.30

TEMPORAL PATTERNS:
  • Dispersion varies significantly across time (r ranges from 1.0 to 69.1)
  • Early period: Low mean (27.5), low dispersion (Var/Mean = 0.58)
  • Late period: High mean (237.8), moderate dispersion (Var/Mean = 4.44)
  • Middle period shows highest dispersion (Var/Mean = 11.85)

MODELING RECOMMENDATIONS:
  1. Use NEGATIVE BINOMIAL likelihood for Bayesian modeling
  2. Consider time-varying dispersion parameter if trend modeling included
  3. Log-Normal approximation acceptable for continuous modeling contexts
  4. Poisson distribution is completely inappropriate for this data
"""

ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1))

plt.suptitle('Count Distribution Analysis: Comprehensive Summary', fontsize=14, fontweight='bold', y=0.995)
plt.savefig('/workspace/eda/analyst_2/visualizations/10_comprehensive_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: 10_comprehensive_summary.png")
print("\n=== Summary visualization complete ===")
