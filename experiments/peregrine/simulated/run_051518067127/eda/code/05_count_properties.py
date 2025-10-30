"""
Count Data Properties and Overdispersion Analysis
================================================
Goal: Test assumptions for count models and assess overdispersion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/workspace/data/data.csv')

print("="*80)
print("COUNT DATA PROPERTIES ANALYSIS")
print("="*80)

# 1. Mean-variance relationship
print("\n1. MEAN-VARIANCE RELATIONSHIP")
print("-" * 80)

# Overall
mean_C = data['C'].mean()
var_C = data['C'].var(ddof=1)
ratio = var_C / mean_C

print(f"Overall statistics:")
print(f"  Mean: {mean_C:.4f}")
print(f"  Variance: {var_C:.4f}")
print(f"  Variance-to-Mean Ratio: {ratio:.4f}")
print(f"\nInterpretation:")
print(f"  Poisson assumption (var=mean): {'VIOLATED' if abs(ratio - 1) > 0.5 else 'Possibly valid'}")
print(f"  {'SEVERE OVERDISPERSION' if ratio > 10 else 'Overdispersion' if ratio > 1 else 'Equidispersion'}")

# By time periods
data['time_idx'] = np.arange(len(data))
data['period'] = pd.cut(data['time_idx'], bins=3, labels=['Early', 'Middle', 'Late'])

print(f"\nMean-variance relationship by period:")
for period in ['Early', 'Middle', 'Late']:
    period_data = data[data['period'] == period]['C']
    p_mean = period_data.mean()
    p_var = period_data.var(ddof=1)
    p_ratio = p_var / p_mean
    print(f"  {period:6s}: mean={p_mean:7.2f}, var={p_var:8.2f}, ratio={p_ratio:6.2f}")

# 2. Test for Poisson distribution
print("\n2. GOODNESS-OF-FIT FOR POISSON DISTRIBUTION")
print("-" * 80)

# Chi-square goodness of fit test
from scipy.stats import poisson, chisquare

# Use observed frequencies
observed_counts = data['C'].values
mean_count = observed_counts.mean()

# Create bins for chi-square test
bins = np.percentile(observed_counts, [0, 25, 50, 75, 100])
observed_freq, _ = np.histogram(observed_counts, bins=bins)

# Expected frequencies under Poisson
expected_freq = []
for i in range(len(bins) - 1):
    prob = poisson.cdf(bins[i+1], mean_count) - poisson.cdf(bins[i], mean_count)
    expected_freq.append(prob * len(observed_counts))

expected_freq = np.array(expected_freq)

# Chi-square test
chi2_stat, p_value = chisquare(observed_freq, expected_freq)
print(f"Chi-square goodness-of-fit test:")
print(f"  Chi-square statistic: {chi2_stat:.4f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Conclusion: {'Reject Poisson' if p_value < 0.05 else 'Cannot reject Poisson'}")

# 3. Index of dispersion
print("\n3. DISPERSION INDICES")
print("-" * 80)

# Index of dispersion (variance-to-mean ratio with test)
dispersion_index = var_C / mean_C
print(f"Index of Dispersion: {dispersion_index:.4f}")

# Test statistic under Poisson assumption
# Under Poisson: (n-1)*D/mean ~ Chi-square(n-1)
n = len(data)
test_stat = (n - 1) * dispersion_index
df = n - 1
p_value_disp = 1 - stats.chi2.cdf(test_stat, df)
print(f"Test for overdispersion:")
print(f"  Test statistic: {test_stat:.4f}")
print(f"  Degrees of freedom: {df}")
print(f"  P-value: {p_value_disp:.4e}")
print(f"  Conclusion: {'Significant overdispersion' if p_value_disp < 0.05 else 'No significant overdispersion'}")

# Coefficient of variation
cv = data['C'].std() / data['C'].mean()
print(f"\nCoefficient of Variation: {cv:.4f}")
print(f"  (For Poisson: CV = 1/sqrt(mean) = {1/np.sqrt(mean_C):.4f})")
print(f"  Observed CV is {'much larger' if cv > 2/np.sqrt(mean_C) else 'larger' if cv > 1/np.sqrt(mean_C) else 'similar'} than Poisson expectation")

# 4. Zero-inflation check
print("\n4. ZERO-INFLATION ASSESSMENT")
print("-" * 80)
n_zeros = (data['C'] == 0).sum()
n_low = (data['C'] < 5).sum()
min_count = data['C'].min()

print(f"Number of zeros: {n_zeros}")
print(f"Number of counts < 5: {n_low}")
print(f"Minimum count: {min_count}")
print(f"Percentage of zeros: {100 * n_zeros / len(data):.2f}%")
print(f"\nConclusion: {'No zero-inflation' if n_zeros == 0 else f'Zero-inflation present ({n_zeros} zeros)'}")

# Expected zeros under Poisson
expected_zeros_poisson = len(data) * np.exp(-mean_C)
print(f"\nExpected zeros under Poisson(mean={mean_C:.2f}): {expected_zeros_poisson:.4f}")
print(f"Observed zeros: {n_zeros}")

# 5. Alternative distributions to consider
print("\n5. ALTERNATIVE DISTRIBUTIONS")
print("-" * 80)

# Negative Binomial parameters estimation
# Method of moments: var = mean + mean^2/size
# size = mean^2 / (var - mean)
if var_C > mean_C:
    nb_size = mean_C**2 / (var_C - mean_C)
    nb_prob = mean_C / var_C
    print(f"Negative Binomial (method of moments):")
    print(f"  Size parameter (r): {nb_size:.4f}")
    print(f"  Probability parameter (p): {nb_prob:.4f}")
    print(f"  Interpretation: Allows for variance = mean + mean²/r")

# Log-normal fit
log_C = np.log(data['C'])
lognorm_mu = log_C.mean()
lognorm_sigma = log_C.std()
print(f"\nLog-Normal parameters:")
print(f"  mu (log-scale mean): {lognorm_mu:.4f}")
print(f"  sigma (log-scale std): {lognorm_sigma:.4f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Count Data Properties and Overdispersion', fontsize=16, y=0.995)

# 1. Mean-variance plot by period
ax = axes[0, 0]
periods = ['Early', 'Middle', 'Late']
period_means = []
period_vars = []
period_colors = ['lightblue', 'lightgreen', 'lightcoral']

for period in periods:
    period_data = data[data['period'] == period]['C']
    period_means.append(period_data.mean())
    period_vars.append(period_data.var(ddof=1))

ax.scatter(period_means, period_vars, s=200, c=period_colors, edgecolor='black',
           linewidth=2, alpha=0.7, zorder=3)

# Add labels
for i, period in enumerate(periods):
    ax.annotate(period, (period_means[i], period_vars[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

# Add reference lines
mean_range = np.linspace(0, max(period_means) * 1.1, 100)
ax.plot(mean_range, mean_range, 'r--', linewidth=2, label='Poisson (var=mean)', alpha=0.7)
ax.plot(mean_range, mean_range * 2, 'orange', linestyle='--',
        linewidth=1.5, label='var=2*mean', alpha=0.7)

# Add overall point
ax.scatter([mean_C], [var_C], s=300, c='red', marker='*',
           edgecolor='black', linewidth=2, label='Overall', zorder=4)

ax.set_xlabel('Mean')
ax.set_ylabel('Variance')
ax.set_title('Mean-Variance Relationship by Period')
ax.legend()
ax.grid(alpha=0.3)

# 2. Histogram with Poisson overlay
ax = axes[0, 1]
counts, bins, _ = ax.hist(data['C'], bins=15, density=True, alpha=0.7,
                          color='steelblue', edgecolor='black', label='Observed')

# Overlay Poisson PMF
x_range = np.arange(data['C'].min(), data['C'].max() + 1)
poisson_pmf = stats.poisson.pmf(x_range, mean_C)
ax.plot(x_range, poisson_pmf, 'ro-', linewidth=2, markersize=4,
        label=f'Poisson(λ={mean_C:.1f})', alpha=0.7)

ax.set_xlabel('Count (C)')
ax.set_ylabel('Density / Probability')
ax.set_title('Distribution vs Poisson Model')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# 3. Q-Q plot against Poisson
ax = axes[1, 0]
# Generate theoretical quantiles from Poisson
sorted_data = np.sort(data['C'])
theoretical_quantiles = stats.poisson.ppf(np.linspace(0.01, 0.99, len(sorted_data)), mean_C)

ax.scatter(theoretical_quantiles, sorted_data, alpha=0.6, s=50, color='purple')
ax.plot([min(theoretical_quantiles), max(theoretical_quantiles)],
        [min(theoretical_quantiles), max(theoretical_quantiles)],
        'r--', linewidth=2)
ax.set_xlabel('Theoretical Quantiles (Poisson)')
ax.set_ylabel('Sample Quantiles')
ax.set_title('Q-Q Plot: Observed vs Poisson')
ax.grid(alpha=0.3)

# 4. Variance vs Mean over moving window
ax = axes[1, 1]
window_size = 10
moving_means = []
moving_vars = []

for i in range(window_size, len(data) + 1):
    window_data = data.iloc[i-window_size:i]['C']
    moving_means.append(window_data.mean())
    moving_vars.append(window_data.var(ddof=1))

ax.scatter(moving_means, moving_vars, alpha=0.6, s=50, color='green',
           edgecolor='black', linewidth=0.5)

# Reference line
mean_range = np.linspace(min(moving_means), max(moving_means), 100)
ax.plot(mean_range, mean_range, 'r--', linewidth=2, label='Poisson (var=mean)')

ax.set_xlabel('Moving Window Mean')
ax.set_ylabel('Moving Window Variance')
ax.set_title(f'Mean-Variance (Rolling Window, n={window_size})')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/04_count_properties.png', dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("Saved: 04_count_properties.png")
print("="*80)
