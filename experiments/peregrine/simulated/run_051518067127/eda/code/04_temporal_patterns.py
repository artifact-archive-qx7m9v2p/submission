"""
Temporal Pattern Analysis
=========================
Goal: Analyze autocorrelation, growth rates, and temporal structure
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

# Add time index
data['time_idx'] = np.arange(len(data))

# Calculate growth rates and changes
data['C_diff'] = data['C'].diff()
data['C_pct_change'] = data['C'].pct_change() * 100
data['log_C'] = np.log(data['C'])
data['log_C_diff'] = data['log_C'].diff()

# Define time periods
n = len(data)
data['period'] = pd.cut(data['time_idx'], bins=3, labels=['Early', 'Middle', 'Late'])

print("="*80)
print("TEMPORAL PATTERN ANALYSIS")
print("="*80)

# 1. Time period statistics
print("\n1. STATISTICS BY TIME PERIOD")
print("-" * 80)
period_stats = data.groupby('period')['C'].agg([
    'count', 'mean', 'std', 'min', 'max',
    ('CV', lambda x: x.std() / x.mean())
])
print(period_stats)

print("\nMean comparison across periods:")
for period in ['Early', 'Middle', 'Late']:
    period_data = data[data['period'] == period]['C']
    print(f"  {period}: mean={period_data.mean():.2f}, std={period_data.std():.2f}")

# ANOVA test
early = data[data['period'] == 'Early']['C']
middle = data[data['period'] == 'Middle']['C']
late = data[data['period'] == 'Late']['C']
f_stat, p_val = stats.f_oneway(early, middle, late)
print(f"\nOne-way ANOVA: F={f_stat:.4f}, p={p_val:.4e}")
print(f"  Conclusion: {'Significant differences' if p_val < 0.05 else 'No significant differences'} across periods")

# 2. Growth rate analysis
print("\n2. GROWTH RATE ANALYSIS")
print("-" * 80)
print(f"Absolute changes (C_diff):")
print(f"  Mean: {data['C_diff'].mean():.4f}")
print(f"  Std: {data['C_diff'].std():.4f}")
print(f"  Min: {data['C_diff'].min():.4f}")
print(f"  Max: {data['C_diff'].max():.4f}")

print(f"\nPercentage changes:")
print(f"  Mean: {data['C_pct_change'].mean():.4f}%")
print(f"  Std: {data['C_pct_change'].std():.4f}%")
print(f"  Median: {data['C_pct_change'].median():.4f}%")

print(f"\nLog differences (approximates growth rate):")
print(f"  Mean: {data['log_C_diff'].mean():.4f}")
print(f"  Std: {data['log_C_diff'].std():.4f}")

# 3. Autocorrelation analysis
print("\n3. AUTOCORRELATION ANALYSIS")
print("-" * 80)

# Calculate ACF manually for raw counts
max_lag = 10
acf_values = []
for lag in range(max_lag + 1):
    if lag == 0:
        acf_values.append(1.0)
    else:
        corr = data['C'].autocorr(lag=lag)
        acf_values.append(corr)
        if lag <= 5:
            print(f"  Lag {lag}: {corr:.4f}")

# Autocorrelation of residuals
from scipy import stats as sp_stats
slope, intercept, _, _, _ = sp_stats.linregress(data['year'], data['C'])
residuals = data['C'] - (slope * data['year'] + intercept)

print(f"\nAutocorrelation of linear model residuals:")
for lag in range(1, 6):
    resid_series = pd.Series(residuals.values)
    acf_resid = resid_series.autocorr(lag=lag)
    print(f"  Lag {lag}: {acf_resid:.4f}")

# Durbin-Watson test
def durbin_watson(residuals):
    """Calculate Durbin-Watson statistic"""
    diff_resid = np.diff(residuals)
    dw = np.sum(diff_resid**2) / np.sum(residuals**2)
    return dw

dw_stat = durbin_watson(residuals.values)
print(f"\nDurbin-Watson statistic: {dw_stat:.4f}")
print(f"  Interpretation: ", end="")
if dw_stat < 1.5:
    print("Positive autocorrelation")
elif dw_stat > 2.5:
    print("Negative autocorrelation")
else:
    print("No strong autocorrelation")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Temporal Pattern Analysis', fontsize=16, y=1.00)

# 1. Time series plot
ax = axes[0, 0]
ax.plot(data['time_idx'], data['C'], 'o-', color='steelblue', linewidth=1.5, markersize=5)
ax.axhline(data['C'].mean(), color='red', linestyle='--', alpha=0.7, label='Overall mean')
# Add period means
for i, period in enumerate(['Early', 'Middle', 'Late']):
    period_data = data[data['period'] == period]
    ax.axhline(period_data['C'].mean(),
               xmin=i/3, xmax=(i+1)/3,
               color='orange', linestyle='--', alpha=0.7, linewidth=2)
ax.set_xlabel('Time Index')
ax.set_ylabel('Count (C)')
ax.set_title('Time Series of Counts')
ax.legend()
ax.grid(alpha=0.3)

# 2. Growth rates over time
ax = axes[0, 1]
ax.plot(data['time_idx'][1:], data['C_diff'][1:], 'o-', color='green', markersize=4)
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.axhline(data['C_diff'].mean(), color='orange', linestyle='--', linewidth=1,
           label=f'Mean: {data["C_diff"].mean():.2f}')
ax.set_xlabel('Time Index')
ax.set_ylabel('Change in Count')
ax.set_title('Absolute Changes Over Time')
ax.legend()
ax.grid(alpha=0.3)

# 3. Percentage changes
ax = axes[0, 2]
ax.plot(data['time_idx'][1:], data['C_pct_change'][1:], 'o-', color='purple', markersize=4)
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.axhline(data['C_pct_change'].mean(), color='orange', linestyle='--', linewidth=1,
           label=f'Mean: {data["C_pct_change"].mean():.2f}%')
ax.set_xlabel('Time Index')
ax.set_ylabel('Percentage Change (%)')
ax.set_title('Percentage Changes Over Time')
ax.legend()
ax.grid(alpha=0.3)

# 4. ACF plot
ax = axes[1, 0]
lags = np.arange(len(acf_values))
ax.bar(lags, acf_values, width=0.3, color='steelblue', alpha=0.7)
# Add confidence bands (95%)
conf_level = 1.96 / np.sqrt(len(data))
ax.axhline(conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(-conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.set_title('Autocorrelation Function (ACF)')
ax.set_xlim(-0.5, max_lag + 0.5)
ax.grid(alpha=0.3, axis='y')

# 5. Box plots by period
ax = axes[1, 1]
period_data_list = [early.values, middle.values, late.values]
bp = ax.boxplot(period_data_list, labels=['Early', 'Middle', 'Late'],
                patch_artist=True, widths=0.5)
for i, box in enumerate(bp['boxes']):
    box.set_facecolor(['lightblue', 'lightgreen', 'lightcoral'][i])
ax.set_ylabel('Count (C)')
ax.set_title('Distribution by Time Period')
ax.grid(alpha=0.3, axis='y')

# 6. Residual ACF
ax = axes[1, 2]
acf_resid_values = []
for lag in range(max_lag + 1):
    if lag == 0:
        acf_resid_values.append(1.0)
    else:
        resid_series = pd.Series(residuals.values)
        acf_resid_values.append(resid_series.autocorr(lag=lag))

ax.bar(lags, acf_resid_values, width=0.3, color='purple', alpha=0.7)
ax.axhline(conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(-conf_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.set_title('ACF of Linear Model Residuals')
ax.set_xlim(-0.5, max_lag + 0.5)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspace/eda/visualizations/03_temporal_patterns.png', dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("Saved: 03_temporal_patterns.png")
print("="*80)
plt.close()

# Additional analysis: changepoint detection
print("\n4. CHANGEPOINT DETECTION (Informal)")
print("-" * 80)
# Split data in half and test for difference
n_half = len(data) // 2
first_half = data.iloc[:n_half]['C']
second_half = data.iloc[n_half:]['C']

t_stat, p_val = stats.ttest_ind(first_half, second_half)
print(f"Two-sample t-test (first half vs second half):")
print(f"  First half mean: {first_half.mean():.2f}")
print(f"  Second half mean: {second_half.mean():.2f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_val:.4e}")
print(f"  Conclusion: {'Significant difference' if p_val < 0.05 else 'No significant difference'}")
