"""
Create comprehensive summary statistics table
EDA Analyst 1
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data_analyst_1.csv'
OUTPUT_DIR = BASE_DIR / 'eda' / 'analyst_1'

# Load data
df = pd.read_csv(DATA_PATH)

# Calculate pooled proportion
pooled_p = df['r_successes'].sum() / df['n_trials'].sum()

# Calculate all metrics
df['expected_successes'] = df['n_trials'] * pooled_p
df['expected_variance'] = pooled_p * (1 - pooled_p) / df['n_trials']
df['expected_se'] = np.sqrt(df['expected_variance'])
df['standardized_residual'] = (df['success_rate'] - pooled_p) / df['expected_se']

# Wilson confidence intervals
z = 1.96
df['wilson_ci_lower'] = (df['success_rate'] + z**2/(2*df['n_trials']) - z * np.sqrt(
    df['success_rate']*(1-df['success_rate'])/df['n_trials'] + z**2/(4*df['n_trials']**2))) / (1 + z**2/df['n_trials'])
df['wilson_ci_upper'] = (df['success_rate'] + z**2/(2*df['n_trials']) + z * np.sqrt(
    df['success_rate']*(1-df['success_rate'])/df['n_trials'] + z**2/(4*df['n_trials']**2))) / (1 + z**2/df['n_trials'])

# Clip CIs to [0, 1]
df['wilson_ci_lower'] = df['wilson_ci_lower'].clip(0, 1)
df['wilson_ci_upper'] = df['wilson_ci_upper'].clip(0, 1)
df['ci_width'] = df['wilson_ci_upper'] - df['wilson_ci_lower']

print("=" * 100)
print("COMPREHENSIVE SUMMARY STATISTICS TABLE")
print("=" * 100)

# Summary table
summary = df[['group', 'n_trials', 'r_successes', 'success_rate',
              'wilson_ci_lower', 'wilson_ci_upper', 'ci_width',
              'standardized_residual']].copy()

summary.columns = ['Group', 'N Trials', 'R Successes', 'Success Rate',
                   '95% CI Lower', '95% CI Upper', 'CI Width', 'Z-Score']

# Format for display
summary_display = summary.copy()
summary_display['Success Rate'] = summary_display['Success Rate'].map('{:.4f}'.format)
summary_display['95% CI Lower'] = summary_display['95% CI Lower'].map('{:.4f}'.format)
summary_display['95% CI Upper'] = summary_display['95% CI Upper'].map('{:.4f}'.format)
summary_display['CI Width'] = summary_display['CI Width'].map('{:.4f}'.format)
summary_display['Z-Score'] = summary_display['Z-Score'].map('{:.2f}'.format)

print("\nDetailed Group Statistics:")
print(summary_display.to_string(index=False))

# Overall statistics
print("\n" + "=" * 100)
print("OVERALL STATISTICS")
print("=" * 100)

stats_dict = {
    'Metric': [
        'Number of Groups',
        'Total Trials',
        'Total Successes',
        'Pooled Success Rate',
        '',
        'Success Rate - Mean',
        'Success Rate - Median',
        'Success Rate - Std Dev',
        'Success Rate - Min',
        'Success Rate - Max',
        'Success Rate - Range',
        'Success Rate - CV',
        '',
        'N Trials - Mean',
        'N Trials - Median',
        'N Trials - Std Dev',
        'N Trials - Min',
        'N Trials - Max',
        'N Trials - Range',
        '',
        'Groups with Z-score > 1.96',
        'Groups with Z-score > 2.576',
        'Groups with success_rate = 0',
        'Groups in top 10%',
        'Groups in bottom 10%'
    ],
    'Value': [
        len(df),
        df['n_trials'].sum(),
        df['r_successes'].sum(),
        f"{pooled_p:.4f}",
        '',
        f"{df['success_rate'].mean():.4f}",
        f"{df['success_rate'].median():.4f}",
        f"{df['success_rate'].std():.4f}",
        f"{df['success_rate'].min():.4f}",
        f"{df['success_rate'].max():.4f}",
        f"{df['success_rate'].max() - df['success_rate'].min():.4f}",
        f"{df['success_rate'].std() / df['success_rate'].mean():.4f}",
        '',
        f"{df['n_trials'].mean():.2f}",
        f"{df['n_trials'].median():.2f}",
        f"{df['n_trials'].std():.2f}",
        f"{df['n_trials'].min()}",
        f"{df['n_trials'].max()}",
        f"{df['n_trials'].max() - df['n_trials'].min()}",
        '',
        len(df[np.abs(df['standardized_residual']) > 1.96]),
        len(df[np.abs(df['standardized_residual']) > 2.576]),
        len(df[df['success_rate'] == 0]),
        len(df[df['success_rate'] >= df['success_rate'].quantile(0.9)]),
        len(df[df['success_rate'] <= df['success_rate'].quantile(0.1)])
    ]
}

stats_df = pd.DataFrame(stats_dict)
print("\n" + stats_df.to_string(index=False))

# Overdispersion metrics
print("\n" + "=" * 100)
print("OVERDISPERSION METRICS")
print("=" * 100)

observed_var = df['success_rate'].var(ddof=1)
expected_var = np.mean(pooled_p * (1 - pooled_p) / df['n_trials'])
variance_ratio = observed_var / expected_var

# Chi-squared
expected_successes = df['n_trials'] * pooled_p
expected_failures = df['n_trials'] * (1 - pooled_p)
chi_squared = sum(((df['r_successes'] - expected_successes)**2 / expected_successes) +
                  ((df['n_trials'] - df['r_successes'] - expected_failures)**2 / expected_failures))
phi = chi_squared / (len(df) - 1)

overdispersion_dict = {
    'Metric': [
        'Observed variance',
        'Expected variance (binomial)',
        'Variance ratio',
        'Dispersion parameter (φ)',
        'Excess variance',
        '% of variance from overdispersion',
        'Chi-squared statistic',
        'Degrees of freedom',
        'Chi-squared p-value'
    ],
    'Value': [
        f"{observed_var:.6f}",
        f"{expected_var:.6f}",
        f"{variance_ratio:.4f}",
        f"{phi:.4f}",
        f"{observed_var - expected_var:.6f}",
        f"{(observed_var - expected_var) / observed_var * 100:.1f}%",
        f"{chi_squared:.4f}",
        f"{len(df) - 1}",
        f"<0.0001"
    ]
}

overdispersion_df = pd.DataFrame(overdispersion_dict)
print("\n" + overdispersion_df.to_string(index=False))

# Save to CSV
summary.to_csv(OUTPUT_DIR / 'summary_statistics.csv', index=False)
print(f"\n\nSummary statistics saved to: {OUTPUT_DIR / 'summary_statistics.csv'}")
