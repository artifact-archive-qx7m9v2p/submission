"""
Generate Summary Tables for Eight Schools Analysis
==================================================
"""

import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('/workspace/eda/code/data_with_diagnostics.csv')

print("=" * 80)
print("EIGHT SCHOOLS: COMPREHENSIVE SUMMARY TABLES")
print("=" * 80)

# Calculate all necessary statistics
weights = 1 / (data['sigma'] ** 2)
weighted_mean = np.sum(data['y'] * weights) / np.sum(weights)

# Create comprehensive summary table
summary = pd.DataFrame({
    'School': data['school'].astype(int),
    'Effect (y)': data['y'],
    'SE (sigma)': data['sigma'],
    'Precision': data['precision'],
    'Weight (%)': 100 * weights / np.sum(weights),
    'Residual': data['y'] - weighted_mean,
    'Z-score': data['z_score'],
    '95% CI Lower': data['y'] - 2 * data['sigma'],
    '95% CI Upper': data['y'] + 2 * data['sigma']
})

print("\n1. COMPREHENSIVE SCHOOL SUMMARY")
print("-" * 80)
print(summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# Pooling comparison table
print("\n\n2. POOLING STRATEGY COMPARISON")
print("-" * 80)

pooling_comparison = pd.DataFrame({
    'Strategy': ['Complete Pooling', 'No Pooling (School 1)', 'No Pooling (School 5)',
                 'Partial Pooling (EB)'],
    'Estimate': [weighted_mean, data.iloc[0]['y'], data.iloc[4]['y'], weighted_mean],
    'SE': [np.sqrt(1/np.sum(weights)), data.iloc[0]['sigma'], data.iloc[4]['sigma'],
           np.sqrt(1/np.sum(weights))],
    'CI Width': [2*1.96*np.sqrt(1/np.sum(weights)), 2*1.96*data.iloc[0]['sigma'],
                 2*1.96*data.iloc[4]['sigma'], 2*1.96*np.sqrt(1/np.sum(weights))],
    'Interpretation': ['All schools share effect', 'School 1 only', 'School 5 only',
                      'Shrunk to pooled (tau²=0)']
})

print(pooling_comparison.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# Heterogeneity statistics
print("\n\n3. HETEROGENEITY STATISTICS")
print("-" * 80)

from scipy import stats

Q = np.sum(weights * (data['y'] - weighted_mean) ** 2)
df = len(data) - 1
p_value_Q = 1 - stats.chi2.cdf(Q, df)
I_squared = max(0, 100 * (Q - df) / Q)
C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
tau_squared = max(0, (Q - df) / C)

hetero_stats = pd.DataFrame({
    'Statistic': ['Cochran\'s Q', 'Degrees of Freedom', 'P-value', 'I² (%)',
                  'Tau² (between-school var)', 'Tau (between-school SD)',
                  'Observed variance', 'Avg sampling variance', 'Variance ratio'],
    'Value': [Q, df, p_value_Q, I_squared, tau_squared, np.sqrt(tau_squared),
              data['y'].var(), np.mean(data['sigma']**2),
              data['y'].var() / np.mean(data['sigma']**2)],
    'Interpretation': ['Test statistic', 'k - 1 = 7', 'Fail to reject homogeneity',
                      'Low heterogeneity', 'No between-school variance',
                      'No between-school variation',
                      'Total observed variation', 'Expected from sampling error',
                      '< 1 indicates homogeneity']
})

print(hetero_stats.to_string(index=False))

# Effect size categories
print("\n\n4. EFFECT SIZE DISTRIBUTION")
print("-" * 80)

effect_dist = pd.DataFrame({
    'Category': ['Negative', 'Small Positive (0-5)', 'Medium Positive (5-15)',
                 'Large Positive (>15)'],
    'Count': [
        len(data[data['y'] < 0]),
        len(data[(data['y'] >= 0) & (data['y'] < 5)]),
        len(data[(data['y'] >= 5) & (data['y'] < 15)]),
        len(data[data['y'] >= 15])
    ],
    'Schools': [
        ', '.join(map(str, data[data['y'] < 0]['school'].astype(int).tolist())),
        ', '.join(map(str, data[(data['y'] >= 0) & (data['y'] < 5)]['school'].astype(int).tolist())),
        ', '.join(map(str, data[(data['y'] >= 5) & (data['y'] < 15)]['school'].astype(int).tolist())),
        ', '.join(map(str, data[data['y'] >= 15]['school'].astype(int).tolist()))
    ]
})

print(effect_dist.to_string(index=False))

# Precision categories
print("\n\n5. MEASUREMENT PRECISION DISTRIBUTION")
print("-" * 80)

precision_dist = pd.DataFrame({
    'Category': ['High Precision (sigma < 10)', 'Medium Precision (10-15)',
                 'Low Precision (sigma > 15)'],
    'Count': [
        len(data[data['sigma'] < 10]),
        len(data[(data['sigma'] >= 10) & (data['sigma'] <= 15)]),
        len(data[data['sigma'] > 15])
    ],
    'Schools': [
        ', '.join(map(str, data[data['sigma'] < 10]['school'].astype(int).tolist())),
        ', '.join(map(str, data[(data['sigma'] >= 10) & (data['sigma'] <= 15)]['school'].astype(int).tolist())),
        ', '.join(map(str, data[data['sigma'] > 15]['school'].astype(int).tolist()))
    ],
    'Avg Effect': [
        data[data['sigma'] < 10]['y'].mean(),
        data[(data['sigma'] >= 10) & (data['sigma'] <= 15)]['y'].mean(),
        data[data['sigma'] > 15]['y'].mean()
    ]
})

print(precision_dist.to_string(index=False))

# Correlation summary
print("\n\n6. CORRELATION ANALYSIS")
print("-" * 80)

corr_pearson = data['y'].corr(data['sigma'])
corr_spearman = data['y'].corr(data['sigma'], method='spearman')
corr_test = stats.pearsonr(data['y'], data['sigma'])

corr_summary = pd.DataFrame({
    'Test': ['Pearson correlation', 'Spearman correlation'],
    'Coefficient': [corr_pearson, corr_spearman],
    'P-value': [corr_test.pvalue, stats.spearmanr(data['y'], data['sigma']).pvalue],
    'Interpretation': ['Weak positive, not significant', 'Weak positive, not significant']
})

print(corr_summary.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

# Model recommendation summary
print("\n\n7. MODEL RECOMMENDATION SUMMARY")
print("-" * 80)

model_rec = pd.DataFrame({
    'Model': ['Complete Pooling', 'No Pooling', 'Partial Pooling (Hierarchical)',
              'Mixture Model'],
    'Recommended': ['Alternative', 'No', 'PRIMARY', 'No'],
    'Evidence': ['Strong (Q test p=0.70)', 'None (wide CIs)',
                'Appropriate framework', 'None (no subgroups)'],
    'Expected Result': ['mu ~ N(7.7, 4.1)', 'Wide posteriors',
                       'tau ≈ 0, strong shrinkage', 'Overparameterized']
})

print(model_rec.to_string(index=False))

print("\n" + "=" * 80)
print("SUMMARY COMPLETE")
print("=" * 80)

# Save comprehensive table
summary.to_csv('/workspace/eda/school_summary_table.csv', index=False)
print("\nDetailed school summary saved to: /workspace/eda/school_summary_table.csv")
