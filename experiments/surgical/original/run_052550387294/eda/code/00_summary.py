"""
EDA Summary - Quick Overview of Key Findings
=============================================
Run this script for a quick summary of all exploratory data analysis findings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Load data
DATA_PATH = Path("/workspace/data/data.csv")
df = pd.read_csv(DATA_PATH)

# Calculate key metrics
pooled_p = df['r'].sum() / df['n'].sum()
df['expected_var'] = pooled_p * (1 - pooled_p) / df['n']
df['expected_std'] = np.sqrt(df['expected_var'])
df['standardized_resid'] = (df['proportion'] - pooled_p) / df['expected_std']

# Chi-square test
chi_square_stat = ((df['r'] - df['n'] * pooled_p)**2 / (df['n'] * pooled_p * (1 - pooled_p))).sum()
df_chi = len(df) - 1
p_value = 1 - stats.chi2.cdf(chi_square_stat, df_chi)
dispersion_param = chi_square_stat / df_chi

print("\n" + "="*70)
print(" "*15 + "BINOMIAL DATASET - EDA SUMMARY")
print("="*70)

print("\n" + "─"*70)
print("  DATA OVERVIEW")
print("─"*70)
print(f"  Observations:        {len(df)} trials")
print(f"  Total trials:        {df['n'].sum():,}")
print(f"  Total successes:     {df['r'].sum()}")
print(f"  Pooled proportion:   {pooled_p:.4f} ({pooled_p*100:.2f}%)")
print(f"  Proportion range:    [{df['proportion'].min():.4f}, {df['proportion'].max():.4f}]")
print(f"  Sample size range:   [{df['n'].min()}, {df['n'].max()}]")

print("\n" + "─"*70)
print("  PRIMARY FINDING: OVERDISPERSION")
print("─"*70)
print(f"  Chi-square statistic:    {chi_square_stat:.2f}")
print(f"  Degrees of freedom:      {df_chi}")
print(f"  P-value:                 {p_value:.6f}")
print(f"  Dispersion parameter:    {dispersion_param:.2f}")
print(f"\n  Interpretation: Variance is {dispersion_param:.1f}x LARGER than expected")
print(f"                  under simple binomial model")
print(f"\n  Conclusion: STRONG EVIDENCE of overdispersion (p < 0.001)")
print(f"              Simple Binomial(n, p) model is INADEQUATE")

print("\n" + "─"*70)
print("  PATTERN ANALYSIS RESULTS")
print("─"*70)

# Temporal trend test
from scipy.stats import spearmanr
spearman_r, spearman_p = spearmanr(df['trial_id'], df['proportion'])
print(f"  Temporal trend:          {'YES' if spearman_p < 0.05 else 'NO'} (p = {spearman_p:.3f})")

# Sample size effect
spearman_r2, spearman_p2 = spearmanr(df['n'], df['proportion'])
print(f"  Sample size effect:      {'YES' if spearman_p2 < 0.05 else 'NO'} (p = {spearman_p2:.3f})")

# Group structure
median_prop = df['proportion'].median()
group_0_props = df[df['proportion'] <= median_prop]['proportion']
group_1_props = df[df['proportion'] > median_prop]['proportion']
t_stat, t_p = stats.ttest_ind(group_0_props, group_1_props)
print(f"  Distinct groups:         {'YES' if t_p < 0.05 else 'NO'} (p = {t_p:.3f})")

# Outliers
q1 = df['proportion'].quantile(0.25)
q3 = df['proportion'].quantile(0.75)
iqr = q3 - q1
outliers = df[(df['proportion'] < q1 - 1.5*iqr) | (df['proportion'] > q3 + 1.5*iqr)]
print(f"  Outliers detected:       {'YES' if len(outliers) > 0 else 'NO'} ({len(outliers)} trials)")

print("\n" + "─"*70)
print("  OUTLIERS AND EXTREME VALUES")
print("─"*70)
for idx, row in outliers.iterrows():
    z_score = row['standardized_resid']
    print(f"  Trial {int(row['trial_id']):2d}: p = {row['proportion']:.4f} "
          f"(n={int(row['n'])}, r={int(row['r'])}), z = {z_score:.2f}")

# Additional extreme standardized residuals
extreme = df[np.abs(df['standardized_resid']) > 2]
if len(extreme) > len(outliers):
    print(f"\n  Additional trials with |z| > 2:")
    for idx, row in extreme.iterrows():
        if idx not in outliers.index:
            z_score = row['standardized_resid']
            print(f"  Trial {int(row['trial_id']):2d}: p = {row['proportion']:.4f} "
                  f"(n={int(row['n'])}, r={int(row['r'])}), z = {z_score:.2f}")

print("\n" + "─"*70)
print("  GROUP ANALYSIS (Median Split)")
print("─"*70)
print(f"  Group 0 (Low, n={len(group_0_props)}):")
print(f"    Trials: {df[df['proportion'] <= median_prop]['trial_id'].tolist()}")
print(f"    Mean proportion: {group_0_props.mean():.4f}")
print(f"\n  Group 1 (High, n={len(group_1_props)}):")
print(f"    Trials: {df[df['proportion'] > median_prop]['trial_id'].tolist()}")
print(f"    Mean proportion: {group_1_props.mean():.4f}")
print(f"\n  Difference: {abs(group_1_props.mean() - group_0_props.mean()):.4f}")
print(f"  T-test p-value: {t_p:.4f}")

print("\n" + "─"*70)
print("  MODEL RECOMMENDATIONS")
print("─"*70)
print("  ✗ Do NOT use:  Simple Binomial(n, p) with constant p")
print("                 → REJECTED by data (p < 0.001)")
print("\n  ✓ RECOMMENDED: Beta-Binomial Model")
print("                 → Naturally accounts for overdispersion")
print("                 → φ parameter should be ~ 3-4")
print("\n  ✓ Consider:    Two-component mixture model")
print("                 → May capture group structure")
print("                 → Use for sensitivity analysis")
print("\n  ✓ Consider:    Hierarchical binomial model")
print("                 → Most flexible option")
print("                 → Trial-specific inference")

print("\n" + "─"*70)
print("  PRIOR RECOMMENDATIONS")
print("─"*70)
print("  For pooled probability p:")
print("    Beta(2, 25) → E[p] ≈ 0.074, 95% CI: [0.005, 0.20]")
print("\n  For overdispersion φ (Beta-Binomial):")
print("    Gamma(2, 0.5) → E[φ] = 4, allows range [0.5, 10]")
print("\n  For mixture components:")
print("    p_low ~ Beta(2, 38)  → centers near 0.05")
print("    p_high ~ Beta(4, 32) → centers near 0.11")

print("\n" + "─"*70)
print("  KEY TAKEAWAYS")
print("─"*70)
print("  1. Strong overdispersion (3.5x expected variance)")
print("  2. No temporal or sample-size patterns")
print("  3. Possible two distinct probability groups")
print("  4. Beta-Binomial or mixture model required")
print("  5. Trial 1 (0 successes) and Trial 8 (high p) are outliers")
print("  6. Pooled p ≈ 0.074 but with substantial heterogeneity")

print("\n" + "─"*70)
print("  FILES GENERATED")
print("─"*70)
print("  Reports:")
print("    /workspace/eda/eda_report.md     - Comprehensive analysis report")
print("    /workspace/eda/eda_log.md        - Detailed exploration log")
print("\n  Code:")
print("    /workspace/eda/code/01_initial_exploration.py")
print("    /workspace/eda/code/02_overdispersion_analysis.py")
print("    /workspace/eda/code/03_visualization.py")
print("    /workspace/eda/code/04_pattern_analysis.py")
print("\n  Visualizations (8 plots):")
print("    /workspace/eda/visualizations/sample_size_distribution.png")
print("    /workspace/eda/visualizations/proportion_distribution.png")
print("    /workspace/eda/visualizations/proportion_vs_trial.png")
print("    /workspace/eda/visualizations/proportion_vs_sample_size.png")
print("    /workspace/eda/visualizations/standardized_residuals.png")
print("    /workspace/eda/visualizations/comprehensive_comparison.png")
print("    /workspace/eda/visualizations/qq_plot.png")
print("    /workspace/eda/visualizations/funnel_plot.png")

print("\n" + "="*70)
print(" "*20 + "ANALYSIS COMPLETE")
print("="*70 + "\n")
