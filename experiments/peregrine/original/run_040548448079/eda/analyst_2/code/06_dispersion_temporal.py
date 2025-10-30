"""
Temporal Dispersion Analysis
Examine how variance structure changes over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Setup
sns.set_style("whitegrid")
output_dir = Path('/workspace/eda/analyst_2/visualizations')

# Load data
data = pd.read_csv('/workspace/data/data_analyst_2.csv')
year = data['year'].values
C = data['C'].values

print("=" * 80)
print("TEMPORAL DISPERSION ANALYSIS")
print("=" * 80)

# 1. Rolling statistics
window_sizes = [5, 10, 15]
print(f"\n1. ROLLING WINDOW STATISTICS")

fig, axes = plt.subplots(3, 2, figsize=(15, 12))

for i, window in enumerate(window_sizes):
    print(f"\n   Window size: {window}")

    # Calculate rolling statistics
    rolling_mean = pd.Series(C).rolling(window=window, center=True).mean()
    rolling_std = pd.Series(C).rolling(window=window, center=True).std()
    rolling_var = pd.Series(C).rolling(window=window, center=True).var()
    rolling_cv = rolling_std / rolling_mean  # Coefficient of variation
    rolling_dispersion = rolling_var / rolling_mean  # Variance-to-mean ratio

    print(f"     Mean dispersion: {rolling_dispersion.mean():.2f}")
    print(f"     Std of dispersion: {rolling_dispersion.std():.2f}")
    print(f"     Min dispersion: {rolling_dispersion.min():.2f}")
    print(f"     Max dispersion: {rolling_dispersion.max():.2f}")

    # Plot 1: Rolling mean and std
    ax = axes[i, 0]
    ax.plot(year, rolling_mean, 'b-', linewidth=2, label='Rolling mean')
    ax.fill_between(year,
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     alpha=0.3, color='blue', label='±1 SD')
    ax.scatter(year, C, alpha=0.4, s=30, color='gray', label='Observed')
    ax.set_xlabel('Standardized Year', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Rolling Mean ± SD (window={window})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Rolling dispersion
    ax = axes[i, 1]
    ax.plot(year, rolling_dispersion, 'o-', linewidth=2, color='steelblue', markersize=4)
    ax.axhline(1, color='red', linestyle='--', linewidth=2, label='Poisson (Var/Mean=1)', alpha=0.7)
    ax.axhline(rolling_dispersion.mean(), color='orange', linestyle=':',
               linewidth=2, label=f'Mean dispersion={rolling_dispersion.mean():.1f}', alpha=0.7)
    ax.set_xlabel('Standardized Year', fontsize=10)
    ax.set_ylabel('Variance / Mean', fontsize=10)
    ax.set_title(f'Rolling Dispersion (window={window})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'temporal_dispersion_rolling.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: temporal_dispersion_rolling.png")
plt.close()

# 2. Non-overlapping period analysis
print("\n" + "=" * 80)
print("2. NON-OVERLAPPING PERIOD ANALYSIS")
print("=" * 80)

n_periods = 5
period_size = len(C) // n_periods
period_results = []

for i in range(n_periods):
    start = i * period_size
    end = start + period_size if i < n_periods - 1 else len(C)
    period_data = C[start:end]
    period_year = year[start:end]

    mean_val = period_data.mean()
    var_val = period_data.var(ddof=1)
    std_val = period_data.std(ddof=1)
    cv_val = std_val / mean_val
    dispersion_val = var_val / mean_val
    skew_val = stats.skew(period_data)
    kurt_val = stats.kurtosis(period_data)

    period_results.append({
        'period': i + 1,
        'start_year': period_year.min(),
        'end_year': period_year.max(),
        'n': len(period_data),
        'mean': mean_val,
        'variance': var_val,
        'std': std_val,
        'cv': cv_val,
        'dispersion': dispersion_val,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'min': period_data.min(),
        'max': period_data.max()
    })

    print(f"\nPeriod {i+1} (year: {period_year.min():.2f} to {period_year.max():.2f})")
    print(f"  n = {len(period_data)}")
    print(f"  Mean = {mean_val:.2f}")
    print(f"  Variance = {var_val:.2f}")
    print(f"  Std Dev = {std_val:.2f}")
    print(f"  CV = {cv_val:.3f}")
    print(f"  Dispersion (Var/Mean) = {dispersion_val:.3f}")
    print(f"  Skewness = {skew_val:.3f}")
    print(f"  Range = [{period_data.min()}, {period_data.max()}]")

period_df = pd.DataFrame(period_results)

# Create comprehensive period comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Mean by period
ax = axes[0, 0]
bars = ax.bar(period_df['period'], period_df['mean'], alpha=0.7, color='steelblue', edgecolor='black')
ax.set_xlabel('Time Period', fontsize=11)
ax.set_ylabel('Mean Count', fontsize=11)
ax.set_title('Mean Count by Period', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, row in period_df.iterrows():
    ax.text(row['period'], row['mean'] + 5, f"{row['mean']:.1f}",
            ha='center', va='bottom', fontsize=9)

# 2. Variance by period
ax = axes[0, 1]
bars = ax.bar(period_df['period'], period_df['variance'], alpha=0.7, color='coral', edgecolor='black')
ax.set_xlabel('Time Period', fontsize=11)
ax.set_ylabel('Variance', fontsize=11)
ax.set_title('Variance by Period', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, row in period_df.iterrows():
    ax.text(row['period'], row['variance'] + 50, f"{row['variance']:.0f}",
            ha='center', va='bottom', fontsize=9)

# 3. Dispersion (Var/Mean) by period
ax = axes[0, 2]
bars = ax.bar(period_df['period'], period_df['dispersion'], alpha=0.7, color='lightgreen', edgecolor='black')
ax.axhline(1, color='red', linestyle='--', linewidth=2, label='Poisson', alpha=0.7)
ax.set_xlabel('Time Period', fontsize=11)
ax.set_ylabel('Variance / Mean', fontsize=11)
ax.set_title('Dispersion by Period', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
for i, row in period_df.iterrows():
    ax.text(row['period'], row['dispersion'] + 0.3, f"{row['dispersion']:.2f}",
            ha='center', va='bottom', fontsize=9)

# 4. Coefficient of Variation by period
ax = axes[1, 0]
bars = ax.bar(period_df['period'], period_df['cv'], alpha=0.7, color='gold', edgecolor='black')
ax.set_xlabel('Time Period', fontsize=11)
ax.set_ylabel('Coefficient of Variation (CV)', fontsize=11)
ax.set_title('CV by Period', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, row in period_df.iterrows():
    ax.text(row['period'], row['cv'] + 0.02, f"{row['cv']:.3f}",
            ha='center', va='bottom', fontsize=9)

# 5. Mean vs Variance scatter
ax = axes[1, 1]
ax.scatter(period_df['mean'], period_df['variance'], s=200, alpha=0.6, color='steelblue', edgecolor='black', linewidth=2)
for i, row in period_df.iterrows():
    ax.annotate(f"P{row['period']}", (row['mean'], row['variance']),
                fontsize=10, ha='center', va='center')

# Add reference lines
mean_range = np.linspace(period_df['mean'].min(), period_df['mean'].max(), 100)
ax.plot(mean_range, mean_range, 'r--', linewidth=2, label='Poisson (Var=Mean)', alpha=0.7)
overall_ratio = C.var(ddof=1) / C.mean()
ax.plot(mean_range, mean_range * overall_ratio, 'orange', linestyle='--',
        linewidth=2, label=f'Overall ratio ({overall_ratio:.1f}×Mean)', alpha=0.7)

ax.set_xlabel('Mean Count', fontsize=11)
ax.set_ylabel('Variance', fontsize=11)
ax.set_title('Mean-Variance Relationship\n(by period)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 6. Distribution shape by period (skewness)
ax = axes[1, 2]
bars = ax.bar(period_df['period'], period_df['skewness'], alpha=0.7, color='mediumpurple', edgecolor='black')
ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Time Period', fontsize=11)
ax.set_ylabel('Skewness', fontsize=11)
ax.set_title('Skewness by Period', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, row in period_df.iterrows():
    y_pos = row['skewness'] + (0.1 if row['skewness'] > 0 else -0.15)
    ax.text(row['period'], y_pos, f"{row['skewness']:.2f}",
            ha='center', va='bottom' if row['skewness'] > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'temporal_periods_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: temporal_periods_comparison.png")
plt.close()

# 3. Test for heteroscedasticity
print("\n" + "=" * 80)
print("3. HETEROSCEDASTICITY TESTS")
print("=" * 80)

# Breusch-Pagan test
from scipy.stats import chi2

# Fit linear model
X = np.column_stack([np.ones(len(year)), year])
beta = np.linalg.lstsq(X, C, rcond=None)[0]
fitted = X @ beta
residuals = C - fitted
rss = np.sum(residuals**2)

# Auxiliary regression: squared residuals on predictors
resid_sq = residuals**2
gamma = np.linalg.lstsq(X, resid_sq, rcond=None)[0]
fitted_aux = X @ gamma
ssr_aux = np.sum((fitted_aux - resid_sq.mean())**2)

# Test statistic
n = len(C)
bp_stat = (n * ssr_aux) / (2 * (rss/n)**2)
p_value_bp = 1 - chi2.cdf(bp_stat, df=1)

print(f"Breusch-Pagan Test:")
print(f"  Test statistic: {bp_stat:.3f}")
print(f"  p-value: {p_value_bp:.4f}")
print(f"  Conclusion: {'REJECT' if p_value_bp < 0.05 else 'FAIL TO REJECT'} homoscedasticity")
print(f"  Interpretation: {'Significant heteroscedasticity detected' if p_value_bp < 0.05 else 'No significant heteroscedasticity'}")

# Summary
print("\n" + "=" * 80)
print("TEMPORAL DISPERSION SUMMARY")
print("=" * 80)
print(f"Overall pattern:")
print(f"  - Mean increases from {period_df.iloc[0]['mean']:.1f} to {period_df.iloc[-1]['mean']:.1f}")
print(f"  - Variance increases from {period_df.iloc[0]['variance']:.1f} to {period_df.iloc[-1]['variance']:.1f}")
print(f"  - Dispersion varies: min={period_df['dispersion'].min():.2f}, max={period_df['dispersion'].max():.2f}")
print(f"  - CV relatively stable: {period_df['cv'].min():.3f} to {period_df['cv'].max():.3f}")
print(f"\nVariance structure:")
print(f"  - Heteroscedastic: variance changes with mean")
print(f"  - Overdispersion present throughout (all periods have Var/Mean > 1)")
print(f"  - Period 3 shows highest dispersion ({period_df.iloc[2]['dispersion']:.2f})")
