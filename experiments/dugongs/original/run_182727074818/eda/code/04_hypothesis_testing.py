"""
Hypothesis Testing and Model Recommendations
=============================================
Purpose: Test specific hypotheses about the data and generate model recommendations
Author: EDA Specialist Agent
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Setup paths
OUTPUT_DIR = Path("/workspace/eda")
VIZ_DIR = OUTPUT_DIR / "visualizations"
DATA_PATH = OUTPUT_DIR / "cleaned_data.csv"

def test_hypothesis_1_saturation(df):
    """
    Hypothesis 1: Y exhibits saturation/asymptotic behavior with increasing x
    Test: Fit asymptotic model and compare incremental changes in Y
    """
    print("\n" + "="*80)
    print("HYPOTHESIS 1: Saturation/Asymptotic Behavior")
    print("="*80)

    x = df['x'].values
    y = df['Y'].values

    # Sort by x for analysis
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Compute incremental changes (rate of change)
    dx = np.diff(x_sorted)
    dy = np.diff(y_sorted)
    rate_of_change = dy / dx

    # Moving average of rate of change
    window = 5
    if len(rate_of_change) >= window:
        moving_avg = np.convolve(rate_of_change, np.ones(window)/window, mode='valid')
    else:
        moving_avg = rate_of_change

    print(f"\nRate of Change Analysis:")
    print(f"  First 5 increments mean rate: {rate_of_change[:5].mean():.6f}")
    print(f"  Last 5 increments mean rate: {rate_of_change[-5:].mean():.6f}")
    print(f"  Ratio (late/early): {rate_of_change[-5:].mean() / rate_of_change[:5].mean():.4f}")

    # Test for decreasing trend in rate of change
    x_mid = x_sorted[:-1] + dx/2
    correlation_rate_x = stats.spearmanr(x_mid, rate_of_change)
    print(f"\nSpearman correlation (rate of change vs x): r={correlation_rate_x[0]:.4f}, p={correlation_rate_x[1]:.4f}")

    if correlation_rate_x[0] < 0 and correlation_rate_x[1] < 0.05:
        print("  RESULT: Significant DECREASING rate of change - SUPPORTS saturation hypothesis")
        verdict = "SUPPORTED"
    else:
        print("  RESULT: No significant decrease in rate of change - WEAK SUPPORT")
        verdict = "WEAK SUPPORT"

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Data with asymptotic fit
    ax = axes[0]
    ax.scatter(df['x'], df['Y'], s=60, alpha=0.6, color='steelblue', edgecolors='black', label='Data')

    # Fit asymptotic model
    def asymptotic(x, ymax, k):
        return ymax * x / (k + x)

    try:
        popt, _ = curve_fit(asymptotic, x, y, p0=[3, 5], maxfev=10000)
        x_range = np.linspace(x.min(), x.max(), 200)
        y_pred = asymptotic(x_range, *popt)
        ax.plot(x_range, y_pred, 'r-', linewidth=2, label=f'Asymptotic fit: Ymax={popt[0]:.3f}, K={popt[1]:.3f}')
        ax.axhline(y=popt[0], color='g', linestyle='--', alpha=0.5, label=f'Asymptote: {popt[0]:.3f}')
    except:
        pass

    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_title('Saturation Hypothesis: Asymptotic Behavior', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Rate of change
    ax = axes[1]
    ax.scatter(x_mid, rate_of_change, s=40, alpha=0.6, color='steelblue', edgecolors='black', label='dY/dx')
    if len(moving_avg) > 0:
        ax.plot(x_mid[window//2:len(moving_avg)+window//2], moving_avg, 'r-', linewidth=2, label='Moving average')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Rate of Change (dY/dx)', fontsize=11)
    ax.set_title('Rate of Change Analysis', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'hypothesis1_saturation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return verdict

def test_hypothesis_2_log_relationship(df):
    """
    Hypothesis 2: Relationship is logarithmic (Y ~ log(x))
    Test: Compare linear fit on log-transformed x vs raw x
    """
    print("\n" + "="*80)
    print("HYPOTHESIS 2: Logarithmic Relationship")
    print("="*80)

    x = df['x'].values
    y = df['Y'].values

    # Fit linear model to raw x
    corr_linear = stats.pearsonr(x, y)[0]

    # Fit linear model to log(x)
    log_x = np.log(x + 1)
    corr_log = stats.pearsonr(log_x, y)[0]

    print(f"\nCorrelation Analysis:")
    print(f"  Pearson r (Y vs x): {corr_linear:.4f}")
    print(f"  Pearson r (Y vs log(x+1)): {corr_log:.4f}")
    print(f"  Improvement with log transform: {corr_log - corr_linear:.4f}")

    # R² comparison
    def compute_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        return 1 - (ss_res / ss_tot)

    # Linear fit
    A_linear = np.vstack([x, np.ones(len(x))]).T
    slope_linear, intercept_linear = np.linalg.lstsq(A_linear, y, rcond=None)[0]
    y_pred_linear = slope_linear * x + intercept_linear
    r2_linear = compute_r2(y, y_pred_linear)

    # Log fit
    A_log = np.vstack([log_x, np.ones(len(log_x))]).T
    slope_log, intercept_log = np.linalg.lstsq(A_log, y, rcond=None)[0]
    y_pred_log = slope_log * log_x + intercept_log
    r2_log = compute_r2(y, y_pred_log)

    print(f"\nR² Comparison:")
    print(f"  R² (linear): {r2_linear:.4f}")
    print(f"  R² (log): {r2_log:.4f}")
    print(f"  Improvement: {r2_log - r2_linear:.4f}")

    if r2_log > r2_linear + 0.05:
        print("  RESULT: Log transformation substantially improves fit - SUPPORTS log hypothesis")
        verdict = "SUPPORTED"
    elif r2_log > r2_linear:
        print("  RESULT: Log transformation slightly improves fit - WEAK SUPPORT")
        verdict = "WEAK SUPPORT"
    else:
        print("  RESULT: Log transformation does not improve fit - NOT SUPPORTED")
        verdict = "NOT SUPPORTED"

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Y vs x (linear scale)
    ax = axes[0]
    ax.scatter(x, y, s=60, alpha=0.6, color='steelblue', edgecolors='black', label='Data')
    x_sorted = np.sort(x)
    ax.plot(x_sorted, slope_linear * x_sorted + intercept_linear, 'r--', linewidth=2,
            label=f'Linear fit (R²={r2_linear:.4f})')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_title('Linear Scale', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Y vs log(x)
    ax = axes[1]
    ax.scatter(log_x, y, s=60, alpha=0.6, color='steelblue', edgecolors='black', label='Data')
    log_x_sorted = np.sort(log_x)
    ax.plot(log_x_sorted, slope_log * log_x_sorted + intercept_log, 'r--', linewidth=2,
            label=f'Linear fit (R²={r2_log:.4f})')
    ax.set_xlabel('log(x+1)', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_title('Log-Transformed Scale', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'hypothesis2_logarithmic.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return verdict

def test_hypothesis_3_constant_variance(df):
    """
    Hypothesis 3: Homoscedasticity (constant variance)
    Test: Breusch-Pagan test and visual inspection
    """
    print("\n" + "="*80)
    print("HYPOTHESIS 3: Homoscedasticity (Constant Variance)")
    print("="*80)

    x = df['x'].values
    y = df['Y'].values

    # Fit linear model
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = slope * x + intercept
    residuals = y - y_pred

    # Squared residuals
    residuals_sq = residuals ** 2

    # Test if squared residuals correlate with x (Breusch-Pagan-like)
    corr_resid_x = stats.spearmanr(x, np.abs(residuals))

    print(f"\nVariance Analysis:")
    print(f"  Correlation (|residuals| vs x): r={corr_resid_x[0]:.4f}, p={corr_resid_x[1]:.4f}")

    # Levene's test on binned groups
    bins = pd.cut(x, bins=3)
    groups = [residuals[bins == cat] for cat in bins.categories if (bins == cat).sum() > 0]
    if len(groups) >= 2:
        levene_stat, levene_p = stats.levene(*groups)
        print(f"  Levene's test: statistic={levene_stat:.4f}, p={levene_p:.4f}")
    else:
        levene_p = None

    if corr_resid_x[1] > 0.05 and (levene_p is None or levene_p > 0.05):
        print("  RESULT: No evidence of heteroscedasticity - SUPPORTS constant variance")
        verdict = "SUPPORTED"
    elif corr_resid_x[1] > 0.05 or (levene_p and levene_p > 0.05):
        print("  RESULT: Mixed evidence - WEAK SUPPORT for constant variance")
        verdict = "WEAK SUPPORT"
    else:
        print("  RESULT: Evidence of heteroscedasticity - NOT SUPPORTED")
        verdict = "NOT SUPPORTED"

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Residuals vs x
    ax = axes[0]
    ax.scatter(x, residuals, s=60, alpha=0.6, color='steelblue', edgecolors='black')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title('Residuals vs Predictor', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Absolute residuals vs x
    ax = axes[1]
    ax.scatter(x, np.abs(residuals), s=60, alpha=0.6, color='steelblue', edgecolors='black')
    # Add trend line
    z = np.polyfit(x, np.abs(residuals), 1)
    p = np.poly1d(z)
    x_sorted = np.sort(x)
    ax.plot(x_sorted, p(x_sorted), 'r-', linewidth=2, label=f'Trend: slope={z[0]:.6f}')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('|Residuals|', fontsize=11)
    ax.set_title('Absolute Residuals vs Predictor', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'hypothesis3_homoscedasticity.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return verdict

def test_hypothesis_4_change_point(df):
    """
    Hypothesis 4: Relationship changes at a specific x value (change point)
    Test: Test for structural break using residuals
    """
    print("\n" + "="*80)
    print("HYPOTHESIS 4: Change Point / Structural Break")
    print("="*80)

    x = df['x'].values
    y = df['Y'].values

    # Fit global linear model
    A = np.vstack([x, np.ones(len(x))]).T
    slope_global, intercept_global = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred_global = slope_global * x + intercept_global
    rss_global = np.sum((y - y_pred_global)**2)

    print(f"\nGlobal model RSS: {rss_global:.6f}")

    # Test different breakpoints
    x_sorted_unique = np.sort(np.unique(x))
    potential_breaks = x_sorted_unique[len(x_sorted_unique)//4:3*len(x_sorted_unique)//4]

    best_break = None
    best_rss_sum = rss_global
    rss_improvements = []

    for break_point in potential_breaks:
        # Split data
        mask_low = x <= break_point
        mask_high = x > break_point

        if mask_low.sum() < 3 or mask_high.sum() < 3:
            continue

        # Fit two separate models
        A_low = np.vstack([x[mask_low], np.ones(mask_low.sum())]).T
        slope_low, intercept_low = np.linalg.lstsq(A_low, y[mask_low], rcond=None)[0]
        y_pred_low = slope_low * x[mask_low] + intercept_low
        rss_low = np.sum((y[mask_low] - y_pred_low)**2)

        A_high = np.vstack([x[mask_high], np.ones(mask_high.sum())]).T
        slope_high, intercept_high = np.linalg.lstsq(A_high, y[mask_high], rcond=None)[0]
        y_pred_high = slope_high * x[mask_high] + intercept_high
        rss_high = np.sum((y[mask_high] - y_pred_high)**2)

        rss_sum = rss_low + rss_high
        improvement = rss_global - rss_sum
        rss_improvements.append((break_point, improvement, rss_sum))

        if rss_sum < best_rss_sum:
            best_rss_sum = rss_sum
            best_break = break_point

    if rss_improvements:
        rss_improvements.sort(key=lambda x: x[1], reverse=True)
        print(f"\nBest breakpoint: x = {rss_improvements[0][0]:.2f}")
        print(f"  RSS improvement: {rss_improvements[0][1]:.6f}")
        print(f"  Relative improvement: {(rss_improvements[0][1] / rss_global) * 100:.2f}%")

        if (rss_improvements[0][1] / rss_global) > 0.15:
            print("  RESULT: Substantial improvement with breakpoint - SUPPORTS change point hypothesis")
            verdict = "SUPPORTED"
        elif (rss_improvements[0][1] / rss_global) > 0.05:
            print("  RESULT: Modest improvement with breakpoint - WEAK SUPPORT")
            verdict = "WEAK SUPPORT"
        else:
            print("  RESULT: Minimal improvement with breakpoint - NOT SUPPORTED")
            verdict = "NOT SUPPORTED"
    else:
        print("  RESULT: Could not test breakpoints - INCONCLUSIVE")
        verdict = "INCONCLUSIVE"

    return verdict

def test_hypothesis_5_replication_variance(df):
    """
    Hypothesis 5: Replicates (same x values) have consistent Y variance
    Test: Compare variance across replicated x values
    """
    print("\n" + "="*80)
    print("HYPOTHESIS 5: Consistent Measurement Error Across Replicates")
    print("="*80)

    # Find x values with replicates
    x_counts = df['x'].value_counts()
    replicates = x_counts[x_counts > 1]

    if len(replicates) == 0:
        print("\nNo replicated x values found.")
        return "INCONCLUSIVE"

    print(f"\nFound {len(replicates)} x values with replicates:")

    variances = []
    for x_val in replicates.index:
        y_vals = df[df['x'] == x_val]['Y'].values
        var = np.var(y_vals, ddof=1) if len(y_vals) > 1 else 0
        variances.append(var)
        print(f"  x={x_val:.1f}: n={len(y_vals)}, Y values={y_vals}, variance={var:.6f}")

    print(f"\nVariance statistics:")
    print(f"  Mean variance: {np.mean(variances):.6f}")
    print(f"  Std of variances: {np.std(variances):.6f}")
    print(f"  Range: [{np.min(variances):.6f}, {np.max(variances):.6f}]")

    # Coefficient of variation of variances
    cv_var = np.std(variances) / np.mean(variances) if np.mean(variances) > 0 else 0
    print(f"  CV of variances: {cv_var:.4f}")

    if cv_var < 0.5:
        print("  RESULT: Low variation in replicate variances - SUPPORTS consistent error")
        verdict = "SUPPORTED"
    elif cv_var < 1.0:
        print("  RESULT: Moderate variation in replicate variances - WEAK SUPPORT")
        verdict = "WEAK SUPPORT"
    else:
        print("  RESULT: High variation in replicate variances - NOT SUPPORTED")
        verdict = "NOT SUPPORTED"

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = replicates.index
    ax.bar(range(len(variances)), variances, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(variances)))
    ax.set_xticklabels([f'{x:.1f}' for x in x_vals], rotation=45)
    ax.set_xlabel('x value', fontsize=11)
    ax.set_ylabel('Variance of Y', fontsize=11)
    ax.set_title('Variance Across Replicated x Values', fontsize=12, fontweight='bold')
    ax.axhline(y=np.mean(variances), color='r', linestyle='--', linewidth=2, label=f'Mean variance: {np.mean(variances):.6f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(VIZ_DIR / 'hypothesis5_replicate_variance.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return verdict

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS - PHASE 4: HYPOTHESIS TESTING")
    print("="*80)

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded data: {df.shape}")

    # Test hypotheses
    results = {}
    results['H1_Saturation'] = test_hypothesis_1_saturation(df)
    results['H2_Logarithmic'] = test_hypothesis_2_log_relationship(df)
    results['H3_Homoscedasticity'] = test_hypothesis_3_constant_variance(df)
    results['H4_ChangePoint'] = test_hypothesis_4_change_point(df)
    results['H5_ReplicateVariance'] = test_hypothesis_5_replication_variance(df)

    # Summary
    print("\n" + "="*80)
    print("HYPOTHESIS TESTING SUMMARY")
    print("="*80)
    for hypothesis, verdict in results.items():
        print(f"{hypothesis}: {verdict}")

    print("\n" + "="*80)
    print("PHASE 4 COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
