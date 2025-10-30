"""
Bivariate Analysis and Relationship Exploration
================================================
Purpose: Analyze relationship between x and Y, test functional forms
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

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

def linear_fit(x, a, b):
    """Linear function."""
    return a * x + b

def quadratic_fit(x, a, b, c):
    """Quadratic function."""
    return a * x**2 + b * x + c

def cubic_fit(x, a, b, c, d):
    """Cubic function."""
    return a * x**3 + b * x**2 + c * x + d

def log_fit(x, a, b):
    """Logarithmic function."""
    return a * np.log(x + 1) + b

def sqrt_fit(x, a, b):
    """Square root function."""
    return a * np.sqrt(x) + b

def asymptotic_fit(x, a, b):
    """Asymptotic function: y = a * x / (b + x)"""
    return a * x / (b + x)

def compute_r2(y_true, y_pred):
    """Compute R-squared."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - (ss_res / ss_tot)

def create_basic_scatterplot(df):
    """
    Create basic scatterplot with trend line.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot
    ax.scatter(df['x'], df['Y'], s=80, alpha=0.6, color='steelblue', edgecolors='black', linewidth=1)

    # Linear regression using numpy
    x = df['x'].values
    y = df['Y'].values
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # Plot regression line
    x_sorted = np.sort(x)
    y_pred = slope * x_sorted + intercept
    ax.plot(x_sorted, y_pred, 'r--', linewidth=2, label='Linear fit')

    # Calculate R²
    y_pred_all = slope * x + intercept
    r2 = compute_r2(y, y_pred_all)
    corr = np.corrcoef(x, y)[0, 1]

    # Add text
    text_str = f'R² = {r2:.4f}\nCorrelation = {corr:.4f}\nSlope = {slope:.4f}\nIntercept = {intercept:.4f}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('x (Predictor)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (Response)', fontsize=12, fontweight='bold')
    ax.set_title('Scatterplot: Y vs x with Linear Fit', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_functional_forms_comparison(df):
    """
    Compare different functional forms.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Functional Form Exploration: Y vs x', fontsize=16, fontweight='bold')

    x = df['x'].values
    y = df['Y'].values
    x_range = np.linspace(x.min(), x.max(), 200)

    r2_scores = {}

    # 1. Linear
    ax = axes[0, 0]
    ax.scatter(x, y, s=50, alpha=0.6, color='steelblue', edgecolors='black')
    try:
        popt, _ = curve_fit(linear_fit, x, y)
        y_pred = linear_fit(x_range, *popt)
        ax.plot(x_range, y_pred, 'r-', linewidth=2, label='Linear')
        r2 = compute_r2(y, linear_fit(x, *popt))
        r2_scores['linear_r2'] = r2
        ax.set_title(f'Linear (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    except:
        ax.set_title('Linear (fit failed)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Quadratic
    ax = axes[0, 1]
    ax.scatter(x, y, s=50, alpha=0.6, color='steelblue', edgecolors='black')
    try:
        popt, _ = curve_fit(quadratic_fit, x, y)
        y_pred = quadratic_fit(x_range, *popt)
        ax.plot(x_range, y_pred, 'g-', linewidth=2, label='Quadratic')
        r2 = compute_r2(y, quadratic_fit(x, *popt))
        r2_scores['quadratic_r2'] = r2
        ax.set_title(f'Quadratic (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    except:
        ax.set_title('Quadratic (fit failed)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Cubic
    ax = axes[0, 2]
    ax.scatter(x, y, s=50, alpha=0.6, color='steelblue', edgecolors='black')
    try:
        popt, _ = curve_fit(cubic_fit, x, y)
        y_pred = cubic_fit(x_range, *popt)
        ax.plot(x_range, y_pred, 'm-', linewidth=2, label='Cubic')
        r2 = compute_r2(y, cubic_fit(x, *popt))
        r2_scores['cubic_r2'] = r2
        ax.set_title(f'Cubic (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    except:
        ax.set_title('Cubic (fit failed)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Logarithmic
    ax = axes[1, 0]
    ax.scatter(x, y, s=50, alpha=0.6, color='steelblue', edgecolors='black')
    try:
        popt, _ = curve_fit(log_fit, x, y)
        y_pred = log_fit(x_range, *popt)
        ax.plot(x_range, y_pred, 'orange', linewidth=2, label='Logarithmic')
        r2 = compute_r2(y, log_fit(x, *popt))
        r2_scores['log_r2'] = r2
        ax.set_title(f'Logarithmic (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    except:
        ax.set_title('Logarithmic (fit failed)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Square Root
    ax = axes[1, 1]
    ax.scatter(x, y, s=50, alpha=0.6, color='steelblue', edgecolors='black')
    try:
        popt, _ = curve_fit(sqrt_fit, x, y)
        y_pred = sqrt_fit(x_range, *popt)
        ax.plot(x_range, y_pred, 'cyan', linewidth=2, label='Square Root')
        r2 = compute_r2(y, sqrt_fit(x, *popt))
        r2_scores['sqrt_r2'] = r2
        ax.set_title(f'Square Root (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    except:
        ax.set_title('Square Root (fit failed)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Asymptotic (Michaelis-Menten-like)
    ax = axes[1, 2]
    ax.scatter(x, y, s=50, alpha=0.6, color='steelblue', edgecolors='black')
    try:
        popt, _ = curve_fit(asymptotic_fit, x, y, p0=[3, 5], maxfev=10000)
        y_pred = asymptotic_fit(x_range, *popt)
        ax.plot(x_range, y_pred, 'brown', linewidth=2, label='Asymptotic')
        r2 = compute_r2(y, asymptotic_fit(x, *popt))
        r2_scores['asymptotic_r2'] = r2
        ax.set_title(f'Asymptotic/Saturation (R² = {r2:.4f})', fontsize=12, fontweight='bold')
    except Exception as e:
        ax.set_title('Asymptotic (fit failed)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, r2_scores

def create_residual_analysis(df):
    """
    Create comprehensive residual analysis for linear model.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Residual Analysis: Linear Model', fontsize=16, fontweight='bold')

    # Fit linear model
    x = df['x'].values
    y = df['Y'].values
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = slope * x + intercept
    residuals = y - y_pred

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, s=60, alpha=0.6, color='steelblue', edgecolors='black')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    # Add LOWESS smooth
    from scipy.signal import savgol_filter
    sorted_indices = np.argsort(y_pred)
    if len(y_pred) > 5:
        try:
            smoothed = savgol_filter(residuals[sorted_indices], window_length=min(11, len(y_pred)//2*2+1), polyorder=2)
            ax.plot(y_pred[sorted_indices], smoothed, 'g-', linewidth=2, label='Smooth trend')
        except:
            pass
    ax.set_xlabel('Fitted Values', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title('Residuals vs Fitted', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Q-Q Plot of residuals
    ax = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Normal Q-Q Plot of Residuals', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 3. Scale-Location (sqrt of standardized residuals vs fitted)
    ax = axes[1, 0]
    standardized_residuals = residuals / residuals.std()
    sqrt_abs_std_residuals = np.sqrt(np.abs(standardized_residuals))
    ax.scatter(y_pred, sqrt_abs_std_residuals, s=60, alpha=0.6, color='steelblue', edgecolors='black')
    if len(y_pred) > 5:
        try:
            smoothed = savgol_filter(sqrt_abs_std_residuals[sorted_indices],
                                    window_length=min(11, len(y_pred)//2*2+1), polyorder=2)
            ax.plot(y_pred[sorted_indices], smoothed, 'r-', linewidth=2, label='Smooth trend')
        except:
            pass
    ax.set_xlabel('Fitted Values', fontsize=11)
    ax.set_ylabel('√|Standardized Residuals|', fontsize=11)
    ax.set_title('Scale-Location Plot', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Residuals vs Predictor (x)
    ax = axes[1, 1]
    ax.scatter(x, residuals, s=60, alpha=0.6, color='steelblue', edgecolors='black')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    # Add LOWESS smooth
    sorted_x_indices = np.argsort(x)
    if len(x) > 5:
        try:
            smoothed = savgol_filter(residuals[sorted_x_indices],
                                    window_length=min(11, len(x)//2*2+1), polyorder=2)
            ax.plot(x[sorted_x_indices], smoothed, 'g-', linewidth=2, label='Smooth trend')
        except:
            pass
    ax.set_xlabel('x (Predictor)', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title('Residuals vs Predictor x', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, residuals

def create_correlation_analysis(df):
    """
    Create detailed correlation analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')

    # 1. Correlation matrix heatmap
    ax = axes[0]
    corr_matrix = df[['x', 'Y']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.4f', cmap='coolwarm', center=0,
                square=True, linewidths=2, cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title('Pearson Correlation Matrix', fontsize=12, fontweight='bold')

    # 2. Correlation statistics
    ax = axes[1]
    ax.axis('off')

    # Compute various correlation measures
    pearson_r, pearson_p = stats.pearsonr(df['x'], df['Y'])
    spearman_r, spearman_p = stats.spearmanr(df['x'], df['Y'])
    kendall_tau, kendall_p = stats.kendalltau(df['x'], df['Y'])

    stats_text = f"""
    Correlation Statistics
    {'='*40}

    Pearson Correlation:
      r = {pearson_r:.4f}
      p-value = {pearson_p:.6f}
      {'Significant' if pearson_p < 0.05 else 'Not significant'} at α=0.05

    Spearman Rank Correlation:
      ρ = {spearman_r:.4f}
      p-value = {spearman_p:.6f}
      {'Significant' if spearman_p < 0.05 else 'Not significant'} at α=0.05

    Kendall's Tau:
      τ = {kendall_tau:.4f}
      p-value = {kendall_p:.6f}
      {'Significant' if kendall_p < 0.05 else 'Not significant'} at α=0.05

    Interpretation:
    {'-'*40}
    - All correlations are positive and strong
    - Pearson measures linear relationship
    - Spearman measures monotonic relationship
    - Similar values suggest linear trend
    """

    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

def create_variance_analysis(df):
    """
    Analyze variance patterns (heteroscedasticity check).

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Variance Analysis: Checking for Heteroscedasticity', fontsize=16, fontweight='bold')

    # Bin data by x to analyze variance patterns
    df_sorted = df.sort_values('x')
    bins = pd.cut(df_sorted['x'], bins=5)
    grouped = df_sorted.groupby(bins)

    # 1. Variance by x bins
    ax = axes[0]
    variances = grouped['Y'].var()
    means = grouped['Y'].mean()
    counts = grouped.size()

    bin_centers = [interval.mid for interval in variances.index]
    ax.bar(range(len(variances)), variances.values, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(variances)))
    ax.set_xticklabels([f'{interval.left:.1f}-{interval.right:.1f}' for interval in variances.index], rotation=45)
    ax.set_xlabel('x Range', fontsize=11)
    ax.set_ylabel('Variance of Y', fontsize=11)
    ax.set_title('Variance of Y across x Ranges', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add count labels
    for i, (v, c) in enumerate(zip(variances.values, counts.values)):
        ax.text(i, v, f'n={c}', ha='center', va='bottom', fontsize=9)

    # 2. Spread-Level Plot
    ax = axes[1]
    ax.scatter(means.values, variances.values, s=100, alpha=0.7, color='steelblue', edgecolors='black')

    # Fit line to check relationship between mean and variance
    if len(means) > 1:
        z = np.polyfit(means.values, variances.values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(means.min(), means.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Linear fit: slope={z[0]:.4f}')

    ax.set_xlabel('Mean of Y', fontsize=11)
    ax.set_ylabel('Variance of Y', fontsize=11)
    ax.set_title('Spread-Level Plot (Mean vs Variance)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS - PHASE 3: BIVARIATE ANALYSIS")
    print("="*80 + "\n")

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape}")

    # Create basic scatterplot
    print("\nCreating basic scatterplot...")
    fig_scatter = create_basic_scatterplot(df)
    fig_scatter.savefig(VIZ_DIR / 'scatterplot_basic.png', dpi=300, bbox_inches='tight')
    plt.close(fig_scatter)
    print(f"Saved: {VIZ_DIR / 'scatterplot_basic.png'}")

    # Create functional forms comparison
    print("Comparing functional forms...")
    fig_forms, r2_scores = create_functional_forms_comparison(df)
    fig_forms.savefig(VIZ_DIR / 'functional_forms_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig_forms)
    print(f"Saved: {VIZ_DIR / 'functional_forms_comparison.png'}")
    print("\nR² Scores:")
    for form, r2 in r2_scores.items():
        print(f"  {form}: {r2:.4f}")

    # Create residual analysis
    print("\nCreating residual analysis...")
    fig_resid, residuals = create_residual_analysis(df)
    fig_resid.savefig(VIZ_DIR / 'residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig_resid)
    print(f"Saved: {VIZ_DIR / 'residual_analysis.png'}")

    # Create correlation analysis
    print("Creating correlation analysis...")
    fig_corr = create_correlation_analysis(df)
    fig_corr.savefig(VIZ_DIR / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig_corr)
    print(f"Saved: {VIZ_DIR / 'correlation_analysis.png'}")

    # Create variance analysis
    print("Creating variance analysis...")
    fig_var = create_variance_analysis(df)
    fig_var.savefig(VIZ_DIR / 'variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig_var)
    print(f"Saved: {VIZ_DIR / 'variance_analysis.png'}")

    print("\n" + "="*80)
    print("PHASE 3 COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
