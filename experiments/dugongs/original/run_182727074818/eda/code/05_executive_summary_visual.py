"""
Executive Summary Visualization
================================
Purpose: Create single comprehensive visual summarizing key EDA findings
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

def create_executive_summary(df):
    """
    Create comprehensive executive summary visualization.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Title
    fig.suptitle('EDA Executive Summary: Y vs x Relationship (n=27)',
                 fontsize=18, fontweight='bold', y=0.98)

    # --- Row 1: Data Overview ---

    # Panel 1: Raw data with best fit
    ax1 = fig.add_subplot(gs[0, 0])
    x = df['x'].values
    y = df['Y'].values

    ax1.scatter(x, y, s=80, alpha=0.7, color='steelblue', edgecolors='black', linewidth=1.5, label='Data')

    # Logarithmic fit
    def log_fit(x, a, b):
        return a * np.log(x + 1) + b

    popt_log, _ = curve_fit(log_fit, x, y)
    x_range = np.linspace(x.min(), x.max(), 200)
    y_pred_log = log_fit(x_range, *popt_log)
    ax1.plot(x_range, y_pred_log, 'r-', linewidth=3, label='Log fit (R²=0.89)', alpha=0.8)

    ax1.set_xlabel('x (Predictor)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y (Response)', fontsize=11, fontweight='bold')
    ax1.set_title('A. Relationship Overview', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Distribution comparison
    ax2 = fig.add_subplot(gs[0, 1])

    # Box plots
    bp1 = ax2.boxplot([x], positions=[1], widths=0.6, patch_artist=True,
                       labels=['x'], vert=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][0].set_alpha(0.7)

    # Normalize Y to similar scale for visualization
    y_scaled = (y - y.min()) / (y.max() - y.min()) * (x.max() - x.min()) + x.min()
    bp2 = ax2.boxplot([y_scaled], positions=[2], widths=0.6, patch_artist=True,
                       labels=['Y (scaled)'], vert=True)
    bp2['boxes'][0].set_facecolor('lightcoral')
    bp2['boxes'][0].set_alpha(0.7)

    ax2.set_ylabel('Value (x in original scale)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Variable Distributions', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add text with key stats
    stats_text = f'x: μ={x.mean():.1f}, σ={x.std():.1f}\nY: μ={y.mean():.2f}, σ={y.std():.2f}'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel 3: Model comparison
    ax3 = fig.add_subplot(gs[0, 2])

    models = ['Linear', 'Sqrt', 'Asymp', 'Quad', 'Cubic', 'Log']
    r2_values = [0.677, 0.826, 0.834, 0.874, 0.880, 0.888]
    colors = ['red' if r2 < 0.85 else 'orange' if r2 < 0.88 else 'green' for r2 in r2_values]

    bars = ax3.barh(models, r2_values, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('R² (Goodness of Fit)', fontsize=11, fontweight='bold')
    ax3.set_title('C. Model Comparison', fontsize=12, fontweight='bold')
    ax3.set_xlim([0.6, 0.9])
    ax3.axvline(x=0.85, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='x')

    # Add values
    for i, (bar, val) in enumerate(zip(bars, r2_values)):
        ax3.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    # --- Row 2: Hypothesis Testing Results ---

    # Panel 4: Log transformation benefit
    ax4 = fig.add_subplot(gs[1, 0])

    log_x = np.log(x + 1)
    ax4.scatter(log_x, y, s=60, alpha=0.6, color='steelblue', edgecolors='black')

    # Linear fit on log scale
    A = np.vstack([log_x, np.ones(len(log_x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    log_x_sorted = np.sort(log_x)
    ax4.plot(log_x_sorted, slope * log_x_sorted + intercept, 'r--', linewidth=2, label='Linear on log(x)')

    ax4.set_xlabel('log(x + 1)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Y', fontsize=11, fontweight='bold')
    ax4.set_title('D. Log Transform Linearizes Relationship', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Residual diagnostics
    ax5 = fig.add_subplot(gs[1, 1])

    # Residuals from linear model
    A_linear = np.vstack([x, np.ones(len(x))]).T
    slope_lin, intercept_lin = np.linalg.lstsq(A_linear, y, rcond=None)[0]
    y_pred_lin = slope_lin * x + intercept_lin
    residuals_lin = y - y_pred_lin

    ax5.scatter(y_pred_lin, residuals_lin, s=60, alpha=0.6, color='red',
                edgecolors='black', label='Linear model')

    # Residuals from log model
    y_pred_log_all = log_fit(x, *popt_log)
    residuals_log = y - y_pred_log_all

    ax5.scatter(y_pred_log_all, residuals_log, s=60, alpha=0.6, color='green',
                edgecolors='black', label='Log model')

    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Residuals', fontsize=11, fontweight='bold')
    ax5.set_title('E. Residual Comparison', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Variance analysis
    ax6 = fig.add_subplot(gs[1, 2])

    # Bin data
    bins = pd.cut(x, bins=5)
    grouped = df.groupby(bins)
    variances = grouped['Y'].var()
    means = grouped['Y'].mean()

    bin_labels = [f'{i.left:.0f}-{i.right:.0f}' for i in variances.index]
    ax6.bar(range(len(variances)), variances.values, alpha=0.7, color='steelblue', edgecolor='black')
    ax6.set_xticks(range(len(variances)))
    ax6.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax6.set_xlabel('x Range', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Variance of Y', fontsize=11, fontweight='bold')
    ax6.set_title('F. Variance Across x (Homoscedastic)', fontsize=12, fontweight='bold')
    ax6.axhline(y=variances.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7, label='Mean')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    # --- Row 3: Key Findings Summary ---

    # Panel 7: Hypothesis summary table
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.axis('tight')
    ax7.axis('off')

    hypotheses = [
        ['Hypothesis', 'Verdict', 'Key Finding'],
        ['H1: Saturation', '⭐⭐☆☆☆', 'Weak evidence, need higher x values'],
        ['H2: Logarithmic', '⭐⭐⭐⭐⭐', 'Strongly supported (R² improvement +31%)'],
        ['H3: Homoscedasticity', '⭐⭐⭐⭐☆', 'Constant variance reasonable'],
        ['H4: Change point (x≈7)', '⭐⭐⭐⭐⭐', 'RSS improves 66% with breakpoint'],
        ['H5: Consistent error', '⭐☆☆☆☆', 'Replicate variance varies'],
    ]

    table = ax7.table(cellText=hypotheses, cellLoc='left', loc='center',
                     colWidths=[0.25, 0.15, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.8)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(hypotheses)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')

    # Highlight strong findings
    table[(2, 1)].set_facecolor('#90EE90')  # H2
    table[(4, 1)].set_facecolor('#90EE90')  # H4

    ax7.set_title('G. Hypothesis Testing Summary', fontsize=12, fontweight='bold',
                  pad=20, loc='left')

    # Panel 8: Recommendations
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    recommendations_text = """
RECOMMENDED MODELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PRIMARY: Logarithmic
   Y ~ Normal(α + β·log(x+1), σ²)

   Reasons:
   • Best fit (R²=0.89)
   • Simple (3 parameters)
   • Interpretable
   • Extrapolates well

2. SECONDARY: Segmented
   Two regimes at x≈7
   (if validated by domain)

3. TERTIARY: Asymptotic
   Y ~ Y_max·x/(K+x)
   (if saturation critical)

━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY INSIGHTS:

✓ Strong non-linearity
✓ Diminishing returns
✓ Constant variance OK
✓ n=27 (favor parsimony)
✓ Quality data, ready to model
    """

    ax8.text(0.05, 0.95, recommendations_text, transform=ax8.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8,
                      edgecolor='black', linewidth=2))

    ax8.set_title('H. Model Recommendations', fontsize=12, fontweight='bold',
                  pad=20, loc='left')

    return fig

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("CREATING EXECUTIVE SUMMARY VISUALIZATION")
    print("="*80 + "\n")

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape}")

    # Create executive summary
    print("\nCreating comprehensive summary visualization...")
    fig = create_executive_summary(df)

    # Save
    output_path = VIZ_DIR / 'EXECUTIVE_SUMMARY.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_path}")

    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
