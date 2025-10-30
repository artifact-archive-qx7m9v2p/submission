"""
Univariate Analysis and Visualizations
=======================================
Purpose: Create comprehensive univariate visualizations for both variables
Author: EDA Specialist Agent
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup paths
OUTPUT_DIR = Path("/workspace/eda")
VIZ_DIR = OUTPUT_DIR / "visualizations"
DATA_PATH = OUTPUT_DIR / "cleaned_data.csv"

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_univariate_panel(df, var_name, bins=15):
    """
    Create comprehensive univariate analysis panel for a variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    var_name : str
        Variable name to analyze
    bins : int
        Number of bins for histogram

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Univariate Analysis: {var_name}', fontsize=16, fontweight='bold')

    data = df[var_name].values

    # 1. Histogram with KDE
    ax1 = axes[0, 0]
    ax1.hist(data, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='black')

    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 100)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # Add normal distribution overlay
    mu, sigma = data.mean(), data.std()
    normal_curve = stats.norm.pdf(x_range, mu, sigma)
    ax1.plot(x_range, normal_curve, 'g--', linewidth=2, label='Normal fit')

    ax1.set_xlabel(var_name, fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Distribution with KDE and Normal Overlay', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Boxplot with statistics
    ax2 = axes[0, 1]
    bp = ax2.boxplot(data, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)

    # Add mean marker
    mean_val = data.mean()
    ax2.scatter([1], [mean_val], color='red', s=100, zorder=3, marker='D', label='Mean')

    # Add text annotations
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    ax2.text(1.3, q1, f'Q1: {q1:.3f}', fontsize=9)
    ax2.text(1.3, median, f'Median: {median:.3f}', fontsize=9, fontweight='bold')
    ax2.text(1.3, q3, f'Q3: {q3:.3f}', fontsize=9)
    ax2.text(1.3, mean_val, f'Mean: {mean_val:.3f}', fontsize=9, color='red')

    ax2.set_ylabel(var_name, fontsize=11)
    ax2.set_title('Boxplot with Summary Statistics', fontsize=12)
    ax2.set_xticks([])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Q-Q Plot
    ax3 = axes[1, 0]
    stats.probplot(data, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normal Distribution)', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. ECDF (Empirical Cumulative Distribution Function)
    ax4 = axes[1, 1]
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax4.plot(sorted_data, ecdf, marker='.', linestyle='none', markersize=8, color='steelblue', alpha=0.7)
    ax4.plot(sorted_data, ecdf, linestyle='-', linewidth=1, color='steelblue', alpha=0.5)

    # Add theoretical normal CDF
    normal_cdf = stats.norm.cdf(sorted_data, mu, sigma)
    ax4.plot(sorted_data, normal_cdf, 'r--', linewidth=2, label='Normal CDF')

    ax4.set_xlabel(var_name, fontsize=11)
    ax4.set_ylabel('Cumulative Probability', fontsize=11)
    ax4.set_title('Empirical CDF vs Normal CDF', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_distribution_comparison(df):
    """
    Create side-by-side distribution comparison of x and Y.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Distribution Comparison: x vs Y', fontsize=16, fontweight='bold')

    for idx, var in enumerate(['x', 'Y']):
        ax = axes[idx]
        data = df[var].values

        # Violin plot
        parts = ax.violinplot([data], positions=[1], widths=0.7,
                              showmeans=True, showmedians=True, showextrema=True)

        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)

        # Overlay individual points with jitter
        jittered_x = np.random.normal(1, 0.04, size=len(data))
        ax.scatter(jittered_x, data, alpha=0.4, s=30, color='darkblue')

        # Add statistics text
        mean_val = data.mean()
        median_val = np.median(data)
        std_val = data.std()

        text_str = f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}'
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_ylabel(var, fontsize=12)
        ax.set_title(f'{var} Distribution', fontsize=13, fontweight='bold')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig

def create_summary_statistics_table(df):
    """
    Create a visual table of summary statistics.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    # Compute statistics
    stats_data = []
    for var in ['x', 'Y']:
        data = df[var].values
        stats_row = [
            var,
            f"{len(data)}",
            f"{data.mean():.4f}",
            f"{np.median(data):.4f}",
            f"{data.std():.4f}",
            f"{data.min():.4f}",
            f"{data.max():.4f}",
            f"{stats.skew(data):.4f}",
            f"{stats.kurtosis(data):.4f}",
            f"{(data.std()/data.mean())*100:.2f}%"
        ]
        stats_data.append(stats_row)

    # Create table
    columns = ['Variable', 'N', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis', 'CV']
    table = ax.table(cellText=stats_data, colLabels=columns, cellLoc='center', loc='center',
                    colWidths=[0.1, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(stats_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#F2F2F2')

    plt.title('Summary Statistics Table', fontsize=14, fontweight='bold', pad=20)
    return fig

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS - PHASE 2: UNIVARIATE ANALYSIS")
    print("="*80 + "\n")

    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data: {df.shape}")

    # Create univariate panel for x
    print("\nCreating univariate panel for x...")
    fig_x = create_univariate_panel(df, 'x', bins=15)
    fig_x.savefig(VIZ_DIR / 'univariate_x.png', dpi=300, bbox_inches='tight')
    plt.close(fig_x)
    print(f"Saved: {VIZ_DIR / 'univariate_x.png'}")

    # Create univariate panel for Y
    print("Creating univariate panel for Y...")
    fig_y = create_univariate_panel(df, 'Y', bins=12)
    fig_y.savefig(VIZ_DIR / 'univariate_Y.png', dpi=300, bbox_inches='tight')
    plt.close(fig_y)
    print(f"Saved: {VIZ_DIR / 'univariate_Y.png'}")

    # Create distribution comparison
    print("Creating distribution comparison...")
    fig_comp = create_distribution_comparison(df)
    fig_comp.savefig(VIZ_DIR / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig_comp)
    print(f"Saved: {VIZ_DIR / 'distribution_comparison.png'}")

    # Create summary statistics table
    print("Creating summary statistics table...")
    fig_table = create_summary_statistics_table(df)
    fig_table.savefig(VIZ_DIR / 'summary_statistics_table.png', dpi=300, bbox_inches='tight')
    plt.close(fig_table)
    print(f"Saved: {VIZ_DIR / 'summary_statistics_table.png'}")

    print("\n" + "="*80)
    print("PHASE 2 COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
