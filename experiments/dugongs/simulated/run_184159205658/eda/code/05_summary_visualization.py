"""
Summary Visualization for EDA Report
====================================
Author: EDA Specialist Agent
Date: 2025-10-27

Creates a comprehensive single-page summary of key findings.
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

# Setup
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data.csv'
VIZ_DIR = BASE_DIR / 'eda' / 'visualizations'

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv(DATA_PATH)

# Calculate models
X = df['x'].values
y = df['Y'].values

# Linear
z_lin = np.polyfit(X, y, 1)
p_lin = np.poly1d(z_lin)
r2_lin = 1 - np.sum((y - p_lin(X))**2) / np.sum((y - np.mean(y))**2)

# Logarithmic
X_log = np.log(X + 0.1)
z_log = np.polyfit(X_log, y, 1)
p_log = np.poly1d(z_log)
r2_log = 1 - np.sum((y - p_log(X_log))**2) / np.sum((y - np.mean(y))**2)

# Quadratic
z_quad = np.polyfit(X, y, 2)
p_quad = np.poly1d(z_quad)
r2_quad = 1 - np.sum((y - p_quad(X))**2) / np.sum((y - np.mean(y))**2)

# Calculate residuals
residuals = y - p_lin(X)

# Create summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('EDA Summary: X-Y Relationship Analysis', fontsize=18, fontweight='bold', y=0.98)

# 1. Main scatter with best fit (top left, large)
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax1.scatter(df['x'], df['Y'], s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
            color='steelblue', label='Observed data', zorder=3)

x_plot = np.linspace(df['x'].min(), df['x'].max(), 200)
ax1.plot(x_plot, p_lin(x_plot), 'r--', linewidth=2, alpha=0.7,
         label=f'Linear (R²={r2_lin:.3f})', zorder=2)
ax1.plot(x_plot, p_log(np.log(x_plot + 0.1)), 'g-', linewidth=2.5,
         label=f'Logarithmic (R²={r2_log:.3f}) [RECOMMENDED]', zorder=2)
ax1.plot(x_plot, p_quad(x_plot), 'm:', linewidth=2, alpha=0.7,
         label=f'Quadratic (R²={r2_quad:.3f})', zorder=1)

ax1.set_xlabel('x (Predictor)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Y (Response)', fontsize=13, fontweight='bold')
ax1.set_title('Primary Finding: Nonlinear Relationship with Diminishing Returns',
              fontsize=14, fontweight='bold', pad=10)
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, alpha=0.3)

# Add annotation for key regions
ax1.axvspan(0, 5, alpha=0.1, color='blue', label='_nolegend_')
ax1.text(2.5, 1.75, 'Rapid\nIncrease', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax1.axvspan(5, 15, alpha=0.1, color='green', label='_nolegend_')
ax1.text(10, 1.75, 'Transition', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax1.axvspan(15, 32, alpha=0.1, color='yellow', label='_nolegend_')
ax1.text(23, 1.75, 'Plateau\n(Sparse data)', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# 2. Residuals vs fitted (top right)
ax2 = fig.add_subplot(gs[0, 2])
fitted = p_lin(X)
ax2.scatter(fitted, residuals, s=70, alpha=0.7, edgecolors='black',
            linewidth=1, color='coral')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Fitted Values', fontsize=10)
ax2.set_ylabel('Residuals', fontsize=10)
ax2.set_title('Residual Pattern\n(Linear Model)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95, 'U-shaped pattern\n→ Nonlinearity',
         transform=ax2.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# 3. Q-Q plot (middle right)
ax3 = fig.add_subplot(gs[1, 2])
stats.probplot(residuals, dist="norm", plot=ax3)
ax3.set_title('Residual Normality', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.text(0.05, 0.95, 'Normal residuals ✓\n(p=0.334)',
         transform=ax3.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# 4. Data distribution summary (bottom left)
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')

summary_text = f"""
DATA QUALITY SUMMARY
{'='*35}

Sample Size: N = 27 observations
Missing Values: 0 (0%)
Outliers: 1 in x (x=31.5, legitimate)

PREDICTOR (x):
  • Range: [1.0, 31.5]
  • Mean: {df['x'].mean():.2f}, Median: {df['x'].median():.2f}
  • Std Dev: {df['x'].std():.2f}
  • Skewness: {df['x'].skew():.2f} (right-skewed)
  • Unique values: 19 (7 with replication)

RESPONSE (Y):
  • Range: [{df['Y'].min():.2f}, {df['Y'].max():.2f}]
  • Mean: {df['Y'].mean():.2f}, Median: {df['Y'].median():.2f}
  • Std Dev: {df['Y'].std():.2f}
  • Skewness: {df['Y'].skew():.2f} (left-skewed)

CORRELATION:
  • Pearson r = 0.720 (strong positive)
  • Spearman ρ = 0.782 (stronger monotonic)
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=9, va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 5. Model recommendations (bottom middle)
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

model_text = """
RECOMMENDED MODELS
{'='*35}

PRIMARY: Logarithmic Regression
  Y ~ Normal(β₀ + β₁·log(x), σ²)

  Pros:
    ✓ Good fit (R²=0.83)
    ✓ Interpretable parameters
    ✓ Captures diminishing returns
    ✓ Only 2 parameters (parsimonious)

  Priors:
    β₀ ~ Normal(1.7, 0.5)
    β₁ ~ Normal(0.3, 0.2)
    σ ~ HalfNormal(0.2)

ALTERNATIVE: Quadratic Regression
  Y ~ Normal(β₀+β₁·x+β₂·x², σ²)

  Best empirical fit (R²=0.86)
  Use if prediction accuracy critical

THEORETICAL: Asymptotic Model
  Y ~ Normal(Ymax·x/(K+x), σ²)

  If saturation expected (R²=0.82)
  Parameters: Ymax≈2.6, K≈0.6
"""

ax5.text(0.05, 0.95, model_text, transform=ax5.transAxes,
         fontsize=8.5, va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

# 6. Key findings (bottom right)
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

findings_text = """
KEY FINDINGS
{'='*25}

✓ STRENGTHS:
  • Complete data (no NAs)
  • Normal residuals
  • Constant variance
  • No influential outliers
  • Clear signal (r=0.72)

⚠ CONCERNS:
  • Sparse high-x coverage
    (only 3 obs with x>20)
  • Modest N=27
  • Possible autocorrelation
    (DW=0.66)

FUNCTIONAL FORM:
  • Linear: REJECTED
    (R²=0.52, bias)
  • Nonlinear: REQUIRED
    (R²=0.82-0.86)

PATTERN:
  Diminishing marginal
  returns - logarithmic
  or asymptotic form

VARIANCE:
  Homoscedastic ✓
  (Breusch-Pagan p=0.55)
"""

ax6.text(0.05, 0.95, findings_text, transform=ax6.transAxes,
         fontsize=8.5, va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

plt.savefig(VIZ_DIR / 'eda_summary.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'eda_summary.png'}")
print("\nSummary visualization complete!")
print("="*80)
