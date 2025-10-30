"""
Residual Diagnostics and Variance Analysis
==========================================
Author: EDA Specialist Agent
Date: 2025-10-27

This script performs detailed residual analysis and checks for heteroscedasticity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Setup
BASE_DIR = Path('/workspace')
DATA_PATH = BASE_DIR / 'data' / 'data.csv'
VIZ_DIR = BASE_DIR / 'eda' / 'visualizations'

plt.style.use('seaborn-v0_8-darkgrid')

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("RESIDUAL DIAGNOSTICS")
print("=" * 80)

# Fit linear model
X = df['x'].values
y = df['Y'].values
z_lin = np.polyfit(X, y, 1)
p_lin = np.poly1d(z_lin)

# Calculate residuals
fitted = p_lin(X)
residuals = y - fitted
std_residuals = residuals / np.std(residuals)

# Calculate additional metrics
leverage = 1/len(X) + (X - np.mean(X))**2 / np.sum((X - np.mean(X))**2)
cooks_d = (residuals**2 / (2 * np.var(residuals))) * (leverage / (1 - leverage)**2)

print("\nRESIDUAL STATISTICS:")
print(f"  Mean: {np.mean(residuals):.6f}")
print(f"  Std Dev: {np.std(residuals):.4f}")
print(f"  Min: {np.min(residuals):.4f}")
print(f"  Max: {np.max(residuals):.4f}")
print(f"  Range: {np.max(residuals) - np.min(residuals):.4f}")

# Figure 1: Comprehensive residual diagnostics
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Residual Diagnostics (Linear Model)', fontsize=16, fontweight='bold')

# 1. Residuals vs Fitted
axes[0, 0].scatter(fitted, residuals, s=80, alpha=0.7, edgecolors='black', linewidth=1, color='steelblue')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted Values', fontsize=12)
axes[0, 0].set_ylabel('Residuals', fontsize=12)
axes[0, 0].set_title('Residuals vs Fitted', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Scale-Location (sqrt of absolute residuals)
sqrt_abs_resid = np.sqrt(np.abs(std_residuals))
axes[0, 2].scatter(fitted, sqrt_abs_resid, s=80, alpha=0.7, edgecolors='black', linewidth=1, color='coral')
# Add smooth line
sorted_idx = np.argsort(fitted)
axes[0, 2].plot(fitted[sorted_idx], np.convolve(sqrt_abs_resid[sorted_idx], np.ones(5)/5, mode='same'),
                'r-', linewidth=2, label='Moving average')
axes[0, 2].set_xlabel('Fitted Values', fontsize=12)
axes[0, 2].set_ylabel('√|Standardized Residuals|', fontsize=12)
axes[0, 2].set_title('Scale-Location Plot', fontsize=13, fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Residuals vs x
axes[1, 0].scatter(X, residuals, s=80, alpha=0.7, edgecolors='black', linewidth=1, color='purple')
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_ylabel('Residuals', fontsize=12)
axes[1, 0].set_title('Residuals vs x', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 5. Histogram of residuals
axes[1, 1].hist(residuals, bins=10, edgecolor='black', alpha=0.7, color='lightgreen', density=True)
# Overlay normal distribution
mu, sigma = np.mean(residuals), np.std(residuals)
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[1, 1].plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', linewidth=2, label='Normal dist')
axes[1, 1].set_xlabel('Residuals', fontsize=12)
axes[1, 1].set_ylabel('Density', fontsize=12)
axes[1, 1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Cook's distance
axes[1, 2].stem(range(len(cooks_d)), cooks_d, linefmt='steelblue', markerfmt='o', basefmt='gray')
axes[1, 2].axhline(y=4/len(X), color='red', linestyle='--', linewidth=2, label='Threshold (4/n)')
axes[1, 2].set_xlabel('Observation Index', fontsize=12)
axes[1, 2].set_ylabel("Cook's Distance", fontsize=12)
axes[1, 2].set_title("Cook's Distance (Influence)", fontsize=13, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'residual_diagnostics.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {VIZ_DIR / 'residual_diagnostics.png'}")
plt.close()

# Figure 2: Heteroscedasticity analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Heteroscedasticity Assessment', fontsize=16, fontweight='bold')

# 1. Residuals^2 vs x
axes[0, 0].scatter(X, residuals**2, s=80, alpha=0.7, edgecolors='black', linewidth=1, color='orange')
# Fit line to squared residuals
z_sq = np.polyfit(X, residuals**2, 1)
p_sq = np.poly1d(z_sq)
x_line = np.linspace(X.min(), X.max(), 100)
axes[0, 0].plot(x_line, p_sq(x_line), 'r--', linewidth=2, label=f'Trend: slope={z_sq[0]:.6f}')
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('Squared Residuals', fontsize=12)
axes[0, 0].set_title('Squared Residuals vs x', fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Absolute residuals vs x
axes[0, 1].scatter(X, np.abs(residuals), s=80, alpha=0.7, edgecolors='black', linewidth=1, color='teal')
z_abs = np.polyfit(X, np.abs(residuals), 1)
p_abs = np.poly1d(z_abs)
axes[0, 1].plot(x_line, p_abs(x_line), 'r--', linewidth=2, label=f'Trend: slope={z_abs[0]:.6f}')
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_ylabel('|Residuals|', fontsize=12)
axes[0, 1].set_title('Absolute Residuals vs x', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals by x segments
x_tertiles = df['x'].quantile([0, 0.33, 0.67, 1.0])
df_temp = df.copy()
df_temp['residuals'] = residuals
df_temp['x_segment'] = pd.cut(df_temp['x'], bins=x_tertiles, labels=['Low', 'Mid', 'High'], include_lowest=True)

segments = ['Low', 'Mid', 'High']
resid_by_seg = [df_temp[df_temp['x_segment'] == seg]['residuals'].values for seg in segments]
bp = axes[1, 0].boxplot(resid_by_seg, labels=segments, patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[1, 0].set_xlabel('x Segment', fontsize=12)
axes[1, 0].set_ylabel('Residuals', fontsize=12)
axes[1, 0].set_title('Residuals by x Range', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Variance by x segments
variances = [df_temp[df_temp['x_segment'] == seg]['residuals'].var() for seg in segments]
std_devs = [df_temp[df_temp['x_segment'] == seg]['residuals'].std() for seg in segments]
x_pos = np.arange(len(segments))
axes[1, 1].bar(x_pos, std_devs, color=['blue', 'green', 'red'], alpha=0.7, edgecolor='black')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(segments)
axes[1, 1].set_xlabel('x Segment', fontsize=12)
axes[1, 1].set_ylabel('Std Dev of Residuals', fontsize=12)
axes[1, 1].set_title('Variance by x Range', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
# Add values on bars
for i, v in enumerate(std_devs):
    axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(VIZ_DIR / 'heteroscedasticity_analysis.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VIZ_DIR / 'heteroscedasticity_analysis.png'}")
plt.close()

# Statistical tests
print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

# Shapiro-Wilk test for normality of residuals
stat_sw, p_sw = stats.shapiro(residuals)
print(f"\nShapiro-Wilk Test (Normality of Residuals):")
print(f"  Statistic: {stat_sw:.4f}")
print(f"  P-value: {p_sw:.4f}")
if p_sw > 0.05:
    print(f"  -> Fail to reject normality (p > 0.05)")
else:
    print(f"  -> Reject normality (p <= 0.05)")

# Breusch-Pagan test for heteroscedasticity (simplified)
# Regress squared residuals on X
from scipy.stats import f as f_dist
resid_sq = residuals**2
z_bp = np.polyfit(X, resid_sq, 1)
p_bp = np.poly1d(z_bp)
resid_sq_pred = p_bp(X)
ss_res_bp = np.sum((resid_sq - resid_sq_pred)**2)
ss_tot_bp = np.sum((resid_sq - np.mean(resid_sq))**2)
r2_bp = 1 - (ss_res_bp / ss_tot_bp)
n = len(X)
bp_stat = n * r2_bp
# Chi-square distribution with 1 df
p_bp = 1 - stats.chi2.cdf(bp_stat, 1)

print(f"\nBreusch-Pagan Test (Heteroscedasticity):")
print(f"  Test statistic: {bp_stat:.4f}")
print(f"  P-value: {p_bp:.4f}")
if p_bp > 0.05:
    print(f"  -> Fail to reject homoscedasticity (p > 0.05)")
else:
    print(f"  -> Reject homoscedasticity - evidence of heteroscedasticity (p <= 0.05)")

# Durbin-Watson test for autocorrelation
dw = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
print(f"\nDurbin-Watson Test (Autocorrelation):")
print(f"  Statistic: {dw:.4f}")
print(f"  (Values near 2.0 suggest no autocorrelation)")
if 1.5 < dw < 2.5:
    print(f"  -> No strong evidence of autocorrelation")
else:
    print(f"  -> Possible autocorrelation detected")

# Variance by segment
print(f"\n" + "=" * 80)
print("VARIANCE ANALYSIS BY X SEGMENT")
print("=" * 80)
for seg in segments:
    seg_data = df_temp[df_temp['x_segment'] == seg]['residuals']
    print(f"\n{seg} x range:")
    print(f"  N: {len(seg_data)}")
    print(f"  Mean residual: {seg_data.mean():.4f}")
    print(f"  Variance: {seg_data.var():.4f}")
    print(f"  Std Dev: {seg_data.std():.4f}")
    print(f"  Range: [{seg_data.min():.4f}, {seg_data.max():.4f}]")

# Levene's test for equality of variances
stat_lev, p_lev = stats.levene(*resid_by_seg)
print(f"\nLevene's Test (Equality of Variances):")
print(f"  Statistic: {stat_lev:.4f}")
print(f"  P-value: {p_lev:.4f}")
if p_lev > 0.05:
    print(f"  -> Fail to reject equal variances (p > 0.05)")
else:
    print(f"  -> Reject equal variances (p <= 0.05)")

print("\n" + "=" * 80)
print("Residual diagnostics complete!")
print("=" * 80)
