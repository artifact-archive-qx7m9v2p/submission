"""
IMPLEMENTATION EXAMPLE: Recommended Model for Count Data

Based on comprehensive transformation and feature engineering analysis,
this script demonstrates the recommended modeling approach:

GLM with log link (Poisson or Negative Binomial) with quadratic terms

Author: Analyst 3
Date: 2025-10-29
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv('/workspace/data/data_analyst_3.csv')
print("Dataset loaded: {} observations".format(len(df)))
print(df.head())

# ============================================================================
# CREATE FEATURES
# ============================================================================

# Extract variables
year = df['year'].values
C = df['C'].values

# Create design matrix with quadratic term
X = pd.DataFrame({
    'year': year,
    'year2': year**2
})
X = sm.add_constant(X)

print("\nFeatures created:")
print(X.head())

# ============================================================================
# FIT RECOMMENDED MODEL: POISSON GLM
# ============================================================================

print("\n" + "="*70)
print("FITTING POISSON GLM WITH LOG LINK")
print("="*70)

# Fit Poisson GLM
model_poisson = sm.GLM(C, X, family=sm.families.Poisson()).fit()

print("\nModel summary:")
print(model_poisson.summary())

# ============================================================================
# CHECK FOR OVERDISPERSION
# ============================================================================

print("\n" + "="*70)
print("CHECKING FOR OVERDISPERSION")
print("="*70)

# Overdispersion parameter
overdispersion = model_poisson.deviance / model_poisson.df_resid
print(f"\nOverdispersion parameter: {overdispersion:.4f}")
print(f"Deviance: {model_poisson.deviance:.2f}")
print(f"Degrees of freedom: {model_poisson.df_resid}")

if overdispersion > 1.5:
    print("\n⚠ Overdispersion detected (φ > 1.5)")
    print("Fitting Negative Binomial GLM instead...")

    model_nb = sm.GLM(C, X, family=sm.families.NegativeBinomial()).fit()
    print("\nNegative Binomial model summary:")
    print(model_nb.summary())

    model_final = model_nb
    model_name = "Negative Binomial"
else:
    print("\n✓ No significant overdispersion (φ ≤ 1.5)")
    print("Poisson model is appropriate")
    model_final = model_poisson
    model_name = "Poisson"

# ============================================================================
# MODEL DIAGNOSTICS
# ============================================================================

print("\n" + "="*70)
print("MODEL DIAGNOSTICS")
print("="*70)

# Get residuals
resid_deviance = model_final.resid_deviance
resid_pearson = model_final.resid_pearson

# Normality test on deviance residuals
shapiro_stat, shapiro_p = stats.shapiro(resid_deviance)
print(f"\nShapiro-Wilk test on deviance residuals:")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  p-value: {shapiro_p:.4f}")

# Durbin-Watson test for autocorrelation (if time series)
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(resid_deviance)
print(f"\nDurbin-Watson statistic: {dw_stat:.4f}")
print(f"  (2 = no autocorrelation, <2 = positive, >2 = negative)")

# ============================================================================
# VISUALIZE DIAGNOSTICS
# ============================================================================

print("\n" + "="*70)
print("GENERATING DIAGNOSTIC PLOTS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'{model_name} GLM: Diagnostic Plots', fontsize=14, fontweight='bold')

# 1. Residuals vs Fitted
axes[0, 0].scatter(model_final.fittedvalues, resid_deviance, alpha=0.6, edgecolors='black')
axes[0, 0].axhline(0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Fitted values (μ̂)', fontsize=10)
axes[0, 0].set_ylabel('Deviance residuals', fontsize=10)
axes[0, 0].set_title('Residuals vs Fitted', fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Q-Q Plot
stats.probplot(resid_deviance, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title(f'Q-Q Plot\nShapiro-Wilk p={shapiro_p:.4f}', fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Scale-Location
axes[1, 0].scatter(model_final.fittedvalues, np.sqrt(np.abs(resid_deviance)),
                   alpha=0.6, edgecolors='black')
axes[1, 0].set_xlabel('Fitted values (μ̂)', fontsize=10)
axes[1, 0].set_ylabel('√|Deviance residuals|', fontsize=10)
axes[1, 0].set_title('Scale-Location Plot', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. Residuals vs Order (for time series patterns)
axes[1, 1].scatter(range(len(resid_deviance)), resid_deviance, alpha=0.6, edgecolors='black')
axes[1, 1].axhline(0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Observation order', fontsize=10)
axes[1, 1].set_ylabel('Deviance residuals', fontsize=10)
axes[1, 1].set_title(f'Residuals vs Order\nDW={dw_stat:.3f}', fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/11_final_model_diagnostics.png',
            dpi=300, bbox_inches='tight')
print("\n✓ Diagnostic plots saved: 11_final_model_diagnostics.png")
plt.close()

# ============================================================================
# MODEL PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

# Get predictions
predictions = model_final.get_prediction(X)
pred_summary = predictions.summary_frame(alpha=0.05)

# Add to dataframe
df_results = pd.DataFrame({
    'year': year,
    'C_observed': C,
    'C_predicted': pred_summary['mean'],
    'lower_95': pred_summary['mean_ci_lower'],
    'upper_95': pred_summary['mean_ci_upper'],
})

print("\nPredictions (first 10 rows):")
print(df_results.head(10))

print("\nPredictions (last 10 rows):")
print(df_results.tail(10))

# ============================================================================
# VISUALIZE FIT
# ============================================================================

print("\n" + "="*70)
print("GENERATING FIT VISUALIZATION")
print("="*70)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Scatter plot of observed data
ax.scatter(year, C, alpha=0.7, s=80, edgecolors='black', linewidths=1.5,
           label='Observed data', zorder=5, color='steelblue')

# Predicted values
year_sorted = np.sort(year)
idx_sorted = np.argsort(year)
ax.plot(year_sorted, pred_summary['mean'][idx_sorted], 'r-', linewidth=2.5,
        label=f'{model_name} GLM fit', zorder=4)

# Confidence interval
ax.fill_between(year_sorted,
                pred_summary['mean_ci_lower'][idx_sorted],
                pred_summary['mean_ci_upper'][idx_sorted],
                alpha=0.3, color='red', label='95% CI', zorder=3)

# Calculate R-squared (pseudo R²)
y_mean = np.mean(C)
ss_tot = np.sum((C - y_mean)**2)
ss_res = np.sum((C - pred_summary['mean'])**2)
r2 = 1 - (ss_res / ss_tot)

ax.set_xlabel('Year (standardized)', fontsize=12, fontweight='bold')
ax.set_ylabel('C (count)', fontsize=12, fontweight='bold')
ax.set_title(f'{model_name} GLM with Log Link: C ~ Poisson(μ), log(μ) = β₀ + β₁×year + β₂×year²\n' +
             f'Pseudo R² = {r2:.4f}, Overdispersion φ = {overdispersion:.3f}',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/eda/analyst_3/visualizations/12_final_model_fit.png',
            dpi=300, bbox_inches='tight')
print("\n✓ Fit visualization saved: 12_final_model_fit.png")
plt.close()

# ============================================================================
# INTERPRET COEFFICIENTS
# ============================================================================

print("\n" + "="*70)
print("COEFFICIENT INTERPRETATION")
print("="*70)

params = model_final.params
print(f"\nModel: log(μ) = β₀ + β₁×year + β₂×year²")
print(f"\nCoefficients:")
print(f"  β₀ (Intercept): {params['const']:.4f}")
print(f"  β₁ (year):      {params['year']:.4f}")
print(f"  β₂ (year²):     {params['year2']:.4f}")

print(f"\nInterpretation:")
print(f"  - At year=0 (mean year), expected count: μ = exp({params['const']:.4f}) = {np.exp(params['const']):.2f}")
print(f"  - One unit increase in year multiplies count by: exp({params['year']:.4f}) = {np.exp(params['year']):.3f}x")
print(f"  - Quadratic term {params['year2']:.4f} indicates {'accelerating' if params['year2'] > 0 else 'decelerating'} growth")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

# Fit alternative models for comparison
# 1. Linear (no quadratic term)
X_linear = sm.add_constant(pd.DataFrame({'year': year}))
model_linear = sm.GLM(C, X_linear, family=sm.families.Poisson()).fit()

# 2. Log-linear OLS (alternative approach)
log_C = np.log(C)
model_log_ols = sm.OLS(log_C, X).fit()

print("\nAIC Comparison:")
print(f"  Poisson GLM (quadratic): {model_poisson.aic:.2f}")
print(f"  Poisson GLM (linear):    {model_linear.aic:.2f}")
print(f"  Log-linear OLS:          {model_log_ols.aic:.2f}")

print(f"\nBest model by AIC: ", end="")
if model_poisson.aic < model_linear.aic and model_poisson.aic < model_log_ols.aic:
    print("Poisson GLM (quadratic) ✓")
elif model_linear.aic < model_log_ols.aic:
    print("Poisson GLM (linear)")
else:
    print("Log-linear OLS")

# ============================================================================
# SAVE PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save predictions to CSV
output_file = '/workspace/eda/analyst_3/model_predictions.csv'
df_results.to_csv(output_file, index=False)
print(f"\n✓ Predictions saved to: {output_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

print(f"""
RECOMMENDED MODEL: {model_name} GLM with log link

Model specification:
  C ~ {model_name}(μ, {'' if model_name == 'Poisson' else 'θ'})
  log(μ) = {params['const']:.4f} + {params['year']:.4f}×year + {params['year2']:.4f}×year²

Model fit:
  AIC: {model_final.aic:.2f}
  Pseudo R²: {r2:.4f}
  Overdispersion: {overdispersion:.3f}

Diagnostics:
  Shapiro-Wilk p-value: {shapiro_p:.4f} {'✓ Normal residuals' if shapiro_p > 0.05 else '⚠ Non-normal residuals'}
  Durbin-Watson: {dw_stat:.3f} {'✓ No autocorrelation' if 1.5 < dw_stat < 2.5 else '⚠ Check autocorrelation'}

Outputs generated:
  - Diagnostic plots: visualizations/11_final_model_diagnostics.png
  - Fit visualization: visualizations/12_final_model_fit.png
  - Predictions: model_predictions.csv

NEXT STEPS:
  1. Review diagnostic plots for any patterns
  2. Consider cross-validation for degree selection (2 vs 3)
  3. Check for influential observations if needed
  4. Validate predictions on holdout set if available
  5. Consider temporal autocorrelation structure for forecasting

RECOMMENDATION BASED ON: Comprehensive transformation analysis
  - Log transformation optimal (variance ratio: 34.7 → 0.58)
  - Quadratic term significant (ΔAIC = -41.4)
  - Count data characteristics evident
  - GLM framework respects data structure
""")

print("="*70)
print("For full analysis details, see:")
print("  - findings.md (comprehensive report)")
print("  - eda_log.md (exploration process)")
print("  - QUICK_REFERENCE.md (TL;DR version)")
print("="*70)
