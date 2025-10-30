# EDA Quick Reference Card

## Critical Finding: OVERDISPERSION DETECTED

```
χ² = 38.56, df = 11, p < 0.001
Dispersion parameter φ = 3.51

⚠️  Simple Binomial(n, p) model is REJECTED
⚠️  Variance is 3.5x larger than expected
```

---

## Data Summary

| Metric | Value |
|--------|-------|
| Observations | 12 trials |
| Total sample size | 2,814 |
| Total successes | 208 |
| Pooled proportion | 0.0739 (7.39%) |
| Proportion range | [0.000, 0.144] |
| Sample size range | [47, 810] |

---

## Statistical Tests Summary

| Test | Result | P-value | Interpretation |
|------|--------|---------|----------------|
| Chi-square GOF | REJECT H0 | < 0.001 | Constant p rejected |
| Temporal trend | NO | 0.199 | No trend over time |
| Sample size effect | NO | 0.787 | No size bias |
| Group structure | YES | 0.012 | Two distinct groups |
| Runs test | Random | 0.226 | No pattern in sequence |

---

## Model Recommendations

### ✗ DO NOT USE
- Simple Binomial(n, p) with constant p

### ✓ RECOMMENDED: Beta-Binomial
```
θ_i ~ Beta(α, β)
r_i | θ_i ~ Binomial(n_i, θ_i)

Priors:
  p ~ Beta(2, 25)        → E[p] ≈ 0.074
  φ ~ Gamma(2, 0.5)      → E[φ] = 4
```

### ✓ ALTERNATIVE: Mixture Model
```
Components:
  p_low ~ Beta(2, 38)    → centers at 0.05
  p_high ~ Beta(4, 32)   → centers at 0.11
  w ~ Dirichlet(1, 1)    → uniform mixing
```

---

## Outliers

| Trial | n | r | Proportion | Status | z-score |
|-------|---|---|------------|--------|---------|
| 1 | 47 | 0 | 0.000 | Low outlier | -1.94 |
| 8 | 215 | 31 | 0.144 | High outlier | 3.94 |
| 2 | 148 | 18 | 0.122 | Extreme | 2.22 |
| 11 | 256 | 29 | 0.113 | Extreme | 2.41 |

---

## Group Structure

**Low probability group (n=6)**
- Trials: 1, 4, 5, 6, 7, 12
- Mean proportion: 0.048

**High probability group (n=6)**
- Trials: 2, 3, 8, 9, 10, 11
- Mean proportion: 0.099

**Difference**: 0.051 (p = 0.012)

---

## Key Visualizations

1. **funnel_plot.png** - Shows overdispersion (points outside funnel)
2. **comprehensive_comparison.png** - 4-panel overview
3. **standardized_residuals.png** - Excess variation evident
4. **proportion_vs_trial.png** - No temporal pattern
5. **proportion_vs_sample_size.png** - No size bias

---

## Files Location

```
/workspace/eda/
├── eda_report.md          ← MAIN REPORT
├── eda_log.md             ← Detailed process
├── README.md              ← Overview
├── code/
│   └── 00_summary.py      ← Run this for quick summary
└── visualizations/         ← 8 plots
```

---

## Run Commands

```bash
# Quick summary
python /workspace/eda/code/00_summary.py

# Full analysis
python /workspace/eda/code/01_initial_exploration.py
python /workspace/eda/code/02_overdispersion_analysis.py
python /workspace/eda/code/03_visualization.py
python /workspace/eda/code/04_pattern_analysis.py
```

---

## Bottom Line

**The data exhibits strong overdispersion that cannot be explained by a simple binomial model. Use Beta-Binomial or mixture models. Expect about 3.5x more variance than a simple binomial model would predict. The pooled success rate is 7.4%, but there's substantial heterogeneity across trials.**
