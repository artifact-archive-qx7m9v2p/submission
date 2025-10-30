# Model Comparison: Experiment 1 vs Experiment 3

**Date**: 2025-10-30
**Status**: Complete
**Recommendation**: Use Experiment 3 (Beta-Binomial)

---

## Quick Summary

**Experiment 3 (Beta-Binomial) is recommended** based on:
- ✓ Equivalent predictive performance (ΔELPD = -1.5 ± 3.7, within 2×SE)
- ✓ Perfect LOO reliability (0/12 bad Pareto k vs 10/12 for Exp1)
- ✓ 7× simpler (2 vs 14 parameters)
- ✓ 15× faster (6 vs 90 seconds)
- ✓ Easier interpretation (probability vs logit scale)

---

## Directory Structure

```
model_comparison/
├── README.md                          # This file
├── comparison_report.md               # Comprehensive technical report (30+ pages)
├── recommendation.md                  # Executive summary and decision guidance
├── code/
│   └── final_comparison.py           # Complete comparison analysis script
├── plots/                            # Key visualizations
│   ├── comprehensive_comparison.png  # 4-panel dashboard (CRITICAL)
│   ├── model_trade_offs_spider.png   # Multi-criteria radar plot
│   └── pareto_k_detailed_comparison.png # Group-by-group LOO reliability
└── diagnostics/                      # Numerical results
    ├── comparison_metrics.csv        # Summary table
    ├── loo_comparison_table.csv      # Group-level LOO metrics
    └── comparison_summary.json       # Structured results
```

---

## Key Files

### Reports (Start Here)

1. **recommendation.md** - Executive recommendation and decision framework
   - Quick decision guide
   - When to use each model
   - One-page summary

2. **comparison_report.md** - Comprehensive technical analysis
   - Detailed LOO/WAIC comparison
   - Group-by-group Pareto k analysis
   - Multi-criteria assessment
   - Publication guidance

### Visualizations (Visual Evidence)

1. **comprehensive_comparison.png** - Most important plot
   - Panel A: ELPD comparison (models equivalent)
   - Panel B: Pareto k diagnostics (Exp3 dominates)
   - Panel C: Model complexity (Exp3 simpler/faster)
   - Panel D: Point-wise LOO scatter (Exp1's bad k highlighted)

2. **model_trade_offs_spider.png** - Multi-criteria radar plot
   - Shows Exp3 dominates in LOO reliability, simplicity, speed, parsimony
   - Predictive accuracy tied

3. **pareto_k_detailed_comparison.png** - Group-level reliability
   - Exp1: 10 red/dark red bars (bad k)
   - Exp3: 12 green bars (all good k)

### Data Files

1. **comparison_metrics.csv** - Summary table (13 metrics × 2 models)
2. **loo_comparison_table.csv** - Group-level LOO and Pareto k values
3. **comparison_summary.json** - Structured results for programmatic access

---

## Key Findings

### 1. Predictive Performance: EQUIVALENT

```
ΔELPD (Exp3 - Exp1): -1.51 ± 3.67 (0.41 × SE)
Decision: EQUIVALENT (|Δ| < 2×SE)
```

Both models have statistically indistinguishable predictive accuracy.

### 2. LOO Reliability: EXP3 DRAMATICALLY SUPERIOR

| Model | Pareto k > 0.7 | Pareto k > 1.0 | Status |
|-------|----------------|----------------|--------|
| Exp1 | 10/12 (83%) | 2/12 | UNRELIABLE |
| Exp3 | 0/12 (0%) | 0/12 | RELIABLE |

Exp3's LOO estimates are trustworthy; Exp1's are not.

### 3. Model Complexity: EXP3 MUCH SIMPLER

| Metric | Exp1 | Exp3 | Advantage |
|--------|------|------|-----------|
| Parameters | 14 | 2 | 7× simpler |
| p_loo | 8.27 | 0.61 | 13.5× more parsimonious |
| Sampling time | 90s | 6s | 15× faster |

### 4. Interpretability: EXP3 EASIER

- Exp1: Logit scale (requires transformation)
- Exp3: Probability scale (direct interpretation)

---

## Visual Evidence Summary

### Comprehensive Comparison (`comprehensive_comparison.png`)

**Panel B (Pareto k)** is the most important:
- Shows 10 red bars (Exp1) vs 0 red bars (Exp3)
- Clear visual evidence of Exp3's superiority in LOO reliability

**Panel A (ELPD)** shows equivalence:
- Error bars overlap substantially
- Difference (1.5 points) is small relative to uncertainty (3.7 SE)

**Interpretation**: Exp3 matches Exp1 in predictive accuracy while being far more reliable.

### Spider Plot (`model_trade_offs_spider.png`)

Shows Exp3 (green) dominates in 4/5 dimensions:
- LOO Reliability: 10/10 vs 1.7/10
- Simplicity: 10/10 vs 1.4/10
- Speed: 10/10 vs 0.7/10
- Parsimony: 10/10 vs 1.0/10
- Predictive Accuracy: ~5/10 vs ~5/10 (tied)

**Interpretation**: Exp3 is superior across almost all evaluation criteria.

---

## When to Use Each Model

### Use Exp3 (Beta-Binomial) - RECOMMENDED

✓ Research question is population-level ("What is the overall success rate?")
✓ Need reliable model comparison (LOO-CV)
✓ Want simple, fast, interpretable analysis
✓ Publication robustness important (no caveats)
✓ Non-technical audience

### Use Exp1 (Hierarchical) - ONLY IF NECESSARY

✓ Need group-specific rate estimates ("What is Group 4's rate?")
✓ Want explicit heterogeneity quantification (τ)
✓ Interested in shrinkage patterns
✓ Can document LOO limitations

---

## Comparison Metrics Summary

| Criterion | Exp1 | Exp3 | Winner |
|-----------|------|------|--------|
| ELPD LOO | -38.76 ± 2.94 | -40.28 ± 2.19 | Equivalent |
| Pareto k > 0.7 | 10/12 | 0/12 | **Exp3** |
| Parameters | 14 | 2 | **Exp3** |
| p_loo | 8.27 | 0.61 | **Exp3** |
| Sampling time | 90s | 6s | **Exp3** |
| PPC tests passed | 4/5 | 5/5 | **Exp3** |
| Interpretability | Logit | Probability | **Exp3** |
| Group estimates | Yes | No | Exp1 (if needed) |

**Overall**: Exp3 wins or ties on 7/8 criteria.

---

## Reproducing the Analysis

```bash
cd /workspace/experiments/model_comparison
python3 code/final_comparison.py
```

**Runtime**: ~10 seconds
**Output**: All plots and diagnostics regenerated

**Requirements**:
- Python 3.13
- ArviZ 0.20+
- NumPy, Pandas, Matplotlib, Seaborn

---

## Citation

If using these results in a publication:

**Preferred (Exp3)**:
> "We compared a hierarchical binomial model and a Beta-Binomial model using leave-one-out cross-validation. Both models showed equivalent predictive performance (ΔELPD = -1.5 ± 3.7, within 2 standard errors). However, the Beta-Binomial model demonstrated superior LOO reliability (0/12 vs 10/12 groups with Pareto k > 0.7) while being substantially simpler (2 vs 14 parameters). Therefore, we selected the Beta-Binomial model for inference."

**If Using Exp1**:
> "We fit a Bayesian hierarchical binomial model to estimate group-specific success rates. The model showed excellent convergence (R̂ = 1.00, ESS > 2400) and passed posterior predictive checks (4/5 tests). However, leave-one-out cross-validation diagnostics indicated high Pareto k values (k > 0.7) for 10 of 12 groups, suggesting sensitivity to individual observations. Therefore, LOO-CV was not used for model comparison, and adequacy was assessed via posterior predictive checks."

---

## Key Takeaway

**Both models fit the data well.** The choice depends on research goals:

- **Population-level inference** → Use Exp3 (simpler, reliable, faster)
- **Group-specific inference** → Use Exp1 (despite LOO issues)

For most applications, Exp3's combination of simplicity, reliability, and adequate performance makes it the superior choice.

---

## Questions?

See detailed analysis in:
- `comparison_report.md` - Technical details
- `recommendation.md` - Decision guidance

Generated by: Model Assessment Specialist (Claude Agent SDK)
Date: 2025-10-30
