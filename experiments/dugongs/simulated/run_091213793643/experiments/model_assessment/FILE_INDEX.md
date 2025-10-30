# Model Assessment File Index

Quick reference for all generated files in the model assessment.

## Key Documents (Start Here)

1. **`README.md`** - Quick summary and getting started guide
2. **`assessment_report.md`** - Comprehensive 10-section assessment report (~8,000 words)
3. **`assessment_metrics.json`** - All numerical results in structured JSON format

## Visualizations (Publication Quality, 300 dpi)

All plots saved in `/workspace/experiments/model_assessment/plots/`

1. **`model_performance_summary.png`** (7 panels)
   - Overall model fit with multiple credible intervals
   - Residuals vs fitted values
   - Residual distribution and Q-Q plot
   - Parameter posteriors (α, β, σ)

2. **`loo_diagnostics_overview.png`** (4 panels)
   - Pareto k diagnostics by observation index
   - Pareto k vs predictor x
   - LOO predictions vs observed
   - LOO residuals vs predictor

3. **`calibration_analysis.png`** (2 panels)
   - Coverage calibration curve (50-99% levels)
   - LOO probability integral transform (LOO-PIT)

4. **`uncertainty_quantification.png`** (4 panels)
   - 95% interval widths vs x
   - Distribution of interval widths
   - Predictions with uncertainty bands
   - Posterior predictive distributions at selected x values

## Code

All code in `/workspace/experiments/model_assessment/code/`

1. **`comprehensive_assessment.py`** - Main assessment script
   - Loads data and posterior samples
   - Computes LOO-CV diagnostics
   - Analyzes calibration
   - Generates all visualizations
   - Saves metrics to JSON

2. **`fix_json.py`** - Utility for JSON type conversion (used internally)

## File Sizes

```
README.md                           ~8 KB
assessment_report.md               ~47 KB
assessment_metrics.json            ~2 KB
comprehensive_assessment.py        ~19 KB
model_performance_summary.png      ~545 KB
loo_diagnostics_overview.png       ~427 KB
calibration_analysis.png           ~841 KB
uncertainty_quantification.png     ~389 KB
```

## Absolute Paths

All files are in: `/workspace/experiments/model_assessment/`

To access from anywhere:
```bash
cd /workspace/experiments/model_assessment

# View main report
cat assessment_report.md

# View metrics
cat assessment_metrics.json

# Re-run assessment
python code/comprehensive_assessment.py
```

## Related Files (from Experiment 1)

- Model specification: `/workspace/experiments/experiment_1/metadata.md`
- Posterior samples: `/workspace/experiments/experiment_1/posterior_inference/diagnostics/posterior_inference.netcdf`
- Model critique: `/workspace/experiments/experiment_1/model_critique/decision.md`
- PPC findings: `/workspace/experiments/experiment_1/posterior_predictive_check/ppc_findings.md`
- Original data: `/workspace/data/data.csv`

## Quick Access Commands

```bash
# View summary
cat /workspace/experiments/model_assessment/README.md

# View full report
cat /workspace/experiments/model_assessment/assessment_report.md

# View metrics
cat /workspace/experiments/model_assessment/assessment_metrics.json | python -m json.tool

# List all plots
ls -lh /workspace/experiments/model_assessment/plots/

# Re-run assessment
python /workspace/experiments/model_assessment/code/comprehensive_assessment.py
```

---

Generated: 2025-10-28
