# Quick Reference: Three Competing Parametric Models

## Model 1: Logarithmic (Primary)
**Form**: `Y = α + β·log(x) + ε`
**Hypothesis**: Unbounded slow growth (Weber-Fechner law)
**Abandon if**: LOO worse than MM by >4, or residuals show systematic curvature at high x

## Model 2: Michaelis-Menten (Alternative)
**Form**: `Y = Y_max - (Y_max - Y_min)·K/(K + x) + ε`
**Hypothesis**: True asymptotic saturation
**Abandon if**: Y_max posterior unbounded, or K > 25 (no saturation observed yet)

## Model 3: Quadratic (Baseline)
**Form**: `Y = α + β₁·x + β₂·x² + ε`
**Hypothesis**: Polynomial approximation (local only)
**Abandon if**: Vertex at x < 31.5 (predicts downturn in data range)

---

## Key Falsification Criteria Summary

| Model | Primary Failure Mode | Decision Threshold |
|-------|---------------------|-------------------|
| Log | Systematic residual pattern at high x | PPC p-value < 0.05 for max(Y) statistic |
| MM | Y_max not identifiable | Posterior SD(Y_max) > 1.0 or mean > 5 |
| Quad | Vertex inside data range | P(vertex_x < 31.5) > 0.1 |
| All | Influential points dominating | LOO Pareto k > 0.7 for >5 observations |

---

## Expected Outcome (Most Likely)

**Winner**: Logarithmic model
- LOO-ELPD ≈ -3 to +3
- Posterior: α≈1.75±0.1, β≈0.27±0.05, σ≈0.12±0.02
- Interpretation: Y increases logarithmically; no evidence of true asymptote

**Runner-up**: Michaelis-Menten
- LOO-ELPD ≈ -5 to -2 (slightly worse)
- Weak identifiability of Y_max (wide posterior)
- Data insufficient to distinguish bounded vs unbounded growth

**Baseline**: Quadratic
- LOO-ELPD ≈ 0 to +2 (slightly better empirically)
- Vertex concerns prevent use for extrapolation
- Use only for interpolation if chosen

---

## Red Flags Requiring Model Class Change

1. **All models fail PPC**: Need robust likelihood (Student-t) or heteroscedastic variance
2. **High Pareto k across models**: Influential points; need sensitivity analysis or robust methods
3. **Gap region shows deviation**: Consider piecewise or change-point models
4. **Divergent transitions in MM**: Reparameterization needed or model inappropriate

---

## Priority Order for Implementation

1. **Logarithmic** (fit first) - most likely winner
2. **Michaelis-Menten** (fit second) - tests key hypothesis
3. **Quadratic** (fit third) - baseline comparison

Skip quadratic if first two are adequate and distinguishable.

---

## File Locations

- **Full proposal**: `/workspace/experiments/designer_1/proposed_models.md`
- **This summary**: `/workspace/experiments/designer_1/model_summary.md`
- **Stan models**: To be created in `stan_models/` subdirectory
- **Results**: To be saved in `results/` subdirectory
