# Designer 3: Temporal Autocorrelation Models

**Focus:** Explicit modeling of temporal dependence in count time series

## Quick Summary

### Key Question
Is the extreme autocorrelation (lag-1 = 0.989) **real** or **spurious** (artifact of smooth trending)?

### Proposed Models
1. **Model T1:** Negative Binomial with Latent AR(1) Process (state-space)
2. **Model T2:** Poisson Dynamic Linear Model (time-varying level & trend)
3. **Model T3:** Count AR(1) with Overdispersed Innovations (direct temporal dependence)

### Primary Hypothesis
**The autocorrelation is spurious** - a well-specified trend model will produce independent residuals.

### Falsification Criteria
- If ρ (autocorrelation parameter) posterior includes 0 → temporal structure not needed
- If LOO-IC is worse than Designer 1's models → simpler models win
- If ACF of residuals from quadratic NB < 0.3 → trending explains everything

### Expected Outcome
Most likely: **Model T3 finds ρ ≈ 0**, confirming that Designer 1's quadratic Negative Binomial model (without explicit temporal structure) is sufficient.

Alternative: If ρ > 0.5 with strong evidence, temporal models are necessary.

## Files
- `proposed_models.md` - Complete model specifications with Stan code
- `/workspace/data/data.csv` - Original data (40 observations)
- `/workspace/eda/eda_report.md` - EDA findings

## Implementation Priority
1. Start with Model T3 (simplest)
2. Only fit T1/T2 if T3 shows strong temporal dependence
3. Always compare to Designer 1's non-temporal models

## Red Flags to Watch
- Divergent transitions >5%
- Prior-posterior ρ overlap >80%
- LOO-IC worse than simpler models
- Computational instabilities

---

**Designer:** 3 (Temporal Structure Specialist)
**Date:** 2025-10-29
