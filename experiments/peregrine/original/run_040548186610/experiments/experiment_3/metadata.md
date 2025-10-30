# Experiment 3: Negative Binomial with Latent AR(1) Process

**Date:** 2025-10-29
**Status:** In Progress
**Model Class:** State-Space / Temporal

---

## Model Specification

### Observation Model
```
C_t ~ NegativeBinomial(μ_t, φ)
log(μ_t) = α_t
```

### Latent State Process (AR(1) on log-scale)
```
α_t = β₀ + β₁·year_t + β₂·year_t² + ε_t
ε_t = ρ·ε_{t-1} + η_t
η_t ~ Normal(0, σ_η)
ε_1 ~ Normal(0, σ_η/√(1-ρ²))  # Stationary initial condition
```

### Parameter Interpretation
- `β₀, β₁, β₂`: Deterministic trend parameters (as in Experiment 1)
- `ρ`: Temporal autocorrelation coefficient (0 = independent, 1 = perfect persistence)
- `σ_η`: Innovation standard deviation (controls temporal variability)
- `φ`: Negative binomial dispersion parameter
- `ε_t`: Latent AR(1) deviations from trend

---

## Prior Distributions

```
# Trend parameters (same as Experiment 1)
β₀ ~ Normal(4.7, 0.3)
β₁ ~ Normal(0.8, 0.2)
β₂ ~ Normal(0.3, 0.1)

# Temporal correlation (informative based on residuals from Exp 1)
ρ ~ Beta(12, 3)              # Mean=0.8, favors high correlation
                             # Based on Exp 1 residual ACF(1)=0.686

# Innovation SD
σ_η ~ HalfNormal(0, 0.5)     # Small innovations maintain smoothness

# Overdispersion
φ ~ Gamma(2, 0.5)             # Same as Experiment 1
```

### Prior Justification

**Trend parameters (β₀, β₁, β₂):**
- Reuse Experiment 1 priors (already validated)
- These capture the deterministic component

**ρ ~ Beta(12, 3):**
- **Informative prior** based on empirical evidence
- Experiment 1 residual ACF(1) = 0.686 suggests high ρ
- Beta(12, 3) has mean=0.8, SD=0.1, mode=0.82
- Still allows data to pull ρ lower if correlation is spurious
- **Key diagnostic:** If posterior ρ << 0.8, autocorrelation was partly trend artifact

**σ_η ~ HalfNormal(0, 0.5):**
- Small innovations keep latent process smooth
- On log-scale, 0.5 SD → exp(±0.5) ≈ 0.6-1.6× multiplier
- Allows moderate temporal fluctuations

**φ ~ Gamma(2, 0.5):**
- Same as Experiment 1 (overdispersion handling unchanged)

---

## Rationale

### Why This Model?

**Problem identified in Experiment 1:**
- Residual ACF(1) = 0.686 (well above 0.5 threshold)
- Systematic temporal wave pattern
- Excessive coverage (100% vs target 90-98%)

**This model addresses the problem:**
1. **Explicit temporal structure:** AR(1) process models correlation directly
2. **Separates trend from correlation:** β parameters for deterministic trend, ρ for stochastic correlation
3. **Maintains count structure:** Negative binomial handles overdispersion
4. **Maintains good features:** Keeps successful quadratic trend from Experiment 1

### Expected Strengths
- Captures temporal autocorrelation explicitly
- Reduces residual ACF(1) from 0.686 → <0.3 (target)
- Improves coverage from 100% → 90-98%
- Maintains interpretable trend parameters
- Should substantially improve LOO-ELPD vs Experiment 1

### Expected Weaknesses
- **High complexity:** 7 parameters + N latent states (47 total unknowns)
- **Identifiability:** β₂ vs ρ may be confounded (both create smooth curves)
- **Computational:** Non-centered parameterization required, may have divergences
- **Overfitting risk:** n=40 may be small for this complexity

---

## Success Criteria

### Convergence
- R̂ < 1.01 for all parameters
- ESS > 400 for all parameters (including latent states summary)
- Divergent transitions < 5% (relaxed from 1% due to complexity)

### Predictive Performance
- Residual ACF(1) < 0.3 (critical improvement over Exp 1's 0.686)
- Coverage: 90-98% (improved from Exp 1's 100%)
- LOO-ELPD substantially better than Experiment 1 (ΔELPD > 5)
- No extreme Pareto-k values (< 5% with k > 0.7)

### Parameter Reasonableness
- ρ ∈ [0.3, 0.95] (not too weak, not unity)
- σ_η reasonable (not near zero, not huge)
- Trend parameters similar to Experiment 1 (should be stable)

---

## Failure Criteria (REJECT if 2+ occur)

1. **Convergence failure:** R̂ > 1.05 despite extensive tuning
2. **No temporal improvement:** Residual ACF(1) > 0.5 still
3. **Identifiability issues:** ρ and β₂ highly correlated (|r| > 0.9), wide posteriors
4. **Prior-posterior conflict:** Posterior ρ peaks at boundary (0 or 1)
5. **Worse LOO:** LOO-ELPD not better than Experiment 1 (complexity not justified)

---

## Implementation Strategy

### Computational Approach
**Primary:** PyMC (Stan showed compiler issues in Experiment 1)
- Use non-centered parameterization for AR(1) errors
- Start with adapt_delta = 0.95
- If divergences: increase to 0.98, increase target_accept

### Non-Centered Parameterization
```python
# Instead of: ε_t = ρ·ε_{t-1} + η_t
# Use: ε_t = ρ·ε_{t-1} + σ_η·η_raw_t, where η_raw_t ~ Normal(0, 1)
# This decorrelates ε from σ_η for better sampling
```

### Sampling Strategy
- 4 chains × 3000 iterations (1500 warmup, 1500 sampling)
- Higher iteration count than Exp 1 due to complexity
- Monitor convergence closely

---

## Falsification Tests

### Prior Predictive Check
- Do priors generate realistic temporal patterns?
- Check that ρ ~ Beta(12, 3) creates appropriate autocorrelation
- Verify σ_η allows smooth but non-deterministic trajectories

### Simulation-Based Calibration
- Critical test: Can we recover ρ reliably?
- Check for identifiability between β₂ and ρ
- May need reduced simulations (M=20-50) due to computational cost

### Posterior Predictive Check
- **Primary goal:** Residual ACF(1) < 0.3
- Secondary: Coverage 90-98%, no systematic patterns
- Temporal structure should be captured

### Model Comparison
- LOO-CV vs Experiment 1 (must be substantially better)
- If ΔELPD < 5: complexity not justified, revert to simpler model

---

## Expected Outcome

### Most Likely Scenario (60% probability)
- Model converges with 2-8% divergences (manageable)
- ρ posterior: mean ≈ 0.7, 95% CI [0.5, 0.85]
- Residual ACF(1): 0.15-0.25 (substantial improvement)
- LOO-ELPD improvement: +10 to +20 points
- **Decision:** ACCEPT (addresses temporal issue successfully)

### Alternative Scenario 1 (25% probability)
- Convergence difficult (>10% divergences despite tuning)
- ρ and β₂ highly correlated (identifiability issues)
- **Decision:** REVISE - simplify to linear trend (β₂=0) or different temporal structure

### Alternative Scenario 2 (15% probability)
- Model converges but ρ posterior: mean ≈ 0.3-0.4 (lower than expected)
- Moderate ACF reduction: 0.686 → 0.4-0.5 (not sufficient)
- **Decision:** ESCALATE - need more complex temporal structure (dynamic level + trend)

---

## Comparison to Experiment 1

| Aspect | Experiment 1 | Experiment 3 |
|--------|--------------|--------------|
| Trend | Quadratic | Quadratic (same) |
| Temporal structure | None (i.i.d.) | AR(1) latent process |
| Parameters | 4 | 7 + N latent states |
| Residual ACF(1) | 0.686 | Target: <0.3 |
| Coverage | 100% | Target: 90-98% |
| Complexity | Simple | Moderate |
| Expected LOO | Baseline | +10 to +20 |

---

## Next Steps

1. ✅ Metadata created
2. ⏳ Prior predictive check
3. ⏳ Simulation-based validation (may be abbreviated due to cost)
4. ⏳ Posterior inference
5. ⏳ Posterior predictive check (focus on temporal diagnostics)
6. ⏳ Model critique

---

## Notes

This model directly addresses the fundamental failure of Experiment 1 (temporal independence violation). Success means residual ACF drops below 0.3 and LOO improves substantially. Failure means we need even more complex temporal structure (state-space with time-varying level and trend) or different modeling approach entirely.

**Key scientific question:** Is the observed autocorrelation (0.989 in data, 0.686 in Exp 1 residuals) primarily a property of the process itself, or an artifact of incorrect trend specification? If ρ posterior is high (>0.6), it's a process property. If low (<0.4), it's partly a trend artifact.
