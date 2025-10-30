# Quick Model Summary - Designer 1

## Model Rankings

### 1. Negative Binomial with Time-Varying Dispersion (PRIMARY)
**Why:** Directly addresses 68× overdispersion and 26× variance increase
**When to abandon:** θ < 0.5 or θ > 100, or variance structure fails stress tests
**LOO comparison with:** Constant dispersion variant

### 2. Negative Binomial with Constant Dispersion (BASELINE)
**Why:** Simpler alternative, NB inherently handles heteroscedasticity
**When to abandon:** ΔLOO > 10 compared to time-varying version
**LOO comparison with:** Time-varying variant

### 3. Random Walk State-Space (FALLBACK)
**Why:** If overdispersion comes from latent heterogeneity not intrinsic variance
**When to abandon:** σ_η → 0, or worse LOO than NB models
**LOO comparison with:** Best NB model

---

## Key Parameters to Monitor

### Model 1a (Time-varying NB):
- **θ_early vs θ_late:** Should differ by 2-5× if time-varying is needed
- **γ₁:** Should be < 0 (decreasing dispersion)
- **β₂:** Should be ≠ 0 (nonlinear trend)

### Model 2a (Random walk):
- **σ_η:** Should be 0.2-0.5 to explain overdispersion
- **ACF(η_t):** Should decay slowly (random walk structure)

---

## Stress Tests (ALL models must pass)

1. **Variance by quintile:** Posterior predictive variance matches empirical ±20%
2. **Extreme value coverage:** 90% intervals cover 80-95% of top/bottom decile
3. **Sequential prediction:** RMSE on last 10 observations < 80

---

## Red Flags → STOP and Pivot

- Posterior predictive variance systematically wrong
- Divergent transitions > 20%
- LOO Pareto k > 0.7 for many points
- Parameters hit prior boundaries
- R-hat > 1.05 for any parameter

---

## Model Code Files

- `model_1a_timevarying_nb.stan`: Primary model
- `model_1b_constant_nb.stan`: Baseline
- `model_2a_random_walk.stan`: Fallback
- `fit_and_compare.py`: Fitting script with LOO comparison

---

## Expected Timeline

**Day 1-2:** Fit Models 1a, 1b → Select winner
**Day 3-4:** Posterior predictive checks, stress tests
**Day 5:** Fit Model 2a if needed
**Day 6-7:** Sensitivity analysis, final validation
