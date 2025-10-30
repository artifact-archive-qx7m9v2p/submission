# Model Critique Summary: Experiment 3

**Model:** Latent AR(1) Negative Binomial
**Date:** 2025-10-29
**Decision:** **REJECT**
**Status:** Informative failure - architectural inadequacy demonstrated

---

## Quick Summary

Experiment 3 achieved **perfect computational convergence** (R̂=1.000, ESS>1100) and successfully estimated temporal correlation (ρ=0.84), but provided **zero improvement in predictive performance** compared to the simpler Experiment 1 baseline.

**Key Finding:** AR(1) on latent log-scale does NOT translate to observation-level temporal correlation due to nonlinear transformation and discrete sampling noise.

**Result:** Residual ACF(1) remained at 0.690 (vs 0.686 in Exp 1) - statistically unchanged.

---

## The Critical Evidence

| Metric | Target | Exp 1 (Baseline) | Exp 3 (AR1) | Improvement |
|--------|--------|------------------|-------------|-------------|
| **Residual ACF(1)** | **< 0.3** | **0.686** | **0.690** | **+0.6% (NONE)** |
| Coverage (95%) | 90-98% | 100.0% | 100.0% | 0.0% |
| R² | Improve | 0.883 | 0.861 | -0.022 (worse) |
| LOO-ELPD | > 2×SE | -174.17 | -169.32 | +4.85 ± 7.47 (weak) |
| Parameters | Simple | 4 | 46 | **11.5x increase** |
| Runtime | Fast | 10 min | 25 min | **2.5x slower** |

**Verdict:** Massive complexity increase (11x parameters, 2.5x time) for zero practical benefit.

---

## Why It Failed

The model specified:
```
α_t = β₀ + β₁·year + β₂·year² + ε_t
ε_t ~ AR(1) with ρ
μ_t = exp(α_t)
C_t ~ NegBinomial(μ_t, φ)
```

**The problem:** AR(1) correlation on log-scale (α_t) ≠ AR(1) on count-scale (C_t)

**Why:**
1. Nonlinear exp() transformation breaks correlation structure
2. Discrete NegBinomial sampling adds uncorrelated noise
3. Small innovations (σ_η=0.09) mean AR process contributes negligible variance
4. Most variation attributed to observation-level overdispersion (φ=20)

**Mathematical reality:** Even with ρ=0.84 on latent scale, observation ACF remains ~0.75 (far below observed 0.944).

---

## What We Learned

**Positive:**
- ✓ Temporal correlation definitively exists (ρ=0.84 clearly estimated)
- ✓ Latent-scale temporal structures are insufficient (architectural class ruled out)
- ✓ Computational methods work perfectly (implementation validated)
- ✓ Informative negative result (know what doesn't work)

**Negative:**
- ✗ Latent AR(1) does not reduce residual autocorrelation
- ✗ Added complexity provides no predictive benefit
- ✗ Coverage calibration worsened
- ✗ Point predictions degraded

---

## Documents in This Directory

### 1. `critique_summary.md` (30KB)
**Comprehensive synthesis** of all validation results:
- Prior predictive assessment
- Convergence diagnostics (perfect)
- Posterior predictive checks (failed)
- Model comparison (weak improvement)
- Detailed analysis of strengths and weaknesses
- Scientific interpretation
- Why complexity didn't help

**Read this for:** Complete understanding of model adequacy issues.

### 2. `decision.md` (15KB)
**Formal rejection** with clear justification:
- Decision framework applied (REJECT criteria met)
- Primary justification (architectural mismatch)
- Why REVISE is inappropriate (unfixable structural problem)
- Why ACCEPT is inappropriate (fails all critical metrics)
- Three recommended paths forward
- High confidence (95%+) in rejection

**Read this for:** Official decision and next steps.

### 3. `improvement_priorities.md` (18KB)
**Action plan** for moving forward:
- Priority 1: Decide if temporal modeling necessary
- Priority 2: Try observation-level conditional AR (Experiment 4)
- Priority 3: Alternative approaches if Exp 4 fails
- Priority 4: Accept Experiment 1 as adequate baseline
- Detailed specifications for Experiment 4
- Clear stopping criteria (max 1 more experiment)

**Read this for:** Practical guidance on what to do next.

---

## Recommendation

### Path Forward (Choose One)

**Option 1: One Final Attempt (Recommended)**
Try **observation-level conditional AR**:
```
C_t ~ NegBinomial(μ_t, φ)
log(μ_t) = β₀ + β₁·year + β₂·year² + γ·log(C_{t-1} + 1)
```

**Why:** Fundamentally different from Exp 3 (observation-scale, not latent-scale). If ACF(1) < 0.3, success. If not, stop.

**Timeline:** 1 week

---

**Option 2: Accept Experiment 1 (Fallback)**
If temporal modeling critical but Exp 4 fails, or if temporal modeling not critical:

**Experiment 1 is adequate for:**
- Estimating mean trend (β₀, β₁, β₂)
- Testing acceleration hypothesis (β₂ > 0?)
- Point predictions with conservative uncertainty
- Descriptive analysis

**Experiment 1 is NOT adequate for:**
- Temporal forecasting (predicting C_{t+1} from C_t)
- Mechanistic understanding of dynamics
- Precise uncertainty quantification

**With clear documentation of limitations** (ACF=0.686), Experiment 1 is scientifically defensible.

---

**Option 3: Try Different Mean Function**
Before giving up on temporal structure, test if residual ACF is artifact:

Try exponential or logistic growth instead of quadratic. Better mean function might eliminate apparent autocorrelation.

**Risk:** May not help if true temporal correlation exists.

---

## Stopping Rule

**Maximum 1 additional experiment.**

Evidence of diminishing returns after 2 failed attempts (Exp 1, Exp 3). If Exp 4 also fails to achieve ACF(1) < 0.3, accept Experiment 1.

**Timeline for decision:** 2 weeks

---

## Key Takeaways

1. **Computational success ≠ scientific success**
   - Perfect convergence (R̂=1.00) doesn't mean good model
   - Must validate with posterior predictive checks

2. **Architecture matters more than parameters**
   - Well-estimated ρ=0.84 is useless if at wrong scale
   - Structural problems cannot be fixed by tuning

3. **Complexity requires justification**
   - 11x more parameters demands proportional benefit
   - Occam's Razor: Simpler model wins when performance tied

4. **Negative results are valuable**
   - Ruling out model class is scientific progress
   - Knowing what doesn't work guides future efforts

5. **Know when to stop**
   - Diminishing returns after multiple failures
   - Accepting imperfection is sometimes correct choice

---

## Citation

When referencing this critique:

```
Experiment 3 (Latent AR(1) Negative Binomial) was rejected due to
architectural inadequacy. Despite perfect convergence diagnostics
(R̂=1.000, ESS>1100) and successful estimation of temporal correlation
(ρ=0.84 [0.69, 0.98]), the model failed to reduce residual autocorrelation
(ACF(1) remained at 0.690 vs 0.686 in baseline) or improve predictive
performance (ΔLOO-ELPD = 4.85 ± 7.47, weak evidence). The failure
demonstrates that latent-scale AR(1) structures cannot capture
observation-level temporal correlation in count data due to nonlinear
transformation and discrete sampling noise. This informative negative
result rules out latent temporal model classes for these data.
```

---

## Contact

**Analyst:** Claude (Model Criticism Specialist)
**Date:** 2025-10-29
**Status:** Complete - Awaiting decision on next steps

---

## Quick Navigation

- **Full critique:** `critique_summary.md` (comprehensive analysis)
- **Formal decision:** `decision.md` (REJECT + justification)
- **Next steps:** `improvement_priorities.md` (action plan)
- **This file:** Executive summary and quick reference

---

**Bottom Line:** Experiment 3 is scientifically inadequate despite computational excellence. Either try observation-level AR (one final attempt) or accept Experiment 1 as adequate baseline with documented limitations. Do not pursue further latent temporal structures.
