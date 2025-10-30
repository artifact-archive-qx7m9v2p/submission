# Quick Reference: Designer 1 Bayesian Models
**Parsimonious & Interpretable Approach**

---

## TL;DR

**Recommended order:** Log → Power → Asymptotic (only if needed)

**Start here:** Logarithmic model (2 parameters, simplest, R²=0.83 from EDA)

**Stop criteria:**
- ✓ R-hat < 1.01, ESS > 400
- ✓ Residuals normal, no patterns
- ✓ 90%+ PPC coverage
- ✓ LOO-CV RMSE < 0.25

**Red flags (abandon model):**
- ✗ Divergent transitions
- ✗ Systematic residual patterns
- ✗ >20% obs outside 95% CI
- ✗ β₁ ≤ 0 (wrong direction)

---

## Three Models at a Glance

### Model 1: Logarithmic (PRIMARY)

```
Y ~ Normal(μ, σ)
μ = β₀ + β₁·log(x)

Priors:
β₀ ~ Normal(1.73, 0.5)
β₁ ~ Normal(0.28, 0.15)
σ ~ Exponential(5)

Pros: Simple, interpretable, fast
Cons: Unbounded growth
```

**When to use:** Default choice, EDA-supported
**When to abandon:** Residuals show clear quadratic pattern

---

### Model 2: Power Law (ALTERNATIVE)

```
Y ~ Normal(μ, σ)
μ = β₀ + β₁·x^β₂

Priors:
β₀ ~ Normal(1.8, 0.5)
β₁ ~ Normal(0.5, 0.3)
β₂ ~ Normal(0.3, 0.2)
σ ~ Exponential(5)

Pros: Flexible, nests linear
Cons: 3 params, correlation, slower
```

**When to use:** Log model fails residual checks
**When to abandon:** No LOO improvement, divergences persist

---

### Model 3: Asymptotic (THEORETICAL)

```
Y ~ Normal(μ, σ)
μ = Y_min + Y_range·x/(K + x)

Priors:
Y_min ~ Normal(1.7, 0.3)
Y_range ~ Normal(0.9, 0.3)
K ~ LogNormal(log(5), 1)
σ ~ Exponential(5)

Pros: Bounded, interpretable max
Cons: Complex, nonlinear, slow
```

**When to use:** Evidence of saturation, unbounded growth implausible
**When to abandon:** K unbounded, computational failure

---

## Decision Flowchart

```
START
  |
  V
Fit Log Model (Model 1)
  |
  +-- Converged? (R-hat < 1.01)
  |     |
  |     NO --> Check priors/data, refit
  |     YES
  |       |
  |       V
  +-- Residuals OK? (no patterns, normal)
  |     |
  |     YES --> PPC good? (>90% coverage)
  |     |         |
  |     |         YES --> LOO RMSE < 0.25?
  |     |         |         |
  |     |         |         YES --> DONE (use Log)
  |     |         |         NO --> Investigate outliers
  |     |         |
  |     |         NO --> Check likelihood (try Student-t?)
  |     |
  |     NO (systematic pattern)
  |       |
  |       V
  +-- Fit Power Model (Model 2)
        |
        Compare LOO: ΔELPD > 3?
        |
        YES --> Use Power
        NO --> Stick with Log (simpler)
```

---

## Failure Criteria Summary

### Computational Red Flags
- Divergent transitions (>1%)
- R-hat > 1.05
- ESS < 400
- Bimodal posteriors

### Statistical Red Flags
- Systematic residual patterns
- <80% PPC coverage
- >20% obs with Pareto k > 0.7
- Prior-posterior conflict

### Scientific Red Flags
- β₁ ≤ 0 (wrong direction)
- σ > 0.4 (R² < 0.3)
- Absurd extrapolations
- Parameters nonsensical

**Any of these → pivot to next model or alternative approach**

---

## Key Parameters Interpretation

### Model 1 (Log)
- **β₀:** Y when log(x)=0 (i.e., x=1)
- **β₁:** Elasticity (% change in Y per % change in x)
- **σ:** Residual SD (unexplained variation)

### Model 2 (Power)
- **β₀:** Baseline Y
- **β₁:** Scale factor
- **β₂:** Curvature (< 1 → concave)
- **σ:** Residual SD

### Model 3 (Asymptotic)
- **Y_min:** Baseline response (x→0)
- **Y_max:** Maximum response (x→∞)
- **K:** Half-max constant (smaller = faster saturation)
- **σ:** Residual SD

---

## Critical Checks Checklist

### Before declaring success:
- [ ] R-hat < 1.01 for all parameters
- [ ] ESS > 400 (preferably >1000)
- [ ] Trace plots show good mixing
- [ ] No divergent transitions
- [ ] Residuals approximately normal (Shapiro p > 0.05)
- [ ] Residual plot shows no patterns
- [ ] PPC: >90% obs in 95% CI
- [ ] LOO: <20% with Pareto k > 0.7
- [ ] Parameters scientifically plausible
- [ ] Extrapolation behavior reasonable

### Before abandoning model:
- [ ] Tried extended tuning (tune=2000+)
- [ ] Tried higher target_accept (0.99)
- [ ] Checked prior specification
- [ ] Verified data quality
- [ ] Attempted reparameterization
- [ ] Consulted diagnostics (trace, pairplot)

---

## Expected Timeline

**Model 1 (Log):** 1-2 days
- Day 1: Fit, diagnose, validate
- Day 2: PPC, LOO, visualizations

**Model 2 (Power):** +1-2 days (if needed)
- Extended sampling/tuning
- Comparison to Model 1

**Model 3 (Asymptotic):** +2-3 days (if needed)
- Complex sampling
- Full three-way comparison

**Total:** 2-7 days depending on how quickly Model 1 succeeds

---

## Common Pitfalls

1. **Overfitting:** N=27 is small, avoid complex models
2. **Extrapolation overconfidence:** Only 3 obs with x>20
3. **Ignoring divergences:** Always investigate, don't just increase target_accept
4. **Prior-data conflict:** If posterior ≈ prior, data not informative
5. **Multiple comparisons:** Don't fit 10 models, stick to 2-3
6. **Confirmation bias:** Actively try to BREAK models, not confirm

---

## Files Generated

**Main outputs:**
- `/workspace/experiments/designer_1/proposed_models.md` (full specification)
- `/workspace/experiments/designer_1/implementation_guide.md` (code templates)
- `/workspace/experiments/designer_1/QUICK_REFERENCE.md` (this file)

**Expected from implementation:**
- `trace_plot_log.png` (convergence)
- `residuals_log.png` (diagnostics)
- `ppc_log.png` (posterior predictive checks)
- `predictions_log.png` (fitted curve)
- `pareto_k_log.png` (LOO diagnostics)
- `model_comparison_loo.png` (if multiple models)
- `*_report.txt` (text summaries)

---

## Contact Points with Other Designers

**If parallel designers exist:**
- Compare LOO-CV across all designs
- Look for consensus on functional form
- Identify where designs diverge (usually complexity vs parsimony)
- Best model may be from another designer!

**Philosophy differences:**
- Designer 1 (me): Parsimony, interpretability
- Designer 2 (?): Might focus on flexibility, splines, GPs
- Designer 3 (?): Might focus on robustness, hierarchical structure

**Synthesis:** Choose based on LOO-CV + interpretability + scientific plausibility

---

## Emergency Escape Routes

**If all 3 models fail:**

1. **Robust likelihood:** Student-t instead of Normal
   ```
   Y ~ StudentT(ν, μ, σ)
   ν ~ Gamma(2, 0.1)  # Estimated df
   ```

2. **Heteroscedastic variance:**
   ```
   σ_i = σ₀ · exp(γ·x_i)
   ```

3. **Gaussian Process:** Non-parametric
   ```
   f ~ GP(0, k(x, x'))
   Y ~ Normal(f, σ)
   ```

4. **Mixture model:** Two regimes?
   ```
   μ_i = w·μ₁(x) + (1-w)·μ₂(x)
   ```

5. **Question data:** Measurement error? Confounders?

---

## Final Advice

**Mindset:**
- Goal is truth, not task completion
- Failing fast = success (learn quickly)
- Simplest adequate model wins
- Interpretability > 0.01 ELPD gain

**Workflow:**
1. Fit simplest model (Log)
2. Validate thoroughly
3. Only add complexity if justified
4. Stop when adequate, not perfect

**Remember:**
- N=27 is modest
- x>20 is sparse (3 obs)
- Normal likelihood justified (EDA)
- Constant variance reasonable (EDA)

**Good luck! Start with Model 1 and be ready to stop there.**

---

*Designer 1 - Parsimonious & Interpretable*
