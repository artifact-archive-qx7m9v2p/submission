# Quick Model Summary - Designer 2

## Three Proposed Model Classes

### Model 1: Hierarchical Beta-Binomial
**Core idea**: Continuous variation in success probabilities via Beta distribution

```
θ_i ~ Beta(μκ, (1-μ)κ)
r_i ~ Binomial(n_i, θ_i)
```

**Key parameters**: μ (pooled probability), κ (concentration)
**Abandon if**: κ at boundary, posterior predictive fails, bimodal θ_i

---

### Model 2: Finite Mixture (2-3 Components)
**Core idea**: Discrete probability regimes (groups)

```
z_i ~ Categorical(π)
r_i | z_i=k ~ Binomial(n_i, p_k)
```

**Key parameters**: π (mixing proportions), p_k (component probabilities)
**Abandon if**: Components collapse, heavy overlap, uncertain assignments

---

### Model 3: Non-Centered Hierarchical Logit
**Core idea**: Trial-specific effects on logit scale with better HMC geometry

```
logit(θ_i) = μ_logit + σ*η_i
η_i ~ Normal(0, 1)
r_i ~ Binomial(n_i, θ_i)
```

**Key parameters**: μ_logit (population mean), σ (heterogeneity scale)
**Abandon if**: σ at boundary, non-normal η_i, computational pathologies

---

## Selection Strategy

1. **Fit all three models**
2. **Compare via LOO-CV** (primary criterion)
3. **Check posterior predictive performance** (can reproduce overdispersion?)
4. **Assess falsification criteria** (did any trigger?)
5. **Sensitivity analysis** (robust to prior choice and influential observations?)

**If models tie**: Use model averaging
**If all fail**: Consider Plan B (robust likelihood) or collect more data

---

## Key Falsification Mindset

- **I will abandon models that fail, not defend them**
- **Passing tests ≠ correct model, failing tests = wrong model**
- **Wide uncertainty is honest, not a failure**
- **12 observations may be insufficient to distinguish models**

---

## Expected Outcome

Most likely: Models 1 and 3 perform similarly, Model 2 slightly worse (overfitting risk).
Action: Model averaging or select based on interpretability + LOO.

---

See `/workspace/experiments/designer_2/proposed_models.md` for full details.
