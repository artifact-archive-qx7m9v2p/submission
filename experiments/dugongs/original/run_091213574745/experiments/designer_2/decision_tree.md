# Decision Tree: Flexible Bayesian Model Selection

```
START: Fit Model 1 (GP with Matern 3/2)
│
├─ Converges well (R̂<1.01, ESS>400, divg<1%)?
│  ├─ YES → Compute LOO-CV
│  │        │
│  │        ├─ LOO better than log baseline (ΔLOO>2)?
│  │        │  ├─ YES → CHECKPOINT: GP is promising
│  │        │  │        ├─ Fit Model 2 (P-splines) for comparison
│  │        │  │        │
│  │        │  │        └─ Model 2 converges & comparable LOO?
│  │        │  │           ├─ YES → Compare both, choose based on:
│  │        │  │           │        - Posterior predictive checks
│  │        │  │           │        - Interpretability
│  │        │  │           │        - Scientific plausibility
│  │        │  │           │        DECISION: Report best flexible model
│  │        │  │           │
│  │        │  │           └─ NO → Model 2 fails
│  │        │  │                   DECISION: Use Model 1 (GP)
│  │        │  │
│  │        │  └─ NO → GP no better than baseline
│  │        │         ├─ Fit Model 2 (P-splines) anyway
│  │        │         │   ├─ Model 2 beats baseline?
│  │        │         │   │  ├─ YES → DECISION: Use Model 2
│  │        │         │   │  └─ NO → ABANDON flexible approaches
│  │        │         │   │          DECISION: Use parametric log model
│  │        │         │   │          Write "Why Flexibility Failed"
│  │        │         │   │
│  │        │         │   └─ Model 2 also fails convergence?
│  │        │         │      DECISION: n=27 too small for flexibility
│  │        │         │                Recommend parametric models
│  │        │         │
│  │        │         └─ Does GP suggest regime change?
│  │        │             (Sharp derivative drop, local clustering)
│  │        │             ├─ YES → Consider Model 3 (Adaptive GP)
│  │        │             │        But only if Models 1-2 show promise
│  │        │             │
│  │        │             └─ NO → Skip Model 3, accept smooth model
│  │        │
│  │        └─ LOO unstable (high k̂ values)?
│  │           ├─ Due to x=31.5 outlier?
│  │           │  ├─ YES → Try Student-t likelihood (Escape Route 1)
│  │           │  │        Refit and reassess
│  │           │  │
│  │           │  └─ NO → Check for other influential points
│  │           │           May indicate model misspecification
│  │           │
│  │           └─ Multiple high k̂?
│  │              DECISION: Model not capturing data structure
│  │                        Try Model 2 or pivot to parametric
│  │
│  └─ NO → Convergence issues (divergences, low ESS, high R̂)
│          │
│          ├─ Try tuning:
│          │  - Increase target_accept to 0.95-0.99
│          │  - More iterations (tune=3000, sample=3000)
│          │  - Wider priors (less informative)
│          │  - Different kernel (try Matern 1/2 or 5/2)
│          │
│          ├─ Still fails after tuning?
│          │  └─ ABANDON Model 1
│          │     ├─ Fit Model 2 (P-splines, simpler)
│          │     │  └─ Model 2 converges?
│          │     │     ├─ YES → DECISION: Use Model 2
│          │     │     └─ NO → ABANDON all flexible models
│          │     │               DECISION: Use parametric approaches
│          │     │
│          │     └─ Investigate why GP failed:
│          │        - Prior-data conflict?
│          │        - Hyperparameter identifiability?
│          │        - Data too sparse for GP?
│          │        Document in report
│          │
│          └─ Specific issue: Length scale ℓ hits prior bounds?
│             ├─ ℓ → 0: Model wants extreme locality
│             │         → Data may have discontinuity
│             │         → Try Model 3 (piecewise GP) or changepoint model
│             │
│             └─ ℓ → ∞: Model wants pure linear trend
│                       → GP overkill, use simple parametric
│                       → DECISION: Recommend logarithmic model


CHECKPOINT: After Models 1 & 2
│
├─ At least one flexible model works well?
│  ├─ YES → Proceed to validation
│  │        │
│  │        ├─ Posterior Predictive Checks
│  │        │  ├─ Monotonic? (must be YES)
│  │        │  ├─ Asymptotic? (must be YES)
│  │        │  ├─ Captures variance correctly? (check)
│  │        │  ├─ No systematic residual patterns? (check)
│  │        │  │
│  │        │  └─ All checks pass?
│  │        │     ├─ YES → Proceed to sensitivity analysis
│  │        │     └─ NO → Investigate failures
│  │        │               - Wrong likelihood? → Try Student-t
│  │        │               - Heteroscedastic? → Try variance model
│  │        │               - Wrong mean structure? → Reconsider
│  │        │
│  │        ├─ Sensitivity Analysis: Remove x=31.5
│  │        │  ├─ Results similar?
│  │        │  │  ├─ YES → Model is robust, PROCEED
│  │        │  │  └─ NO → Model overly influenced by outlier
│  │        │  │           → Try Student-t or document limitation
│  │        │  │
│  │        │  └─ Conclusions change substantially?
│  │        │     DECISION: Flag uncertainty, recommend caution
│  │        │
│  │        └─ Final Comparison
│  │           ├─ ΔLOO > 4 for one model?
│  │           │  ├─ YES → Clear winner, DECISION: Report it
│  │           │  └─ NO → Models within 2 SE
│  │           │           DECISION: Model averaging OR
│  │           │                     Choose based on interpretability
│  │           │
│  │           └─ All models agree on key features?
│  │              ├─ YES → Strong evidence, confident report
│  │              └─ NO → Models disagree fundamentally
│  │                       DECISION: Report uncertainty
│  │                                 Recommend more data
│  │
│  └─ NO → Both Models 1 & 2 fail
│           │
│           └─ DECISION: Flexible approaches not suitable for n=27
│              - Document why (overfitting, poor convergence, etc.)
│              - Recommend parametric models from Designer 1
│              - Write "Why Flexibility Failed with n=27"
│              - Suggest minimum sample size for flexible modeling


OPTIONAL: Model 3 (Adaptive GP with Changepoint)
│
├─ Only fit if Models 1-2 suggest regime change AND
│  at least one flexible model already successful
│
├─ Attempt Model 3 fit
│  ├─ Converges? (expect difficulties)
│  │  ├─ YES → Amazing! Proceed with comparison
│  │  │        │
│  │  │        ├─ τ posterior concentrated?
│  │  │        │  ├─ YES (e.g., 95% CI = [6.2, 7.8])
│  │  │        │  │  → Strong evidence for changepoint at x≈7
│  │  │        │  │
│  │  │        │  └─ NO (e.g., 95% CI = [4.5, 11.2])
│  │  │        │     → Changepoint location uncertain
│  │  │        │        Model may be overparameterized
│  │  │        │
│  │  │        ├─ ℓ₁ ≠ ℓ₂ in posterior?
│  │  │        │  ├─ YES → Different smoothness per regime
│  │  │        │  │        Scientific insight!
│  │  │        │  │
│  │  │        │  └─ NO → Regimes have same smoothness
│  │  │        │          Model 3 overkill, use simpler version
│  │  │        │
│  │  │        └─ LOO comparison with Models 1 & 2
│  │  │           ├─ Model 3 best (ΔLOO>4)?
│  │  │           │  ├─ YES → DECISION: Regime change confirmed
│  │  │           │  │                 Report Model 3 as winner
│  │  │           │  │
│  │  │           │  └─ NO → Model 3 worse than simpler models
│  │  │           │          Complexity not justified
│  │  │           │          DECISION: Use Model 1 or 2
│  │  │           │
│  │  │           └─ All models comparable (ΔLOO<2)?
│  │  │              DECISION: Report most interpretable
│  │  │                        (likely Model 2 or piecewise linear)
│  │  │
│  │  └─ NO → Convergence issues (expected)
│  │          │
│  │          ├─ Try simplifications:
│  │          │  - Fix τ=7 (based on EDA)
│  │          │  - Remove GP components (piecewise linear only)
│  │          │  - Use variational inference (faster, approximate)
│  │          │
│  │          └─ Still fails?
│  │             DECISION: Model 3 too complex for n=27
│  │                       Report Models 1-2 results only
│  │                       Mention Model 3 attempt in limitations
│  │
│  └─ Skip Model 3 if:
│     - Models 1-2 show smooth saturation (no regime change)
│     - Models 1-2 both fail (no point trying more complex)
│     - LOO for Models 1-2 much worse than parametric
│     - Time/resource constraints


FINAL DELIVERABLE
│
├─ Successful Outcome:
│  - One or more flexible models converged and validated
│  - LOO comparison complete
│  - Sensitivity analyses done
│  - Posterior predictive checks passed
│  - Clear recommendation with evidence
│
│  DELIVERABLE: Report with:
│  ├─ Best model specification + rationale
│  ├─ Parameter posteriors with interpretation
│  ├─ Predictions with uncertainty
│  ├─ Comparison table (LOO, WAIC, diagnostics)
│  ├─ Plots: posterior mean ± CI, residuals, derivatives
│  └─ Honest assessment of limitations
│
└─ Failure Outcome:
   - All flexible models failed validation
   - Parametric models superior
   - n=27 insufficient for flexibility

   DELIVERABLE: Report titled "Why Flexibility Failed"
   ├─ Document what went wrong (technical details)
   ├─ Explain why simple models won
   ├─ Recommend minimum sample size for flexible methods
   ├─ Suggest parametric alternatives (Designer 1's work)
   └─ Lessons learned for future analyses


STOPPING RULES
│
├─ Stop and accept simple parametric model if:
│  - All flexible models have LOO worse by ΔLOO > 4
│  - Computational issues persist despite extensive tuning
│  - Posterior uncertainty so large that inferences meaningless
│  - Models cannot pass basic posterior predictive checks
│
├─ Stop and request more data if:
│  - Models disagree fundamentally on basic features
│  - LOO is unstable (many high k̂ values)
│  - Sensitivity to single observations is extreme
│  - Cannot distinguish between competing hypotheses
│
└─ Stop and report success if:
   - Clear winner emerges (ΔLOO > 4)
   - Posterior predictive checks pass
   - Sensitivity analysis shows robustness
   - Can answer scientific questions with confidence


ESCAPE HATCHES (If Things Go Wrong)
│
├─ Computational issues → Variational inference
├─ Outlier issues → Student-t likelihood
├─ Heteroscedasticity → Model log(σ)
├─ Non-monotonic → Transform Y (log or sqrt)
├─ Sample size too small → Strongly informative priors
├─ All models fail → Abandon, use parametric
└─ Fundamental confusion → Request domain expertise
```

---

## Key Principle: Fail Fast, Learn Quickly

This decision tree prioritizes **rapid falsification** over exhaustive model exploration. At each node, we ask: "Does this approach work?" If no, we pivot quickly rather than persisting.

**Time budget allocation**:
- Model 1 (GP): 50% of effort (most promising)
- Model 2 (P-splines): 30% of effort (fast alternative)
- Model 3 (Adaptive GP): 20% of effort (only if warranted)

**No-go thresholds** (immediate abandonment):
- Divergences > 10%
- R̂ > 1.05 after tuning
- ESS < 100 after tuning
- LOO worse than baseline by ΔLOO < -4
- Cannot pass monotonicity check in PPC

**Success indicators** (high confidence):
- ΔLOO > 4 favoring flexible model
- All diagnostics green (R̂<1.01, ESS>400, divg<1%)
- Posterior predictive p-values in [0.05, 0.95]
- Robust to outlier removal
- Interpretable scientific story

---

**Use this tree during implementation to avoid getting stuck!**
