# Document Navigation Map - Designer 3

**Total Documentation**: ~10,000 words across 4 documents

---

## How to Navigate This Design

### If You Have 5 Minutes: Read This File + README

**Start Here**: `/workspace/experiments/designer_3/README.md` (1,300 words)
- Quick overview of three models
- Key principles
- Expected outcomes

### If You Have 30 Minutes: Add Implementation Priority

**Then Read**: `/workspace/experiments/designer_3/implementation_priority.md` (1,500 words)
- Step-by-step fitting order
- Falsification test checklist
- Decision tree
- Quick start commands

### If You Have 1 Hour: Add Visual Comparison

**Then Read**: `/workspace/experiments/designer_3/model_comparison_summary.md` (2,200 words)
- Side-by-side model diagrams
- Conceptual differences
- Strengths/weaknesses tables
- Expected parameter posteriors
- Posterior predictive check guidance

### If You're Implementing: Read Everything

**Finally Read**: `/workspace/experiments/designer_3/proposed_models.md` (5,000 words)
- Complete mathematical specifications
- Full Stan code for all three models
- Detailed prior justifications
- Comprehensive falsification criteria
- Model comparison strategy
- Red flags and pivot plans
- Alternative approaches if all models fail

---

## Document Purpose Matrix

| Document | Audience | Purpose | Detail Level |
|----------|----------|---------|--------------|
| **README** | Everyone | Quick overview, key principles | High-level |
| **Implementation Priority** | Practitioners | How to fit models in order | Tactical |
| **Model Comparison Summary** | Researchers | Understand model differences | Conceptual |
| **Proposed Models** | Implementers | Complete specifications | Full detail |

---

## Content Cross-Reference

### Where to Find Specific Information

#### Model Specifications
- **Equations**: All documents have them, but full detail in `proposed_models.md`
- **Stan code**: Only in `proposed_models.md` (complete templates)
- **Priors**: Overview in all docs, full justification in `proposed_models.md`

#### Falsification Criteria
- **Checklist**: `implementation_priority.md` (Section: Falsification Test Checklist)
- **Full details**: `proposed_models.md` (each model has dedicated section)
- **Summary table**: `model_comparison_summary.md` (Section: Falsification Criteria)

#### Implementation Guidance
- **Priority order**: `implementation_priority.md` (Section: Priority Order)
- **Computational tips**: `implementation_priority.md` (Section: Computational Tips)
- **Stan templates**: `proposed_models.md` (each model section)
- **Expected runtime**: All documents mention, summary in `model_comparison_summary.md`

#### Decision Making
- **Decision tree**: `implementation_priority.md` (visual flowchart)
- **Model comparison strategy**: `proposed_models.md` (Section: Comparative Model Evaluation)
- **LOO comparison**: `model_comparison_summary.md` (Section: LOO Comparison)
- **When to pivot**: All documents, comprehensive in `proposed_models.md` (Section: Red Flags)

#### Conceptual Understanding
- **Mental models**: `model_comparison_summary.md` (Section: Conceptual Differences)
- **Why these models**: `proposed_models.md` (each model's "Core Philosophy")
- **Model philosophy**: `README.md` and `proposed_models.md` (final sections)

---

## Reading Paths for Different Goals

### Goal: Quick Understanding (15 min)

```
README.md (full)
  └─> model_comparison_summary.md (Sections: At a Glance, Conceptual Differences)
```

### Goal: Implementation Planning (45 min)

```
README.md (full)
  └─> implementation_priority.md (full)
      └─> model_comparison_summary.md (Section: Computational Comparison)
```

### Goal: Deep Dive Before Coding (2 hours)

```
README.md
  └─> model_comparison_summary.md (full)
      └─> proposed_models.md (Model 1 section, then Model 3, then Model 2)
          └─> implementation_priority.md (as reference while coding)
```

### Goal: Troubleshooting (as needed)

```
Having convergence issues?
  └─> implementation_priority.md (Section: Computational Tips)
      └─> proposed_models.md (specific model's Implementation Considerations)

Model failing tests?
  └─> implementation_priority.md (Section: Red Flag Triggers)
      └─> proposed_models.md (specific model's Falsification Criteria)
          └─> proposed_models.md (Section: Red Flags Requiring Model Class Pivot)

Need to compare models?
  └─> model_comparison_summary.md (Section: LOO Comparison)
      └─> proposed_models.md (Section: Comparative Model Evaluation Strategy)
```

---

## Document Statistics

```
File                              Lines    Words    Size
─────────────────────────────────────────────────────────
README.md                          305     1,312    9.1K
implementation_priority.md         364     1,470     10K
model_comparison_summary.md        471     2,245     20K
proposed_models.md                 879     4,964     35K
─────────────────────────────────────────────────────────
TOTAL                            2,019     9,991     74K
```

---

## Key Sections Quick Reference

### In proposed_models.md

| Section | Line/Location | What It Contains |
|---------|---------------|------------------|
| Executive Summary | Top | High-level overview, key insight |
| Model 1: DLM | ~50 | Full spec, Stan code, priors, falsification |
| Model 2: NB-AR | ~200 | Full spec, Stan code, priors, falsification |
| Model 3: GP | ~350 | Full spec, Stan code, priors, falsification |
| Comparative Evaluation | ~500 | 4-phase strategy, metrics, decision points |
| Red Flags | ~650 | Catastrophic failures, when to pivot |
| Model Ranking | ~700 | Primary/secondary/tertiary recommendations |
| Implementation Roadmap | ~750 | Week-by-week timeline |
| Alternative Approaches | ~800 | If primary models fail, what next |
| Success Criteria | ~850 | 10 criteria for "successful" model |

### In implementation_priority.md

| Section | What It Contains |
|---------|------------------|
| Priority Order | Which model to fit first (Model 1 > 3 > 2) |
| Falsification Test Checklist | 4 tests with code snippets |
| Red Flag Triggers | Computational, statistical, model-specific |
| Decision Tree | Visual flowchart for model selection |
| Expected Outcomes | Predictions for each model |
| Computational Tips | Stan tricks, initialization, sampling params |
| File Organization | Directory structure, file naming |
| Timeline Estimate | 3-week detailed schedule |

### In model_comparison_summary.md

| Section | What It Contains |
|---------|------------------|
| At a Glance | Quick visual comparison |
| Conceptual Differences | Mental models with diagrams |
| Mathematical Equations | Side-by-side equation table |
| Strengths and Weaknesses | Detailed for each model |
| Falsification Criteria | Summary tables with thresholds |
| Expected Parameter Values | Posterior predictions |
| LOO Comparison | Predicted ranking |
| Computational Comparison | Runtime, memory, ease |
| Posterior Predictive Checks | Visual and numerical checks |
| When to Pivot | Scenarios and next steps |
| Summary Flowchart | Complete decision process |

---

## Search Keywords (Ctrl+F Guide)

Want to find information about...

**Autocorrelation**:
- Search "ACF" or "autocorrelation" in any document
- Most detail in `proposed_models.md` falsification sections

**Changepoint/Structural Break**:
- Search "τ" (tau) or "changepoint" or "regime"
- Visual explanation in `model_comparison_summary.md`

**Priors**:
- Search "prior" in any document
- Full justification in `proposed_models.md` (each model has "Prior Recommendations")

**Computational Issues**:
- Search "divergence" or "computational"
- Troubleshooting in `implementation_priority.md`

**Falsification**:
- Search "falsification" or "FAIL" or "abandon"
- Checklist in `implementation_priority.md`
- Full criteria in `proposed_models.md`

**Stan Code**:
- Only in `proposed_models.md`
- Search "```stan" to find code blocks

**Timeline**:
- Search "Week" or "timeline"
- In both `proposed_models.md` and `implementation_priority.md`

**LOO/Model Comparison**:
- Search "LOO" or "ELPD"
- Strategy in `proposed_models.md`, predictions in `model_comparison_summary.md`

**Red Flags**:
- Search "red flag" or "STOP"
- All documents have them, comprehensive in `proposed_models.md`

---

## Printed Reading Order

If you're printing these documents:

1. **README.md** (9 pages) - Overview
2. **model_comparison_summary.md** (20 pages) - Visual guide
3. **implementation_priority.md** (10 pages) - Practical steps
4. **proposed_models.md** (35 pages) - Full reference

**Total**: ~74 pages (single-spaced)

---

## Digital Reading Order

If you're reading on screen:

1. **README.md** - Get oriented (5 min)
2. **model_comparison_summary.md** - Understand differences (20 min)
3. **proposed_models.md** - Dive deep on Model 1 only (30 min)
4. **implementation_priority.md** - Make implementation plan (15 min)
5. Return to **proposed_models.md** when coding each model

**Total**: ~70 min for initial read-through

---

## FAQ: Which Document Answers What?

**Q: Which model should I fit first?**
A: `implementation_priority.md` Section 1 (Answer: Model 1 with τ=17 fixed)

**Q: What are the complete Stan code templates?**
A: `proposed_models.md` - each model section has full Stan code

**Q: How do I know if a model has failed?**
A: `implementation_priority.md` Falsification Test Checklist (4 tests)

**Q: What if all three models fail?**
A: `proposed_models.md` Section "Red Flags Requiring Model Class Pivot"

**Q: How long will this take?**
A: `implementation_priority.md` Timeline (Answer: 15 working days)

**Q: Why these three specific models?**
A: `proposed_models.md` Executive Summary and each model's "Core Philosophy"

**Q: What are the key differences between models?**
A: `model_comparison_summary.md` Sections 1-3 (visual diagrams)

**Q: What parameter values should I expect?**
A: `model_comparison_summary.md` Section "Expected Parameter Values"

**Q: How do I compare models?**
A: `proposed_models.md` Section "Comparative Model Evaluation Strategy"

**Q: What does success look like?**
A: `proposed_models.md` Section "Success Criteria" (10 criteria)

---

## Version Control

**Version**: 1.0
**Date**: 2025-10-29
**Author**: Model Designer 3 (Time-Series Specialist)
**Based on**: EDA findings in `/workspace/eda/eda_report.md`

**Change Log**:
- v1.0 (2025-10-29): Initial comprehensive design
  - Three model classes proposed
  - Complete specifications with Stan code
  - Falsification criteria defined
  - Implementation guidance provided

---

## External Dependencies

These documents assume you have access to:

1. **EDA Report**: `/workspace/eda/eda_report.md`
   - Referenced for prior justification
   - Summary of key findings

2. **Data**: 40 observations of count data
   - Located at: (to be specified by main agent)
   - Variables: `year` (standardized), `C` (counts)

3. **Software**:
   - Stan (cmdstan or pystan/rstan)
   - PyMC (alternative to Stan for some models)
   - Python/R for analysis
   - LOO package for model comparison

---

## Final Note

These documents embody a **falsification-first** approach to Bayesian modeling:

1. We propose models with the INTENT of discovering their limitations
2. Each model has explicit failure criteria
3. Failure is success (it teaches us about the data generation process)
4. We pivot quickly when models fail
5. The goal is scientific truth, not completing a predetermined plan

**If you find yourself "forcing" a model to work despite failures, you're doing it wrong. Trust the falsification criteria.**

---

**Navigation Complete**

Return to: `/workspace/experiments/designer_3/README.md` to begin implementation.
