## Bayesian Requirement (Hard Constraint)
- The final deliverable must be a Bayesian model: specify priors, perform posterior inference, and evaluate via posterior predictive checks.
- Non-Bayesian methods may be explored as baselines/context but cannot be selected or reported as the solution.

## Implementation Requirements
- **PPL usage**: All Bayesian models must use probabilistic programming (Stan via CmdStanPy or PyMC) with MCMC or VI
- **Applies to all model types**: regression, hierarchical, Gaussian Processes, state-space, splines, etc.
- **Allowed non-PPL**: sklearn/scipy/etc. only for preprocessing (e.g., scalers, splits), EDA (e.g., PCA, clustering), and baseline comparisons (not the final Bayesian model)
- **Pivot accountability**: If claiming "numerical issues" and abandoning a model class or switching away from Stan→PyMC, include the error output in experiments/<name>/diagnostics/ before pivoting
- **Log-likelihood requirement**: Every accepted Bayesian model must save ArviZ InferenceData with pointwise log_likelihood suitable for LOO (ArviZ `az.loo/az.compare`). For Stan, compute `vector[N] log_lik` in generated quantities and convert with `az.from_cmdstanpy(..., log_likelihood='log_lik')`. For PyMC, ensure the `log_likelihood` group is present (e.g., `az.from_pymc(..., log_likelihood=True)`)
- **Standard paths**: Save comparison-ready InferenceData to `experiments/experiment_X/posterior_inference/diagnostics/posterior_inference.netcdf`

# Bayesian Model Building Guidelines

This document defines the systematic workflow for autonomous Bayesian model development.

## Software Stack

- CmdStanPy, ArviZ [PRIMARY - required for all Bayesian models]
- PyMC [FALLBACK - only if Stan fails with documented errors]
- Matplotlib, Seaborn
- Pandas, NumPy
- scikit-learn [preprocessing/EDA/baselines only]

- Use `uv` exclusively (never pip)
    - Setup: `uv sync`
    - Install: `uv add package`
    - Run: `uvx run tool`

## Parallel Agent Execution

Parallel execution prevents blind spots and increases robustness. The workflow diagrams show when to use parallel agents (mandatory for model design, recommended for complex EDA, optional for borderline critique).

### Key Implementation Notes
- Each parallel agent works in complete isolation with its own data copy and output directory
- Main agent must prepare data copies before spawning: `data_analyst_1.csv`, `data_analyst_2.csv`, etc.
- Choose number of parallel agents based on problem complexity, uncertainty level, and available resources (typically 2-3)
- Always explicitly specify output paths when spawning agents:
  - Parallel: "Write all outputs to `eda/analyst_1/`"
  - Solo: "Write outputs to `eda/`"
- If a parallel agent fails, relaunch it once with the same instructions; if it fails again, proceed with successful agents
- After parallel execution, main agent synthesizes convergent and divergent findings
- Use different focus areas for each parallel agent to maximize coverage

## Modeling Workflow

### Phase 1: Data Understanding
```
START → [Complexity Assessment]
         ├─ Simple/Familiar data:
         │   └─→ eda-analyst → eda/eda_report.md
         │
         └─ Complex/Unknown data (recommended default):
             ├─→ Prepare data copies for parallel agents
             ├─→ eda-analyst [parallel 1-3] → eda/analyst_N/
             └─→ Main agent synthesis → eda/synthesis.md → eda/eda_report.md
```
- Understand data characteristics from multiple perspectives
- Identify convergent and divergent patterns
- Generate comprehensive modeling hypotheses

### Phase 2: Model Design
```
EDA Report → [Always run parallel designers to avoid blind spots]
             ├─→ model-designer [parallel 2-3] → experiments/designer_N/proposed_models.md
             └─→ Main agent synthesis → experiments/experiment_plan.md
                 (Combines all proposed model classes, removes duplicates,
                  prioritizes by theoretical justification)
```
- Each designer independently proposes 2-3 model classes
- Main agent synthesizes to catch missed model types
- Define falsification criteria for all models
- Plan iteration strategy

### Phase 3: Model Development Loop
```
For each model in Experiment Plan:
    
    VALIDATION PIPELINE:
    ├─→ prior-predictive-checker
    │   ├─ FAIL → Document issue → Skip to next model
    │   └─ PASS ↓
    ├─→ simulation-based-validator  
    │   ├─ FAIL → Document issue → Skip to next model
    │   └─ PASS ↓
    ├─→ model-fitter (must save log_likelihood in InferenceData)
    │   ├─ FAIL → Document convergence issue → Try refinement or skip
    │   └─ PASS ↓
    ├─→ posterior-predictive-checker
    │   └─→ Continue regardless (document fit quality)
    │
    EVALUATION → [Adequacy Assessment]
    ├─ Clear pass/fail:
    │   └─→ model-critique → [ACCEPT/REVISE/REJECT]
    │
    └─ Borderline/Complex tradeoffs:
        ├─→ model-critique [parallel 2] → experiment_N/critique_N/
        └─→ Synthesis → [ACCEPT/REVISE/REJECT]
    
    Decision outcomes:
    ├─ ACCEPT → Add to successful models
    ├─ REVISE → model-refiner → Create new experiment → Loop
    └─ REJECT → Document failure → Next model class
    
    └─→ Continue per Minimum Attempt Policy (below)
```

**Minimum Attempt Policy:**
- Attempt at least the first two models in `experiments/experiment_plan.md` unless Model 1 fails pre-fit validation
- If fewer than two models attempted, document reason in `log.md`
- After attempting required models, proceed to Phase 4

### Phase 4: Model Assessment & Comparison (always run)
```
All ACCEPT models → model-assessment-analyst → Assessment Report (+ Comparison if 2+ models)
```

**Single Model (1 ACCEPT):**
- LOO diagnostics: Compute ELPD ± SE, Pareto k summary
- Calibration: LOO-PIT plot and 90% coverage check
- Absolute metrics: RMSE/MAE or task-appropriate error
- Document in `experiments/model_assessment/assessment_report.md`

**Multiple Models (2+ ACCEPTs):**
- Everything from single model assessment, plus:
- Model comparison via `az.compare`: ΔELPD ± SE, rankings
- Apply parsimony rule when |ΔELPD| < 2×SE
- Document in `experiments/model_comparison/comparison_report.md`

**Note:** Phase 4 always runs after Phase 3 completes. It provides assessment context but does not re-judge ACCEPT/REJECT decisions from Phase 3.

### Phase 5: Adequacy Assessment
```
All Experiments → model-adequacy-assessor → [ADEQUATE/CONTINUE/STOP]
├─ ADEQUATE → Proceed to reporting
├─ CONTINUE → model-refiner → New experiments
└─ STOP → Document limitations → Consider simpler Bayesian approach (remain within the Bayesian paradigm)
```

### Phase 6: Final Reporting
```
Adequate Model → report-writer → Final Report
```

## Decision Logic

**When to iterate within a model class:**
- Fixable issues identified (e.g., missing predictor, wrong likelihood)
- Clear improvement path exists
- Convergence/computational issues can be resolved

**When to switch model classes (within Bayes):**
- Fundamental misspecification evident  
- Multiple refinements fail
- Prior-data conflict unresolvable
- Switching to non-Bayesian methods is not permitted

**When to stop entirely:**
- Adequate Bayesian model found
- Diminishing returns (improvements < 2*SE)
- Computational limits reached
- Data quality issues discovered

## Progress Tracking

**Use log.md for all progress tracking** - Never use TodoWrite tool. All task progress, decisions, and status updates should be recorded in log.md at the project root.
**Subagent reporting** - After every subagent completes their task, update log.md with their findings, decisions, and next steps.
**Visual evidence documentation** - When plots are created, document in logs:
  - What question each plot addresses
  - Key findings from the visualization
  - Plot filename for future reference
  - How the visual evidence influences decisions

## Project Structure

```
log.md                             # Progress tracking - record all decisions, task status, and experiment state

data/
├── data.csv                       # Original data
└── data_analyst_N.csv             # Copies for parallel agents (when used)

eda/
├── analyst_N/                     # Parallel analyst N outputs (when parallel used)
│   ├── code/
│   ├── visualizations/
│   └── findings.md
├── synthesis.md                   # Synthesis of parallel findings (when parallel used)
├── eda_report.md                  # Final consolidated EDA report
└── eda_log.md                     # Detailed process log

experiments/
├── designer_N/                    # Parallel designer N proposals (always used)
│   └── proposed_models.md
├── experiment_plan.md             # Synthesized plan from all designers
├── iteration_log.md               # Refinement history
├── adequacy_assessment.md         # Final assessment
├── model_comparison/              # Comparison results (if multiple models)
├── experiment_1/
│   ├── metadata.md                # Model specification
│   ├── prior_predictive_check/
│   │   ├── code/
│   │   ├── plots/
│   │   └── findings.md
│   ├── simulation_based_validation/
│   │   ├── code/
│   │   ├── plots/
│   │   └── recovery_metrics.md
│   ├── posterior_inference/
│   │   ├── code/
│   │   ├── diagnostics/
│   │   ├── plots/
│   │   └── inference_summary.md
│   ├── posterior_predictive_check/
│   │   ├── code/
│   │   ├── plots/
│   │   └── ppc_findings.md
│   ├── critique_N/                # Parallel critique N (when parallel used)
│   │   └── assessment.md
│   ├── model_critique/            # Final critique synthesis/decision
│   │   ├── critique_summary.md
│   │   ├── decision.md           # ACCEPT/REVISE/REJECT
│   │   └── improvement_priorities.md
│   └── refinement_rationale.md   # If this led to refinement
└── experiment_2/...

final_report/
├── report.md                      # Main report
├── figures/                       # Key visualizations
└── supplementary/                 # Additional details
```

## Code Generation Guidelines

When writing analysis code, prioritize clarity, reproducibility, and maintainability:

### Path Management
- Use `pathlib.Path` for all file operations - it's more robust than string concatenation
- Define paths relative to project root or script location: `Path(__file__).parent` is better than hardcoded paths
- Create a single configuration section at the top of each script with all paths defined

### Code Organization
- Functions should do one thing well - aim for 20-50 lines per function
- Use classes when managing state across multiple operations (like `EDAAnalyst`), otherwise simple functions are better
- Group related functionality: data loading together, plotting together, analysis together
- Import statements at the top, configuration next, then functions, then main execution

### Error Handling & Validation
- Check data assumptions explicitly: `assert df['group'].nunique() == expected_groups, f"Expected {expected_groups} groups"`
- Return meaningful error messages that help diagnose issues: `f"Convergence failed: R-hat={rhat:.3f} > 1.01 for parameter {param}"`
- Use specific exception types when catching errors: `except ValueError as e:` is better than bare `except:`
- Validate inputs early: check data exists and has expected structure before processing

### Constants & Configuration
- Define constants at the top: `CONVERGENCE_THRESHOLD = 1.01` with a comment explaining why
- Group related parameters in dictionaries: `PLOT_CONFIG = {'dpi': 300, 'figsize': (10, 6)}`
- Use descriptive names: `N_POSTERIOR_SAMPLES = 1000` is better than `n = 1000`

### Plotting & Visualization
- Create a setup function for consistent plot styling across all scripts
- Close figures explicitly after saving: `plt.close(fig)` prevents memory issues
- Save plots with descriptive names indicating what they show, not their format
- Consider plot density vs clarity - sometimes separate plots communicate better than subplots

### Data Processing
- Use vectorized operations: `df['log_y'] = np.log(df['y'] + 1)` is better than loops
- Chain pandas operations for readability: `df.groupby('group').agg({'y': ['mean', 'std']})`
- Keep original data unchanged - create new columns/dataframes for transformations

### Documentation
- Write docstrings for functions that explain purpose, not just mechanics
- Add inline comments for non-obvious logic: explain the "why" not the "what"
- Include example usage in complex function docstrings
- Type hints are helpful for key functions: `def fit_model(data: dict, chains: int = 4) -> Optional[CmdStanPyFit]:`

### Resource Management
- Use context managers for file operations: `with open(file, 'w') as f:`
- Set random seeds for reproducibility: `np.random.seed(42)` at script start
- Suppress warnings selectively, not globally: use `with warnings.catch_warnings():`

### Output & Logging
- Use print statements that indicate progress: `print(f"Step 2/5: Fitting model (this may take 2-3 minutes)...")`
- Save intermediate results to enable restart after failures
- Create summary reports that link to detailed outputs
- Timestamp outputs when relevant: `output_dir / f"fit_{datetime.now():%Y%m%d_%H%M%S}.csv"`

### Testing & Validation
- Include sanity checks: after fitting, verify parameters are in reasonable ranges
- Test with small data subsets first: `df.sample(100)` for quick iteration
- Add assertion checks that can be disabled: `if DEBUG: assert all(phi > 0)`
- Document expected runtime for long operations

### Software Architecture Considerations
- Think about code organization before writing - will this be reused? Will it grow? Is it complex enough to benefit from modularity?
- Consider splitting into multiple files when it improves clarity or reusability (utilities, configuration, analysis modules)
- A single well-structured file is perfectly fine for focused analyses
- Let the problem complexity guide the architecture - simple problems deserve simple solutions
- If you find yourself copying code between scripts, that's a signal to extract shared utilities

## Key Principles

1. **Falsification over confirmation** - Actively try to break models
2. **Document everything** - Failed attempts are valuable information
3. **One change at a time** - Isolate effects of modifications
4. **Practical adequacy (Bayesian scope)** - Prefer a simple, calibrated Bayesian model over a better-fitting non-Bayesian alternative
5. **Computational efficiency** - Don't waste resources on doomed models
6. **Simplify within Bayes** - When complex models struggle, move to simpler Bayesian formulations (fewer parameters, simpler structure, stronger regularization) before stopping
7. **PPL-first implementation** - Use Stan/PyMC for all Bayesian models; sklearn only for EDA/preprocessing/baselines

## Finalization Checklist
Before declaring a model complete, verify:
- **PPL compliance**: Stan/PyMC used for the selected model's posterior inference
- **Artifacts present**: ArviZ InferenceData (.netcdf) and `az.summary` with Rhat/ESS saved in `posterior_inference/diagnostics/`
- **Posterior predictive**: PPC produced from InferenceData (not bootstrap-only)

## Common Pitfalls to Avoid
- Do not use `sklearn.gaussian_process` for the final Bayesian GP - use Stan/PyMC GP with priors and MCMC/VI
- Do not equate MLE/MAP with full Bayesian inference
- Do not label bootstrap-based checks as posterior predictive checks

## Agents Summary

| Agent | Purpose | When to Use |
|-------|---------|-------------|
| eda-analyst | Exploratory data analysis | Start of workflow |
| model-designer | Create experiment plan | After EDA |
| prior-predictive-checker | Validate priors | Before fitting each model |
| simulation-based-validator | Test parameter recovery | After prior checks |
| model-fitter | Fit model with HMC | After validation passes |
| posterior-predictive-checker | Check model fit | After successful fitting |
| model-critique | Evaluate model adequacy | After all checks complete |
| model-assessment-analyst | Assess model(s) and compare if multiple | Always after Phase 3 completes |
| model-refiner | Propose improvements | When critique suggests revision |
| model-adequacy-assessor | Decide if done (Bayesian adequacy only) | After multiple iterations |
| report-writer | Create final documentation | When adequate model found |