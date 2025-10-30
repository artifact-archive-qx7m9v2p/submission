# Designer 2: Document Index

**Quick navigation guide for all Designer 2 documents**

---

## Start Here

**New to this project?** Start with these in order:

1. **README.md** - High-level summary (5 min read)
2. **SUMMARY.txt** - Visual overview (2 min read)
3. **proposed_models.md** - Full model specifications (30 min read)

---

## Document Guide

### Executive Documents

#### README.md (2 KB)
- **Purpose**: Quick introduction
- **Audience**: Anyone new to the project
- **Contents**: Philosophy, expected outcome, falsification mindset
- **Read if**: You need a 5-minute overview

#### SUMMARY.txt (4 KB)
- **Purpose**: Visual summary with ASCII boxes
- **Audience**: Quick reference
- **Contents**: Models, decision points, red flags
- **Read if**: You want the key points in visual format

#### DELIVERABLES.md (7 KB)
- **Purpose**: Complete list of what was delivered
- **Audience**: Project managers, reviewers
- **Contents**: All files, models, contributions, next steps
- **Read if**: You need to verify completeness

---

### Technical Specifications

#### proposed_models.md (27 KB) ⭐ MOST IMPORTANT
- **Purpose**: Complete model specifications
- **Audience**: Modelers, implementers
- **Contents**:
  - Model 1: GP-NegBin (full Stan code)
  - Model 2: P-splines (full Stan code)
  - Model 3: Semi-parametric (full PyMC code)
  - All priors with justifications
  - Falsification criteria for each model
  - Expected insights and failure modes
- **Read if**: You're implementing any of the models

#### experiment_plan.md (16 KB) ⭐ STRATEGIC DOCUMENT
- **Purpose**: Overall strategy and decision framework
- **Audience**: Research leads, designers
- **Contents**:
  - Problem formulation with competing hypotheses
  - Model variants and configurations
  - Red flags and decision points
  - Alternative approaches if models fail
  - Domain constraints
  - Stopping rules
- **Read if**: You're making strategic decisions about the project

---

### Implementation Resources

#### implementation_guide.md (13 KB) ⭐ FOR CODERS
- **Purpose**: Copy-paste ready code
- **Audience**: Programmers implementing models
- **Contents**:
  - Complete Stan code for Models 1-2
  - Complete PyMC code for Model 3
  - Data preprocessing steps
  - Fitting procedures
  - Diagnostic checking code
  - Model comparison code
- **Read if**: You're writing code to fit the models

#### model_comparison_matrix.md (5 KB)
- **Purpose**: Quick reference table
- **Audience**: Anyone comparing models
- **Contents**:
  - Side-by-side model comparison
  - Falsification criteria summary
  - Decision tree flowchart
  - Expected behavior scenarios
- **Read if**: You need to quickly compare models or make decisions

---

## Reading Paths

### Path 1: "I need to understand the approach" (45 min)
1. README.md (5 min)
2. proposed_models.md, sections 1-3 (20 min)
3. experiment_plan.md, sections 1-2 (20 min)

### Path 2: "I need to implement these models" (60 min)
1. implementation_guide.md (30 min)
2. proposed_models.md, implementation notes (20 min)
3. model_comparison_matrix.md (10 min)

### Path 3: "I need to make strategic decisions" (40 min)
1. SUMMARY.txt (5 min)
2. experiment_plan.md, sections on decision points (15 min)
3. proposed_models.md, falsification sections (20 min)

### Path 4: "I'm reviewing the deliverables" (30 min)
1. DELIVERABLES.md (10 min)
2. README.md (5 min)
3. Skim all other documents (15 min)

---

## Key Sections by Topic

### Falsification Criteria
- **proposed_models.md**: Each model has "Why This Might FAIL" section
- **experiment_plan.md**: "Red Flags" and "Evidence to Switch" sections
- **model_comparison_matrix.md**: Falsification summary table

### Implementation Code
- **implementation_guide.md**: All code snippets
- **proposed_models.md**: Stan/PyMC code within model specs

### Decision Rules
- **experiment_plan.md**: "Decision Points" section
- **model_comparison_matrix.md**: Decision tree
- **SUMMARY.txt**: Visual decision flowchart

### Prior Justifications
- **proposed_models.md**: "Prior Distributions" for each model
- **proposed_models.md**: "Prior Justification and Sensitivity" section

### Model Comparison
- **model_comparison_matrix.md**: Entire document
- **implementation_guide.md**: "Model Comparison" section
- **experiment_plan.md**: Cross-designer comparison strategy

---

## Quick Reference: Where to Find...

**Stan code for GP model?**
→ proposed_models.md, Model 1 section
→ implementation_guide.md, Model 1 section

**Stan code for P-splines?**
→ proposed_models.md, Model 2 section
→ implementation_guide.md, Model 2 section

**PyMC code for semi-parametric?**
→ proposed_models.md, Model 3 section
→ implementation_guide.md, Model 3 section

**When to abandon a model?**
→ proposed_models.md, "Why This Might FAIL" for each model
→ model_comparison_matrix.md, falsification table

**How to decide which model is best?**
→ experiment_plan.md, "Decision Points" section
→ implementation_guide.md, "Decision Rules Implementation"

**What to do if models fail?**
→ experiment_plan.md, "Alternative Approaches" and "Backup Plans"
→ proposed_models.md, stress test sections

**Prior choices and why?**
→ proposed_models.md, each model's "Prior Distributions" section
→ proposed_models.md, "Prior Justification and Sensitivity"

**Expected outcomes?**
→ SUMMARY.txt, expected winner
→ proposed_models.md, "Expected Insights" for each model

**Cross-designer coordination?**
→ experiment_plan.md, cross-designer sections
→ README.md, distinctions from other designers

---

## File Sizes and Complexity

```
Simple/Overview:
├─ README.md          (2 KB)  ⚡ Quick read
├─ SUMMARY.txt        (4 KB)  ⚡ Quick read
└─ DELIVERABLES.md    (7 KB)  ⚡ Quick read

Reference:
├─ model_comparison_matrix.md  (5 KB)  ⚡ Quick reference
└─ INDEX.md                    (4 KB)  ⚡ This file

Detailed/Technical:
├─ implementation_guide.md  (13 KB)  🔧 For implementation
├─ experiment_plan.md       (16 KB)  📊 Strategic
└─ proposed_models.md       (27 KB)  📚 Complete specs

Total: 92 KB documentation
```

---

## Recommended Reading Order by Role

### If you're a **Statistical Modeler**:
1. proposed_models.md (complete read)
2. experiment_plan.md (skim for strategy)
3. model_comparison_matrix.md (quick reference)

### If you're a **Programmer/Implementer**:
1. implementation_guide.md (complete read)
2. proposed_models.md (focus on model specs)
3. Keep model_comparison_matrix.md handy

### If you're a **Project Lead**:
1. DELIVERABLES.md (verify completeness)
2. experiment_plan.md (understand strategy)
3. SUMMARY.txt (overview for meetings)

### If you're a **Reviewer**:
1. README.md (context)
2. proposed_models.md (check technical rigor)
3. experiment_plan.md (check falsification thinking)

---

## Changelog

**2025-10-29**: Initial creation
- All 7 documents completed
- Total 92 KB documentation
- Design phase complete

---

## Next Steps

**After reading appropriate documents**:
1. Create `.stan` and `.py` files from code in implementation_guide.md
2. Set up Python environment with required packages
3. Begin with Model 2 (P-splines) as highest priority
4. Follow diagnostic procedures in implementation_guide.md
5. Apply decision rules from experiment_plan.md

---

**Location**: `/workspace/experiments/designer_2/`
**Status**: Design complete, ready for implementation
**Contact**: Designer 2 (Non-parametric Specialist)
