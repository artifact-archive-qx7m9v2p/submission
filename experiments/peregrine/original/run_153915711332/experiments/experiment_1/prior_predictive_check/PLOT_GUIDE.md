# Prior Predictive Check: Visual Diagnostics Guide

This guide explains what each plot reveals and how to interpret it.

---

## Plot 1: parameter_prior_marginals.png

**Purpose:** Verify that parameter priors match intended distributions

**What to look for:**
- Delta (left): Should be centered at 0.05 with narrow spread
- Sigma_eta (middle): Should have median around 0.05-0.10
- Phi (right): Should have median around 10-20

**What we found:**
- ✓ Delta: Excellent - mean 0.049, well-concentrated
- ✗ Sigma_eta: Too diffuse - long tail to 0.5+
- ✗ Phi: Too diffuse - wide range from 0.3 to 60

**Key insight:** The marginal priors for sigma_eta and phi extend too far into implausible territory.

---

## Plot 2: prior_predictive_trajectories.png

**Purpose:** Assess whether prior generates plausible count trajectories

**What to look for:**
- Do the blue trajectories cover observed data (red line)?
- Are trajectories staying within scientifically plausible bounds?
- Does the 95% CI (shaded region) remain reasonable?

**What we found:**
- ✓ Coverage: Observed data within prior envelope
- ✗ Scale: Prior 95% CI extends to 7,000+ by t=40 (observed max: 272)
- ✗ Median trajectory: Flatter than observed data (prior doesn't favor growth)

**Key insight:** The prior is TOO WIDE - it covers the data but also covers ridiculous scenarios.

---

## Plot 3: prior_predictive_coverage.png (4-panel)

**Purpose:** Detailed diagnostic of prior predictive distribution properties

### Panel A: Distribution of Prior Predictive Means
**What we found:** Observed mean (109) falls at extreme left tail; prior mean is 419

### Panel B: Distribution of Prior Predictive Max
**What we found:** Observed max (272) well below prior median (550); 95% CI extends to 11,610

### Panel C: Distribution of Prior Growth Factors
**What we found:** Observed growth (8.45x) near prior median (6.6x) - this is actually reasonable!

### Panel D: Prior Coverage at Selected Time Points
**What we found:** Violin plots show increasing right skew over time; observed data at bottom of distribution by t=39

**Key insight:** The prior fails to concentrate probability mass around observed data characteristics. While it "covers" the data, it treats the observed data as implausibly small.

---

## Plot 4: computational_red_flags.png (4-panel)

**Purpose:** Identify extreme values and parameter regions that generate them

### Panel A: Distribution of All Prior Predictive Counts
**What we found:** Vast majority of counts <1000, but right tail extends to 7,000+

### Panel B: Log-Scale Distribution
**What we found:** Even on log scale, distribution shows heavy right tail

### Panel C: Growth Factor Distribution (with red flags)
**What we found:** Most growth factors reasonable (<20x), but 1.6% exceed 100x threshold

### Panel D: Parameter Space - Extreme Count Regions
**Critical finding:** Extreme counts (red points) cluster at HIGH sigma_eta values
**Mechanism:** Large sigma_eta → volatile random walk → extreme cumulative changes → explosive counts

**Key insight:** This plot directly identifies WHY priors are too diffuse - high sigma_eta is the primary culprit.

---

## Plot 5: latent_state_prior.png (2-panel)

**Purpose:** Examine latent state (eta) behavior before observation model

### Panel A: Prior Predictive Latent State Trajectories
**What we found:**
- Prior 95% CI on eta spans [2.3, 8.5] by t=40
- Observed log-counts stay near bottom of envelope
- Prior median trajectory is nearly flat (δ≈0.05 is small relative to sigma_eta variability)

### Panel B: Prior Distribution of Initial State
**What we found:**
- ✓ Well-specified: Prior mean log(50)=3.91 is reasonable
- Observed initial log(29)=3.37 is well within prior

**Key insight:** The initial state prior is fine, but the compounding effect of sigma_eta causes the envelope to explode over time. This is the "random walk cumulative effect" problem.

---

## Plot 6: joint_prior_diagnostics.png (6-panel)

**Purpose:** Understand joint behavior of parameters and their impact on predictions

### Panel A: Drift vs Innovation SD
**Finding:** No correlation (independent priors, as intended)

### Panel B: Drift vs Dispersion
**Finding:** No correlation (independent priors, as intended)

### Panel C: Innovation SD vs Dispersion
**Finding:** No correlation, but both contribute to extreme predictions

### Panel D: Drift vs Realized Growth
**Critical finding:**
- Green = plausible growth (1-50x)
- Red = extreme growth (>50x or <1x)
- Higher drift → higher growth (expected)
- But: Even at δ=0.05, some draws produce extreme growth due to sigma_eta

### Panel E: Innovation SD vs Growth
**Most important panel:**
- Clear relationship: high sigma_eta → extreme growth
- Confirms that sigma_eta is the primary driver of implausible predictions
- Even modest drift can produce wild growth if sigma_eta is large

### Panel F: Prior Predictive Space (Mean vs Max)
**Finding:**
- Purple cloud = prior predictive samples
- Red star = observed data (mean=109, max=272)
- Observed data in lower-left corner; prior spreads to (5000, 50000+)

**Key insight:** The joint prior behavior reveals that sigma_eta and phi interact to create extreme predictions. The problem is not just marginal distributions but their compounding effect.

---

## Overall Diagnostic Story

### The Chain of Evidence

1. **Marginal priors** (Plot 1) show sigma_eta and phi are diffuse
2. **Prior trajectories** (Plot 2) show this creates explosive 95% CIs
3. **Coverage diagnostics** (Plot 3) show observed data at tail of prior
4. **Red flag analysis** (Plot 4, Panel D) identifies HIGH SIGMA_ETA as culprit
5. **Latent state view** (Plot 5) shows cumulative compounding effect
6. **Joint diagnostics** (Plot 6, Panel E) confirms sigma_eta drives extremes

### The Smoking Gun

**Plot 4, Panel D** and **Plot 6, Panel E** are the most damning:
- They show that extreme counts cluster at sigma_eta > 0.2
- This directly points to the solution: tighten sigma_eta prior
- Secondary issue: Low phi (<2) also contributes via observation noise

---

## How to Use These Plots for Revised Priors

### After implementing revised priors, check:

1. **Plot 1:** Sigma_eta should have 95% CI ending around 0.15 (not 0.37)
2. **Plot 2:** 95% CI at t=40 should be roughly [50, 2000] (not [50, 7000])
3. **Plot 3, Panel A:** Prior mean should be closer to 109 (not 419)
4. **Plot 3, Panel B:** Prior 95% max should be <1000 (not 11,610)
5. **Plot 4, Panel D:** Red points (extremes) should be <0.1% (not 3.1%)
6. **Plot 6, Panel F:** Prior cloud should concentrate around observed star

---

## Interpretive Framework

### What makes a GOOD prior predictive plot?

**PASS criteria:**
- Observed data within prior IQR (not tail)
- Extreme values <0.1% of samples
- Prior concentrates mass around plausible values
- 95% CI stays within order-of-magnitude of observed range

**FAIL criteria:**
- Observed data at tail of prior distribution
- Frequent extreme values (>0.5%)
- Prior treats observed data as implausibly small/large
- Computational red flags in >1% of samples

### Our assessment:

Current priors: **FAIL**
- Observed data consistently at left tail
- 0.4% extreme counts, 1.6% extreme growth
- Prior mean 4x higher than observed mean
- 95% CI extends to 40x observed maximum

**Conclusion:** Priors are technically proper and cover the data, but they're NOT weakly informative. They assign too much probability mass to implausible extremes.
