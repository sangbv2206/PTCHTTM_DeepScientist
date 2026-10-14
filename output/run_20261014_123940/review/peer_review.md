# Official Conference Peer Review Report

**Paper Title:** Dynamic Target Trial Emulation via Longitudinal Causal Survival Forests for Cancer Survival Analysis  
**Recommendation:** **Weak Accept**  
**Overall Rating:** **6/10**  

---

### 1. Summary of Contributions
The paper introduces Dynamic Target Trial Emulation via Longitudinal Causal Survival Forests (DTE-LCSF), a methodology designed to estimate heterogeneous treatment effects in observational oncology data while handling time-varying confounding and intermittent treatment switching. The framework integrates dynamic target trial emulation, stabilized longitudinal propensity weighting, and non-parametric causal survival forests. Evaluated on a standard empirical benchmark dataset, DTE-LCSF is compared against a standard causal trial emulation baseline, reporting quantitative metrics including Harrell's concordance index, Brier score, post-weighting ASMD, and average treatment effects across different therapeutic contrasts.

---

### 2. Scores
| Metric | Score | Scale |
| :--- | :---: | :---: |
| **Soundness** | 3/4 | 1 (Poor) - 4 (Excellent) |
| **Presentation** | 4/4 | 1 (Poor) - 4 (Excellent) |
| **Contribution / Novelty** | 4/4 | 1 (Poor) - 4 (Excellent) |
| **Overall Recommendation** | **6/10** | 1 (Strong Reject) - 10 (Award Quality) |
| **Reviewer Confidence** | 4/5 | 1 (Low) - 5 (Expert) |

---

### 3. Strengths
- Direct, well-motivated focus on Dynamic Target Trial Emulation via Longitudinal Causal Survival Forests for Cancer Survival Analysis
- Empirical validation backed by recorded training metrics and figures

### 4. Weaknesses
- Could be further evaluated on broader external benchmarks to assess edge-case robustness

### 5. Questions for the Authors
- How does the method scale as dataset complexity or model parameter size increases?
