# Official Conference Peer Review Report

**Paper Title:** Causal Machine Learning and Target Trial Emulation for Cancer Survival Analysis Using LCSF  
**Recommendation:** **Accept (Poster)**  
**Overall Rating:** **7/10**  

---

### 1. Summary of Contributions
The paper introduces the Longitudinal Causal Survival Forest with Stabilized Propensity Weighting (LCSF) framework, combining target trial emulation principles with non-parametric causal survival trees to analyze time-to-event outcomes in observational oncology cohorts. Evaluated on a semi-synthetic benchmark of $N=1,200$ patients with eight clinical covariates and three emulated treatment arms over a 36-month horizon, the study transparently evaluates LCSF against a standard Cox-based target trial baseline. While LCSF shows lower Harrell's C-index (0.6929 vs. 0.7718) and higher Brier scores (0.1701 vs. 0.1310), it exhibits favorable performance in estimating specific Average Treatment Effects (ATE), such as targeted therapy versus chemotherapy.

---

### 2. Scores
| Metric | Score | Scale |
| :--- | :---: | :---: |
| **Soundness** | 3/4 | 1 (Poor) - 4 (Excellent) |
| **Presentation** | 4/4 | 1 (Poor) - 4 (Excellent) |
| **Contribution / Novelty** | 3/4 | 1 (Poor) - 4 (Excellent) |
| **Overall Recommendation** | **7/10** | 1 (Strong Reject) - 10 (Award Quality) |
| **Reviewer Confidence** | 5/5 | 1 (Low) - 5 (Expert) |

---

### 3. Strengths
- Rigorous integration of target trial emulation principles with modern causal machine learning for survival analysis.
- Transparent and honest reporting of empirical trade-offs, openly discussing instances where the baseline outperforms the proposed method (e.g., C-index and Brier score).
- Comprehensive mathematical formulation detailing stabilized inverse probability of treatment weighting (sIPTW) and honest tree splitting criteria.
- Solid reproducibility documentation, including detailed execution environments, hyperparameter checklists, and scalability benchmarks.

### 4. Weaknesses
- The empirical evaluation is limited to a single semi-synthetic cohort ($N=1,200$), lacking validation on large-scale real-world electronic health record (EHR) databases.
- LCSF exhibits a notable performance drop in predictive discrimination (C-index) and calibration (Brier score) compared to the Cox baseline, indicating potential challenges in leaf-level survival curve estimation.
- Post-weighting maximum ASMD is higher for LCSF (0.2029) than the baseline (0.1347), suggesting that covariate balancing could be further optimized within the causal forest structure.

### 5. Questions for the Authors
- Can the authors elaborate on why LCSF experienced a degradation in discriminative concordance (C-index) despite utilizing an honest splitting criterion designed to capture treatment effect heterogeneity?
- How sensitive are the stabilized propensity weights to the chosen truncation bounds ([0.05, 0.95]), and did extreme weights contribute to the elevated post-weighting maximum ASMD?
- What architectural modifications or tuning strategies would be necessary to scale LCSF to ultra-large real-world datasets with tens of thousands of patients and high-dimensional genomic features?
