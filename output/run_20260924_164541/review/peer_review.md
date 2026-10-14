# Official Conference Peer Review Report

**Paper Title:** Adaptive Grid-Refining Physics-Informed Kolmogorov-Arnold Networks KAN for Scientific Machine Learning with Conformal Uncertainty Quantification  
**Recommendation:** **Weak Accept**  
**Overall Rating:** **7/10**  

---

### 1. Summary of Contributions
The paper introduces Adaptive Grid-Refining Physics-Informed Kolmogorov-Arnold Networks with Conformal Uncertainty Quantification (AG-PIKAN), a unified framework for Scientific Machine Learning (SciML). It aims to solve the limitations of static B-spline grids in traditional KANs when applied to high-dimensional, stiff, and multi-scale differential equations. The method dynamically adapts knot vectors based on the local curvature (Hessian trace) of physical residuals, integrates symbolic pruning for equation discovery, and incorporates a conformal prediction wrapper to provide distribution-free, finite-sample uncertainty coverage.

---

### 2. Scores
| Metric | Score | Scale |
| :--- | :---: | :---: |
| **Soundness** | 3/4 | 1 (Poor) - 4 (Excellent) |
| **Presentation** | 4/4 | 1 (Poor) - 4 (Excellent) |
| **Contribution / Novelty** | 3/4 | 1 (Poor) - 4 (Excellent) |
| **Overall Recommendation** | **7/10** | 1 (Strong Reject) - 10 (Award Quality) |
| **Reviewer Confidence** | 4/5 | 1 (Low) - 5 (Expert) |

---

### 3. Strengths
- Novel integration of adaptive B-spline grid refinement with physics-informed learning, addressing the well-known interpolation bottlenecks and spectral bias of static-grid KANs in stiff PDEs.
- Principled pipeline combining symbolic pruning/regularization for mathematical expression extraction with rigorous conformal uncertainty quantification.
- Clear, mathematically rigorous formulation of the adaptive knot-allocation strategy driven by physical residual Hessians.

### 4. Weaknesses
- The experimental section in the provided source text is incomplete (cuts off abruptly in the problem formulation section), meaning empirical validation details, runtime overhead comparisons, and baseline comparisons are missing from the current draft.
- The computational cost and wall-clock overhead of computing Hessian traces for dynamic grid refinement on large collocation meshes are not explicitly analyzed.
- Lack of explicit discussion regarding how conformal calibration handles temporal-spatial correlations in trajectory forecasting.

### 5. Questions for the Authors
- What is the computational overhead of computing the Hessian trace $	au(x,t)$ for adaptive knot allocation compared to standard static-grid training epochs?
- How does the conformal prediction framework account for temporal and spatial dependencies when computing non-conformity scores on spatio-temporal trajectories?
- Could the authors elaborate on the exact symbolic pruning thresholding criteria used to transition from unconstrained B-splines to clean analytical expressions?
