# Bảng Kiểm Chứng Tuyên Bố Thực Nghiệm (Claim Ledger)

Tất cả các con số xuất hiện trong bài báo được trích xuất trực tiếp từ các khoản mục dưới đây:

| ID | Chỉ số / Metric | Mô hình đề xuất | Giá trị đo được | Baseline so sánh | Chênh lệch (Delta) | Cải thiện (%) | Chiều tốt |
|---|---|---|---|---|---|---|---|
| **R1** | `best_c_index` | Proposed (DTE-LCSF) | 0.6929 | -- | -- | -- | Cao hơn tốt hơn |
| **R2** | `best_c_index` | Proposed (DTE-LCSF) | 0.6929 | Standard Baseline (Causal Trial Emulation) (0.7718) | -0.0789 | -10.2% | Cao hơn tốt hơn |
| **R3** | `brier_score` | Proposed (DTE-LCSF) | 0.1701 | -- | -- | -- | Thấp hơn tốt hơn |
| **R4** | `brier_score` | Proposed (DTE-LCSF) | 0.1701 | Standard Baseline (Causal Trial Emulation) (0.131) | +0.0391 | +29.9% | Thấp hơn tốt hơn |
| **R5** | `unadjusted_max_asmd` | Proposed (DTE-LCSF) | 0.7376 | -- | -- | -- | Thấp hơn tốt hơn |
| **R6** | `unadjusted_max_asmd` | Proposed (DTE-LCSF) | 0.7376 | Standard Baseline (Causal Trial Emulation) (0.7376) | +0.0000 | +0.0% | Thấp hơn tốt hơn |
| **R7** | `post_weighting_max_asmd` | Proposed (DTE-LCSF) | 0.2029 | -- | -- | -- | Thấp hơn tốt hơn |
| **R8** | `post_weighting_max_asmd` | Proposed (DTE-LCSF) | 0.2029 | Standard Baseline (Causal Trial Emulation) (0.1347) | +0.0682 | +50.6% | Thấp hơn tốt hơn |
| **R9** | `ate_dual_vs_sglt2` | Proposed (DTE-LCSF) | -0.074 | -- | -- | -- | Thấp hơn tốt hơn |
| **R10** | `ate_dual_vs_sglt2` | Proposed (DTE-LCSF) | -0.074 | Standard Baseline (Causal Trial Emulation) (-0.0536) | -0.0204 | +38.1% | Thấp hơn tốt hơn |
| **R11** | `ate_triple_vs_sglt2` | Proposed (DTE-LCSF) | -0.0152 | -- | -- | -- | Thấp hơn tốt hơn |
| **R12** | `ate_triple_vs_sglt2` | Proposed (DTE-LCSF) | -0.0152 | Standard Baseline (Causal Trial Emulation) (-0.0232) | +0.0080 | -34.5% | Thấp hơn tốt hơn |
| **R13** | `true_relative_risk_triple` | Proposed (DTE-LCSF) | 0.65 | -- | -- | -- | Thấp hơn tốt hơn |
| **R14** | `true_relative_risk_triple` | Proposed (DTE-LCSF) | 0.65 | Standard Baseline (Causal Trial Emulation) (0.65) | +0.0000 | +0.0% | Thấp hơn tốt hơn |