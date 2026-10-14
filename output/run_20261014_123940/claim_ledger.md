# Bảng Kiểm Chứng Tuyên Bố Thực Nghiệm (Claim Ledger)

Tất cả các con số xuất hiện trong bài báo được trích xuất trực tiếp từ các khoản mục dưới đây:

| ID | Chỉ số / Metric | Mô hình đề xuất | Giá trị đo được | Baseline so sánh | Chênh lệch (Delta) | Thay đổi (%) | Chiều tốt | Đánh giá |
|---|---|---|---|---|---|---|---|---|
| **R1** | `best_c_index` | Proposed Method (Longitudinal Causal Forest) | 0.6929 | -- | -- | -- | Cao hơn tốt hơn | **Độc lập** |
| **R2** | `best_c_index` | Proposed Method (Longitudinal Causal Forest) | 0.6929 | Standard Baseline (Cox Target Trial) (0.7718) | -0.0789 | -10.2% | Cao hơn tốt hơn | **KÉM HƠN (DEGRADED)** |
| **R3** | `brier_score` | Proposed Method (Longitudinal Causal Forest) | 0.1701 | -- | -- | -- | Thấp hơn tốt hơn | **Độc lập** |
| **R4** | `brier_score` | Proposed Method (Longitudinal Causal Forest) | 0.1701 | Standard Baseline (Cox Target Trial) (0.131) | +0.0391 | +29.9% | Thấp hơn tốt hơn | **KÉM HƠN (DEGRADED)** |
| **R5** | `unadjusted_max_asmd` | Proposed Method (Longitudinal Causal Forest) | 0.7376 | -- | -- | -- | Thấp hơn tốt hơn | **Độc lập** |
| **R6** | `unadjusted_max_asmd` | Proposed Method (Longitudinal Causal Forest) | 0.7376 | Standard Baseline (Cox Target Trial) (0.7376) | +0.0000 | +0.0% | Thấp hơn tốt hơn | **Tương đương (0.0%)** |
| **R7** | `post_weighting_max_asmd` | Proposed Method (Longitudinal Causal Forest) | 0.2029 | -- | -- | -- | Thấp hơn tốt hơn | **Độc lập** |
| **R8** | `post_weighting_max_asmd` | Proposed Method (Longitudinal Causal Forest) | 0.2029 | Standard Baseline (Cox Target Trial) (0.1347) | +0.0682 | +50.6% | Thấp hơn tốt hơn | **KÉM HƠN (DEGRADED)** |
| **R9** | `ate_targeted_vs_chemo` | Proposed Method (Longitudinal Causal Forest) | -0.074 | -- | -- | -- | Thấp hơn tốt hơn | **Độc lập** |
| **R10** | `ate_targeted_vs_chemo` | Proposed Method (Longitudinal Causal Forest) | -0.074 | Standard Baseline (Cox Target Trial) (-0.0536) | -0.0204 | +38.1% | Thấp hơn tốt hơn | **TỐT HƠN** |
| **R11** | `ate_immuno_vs_chemo` | Proposed Method (Longitudinal Causal Forest) | -0.0152 | -- | -- | -- | Thấp hơn tốt hơn | **Độc lập** |
| **R12** | `ate_immuno_vs_chemo` | Proposed Method (Longitudinal Causal Forest) | -0.0152 | Standard Baseline (Cox Target Trial) (-0.0232) | +0.0080 | -34.5% | Thấp hơn tốt hơn | **KÉM HƠN (DEGRADED)** |
| **R13** | `ground_truth_hr_immuno` | Proposed Method (Longitudinal Causal Forest) | 0.65 | -- | -- | -- | Thấp hơn tốt hơn | **Độc lập** |
| **R14** | `ground_truth_hr_immuno` | Proposed Method (Longitudinal Causal Forest) | 0.65 | Standard Baseline (Cox Target Trial) (0.65) | +0.0000 | +0.0% | Thấp hơn tốt hơn | **Tương đương (0.0%)** |

### CHỈ DẪN TRUNG THỰC BẮT BUỘC CHO NGÔN NGỮ BÀI BÁO (ZERO-SPIN DIRECTIVES):
Khi viết văn xuôi (Abstract, Introduction, Experiments, Conclusion), bạn PHẢI tuân thủ các quy tắc cực tính sau:
- **best_c_index**: Proposed (0.6929) KÉM HƠN baseline (0.7718) chênh lệch 10.2%. BẮT BUỘC PHẢI THỪA NHẬN TRUNG THỰC: Đây là sự suy giảm / đánh đổi phương pháp luận (performance decline / trade-off). CẤM TUYỆT ĐỐI dùng từ 'outperform', 'superior', 'effectively handles', 'robust' cho chỉ số này!
- **brier_score**: Proposed (0.1701) KÉM HƠN baseline (0.131) chênh lệch 29.9%. BẮT BUỘC PHẢI THỪA NHẬN TRUNG THỰC: Đây là sự suy giảm / đánh đổi phương pháp luận (performance decline / trade-off). CẤM TUYỆT ĐỐI dùng từ 'outperform', 'superior', 'effectively handles', 'robust' cho chỉ số này!
- **post_weighting_max_asmd**: Proposed (0.2029) KÉM HƠN baseline (0.1347) chênh lệch 50.6%. BẮT BUỘC PHẢI THỪA NHẬN TRUNG THỰC: Đây là sự suy giảm / đánh đổi phương pháp luận (performance decline / trade-off). CẤM TUYỆT ĐỐI dùng từ 'outperform', 'superior', 'effectively handles', 'robust' cho chỉ số này!
- **ate_targeted_vs_chemo**: Proposed (-0.074) cải thiện so với baseline (-0.0536) thêm 38.1%. Cho phép dùng từ 'improves' hoặc 'superior'.
- **ate_immuno_vs_chemo**: Proposed (-0.0152) KÉM HƠN baseline (-0.0232) chênh lệch 34.5%. BẮT BUỘC PHẢI THỪA NHẬN TRUNG THỰC: Đây là sự suy giảm / đánh đổi phương pháp luận (performance decline / trade-off). CẤM TUYỆT ĐỐI dùng từ 'outperform', 'superior', 'effectively handles', 'robust' cho chỉ số này!
- **Tham số giả lập**: Nếu có chỉ số Relative Risk hoặc Hazard Ratio ground-truth cố định, phải nêu rõ đây là giá trị chuẩn của mô phỏng (benchmark simulation ground truth), KHÔNG ĐƯỢC viết như thể đây là số đo do mô hình ước lượng được.