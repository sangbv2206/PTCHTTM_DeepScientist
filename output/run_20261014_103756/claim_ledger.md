# Bảng Kiểm Chứng Tuyên Bố Thực Nghiệm (Claim Ledger)

Tất cả các con số xuất hiện trong bài báo được trích xuất trực tiếp từ các khoản mục dưới đây:

| ID | Chỉ số / Metric | Mô hình đề xuất | Giá trị đo được | Baseline so sánh | Chênh lệch (Delta) | Cải thiện (%) | Chiều tốt |
|---|---|---|---|---|---|---|---|
| **R1** | `final_test_accuracy` | Proposed (AFU) | 0.975 | -- | -- | -- | Cao hơn tốt hơn |
| **R2** | `final_test_accuracy` | Proposed (AFU) | 0.975 | Standard Baseline (Gnn Graph Learning) (0.95) | +0.0250 | +2.6% | Cao hơn tốt hơn |
| **R3** | `best_val_accuracy` | Proposed (AFU) | 0.975 | -- | -- | -- | Cao hơn tốt hơn |
| **R4** | `best_val_accuracy` | Proposed (AFU) | 0.975 | Standard Baseline (Gnn Graph Learning) (0.9583) | +0.0167 | +1.7% | Cao hơn tốt hơn |
| **R5** | `final_train_loss` | Proposed (AFU) | 0.0409 | -- | -- | -- | Thấp hơn tốt hơn |
| **R6** | `final_train_loss` | Proposed (AFU) | 0.0409 | Standard Baseline (Gnn Graph Learning) (0.0645) | -0.0236 | -36.5% | Thấp hơn tốt hơn |
| **R7** | `final_val_loss` | Proposed (AFU) | 0.0772 | -- | -- | -- | Thấp hơn tốt hơn |
| **R8** | `final_val_loss` | Proposed (AFU) | 0.0772 | Standard Baseline (Gnn Graph Learning) (0.1375) | -0.0603 | -43.9% | Thấp hơn tốt hơn |
| **R9** | `training_time_sec` | Proposed (AFU) | 0.3695 | -- | -- | -- | Thấp hơn tốt hơn |
| **R10** | `training_time_sec` | Proposed (AFU) | 0.3695 | Standard Baseline (Gnn Graph Learning) (0.1994) | +0.1701 | +85.3% | Thấp hơn tốt hơn |
| **R11** | `epochs` | Proposed (AFU) | 60.0 | -- | -- | -- | Thấp hơn tốt hơn |
| **R12** | `epochs` | Proposed (AFU) | 60.0 | Standard Baseline (Gnn Graph Learning) (60.0) | +0.0000 | +0.0% | Thấp hơn tốt hơn |