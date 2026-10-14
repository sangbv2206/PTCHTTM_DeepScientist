# Bảng Kiểm Chứng Tuyên Bố Thực Nghiệm (Claim Ledger)

Tất cả các con số xuất hiện trong bài báo được trích xuất trực tiếp từ các khoản mục dưới đây:

| ID | Chỉ số / Metric | Mô hình đề xuất | Giá trị đo được | Baseline so sánh | Chênh lệch (Delta) | Cải thiện (%) | Chiều tốt |
|---|---|---|---|---|---|---|---|
| **R1** | `final_test_accuracy` | Proposed (AFU) | 0.9833 | -- | -- | -- | Cao hơn tốt hơn |
| **R2** | `final_test_accuracy` | Proposed (AFU) | 0.9833 | Standard Baseline (Gnn Graph Learning) (0.95) | +0.0333 | +3.5% | Cao hơn tốt hơn |
| **R3** | `best_val_accuracy` | Proposed (AFU) | 0.975 | -- | -- | -- | Cao hơn tốt hơn |
| **R4** | `best_val_accuracy` | Proposed (AFU) | 0.975 | Standard Baseline (Gnn Graph Learning) (0.9583) | +0.0167 | +1.7% | Cao hơn tốt hơn |
| **R5** | `final_train_loss` | Proposed (AFU) | 0.0313 | -- | -- | -- | Thấp hơn tốt hơn |
| **R6** | `final_train_loss` | Proposed (AFU) | 0.0313 | Standard Baseline (Gnn Graph Learning) (0.0645) | -0.0332 | -51.5% | Thấp hơn tốt hơn |
| **R7** | `final_val_loss` | Proposed (AFU) | 0.0803 | -- | -- | -- | Thấp hơn tốt hơn |
| **R8** | `final_val_loss` | Proposed (AFU) | 0.0803 | Standard Baseline (Gnn Graph Learning) (0.1375) | -0.0571 | -41.6% | Thấp hơn tốt hơn |
| **R9** | `training_time_sec` | Proposed (AFU) | 0.2856 | -- | -- | -- | Thấp hơn tốt hơn |
| **R10** | `training_time_sec` | Proposed (AFU) | 0.2856 | Standard Baseline (Gnn Graph Learning) (0.1728) | +0.1127 | +65.2% | Thấp hơn tốt hơn |
| **R11** | `epochs` | Proposed (AFU) | 60.0 | -- | -- | -- | Thấp hơn tốt hơn |
| **R12** | `epochs` | Proposed (AFU) | 60.0 | Standard Baseline (Gnn Graph Learning) (60.0) | +0.0000 | +0.0% | Thấp hơn tốt hơn |