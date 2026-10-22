import os
import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset(n_patients=1200, n_covariates=8, csv_path=None):
    """
    Nạp dữ liệu thử nghiệm lâm sàng giả lập (Target Trial Emulation Dataset):
    1. Ưu tiên 1 (Real Cohort Priority): Nạp từ file CSV ('cohort.csv' hoặc 'data.csv').
    2. Ưu tiên 2 (Synthetic Causal Generator): Sinh đoàn hệ giả lập có hiệu ứng chỉ định (Indication Bias) và Hazard thật.
    """
    set_seed(42)
    target_csv = csv_path or ("cohort.csv" if os.path.exists("cohort.csv") else ("data.csv" if os.path.exists("data.csv") else None))
    if target_csv and os.path.exists(target_csv):
        try:
            print(f"[Datasets Loader] Đang nạp dữ liệu đoàn hệ thực tế từ: {target_csv}")
            df = pd.read_csv(target_csv)
            # Giả định cột cuối là event (y), cột áp chót là treatment, các cột trước là covariates
            cols = df.columns.tolist()
            if len(cols) >= 3:
                covariate_names = cols[:-2]
                X = df[covariate_names].values.astype(np.float32)
                treatment = df[cols[-2]].values.astype(np.int64)
                y_event = df[cols[-1]].values.astype(np.float32)

                X_mean = X.mean(axis=0, keepdims=True)
                X_std = X.std(axis=0, keepdims=True) + 1e-6
                X_norm = (X - X_mean) / X_std

                # Khởi tạo giả định propensities đồng đều
                n_arms = len(np.unique(treatment))
                true_propensities = np.ones((len(X), n_arms)) / n_arms
                true_hr = np.ones(n_arms)

                return X, X_norm, treatment, y_event, true_propensities, true_hr, covariate_names
        except Exception as e:
            print(f"[Datasets Loader] Lỗi khi đọc CSV ({e}), kích hoạt Synthetic Cohort Generator.")

    # Synthetic Cohort Generator
    X = np.zeros((n_patients, n_covariates), dtype=np.float32)
    X[:, 0] = np.random.normal(63.5, 9.2, n_patients)
    X[:, 1] = np.random.choice([1, 2, 3, 4], size=n_patients, p=[0.15, 0.25, 0.35, 0.25])
    X[:, 2] = np.random.choice([0, 1, 2, 3], size=n_patients, p=[0.30, 0.40, 0.20, 0.10])
    X[:, 3] = np.random.poisson(1.2, n_patients).astype(np.float32)
    X[:, 4] = np.random.normal(3.8, 0.5, n_patients)
    X[:, 5] = np.random.normal(215.0, 45.0, n_patients)
    X[:, 6] = np.random.exponential(8.5, n_patients)
    X[:, 7] = np.random.binomial(1, 0.45, n_patients)

    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X - X_mean) / X_std

    logits_0 = np.zeros(n_patients)
    logits_1 = 0.5 * X_norm[:, 1] + 0.4 * X_norm[:, 3] + 0.6 * X_norm[:, 6] - 0.2
    logits_2 = 0.8 * X_norm[:, 1] + 0.7 * X_norm[:, 2] + 0.9 * X_norm[:, 6] - 0.5
    logits = np.stack([logits_0, logits_1, logits_2], axis=1)
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    true_propensities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    treatment = np.array([np.random.choice(3, p=true_propensities[i]) for i in range(n_patients)], dtype=np.int64)

    true_hr = np.array([1.0, 0.80, 0.65])
    baseline_log_hazard = (
        0.4 * X_norm[:, 0] + 0.7 * X_norm[:, 1] + 0.6 * X_norm[:, 2]
        + 0.5 * X_norm[:, 3] - 0.5 * X_norm[:, 4] + 0.6 * X_norm[:, 5]
    )
    treatment_effect = np.log(true_hr[treatment])
    total_log_hazard = baseline_log_hazard + treatment_effect

    hazard = np.exp(np.clip(total_log_hazard, -4.0, 4.0)) * 0.15
    event_prob = 1.0 - np.exp(-hazard)
    y_event = np.random.binomial(1, np.clip(event_prob, 0.02, 0.95)).astype(np.float32)

    covariate_names = ["Age", "Tumor Stage", "ECOG Status", "Prior Regimens", "Serum Albumin", "Serum LDH", "TMB Marker", "Smoking History"]
    return X, X_norm, treatment, y_event, true_propensities, true_hr, covariate_names
