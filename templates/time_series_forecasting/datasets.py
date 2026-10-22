import json
import os
import sys
import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")



def load_domain_context(context_path: str = "domain_context.json") -> dict:
    """Tự động tải ngữ cảnh bài toán do Pipeline điều phối inject vào."""
    if os.path.exists(context_path):
        try:
            with open(context_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "domain": "generic_sequential",
        "series_name": "Sequential Trajectory Forecast",
        "unit_label": "Normalized Amplitude",
        "target_name": "Continuous Target State",
        "feature_names": ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]
    }


def get_dataset(
    domain_ctx: dict = None,
    num_timesteps: int = 1500,
    seq_len: int = 24,
    seed: int = 42,
    csv_path: str = None
):
    """
    Bộ nạp & sinh dữ liệu chuỗi thời gian thích ứng:
    1. Ưu tiên 1 (Real Data Priority): Nếu có file CSV thật (ví dụ 'data.csv' hoặc đường dẫn được chỉ định), nạp dữ liệu thật.
    2. Ưu tiên 2 (Adaptive Physics Simulator): Tự động sinh dữ liệu theo đặc trưng bài toán (thủy văn, năng lượng, môi trường).
    """
    if domain_ctx is None:
        domain_ctx = load_domain_context()

    # Kiểm tra xem có file CSV thật nào trong thư mục làm việc hay không
    target_csv = csv_path or ("data.csv" if os.path.exists("data.csv") else ("dataset.csv" if os.path.exists("dataset.csv") else None))
    if target_csv and os.path.exists(target_csv):
        try:
            print(f"[Datasets Loader] Đang nạp dữ liệu thực tế từ: {target_csv}")
            df = pd.read_csv(target_csv)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                features_data = df[numeric_cols[:-1]].values.astype(np.float32)
                target_data = df[numeric_cols[-1]].values.astype(np.float32)

                # Min-max scaling
                f_min, f_max = features_data.min(axis=0), features_data.max(axis=0)
                features = (features_data - f_min) / (f_max - f_min + 1e-8)
                t_min, t_max = target_data.min(), target_data.max()
                y_series = (target_data - t_min) / (t_max - t_min + 1e-8)

                X_list, y_list = [], []
                n_pts = len(features)
                for i in range(n_pts - seq_len):
                    X_list.append(features[i : i + seq_len])
                    y_list.append(y_series[i + seq_len])

                X = torch.tensor(np.array(X_list), dtype=torch.float32)
                y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1)

                n_samples = len(X)
                n_train = int(0.7 * n_samples)
                n_val = int(0.15 * n_samples)

                return (
                    X[:n_train], y[:n_train],
                    X[n_train : n_train + n_val], y[n_train : n_train + n_val],
                    X[n_train + n_val :], y[n_train + n_val :],
                    domain_ctx
                )
        except Exception as e:
            print(f"[Datasets Loader] Cảnh báo khi đọc CSV ({e}), chuyển sang chế độ Domain Simulator.")

    # Mô phỏng động lực học thích ứng (Adaptive Physics Simulation)
    np.random.seed(seed)
    torch.manual_seed(seed)

    t = np.arange(num_timesteps, dtype=np.float32)
    domain = domain_ctx.get("domain", "generic_sequential")

    def smooth(arr, window=7):
        kernel = np.ones(window, dtype=np.float32) / window
        return np.convolve(arr, kernel, mode="same")

    def minmax(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)

    if domain == "hydrology_meteorology":
        storm_pulses = np.zeros(num_timesteps, dtype=np.float32)
        for burst_time in [140, 360, 620, 880, 1120, 1380]:
            width = np.random.uniform(16.0, 32.0)
            intensity = np.random.uniform(55.0, 110.0)
            storm_pulses += intensity * np.exp(-0.5 * ((t - burst_time) / width) ** 2)
        base_rain = np.maximum(0.0, 6.0 * np.sin(2 * np.pi * t / 72.0) + np.random.normal(0, 3.0, num_timesteps))
        rainfall = storm_pulses + base_rain

        runoff_lag = smooth(rainfall, window=15) * 0.085
        upstream_stage = 2.2 + runoff_lag + np.random.normal(0, 0.08, num_timesteps)

        soil_moisture = 35.0 + smooth(rainfall, window=35) * 0.45
        soil_moisture = np.clip(soil_moisture, 20.0, 98.0)

        terrain_slope = 145.0 + 35.0 * np.sin(2 * np.pi * t / 200.0) + np.random.normal(0, 1.2, num_timesteps)

        water_level = 1.35 + 0.68 * upstream_stage + 0.022 * (soil_moisture - 45.0) + 0.016 * smooth(rainfall, window=7)
        noise = np.random.normal(0, 0.07, num_timesteps)
        target_series = np.maximum(0.6, water_level + noise)

        features = np.stack([minmax(rainfall), minmax(upstream_stage), minmax(soil_moisture), minmax(terrain_slope)], axis=1)
        y_series = minmax(target_series)

    elif domain == "energy_power":
        hour = t % 24
        day = (t // 24) % 7
        temp = 22.0 + 7.0 * np.sin(2 * np.pi * (hour - 9) / 24) + np.random.normal(0, 1.0, num_timesteps)
        base_load = 1.2
        morning_peak = 2.0 * np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
        evening_peak = 3.5 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
        weekend_factor = np.where(day >= 5, 1.25, 1.0)
        temp_cooling = np.maximum(temp - 25.0, 0.0) * 0.15
        noise = np.random.normal(0, 0.25, num_timesteps)

        total_demand = (base_load + morning_peak + evening_peak) * weekend_factor + temp_cooling + noise
        total_demand = np.maximum(0.5, total_demand)

        f0_temp = temp
        f1_humidity = 65.0 - 15.0 * np.sin(2 * np.pi * (hour - 10) / 24) + np.random.normal(0, 3.0, num_timesteps)
        f2_industrial = np.where((hour >= 8) & (hour <= 18) & (day < 5), 2.8, 0.4) + np.random.normal(0, 0.2, num_timesteps)
        f3_solar = np.maximum(0.0, 4.5 * np.sin(np.pi * (hour - 6) / 12)) * (1.0 - 0.2 * np.random.uniform(0, 1, num_timesteps))

        features = np.stack([minmax(f0_temp), minmax(f1_humidity), minmax(f2_industrial), minmax(f3_solar)], axis=1)
        y_series = minmax(total_demand)

    elif domain == "air_pollution_environment":
        wind = np.maximum(0.5, 3.5 + 2.0 * np.sin(2 * np.pi * t / 48.0) + np.random.normal(0, 0.8, num_timesteps))
        humidity = 70.0 + 18.0 * np.sin(2 * np.pi * t / 24.0) + np.random.normal(0, 4.0, num_timesteps)
        traffic = np.maximum(0.2, 1.0 + np.sin(2 * np.pi * (t % 24 - 7) / 24) + np.random.normal(0, 0.2, num_timesteps))
        temp = 25.0 + 6.0 * np.sin(2 * np.pi * (t % 24 - 11) / 24)

        pm25 = 35.0 + 25.0 * (traffic / (wind + 0.5)) + 0.15 * humidity + np.random.normal(0, 3.5, num_timesteps)
        pm25 = np.maximum(5.0, pm25)

        features = np.stack([minmax(wind), minmax(humidity), minmax(traffic), minmax(temp)], axis=1)
        y_series = minmax(pm25)

    else:
        f0 = np.sin(2 * np.pi * t / 48.0) + 0.4 * np.sin(2 * np.pi * t / 14.0)
        f1 = np.cos(2 * np.pi * t / 32.0) + np.random.normal(0, 0.15, num_timesteps)
        f2 = np.cumsum(np.random.normal(0, 0.04, num_timesteps))
        f3 = smooth(f0 * f1, window=10)
        target = 0.45 * f0 + 0.3 * f1 + 0.15 * minmax(f2) + np.random.normal(0, 0.08, num_timesteps)

        features = np.stack([minmax(f0), minmax(f1), minmax(f2), minmax(f3)], axis=1)
        y_series = minmax(target)

    X_list, y_list = [], []
    for i in range(num_timesteps - seq_len):
        X_list.append(features[i : i + seq_len])
        y_list.append(y_series[i + seq_len])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1)

    n_samples = len(X)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    train_X, train_y = X[:n_train], y[:n_train]
    val_X, val_y = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    test_X, test_y = X[n_train + n_val :], y[n_train + n_val :]

    return train_X, train_y, val_X, val_y, test_X, test_y, domain_ctx
