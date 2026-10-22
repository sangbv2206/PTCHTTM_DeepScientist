import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    out_dir = "run_1" if os.path.exists("run_1") else ("run_0" if os.path.exists("run_0") else ".")

    history_path = os.path.join(out_dir, "history.json")
    final_info_path = os.path.join(out_dir, "final_info.json")
    domain_context_path = "domain_context.json"

    # 1. Đọc metadata ngữ cảnh miền (Domain Context) & thông số thực nghiệm
    meta = {}
    if os.path.exists(domain_context_path):
        try:
            with open(domain_context_path, "r", encoding="utf-8") as f:
                meta.update(json.load(f))
        except Exception:
            pass

    if os.path.exists(final_info_path):
        try:
            with open(final_info_path, "r", encoding="utf-8") as f:
                meta.update(json.load(f))
        except Exception:
            pass

    series_title = meta.get("series_name", "Sequential Trajectory Forecast")
    unit_label = meta.get("unit_label", "Normalized Amplitude")
    domain_name = meta.get("domain", "General")

    # 2. Đọc lịch sử huấn luyện
    history = {"train_loss": [], "val_loss": [], "test_r2": [], "predictions": [], "targets": []}
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    # =========================================================================
    # FIGURE 1: Convergence Curves & Actual vs Predicted Sequential Trajectory
    # =========================================================================
    plt.figure(figsize=(10, 4.5), dpi=300)

    # Subplot 1: Convergence Dynamics
    plt.subplot(1, 2, 1)
    if epochs:
        plt.plot(epochs, history["train_loss"], label="Train Loss (MSE)", color="#1f77b4", lw=2)
        plt.plot(epochs, history["val_loss"], label="Val Loss (MSE)", color="#ff7f0e", lw=2, linestyle="--")
    plt.title("(a) Convergence Dynamics", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    # Subplot 2: Actual vs Predicted Curve
    plt.subplot(1, 2, 2)
    preds = history.get("predictions", [])[:75]
    targets = history.get("targets", [])[:75]
    if preds and targets:
        time_steps = list(range(len(preds)))
        plt.plot(time_steps, targets, label="Ground Truth Observed", color="#2ca02c", lw=2)
        plt.plot(time_steps, preds, label="Model Prediction", color="#d62728", lw=2, linestyle="--")
    clean_series_title = series_title if len(series_title) <= 30 else series_title[:27] + "..."
    plt.title(f"(b) {clean_series_title}", fontsize=11, fontweight="bold")
    plt.xlabel("Sequential Horizon Steps", fontsize=10)
    plt.ylabel(f"Normalized Level ({unit_label})", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("figure_1.png", dpi=300)
    plt.close()
    print(f"[PLOTTING] Saved figure_1.png ({series_title})")

    # =========================================================================
    # FIGURE 2: Baseline vs Proposed Innovation Benchmark Comparison
    # =========================================================================
    baseline_r2 = None
    proposed_r2 = None
    baseline_name = "Standard Baseline"
    metric_label = "Determination Score R² (%)"

    if os.path.exists("run_0/final_info.json"):
        try:
            with open("run_0/final_info.json", "r", encoding="utf-8") as f:
                b_info = json.load(f)
                baseline_r2 = float(b_info.get("best_test_r2", b_info.get("r2", 0.0)))
                if b_info.get("model_name"):
                    baseline_name = b_info["model_name"]
        except Exception:
            pass

    if os.path.exists("run_1/final_info.json"):
        try:
            with open("run_1/final_info.json", "r", encoding="utf-8") as f:
                p_info = json.load(f)
                proposed_r2 = float(p_info.get("best_test_r2", p_info.get("r2", 0.0)))
        except Exception:
            pass
    elif os.path.exists(final_info_path):
        try:
            with open(final_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
                proposed_r2 = float(info.get("best_test_r2", info.get("r2", 0.0)))
        except Exception:
            pass

    if baseline_r2 is not None and proposed_r2 is not None:
        models = [baseline_name, "Proposed Method"]
        r2_scores = [
            baseline_r2 * 100 if baseline_r2 <= 1.0 else baseline_r2,
            proposed_r2 * 100 if proposed_r2 <= 1.0 else proposed_r2
        ]
        colors = ["#4a90e2", "#27ae60"]
    elif proposed_r2 is not None:
        models = ["Proposed Method"]
        r2_scores = [proposed_r2 * 100 if proposed_r2 <= 1.0 else proposed_r2]
        colors = ["#27ae60"]
    elif baseline_r2 is not None:
        models = [baseline_name]
        r2_scores = [baseline_r2 * 100 if baseline_r2 <= 1.0 else baseline_r2]
        colors = ["#4a90e2"]
    else:
        models = []
        r2_scores = []
        colors = []

    if models:
        plt.figure(figsize=(7.0, 4.5), dpi=300)
        bars = plt.bar(models, r2_scores, color=colors, width=0.45 if len(models) > 1 else 0.3, edgecolor="black", linewidth=0.8)
        plt.ylabel(metric_label, fontsize=11, fontweight="bold")
        plt.title(f"Benchmark Generalization Comparison ({clean_series_title})", fontsize=11, fontweight="bold")
        
        # Tự động điều chỉnh trục y nếu chênh lệch rất nhỏ để biểu đồ phản ánh đúng thực tế
        if len(r2_scores) > 1 and abs(r2_scores[1] - r2_scores[0]) < 2.0:
            y_min = max(0.0, min(r2_scores) - 1.5)
            y_max = max(r2_scores) + 1.5
            plt.ylim(y_min, y_max)
        else:
            plt.ylim(0, max(max(r2_scores) * 1.25, 100.0) if r2_scores else 100)
        plt.grid(axis="y", linestyle=":", alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05, f"{yval:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

        plt.tight_layout()
        plt.savefig("figure_2.png", dpi=300)
        plt.close()
        print("[PLOTTING] Saved figure_2.png (Forecasting Benchmark Comparison)")

    # =========================================================================
    # FIGURE 3: Ground Truth vs Prediction Scatter & Residual Distribution
    # =========================================================================
    preds_arr = np.array(history.get("predictions", []))
    targets_arr = np.array(history.get("targets", []))

    if len(preds_arr) > 0 and len(targets_arr) > 0 and len(preds_arr) == len(targets_arr):
        residuals = targets_arr - preds_arr
        plt.figure(figsize=(10.5, 4.5), dpi=300)

        # Subplot 1: Actual vs Predicted Scatter Correlation Plot (y = x reference)
        plt.subplot(1, 2, 1)
        plt.scatter(targets_arr, preds_arr, color="#1f77b4", alpha=0.65, edgecolors="none", s=25, label="Test Samples")
        min_v = min(targets_arr.min(), preds_arr.min())
        max_v = max(targets_arr.max(), preds_arr.max())
        plt.plot([min_v, max_v], [min_v, max_v], color="#d62728", lw=2, linestyle="--", label="Ideal 1:1 Identity Line")
        plt.title("(a) Actual vs. Predicted Correlation Fit", fontsize=11, fontweight="bold")
        plt.xlabel(f"Ground Truth ({unit_label})", fontsize=10)
        plt.ylabel(f"Model Prediction ({unit_label})", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()

        # Subplot 2: Forecast Residual Distribution
        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=25, density=True, color="#2ca02c", alpha=0.65, edgecolor="black", label="Error Residuals")
        kde_x = np.linspace(min(residuals), max(residuals), 100)
        from scipy.stats import norm
        try:
            mu, std = norm.fit(residuals)
            plt.plot(kde_x, norm.pdf(kde_x, mu, std), color="#d62728", lw=2, label=f"Normal Fit (σ={std:.3f})")
        except Exception:
            pass
        plt.axvline(x=0, color="black", linestyle="--", alpha=0.7)
        plt.title("(b) Forecast Residual Error Distribution", fontsize=11, fontweight="bold")
        plt.xlabel("Prediction Error (Residual)", fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()

        plt.tight_layout()
        plt.savefig("figure_3.png", dpi=300)
        plt.close()
        print("[PLOTTING] Saved figure_3.png (Correlation Scatter & Residual Dynamics)")


if __name__ == "__main__":
    main()
