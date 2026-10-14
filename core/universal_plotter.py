import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class UniversalPlotter:
 

    @staticmethod
    def _format_metric_name(k: str) -> str:
        name = k.replace("_", " ").title()
        name = name.replace("Ate", "ATE").replace("Hr", "HR").replace("Asmd", "ASMD")
        name = name.replace("C Index", "C-Index").replace("Ndcg", "NDCG").replace("R2", "R²")
        name = name.replace("Mse", "MSE").replace("Acc", "Accuracy")
        return name

    @classmethod
    def generate_grounded_figures(
        cls,
        exp_work_dir: str,
        full_metrics: Dict[str, Dict[str, Any]],
        user_topic: str = ""
    ) -> List[str]:
        """
        Sinh 2-3 đồ thị chuẩn học thuật trực tiếp từ full_metrics và history.json thực tế.
        """
        os.makedirs(exp_work_dir, exist_ok=True)
        generated_figs = []

        # -------------------------------------------------------------
        # 1. Thu thập dữ liệu lịch sử huấn luyện (history.json)
        # -------------------------------------------------------------
        histories = {}
        for run_name in ["run_0", "run_1"]:
            h_path = os.path.join(exp_work_dir, run_name, "history.json")
            if not os.path.exists(h_path):
                h_path = os.path.join(exp_work_dir, "history.json")
            if os.path.exists(h_path):
                try:
                    with open(h_path, "r", encoding="utf-8") as f:
                        histories[run_name] = json.load(f)
                except Exception:
                    pass

        # =============================================================
        # FIGURE 1: Training & Validation Dynamics (Nếu có lịch sử epoch)
        # =============================================================
        fig1_path = os.path.join(exp_work_dir, "figure_1.png")
        if not os.path.exists(fig1_path):
            cls._plot_figure_1_dynamics(exp_work_dir, histories, full_metrics, user_topic)
            if os.path.exists(fig1_path):
                generated_figs.append("figure_1.png")

        # =============================================================
        # FIGURE 2: Grounded Benchmark Comparison (Baseline vs Proposed)
        # =============================================================
        fig2_path = os.path.join(exp_work_dir, "figure_2.png")
        if not os.path.exists(fig2_path):
            cls._plot_figure_2_comparison(exp_work_dir, full_metrics, user_topic)
            if os.path.exists(fig2_path):
                generated_figs.append("figure_2.png")

        # =============================================================
        # FIGURE 3: Metric Delta & Honest Trade-off Analysis
        # =============================================================
        fig3_path = os.path.join(exp_work_dir, "figure_3.png")
        if not os.path.exists(fig3_path):
            cls._plot_figure_3_deltas(exp_work_dir, full_metrics, user_topic)
            if os.path.exists(fig3_path):
                generated_figs.append("figure_3.png")

        return generated_figs

    @classmethod
    def _plot_figure_1_dynamics(
        cls,
        exp_work_dir: str,
        histories: Dict[str, Any],
        full_metrics: Dict[str, Dict[str, Any]],
        user_topic: str
    ):
        target_hist = histories.get("run_1") or histories.get("run_0") or {}
        train_loss = target_hist.get("train_loss") or target_hist.get("loss") or []
        val_loss = target_hist.get("val_loss") or []

        plt.figure(figsize=(9, 4.2), dpi=300)

        if train_loss:
            epochs = list(range(1, len(train_loss) + 1))
            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", lw=2)
            if val_loss and len(val_loss) == len(train_loss):
                plt.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", lw=2, linestyle="--")
            plt.title("Optimization Convergence Dynamics", fontsize=11, fontweight="bold")
            plt.xlabel("Epoch / Iteration", fontsize=10)
            plt.ylabel("Loss Criterion", fontsize=10)
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(frameon=True)

            # Subplot 2: Metric trajectory hoặc summary bar
            plt.subplot(1, 2, 2)
            metric_keys = [k for k in target_hist.keys() if k not in ["train_loss", "val_loss", "loss", "predictions", "targets"]]
            plotted_metric = False
            for mk in metric_keys[:2]:
                m_vals = target_hist[mk]
                if isinstance(m_vals, list) and len(m_vals) == len(epochs) and all(isinstance(v, (int, float)) for v in m_vals):
                    plt.plot(epochs, m_vals, label=cls._format_metric_name(mk), lw=2)
                    plotted_metric = True
            if plotted_metric:
                plt.title("Empirical Metric Trajectory", fontsize=11, fontweight="bold")
                plt.xlabel("Epoch", fontsize=10)
                plt.ylabel("Metric Score", fontsize=10)
                plt.grid(True, linestyle=":", alpha=0.6)
                plt.legend(frameon=True)
            else:
                # Nếu không có metric theo epoch, vẽ phân phối loss
                plt.hist(train_loss, bins=15, color="#2ca02c", alpha=0.7, edgecolor="black")
                plt.title("Empirical Loss Distribution", fontsize=11, fontweight="bold")
                plt.xlabel("Loss Value", fontsize=10)
                plt.ylabel("Frequency", fontsize=10)
                plt.grid(True, linestyle=":", alpha=0.6)
        else:
            # Fallback nếu không có history theo epoch (ví dụ: mô hình cây, random forest)
            plt.subplot(1, 1, 1)
            models = list(full_metrics.keys())
            first_m = list(full_metrics[models[0]].keys())[0] if models and full_metrics[models[0]] else "Metric"
            vals = [full_metrics[m].get(first_m, 0.0) for m in models]
            bars = plt.bar(models, vals, color=["#4a90e2", "#2ecc71"][:len(models)], width=0.45, edgecolor="black")
            plt.title(f"Empirical Evaluation Summary ({cls._format_metric_name(first_m)})", fontsize=12, fontweight="bold")
            plt.ylabel(cls._format_metric_name(first_m), fontsize=11)
            plt.grid(axis="y", linestyle=":", alpha=0.6)
            for bar in bars:
                y = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, y * 0.95, f"{y:.4f}", ha="center", va="top", color="white", fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(exp_work_dir, "figure_1.png"), dpi=300)
        plt.close()

    @classmethod
    def _plot_figure_2_comparison(
        cls,
        exp_work_dir: str,
        full_metrics: Dict[str, Dict[str, Any]],
        user_topic: str
    ):
        models = list(full_metrics.keys())
        if not models:
            return

        # Lấy danh sách metrics chung
        all_keys = []
        for m in models:
            for k, v in full_metrics[m].items():
                if isinstance(v, (int, float)) and k not in all_keys:
                    all_keys.append(k)

        if not all_keys:
            return

        # Chọn tối đa 3-4 metrics quan trọng nhất để vẽ bar plot
        selected_keys = all_keys[:4]
        n_metrics = len(selected_keys)
        n_models = len(models)

        fig, ax = plt.subplots(figsize=(max(6.5, n_metrics * 2.2), 4.5), dpi=300)
        x = np.arange(n_metrics)
        width = 0.35 if n_models == 2 else 0.8 / max(n_models, 1)
        colors = ["#4a90e2", "#27ae60", "#e67e22", "#9b59b6"]

        for i, m_name in enumerate(models):
            vals = [float(full_metrics[m_name].get(k, 0.0)) for k in selected_keys]
            offset = (i - (n_models - 1) / 2) * width
            label = "Baseline" if "baseline" in m_name.lower() or i == 0 else "Proposed Framework"
            bars = ax.bar(x + offset, vals, width, label=label, color=colors[i % len(colors)], edgecolor="black", linewidth=0.8)

            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -10),
                    textcoords="offset points",
                    ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=8.5, fontweight="bold"
                )

        ax.set_ylabel("Empirical Metric Value", fontsize=11, fontweight="bold")
        ax.set_title("Standardized Benchmark Evaluation (Grounded in Verified Logs)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([cls._format_metric_name(k) for k in selected_keys], fontsize=9.5, fontweight="bold")
        ax.legend(frameon=True, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(exp_work_dir, "figure_2.png"), dpi=300)
        plt.close()

    @classmethod
    def _plot_figure_3_deltas(
        cls,
        exp_work_dir: str,
        full_metrics: Dict[str, Dict[str, Any]],
        user_topic: str
    ):
        models = list(full_metrics.keys())
        if len(models) < 2:
            return

        baseline_name = models[0]
        proposed_name = models[1]
        b_metrics = full_metrics[baseline_name]
        p_metrics = full_metrics[proposed_name]

        common_keys = [k for k in b_metrics if k in p_metrics and isinstance(b_metrics[k], (int, float)) and isinstance(p_metrics[k], (int, float))]
        if not common_keys:
            return

        deltas = []
        labels = []
        colors = []
        for k in common_keys[:6]:
            b_val = float(b_metrics[k])
            p_val = float(p_metrics[k])
            if abs(b_val) > 1e-6:
                pct = ((p_val - b_val) / abs(b_val)) * 100.0
            else:
                pct = (p_val - b_val) * 100.0

            # Đối với metric mà lower-is-better (loss, asmd, brier, error)
            is_lower_better = any(w in k.lower() for w in ["loss", "asmd", "brier", "mse", "mae", "error"])
            is_improvement = (pct < 0) if is_lower_better else (pct > 0)

            deltas.append(pct)
            labels.append(cls._format_metric_name(k))
            colors.append("#27ae60" if is_improvement else "#d35400")

        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
        bars = ax.barh(labels, deltas, color=colors, height=0.55, edgecolor="black", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Relative Metric Difference vs Baseline (%)", fontsize=10, fontweight="bold")
        ax.set_title("Empirical Delta & Scientific Trade-Off Analysis", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        for bar in bars:
            w = bar.get_width()
            xpos = w + (1.0 if w >= 0 else -1.0)
            ha = "left" if w >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{w:+.1f}%", va="center", ha=ha, fontsize=8.5, fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(exp_work_dir, "figure_3.png"), dpi=300)
        plt.close()

    @classmethod
    def ensure_grounded_figures(
        cls,
        exp_work_dir: str,
        full_metrics: Dict[str, Dict[str, Any]],
        user_topic: str = ""
    ) -> List[str]:
        """
        Kiểm tra và sinh bổ sung nếu thiếu hình vẽ, đảm bảo tối thiểu figure_1.png và figure_2.png
        luôn tồn tại và gắn chặt với dữ liệu thực nghiệm thực tế.
        """
        existing = [f for f in os.listdir(exp_work_dir) if f.endswith(".png")]
        if "figure_1.png" not in existing or "figure_2.png" not in existing:
            print("[UniversalPlotter] Phát hiện thiếu hoặc lỗi đồ thị thực nghiệm. Đang tự động sinh biểu đồ chuẩn hóa từ log thực tế...")
            cls.generate_grounded_figures(exp_work_dir, full_metrics, user_topic)

        final_figs = sorted([f for f in os.listdir(exp_work_dir) if f.endswith(".png")])
        print(f"[UniversalPlotter] Verified figure manifest ({len(final_figs)} figures): {final_figs}")
        return final_figs
