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
    """
    Bộ vẽ đồ thị phổ quát thích ứng (Adaptive Universal Plotter) cho DeepScientist:
    - NGUYÊN TẮC 1 (Tôn trọng đặc thù miền): Nếu template tự sinh được ảnh hợp lệ
      (bất kể 1, 2, 4 hay 5 ảnh, ví dụ: ảnh phân tử, ảnh scatter diffusion, ảnh stroke),
      hệ thống giữ nguyên 100%, tuyệt đối không can thiệp hay ghi đè.
    - NGUYÊN TẮC 2 (Cứu cánh thích ứng linh hoạt - Dynamic Fallback):
      Chỉ can thiệp khi template KHÔNG sinh được ảnh nào (bị crash hoặc thiếu plot.py).
      Số lượng ảnh sinh ra hoàn toàn thích ứng theo dữ liệu thật:
        * Chỉ có 1 metric kết quả -> Sinh đúng 1 biểu đồ so sánh.
        * Có thêm lịch sử epoch (loss/acc) -> Sinh thêm 1 biểu đồ hội tụ.
        * Có nhiều metric đối chiếu đa chiều -> Mới sinh thêm biểu đồ Trade-off Delta.
    - 100% Zero-Hardcode: Tuyệt đối không chứa số ảo hay mảng số gán cứng.
    """

    @staticmethod
    def _format_metric_name(k: str) -> str:
        name = k.replace("_", " ").title()
        name = name.replace("Ate", "ATE").replace("Hr", "HR").replace("Asmd", "ASMD")
        name = name.replace("C Index", "C-Index").replace("Ndcg", "NDCG").replace("R2", "R²")
        name = name.replace("Mse", "MSE").replace("Acc", "Accuracy")
        return name

    @classmethod
    def ensure_grounded_figures(
        cls,
        exp_work_dir: str,
        full_metrics: Dict[str, Dict[str, Any]],
        user_topic: str = ""
    ) -> List[str]:
        """
        Kiểm tra và bảo đảm hình ảnh cho bài báo:
        - Nếu template đã sinh được ảnh hợp lệ (> 0 ảnh .png không rỗng), giữ nguyên 100%.
        - Nếu template không có ảnh nào (0 ảnh), kích hoạt vẽ cứu cánh thích ứng theo dữ liệu.
        """
        existing = [
            f for f in os.listdir(exp_work_dir)
            if f.endswith(".png") and os.path.getsize(os.path.join(exp_work_dir, f)) > 1000
        ]

        if existing:
            print(f"[UniversalPlotter] Template da sinh thanh cong {len(existing)} do thi dac thu mien ({existing}). Giu nguyen 100%, khong can thiep.")
            return sorted(existing)

        print("[UniversalPlotter] Template khong sinh duoc do thi nao (0 anh). Dang tu dong kich hoat bo ve thich ung theo log...")
        generated_figs = cls.generate_grounded_figures(exp_work_dir, full_metrics, user_topic)
        final_figs = sorted([f for f in os.listdir(exp_work_dir) if f.endswith(".png")])
        print(f"[UniversalPlotter] Verified figure manifest ({len(final_figs)} figures): {final_figs}")
        return final_figs

    @classmethod
    def generate_grounded_figures(
        cls,
        exp_work_dir: str,
        full_metrics: Dict[str, Dict[str, Any]],
        user_topic: str = ""
    ) -> List[str]:
        """
        Sinh số lượng đồ thị hoàn toàn linh hoạt (1, 2, hoặc 3 ảnh) dựa trên bản chất dữ liệu thật.
        """
        os.makedirs(exp_work_dir, exist_ok=True)
        generated_figs = []

        # 1. Thu thập dữ liệu lịch sử huấn luyện (history.json)
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

        target_hist = histories.get("run_1") or histories.get("run_0") or {}
        train_loss = target_hist.get("train_loss") or target_hist.get("loss") or []

        # -----------------------------------------------------------------
        # HÌNH A: Động học tối ưu hóa (Chỉ sinh nếu thực sự có lịch sử epoch)
        # -----------------------------------------------------------------
        has_epochs = bool(isinstance(train_loss, list) and len(train_loss) > 1)
        if has_epochs:
            fig_a_name = f"figure_{len(generated_figs) + 1}.png"
            cls._plot_figure_dynamics(exp_work_dir, fig_a_name, target_hist, user_topic)
            generated_figs.append(fig_a_name)

        # -----------------------------------------------------------------
        # HÌNH B: So sánh Benchmark định lượng (Sinh nếu có metrics)
        # -----------------------------------------------------------------
        if full_metrics:
            fig_b_name = f"figure_{len(generated_figs) + 1}.png"
            cls._plot_figure_comparison(exp_work_dir, fig_b_name, full_metrics, user_topic)
            generated_figs.append(fig_b_name)

        # -----------------------------------------------------------------
        # HÌNH C: Phân tích Đánh đổi / Delta (Chỉ sinh khi có >= 2 models và >= 2 metrics)
        # -----------------------------------------------------------------
        models = list(full_metrics.keys())
        can_plot_delta = False
        if len(models) >= 2:
            m0, m1 = models[0], models[1]
            shared_metrics = [
                k for k in full_metrics[m0]
                if k in full_metrics[m1] and isinstance(full_metrics[m0][k], (int, float))
            ]
            if len(shared_metrics) >= 2:
                can_plot_delta = True

        if can_plot_delta:
            fig_c_name = f"figure_{len(generated_figs) + 1}.png"
            cls._plot_figure_deltas(exp_work_dir, fig_c_name, full_metrics, user_topic)
            generated_figs.append(fig_c_name)

        return generated_figs

    @classmethod
    def _plot_figure_dynamics(
        cls,
        exp_work_dir: str,
        fig_name: str,
        target_hist: Dict[str, Any],
        user_topic: str
    ):
        train_loss = target_hist.get("train_loss") or target_hist.get("loss") or []
        val_loss = target_hist.get("val_loss") or []
        epochs = list(range(1, len(train_loss) + 1))

        plt.figure(figsize=(9, 4.2), dpi=300)

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", lw=2)
        if val_loss and len(val_loss) == len(train_loss):
            plt.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", lw=2, linestyle="--")
        plt.title("Optimization Convergence Dynamics", fontsize=11, fontweight="bold")
        plt.xlabel("Epoch / Iteration", fontsize=10)
        plt.ylabel("Loss Criterion", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True)

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
            plt.hist(train_loss, bins=15, color="#2ca02c", alpha=0.7, edgecolor="black")
            plt.title("Empirical Loss Distribution", fontsize=11, fontweight="bold")
            plt.xlabel("Loss Value", fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(exp_work_dir, fig_name), dpi=300)
        plt.close()

    @classmethod
    def _plot_figure_comparison(
        cls,
        exp_work_dir: str,
        fig_name: str,
        full_metrics: Dict[str, Dict[str, Any]],
        user_topic: str
    ):
        models = list(full_metrics.keys())
        if not models:
            return

        all_keys = []
        for m in models:
            for k, v in full_metrics[m].items():
                if isinstance(v, (int, float)) and k not in all_keys:
                    all_keys.append(k)

        if not all_keys:
            return

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
        plt.savefig(os.path.join(exp_work_dir, fig_name), dpi=300)
        plt.close()

    @classmethod
    def _plot_figure_deltas(
        cls,
        exp_work_dir: str,
        fig_name: str,
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

        common_keys = [
            k for k in b_metrics
            if k in p_metrics and isinstance(b_metrics[k], (int, float)) and isinstance(p_metrics[k], (int, float))
        ]
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
        plt.savefig(os.path.join(exp_work_dir, fig_name), dpi=300)
        plt.close()
