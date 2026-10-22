import os
import sys
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Optional
from core.claim_ledger import flatten_template_metrics, compute_mean_std, is_higher_is_better, EXCLUDE_COLUMNS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class UniversalPlotter:
    """
    Bộ vẽ đồ thị cứu cánh đa năng (Grounded Fallback Plotter) cho DeepScientist:
    - Nếu template đã tự sinh được ảnh (> 0 file .png), giữ nguyên 100%, không can thiệp.
    - Nếu template không sinh được ảnh (0 ảnh hoặc plot.py bị lỗi), tự động trích xuất
      số liệu thực từ log (history.json, training_dynamics, full_metrics) để vẽ đồ thị
      khoa học chuẩn mực, đúng tỉ lệ và tương thích hoàn toàn với LaTeX Writer.
    """

    @staticmethod
    def _format_metric_name(k: str) -> str:
        name = k.replace("_", " ").replace("best ", "").replace("final ", "").title()
        name = name.replace("Ate", "ATE").replace("Hr", "HR").replace("Asmd", "ASMD")
        name = name.replace("C Index", "C-Index").replace("Ndcg", "NDCG").replace("R2", "R²")
        name = name.replace("Mse", "MSE").replace("Mae", "MAE").replace("Acc", "Accuracy")
        name = name.replace("Td Loss", "TD Loss").replace("Ppl", "PPL")
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
        - Nếu template không có ảnh nào (0 ảnh), kích hoạt vẽ cứu cánh thích ứng theo dữ liệu thật.
        """
        existing = [
            f for f in os.listdir(exp_work_dir)
            if f.endswith(".png") and os.path.getsize(os.path.join(exp_work_dir, f)) > 1000
        ]

        if existing:
            print(f"[UniversalPlotter] Template đã sinh thành công {len(existing)} đồ thị ({existing}). Bảo toàn 100% dữ liệu gốc.")
            return sorted(existing)

        print("[UniversalPlotter] Template chưa có biểu đồ. Đang tự động kích hoạt bộ vẽ thích ứng theo log thực nghiệm...")
        generated_figs = cls.generate_grounded_figures(exp_work_dir, full_metrics, user_topic)
        final_figs = sorted([f for f in os.listdir(exp_work_dir) if f.endswith(".png")])
        print(f"[UniversalPlotter] Đã xác thực danh mục hình ảnh ({len(final_figs)} hình): {final_figs}")
        return final_figs

    @classmethod
    def generate_grounded_figures(
        cls,
        exp_work_dir: str,
        full_metrics: Dict[str, Dict[str, Any]],
        user_topic: str = ""
    ) -> List[str]:
        """
        Sinh các đồ thị khoa học cứu cánh dựa hoàn toàn trên số liệu thật từ log huấn luyện.
        Đặt tên tệp đồng bộ với bộ nhận diện Caption trong LaTeX Writer:
        - convergence_dynamics.png
        - benchmark_comparison.png
        - ablation_tradeoff_analysis.png
        """
        os.makedirs(exp_work_dir, exist_ok=True)
        generated_figs = []

        # 1. Thu thập dữ liệu lịch sử huấn luyện từ các nguồn log thực tế
        histories: Dict[str, Dict[str, Any]] = {}
        for run_name in ["run_0", "run_1"]:
            for f_cand in ["history.json", "final_info.json"]:
                h_path = os.path.join(exp_work_dir, run_name, f_cand)
                if os.path.exists(h_path):
                    try:
                        with open(h_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if "training_dynamics" in data and isinstance(data["training_dynamics"], dict):
                                histories[run_name] = data["training_dynamics"]
                            elif f_cand == "history.json" and run_name not in histories:
                                histories[run_name] = data
                    except Exception:
                        pass

        root_h = os.path.join(exp_work_dir, "history.json")
        if os.path.exists(root_h) and "run_0" not in histories and "run_1" not in histories:
            try:
                with open(root_h, "r", encoding="utf-8") as f:
                    histories["run_0"] = json.load(f)
            except Exception:
                pass

        # Thu thập thêm từ training_dynamics trong metrics (ví dụ template RL)
        for m_name, m_data in full_metrics.items():
            if isinstance(m_data, dict) and "training_dynamics" in m_data:
                td = m_data["training_dynamics"]
                if isinstance(td, dict):
                    # Nhận diện run_0 vs run_1 dựa trên nhãn
                    is_prop = any(x in m_name.lower() for x in ["ours", "proposed", "afdqld", "run_1", "model_1"])
                    tag = "run_1" if is_prop else "run_0"
                    if tag not in histories:
                        histories[tag] = td

        # -----------------------------------------------------------------
        # HÌNH 1: Động học hội tụ (convergence_dynamics.png)
        # -----------------------------------------------------------------
        has_loss_curve = any(
            isinstance(h.get("train_loss") or h.get("td_losses") or h.get("loss"), list) and
            len(h.get("train_loss") or h.get("td_losses") or h.get("loss")) > 1
            for h in histories.values()
        ) or any(
            isinstance(h.get("episode_rewards"), list) and len(h.get("episode_rewards")) > 1
            for h in histories.values()
        )

        if has_loss_curve:
            fig_name = "convergence_dynamics.png"
            cls._plot_figure_dynamics(exp_work_dir, fig_name, histories, user_topic)
            generated_figs.append(fig_name)

        # -----------------------------------------------------------------
        # Chuẩn hóa & Làm phẳng toàn bộ metrics đa cấu trúc (Flat & Nested)
        # -----------------------------------------------------------------
        norm_metrics: Dict[str, Dict[str, float]] = {}
        for m_name, m_data in full_metrics.items():
            flattened = flatten_template_metrics(m_data)
            model_means = {}
            for k, vals in flattened.items():
                if k.lower() in EXCLUDE_COLUMNS:
                    continue
                mean_val, _ = compute_mean_std(vals)
                model_means[k] = mean_val
            if model_means:
                norm_metrics[m_name] = model_means

        # -----------------------------------------------------------------
        # HÌNH 2: Đối sánh Benchmark định lượng (benchmark_comparison.png)
        # -----------------------------------------------------------------
        if norm_metrics:
            fig_b_name = "benchmark_comparison.png"
            plotted = cls._plot_figure_comparison(exp_work_dir, fig_b_name, norm_metrics, user_topic)
            if plotted:
                generated_figs.append(fig_b_name)

        # -----------------------------------------------------------------
        # HÌNH 3: Phân tích Chênh lệch & Đánh đổi (ablation_tradeoff_analysis.png)
        # -----------------------------------------------------------------
        if len(norm_metrics) >= 2:
            fig_c_name = "ablation_tradeoff_analysis.png"
            plotted_delta = cls._plot_figure_deltas(exp_work_dir, fig_c_name, norm_metrics, user_topic)
            if plotted_delta:
                generated_figs.append(fig_c_name)

        return generated_figs

    @classmethod
    def _plot_figure_dynamics(
        cls,
        exp_work_dir: str,
        fig_name: str,
        histories: Dict[str, Dict[str, Any]],
        user_topic: str
    ):
        """Vẽ biểu đồ động học tối ưu hóa: TD Loss / Loss trajectory và Episodic Return trajectory (cả Baseline và Proposed)."""
        plt.figure(figsize=(10.0, 4.2), dpi=300)

        # Subplot 1: Convergence / TD Loss Trajectory
        plt.subplot(1, 2, 1)
        plotted_any = False

        if "run_0" in histories:
            h0 = histories["run_0"]
            l0 = h0.get("train_loss") or h0.get("td_losses") or h0.get("loss") or []
            if isinstance(l0, list) and len(l0) > 1:
                plt.plot(range(1, len(l0) + 1), l0, label="Standard Baseline", color="#7f8c8d", lw=1.8, linestyle="--")
                plotted_any = True

        if "run_1" in histories:
            h1 = histories["run_1"]
            l1 = h1.get("train_loss") or h1.get("td_losses") or h1.get("loss") or []
            if isinstance(l1, list) and len(l1) > 1:
                plt.plot(range(1, len(l1) + 1), l1, label="Proposed Method", color="#2980b9", lw=2.0)
                plotted_any = True

        # Nhận diện nhãn trục x phù hợp
        is_rl = any("episode" in k or "td_loss" in k for h in histories.values() for k in h.keys())
        x_lbl = "Training Episode" if is_rl else "Training Step / Epoch"
        y_lbl = "TD Loss Error" if is_rl else "Loss Criterion (Objective Function)"

        plt.title("Optimization Loss Dynamics", fontsize=11, fontweight="bold")
        plt.xlabel(x_lbl, fontsize=10)
        plt.ylabel(y_lbl, fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        if plotted_any:
            plt.legend(frameon=True, loc="upper right")

        # Subplot 2: Metric / Return / Validation Dynamics
        plt.subplot(1, 2, 2)
        has_rewards = any(
            isinstance(h.get("episode_rewards"), list) and len(h.get("episode_rewards")) > 1
            for h in histories.values()
        )
        has_val_loss = any(
            isinstance(h.get("val_loss"), list) and len(h.get("val_loss")) > 1
            for h in histories.values()
        )

        if has_rewards:
            # Vẽ Episode Rewards cho cả Baseline và Proposed
            if "run_0" in histories and isinstance(histories["run_0"].get("episode_rewards"), list):
                r0 = histories["run_0"]["episode_rewards"]
                plt.plot(range(1, len(r0) + 1), r0, label="Standard Baseline", color="#7f8c8d", lw=1.8, linestyle="--")
            if "run_1" in histories and isinstance(histories["run_1"].get("episode_rewards"), list):
                r1 = histories["run_1"]["episode_rewards"]
                plt.plot(range(1, len(r1) + 1), r1, label="Proposed Method", color="#27ae60", lw=2.0)

            plt.title("Episodic Return Trajectory", fontsize=11, fontweight="bold")
            plt.xlabel("Training Episode", fontsize=10)
            plt.ylabel("Episodic Reward", fontsize=10)
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(frameon=True, loc="lower right")

        elif has_val_loss:
            if "run_0" in histories and isinstance(histories["run_0"].get("val_loss"), list):
                v0 = histories["run_0"]["val_loss"]
                plt.plot(range(1, len(v0) + 1), v0, label="Standard Baseline", color="#7f8c8d", lw=1.8, linestyle="--")
            if "run_1" in histories and isinstance(histories["run_1"].get("val_loss"), list):
                v1 = histories["run_1"]["val_loss"]
                plt.plot(range(1, len(v1) + 1), v1, label="Proposed Method", color="#27ae60", lw=2.0)

            plt.title("Validation Loss Trajectory", fontsize=11, fontweight="bold")
            plt.xlabel("Evaluation Step / Epoch", fontsize=10)
            plt.ylabel("Validation Loss", fontsize=10)
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(frameon=True, loc="upper right")

        else:
            target_h = histories.get("run_1") or histories.get("run_0") or {}
            primary_loss = target_h.get("train_loss") or target_h.get("td_losses") or target_h.get("loss") or []
            if isinstance(primary_loss, list) and primary_loss:
                plt.hist(primary_loss, bins=min(15, max(5, len(primary_loss)//2)), color="#16a085", alpha=0.75, edgecolor="black")
            plt.title("Empirical Loss Distribution", fontsize=11, fontweight="bold")
            plt.xlabel("Loss Value", fontsize=10)
            plt.ylabel("Sample Density", fontsize=10)
            plt.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(exp_work_dir, fig_name), dpi=300)
        plt.close()

    @classmethod
    def _plot_figure_comparison(
        cls,
        exp_work_dir: str,
        fig_name: str,
        norm_metrics: Dict[str, Dict[str, float]],
        user_topic: str
    ) -> bool:
        """Vẽ biểu đồ đối sánh benchmark định lượng theo từng subplot độc lập (tránh lỗi lệch thang đo giữa Reward ~200 và Loss ~0.08)."""
        models = list(norm_metrics.keys())
        if not models:
            return False

        # Tìm các chỉ số chung của các mô hình
        all_candidate_keys = []
        for m in models:
            for k in norm_metrics[m].keys():
                if k.lower() in EXCLUDE_COLUMNS:
                    continue
                if k not in all_candidate_keys:
                    all_candidate_keys.append(k)

        if not all_candidate_keys:
            return False

        # Chọn tối đa 4 chỉ số có ý nghĩa nhất
        selected_keys = all_candidate_keys[:4]
        n_metrics = len(selected_keys)
        n_models = len(models)

        # Sử dụng multi-panel subplots (1 x n_metrics) để mỗi chỉ số có trục tung độc lập với thang đo chuẩn xác
        fig, axes = plt.subplots(1, n_metrics, figsize=(max(8.0, n_metrics * 2.8), 3.8), dpi=300)
        if n_metrics == 1:
            axes = [axes]

        model_palette = ["#7f8c8d", "#2980b9", "#27ae60", "#8e44ad"]
        model_labels = []
        for i, m_name in enumerate(models):
            lbl = "Baseline" if "baseline" in m_name.lower() or i == 0 else "Proposed"
            model_labels.append(lbl)

        for col_idx, k in enumerate(selected_keys):
            ax = axes[col_idx]
            vals = [float(norm_metrics[m].get(k, 0.0)) for m in models]
            x_pos = np.arange(n_models)
            bars = ax.bar(x_pos, vals, width=0.55, color=[model_palette[i % len(model_palette)] for i in range(n_models)], edgecolor="black", linewidth=0.8)

            # Căn chỉnh giới hạn y hợp lý
            max_val = max(vals) if vals else 1.0
            min_val = min(vals) if vals else 0.0
            if max_val > 0:
                ax.set_ylim(0, max_val * 1.25)
            elif min_val < 0:
                ax.set_ylim(min_val * 1.25, 0)

            for bar in bars:
                h = bar.get_height()
                fmt = f"{h:.4f}" if abs(h) < 1.0 else (f"{h:.1f}" if abs(h) >= 10.0 else f"{h:.2f}")
                ax.annotate(
                    fmt,
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4 if h >= 0 else -10),
                    textcoords="offset points",
                    ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=8.5, fontweight="bold"
                )

            ax.set_title(cls._format_metric_name(k), fontsize=9.5, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_labels, fontsize=8.5, fontweight="bold")
            ax.grid(axis="y", linestyle=":", alpha=0.5)

        fig.suptitle("Quantitative Benchmark Performance Comparison", fontsize=11, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_work_dir, fig_name), dpi=300, bbox_inches="tight")
        plt.close()
        return True

    @classmethod
    def _plot_figure_deltas(
        cls,
        exp_work_dir: str,
        fig_name: str,
        norm_metrics: Dict[str, Dict[str, float]],
        user_topic: str
    ) -> bool:
        """Vẽ biểu đồ thanh ngang phân tích chênh lệch và đánh đổi phương pháp luận (Scientific Trade-off)."""
        models = list(norm_metrics.keys())
        if len(models) < 2:
            return False

        baseline_name = models[0]
        proposed_name = models[-1]
        b_dict = norm_metrics[baseline_name]
        p_dict = norm_metrics[proposed_name]

        common_keys = [k for k in b_dict if k in p_dict and k.lower() not in EXCLUDE_COLUMNS]
        if not common_keys:
            return False

        deltas = []
        labels = []
        colors = []
        for k in common_keys[:6]:
            b_val = float(b_dict[k])
            p_val = float(p_dict[k])
            if abs(b_val) > 1e-9:
                pct = ((p_val - b_val) / abs(b_val)) * 100.0
            else:
                pct = (p_val - b_val) * 100.0

            hib = is_higher_is_better(k)
            is_improvement = (pct > 0) if hib else (pct < 0)

            deltas.append(pct)
            labels.append(cls._format_metric_name(k))
            colors.append("#27ae60" if is_improvement else "#d35400")

        fig, ax = plt.subplots(figsize=(8.2, max(4.0, len(labels) * 0.75)), dpi=300)
        bars = ax.barh(labels, deltas, color=colors, height=0.55, edgecolor="black", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Relative Metric Difference vs Baseline (%)", fontsize=10, fontweight="bold")
        ax.set_title("Empirical Delta & Scientific Trade-Off Analysis", fontsize=11, fontweight="bold")
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        # Mở rộng giới hạn trục hoành (x-limits) để nhãn số liệu không bao giờ bị đè lên nhãn trục tung
        min_w = min(deltas) if deltas else 0.0
        max_w = max(deltas) if deltas else 0.0
        x_span = max(abs(min_w), abs(max_w), 25.0)
        left_pad = max(abs(min_w) * 0.55, 38.0) if min_w < 0 else 15.0
        right_pad = max(abs(max_w) * 0.35, 25.0) if max_w > 0 else 15.0
        ax.set_xlim(min(0.0, min_w) - left_pad, max(0.0, max_w) + right_pad)

        for bar in bars:
            w = bar.get_width()
            if w >= 0:
                xpos = w + (x_span * 0.02)
                ha = "left"
            else:
                xpos = w - (x_span * 0.02)
                ha = "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2, f"{w:+.1f}%", va="center", ha=ha, fontsize=8.5, fontweight="bold")

        plt.subplots_adjust(left=0.28, right=0.92, top=0.90, bottom=0.15)
        plt.savefig(os.path.join(exp_work_dir, fig_name), dpi=300)
        plt.close()
        return True
