import json
import math
import re
from typing import Dict, Any, List, Optional, Tuple
from core.contracts import ClaimItem, EnvironmentInfo


HIGHER_IS_BETTER_METRICS = {
    "accuracy", "acc", "top1_acc", "top5_acc", "f1", "f1_macro", "f1_micro", "f1_weighted",
    "precision", "recall", "auc", "roc_auc", "r2", "r2_score", "c_index", "retention",
    "reward", "return", "hit_ratio", "ndcg", "mrr", "bleu", "rouge", "success_rate",
    "throughput", "throughput_steps_per_sec", "tokens_per_second", "steps_per_sec"
}

LOWER_IS_BETTER_METRICS = {
    "loss", "val_loss", "train_loss", "test_loss", "mse", "test_mse", "mae", "test_mae",
    "rmse", "brier", "brier_score", "latency_ms", "inference_latency", "perplexity", "ppl",
    "energy_joules", "error", "error_rate", "asmd", "kl_divergence", "kldiv", "time", "seconds", "elapsed",
    "convergence_episode", "s_per_1000_steps", "s_per_step"
}

EXCLUDE_COLUMNS = {
    "epochs", "epoch", "steps", "step", "batch_size", "seed", "seed_offset",
    "episodes_trained", "num_episodes", "domain", "series_name", "unit_label",
    "target_name", "feature_names", "ground_truth_hr_immuno", "training_dynamics",
    "dataset", "method", "notes", "all_results", "history", "treatment_names", "covariate_names",
    "all_seeds", "num_seeds"
}


def is_higher_is_better(metric_name: str) -> bool:
    m = metric_name.lower().replace("-", "_")
    if any(l in m for l in LOWER_IS_BETTER_METRICS):
        return False
    if any(h in m for h in HIGHER_IS_BETTER_METRICS) or m == "score":
        return True
    return False


def is_timing_metric(metric_name: str) -> bool:
    m = metric_name.lower()
    return any(k in m for k in ["time", "sec", "latency", "elapsed", "wall"])


# Ngưỡng nhiễu đo thời gian (wall-clock) trên 1 lần chạy: chênh lệch nhỏ hơn mức này không được coi là cải thiện
TIMING_NOISE_PCT = 5.0


def split_grouped_metrics(keys: List[str]) -> Optional[Tuple[List[str], List[str]]]:
    """
    Phát hiện metric có tiền tố theo dataset (vd: circle_eval_loss, dino_eval_loss).
    Trả về (danh sách dataset, danh sách metric gốc) nếu mọi key đều tách được theo dạng <dataset>_<metric>
    với ít nhất 2 dataset dùng chung metric; ngược lại trả về None.
    """
    if len(keys) < 4:
        return None
    groups: Dict[str, List[str]] = {}
    pairs = []
    for k in keys:
        if "_" not in k:
            return None
        prefix, suffix = k.split("_", 1)
        pairs.append((prefix, suffix))
        groups.setdefault(suffix, []).append(prefix)
    datasets = []
    for p, _ in pairs:
        if p not in datasets:
            datasets.append(p)
    metrics = [s for s in groups if len(groups[s]) >= 2]
    if len(datasets) < 2 or len(metrics) != len(groups):
        return None
    return datasets, metrics


def latex_escape(text: str) -> str:
    """Thoát các ký tự đặc biệt LaTeX để tránh lỗi pdflatex."""
    return str(text).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")


def compute_mean_std(values: List[Any]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    clean_vals = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
            if not math.isnan(f) and not math.isinf(f):
                clean_vals.append(f)
        except (ValueError, TypeError):
            continue
    if not clean_vals:
        return 0.0, 0.0
    mean = sum(clean_vals) / len(clean_vals)
    if len(clean_vals) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in clean_vals) / (len(clean_vals) - 1)
    std = math.sqrt(var)
    return mean, std


def flatten_template_metrics(raw: Dict[str, Any], prefix: str = "") -> Dict[str, List[float]]:
    """
    Chuẩn hóa và làm phẳng số liệu thực nghiệm từ đa dạng cấu trúc Template:
    - Hỗ trợ định dạng phẳng (time_series_forecasting, causal_trial_emulation, reinforcement_learning).
    - Hỗ trợ định dạng lồng nhiều cấp (nanoGPT, nanoGPT_lite, 2d_diffusion, grokking).
    """
    flattened: Dict[str, List[float]] = {}
    if not isinstance(raw, dict):
        return flattened

    # Nhánh A: Template Sakana AI chứa final_info_dict (danh sách seeds)
    if "final_info_dict" in raw and isinstance(raw["final_info_dict"], dict):
        for k, vals in raw["final_info_dict"].items():
            if k.lower() in EXCLUDE_COLUMNS:
                continue
            clean_k = k.replace("_mean", "").replace("_stderr", "")
            full_k = f"{prefix}_{clean_k}" if prefix else clean_k
            if isinstance(vals, list):
                flattened[full_k] = [float(x) for x in vals if x is not None and not math.isnan(float(x))]
            elif isinstance(vals, (int, float)):
                flattened[full_k] = [float(vals)]
        return flattened

    # Nhánh B: Template Sakana AI chứa means dict
    if "means" in raw and isinstance(raw["means"], dict):
        for k, v in raw["means"].items():
            if k.lower() in EXCLUDE_COLUMNS:
                continue
            clean_k = k.replace("_mean", "")
            full_k = f"{prefix}_{clean_k}" if prefix else clean_k
            if isinstance(v, (int, float)) and not math.isnan(float(v)) and not math.isinf(float(v)):
                flattened[full_k] = [float(v)]
        return flattened

    # Nhánh C: Dict thông thường hoặc lồng theo dataset/module
    for k, v in raw.items():
        if k.lower() in EXCLUDE_COLUMNS:
            continue
        curr_key = f"{prefix}_{k}" if prefix else k

        if isinstance(v, dict):
            sub_res = flatten_template_metrics(v, prefix=curr_key)
            flattened.update(sub_res)
        elif isinstance(v, list):
            # Nếu là danh sách giá trị số hữu hạn (<= 20 phần tử, ví dụ qua các seeds/folds)
            if v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                if len(v) <= 20:
                    flattened[curr_key] = [float(x) for x in v if not math.isnan(float(x)) and not math.isinf(float(x))]
                else:
                    # Nếu là chuỗi dữ liệu dài (ví dụ learning curve), chỉ lấy điểm kết thúc
                    flattened[curr_key] = [float(v[-1])]
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                fv = float(v)
                if not math.isnan(fv) and not math.isinf(fv):
                    flattened[curr_key] = [fv]
            except (ValueError, TypeError):
                pass

    return flattened


class ClaimLedger:
    def __init__(self, results_data: Dict[str, Any], config_data: Optional[Dict[str, Any]] = None, env_data: Optional[Dict[str, Any]] = None):
        self.results_data = results_data or {}
        self.config_data = config_data or {}
        self.env_data = env_data or {}
        self.claims: Dict[str, ClaimItem] = {}
        self.placeholders: Dict[str, str] = {}
        self.normalized_metrics: Dict[str, Dict[str, List[float]]] = {}
        self._build_ledger()

    def _build_ledger(self):
        raw_metrics = self.results_data.get("metrics", {})
        if not raw_metrics:
            return

        # 0. Chuẩn hóa và làm phẳng số liệu cho tất cả các mô hình
        for m_name, m_data in raw_metrics.items():
            self.normalized_metrics[m_name] = flatten_template_metrics(m_data)

        # 1. Tìm phương pháp đề xuất (Proposed/Ours) và các Baselines
        model_names = list(self.normalized_metrics.keys())
        if not model_names:
            return

        proposed_name = None
        for name in model_names:
            if any(k in name.lower() for k in ["ours", "proposed", "deepscientist", "upgraded", "model_1", "run_1"]):
                proposed_name = name
                break
        if not proposed_name:
            proposed_name = model_names[-1]

        baselines = [m for m in model_names if m != proposed_name]

        # 2. Xây dựng các Claim số liệu (R1, R2, ...)
        proposed_norm = self.normalized_metrics.get(proposed_name, {})
        claim_counter = 1

        for metric_key, vals_list in proposed_norm.items():
            if not vals_list:
                continue

            p_mean, p_std = compute_mean_std(vals_list)
            hib = is_higher_is_better(metric_key)
            direction = "higher_is_better" if hib else "lower_is_better"

            # Tuyên bố độc lập cho chính model đề xuất
            c_id = f"R{claim_counter}"
            claim_counter += 1
            std_str = f" ± {p_std:.4f}" if p_std > 0 else ""
            self.claims[c_id] = ClaimItem(
                claim_id=c_id,
                metric_key=metric_key,
                method_name=proposed_name,
                method_mean=round(p_mean, 4),
                method_std=round(p_std, 4),
                direction=direction,
            )
            self.placeholders[c_id] = f"{p_mean:.4f}{std_str}"

            # Tuyên bố đối sánh so với từng baseline
            for b_name in baselines:
                b_list = self.normalized_metrics.get(b_name, {}).get(metric_key)
                if b_list:
                    b_mean, b_std = compute_mean_std(b_list)
                    delta = p_mean - b_mean
                    rel_pct = 0.0
                    if abs(b_mean) > 1e-9:
                        rel_pct = (delta / b_mean) * 100.0

                    comp_id = f"R{claim_counter}"
                    claim_counter += 1
                    self.claims[comp_id] = ClaimItem(
                        claim_id=comp_id,
                        metric_key=metric_key,
                        method_name=proposed_name,
                        method_mean=round(p_mean, 4),
                        method_std=round(p_std, 4),
                        baseline_name=b_name,
                        baseline_mean=round(b_mean, 4),
                        baseline_std=round(b_std, 4),
                        delta=round(delta, 4),
                        rel_improvement_pct=round(rel_pct, 2),
                        direction=direction,
                    )
                    dir_sym = "+" if delta > 0 else ""
                    self.placeholders[comp_id] = f"{dir_sym}{rel_pct:.1f}\\%"

        # 3. Xây dựng các Placeholder Cấu hình (C_...)
        for k, v in self._flatten_dict(self.config_data, prefix="C").items():
            self.placeholders[k] = str(v)

        # 4. Xây dựng các Placeholder Môi trường (E_...)
        for k, v in self.env_data.items():
            self.placeholders[f"E_{k}"] = str(v)

    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        items = {}
        for k, v in d.items():
            new_key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items

    @staticmethod
    def format_metric_header(k: str) -> str:
        name = k.replace('_', ' ').replace('best ', '').replace('final ', '').title()
        name = re.sub(r"\bMse\b", "MSE", name)
        name = re.sub(r"\bMae\b", "MAE", name)
        name = re.sub(r"\bRmse\b", "RMSE", name)
        name = re.sub(r"\bR2\b", r"R$^2$", name)
        name = re.sub(r"\bAuc\b", "AUC", name)
        name = re.sub(r"\bAte\b", "ATE", name)
        name = re.sub(r"\bAsmd\b", "ASMD", name)
        name = re.sub(r"\bKl Divergence\b", "KL Div.", name)
        name = re.sub(r"\bKl\b", "KL", name)
        if "accuracy" in k.lower() and "(%)" not in name:
            name += " (\\%)"
        elif "sec" in k.lower() or "time" in k.lower():
            name = re.sub(r"\b(?:Seconds?|Sec)\b", "(s)", name)
            if "Time" in name and "(s)" not in name:
                name = name.replace("Time", "Time (s)")
        return name

    @staticmethod
    def _short_model_name(model: str) -> str:
        """'Proposed (IMS-DDPM)' -> 'IMS-DDPM'; 'Standard Baseline (2D Diffusion)' -> 'Baseline (2D Diffusion)'."""
        m = re.match(r"^\s*Proposed\s*\((.+)\)\s*$", model)
        if m:
            return m.group(1).strip()
        return re.sub(r"^\s*Standard\s+", "", model)

    @staticmethod
    def _format_value(m_key: str, mean: float, std: float) -> str:
        if "accuracy" in m_key.lower():
            display_val = f"{mean * 100:.2f}" if mean <= 1.0 else f"{mean:.2f}"
            return display_val if std <= 0 else f"{display_val} $\\pm$ {std*100:.2f}"
        if is_timing_metric(m_key):
            return f"{mean:.2f}" if std <= 0 else f"{mean:.2f} $\\pm$ {std:.2f}"
        return f"{mean:.4f}" if std <= 0 else f"{mean:.4f} $\\pm$ {std:.4f}"

    def _render_grouped_table(self, models: List[str], datasets: List[str], metrics: List[str], caption: str) -> str:
        """Bảng chuyển vị: mỗi dataset một khối hàng (Baseline / Proposed), cột là metric gốc."""
        col_spec = "ll" + "c" * len(metrics)
        arrows = ["$\\uparrow$" if is_higher_is_better(m) else "$\\downarrow$" for m in metrics]
        header = "\\textbf{Dataset} & \\textbf{Method} & " + " & ".join(
            f"\\textbf{{{self.format_metric_header(m)}}} {a}" for m, a in zip(metrics, arrows)
        ) + " \\\\"
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\vspace{0.3em}",
            "\\label{tab:empirical_results}",
            "\\small",
            "\\begin{adjustbox}{max width=\\columnwidth}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            header,
            "\\midrule",
        ]
        for d_idx, ds in enumerate(datasets):
            # Giá trị tốt nhất trong từng ô (dataset, metric)
            best = {}
            for m in metrics:
                vals = []
                for model in models:
                    v = self.normalized_metrics[model].get(f"{ds}_{m}")
                    if v:
                        vals.append(compute_mean_std(v)[0])
                if len(vals) >= 2:
                    best[m] = max(vals) if is_higher_is_better(m) else min(vals)
            for r_idx, model in enumerate(models):
                ds_cell = f"\\multirow{{{len(models)}}}{{*}}{{{latex_escape(ds.title())}}}" if r_idx == 0 else ""
                is_ours = any(k in model.lower() for k in ["ours", "proposed"])
                name = latex_escape(self._short_model_name(model))
                row = [ds_cell, f"\\textsc{{{name}}}" if is_ours else name]
                for m in metrics:
                    v = self.normalized_metrics[model].get(f"{ds}_{m}")
                    if not v:
                        row.append("--")
                        continue
                    mean, std = compute_mean_std(v)
                    s = self._format_value(m, mean, std)
                    if m in best and abs(mean - best[m]) < 1e-9:
                        s = f"\\textbf{{{s}}}"
                    row.append(s)
                lines.append(" & ".join(row) + " \\\\")
            if d_idx < len(datasets) - 1:
                lines.append("\\midrule")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"])
        return "\n".join(lines)

    def get_metric_names(self) -> List[str]:
        """Danh sách tên metric gốc (đã gộp theo dataset nếu có) để mô tả trong bài báo."""
        keys = []
        for m in self.normalized_metrics.values():
            for k in m.keys():
                if k not in keys:
                    keys.append(k)
        grouped = split_grouped_metrics(keys)
        base = grouped[1] if grouped else keys
        return [self.format_metric_header(k) for k in base]

    def get_dataset_names(self) -> List[str]:
        keys = []
        for m in self.normalized_metrics.values():
            for k in m.keys():
                if k not in keys:
                    keys.append(k)
        grouped = split_grouped_metrics(keys)
        return [d.title() for d in grouped[0]] if grouped else []

    def render_results_table_latex(self, caption: str = "Quantitative Evaluation Benchmark") -> str:
        """Sinh bảng kết quả định lượng tự động 100% từ dữ liệu thực tế."""
        models = list(self.normalized_metrics.keys())
        if not models:
            return "% Không có số liệu thực nghiệm để tạo bảng."

        all_metrics_keys = []
        for m in models:
            for k in self.normalized_metrics[m].keys():
                if k not in all_metrics_keys:
                    all_metrics_keys.append(k)

        if not all_metrics_keys:
            return "% Không có chỉ số định lượng hợp lệ để tạo bảng."

        # Điều chỉnh caption cho số ít/số nhiều baseline
        n_baselines = len(models) - 1
        if n_baselines <= 1:
            caption = re.sub(r"against (?:the )?competitive baselines", "against the standard baseline", caption)
            caption = caption.replace("competitive baselines", "the standard baseline")

        grouped = split_grouped_metrics(all_metrics_keys)
        if grouped:
            datasets, base_metrics = grouped
            return self._render_grouped_table(models, datasets, base_metrics, caption)

        # Tìm giá trị tốt nhất trong từng cột để bôi đậm
        best_vals = {}
        for m_key in all_metrics_keys:
            hib = is_higher_is_better(m_key)
            col_vals = []
            for m in models:
                v_list = self.normalized_metrics[m].get(m_key)
                if v_list:
                    m_val, _ = compute_mean_std(v_list)
                    col_vals.append(m_val)
            if col_vals:
                best_vals[m_key] = max(col_vals) if hib else min(col_vals)

        format_col_header = self.format_metric_header

        is_wide = len(all_metrics_keys) >= 5
        tbl_env = "table*" if is_wide else "table"
        tbl_pos = "[t]" if is_wide else "[htbp]"
        max_w = r"\textwidth" if is_wide else r"\columnwidth"
        col_spec = "l" + "c" * len(all_metrics_keys)
        latex = [
            f"\\begin{{{tbl_env}}}{tbl_pos}",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\vspace{0.3em}",
            "\\label{tab:empirical_results}",
            "\\small",
            f"\\begin{{adjustbox}}{{max width={max_w}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            "\\textbf{Method / Architecture} & " + " & ".join(f"\\textbf{{{format_col_header(k)}}}" for k in all_metrics_keys) + " \\\\",
            "\\midrule"
        ]

        for model in models:
            is_ours = any(k in model.lower() for k in ["ours", "proposed"])
            disp_model = latex_escape(model)
            row_items = [f"\\textbf{{{disp_model}}}" if is_ours else disp_model]
            for m_key in all_metrics_keys:
                v_list = self.normalized_metrics[model].get(m_key)
                if not v_list:
                    row_items.append("--")
                else:
                    mean, std = compute_mean_std(v_list)
                    is_best = (m_key in best_vals and abs(mean - best_vals[m_key]) < 1e-6)
                    
                    val_str = self._format_value(m_key, mean, std)

                    if is_best:
                        val_str = f"\\textbf{{{val_str}}}"
                    row_items.append(val_str)
            latex.append(" & ".join(row_items) + " \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{adjustbox}",
            f"\\end{{{tbl_env}}}"
        ])
        return "\n".join(latex)

    def render_claim_ledger_markdown(self) -> str:
        """Sinh tài liệu Markdown báo cáo kiểm chứng cho Bundle kèm chỉ dẫn cực tính trung thực."""
        lines = [
            "# Bảng Kiểm Chứng Tuyên Bố Thực Nghiệm (Claim Ledger)",
            "",
            "Tất cả các con số xuất hiện trong bài báo được trích xuất trực tiếp từ các khoản mục dưới đây:",
            "",
            "| ID | Chỉ số / Metric | Mô hình đề xuất | Giá trị đo được | Baseline so sánh | Chênh lệch (Delta) | Thay đổi (%) | Chiều tốt | Đánh giá |",
            "|---|---|---|---|---|---|---|---|---|"
        ]
        directives = []
        for cid, c in self.claims.items():
            b_str = f"{c.baseline_name} ({c.baseline_mean})" if c.baseline_name else "--"
            delta_str = f"{c.delta:+.4f}" if c.baseline_name else "--"
            rel_str = f"{c.rel_improvement_pct:+.1f}%" if c.baseline_name else "--"
            dir_str = "Cao hơn tốt hơn" if c.direction == "higher_is_better" else "Thấp hơn tốt hơn"
            std_str = f" ± {c.method_std:.4f}" if c.method_std > 0 else ""

            eval_str = "Độc lập"
            if c.baseline_name:
                is_better = (c.delta > 0) if c.direction == "higher_is_better" else (c.delta < 0)
                if abs(c.delta) < 1e-6:
                    eval_str = "Tương đương (0.0%)"
                elif is_timing_metric(c.metric_key) and abs(c.rel_improvement_pct) < TIMING_NOISE_PCT:
                    eval_str = "NHIỄU ĐO THỜI GIAN (NOT SIGNIFICANT)"
                    directives.append(f"- **{c.metric_key}**: Chênh lệch {c.rel_improvement_pct:+.1f}% ({c.method_mean} vs {c.baseline_mean} giây) nằm trong biên nhiễu đo wall-clock của 1 lần chạy (< {TIMING_NOISE_PCT:.0f}%). BẮT BUỘC mô tả là 'comparable / within run-to-run timing noise'. CẤM dùng 'improves', 'faster', 'superior', 'competitive or superior' cho chỉ số này.")
                elif is_better:
                    if abs(c.rel_improvement_pct) < 1.0 or abs(c.delta) < 0.002:
                        eval_str = "TƯƠNG ĐƯƠNG / CẢI THIỆN BIÊN ĐỘ NHỎ (MARGINAL)"
                        directives.append(f"- **{c.metric_key}**: Proposed ({c.method_mean}) cải thiện rất nhỏ so với baseline ({c.baseline_mean}) ({c.rel_improvement_pct:+.1f}%). TRUNG THỰC HỌC THUẬT: CẤM TUYỆT ĐỐI dùng từ 'superior' hoặc 'outperforms'! BẮT BUỘC dùng từ thận trọng: 'performs competitively / on par with the baseline', 'yields a slight/marginal increase', và ghi rõ kết quả trên 1 seed ngẫu nhiên duy nhất chưa khẳng định khác biệt có ý nghĩa thống kê.")
                    else:
                        eval_str = "TỐT HƠN"
                        directives.append(f"- **{c.metric_key}**: Proposed ({c.method_mean}) cải thiện so với baseline ({c.baseline_mean}) thêm {abs(c.rel_improvement_pct):.1f}%. Cho phép dùng từ 'improves' hoặc 'demonstrates gains'.")
                else:
                    if abs(c.rel_improvement_pct) < 1.0:
                        eval_str = "TƯƠNG ĐƯƠNG (CHÊNH LỆCH < 1%)"
                        directives.append(f"- **{c.metric_key}**: Proposed ({c.method_mean}) vs baseline ({c.baseline_mean}) chênh {c.rel_improvement_pct:+.1f}% — trên 1 seed duy nhất, đây là mức TƯƠNG ĐƯƠNG (on par). KHÔNG gọi là 'degradation' và KHÔNG gọi là 'improvement'.")
                        lines.append(
                            f"| **{cid}** | `{c.metric_key}` | {c.method_name} | {c.method_mean}{std_str} | {b_str} | {delta_str} | {rel_str} | {dir_str} | **{eval_str}** |"
                        )
                        continue
                    eval_str = "KÉM HƠN (DEGRADED)"
                    directives.append(f"- **{c.metric_key}**: Proposed ({c.method_mean}) KÉM HƠN baseline ({c.baseline_mean}) chênh lệch {abs(c.rel_improvement_pct):.1f}%. BẮT BUỘC PHẢI THỪA NHẬN TRUNG THỰC: Đây là sự suy giảm / đánh đổi phương pháp luận (performance decline / trade-off). CẤM TUYỆT ĐỐI dùng từ 'outperform', 'superior', 'effectively handles', 'robust' cho chỉ số này!")

            lines.append(
                f"| **{cid}** | `{c.metric_key}` | {c.method_name} | {c.method_mean}{std_str} | {b_str} | {delta_str} | {rel_str} | {dir_str} | **{eval_str}** |"
            )

        if directives:
            lines.extend([
                "",
                "### CHỈ DẪN TRUNG THỰC BẮT BUỘC CHO NGÔN NGỮ BÀI BÁO (ZERO-SPIN DIRECTIVES):",
                "Khi viết văn xuôi (Abstract, Introduction, Experiments, Conclusion), bạn PHẢI tuân thủ các quy tắc cực tính sau:",
                *directives,
                "- **Thận trọng khoa học**: Khi chênh lệch < 1.0% hoặc trên 1 seed duy nhất, không thổi phồng thành đột phá. Báo cáo khách quan như một đối sánh ngang ngửa (on par / comparable)."
            ])

        return "\n".join(lines)

    def render_claim_ledger_markdown_en(self) -> str:
        """Generate English verified claim ledger markdown with zero-spin directives for academic paper writing."""
        lines = [
            "# Verified Empirical Claim Ledger",
            "",
            "All quantitative figures reported in the manuscript are strictly verified against the following items:",
            "",
            "| ID | Metric | Evaluated Model | Measured Value | Compared Baseline | Delta | Change (%) | Preferred Direction | Assessment |",
            "|---|---|---|---|---|---|---|---|---|"
        ]
        directives = []
        for cid, c in self.claims.items():
            b_str = f"{c.baseline_name} ({c.baseline_mean})" if c.baseline_name else "--"
            delta_str = f"{c.delta:+.4f}" if c.baseline_name else "--"
            rel_str = f"{c.rel_improvement_pct:+.1f}%" if c.baseline_name else "--"
            dir_str = "Higher is better" if c.direction == "higher_is_better" else "Lower is better"
            std_str = f" ± {c.method_std:.4f}" if c.method_std > 0 else ""

            eval_str = "Independent"
            if c.baseline_name:
                is_better = (c.delta > 0) if c.direction == "higher_is_better" else (c.delta < 0)
                if abs(c.delta) < 1e-6:
                    eval_str = "On par (0.0%)"
                elif is_timing_metric(c.metric_key) and abs(c.rel_improvement_pct) < TIMING_NOISE_PCT:
                    eval_str = "Run-to-run timing noise (not statistically significant)"
                    directives.append(f"- **{c.metric_key}**: Difference of {c.rel_improvement_pct:+.1f}% ({c.method_mean} vs {c.baseline_mean} seconds) falls within single-run wall-clock measurement noise (< {TIMING_NOISE_PCT:.0f}%). MANDATORY: Describe as 'comparable / within run-to-run timing noise'. NEVER use 'improves', 'faster', 'superior', or 'competitive or superior' for this runtime metric.")
                elif is_better:
                    if abs(c.rel_improvement_pct) < 1.0 or abs(c.delta) < 0.002:
                        eval_str = "Comparable / Marginal difference (< 1%)"
                        directives.append(f"- **{c.metric_key}**: Proposed ({c.method_mean}) shows marginal change over baseline ({c.baseline_mean}) ({c.rel_improvement_pct:+.1f}%). SCIENTIFIC INTEGRITY: DO NOT claim 'superior' or 'outperforms'! State that performance is 'on par with or comparable to baseline' and note that single-seed runs cannot establish statistical superiority.")
                    else:
                        eval_str = "Improved"
                        directives.append(f"- **{c.metric_key}**: Proposed ({c.method_mean}) demonstrates gains over baseline ({c.baseline_mean}) by {abs(c.rel_improvement_pct):.1f}%. Permissible to use 'improves' or 'demonstrates gains'.")
                else:
                    if abs(c.rel_improvement_pct) < 1.0:
                        eval_str = "On par (difference < 1%)"
                        directives.append(f"- **{c.metric_key}**: Proposed ({c.method_mean}) vs baseline ({c.baseline_mean}) differs by {c.rel_improvement_pct:+.1f}% — on a single seed, this represents comparable performance (on par). Do not label as degradation or significant improvement.")
                        lines.append(
                            f"| **{cid}** | `{c.metric_key}` | {c.method_name} | {c.method_mean}{std_str} | {b_str} | {delta_str} | {rel_str} | {dir_str} | **{eval_str}** |"
                        )
                        continue
                    eval_str = "Degraded / Methodological Trade-off"
                    directives.append(f"- **{c.metric_key}**: Proposed ({c.method_mean}) degrades compared to baseline ({c.baseline_mean}) by {abs(c.rel_improvement_pct):.1f}%. MANDATORY SCIENTIFIC HONESTY: Explicitly acknowledge this as a performance decline or methodological trade-off. FORBIDDEN to use 'outperform', 'superior', 'effectively handles', or 'robust' for this degraded metric!")

            lines.append(
                f"| **{cid}** | `{c.metric_key}` | {c.method_name} | {c.method_mean}{std_str} | {b_str} | {delta_str} | {rel_str} | {dir_str} | **{eval_str}** |"
            )

        if directives:
            lines.extend([
                "",
                "### ZERO-SPIN SCIENTIFIC INTEGRITY DIRECTIVES:",
                "When composing prose (Abstract, Introduction, Experiments, Conclusion), you MUST strictly comply with these directives:",
                *directives,
                "- **Scientific Prudence**: When differences are < 1.0% or derived from a single seed, do not characterize results as breakthroughs. Report objectively as competitive or on-par comparisons."
            ])

        return "\n".join(lines)

