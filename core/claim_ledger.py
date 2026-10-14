import json
import math
from typing import Dict, Any, List, Optional, Tuple
from core.contracts import ClaimItem, EnvironmentInfo


HIGHER_IS_BETTER_METRICS = {
    "accuracy", "acc", "top1_acc", "top5_acc", "f1", "f1_macro", "f1_micro", "f1_weighted",
    "precision", "recall", "auc", "roc_auc", "r2", "r2_score", "c_index", "retention"
}

LOWER_IS_BETTER_METRICS = {
    "loss", "val_loss", "train_loss", "test_loss", "mse", "test_mse", "mae", "test_mae",
    "rmse", "brier_score", "latency_ms", "inference_latency", "perplexity", "ppl",
    "energy_joules", "error", "error_rate"
}


def is_higher_is_better(metric_name: str) -> bool:
    m = metric_name.lower().replace("-", "_")
    if any(h in m for h in HIGHER_IS_BETTER_METRICS):
        return True
    return False


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


class ClaimLedger:
    def __init__(self, results_data: Dict[str, Any], config_data: Optional[Dict[str, Any]] = None, env_data: Optional[Dict[str, Any]] = None):
        self.results_data = results_data or {}
        self.config_data = config_data or {}
        self.env_data = env_data or {}
        self.claims: Dict[str, ClaimItem] = {}
        self.placeholders: Dict[str, str] = {}
        self._build_ledger()

    def _build_ledger(self):
        metrics = self.results_data.get("metrics", {})
        if not metrics:
            return

        # 1. Tìm phương pháp đề xuất (Proposed/Ours) và các Baselines
        model_names = list(metrics.keys())
        proposed_name = None
        for name in model_names:
            if any(k in name.lower() for k in ["ours", "proposed", "deepscientist", "upgraded", "model_1", "run_1"]):
                proposed_name = name
                break
        if not proposed_name:
            proposed_name = model_names[-1]  # Mặc định lấy model cuối

        baselines = [m for m in model_names if m != proposed_name]

        # 2. Xây dựng các Claim số liệu (R1, R2, ...)
        proposed_metrics = metrics.get(proposed_name, {})
        claim_counter = 1

        for metric_key, val in proposed_metrics.items():
            vals_list = val if isinstance(val, list) else [val]
            # Bỏ qua các trường metadata không phải số (ví dụ: "method", "dataset", "notes")
            has_num = False
            for v in vals_list:
                try:
                    if v is not None and not math.isnan(float(v)) and not math.isinf(float(v)):
                        has_num = True
                        break
                except (ValueError, TypeError):
                    pass
            if not has_num:
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
                b_val = metrics.get(b_name, {}).get(metric_key)
                if b_val is not None:
                    b_list = b_val if isinstance(b_val, list) else [b_val]
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

    def render_results_table_latex(self, caption: str = "Quantitative Evaluation Benchmark") -> str:
        """Sinh bảng kết quả định lượng tự động 100% từ dữ liệu thực tế (thay thế TableEngine hoàn toàn)."""
        metrics = self.results_data.get("metrics", {})
        if not metrics:
            return "% Không có số liệu thực nghiệm để tạo bảng."

        # Lọc danh sách cột: loại bỏ hyperparameter như epochs, steps, batch_size
        EXCLUDE_COLUMNS = {"epochs", "epoch", "steps", "step", "batch_size", "seed"}
        models = list(metrics.keys())
        all_metrics_keys = []
        for m in models:
            for k, val in metrics[m].items():
                if k.lower() in EXCLUDE_COLUMNS:
                    continue
                v_list = val if isinstance(val, list) else [val]
                has_num = False
                for v in v_list:
                    try:
                        if v is not None and not math.isnan(float(v)) and not math.isinf(float(v)):
                            has_num = True
                            break
                    except (ValueError, TypeError):
                        pass
                if has_num and k not in all_metrics_keys:
                    all_metrics_keys.append(k)

        if not all_metrics_keys:
            return "% Không có chỉ số định lượng hợp lệ để tạo bảng."

        # Tìm giá trị tốt nhất trong từng cột để bôi đậm
        best_vals = {}
        for m_key in all_metrics_keys:
            hib = is_higher_is_better(m_key)
            col_vals = []
            for m in models:
                v = metrics[m].get(m_key)
                if v is not None:
                    m_val, _ = compute_mean_std(v if isinstance(v, list) else [v])
                    col_vals.append(m_val)
            if col_vals:
                best_vals[m_key] = max(col_vals) if hib else min(col_vals)

        # Định dạng tiêu đề cột
        def format_col_header(k: str) -> str:
            name = k.replace('_', ' ').replace('best ', '').replace('final ', '').title()
            if "accuracy" in k.lower() and "(%)" not in name:
                name += " (\\%)"
            elif "sec" in k.lower():
                name = name.replace(" Sec", " (s)")
            return name

        is_wide = len(all_metrics_keys) >= 3
        tbl_env = "table*" if is_wide else "table"
        tbl_pos = "[t]" if is_wide else "[htbp]"
        target_width = r"\textwidth" if is_wide else r"\columnwidth"
        latex = [
            f"\\begin{{{tbl_env}}}{tbl_pos}",
            "\\centering",
            f"\\caption{{{caption}}}",
            "\\vspace{0.3em}",
            "\\label{tab:empirical_results}",
            "\\small",
            f"\\resizebox{{{target_width}}}{{!}}{{",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            "\\textbf{Method / Architecture} & " + " & ".join(f"\\textbf{{{format_col_header(k)}}}" for k in all_metrics_keys) + " \\\\",
            "\\midrule"
        ]

        for model in models:
            is_ours = any(k in model.lower() for k in ["ours", "proposed"])
            row_items = [f"\\textbf{{{model}}}" if is_ours else model]
            for m_key in all_metrics_keys:
                val = metrics[model].get(m_key)
                if val is None:
                    row_items.append("--")
                else:
                    v_list = val if isinstance(val, list) else [val]
                    mean, std = compute_mean_std(v_list)
                    is_best = (m_key in best_vals and abs(mean - best_vals[m_key]) < 1e-6)
                    
                    if "accuracy" in m_key.lower():
                        display_val = f"{mean * 100:.2f}" if mean <= 1.0 else f"{mean:.2f}"
                        val_str = f"{display_val}" if std <= 0 else f"{display_val} $\\pm$ {std*100:.2f}"
                    elif "time" in m_key.lower() or "sec" in m_key.lower():
                        val_str = f"{mean:.2f}s"
                    elif abs(mean) < 0.01:
                        val_str = f"{mean:.4f}" if std <= 0 else f"{mean:.4f} $\\pm$ {std:.4f}"
                    else:
                        val_str = f"{mean:.3f}" if std <= 0 else f"{mean:.3f} $\\pm$ {std:.3f}"

                    if is_best:
                        val_str = f"\\textbf{{{val_str}}}"
                    row_items.append(val_str)
            latex.append(" & ".join(row_items) + " \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            f"\\end{{{tbl_env}}}"
        ])
        return "\n".join(latex)

    def render_claim_ledger_markdown(self) -> str:
        """Sinh tài liệu Markdown báo cáo kiểm chứng cho Bundle (Điều #1, #4, #15) kèm chỉ dẫn cực tính trung thực."""
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
                elif is_better:
                    eval_str = "TỐT HƠN"
                    directives.append(f"- **{c.metric_key}**: Proposed ({c.method_mean}) cải thiện so với baseline ({c.baseline_mean}) thêm {abs(c.rel_improvement_pct):.1f}%. Cho phép dùng từ 'improves' hoặc 'superior'.")
                else:
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
                "- **Tham số giả lập**: Nếu có chỉ số Relative Risk hoặc Hazard Ratio ground-truth cố định, phải nêu rõ đây là giá trị chuẩn của mô phỏng (benchmark simulation ground truth), KHÔNG ĐƯỢC viết như thể đây là số đo do mô hình ước lượng được."
            ])

        return "\n".join(lines)
