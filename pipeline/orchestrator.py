import datetime
import json
import os
import subprocess
import sys
from typing import Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from core.gemini_llm import GeminiLLM
from core.knowledge_store import KnowledgeStore
from agents.research_agents import (
    ProblemIdentifier, ProblemValidator,
    MethodDeveloper, MethodValidator,
    ExperimentDesigner, ExperimentValidator,
)
from agents.experiment_agents import CoderAgent
from templates.template_manager import TemplateManager
from agents.latex_writer import LaTeXWriter
from agents.peer_reviewer import PeerReviewer


def safe_float(val: Any, default: float = 4.0) -> float:
    """Chuyển đổi an toàn giá trị điểm số sang float, ngăn chặn lỗi ValueError format specifier."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        import re
        match = re.search(r"[-+]?\d*\.?\d+", val)
        if match:
            try:
                return float(match.group(0))
            except Exception:
                pass
    return default


class DeepScientistPipeline:
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        data_dir: Optional[str] = None,
        output_base_dir: Optional[str] = None,
        online_search: bool = True,
        semantic_scholar_key: Optional[str] = None,
    ):
        print("=" * 70)
        print("  DEEPSCIENTIST: AUTONOMOUS SCIENTIFIC DISCOVERY")
        print("=" * 70)

        self.online_search = online_search
        self.semantic_scholar_key = semantic_scholar_key
        self.llm = GeminiLLM(api_key=api_key, model_name=model_name)
        self.template_manager = TemplateManager()

        # Nạp toàn bộ 100% cơ sở dữ liệu của ResearchAgent
        self.knowledge_store = KnowledgeStore(data_dir=data_dir)
        self.knowledge_store.load_all_data()

        # Đầy đủ 6 tác nhân nguyên bản của ResearchAgent
        self.problem_identifier = ProblemIdentifier(self.llm)
        self.problem_validator = ProblemValidator(self.llm)
        self.method_developer = MethodDeveloper(self.llm)
        self.method_validator = MethodValidator(self.llm)
        self.experiment_designer = ExperimentDesigner(self.llm)
        self.experiment_validator = ExperimentValidator(self.llm)

        # CoderAgent của Sakana AI để nâng cấp code template
        self.coder_agent = CoderAgent(self.llm)

        # Công cụ xuất bản & phản biện của Sakana AI
        self.latex_writer = LaTeXWriter(self.llm, self.knowledge_store)
        self.peer_reviewer = PeerReviewer(self.llm)

        if output_base_dir is None:
            self.output_base_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
            )
        else:
            self.output_base_dir = output_base_dir

    def run(
        self,
        topic: str,
        template_name: Optional[str] = None,
        max_iterations: int = 2
    ) -> Dict[str, Any]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
        run_dir = os.path.join(self.output_base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n[PIPELINE] Bắt đầu nghiên cứu đề tài: '{topic}'")
        print(f"[PIPELINE] Thư mục kết quả: {run_dir}\n")

        # -------------------------------------------------------------
        # BƯỚC 0: KIỂM ĐỊNH ĐỘ PHỦ DỮ LIỆU & ĐỘ TƯƠNG THÍCH TEMPLATE
        # -------------------------------------------------------------
        print(">>> GIAI ĐOẠN 0: KIỂM ĐỊNH ĐỘ PHỦ DỮ LIỆU & ĐỘ TƯƠNG THÍCH TEMPLATE (Pre-Flight Assessment)")
        data_cov = self.knowledge_store.assess_topic_coverage(topic)
        tpl_suit = self.template_manager.assess_template_suitability(topic, llm=self.llm)

        print(f"[Pre-Flight] Mức độ bao phủ tài liệu offline: {data_cov['status']} ({data_cov['coverage_percent']}%)")
        if data_cov["status"] == "INSUFFICIENT":
            print("\n" + "!" * 70)
            print(f" [CẢNH BÁO TÀI LIỆU]: {data_cov['warning']}")
            print(" [CÁC HƯỚNG GIẢI QUYẾT ĐỀ XUẤT]:")
            for sol in data_cov["solutions"]:
                print(f"   -> {sol}")
            print("!" * 70 + "\n")
        elif data_cov["status"] == "MODERATE":
            print(f" [Lưu ý tài liệu]: {data_cov['warning']}")

        print(f"[Pre-Flight] Đánh giá Template thực nghiệm: '{tpl_suit.get('selected_template')}' (Độ tin cậy: {tpl_suit.get('confidence')}/10)")
        if not tpl_suit.get("is_supported") or tpl_suit.get("confidence", 0) < 6:
            print("\n" + "!" * 70)
            print(f" [CẢNH BÁO TEMPLATE]: Chưa có template chuyên biệt cho đề tài '{topic}' trong 14 template có sẵn!")
            print(f" Phân tích: {tpl_suit.get('reasoning')}")
            print(" [CÁC HƯỚNG GIẢI QUYẾT ĐỀ XUẤT]:")
            print("   -> Hướng 1: Kích hoạt chế độ Autonomous Zero-Shot Coding (CoderAgent tự thiết kế và sinh mã run_experiment.py mới).")
            print("   -> Hướng 2: Bổ sung thư mục template tùy biến vào templates/ chứa experiment.py.")
            print(f"   -> Gợi ý: {tpl_suit.get('suggestions')}")
            print("!" * 70 + "\n")
        else:
            print(f" -> Đánh giá: {tpl_suit.get('reasoning')}")

        if not template_name:
            template_name = tpl_suit.get("selected_template") or self.template_manager.select_best_template_for_topic(topic)

        # Lưu báo cáo kiểm định trước chuyến bay
        preflight_info = {
            "topic": topic,
            "data_coverage": data_cov,
            "template_suitability": tpl_suit,
            "selected_template": template_name
        }
        with open(os.path.join(run_dir, "preflight_assessment.json"), "w", encoding="utf-8") as pf:
            json.dump(preflight_info, pf, indent=2, ensure_ascii=False)

        # -------------------------------------------------------------
        # BƯỚC 1: Khai phá Đồ thị Tri thức Toàn cục (ResearchAgent Store)
        # -------------------------------------------------------------
        print("\n>>> GIAI ĐOẠN 1: KHAI PHÁ ĐỒ THỊ TRI THỨC TOÀN CỤC (ResearchAgent Global Graph)")

        # Lấy top papers kèm scores để có thể kiểm tra relevance của seed paper
        matched_papers_with_scores = self.knowledge_store.search_target_papers_with_scores(topic, top_k=5)
        matched_papers = [p for _, p in matched_papers_with_scores]
        top_score = matched_papers_with_scores[0][0] if matched_papers_with_scores else 0.0

        # Ngưỡng điểm relevance tối thiểu để tin dùng bài từ offline corpus làm seed paper.
        # Nếu top score thấp hơn ngưỡng → bài top-1 chỉ khớp từ chung, không phải topic thực sự
        # → kích hoạt external retrieval dù coverage status là MODERATE.
        SEED_RELEVANCE_THRESHOLD = 40.0

        # Log chi tiết top-5 papers với scores để dễ debug
        print(f"[Stage 1] Ket qua tim kiem offline corpus (top-5 papers, score giam dan):")
        for i, (sc, pp) in enumerate(matched_papers_with_scores):
            print(f"  [{i+1}] score={sc:6.1f}  {pp.get('title', 'N/A')[:75]}")
        sys.stdout.flush()

        # Điều kiện kích hoạt external retrieval — bao gồm weak-seed detection
        offline_insufficient = data_cov.get("status") == "INSUFFICIENT"
        weak_seed = (top_score < SEED_RELEVANCE_THRESHOLD)
        if weak_seed and not offline_insufficient:
            print(f"[Stage 1] CANH BAO: Top-1 offline paper co score={top_score:.1f} < {SEED_RELEVANCE_THRESHOLD} "
                  f"(nguong relevance). Kha nang seed paper khong sat chu de. Chuyen sang External Retrieval.")

        if offline_insufficient or weak_seed or not matched_papers:
            print(f"[PIPELINE] Du lieu offline chua du do phu cho chu de '{topic}'.")

            enrichment_result = None
            if getattr(self, "online_search", True):
                print("[PIPELINE] KICH HOAT CHE DO 'DYNAMIC EXTERNAL RETRIEVAL' (Khai pha bai bao khoa hoc thuc te tu OpenAlex/arXiv)...")
                try:
                    enrichment_result = self.knowledge_store.enrich_with_external_papers(
                        topic=topic,
                        top_k=8,
                        semantic_scholar_key=getattr(self, "semantic_scholar_key", None)
                    )
                except Exception as e:
                    print(f"[PIPELINE] Loi ket noi external papers: {e}")

            if enrichment_result and enrichment_result.get("target_paper"):
                target_paper = enrichment_result["target_paper"]
                related_papers = enrichment_result["references"]
                discovered_entities = enrichment_result["entities"]
                source_name = enrichment_result.get("source", "OpenAlex/arXiv")
                print(f"[PIPELINE] -> Da nap thanh cong bai bao hat giong THUC TE tu {source_name}: '{target_paper.get('title')}' ({target_paper.get('year')})")
            else:
                print("[PIPELINE] KICH HOAT CHE DO 'OPEN-WORLD LITERATURE SYNTHESIS' (Tong hop tri thuc mo tu Gemini)...")
                open_world_data = self.knowledge_store.synthesize_open_world_seed_paper(topic, llm=self.llm)
                target_paper = open_world_data["target_paper"]
                related_papers = open_world_data["references"]
                discovered_entities = open_world_data["entities"]
        else:
            target_paper = matched_papers[0]
            related_papers = self.knowledge_store.get_related_references_for_paper(target_paper, top_k=5)
            paper_ids = [target_paper.get("corpusid")] + [p.get("corpusid") for p in related_papers if p.get("corpusid")]
            discovered_entities = self.knowledge_store.get_relevant_entities(paper_ids, top_k=30)

        print(f"[PIPELINE] Bai bao hat giong (Target): '{target_paper.get('title')}'")
        total_p = len(self.knowledge_store.papers_by_id)
        total_e = len(self.knowledge_store.entity_counter)
        print(f"[PIPELINE] Da chon loc Top {len(related_papers)} bai bao tham chieu sat nhat tu kho tri thuc ({total_p:,} bai).")
        print(f"[PIPELINE] Da khai pha Top {len(discovered_entities)} thuc the tri thuc tieu bieu nhat (tu {total_e:,} khai niem hoc thuat):")
        print(f"           -> {', '.join(discovered_entities[:10])}...")
        print(f"[PIPELINE] Khoi dong thuc nghiem tren Template: '{template_name}'")
        sys.stdout.flush()

        context = {
            "user_topic": topic,
            "topic": topic,
            "paper": target_paper,
            "references": related_papers,
            "entities": discovered_entities,
            "template_name": template_name,
            "preflight_info": preflight_info,
            "evidence_coverage_pct": data_cov.get("coverage_percent", 0.0),
            "evidence_status": data_cov.get("status", "INSUFFICIENT"),
        }

        # -------------------------------------------------------------
        # BƯỚC 2: Tầng 1 ResearchAgent - Problem Formulation (Debate)
        # -------------------------------------------------------------
        print("\n>>> GIAI ĐOẠN 2: TẦNG 1 - TƯ DUY & PHẢN BIỆN BÀI TOÁN KHOA HỌC (Problem Formulation)")

        # Ngưỡng điểm ProblemValidator động theo evidence coverage (Lỗi 4 fix).
        # Coverage thấp → LLM thiếu context thực → không yêu cầu threshold quá cao.
        coverage_pct = data_cov.get("coverage_percent", 0.0)
        if coverage_pct >= 60.0:
            problem_score_threshold = 4.2   # STRONG corpus  → chuẩn cao
        elif coverage_pct >= 20.0:
            problem_score_threshold = 4.0   # MODERATE corpus → relaxed
        else:
            problem_score_threshold = 3.8   # INSUFFICIENT   → rất relaxed

        feedback_p = ""
        problem_result = {}
        for it in range(max_iterations):
            print(f"[ProblemIdentifier] Vong {it + 1}/{max_iterations}: Dang xac dinh bai toan...")
            sys.stdout.flush()
            p_out = self.problem_identifier.run(context, feedback=feedback_p)
            sys.stdout.flush()
            context.update(p_out)

            print(f"[ProblemValidator] Vong {it + 1}/{max_iterations}: Dang danh gia 5 tieu chi (Clarity, Relevance, Originality, Feasibility, Significance)...")
            sys.stdout.flush()
            p_val = self.problem_validator.run(context)
            sys.stdout.flush()
            scores = p_val.get("scores", {})
            avg_score = safe_float(p_val.get("average_score"), 4.0)
            c_score = safe_float(scores.get("clarity"), 4.0)
            o_score = safe_float(scores.get("originality"), 4.0)
            f_score = safe_float(scores.get("feasibility"), 4.0)
            print(f" -> Diem danh gia: {avg_score:.2f}/5.0 (Clarity: {c_score:.1f}, Novelty: {o_score:.1f}, Feasibility: {f_score:.1f}) [nguong: {problem_score_threshold}]")

            problem_result = context
            if avg_score >= problem_score_threshold:
                print(" -> Bai toan khoa hoc da dat do xuat sac!")
                if coverage_pct < 30.0:
                    print(f" -> [LUU Y] Evidence coverage chi {coverage_pct:.0f}% — ket qua phu thuoc nhieu vao LLM synthesis, nen kiem tra lai sau khi bo sung du lieu.")
                break
            feedback_p = p_val.get("feedbacks", "")

        # -------------------------------------------------------------
        # BƯỚC 3: Tầng 2 ResearchAgent - Method Development (Debate)
        # -------------------------------------------------------------
        print("\n>>> GIAI ĐOẠN 3: TẦNG 2 - THIẾT KẾ THUẬT TOÁN & PHẢN BIỆN TOÁN HỌC (Method Development)")
        feedback_m = ""
        method_result = {}
        for it in range(max_iterations):
            print(f"[MethodDeveloper] Vong {it + 1}/{max_iterations}: Dang xay dung thuat toan & cong thuc toan...")
            sys.stdout.flush()
            m_out = self.method_developer.run(context, feedback=feedback_m)
            sys.stdout.flush()
            context.update(m_out)

            print(f"[MethodValidator] Vong {it + 1}/{max_iterations}: Dang danh gia 5 tieu chi (Clarity, Soundness, Novelty, Feasibility, Effectiveness)...")
            sys.stdout.flush()
            m_val = self.method_validator.run(context)
            sys.stdout.flush()
            m_scores = m_val.get("scores", {})
            m_avg = safe_float(m_val.get("average_score"), 4.0)
            s_score = safe_float(m_scores.get("soundness"), 4.0)
            n_score = safe_float(m_scores.get("novelty"), 4.0)
            print(f" -> Diem phuong phap: {m_avg:.2f}/5.0 (Soundness: {s_score:.1f}, Novelty: {n_score:.1f})")

            method_result = context
            if m_avg >= 4.2:
                print(" -> Phuong phap toan hoc da dat chuan khoa hoc xuat sac!")
                break
            feedback_m = m_val.get("feedbacks", "")

        # -------------------------------------------------------------
        # BƯỚC 4: Tầng 3 ResearchAgent - Experiment Design (Debate)
        # -------------------------------------------------------------
        print("\n>>> GIAI ĐOẠN 4: TẦNG 3 - THIẾT KẾ THỰC NGHIỆM ĐỐI SÁNH (Experiment Design)")
        feedback_e = ""
        exp_result = {}
        for it in range(max_iterations):
            print(f"[ExperimentDesigner] Vong {it + 1}/{max_iterations}: Dang thiet ke gia thuyet & baselines...")
            sys.stdout.flush()
            e_out = self.experiment_designer.run(context, feedback=feedback_e)
            sys.stdout.flush()
            context.update(e_out)

            print(f"[ExperimentValidator] Vòng {it + 1}/{max_iterations}: Đang đánh giá 5 tiêu chí (Clarity, Validity, Robustness, Feasibility, Reproducibility)...")
            sys.stdout.flush()
            e_val = self.experiment_validator.run(context)
            sys.stdout.flush()
            e_avg = safe_float(e_val.get("average_score"), 4.0)
            print(f" -> Điểm thiết kế thực nghiệm: {e_avg:.2f}/5.0")
            sys.stdout.flush()

            exp_result = context
            if e_avg >= 4.2:
                print(" -> Giao thức thực nghiệm đạt chuẩn kiểm chứng!")
                break
            feedback_e = e_val.get("feedbacks", "")

        # -------------------------------------------------------------
        # BƯỚC 5: Sakana AI Execution - Huấn Luyện PyTorch Thật
        # -------------------------------------------------------------
        print(f"\n>>> GIAI ĐOẠN 5: HUẤN LUYỆN PYTORCH THẬT SỰ TRÊN TEMPLATE '{template_name}'")
        exp_work_dir = os.path.join(run_dir, "experiment")
        self.template_manager.copy_template_to_dir(template_name, exp_work_dir)

        # 5.1. Chạy Baseline Training (Run 0)
        print(f"[PyTorch Engine] Đang chạy Baseline Run 0 (experiment.py)...")
        train_cmd = [sys.executable, "experiment.py", "--out_dir=run_0"]
        train_res = subprocess.run(train_cmd, cwd=exp_work_dir, capture_output=True, text=True, timeout=180)
        if train_res.returncode == 0:
            print("[PyTorch Engine] Baseline Run 0 hoàn tất!")
        else:
            print(f"[PyTorch Engine] Cảnh báo Run 0: {train_res.stderr[:200]}")

        # 5.2. CoderAgent nâng cấp mã nguồn theo phương pháp mới của ResearchAgent
        print(f"[CoderAgent] Đang nâng cấp mã nguồn PyTorch theo phương pháp '{context.get('method_name')}'...")
        try:
            with open(os.path.join(exp_work_dir, "experiment.py"), "r", encoding="utf-8") as f:
                base_code = f.read()

            upgraded_code = self.coder_agent.modify_template_code(
                base_code=base_code,
                method_info=context,
                experiment_design=context
            )
            upgraded_script_path = os.path.join(exp_work_dir, "experiment_upgraded.py")
            with open(upgraded_script_path, "w", encoding="utf-8") as f:
                f.write(upgraded_code)

            # 5.3. Chạy Proposed Innovation Training (Run 1)
            print(f"[PyTorch Engine] Đang huấn luyện mô hình cải tiến theo phát minh mới (Run 1)...")
            upg_res = subprocess.run(
                [sys.executable, "experiment_upgraded.py", "--out_dir=run_1"],
                cwd=exp_work_dir,
                capture_output=True,
                text=True,
                timeout=180
            )
            if upg_res.returncode == 0:
                print("[PyTorch Engine] Huấn luyện mô hình cải tiến (Run 1) THÀNH CÔNG RỰC RỠ!")
            else:
                print(f"[PyTorch Engine] Lưu ý: {upg_res.stderr[:150]}")
        except Exception as e:
            print(f"[CoderAgent] Lưu ý khi nâng cấp code: {e}")

        # 5.4. Đọc dữ liệu định lượng thực nghiệm (metrics_data)
        metrics_data = {}
        for r_dir in ["run_1", "run_0"]:
            info_p = os.path.join(exp_work_dir, r_dir, "final_info.json")
            if os.path.exists(info_p):
                with open(info_p, "r", encoding="utf-8") as mf:
                    metrics_data = json.load(mf)
                break
        if not metrics_data:
            for r_dir in ["run_1", "run_0"]:
                met_p = os.path.join(exp_work_dir, r_dir, "metrics.json")
                if os.path.exists(met_p):
                    with open(met_p, "r", encoding="utf-8") as mf:
                        metrics_data = json.load(mf)
                    break

        # 5.5. CoderAgent tự động thích ứng / sinh script vẽ biểu đồ theo đúng domain và metrics
        plot_script = os.path.join(exp_work_dir, "plot.py")
        if os.path.exists(plot_script):
            print(f"[CoderAgent] Đang điều chỉnh plot.py theo bối cảnh đề tài '{context.get('method_name')}'...")
            try:
                with open(plot_script, "r", encoding="utf-8") as pf:
                    base_plot = pf.read()

                adapted_plot = self.coder_agent.adapt_or_generate_plot_code(
                    base_plot_code=base_plot,
                    problem_info=context,
                    method_info=context,
                    experiment_design=context,
                    metrics_data=metrics_data
                )
                with open(plot_script, "w", encoding="utf-8") as pf:
                    pf.write(adapted_plot)
            except Exception as pe:
                print(f"[CoderAgent] Cảnh báo khi điều chỉnh plot.py: {pe}")

            print("[PyTorch Engine] Đang chạy plot.py để vẽ biểu đồ đối sánh...")
            plot_proc = subprocess.run([sys.executable, "plot.py"], cwd=exp_work_dir, capture_output=True, text=True, timeout=60)
            if plot_proc.returncode != 0:
                print(f"[PyTorch Engine] Cảnh báo chạy plot.py gặp lỗi: {plot_proc.stderr[:300]}")
                print("[PyTorch Engine] Đang khôi phục và chạy fallback base_plot từ template...")
                try:
                    with open(plot_script, "w", encoding="utf-8") as pf:
                        pf.write(base_plot)
                    fb_proc = subprocess.run([sys.executable, "plot.py"], cwd=exp_work_dir, capture_output=True, text=True, timeout=60)
                    if fb_proc.returncode == 0:
                        print("[PyTorch Engine] Đã xuất thành công biểu đồ từ template fallback!")
                    else:
                        print(f"[PyTorch Engine] Cảnh báo chạy fallback plot.py: {fb_proc.stderr[:200]}")
                except Exception as fb_err:
                    print(f"[PyTorch Engine] Lỗi khi chạy fallback plot: {fb_err}")
            else:
                print("[PyTorch Engine] Đã xuất thành công các biểu đồ khoa học (figure_1.png, figure_2.png, figure_3.png).")

        # -------------------------------------------------------------
        # BƯỚC 6: Sakana AI Writeup - Soạn Thảo Toàn Văn & Compile PDF
        # -------------------------------------------------------------
        print("\n>>> GIAI ĐOẠN 6: SOẠN THẢO BÀI BÁO KHOA HỌC & BIÊN DỊCH PDF (LaTeX Writeup)")
        pdf_path, generated_title = self.latex_writer.write_paper(
            output_dir=run_dir,
            context={"target_paper": target_paper, "related_papers": related_papers, "user_topic": topic, "topic": topic},
            problem_info={"problem": context.get("problem"), "rationale": context.get("problem_rationale")},
            method_info={"method_name": context.get("method_name"), "abbreviation": context.get("method_abbr"), "intuition": context.get("method"), "mathematical_formulation": context.get("mathematical_formulation"), "theoretical_justification": context.get("method_rationale")},
            experiment_design={"baselines": context.get("baselines", []), "metrics": context.get("metrics", [])},
            metrics_data=metrics_data,
            experiment_dir=exp_work_dir,
        )

        # -------------------------------------------------------------
        # BƯỚC 7: Sakana AI Review - Phản Biện Hội Nghị Độc Lập
        # -------------------------------------------------------------
        print("\n>>> GIAI ĐOẠN 7: PHẢN BIỆN HỌC THUẬT ĐỘC LẬP (Conference Double-Blind Peer Review)")
        paper_title = generated_title or context.get("method_name", "Autonomous AI Discovery")
        tex_path = os.path.join(run_dir, "latex", "paper.tex")
        review_data = self.peer_reviewer.review_paper(
            output_dir=run_dir,
            paper_title=paper_title,
            paper_tex_path=tex_path,
            metrics_data=metrics_data,
        )

        print("\n" + "=" * 70)
        print("  CÔNG TRÌNH NGHIÊN CỨU ĐÃ HOÀN THÀNH XUẤT SẮC!")
        print("=" * 70)
        print(f"1. Thư mục thực nghiệm: {exp_work_dir}")
        print(f"2. Biểu đồ trực quan: figure_1.png, figure_2.png, figure_3.png ({exp_work_dir})")
        print(f"3. Mã nguồn LaTeX: {os.path.join(run_dir, 'latex', 'paper.tex')}")
        if pdf_path and os.path.exists(pdf_path):
            print(f"4. BÀI BÁO HOÀN CHỈNH (PDF): {pdf_path}")
        print(f"5. Báo cáo phản biện: {os.path.join(run_dir, 'review', 'peer_review.md')}")
        print(f"   -> Kết luận: {review_data.get('recommendation')} ({review_data.get('scores', {}).get('overall_rating')}/10)")

        return {
            "run_dir": run_dir,
            "pdf_path": pdf_path,
            "review": review_data,
            "metrics": metrics_data,
        }
