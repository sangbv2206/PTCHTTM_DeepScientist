import datetime
import json
import os
import subprocess
import sys
from typing import Callable, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from core.gemini_llm import GeminiLLM
from core.knowledge_store import KnowledgeStore
from core.external_retriever import check_internet_connection
from agents.research_agents import (
    ProblemIdentifier, ProblemValidator,
    MethodDeveloper, MethodValidator,
    ExperimentDesigner, ExperimentValidator,
)
from agents.experiment_agents import CoderAgent
from templates.template_manager import TemplateManager
from agents.latex_writer import LaTeXWriter
from agents.peer_reviewer import PeerReviewer
import time
from core.contracts import EnvironmentInfo, FigureManifestItem, validate_results_data
from core.claim_ledger import ClaimLedger
from core.universal_plotter import UniversalPlotter


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


from pipeline.domain_registry import (
    extract_domain_context,
    get_dataset_info_for_template,
    get_domain_hyperparameters,
    TEMPLATE_DATASET_INFO,
)



class DeepScientistPipeline:
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        data_dir: Optional[str] = None,
        output_base_dir: Optional[str] = None,
        online_search: bool = True,
        semantic_scholar_key: Optional[str] = None,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        print("=" * 70)
        print("  DEEPSCIENTIST: AUTONOMOUS SCIENTIFIC DISCOVERY")
        print("=" * 70)

        self.online_search = online_search
        self.semantic_scholar_key = semantic_scholar_key
        self.event_callback = event_callback
        self.llm = GeminiLLM(api_key=api_key, model_name=model_name)
        self.template_manager = TemplateManager()

        self.knowledge_store = KnowledgeStore(data_dir=data_dir)
        if not check_internet_connection(timeout=2.0):
            print("[KnowledgeStore] Không phát hiện kết nối Internet. Kích hoạt chế độ dự phòng ngoại tuyến (papers.jsonl)...")
            self.knowledge_store.load_all_data()
        else:
            print("[KnowledgeStore] Đã xác thực kết nối Internet. Kích hoạt cơ chế cào trực tuyến (OpenAlex & arXiv API). Tối ưu hóa bộ nhớ RAM.")

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

    def _emit_event(
        self,
        run_id: str,
        stage: int,
        status: str,
        message: str,
        progress: float,
        run_dir: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.event_callback:
            return
        payload: Dict[str, Any] = {
            "run_id": run_id,
            "stage": stage,
            "status": status,
            "message": message,
            "progress": progress,
        }
        if run_dir:
            payload["run_dir"] = run_dir
        if result is not None:
            payload["result"] = result
        try:
            self.event_callback(payload)
        except Exception as error:
            print(f"[EventCallback] Error emitting event: {error}")

    def _normalize_topic_to_english(self, topic: str) -> str:
        """Chuẩn hóa đề tài sang tiếng Anh học thuật nếu người dùng nhập tiếng Việt hoặc ngôn ngữ khác."""
        topic_clean = (topic or "").strip()
        has_non_ascii = any(ord(c) > 127 for c in topic_clean)
        if not has_non_ascii:
            return topic_clean

        prompt = f"""You are an elite scientific translator and academic taxonomy expert.
Translate and refine the following research topic into a standard, concise, high-impact English academic research topic title (3 to 10 words):
"{topic_clean}"

Rules:
1. Translate accurately into professional academic English commonly used in top-tier machine learning and computer science venues (e.g., NeurIPS, ICML, ICLR, IEEE Transactions).
2. DO NOT add conversational filler, quotes, explanations, markdown formatting, or prefix/suffix.
3. Output ONLY the raw English topic title.
"""
        try:
            translated = self.llm.generate(prompt).strip()
            translated = re.sub(r'^["\'`]+|["\'`]+$', '', translated).strip()
            if translated and len(translated) >= 3:
                print(f"[PIPELINE] Tự động chuẩn hóa đề tài sang tiếng Anh học thuật:")
                print(f"           Gốc: '{topic_clean}'")
                print(f"           Anh: '{translated}'\n")
                return translated
        except Exception as e:
            print(f"[PIPELINE] Lưu ý khi dịch chuẩn hóa đề tài: {e}")
        return topic_clean

    def run(
        self,
        topic: str,
        template_name: Optional[str] = None,
        max_iterations: int = 2,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{run_id}" if run_id else f"run_{timestamp}"
        run_dir = os.path.join(self.output_base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        original_topic = (topic or "").strip()
        topic_en = self._normalize_topic_to_english(original_topic)
        topic = topic_en

        self._emit_event(run_id or run_name, 0, "running", "Research run started", 0.0, run_dir=run_dir)

        print(f"\n[PIPELINE] Bắt đầu nghiên cứu đề tài: '{topic}' (Gốc: '{original_topic}')")
        print(f"[PIPELINE] Thư mục kết quả: {run_dir}\n")

        self._emit_event(run_id or run_name, 0, "running", "Running preflight assessment", 0.04, run_dir=run_dir)

        # -------------------------------------------------------------
        # BƯỚC 0: KIỂM ĐỊNH ĐỘ PHỦ DỮ LIỆU & ĐỘ TƯƠNG THÍCH TEMPLATE
        # -------------------------------------------------------------
        print(">>> GIAI ĐOẠN 0: KIỂM ĐỊNH ĐỘ PHỦ DỮ LIỆU & ĐỘ TƯƠNG THÍCH TEMPLATE (Pre-Flight Assessment)")
        is_net_available = check_internet_connection(timeout=2.0)
        data_cov = self.knowledge_store.assess_topic_coverage(topic, is_online=is_net_available)
        tpl_suit = self.template_manager.assess_template_suitability(topic, llm=self.llm)

        if is_net_available:
            print("[Pre-Flight] Chế độ tài liệu: TRỰC TUYẾN 100% (OpenAlex & arXiv API) - Tối ưu hóa bộ nhớ, không nạp file tĩnh.")
        else:
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

        # Kiểm tra tính nhất quán ngữ nghĩa giữa Đề tài và Bản chất Template
        t_low = (topic or "").lower()
        if "continuous" in t_low and template_name == "reinforcement_learning":
            print("\n" + "=" * 70)
            print(" [KIỂM DUYỆT NHẤT QUÁN ĐỀ TÀI & TEMPLATE (SEMANTIC FIDELITY GUARD)]:")
            print(f" Đề tài: '{topic}'")
            print(" -> PHÁT HIỆN: Đề tài yêu cầu điều khiển liên tục ('continuous'),")
            print("    nhưng template 'reinforcement_learning' sử dụng CartPole rời rạc (Discrete Action: Left/Right).")
            print(" -> GIẢI PHÁP HỆ THỐNG: Kích hoạt chỉ dẫn 'Action-Space Discretization Mapping'")
            print("    để các Agent bám sát bản chất CartPole, tuyệt đối không bịa đặt môi trường ảo!")
            print("=" * 70 + "\n")

        # Lưu báo cáo kiểm định trước chuyến bay
        preflight_info = {
            "topic": topic,
            "original_topic": original_topic,
            "data_coverage": data_cov,
            "template_suitability": tpl_suit,
            "selected_template": template_name
        }
        with open(os.path.join(run_dir, "preflight_assessment.json"), "w", encoding="utf-8") as pf:
            json.dump(preflight_info, pf, indent=2, ensure_ascii=False)

        # -------------------------------------------------------------
        # BƯỚC 1: Khai phá Đồ thị Tri thức Toàn cục (ResearchAgent Store)
        # -------------------------------------------------------------
        self._emit_event(run_id or run_name, 1, "running", "Searching literature and knowledge graph", 0.14, run_dir=run_dir)
        print("\n>>> GIAI ĐOẠN 1: KHAI PHÁ ĐỒ THỊ TRI THỨC TOÀN CỤC (ResearchAgent Global Graph)")

        # Kiểm tra nhanh kết nối Internet (Fast Ping 2s)
        is_online = check_internet_connection(timeout=2.0)
        target_paper = None
        related_papers = []
        discovered_entities = []

        # 1. Ưu tiên hàng đầu khi có mạng: 100% lấy dữ liệu trực tuyến từ OpenAlex & arXiv API (Không dùng kho offline)
        if is_online:
            print("[PIPELINE] Kiểm tra kết nối Internet (Fast Ping 2s): KẾT NỐI THÀNH CÔNG!")
            print("[PIPELINE] CHẾ ĐỘ TRỰC TUYẾN: Thu thập dữ liệu học thuật thời gian thực từ OpenAlex & arXiv API.")
            try:
                enrichment_result = self.knowledge_store.enrich_with_external_papers(
                    topic=topic,
                    top_k=40,
                    semantic_scholar_key=getattr(self, "semantic_scholar_key", None),
                    save_cache=False
                )
                if enrichment_result and enrichment_result.get("target_paper"):
                    target_paper = enrichment_result["target_paper"]
                    related_papers = enrichment_result["references"]
                    discovered_entities = enrichment_result["entities"]
                    source_name = enrichment_result.get("source", "OpenAlex/arXiv")
                    print(f"[PIPELINE] -> KẾT NỐI ONLINE THÀNH CÔNG! Đã nạp bài báo hạt giống: '{target_paper.get('title')}' ({target_paper.get('year')}) từ {source_name}")
                    print(f"[PIPELINE] -> Thu thập được {len(related_papers)} tài liệu tham khảo thực tế trực tiếp từ mạng!")
            except Exception as e:
                print(f"[PIPELINE] Cảnh báo lỗi truy vấn trực tuyến: {e}")

        # 2. Phương án dự phòng: CHỈ KÍCH HOẠT KHI MẤT KẾT NỐI MẠNG HOÀN TOÀN
        if not target_paper:
            print("[PIPELINE] Phát hiện KHÔNG CÓ KẾT NỐI MẠNG (Hoặc mất mạng trên laptop).")
            print("[PIPELINE] KÍCH HOẠT DỰ PHÒNG NGOẠI TUYẾN: Nạp kho papers.jsonl trong máy để phục vụ chạy terminal offline...")
            if not self.knowledge_store.is_loaded:
                self.knowledge_store.load_all_data()
            matched_papers_with_scores = self.knowledge_store.search_target_papers_with_scores(topic, top_k=50)
            matched_papers = [p for _, p in matched_papers_with_scores]
            if matched_papers_with_scores:
                print(f"[PIPELINE] Kết quả tìm kiếm trong kho offline (top 5 bài phù hợp nhất):")
                for i, (sc, pp) in enumerate(matched_papers_with_scores[:5]):
                    print(f"  [{i+1}] score={sc:6.1f}  {pp.get('title', 'N/A')[:75]}")
            if matched_papers:
                target_paper = matched_papers[0]
                direct_refs = self.knowledge_store.get_related_references_for_paper(target_paper, top_k=25)
                seen_cids = {target_paper.get("corpusid")}
                combined_refs = []
                for p in direct_refs + matched_papers[1:40]:
                    cid = p.get("corpusid")
                    if cid and cid not in seen_cids:
                        seen_cids.add(cid)
                        combined_refs.append(p)
                    elif not cid and p not in combined_refs:
                        combined_refs.append(p)
                related_papers = combined_refs[:35]
                paper_ids = [target_paper.get("corpusid")] + [p.get("corpusid") for p in related_papers if p.get("corpusid")]
                discovered_entities = self.knowledge_store.get_relevant_entities(paper_ids, top_k=30)
            else:
                print("[PIPELINE] KICH HOAT CHE DO 'OPEN-WORLD LITERATURE SYNTHESIS' (Tong hop tri thuc mo tu Gemini)...")
                open_world_data = self.knowledge_store.synthesize_open_world_seed_paper(topic, llm=self.llm)
                target_paper = open_world_data["target_paper"]
                related_papers = open_world_data["references"]
                discovered_entities = open_world_data["entities"]

        print(f"[PIPELINE] Bài báo hạt giống định hướng (Seed Paper): '{target_paper.get('title')}'")
        total_p = len(self.knowledge_store.papers_by_id)
        total_e = len(self.knowledge_store.entity_counter)
        if is_online:
            print(f"[PIPELINE] Đã tuyển chọn Top {len(related_papers)} tài liệu tham khảo trực tuyến sát nhất làm tập ứng viên trích dẫn.")
        else:
            print(f"[PIPELINE] Đã tuyển chọn Top {len(related_papers)} tài liệu tham khảo sát nhất từ kho ngoại tuyến ({total_p:,} bài).")
        print(f"[PIPELINE] Đã khai phá Top {len(discovered_entities)} khái niệm và thực thể học thuật tiêu biểu:")
        print(f"           -> {', '.join(discovered_entities[:10])}...")
        print(f"[PIPELINE] Khởi động thực nghiệm trên Template: '{template_name}'")
        sys.stdout.flush()

        context = {
            "user_topic": topic,
            "topic": topic,
            "original_topic": original_topic,
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
        self._emit_event(run_id or run_name, 2, "running", "Formulating and validating the research problem", 0.28, run_dir=run_dir)
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
            print(f"[ProblemIdentifier] Vòng {it + 1}/{max_iterations}: Đang xác định bài toán khoa học...")
            sys.stdout.flush()
            p_out = self.problem_identifier.run(context, feedback=feedback_p)
            sys.stdout.flush()
            context.update(p_out)

            print(f"[ProblemValidator] Vòng {it + 1}/{max_iterations}: Đang thẩm định 5 tiêu chí (Rõ ràng, Thích hợp, Tính mới, Khả thi, Ý nghĩa)...")
            sys.stdout.flush()
            p_val = self.problem_validator.run(context)
            sys.stdout.flush()
            scores = p_val.get("scores", {})
            avg_score = safe_float(p_val.get("average_score"), 4.0)
            c_score = safe_float(scores.get("clarity"), 4.0)
            o_score = safe_float(scores.get("originality"), 4.0)
            f_score = safe_float(scores.get("feasibility"), 4.0)
            print(f" -> Điểm đánh giá: {avg_score:.2f}/5.0 (Clarity: {c_score:.1f}, Novelty: {o_score:.1f}, Feasibility: {f_score:.1f}) [Ngưỡng: {problem_score_threshold}]")

            problem_result = context
            if avg_score >= problem_score_threshold:
                print(" -> Bài toán khoa học đã đạt độ xuất sắc!")
                if coverage_pct < 30.0:
                    print(f" -> [LƯU Ý] Độ phủ tri thức chỉ đạt {coverage_pct:.0f}% — kết quả phụ thuộc nhiều vào tổng hợp của mô hình.")
                break
            feedback_p = p_val.get("feedbacks", "")

        # -------------------------------------------------------------
        # BƯỚC 3: Tầng 2 ResearchAgent - Method Development (Debate)
        # -------------------------------------------------------------
        self._emit_event(run_id or run_name, 3, "running", "Developing and validating the method", 0.42, run_dir=run_dir)
        print("\n>>> GIAI ĐOẠN 3: TẦNG 2 - THIẾT KẾ THUẬT TOÁN & PHẢN BIỆN TOÁN HỌC (Method Development)")
        feedback_m = ""
        method_result = {}
        for it in range(max_iterations):
            print(f"[MethodDeveloper] Vòng {it + 1}/{max_iterations}: Đang xây dựng thuật toán & công thức toán học...")
            sys.stdout.flush()
            m_out = self.method_developer.run(context, feedback=feedback_m)
            sys.stdout.flush()
            context.update(m_out)

            print(f"[MethodValidator] Vòng {it + 1}/{max_iterations}: Đang thẩm định 5 tiêu chí (Rõ ràng, Chặt chẽ, Tính mới, Khả thi, Hiệu quả)...")
            sys.stdout.flush()
            m_val = self.method_validator.run(context)
            sys.stdout.flush()
            m_scores = m_val.get("scores", {})
            m_avg = safe_float(m_val.get("average_score"), 4.0)
            s_score = safe_float(m_scores.get("soundness"), 4.0)
            n_score = safe_float(m_scores.get("novelty"), 4.0)
            print(f" -> Điểm phương pháp: {m_avg:.2f}/5.0 (Soundness: {s_score:.1f}, Novelty: {n_score:.1f})")

            method_result = context
            if m_avg >= 4.2:
                print(" -> Phương pháp toán học đã đạt chuẩn khoa học xuất sắc!")
                break
            feedback_m = m_val.get("feedbacks", "")

        # -------------------------------------------------------------
        # BƯỚC 4: Tầng 3 ResearchAgent - Experiment Design (Debate)
        # -------------------------------------------------------------
        self._emit_event(run_id or run_name, 4, "running", "Designing and validating the experiment", 0.54, run_dir=run_dir)
        print("\n>>> GIAI ĐOẠN 4: TẦNG 3 - THIẾT KẾ THỰC NGHIỆM ĐỐI SÁNH (Experiment Design)")
        feedback_e = ""
        exp_result = {}
        for it in range(max_iterations):
            print(f"[ExperimentDesigner] Vòng {it + 1}/{max_iterations}: Đang thiết kế giả thuyết & phương pháp đối chứng (Baselines)...")
            sys.stdout.flush()
            e_out = self.experiment_designer.run(context, feedback=feedback_e)
            sys.stdout.flush()
            context.update(e_out)

            print(f"[ExperimentValidator] Vòng {it + 1}/{max_iterations}: Đang thẩm định 5 tiêu chí (Rõ ràng, Tính hợp lệ, Độ bền vững, Khả thi, Tái lập)...")
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
        self._emit_event(run_id or run_name, 5, "running", "Running baseline and proposed experiments", 0.66, run_dir=run_dir)
        print(f"\n>>> GIAI ĐOẠN 5: HUẤN LUYỆN PYTORCH THẬT SỰ TRÊN TEMPLATE '{template_name}'")
        exp_work_dir = os.path.join(run_dir, "experiment")
        self.template_manager.copy_template_to_dir(template_name, exp_work_dir)

        # Trích xuất và inject domain_context.json tự động bằng AI từ template thực tế
        domain_ctx = extract_domain_context(topic, template_name, llm=self.llm)
        with open(os.path.join(exp_work_dir, "domain_context.json"), "w", encoding="utf-8") as f_ctx:
            json.dump(domain_ctx, f_ctx, indent=2, ensure_ascii=False)
        print(f"[Evidence Tracker] AI tự động phân giải ngữ cảnh miền: '{domain_ctx['domain']}' -> {domain_ctx['series_name']} ({domain_ctx['unit_label']})")
        print(f"[Evidence Tracker] Dataset thực tế được AI xác định: '{domain_ctx.get('dataset_info')}'")

        # 5.1. Chạy Baseline Training (Run 0)
        TRAIN_TIMEOUT = 1800
        print(f"[PyTorch Engine] Đang chạy Baseline Run 0 (experiment.py)... [timeout={TRAIN_TIMEOUT}s]")
        train_cmd = [sys.executable, "-X", "utf8", "experiment.py", "--out_dir=run_0"]
        import copy
        train_env = copy.copy(os.environ.copy())
        train_env["PYTHONIOENCODING"] = "utf-8"
        train_env["PYTHONUTF8"] = "1"
        train_env["DEEPSCIENTIST_MAX_STEPS"] = "1000"  # Quick-run 1000 steps
        
        train_start_time = time.time()
        baseline_stderr_log = os.path.join(exp_work_dir, "baseline_stderr.log")
        
        try:
            train_res = subprocess.run(
                train_cmd, cwd=exp_work_dir, capture_output=True, text=True,
                timeout=TRAIN_TIMEOUT, env=train_env, encoding="utf-8", errors="replace"
            )
            with open(baseline_stderr_log, "w", encoding="utf-8") as f_err:
                f_err.write(train_res.stderr or "")
            if train_res.returncode == 0:
                print("[PyTorch Engine] Baseline Run 0 hoàn tất!")
            else:
                print(f"[PyTorch Engine] Cảnh báo Run 0 (Exit {train_res.returncode}): Log ghi tại {baseline_stderr_log}")
        except subprocess.TimeoutExpired:
            print(f"[PyTorch Engine] Run 0 vượt quá {TRAIN_TIMEOUT}s timeout — bỏ qua, tiếp tục pipeline.")
            with open(baseline_stderr_log, "w", encoding="utf-8") as f_err:
                f_err.write("TimeoutExpired")

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

            # 5.3. Chạy Proposed Innovation Training (Run 1) với cơ chế Tự sửa lỗi (Self-Healing Loop)
            print(f"[PyTorch Engine] Đang huấn luyện mô hình cải tiến theo phát minh mới (Run 1)... [timeout={TRAIN_TIMEOUT}s]")
            upgraded_stderr_log = os.path.join(exp_work_dir, "upgraded_stderr.log")
            
            MAX_CODE_RETRIES = 2
            run1_success = False

            for attempt in range(MAX_CODE_RETRIES + 1):
                try:
                    upg_res = subprocess.run(
                        [sys.executable, "-X", "utf8", "experiment_upgraded.py", "--out_dir=run_1"],
                        cwd=exp_work_dir,
                        capture_output=True,
                        text=True,
                        timeout=TRAIN_TIMEOUT,
                        env=train_env,
                        encoding="utf-8",
                        errors="replace"
                    )
                    with open(upgraded_stderr_log, "w", encoding="utf-8") as f_err:
                        f_err.write(upg_res.stderr or "")

                    if upg_res.returncode == 0:
                        print("[PyTorch Engine] Huấn luyện mô hình cải tiến (Run 1) THÀNH CÔNG RỰC RỠ!")
                        run1_success = True
                        break
                    else:
                        err_snippet = (upg_res.stderr or "").strip()
                        last_err = err_snippet.splitlines()[-1] if err_snippet else "Unknown error"
                        print(f"[PyTorch Engine] Cảnh báo Run 1 (Lần {attempt+1}/{MAX_CODE_RETRIES+1} - Exit {upg_res.returncode}): {last_err}")
                        if attempt < MAX_CODE_RETRIES:
                            print(f"[CoderAgent] Kích hoạt VÒNG LẶP TỰ SỬA LỖI (Self-Healing)... Đang yêu cầu CoderAgent phân tích lỗi và sửa lại mã nguồn...")
                            upgraded_code = self.coder_agent.modify_template_code(
                                base_code=upgraded_code,
                                method_info=context,
                                experiment_design=context,
                                error_context=err_snippet
                            )
                            with open(upgraded_script_path, "w", encoding="utf-8") as f:
                                f.write(upgraded_code)
                except subprocess.TimeoutExpired:
                    print(f"[PyTorch Engine] Run 1 vượt quá {TRAIN_TIMEOUT}s timeout.")
                    with open(upgraded_stderr_log, "w", encoding="utf-8") as f_err:
                        f_err.write("TimeoutExpired")
                    break
        except Exception as e:
            print(f"[CoderAgent] Lưu ý khi nâng cấp code: {e}")

        total_wall_time = time.time() - train_start_time

        # Thu thập thông số môi trường thực tế (Điều #2 - Không bịa môi trường)
        env_info = EnvironmentInfo.collect(wall_time_sec=total_wall_time, is_sandboxed=False)
        env_path = os.path.join(run_dir, "environment.json")
        env_info.save(env_path)
        print(f"[Evidence Tracker] Đã ghi nhận thông số môi trường thực tế: {env_info.device_name}, CPU: {env_info.cpu_count}, RAM: {env_info.ram_gb}GB")

        # 5.4. Đọc dữ liệu định lượng thực nghiệm cho cả Baseline (run_0) và Proposed (run_1)
        baseline_metrics = {}
        proposed_metrics = {}

        # 5.4a. Đọc Baseline metrics
        for fname in ["final_info.json", "metrics.json"]:
            p = os.path.join(exp_work_dir, "run_0", fname)
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as mf:
                        baseline_metrics = json.load(mf)
                    break
                except Exception:
                    pass

        # 5.4b. Đọc Proposed metrics
        for fname in ["final_info.json", "metrics.json"]:
            p = os.path.join(exp_work_dir, "run_1", fname)
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as mf:
                        proposed_metrics = json.load(mf)
                    break
                except Exception:
                    pass

        # TUYỆT ĐỐI KHÔNG COPY số liệu Baseline sang Proposed nếu Run 1 thất bại!
        if not proposed_metrics and baseline_metrics:
            print("[Evidence Tracker] LƯU Ý: Run 1 không sinh ra metrics độc lập, chỉ hiển thị số liệu Baseline chuẩn.")

        # Thiết lập full_metrics đa mô hình để ClaimLedger và Table 1 có đầy đủ đối sánh
        full_metrics = {}
        baseline_label = f"Standard Baseline ({template_name.replace('_', ' ').title()})"
        proposed_label = f"Proposed ({context.get('method_abbr') or context.get('method_name', 'Ours')})"

        if baseline_metrics:
            full_metrics[baseline_label] = baseline_metrics
        if proposed_metrics:
            full_metrics[proposed_label] = proposed_metrics

        metrics_data = proposed_metrics or baseline_metrics

        # -------------------------------------------------------------
        # ĐIỂM DỪNG BẮT BUỘC (FAIL-CLOSED GATE - Điều #1, #18)
        # -------------------------------------------------------------
        val_errors = validate_results_data({"metrics": full_metrics} if full_metrics else {})
        if val_errors:
            print("\n" + "!" * 75)
            print(" [KIỂM DUYỆT SỐ LIỆU - PHÁT HIỆN BẤT THƯỜNG]:")
            for err in val_errors[:10]:
                print(f"  - {err}")
            print("!" * 75 + "\n")

        if not full_metrics:
            print("\n" + "!" * 75)
            print(" [HỢP ĐỒNG ĐẦU RA - VI PHẠM ĐIỀU #1, #18]: THỰC NGHIỆM THẤT BẠI!")
            print(" Quá trình huấn luyện không sinh ra file kết quả (metrics.json / final_info.json).")
            print(" HỆ THỐNG DỪNG LẠI (FAIL-CLOSED): Tuyệt đối không sinh bài báo giả mạo số liệu.")
            print(" Bạn có thể kiểm tra baseline_stderr.log và upgraded_stderr.log trong thư mục experiment/.")
            print("!" * 75 + "\n")
            failure_report = {
                "topic": topic,
                "status": "EXPERIMENT_FAILED",
                "reason": "Training process produced no valid metrics. Execution halted to prevent hallucinated paper.",
                "env": env_info.to_dict(),
                "logs": {
                    "baseline_log": baseline_stderr_log,
                    "upgraded_log": upgraded_stderr_log
                }
            }
            with open(os.path.join(run_dir, "failure_report.json"), "w", encoding="utf-8") as ff:
                json.dump(failure_report, ff, indent=2, ensure_ascii=False)
            return {
                "status": "FAILED",
                "run_dir": run_dir,
                "reason": "EXPERIMENT_FAILED"
            }

        # Kiểm tra tính hợp lệ của số liệu (Sanity Check: Chance-level performance)
        for label, m_dict in full_metrics.items():
            if isinstance(m_dict, dict):
                test_acc = m_dict.get("test_acc") or m_dict.get("test_accuracy") or m_dict.get("accuracy")
                if test_acc is not None:
                    try:
                        acc_val = float(test_acc)
                        # Nếu kết quả accuracy <= 0.15 (hoặc <= 15.0%) trên bài toán phân loại đa lớp
                        norm_acc = acc_val if acc_val <= 1.0 else acc_val / 100.0
                        if 0.0 < norm_acc <= 0.15:
                            print("\n" + "!" * 75)
                            print(f" [CẢNH BÁO CHẤT LƯỢNG THỰC NGHIỆM]: {label} có test accuracy = {norm_acc*100:.2f}% (Mức đoán ngẫu nhiên / Chance level).")
                            print(" Mô hình chưa học được đặc trưng hữu ích (underfitting/mock data).")
                            print("!" * 75 + "\n")
                    except (ValueError, TypeError):
                        pass

        # 5.4b. Sinh biểu đồ thực nghiệm từ dữ liệu log thực tế (plot.py)
        # Ghi nhãn hiển thị chính thức cho plot.py thay vì tên folder nội bộ run_0/run_1
        run_labels_data = {
            "run_0": f"Baseline ({template_name.replace('_', ' ').title()})",
            "run_1": context.get("method_abbr") or "Proposed"
        }
        try:
            with open(os.path.join(exp_work_dir, "run_labels.json"), "w", encoding="utf-8") as rlf:
                json.dump(run_labels_data, rlf, indent=2, ensure_ascii=False)
        except Exception:
            pass

        plot_script = os.path.join(exp_work_dir, "plot.py")
        if os.path.exists(plot_script):
            print(f"[PyTorch Engine] Đang chạy plot.py để vẽ biểu đồ trực quan từ số liệu thật...")
            try:
                plot_res = subprocess.run(
                    [sys.executable, "-X", "utf8", "plot.py"],
                    cwd=exp_work_dir,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=train_env,
                    encoding="utf-8",
                    errors="replace"
                )
                if plot_res.returncode == 0:
                    print("[PyTorch Engine] Đã sinh thành công các biểu đồ thực nghiệm từ log thật!")
                else:
                    print(f"[PyTorch Engine] Cảnh báo khi chạy plot.py: {plot_res.stderr[:200]}")
            except Exception as e:
                print(f"[PyTorch Engine] Ngoại lệ khi chạy plot.py: {e}")

        # 5.4c. Bảo đảm các hình ảnh thực nghiệm luôn tồn tại và bám sát log số liệu thật (UniversalPlotter)
        UniversalPlotter.ensure_grounded_figures(exp_work_dir, full_metrics, user_topic=topic)

        # 5.5. Quét hình ảnh thực tế từ thư mục thực nghiệm (Không hardcode 8 ảnh)
        figure_manifest = []
        for f in sorted(os.listdir(exp_work_dir)):
            if f.endswith(".png"):
                figure_manifest.append(FigureManifestItem(
                    filename=f,
                    figure_type="empirical_result",
                    caption=f"Empirical outcome visualization ({f})",
                    data_source=os.path.join(exp_work_dir, f)
                ))
        print(f"[Evidence Tracker] Tìm thấy {len(figure_manifest)} hình ảnh thực tế từ thực nghiệm: {[item.filename for item in figure_manifest]}")

        # 5.6. Thiết lập Sổ Cái Tuyên Bố Thực Nghiệm (ClaimLedger)
        claim_ledger = ClaimLedger(
            results_data={"metrics": full_metrics},
            config_data=context,
            env_data=env_info.to_dict()
        )
        ledger_path = os.path.join(run_dir, "claim_ledger.md")
        with open(ledger_path, "w", encoding="utf-8") as lf:
            lf.write(claim_ledger.render_claim_ledger_markdown())
        print(f"[Evidence Tracker] Đã lưu Sổ Cái Tuyên Bố (Claim Ledger) tại: {ledger_path}")

        # -------------------------------------------------------------
        # BƯỚC 6: Sakana AI Writeup - Soạn Thảo Toàn Văn & Compile PDF
        # -------------------------------------------------------------
        self._emit_event(run_id or run_name, 6, "running", "Writing and compiling the paper", 0.82, run_dir=run_dir)
        # Sử dụng lại domain_ctx đã được AI phân giải ở Bước 5 hoặc gọi AI trích xuất động
        if "domain_ctx" not in locals() or not domain_ctx:
            domain_ctx = extract_domain_context(topic, template_name, llm=self.llm)
        dataset_info_str = get_dataset_info_for_template(template_name, domain_ctx)
        pdf_path, generated_title = self.latex_writer.write_paper(
            output_dir=run_dir,
            context={"target_paper": target_paper, "related_papers": related_papers, "user_topic": topic, "topic": topic},
            problem_info={"problem": context.get("problem"), "rationale": context.get("problem_rationale")},
            method_info={"method_name": context.get("method_name"), "abbreviation": context.get("method_abbr"), "intuition": context.get("method"), "mathematical_formulation": context.get("mathematical_formulation"), "theoretical_justification": context.get("method_rationale")},
            experiment_design={
                "baselines": [baseline_label] if baseline_metrics else context.get("baselines", []),
                "metrics": context.get("metrics", []),
                "dataset_info": dataset_info_str,
                "template_name": template_name,
                "hyperparameters": get_domain_hyperparameters(template_name),
            },
            metrics_data=full_metrics,
            experiment_dir=exp_work_dir,
            claim_ledger=claim_ledger,
            figure_manifest=figure_manifest,
            env_info=env_info,
        )

        # -------------------------------------------------------------
        # BƯỚC 7: Sakana AI Review - Phản Biện Hội Nghị Độc Lập
        # -------------------------------------------------------------
        self._emit_event(run_id or run_name, 7, "running", "Running independent peer review", 0.93, run_dir=run_dir)
        print("\n>>> GIAI ĐOẠN 7: PHẢN BIỆN HỌC THUẬT ĐỘC LẬP (Conference Double-Blind Peer Review)")
        paper_title = generated_title or context.get("method_name", "Autonomous AI Discovery")
        tex_path = os.path.join(run_dir, "latex", "paper.tex")
        review_data = self.peer_reviewer.review_paper(
            output_dir=run_dir,
            paper_title=paper_title,
            paper_tex_path=tex_path,
            metrics_data=full_metrics,
        )

        print("\n" + "=" * 70)
        print("  CÔNG TRÌNH NGHIÊN CỨU ĐÃ HOÀN THÀNH (TUÂN THỦ HỢP ĐỒNG ĐẦU RA)!")
        print("=" * 70)
        print(f"1. Thư mục thực nghiệm: {exp_work_dir}")
        print(f"2. Biểu đồ trực quan thực tế: {len(figure_manifest)} ảnh ({[item.filename for item in figure_manifest]})")
        print(f"3. Sổ cái tuyên bố số liệu: {ledger_path}")
        print(f"4. Mã nguồn LaTeX: {os.path.join(run_dir, 'latex', 'paper.tex')}")
        if pdf_path and os.path.exists(pdf_path):
            print(f"5. BÀI BÁO HOÀN CHỈNH (PDF): {pdf_path}")
        print(f"6. Báo cáo phản biện: {os.path.join(run_dir, 'review', 'peer_review.md')}")
        print(f"   -> Kết luận: {review_data.get('recommendation')} ({review_data.get('scores', {}).get('overall_rating')}/10)")

        result = {
            "run_dir": run_dir,
            "pdf_path": pdf_path,
            "review": review_data,
            "metrics": metrics_data,
            "claim_ledger": ledger_path,
            "environment": env_path,
        }
        self._emit_event(run_id or run_name, 7, "completed", "Research run completed", 1.0, run_dir=run_dir, result=result)

        return result

