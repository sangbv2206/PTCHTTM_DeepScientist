"""
Paper Reader: đọc một bài báo PDF (seed paper) và trích xuất "bản tóm lược kỹ thuật" có cấu trúc
(phương pháp, thuật toán từng bước, dataset, baseline, metric, siêu tham số, kết quả chính).

Bản tóm lược này được đưa vào context của pipeline để các Agent (ProblemIdentifier, MethodDeveloper,
ExperimentDesigner, CoderAgent) bám sát đúng nội dung bài báo thay vì chỉ dựa vào tiêu đề.
"""
import json
import os
import re
from typing import Any, Dict, Optional


def extract_pdf_text(pdf_path: str, max_chars: int = 60000) -> str:
    """Trích xuất text thô từ PDF (dùng pypdf). Cắt bớt phần References/Appendix nếu quá dài."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)

    # Ưu tiên phần thân bài: cắt trước mục References nếu tìm thấy ở nửa sau tài liệu
    m = re.search(r"\n\s*References\s*\n", text)
    if m and m.start() > len(text) * 0.3:
        body = text[: m.start()]
        appendix = text[m.end():]
        # Giữ lại một phần appendix (thường chứa prompt/hyperparameter chi tiết)
        text = body + "\n\n[APPENDIX EXCERPT]\n" + appendix[: max(0, max_chars - len(body))]
    return text[:max_chars]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(raw)
    except Exception:
        return None


def build_seed_digest(pdf_path: str, llm: Any) -> Dict[str, Any]:
    """
    Đọc PDF và dùng LLM trích xuất bản tóm lược kỹ thuật dạng JSON.
    Trả về dict gồm: title, abstract, problem_family, core_idea, algorithm_steps, datasets,
    baselines, metrics, hyperparameters, key_results, figures, raw_excerpt.
    """
    text = extract_pdf_text(pdf_path)
    first_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    fallback_title = " ".join(first_lines[:2])[:200] if first_lines else os.path.basename(pdf_path)

    prompt = f"""You are a senior ML researcher. Read the following research paper text and extract a precise,
faithful technical digest that would allow an engineer to re-implement its core method and experiments.
Do NOT invent anything that is not in the text. If something is missing, write "not specified".

PAPER TEXT:
\"\"\"
{text}
\"\"\"

Respond strictly in JSON:
```json
{{
  "title": "exact paper title",
  "abstract": "the abstract (verbatim or near-verbatim)",
  "problem_family": "one short label for the experimental task family, e.g. 'retrieval-augmented question answering', 'time-series forecasting', 'image classification', 'graph node classification', 'causal effect estimation'",
  "core_idea": "2-4 sentences describing the key mechanism",
  "algorithm_steps": ["Step 1 ...", "Step 2 ...", "..."],
  "datasets": ["dataset name - short description"],
  "baselines": ["baseline name - one line description"],
  "metrics": ["metric name"],
  "hyperparameters": ["name = value"],
  "key_results": ["quantitative claim with numbers as reported"],
  "figures_and_tables": ["Table/Figure id - what it shows"]
}}
```"""
    digest: Dict[str, Any] = {}
    try:
        res = llm.generate(prompt)
        digest = _extract_json(res) or {}
    except Exception as e:
        print(f"[PaperReader] LLM digest failed: {e}")

    digest.setdefault("title", fallback_title)
    digest.setdefault("abstract", text[:1500])
    digest.setdefault("problem_family", "not specified")
    digest.setdefault("core_idea", "")
    for k in ["algorithm_steps", "datasets", "baselines", "metrics", "hyperparameters", "key_results", "figures_and_tables"]:
        if not isinstance(digest.get(k), list):
            digest[k] = []
    digest["source_pdf"] = os.path.abspath(pdf_path)
    digest["raw_excerpt"] = text[:12000]
    return digest


def render_seed_digest(digest: Optional[Dict[str, Any]], include_results: bool = True) -> str:
    """Định dạng digest thành khối văn bản để chèn vào prompt."""
    if not digest:
        return ""

    def bullets(items):
        return "\n".join(f"  - {x}" for x in items) if items else "  - not specified"

    parts = [
        f"Title: {digest.get('title')}",
        f"Problem family: {digest.get('problem_family')}",
        f"Core idea: {digest.get('core_idea')}",
        "Algorithm steps:\n" + bullets(digest.get("algorithm_steps", [])),
        "Datasets:\n" + bullets(digest.get("datasets", [])),
        "Baselines:\n" + bullets(digest.get("baselines", [])),
        "Metrics:\n" + bullets(digest.get("metrics", [])),
        "Hyperparameters:\n" + bullets(digest.get("hyperparameters", [])),
    ]
    if include_results:
        parts.append("Reported results:\n" + bullets(digest.get("key_results", [])))
    return "\n".join(parts)
