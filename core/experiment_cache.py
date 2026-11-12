"""
Experiment Cache (bộ nhớ thực nghiệm tự mở rộng).

Mỗi khi nhánh Zero-Shot sinh thành công một kịch bản thực nghiệm cho đề tài chưa có template,
kịch bản đó được lưu vào templates/_synthesized/<slug>/ kèm meta.json.
Lần sau gặp đề tài cùng "họ bài toán" (problem_family) hoặc trùng từ khóa, kịch bản cũ được đưa cho
CoderAgent làm mã tham chiếu để chỉnh sửa, thay vì viết lại từ đầu.

Thư mục _synthesized không chứa experiment.py ở cấp gốc nên không lẫn vào kho template chuẩn.
"""
import datetime
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional

_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "in", "on", "with", "to", "via", "through", "by",
    "using", "based", "towards", "toward", "from", "beyond", "into", "is", "are", "novel", "new",
    "approach", "method", "framework", "study", "analysis",
}


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:max_len] or "experiment"


class ExperimentCache:
    def __init__(self, root_dir: Optional[str] = None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.root_dir = root_dir or os.path.join(base, "templates", "_synthesized")
        os.makedirs(self.root_dir, exist_ok=True)

    def list_entries(self) -> List[Dict[str, Any]]:
        entries = []
        for name in sorted(os.listdir(self.root_dir)):
            meta_path = os.path.join(self.root_dir, name, "meta.json")
            code_path = os.path.join(self.root_dir, name, "run_experiment.py")
            if os.path.exists(meta_path) and os.path.exists(code_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["_dir"] = os.path.join(self.root_dir, name)
                    entries.append(meta)
                except Exception:
                    pass
        return entries

    def find_best_match(self, topic: str, problem_family: str = "", min_score: float = 0.25) -> Optional[Dict[str, Any]]:
        """Tìm kịch bản đã lưu gần nhất theo problem_family (ưu tiên) và độ trùng từ khóa (Jaccard)."""
        query_tokens = _tokens(topic) | _tokens(problem_family)
        if not query_tokens:
            return None
        best, best_score = None, 0.0
        for meta in self.list_entries():
            cand_tokens = _tokens(meta.get("topic", "")) | _tokens(meta.get("problem_family", "")) | set(meta.get("keywords", []))
            if not cand_tokens:
                continue
            score = len(query_tokens & cand_tokens) / len(query_tokens | cand_tokens)
            if problem_family and meta.get("problem_family", "").strip().lower() == problem_family.strip().lower():
                score += 0.5
            if score > best_score:
                best, best_score = meta, score
        if best and best_score >= min_score:
            best["_match_score"] = round(best_score, 3)
            return best
        return None

    def load_code(self, entry: Dict[str, Any]) -> str:
        with open(os.path.join(entry["_dir"], "run_experiment.py"), "r", encoding="utf-8") as f:
            return f.read()

    def save(self, work_dir: str, topic: str, problem_family: str, method_name: str, metrics: List[str]) -> str:
        """Lưu kịch bản thành công vào kho. Trả về đường dẫn thư mục đã lưu."""
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _slugify(problem_family if problem_family and problem_family != "not specified" else topic)
        target = os.path.join(self.root_dir, f"{slug}_{stamp}")
        os.makedirs(target, exist_ok=True)
        for fname in ["run_experiment.py", "experiment_config.json"]:
            src = os.path.join(work_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(target, fname))
        meta = {
            "topic": topic,
            "problem_family": problem_family,
            "method_name": method_name,
            "metrics": metrics,
            "keywords": sorted(_tokens(topic) | _tokens(problem_family)),
            "created_at": stamp,
        }
        with open(os.path.join(target, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return target
