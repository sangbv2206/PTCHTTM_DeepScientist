import hashlib
import json
import os
import re
import socket
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import requests

# Ensure IPv4-first DNS resolution on Windows/corporate networks
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
    try:
        res = _orig_getaddrinfo(host, port, family, type, proto, flags)
        ipv4 = [r for r in res if r[0] == socket.AF_INET]
        return ipv4 if ipv4 else res
    except Exception:
        return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_first


HF_API_URL = "https://huggingface.co/api/datasets"
HF_SERVER_SPLITS = "https://datasets-server.huggingface.co/splits"
HF_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"

# Built-in curated Hugging Face dataset fallbacks by domain keyword (used if LLM suggestions miss)
DOMAIN_HF_FALLBACKS: List[Tuple[Tuple[str, ...], List[str]]] = [
    (("rag", "triplet", "multi-hop", "multihop", "hotpot", "retrieval", "qa", "question answering"),
     ["hotpotqa/hotpot_qa", "hotpot_qa", "rajpurkar/squad", "framolfese/2WikiMultihopQA"]),
    (("sentiment", "classification", "nlp", "text", "review", "nli", "entailment"),
     ["stanfordnlp/sst2", "nyu-mll/glue", "fancyzhx/ag_news", "imdb"]),
    (("math", "reasoning", "gsm8k", "arithmetic"),
     ["openai/gsm8k", "gsm8k"]),
    (("summarization", "summary", "abstract"),
     ["abisee/cnn_dailymail", "EdinburghNLP/xsum"]),
    (("code", "programming", "python", "test case", "testcase", "contest", "competitive"),
     ["deepmind/code_contests", "evalplus/humanevalplus", "openai/openai_humaneval", "mbpp"]),
    (("medical", "biomedical", "clinical", "pubmed"),
     ["qiaojin/PubMedQA", "bigbio/med_qa"]),
]

SKLEARN_LOADERS = {
    "breast_cancer": ("sklearn.datasets.load_breast_cancer", "Breast Cancer Wisconsin Diagnostic Dataset (UCI/sklearn, 569 samples, 30 features)"),
    "wine": ("sklearn.datasets.load_wine", "Wine Recognition Dataset (UCI/sklearn, 178 samples, 13 features)"),
    "digits": ("sklearn.datasets.load_digits", "Optical Recognition of Handwritten Digits (UCI/sklearn, 1797 samples, 64 features)"),
    "iris": ("sklearn.datasets.load_iris", "Iris Plants Dataset (UCI/sklearn, 150 samples, 4 features)"),
    "diabetes": ("sklearn.datasets.load_diabetes", "Diabetes Regression Dataset (sklearn, 442 samples, 10 features)"),
    "california_housing": ("sklearn.datasets.fetch_california_housing", "California Housing Regression Dataset (Pace & Barry / sklearn, 20640 samples)"),
}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    raw = match.group(1) if match else ""
    if not raw:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            raw = text[start:end + 1]
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _truncate_row(row: Any, max_str: int = 320, max_list: int = 10) -> Any:
    if isinstance(row, str):
        return row if len(row) <= max_str else row[:max_str] + "..."
    if isinstance(row, list):
        trimmed = [_truncate_row(x, max_str=max_str, max_list=max_list) for x in row[:max_list]]
        return trimmed
    if isinstance(row, dict):
        return {k: _truncate_row(v, max_str=max_str, max_list=max_list) for k, v in list(row.items())[:12]}
    return row


def _compact_row_for_storage(row: Dict[str, Any], max_field_chars: int = 12000) -> Dict[str, Any]:
    """Keep real benchmark fields intact while bounding extraordinarily huge passages."""
    cleaned: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, str) and len(v) > max_field_chars:
            cleaned[k] = v[:max_field_chars]
        elif isinstance(v, bytes):
            continue
        else:
            cleaned[k] = v
    return cleaned


class DataAgent:
    """
    Autonomous Public Dataset Acquisition Agent.
    Resolves a research topic to a verifiable, real public benchmark subset (Hugging Face Datasets Server
    or scikit-learn built-ins) and records provenance in `data/data_manifest.json`.
    """

    def __init__(self, llm: Optional[Any] = None, timeout: float = 12.0):
        self.llm = llm
        self.timeout = timeout

    def _plan_candidates(
        self,
        topic: str,
        problem_info: Optional[Dict[str, Any]] = None,
        seed_digest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        topic_lower = (topic or "").lower()
        fallback_hf: List[str] = []
        for kws, ds_list in DOMAIN_HF_FALLBACKS:
            if any(kw in topic_lower for kw in kws):
                fallback_hf.extend(ds_list)

        plan = {
            "task_nature": "empirical_real",
            "task_family": "general",
            "hf_candidates": fallback_hf or ["rajpurkar/squad"],
            "hf_search_queries": [topic[:80]],
            "sklearn_candidate": "",
        }
        if not self.llm:
            return plan

        seed_ds = (seed_digest or {}).get("datasets", [])
        prob = (problem_info or {}).get("problem", "")
        prompt = f"""You are a scientific data acquisition planner.
Identify the best real, publicly available benchmark dataset for this research topic.

Topic: "{topic}"
Problem: "{prob}"
Seed Paper Datasets (if any): {json.dumps(seed_ds[:6])}

Respond ONLY in JSON:
```json
{{
  "task_nature": "empirical_real or simulation (use 'simulation' ONLY if the topic is inherently synthetic control, abstract game theory, or theoretical algorithm stress-testing)",
  "task_family": "qa_rag | text_classification | tabular_classification | tabular_regression | time_series | graph | vision | math_reasoning | simulation",
  "hf_candidates": ["owner/dataset_id_1", "owner/dataset_id_2", "owner/dataset_id_3"],
  "hf_search_queries": ["short keyword query 1", "short keyword query 2"],
  "sklearn_candidate": "optional one of: breast_cancer, wine, digits, iris, diabetes, california_housing, or empty string"
}}
```
Rules:
- Prefer widely cited, lightweight Hugging Face evaluation benchmarks (e.g., hotpotqa/hotpot_qa, rajpurkar/squad, stanfordnlp/sst2, deepmind/code_contests, openai/gsm8k, qiaojin/PubMedQA).
- AVOID `*-instruct` or `*-sft` prompt-only datasets unless the topic is specifically instruction fine-tuning; prefer benchmarks that include ground-truth labels, answers, or test cases/incorrect solutions for genuine evaluation.
- For QA/RAG topics, put hotpotqa/hotpot_qa or rajpurkar/squad first. For competitive programming / test case generation, put deepmind/code_contests first.
"""
        try:
            parsed = _extract_json(self.llm.generate(prompt))
            if parsed:
                hf_cands = [c for c in parsed.get("hf_candidates", []) if isinstance(c, str) and c.strip()]
                # Merge curated domain fallbacks after LLM choices
                for fb in fallback_hf:
                    if fb not in hf_cands:
                        hf_cands.append(fb)
                plan["task_nature"] = parsed.get("task_nature", plan["task_nature"])
                plan["task_family"] = parsed.get("task_family", plan["task_family"])
                plan["hf_candidates"] = hf_cands[:8]
                plan["hf_search_queries"] = [q for q in parsed.get("hf_search_queries", []) if isinstance(q, str) and q.strip()][:3] or plan["hf_search_queries"]
                plan["sklearn_candidate"] = (parsed.get("sklearn_candidate") or "").strip()
        except Exception as e:
            print(f"[DataAgent] Warning during candidate planning ({e}); using heuristic candidates.")
        return plan

    def _search_hf_hub(self, query: str, limit: int = 6) -> List[str]:
        try:
            resp = requests.get(
                HF_API_URL,
                params={"search": query, "sort": "downloads", "direction": "-1", "limit": limit},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                items = resp.json()
                return [it["id"] for it in items if isinstance(it, dict) and it.get("id")]
        except Exception:
            pass
        return []

    def _pick_split(self, dataset_id: str) -> Optional[Tuple[str, str]]:
        try:
            resp = requests.get(HF_SERVER_SPLITS, params={"dataset": dataset_id}, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            splits = resp.json().get("splits", [])
            if not splits:
                return None
            # Prefer validation / test / dev splits for evaluation, else train; prefer smaller/standard configs
            preferred_splits = ["validation", "dev", "test", "train"]
            for sp_pref in preferred_splits:
                for item in splits:
                    if item.get("split") == sp_pref:
                        cfg = item.get("config", "default")
                        if cfg in ("distractor", "default", "plain_text", "main"):
                            return cfg, sp_pref
            for sp_pref in preferred_splits:
                for item in splits:
                    if item.get("split") == sp_pref:
                        return item.get("config", "default"), sp_pref
            return splits[0].get("config", "default"), splits[0].get("split", "train")
        except Exception:
            return None

    def _fetch_hf_rows(self, dataset_id: str, config: str, split: str, max_rows: int = 120) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        offset = 0
        page_size = min(25, max_rows)
        req_timeout = max(25.0, self.timeout)
        while len(rows) < max_rows:
            batch = min(page_size, max_rows - len(rows))
            try:
                resp = requests.get(
                    HF_SERVER_ROWS,
                    params={"dataset": dataset_id, "config": config, "split": split, "offset": offset, "length": batch},
                    timeout=req_timeout,
                )
            except Exception as e:
                print(f"[DataAgent] HF rows fetch error for '{dataset_id}' (offset={offset}): {e}")
                break
            if resp.status_code != 200:
                break
            data = resp.json()
            fetched = data.get("rows", [])
            if not fetched:
                break
            for r in fetched:
                raw_row = r.get("row") if isinstance(r, dict) else None
                if isinstance(raw_row, dict):
                    rows.append(_compact_row_for_storage(raw_row))
            offset += len(fetched)
            if len(fetched) < batch:
                break
        return rows

    def _verify_dataset_schema(self, topic: str, dataset_id: str, rows: List[Dict[str, Any]]) -> bool:
        """Generic schema check: verify the dataset has both inputs and ground-truth targets/tests for the topic."""
        if not rows:
            return False
        cols = list(rows[0].keys())
        # Quick generic heuristic: avoid instruction-only datasets when not doing SFT
        if "instruct" in dataset_id.lower() and "instruct" not in topic.lower() and "fine-tun" not in topic.lower():
            if not any(c in cols for c in ("incorrect_solutions", "public_tests", "test", "tests", "answer", "answers", "label", "labels")):
                print(f"[DataAgent] Skipping '{dataset_id}' (instruction-only schema without evaluation labels/tests: {cols}).")
                return False
        if not self.llm:
            return True
        preview = _truncate_row(rows[0], max_str=180, max_list=3)
        prompt = f"""Check if this dataset is suitable for authentic empirical evaluation of the topic below.
Topic: "{topic}"
Dataset ID: "{dataset_id}"
Columns: {json.dumps(cols)}
Sample Row Preview: {json.dumps(preview, ensure_ascii=False)[:900]}

Does this dataset contain BOTH the task inputs AND the ground-truth labels / answers / reference tests needed to evaluate "{topic}" WITHOUT fabricating missing ground-truth?
Respond ONLY in JSON: {{"suitable": true, "reason": "brief reason"}}"""
        try:
            parsed = _extract_json(self.llm.generate(prompt))
            if parsed is not None and parsed.get("suitable") is False:
                print(f"[DataAgent] Schema check rejected '{dataset_id}': {parsed.get('reason', 'missing evaluation columns')}")
                return False
        except Exception:
            pass
        return True

    def _try_sklearn(self, name: str, data_dir: str, max_rows: int = 500) -> Optional[Dict[str, Any]]:
        if name not in SKLEARN_LOADERS:
            return None
        try:
            from sklearn import datasets as sk_ds
            fn_map = {
                "breast_cancer": sk_ds.load_breast_cancer,
                "wine": sk_ds.load_wine,
                "digits": sk_ds.load_digits,
                "iris": sk_ds.load_iris,
                "diabetes": sk_ds.load_diabetes,
                "california_housing": sk_ds.fetch_california_housing,
            }
            bunch = fn_map[name](as_frame=True)
            df = bunch.frame.head(max_rows)
            out_path = os.path.join(data_dir, "dataset.jsonl")
            records = df.to_dict(orient="records")
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            loader_path, desc = SKLEARN_LOADERS[name]
            return {
                "data_source": f"real:sklearn:{name}",
                "dataset_id": name,
                "dataset_name": desc,
                "provider": "scikit-learn",
                "url": "https://scikit-learn.org/stable/datasets.html",
                "split": "full_subset",
                "num_samples": len(records),
                "local_path": "data/dataset.jsonl",
                "sha256": _sha256_file(out_path),
                "columns": list(df.columns),
                "sample_preview": [_truncate_row(r) for r in records[:2]],
                "requires_disclosure": False,
            }
        except Exception as e:
            print(f"[DataAgent] sklearn fallback '{name}' failed: {e}")
            return None

    def acquire(
        self,
        topic: str,
        exp_work_dir: str,
        problem_info: Optional[Dict[str, Any]] = None,
        seed_digest: Optional[Dict[str, Any]] = None,
        online: bool = True,
        max_rows: int = 120,
    ) -> Dict[str, Any]:
        """
        Acquire a real public dataset into `<exp_work_dir>/data/` and write `data_manifest.json`.
        Falls back cleanly to explicit simulation/synthetic disclosure only if no real dataset is obtainable.
        """
        data_dir = os.path.join(exp_work_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        manifest_path = os.path.join(data_dir, "data_manifest.json")

        # 0. Reuse if user already placed a dataset file in <exp_work_dir>/data/
        for fname in ("dataset.jsonl", "dataset.json", "dataset.csv"):
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath) and os.path.getsize(fpath) > 50:
                manifest = {
                    "data_source": f"real:local_file:{fname}",
                    "dataset_id": fname,
                    "dataset_name": f"User-provided dataset ({fname})",
                    "provider": "local",
                    "url": "",
                    "split": "provided",
                    "num_samples": None,
                    "local_path": f"data/{fname}",
                    "sha256": _sha256_file(fpath),
                    "columns": [],
                    "sample_preview": [],
                    "requires_disclosure": False,
                }
                with open(manifest_path, "w", encoding="utf-8") as mf:
                    json.dump(manifest, mf, indent=2, ensure_ascii=False)
                return manifest

        plan = self._plan_candidates(topic, problem_info=problem_info, seed_digest=seed_digest)
        task_nature = plan.get("task_nature", "empirical_real")
        task_family = plan.get("task_family", "general")

        # 1. Try Hugging Face Datasets Server when online and task is not purely abstract simulation
        if online and task_nature != "simulation":
            candidates = list(plan.get("hf_candidates", []))
            for q in plan.get("hf_search_queries", []):
                for hit in self._search_hf_hub(q, limit=5):
                    if hit not in candidates:
                        candidates.append(hit)

            for ds_id in candidates:
                split_info = self._pick_split(ds_id)
                if not split_info:
                    continue
                cfg, sp = split_info
                rows = self._fetch_hf_rows(ds_id, cfg, sp, max_rows=max_rows)
                if len(rows) >= 20 and self._verify_dataset_schema(topic, ds_id, rows):
                    out_path = os.path.join(data_dir, "dataset.jsonl")
                    with open(out_path, "w", encoding="utf-8") as f:
                        for row in rows:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    cols = list(rows[0].keys()) if rows else []
                    manifest = {
                        "data_source": f"real:huggingface:{ds_id}",
                        "dataset_id": ds_id,
                        "dataset_name": f"{ds_id} (config={cfg}, split={sp}, N={len(rows)} real benchmark samples)",
                        "provider": "Hugging Face Datasets Server",
                        "url": f"https://huggingface.co/datasets/{ds_id}",
                        "config": cfg,
                        "split": sp,
                        "num_samples": len(rows),
                        "local_path": "data/dataset.jsonl",
                        "sha256": _sha256_file(out_path),
                        "columns": cols,
                        "sample_preview": [_truncate_row(r) for r in rows[:2]],
                        "task_family": task_family,
                        "requires_disclosure": False,
                    }
                    with open(manifest_path, "w", encoding="utf-8") as mf:
                        json.dump(manifest, mf, indent=2, ensure_ascii=False)
                    print(f"[DataAgent] Acquired REAL dataset '{ds_id}' ({cfg}/{sp}, {len(rows)} rows) -> {out_path}")
                    return manifest

        # 2. Try scikit-learn real benchmark if applicable to tabular/regression/digits
        sk_cand = plan.get("sklearn_candidate", "")
        if sk_cand and task_family in ("tabular_classification", "tabular_regression", "vision", "general"):
            sk_manifest = self._try_sklearn(sk_cand, data_dir)
            if sk_manifest:
                sk_manifest["task_family"] = task_family
                with open(manifest_path, "w", encoding="utf-8") as mf:
                    json.dump(sk_manifest, mf, indent=2, ensure_ascii=False)
                print(f"[DataAgent] Acquired REAL sklearn dataset '{sk_cand}' ({sk_manifest['num_samples']} rows)")
                return sk_manifest

        # 3. Controlled Simulation / Synthetic Disclosure Fallback
        is_sim = task_nature == "simulation"
        manifest = {
            "data_source": "synthetic_simulation" if is_sim else "synthetic_fallback",
            "dataset_id": "synthetic_benchmark",
            "dataset_name": (
                f"Controlled synthetic simulation environment for {topic}"
                if is_sim
                else f"Offline synthetic proxy benchmark for {topic} (public dataset fetch unavailable)"
            ),
            "provider": "generated_in_experiment",
            "url": "",
            "split": "synthetic",
            "num_samples": 0,
            "local_path": "",
            "sha256": "",
            "columns": [],
            "sample_preview": [],
            "task_family": task_family,
            "requires_disclosure": True,
            "disclosure_note": (
                "Experiments were conducted in a controlled synthetic simulation matching the task's analytical assumptions."
                if is_sim
                else "Public benchmark download was unavailable in the execution environment; results reflect a controlled synthetic proxy and must not be interpreted as full-scale real-world benchmark scores."
            ),
        }
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2, ensure_ascii=False)
        print(f"[DataAgent] Fallback manifest recorded ({manifest['data_source']}).")
        return manifest

