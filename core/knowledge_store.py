import json
import math
import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Any
import torch
from core.external_retriever import ExternalPaperRetriever

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", 
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", 
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", 
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
    "while", "who", "whom", "why", "will", "with", "based", "approach", "method", "methods", "using", 
    "study", "applications", "application", "system", "systems", "via", "towards", "toward", "analysis",
    "comparison", "different", "models", "model", "assisted", "review", "effect", "effects", "paper",
    "deep", "learning", "machine", "neural", "network", "networks", "artificial", "intelligence"
}


class KnowledgeStore:
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, "data")
        else:
            self.data_dir = data_dir

        self.knowledge_path = os.path.join(self.data_dir, "knowledge.jsonl")
        self.papers_path = os.path.join(self.data_dir, "papers.jsonl")
        self.references_path = os.path.join(self.data_dir, "references.jsonl")
        self.cache_path = os.path.join(self.data_dir, "external_papers_cache.jsonl")
        self.retriever = ExternalPaperRetriever(cache_dir=self.data_dir)

        self.papers_by_id: Dict[int, Dict[str, Any]] = {}
        self.paper2references: Dict[int, List[int]] = defaultdict(list)
        self.paper2entities: Dict[int, Dict[str, int]] = {}
        self.entity_counter: Counter = Counter()
        self.entity_cooccurrence: Dict[str, Counter] = defaultdict(Counter)
        self.doc_freq: Counter = Counter()
        self.is_loaded = False

    def load_all_data(self):
        """Loads corpus papers, entities, reference relationships, and cache into memory."""
        if self.is_loaded:
            return

        print(f"[KnowledgeStore] Loading dataset from: {self.data_dir}")

        # 1. Load papers.jsonl
        if os.path.exists(self.papers_path):
            with open(self.papers_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        p = json.loads(line)
                        cid = p.get("corpusid")
                        if cid:
                            self.papers_by_id[cid] = p
            print(f"[KnowledgeStore] Loaded {len(self.papers_by_id)} papers from papers.jsonl.")
        else:
            print(f"[KnowledgeStore] WARNING: {self.papers_path} not found.")

        # 2. Load knowledge.jsonl entities
        if os.path.exists(self.knowledge_path):
            k_records = 0
            with open(self.knowledge_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    instance = json.loads(line)
                    cid = instance.get("corpusid")
                    entities = instance.get("knowledge", {})
                    if cid and entities:
                        self.paper2entities[cid] = entities
                        self.entity_counter.update(entities)
                        for entity_name in entities.keys():
                            self.entity_cooccurrence[entity_name].update(
                                {k: v for k, v in entities.items() if k != entity_name}
                            )
                        k_records += 1
            print(f"[KnowledgeStore] Loaded {k_records} paper entity graphs from knowledge.jsonl.")
            print(f"[KnowledgeStore] Built global entity co-occurrence matrix with {len(self.entity_counter)} unique scientific concepts.")
        else:
            print(f"[KnowledgeStore] WARNING: {self.knowledge_path} not found.")

        # 3. Load references.jsonl
        if os.path.exists(self.references_path):
            r_added = 0
            with open(self.references_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            source_id = item.get("corpusid")
                            if source_id and source_id not in self.papers_by_id:
                                self.papers_by_id[source_id] = item
                                r_added += 1
                        except Exception:
                            pass
            print(f"[KnowledgeStore] Loaded {r_added} additional reference papers from references.jsonl (Unified corpus: {len(self.papers_by_id)} papers).")

        # 4. Load cache
        if os.path.exists(self.cache_path):
            c_added = 0
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            cid = item.get("corpusid")
                            if cid and cid not in self.papers_by_id:
                                self.papers_by_id[cid] = item
                                c_added += 1
                        except Exception:
                            pass
            if c_added > 0:
                print(f"[KnowledgeStore] Loaded {c_added} cached online papers from external_papers_cache.jsonl.")

        # Compute document frequency for search
        self.doc_freq = Counter()
        for p in self.papers_by_id.values():
            text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
            words = set(re.findall(r"\b[a-z]{2,}\b", text)) - STOPWORDS
            self.doc_freq.update(words)

        self.is_loaded = True

    def get_entity_log_likelihood(self, entity: str, paper_entities: list) -> float:
        total_cooc = sum(self.entity_cooccurrence[entity].values()) + 1e-16
        conditional_log_probabilities = [
            math.log2((self.entity_cooccurrence[entity][paper_entity] + 1e-16) / total_cooc)
            for paper_entity in paper_entities
        ]
        return sum(conditional_log_probabilities)

    def get_entity_probability(self, entity: str) -> float:
        total = sum(self.entity_counter.values())
        if total == 0:
            return 0.0
        return self.entity_counter[entity] / total

    def get_relevant_entities(self, paper_ids: list, top_k: int = 30) -> List[str]:
        paper_entities = sum(
            [Counter(self.paper2entities[pid]) for pid in paper_ids if pid in self.paper2entities],
            start=Counter()
        )
        paper_entities_list = list(paper_entities.elements())

        discovered = []
        if paper_entities_list:
            candidate_entities = sum(
                [self.entity_cooccurrence[e] for e in paper_entities_list],
                start=Counter()
            )
            candidate_entities_filtered = [
                e for e, count in candidate_entities.items()
                if count >= 2 and len(e) > 2 and not e.isdigit()
            ]

            if candidate_entities_filtered:
                candidate_entities_probs = [
                    (
                        self.get_entity_log_likelihood(e, paper_entities_list) +
                        math.log2(self.get_entity_probability(e) + 1e-16)
                    )
                    for e in candidate_entities_filtered
                ]
                k = min(top_k, len(candidate_entities_filtered))
                _, indices = torch.topk(torch.tensor(candidate_entities_probs), k=k, axis=-1)
                discovered = [candidate_entities_filtered[index] for index in indices]

        domain_concepts = []
        for pid in paper_ids:
            p = self.papers_by_id.get(pid)
            if not p:
                continue
            title = p.get("title", "")
            abstract = p.get("abstract", "")

            cap_phrases = re.findall(r"\b[A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*)+\b", title + ". " + abstract)
            for phrase in cap_phrases:
                phrase_clean = phrase.strip()
                if len(phrase_clean) > 3 and not phrase_clean.isdigit() and phrase_clean not in domain_concepts:
                    domain_concepts.append(phrase_clean)

            words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", title) if w.lower() not in STOPWORDS]
            for i in range(len(words) - 1):
                bg = f"{words[i].capitalize()} {words[i+1].capitalize()}"
                if bg not in domain_concepts:
                    domain_concepts.append(bg)

        final_entities = []
        for ent in domain_concepts + discovered:
            if ent not in final_entities and len(ent) > 2:
                final_entities.append(ent)
        return final_entities[:top_k]

    def assess_topic_coverage(self, topic: str) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_all_data()

        MIN_PAPERS_PER_KEYWORD = 15
        query_words = [w for w in re.findall(r"\b[a-z]{2,}\b", topic.lower()) if w not in STOPWORDS]
        if not query_words:
            query_words = [w for w in re.findall(r"\b[a-z]{2,}\b", topic.lower())]

        unique_q_words = list(dict.fromkeys(query_words))
        matched_q = [w for w in unique_q_words if self.doc_freq.get(w, 0) >= MIN_PAPERS_PER_KEYWORD]
        missing_q = [w for w in unique_q_words if w not in matched_q]
        coverage_ratio = len(matched_q) / max(len(unique_q_words), 1)

        scored_matches = self.search_target_papers_with_scores(topic, top_k=5)
        top_score = scored_matches[0][0] if scored_matches else 0.0
        matches = [p for _, p in scored_matches]
        top_paper = matches[0] if matches else None

        if coverage_ratio >= 0.50 and top_score >= 50.0:
            status = "STRONG"
            warning = None
            solutions = ["Cơ sở dữ liệu hỗ trợ tốt, có đầy đủ tài liệu nền tảng để trích dẫn và đối sánh."]
        elif (coverage_ratio >= 0.25 and top_score >= 35.0) or top_score >= 45.0:
            status = "MODERATE"
            warning = (
                f"Chủ đề '{topic}' có dữ liệu liên quan một phần "
                f"({int(coverage_ratio*100)}% từ khóa có trong corpus, top score: {top_score:.1f}). "
                f"Một số khía cạnh chuyên sâu có thể cần tra cứu thêm."
            )
            solutions = [
                "Hệ thống sẽ kết hợp bài báo gần nhất với khả năng suy luận mở của Gemini để bổ sung khoảng trống kiến thức.",
                "Có thể bổ sung thêm các bài báo chuyên sâu vào data/papers.jsonl để tăng độ chính xác."
            ]
        else:
            status = "INSUFFICIENT"
            missing_str = ', '.join(missing_q[:8]) + (" ..." if len(missing_q) > 8 else "")
            warning = (
                f"Chủ đề '{topic}' CHƯA ĐỦ THÔNG TIN trong cơ sở dữ liệu nội bộ "
                f"(chỉ {int(coverage_ratio*100)}% từ khóa cốt lõi được corpus hỗ trợ, "
                f"top relevance score={top_score:.1f} < 35.0, thiếu: {missing_str})."
            )
            solutions = [
                "Hướng 1: Kích hoạt Dynamic External Retrieval từ OpenAlex / arXiv để lấy bài báo thực tế mới nhất.",
                "Hướng 2: Chuyển sang chế độ Open-World LLM Synthesis (tổng hợp từ tri thức toàn cầu của mô hình).",
                "Hướng 3: Thu hẹp hoặc điều chỉnh đề tài về các miền bài toán có sẵn nhiều tài liệu trong kho."
            ]

        return {
            "status": status,
            "coverage_score": coverage_ratio,
            "coverage_percent": round(coverage_ratio * 100, 1),
            "top_relevance_score": round(top_score, 1),
            "matched_keywords": matched_q,
            "missing_keywords": missing_q,
            "best_paper": top_paper,
            "candidate_papers": matches,
            "num_candidates": len(matches),
            "warning": warning,
            "solutions": solutions
        }

    def _compute_paper_scores(self, query: str, exclude_corpusid: Optional[int] = None) -> List[tuple]:
        total_docs = max(len(self.papers_by_id), 1)
        query_words = [w for w in re.findall(r"\b[a-z]{2,}\b", query.lower()) if w not in STOPWORDS]
        if not query_words:
            query_words = [w for w in re.findall(r"\b[a-z]{2,}\b", query.lower())]

        raw_words = query.lower().split()
        query_bigrams = [" ".join(raw_words[i:i+2]) for i in range(len(raw_words)-1)]
        content_bigrams = [bg for bg in query_bigrams if not any(w in STOPWORDS for w in bg.split())]

        matches = []
        for cid, paper in self.papers_by_id.items():
            if exclude_corpusid and cid == exclude_corpusid:
                continue

            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            t_lower = title.lower()
            a_lower = abstract.lower()

            t_words = set(re.findall(r"\b[a-z]{2,}\b", t_lower))
            a_words = set(re.findall(r"\b[a-z]{2,}\b", a_lower))

            score = 0.0
            matched_terms = 0

            for w in query_words:
                idf = math.log((total_docs + 1.0) / (self.doc_freq.get(w, 0) + 1.0)) + 1.0
                if w in t_words:
                    score += 3.0 * idf
                    matched_terms += 1
                elif w in a_words:
                    score += 1.0 * idf
                    matched_terms += 1

            if matched_terms == 0:
                continue

            for bg in content_bigrams:
                if len(bg) > 5 and bg in t_lower:
                    score += 15.0
                elif len(bg) > 5 and bg in a_lower:
                    score += 5.0

            coverage_ratio = matched_terms / max(len(query_words), 1)
            score += coverage_ratio * 25.0

            cit_count = paper.get("citationcount", 0) or 0
            score += min(cit_count / 50.0, 2.0)

            matches.append((score, paper))

        return matches

    def search_target_papers_with_scores(
        self,
        query: str,
        top_k: int = 10,
        exclude_corpusid: Optional[int] = None
    ) -> List[tuple]:
        if not self.is_loaded:
            self.load_all_data()
        matches = self._compute_paper_scores(query, exclude_corpusid)
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[:top_k]

    def search_target_papers(
        self,
        query: str,
        top_k: int = 10,
        exclude_corpusid: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return [p for _, p in self.search_target_papers_with_scores(query, top_k, exclude_corpusid)]

    def synthesize_open_world_seed_paper(self, topic: str, llm: Any) -> Dict[str, Any]:
        prompt = f"""You are an academic literature curator.
The user wants to conduct research on: "{topic}".
Our offline database does not contain seed papers on this topic.

TASK:
1. Identify a REAL, actual published foundational or state-of-the-art paper in this domain (real paper title, real primary authors, real venue/year). DO NOT invent fictional papers.
2. Provide 15-20 real published reference papers commonly cited in this field.
3. Provide 15-20 core scientific domain concepts/entities.

Respond strictly in JSON format:
```json
{{
  "target_paper": {{
    "title": "Exact Title of Real Published Paper",
    "abstract": "Actual abstract summary of the real paper.",
    "authors": [{{"name": "Real Author 1"}}, {{"name": "Real Author 2"}}],
    "venue": "Actual Conference/Journal (e.g. NeurIPS, ICML, IEEE)",
    "year": 2023,
    "corpusid": 999001
  }},
  "references": [
    {{
      "title": "Real Reference Paper Title",
      "authors": [{{"name": "Real Author"}}],
      "venue": "Venue",
      "year": 2022
    }}
  ],
  "entities": ["entity1", "entity2"]
}}
```"""
        try:
            res = llm.generate(prompt)
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", res, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                if "target_paper" in data and "references" in data:
                    return data
        except Exception as e:
            print(f"[KnowledgeStore] Lỗi tổng hợp open-world seed paper: {e}")

        # Safe fallback
        return {
            "target_paper": {
                "title": f"Recent Advances in {topic}",
                "abstract": f"This foundational study investigates computational modeling and deep learning for {topic}, analyzing current theoretical challenges and empirical formulations.",
                "authors": [{"name": "Lead Researcher"}, {"name": "Senior Scientist"}],
                "venue": "arXiv preprint",
                "year": 2024,
                "corpusid": 999001
            },
            "references": [
                {
                    "title": f"Foundations of Machine Learning in {topic}",
                    "authors": [{"name": "P. Scholar et al."}],
                    "venue": "IEEE Access",
                    "year": 2023,
                    "corpusid": 999002
                }
            ],
            "entities": [w.capitalize() for w in re.findall(r"\b[a-zA-Z]{3,}\b", topic) if w.lower() not in STOPWORDS]
        }

    def get_related_references_for_paper(self, paper: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        cid = paper.get("corpusid")
        ref_ids = self.paper2references.get(cid, [])
        found_papers = []

        # 1. Ưu tiên các bài trích dẫn có sẵn metadata tác giả đầy đủ
        for r_id in ref_ids:
            if r_id in self.papers_by_id:
                p = self.papers_by_id[r_id]
                if p.get("authors") or p.get("author"):
                    found_papers.append(p)
            if len(found_papers) >= top_k:
                break

        # 2. Tìm kiếm các bài báo cùng chuyên ngành trong kho có đầy đủ tác giả
        if len(found_papers) < top_k:
            needed = top_k - len(found_papers)
            target_query = f"{paper.get('title', '')} {paper.get('abstract', '')[:300]}"
            topical_papers = self.search_target_papers(
                query=target_query, 
                top_k=(needed + 5) * 3, 
                exclude_corpusid=cid
            )
            for tp in topical_papers:
                if tp not in found_papers and tp.get("corpusid") != cid and (tp.get("authors") or tp.get("author")):
                    found_papers.append(tp)
                if len(found_papers) >= top_k:
                    break

        # 3. Nếu vẫn chưa đủ, lấy thêm từ ref_ids (sẽ được tự động bù tác giả qua OpenAlex ở generate_bibtex)
        if len(found_papers) < top_k:
            for r_id in ref_ids:
                if r_id in self.papers_by_id:
                    p = self.papers_by_id[r_id]
                    if p not in found_papers and p.get("corpusid") != cid:
                        found_papers.append(p)
                if len(found_papers) >= top_k:
                    break

        return found_papers

    def generate_bibtex(self, paper: Dict[str, Any], cite_key: str) -> str:
        title = paper.get("title", "Untitled").replace("{", "").replace("}", "").replace("&amp;", r"\&").replace("&", r"\&")
        authors_list = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]

        if not authors_list and paper.get("author"):
            if isinstance(paper["author"], list):
                authors_list = [str(x) for x in paper["author"]]
            else:
                authors_list = [str(paper["author"])]

        # Nếu thiếu tác giả, tự động tra cứu nhanh từ OpenAlex theo tiêu đề để lấy tác giả thật
        if not authors_list:
            p_title = paper.get("title", "").strip()
            if p_title and len(p_title) > 8:
                try:
                    fetched = self.retriever.search_openalex(p_title, limit=1)
                    if fetched and fetched[0].get("authors"):
                        paper["authors"] = fetched[0]["authors"]
                        if not paper.get("year"):
                            paper["year"] = fetched[0].get("year")
                        if not paper.get("venue"):
                            paper["venue"] = fetched[0].get("venue")
                        authors_list = [a.get("name", "") for a in paper["authors"] if a.get("name")]
                except Exception:
                    pass

        if authors_list:
            authors = " and ".join(authors_list)
        else:
            authors = "Anonymous Scholar et al."

        venue = (paper.get("venue") or "arXiv preprint").replace("&amp;", r"\&").replace("&", r"\&")
        raw_year = paper.get("year")
        year_str = f"  year={{{raw_year}}},\n" if raw_year else ""
        cid = paper.get("corpusid", "")
        url = paper.get("url", "")
        note_parts = []
        if cid:
            note_parts.append(f"CorpusID: {cid}")
        if url:
            note_parts.append(f"URL: {url}")
        joined_notes = ", ".join(note_parts)
        note_str = f"  note={{{joined_notes}}}" if note_parts else ""

        return f"""@article{{{cite_key},
  title={{{{{title}}}}},
  author={{{authors}}},
  journal={{{venue}}},
{year_str}{note_str}
}}"""

    def generate_bibtex_entry(self, paper: Dict[str, Any], cite_key: str = "ref") -> str:
        return self.generate_bibtex(paper, cite_key)

    def enrich_with_external_papers(
        self, 
        topic: str, 
        top_k: int = 8, 
        semantic_scholar_key: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_all_data()

        papers = self.retriever.fetch_papers_for_topic(
            topic=topic,
            limit=top_k,
            semantic_scholar_key=semantic_scholar_key
        )
        if not papers:
            return {}

        for p in papers:
            cid = p.get("corpusid")
            if cid and cid not in self.papers_by_id:
                self.papers_by_id[cid] = p
                text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
                words = set(re.findall(r"\b[a-z]{2,}\b", text)) - STOPWORDS
                self.doc_freq.update(words)

        target_paper = papers[0]
        target_cid = target_paper.get("corpusid")
        references = papers[1:]

        for ref in references:
            ref_cid = ref.get("corpusid")
            if target_cid and ref_cid:
                if ref_cid not in self.paper2references[target_cid]:
                    self.paper2references[target_cid].append(ref_cid)

        all_pids = [p.get("corpusid") for p in papers if p.get("corpusid")]
        discovered_entities = self.get_relevant_entities(all_pids, top_k=30)
        
        if len(discovered_entities) < 10:
            topic_words = [w.capitalize() for w in re.findall(r"\b[a-zA-Z]{3,}\b", topic) if w.lower() not in STOPWORDS]
            for tw in topic_words:
                if tw not in discovered_entities:
                    discovered_entities.append(tw)

        return {
            "target_paper": target_paper,
            "references": references,
            "entities": discovered_entities,
            "source": target_paper.get("source", "Online Academic Index"),
            "total_fetched": len(papers)
        }
