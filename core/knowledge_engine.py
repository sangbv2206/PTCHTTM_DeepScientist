import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any


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
    "study", "applications", "application", "system", "systems", "via", "towards", "toward", "analysis"
}


class KnowledgeEngine:

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, "data")
        else:
            self.data_dir = data_dir

        self.papers_path = os.path.join(self.data_dir, "papers.jsonl")
        self.knowledge_path = os.path.join(self.data_dir, "knowledge.jsonl")
        self.references_path = os.path.join(self.data_dir, "references.jsonl")

        self.papers_by_id: Dict[int, Dict[str, Any]] = {}
        self.paper2entities: Dict[int, Dict[str, int]] = {}
        self.entity_counter: Counter = Counter()
        self.is_loaded = False

    def load_data(self, max_papers: Optional[int] = None):
        """Tải dữ liệu bài báo và thực thể vào bộ nhớ với index tối ưu."""
        if self.is_loaded:
            return

        print(f"[KnowledgeEngine] Loading dataset from: {self.data_dir}...")
        
        # 1. Load papers
        count = 0
        if os.path.exists(self.papers_path):
            with open(self.papers_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    paper = json.loads(line)
                    corpusid = paper.get("corpusid")
                    if corpusid:
                        self.papers_by_id[corpusid] = paper
                        count += 1
                        if max_papers and count >= max_papers:
                            break
            print(f"[KnowledgeEngine] Loaded {len(self.papers_by_id)} papers from papers.jsonl.")
        else:
            print(f"[KnowledgeEngine] WARNING: File not found: {self.papers_path}")

        # 2. Load entities from knowledge.jsonl
        if os.path.exists(self.knowledge_path):
            k_count = 0
            with open(self.knowledge_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    cid = item.get("corpusid")
                    knowledge = item.get("knowledge", {})
                    if cid and knowledge:
                        self.paper2entities[cid] = knowledge
                        self.entity_counter.update(knowledge)
                        k_count += 1
            print(f"[KnowledgeEngine] Loaded {k_count} entity records from knowledge.jsonl.")
        
        self.is_loaded = True

    def search_papers(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm các bài báo liên quan nhất trong dataset theo từ khóa/chủ đề (đã loại bỏ stopwords).
        """
        if not self.is_loaded:
            self.load_data()

        query_tokens = [w for w in re.findall(r"\w+", query.lower()) if w not in STOPWORDS]
        if not query_tokens:
            query_tokens = re.findall(r"\w+", query.lower())

        scores = []
        for cid, paper in self.papers_by_id.items():
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            t_lower = title.lower()
            a_lower = abstract.lower()

            t_tokens = set(re.findall(r"\w+", t_lower))
            a_tokens = set(re.findall(r"\w+", a_lower))
            
            t_overlap = len(set(query_tokens).intersection(t_tokens))
            a_overlap = len(set(query_tokens).intersection(a_tokens))

            if t_overlap > 0 or a_overlap > 0:
                cit_count = paper.get("citationcount", 0) or 0
                score = (t_overlap * 12.0) + (a_overlap * 3.0) + min(cit_count / 25.0, 3.0)
                scores.append((score, paper))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = [p for _, p in scores[:top_k]]

        if len(results) < top_k:
            all_sorted = sorted(
                self.papers_by_id.values(),
                key=lambda p: p.get("citationcount", 0) or 0,
                reverse=True
            )
            for p in all_sorted:
                if p not in results:
                    results.append(p)
                if len(results) >= top_k:
                    break

        return results

    def get_paper_entities(self, corpusid: int) -> List[str]:
        """Lấy danh sách các khái niệm/thực thể khoa học gắn với bài báo."""
        entities = self.paper2entities.get(corpusid, {})
        # Sắp xếp theo mức độ quan trọng
        sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_entities]

    def format_paper_context(self, paper: Dict[str, Any]) -> str:
        """Định dạng thông tin bài báo để đưa vào prompt cho LLM."""
        authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])][:5])
        year = paper.get("year", "N/A")
        venue = paper.get("venue", "N/A")
        title = paper.get("title", "")
        cid = paper.get("corpusid", "")
        
        entities = self.get_paper_entities(cid)
        entities_str = ", ".join(entities[:10]) if entities else "General machine learning"

        context = f"""Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}
CorpusID: {cid}
Key Scientific Entities: {entities_str}"""
        return context

    def generate_bibtex_entry(self, paper: Dict[str, Any], cite_key: Optional[str] = None) -> str:
        """Tạo trích dẫn BibTeX chuẩn từ dữ liệu bài báo thật."""
        cid = paper.get("corpusid", "ref")
        title = paper.get("title", "Untitled").replace("{", "").replace("}", "")
        
        raw_authors = paper.get("authors", [])
        author_names = [a.get("name", "") for a in raw_authors if isinstance(a, dict) and a.get("name")]
        if not author_names and paper.get("author"):
            author_names = [paper["author"]] if isinstance(paper["author"], str) else paper["author"]
            
        if author_names:
            authors = " and ".join(author_names)
            first_author_str = author_names[0]
        else:
            venue_name = paper.get("venue")
            if venue_name and "arxiv" not in venue_name.lower() and len(venue_name) < 40:
                authors = f"{venue_name.strip()} Research Group"
                first_author_str = venue_name
            else:
                title_lead = title.split()[0] if title else "Author"
                authors = f"{title_lead} et al."
                first_author_str = title_lead

        if not cite_key:
            first_author = re.sub(r"\W+", "", first_author_str.split()[-1].lower()) if first_author_str else "author"
            year = paper.get("year", 2024)
            cite_key = f"{first_author}{year}_{cid}"

        venue = paper.get("venue", "arXiv preprint")
        year = paper.get("year", 2024)

        bibtex = f"""@article{{{cite_key},
  title={{{{{{title}}}}}},
  author={{{{{{authors}}}}}},
  journal={{{{{{venue}}}}}},
  year={{{{{{year}}}}}},
  note={{CorpusID: {cid}}}
}}"""
        return bibtex
