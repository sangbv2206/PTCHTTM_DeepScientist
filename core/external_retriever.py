import hashlib
import json
import os
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def normalize_paper_title(title: str) -> str:
    """Chuẩn hóa tiêu đề bài báo để khử trùng lặp chính xác (loại bỏ dấu câu và khoảng trắng thừa)."""
    return re.sub(r"[^\w\s]", "", title.lower()).strip()


def reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Tái cấu trúc chuỗi tóm tắt (abstract) từ cấu trúc inverted index của OpenAlex API."""
    if not inverted_index:
        return ""
    
    position_word_map = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word_map[pos] = word
            
    sorted_positions = sorted(position_word_map.keys())
    return " ".join([position_word_map[p] for p in sorted_positions])


def generate_stable_corpusid(title: str, prefix: int = 900) -> int:
    """Tạo một mã corpusid dạng số nguyên ổn định dựa trên hàm băm tiêu đề bài báo."""
    h = hashlib.md5(title.strip().lower().encode("utf-8")).hexdigest()
    num_part = int(h[:6], 16) % 900000
    return prefix * 1000000 + num_part


def check_internet_connection(timeout: float = 2.0) -> bool:
    """
    Kiểm tra nhanh kết nối Internet tới OpenAlex hoặc arXiv (Fast Ping timeout 2s).
    Trả về True nếu kết nối thành công, False nếu mất mạng hoàn toàn.
    """
    test_urls = [
        "https://api.openalex.org",
        "https://export.arxiv.org",
        "https://1.1.1.1"
    ]
    for url in test_urls:
        try:
            resp = requests.head(url, timeout=timeout)
            if resp.status_code < 500:
                return True
        except Exception:
            try:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code < 500:
                    return True
            except Exception:
                continue
    return False


class ExternalPaperRetriever:
    """
    Module thu thập dữ liệu bài báo khoa học trực tuyến từ các kho học thuật mở toàn cầu:
    1. OpenAlex API (Hơn 250 triệu bài báo, không cần API key, cực nhanh và phong phú).
    2. arXiv API (Kho preprint hàng đầu thế giới về AI, CS, Math).
    3. Semantic Scholar API (Tự động kích hoạt khi có API Key, fallback nếu bị 429).
    4. Local Disk Cache (Tự động lưu lại để chạy offline trong các lần tiếp theo).
    """

    def __init__(self, cache_dir: Optional[str] = None, email: str = "deepscientist@ptit.edu.vn"):
        if cache_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.cache_dir = os.path.join(base_dir, "data")
        else:
            self.cache_dir = cache_dir

        self.cache_file = os.path.join(self.cache_dir, "external_papers_cache.jsonl")
        self.email = email
        self.headers = {
            "User-Agent": f"DeepScientist/1.0 (mailto:{self.email})"
        }

    def search_openalex(self, query: str, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Tìm kiếm bài báo khoa học qua OpenAlex Works API (100% Miễn phí, không cần Key).
        """
        clean_q = query.strip()
        words = clean_q.split()
        if len(words) > 12:
            clean_q = " ".join(words[:12])
        encoded_query = urllib.parse.quote_plus(clean_q)
        url = f"https://api.openalex.org/works?search={encoded_query}&per-page={limit}&sort=relevance_score:desc"
        
        papers = []
        try:
            resp = requests.get(url, headers=self.headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", []):
                    title = item.get("display_name") or item.get("title") or ""
                    if not title or len(title.strip()) < 5:
                        continue

                    abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
                    
                    authors = []
                    for authorship in item.get("authorships", []):
                        author_name = authorship.get("author", {}).get("display_name")
                        if author_name:
                            authors.append({"name": author_name})

                    venue = "Academic Conference/Journal"
                    primary_loc = item.get("primary_location") or {}
                    source = primary_loc.get("source") or {}
                    if source.get("display_name"):
                        venue = source.get("display_name")
                    elif item.get("type"):
                        venue = item.get("type").replace("-", " ").title()

                    year = item.get("publication_year") or 2024
                    cit_count = item.get("cited_by_count", 0) or 0
                    cid = generate_stable_corpusid(title, prefix=901)

                    papers.append({
                        "corpusid": cid,
                        "title": title.strip(),
                        "abstract": abstract.strip(),
                        "authors": authors if authors else [{"name": f"{venue.split()[0]} Collaboration" if venue else "DeepScientist Consortium"}],
                        "venue": venue,
                        "year": year,
                        "citationcount": cit_count,
                        "url": item.get("doi") or item.get("id") or "",
                        "source": "OpenAlex"
                    })
        except Exception as e:
            print(f"[ExternalRetriever] OpenAlex request error: {e}")

        return papers

    def search_arxiv(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Tìm kiếm bài báo qua arXiv API (XML Atom Feed, chuyên sâu AI/ML/CS).
        """
        clean_q = re.sub(r"[^\w\s]", " ", query).strip()
        words = clean_q.split()
        if len(words) > 10:
            clean_q = " ".join(words[:10])
        encoded_query = urllib.parse.quote_plus(clean_q)
        url = f"https://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={limit}&sortBy=relevance&sortOrder=descending"

        papers = []
        try:
            resp = requests.get(url, headers=self.headers, timeout=12)
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                
                for entry in root.findall("atom:entry", ns):
                    title_elem = entry.find("atom:title", ns)
                    summary_elem = entry.find("atom:summary", ns)
                    published_elem = entry.find("atom:published", ns)
                    id_elem = entry.find("atom:id", ns)

                    if title_elem is None or not title_elem.text:
                        continue

                    title = " ".join(title_elem.text.split())
                    abstract = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else ""
                    
                    year = 2024
                    if published_elem is not None and published_elem.text:
                        year = int(published_elem.text[:4])

                    authors = []
                    for author in entry.findall("atom:author", ns):
                        name_elem = author.find("atom:name", ns)
                        if name_elem is not None and name_elem.text:
                            authors.append({"name": name_elem.text.strip()})

                    cid = generate_stable_corpusid(title, prefix=902)
                    papers.append({
                        "corpusid": cid,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors if authors else [{"name": "arXiv Researcher"}],
                        "venue": "arXiv.org",
                        "year": year,
                        "citationcount": 5,
                        "url": id_elem.text.strip() if id_elem is not None else "",
                        "source": "arXiv"
                    })
        except Exception as e:
            print(f"[ExternalRetriever] arXiv request error: {e}")

        return papers

    def search_semantic_scholar(
        self, 
        query: str, 
        limit: int = 5, 
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm bài báo qua Semantic Scholar Academic Graph (S2AG) API.
        """
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "corpusId,title,abstract,authors,venue,year,citationCount,url"
        }
        headers = dict(self.headers)
        if api_key:
            headers["x-api-key"] = api_key

        papers = []
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("data", []):
                    title = item.get("title")
                    if not title:
                        continue
                    cid = item.get("corpusId") or generate_stable_corpusid(title, prefix=903)
                    papers.append({
                        "corpusid": cid,
                        "title": title.strip(),
                        "abstract": (item.get("abstract") or "").strip(),
                        "authors": item.get("authors") or [{"name": "Scholar"}],
                        "venue": item.get("venue") or "arXiv.org",
                        "year": item.get("year") or 2024,
                        "citationcount": item.get("citationCount", 0) or 0,
                        "url": item.get("url") or "",
                        "source": "Semantic Scholar"
                    })
            elif resp.status_code == 429:
                print("[ExternalRetriever] Semantic Scholar Rate Limit (429), switching to OpenAlex/arXiv.")
        except Exception as e:
            print(f"[ExternalRetriever] Semantic Scholar error: {e}")

        return papers

    def fetch_papers_for_topic(
        self, 
        topic: str, 
        limit: int = 35, 
        semantic_scholar_key: Optional[str] = None,
        save_cache: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Chiến lược đa nguồn thông minh:
        1. Thử Semantic Scholar nếu có API Key.
        2. Tìm trên OpenAlex (ưu tiên số 1 vì tốc độ và độ bao phủ toàn diện).
        3. Nếu thiếu bài, bổ sung từ arXiv.
        4. Hợp nhất, loại bỏ trùng lặp và trả về danh sách bài mới nhất.
        """
        print(f"[ExternalRetriever] Đang tìm kiếm tài liệu học thuật trực tuyến cho chủ đề: '{topic}'...")
        all_papers: List[Dict[str, Any]] = []
        seen_titles = set()

        # 1. Thử Semantic Scholar nếu có API Key
        if semantic_scholar_key:
            s2_papers = self.search_semantic_scholar(topic, limit=limit, api_key=semantic_scholar_key)
            for p in s2_papers:
                t_norm = normalize_paper_title(p["title"])
                if t_norm not in seen_titles:
                    seen_titles.add(t_norm)
                    all_papers.append(p)

        # 2. Truy vấn OpenAlex (Không cần key, cực kỳ ổn định)
        if len(all_papers) < limit:
            oa_papers = self.search_openalex(topic, limit=limit)
            for p in oa_papers:
                t_norm = normalize_paper_title(p["title"])
                if t_norm not in seen_titles:
                    seen_titles.add(t_norm)
                    all_papers.append(p)

        # 3. Bổ sung từ arXiv nếu vẫn chưa đủ số lượng
        if len(all_papers) < limit:
            arxiv_papers = self.search_arxiv(topic, limit=limit)
            for p in arxiv_papers:
                t_norm = normalize_paper_title(p["title"])
                if t_norm not in seen_titles:
                    seen_titles.add(t_norm)
                    all_papers.append(p)

        if all_papers:
            print(f"[ExternalRetriever] Thu thập thành công {len(all_papers)} bài báo khoa học thực tế từ mạng!")
            if save_cache:
                self.save_to_cache(all_papers)

        return all_papers[:limit]

    def save_to_cache(self, papers: List[Dict[str, Any]]):
        """Lưu bài báo vào cache JSONL để tái sử dụng offline."""
        os.makedirs(self.cache_dir, exist_ok=True)
        existing_cids = set()
        existing_titles = set()
        
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            p = json.loads(line)
                            if p.get("corpusid"):
                                existing_cids.add(p["corpusid"])
                            if p.get("title"):
                                existing_titles.add(normalize_paper_title(p["title"]))
            except Exception:
                pass

        new_count = 0
        with open(self.cache_file, "a", encoding="utf-8") as f:
            for p in papers:
                cid = p.get("corpusid")
                t_norm = normalize_paper_title(p.get("title", ""))
                if cid and cid not in existing_cids and t_norm and t_norm not in existing_titles:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
                    existing_cids.add(cid)
                    existing_titles.add(t_norm)
                    new_count += 1

        if new_count > 0:
            print(f"[ExternalRetriever] Da ghi nho {new_count} bai bao moi vao cache offline: {self.cache_file}")

    def load_from_cache(self) -> List[Dict[str, Any]]:
        """Tải các bài báo đã được cache từ trước."""
        if not os.path.exists(self.cache_file):
            return []

        cached_papers = []
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cached_papers.append(json.loads(line))
        except Exception as e:
            print(f"[ExternalRetriever] Loi doc cache: {e}")

        return cached_papers
