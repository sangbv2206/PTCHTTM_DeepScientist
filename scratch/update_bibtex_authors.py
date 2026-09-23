import re

# 2. Update core/knowledge_engine.py
ke_path = r"d:\PTIT\Y3-1\PTCHTTM\DeepScientist\core\knowledge_engine.py"
with open(ke_path, "r", encoding="utf-8") as f:
    ke_text = f.read()

pattern_ke = r'    def generate_bibtex_entry\(self, paper: Dict\[str, Any\], cite_key: Optional\[str\] = None\) -> str:[\s\S]*?return bibtex'

replacement_ke = '''    def generate_bibtex_entry(self, paper: Dict[str, Any], cite_key: Optional[str] = None) -> str:
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
            first_author = re.sub(r"\\W+", "", first_author_str.split()[-1].lower()) if first_author_str else "author"
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
        return bibtex'''

ke_new, count_ke = re.subn(pattern_ke, lambda m: replacement_ke, ke_text)
if count_ke > 0:
    with open(ke_path, "w", encoding="utf-8") as f:
        f.write(ke_new)
    print(f"Updated core/knowledge_engine.py ({count_ke} matches)")
else:
    print("Pattern not found in core/knowledge_engine.py")
