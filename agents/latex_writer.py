import json
import os
import re
import shutil
import subprocess
from typing import Dict, Any, List, Optional
from core.gemini_llm import GeminiLLM
from core.knowledge_engine import KnowledgeEngine
from core.knowledge_store import STOPWORDS


class LaTeXWriter:
    """
    Tác nhân soạn thảo bài báo khoa học chuẩn quốc tế (kế thừa triết lý perform_writeup của Sakana AI):
    - Tự động sinh trọn vẹn từng section bằng văn phong học thuật cao cấp.
    - Tạo file trích dẫn chuẩn `references.bib` từ các bài báo thật trong dataset của ResearchAgent.
    - Nhúng biểu đồ `.png` và bảng số liệu thực nghiệm thật vào mã nguồn LaTeX.
    - Biên dịch tự động ra `paper.pdf` bằng pdflatex.
    """

    def __init__(self, llm: GeminiLLM, knowledge_engine: KnowledgeEngine):
        self.llm = llm
        self.knowledge_engine = knowledge_engine

    def write_paper(
        self,
        output_dir: str,
        context: Dict[str, Any],
        problem_info: Dict[str, Any],
        method_info: Dict[str, Any],
        experiment_design: Dict[str, Any],
        metrics_data: Dict[str, Any],
        experiment_dir: str,
    ) -> str:
        latex_dir = os.path.join(output_dir, "latex")
        os.makedirs(latex_dir, exist_ok=True)

        # 1. Sao chép các biểu đồ thực nghiệm vào thư mục latex
        for fig_name in ["figure_1.png", "figure_2.png", "figure_3.png"]:
            fig_src = os.path.join(experiment_dir, fig_name)
            if os.path.exists(fig_src):
                shutil.copy(fig_src, os.path.join(latex_dir, fig_name))
            else:
                for root, _, files in os.walk(experiment_dir):
                    if fig_name in files:
                        shutil.copy(os.path.join(root, fig_name), os.path.join(latex_dir, fig_name))
                        break

        # 2. Xây dựng file references.bib từ dataset thật
        bib_entries = []
        cite_keys = []
        target_paper = context.get("target_paper")
        if target_paper:
            key = "target_ref"
            bib = self.knowledge_engine.generate_bibtex_entry(target_paper, cite_key=key)
            bib_entries.append(bib)
            cite_keys.append((key, target_paper.get("title", "")))

        related_papers = context.get("related_papers", [])
        for i, paper in enumerate(related_papers[:8]):
            key = f"ref_{i+1}"
            bib = self.knowledge_engine.generate_bibtex_entry(paper, cite_key=key)
            bib_entries.append(bib)
            cite_keys.append((key, paper.get("title", "")))

        bibtex_content = "\n\n".join(bib_entries)
        bib_path = os.path.join(latex_dir, "references.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write(bibtex_content)

        cite_keys_prompt = "\n".join([f"- \\cite{{{k}}}: {t}" for k, t in cite_keys])

        # 3. Đọc template LaTeX
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "templates",
            "academic_paper.tex"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            tex_content = f.read()

        user_topic = context.get("user_topic") or context.get("topic") or ""

        # 4. Sinh từng Section của bài báo qua Gemini 2.0 Flash
        print("[LaTeXWriter] Đang tạo Tiêu đề và Tóm tắt (Abstract)...")
        title, abstract = self._write_title_and_abstract(problem_info, method_info, user_topic=user_topic)

        print("[LaTeXWriter] Đang viết Introduction...")
        intro = self._write_introduction(problem_info, method_info, cite_keys_prompt, user_topic=user_topic)

        print("[LaTeXWriter] Đang viết Related Work & Literature Grounding...")
        related_work = self._write_related_work(cite_keys_prompt, problem_info, user_topic=user_topic)

        print("[LaTeXWriter] Đang viết Problem Formulation & Theoretical Background...")
        formulation = self._write_formulation(problem_info, method_info, user_topic=user_topic)

        print("[LaTeXWriter] Đang viết Proposed Methodology...")
        methodology = self._write_methodology(method_info, user_topic=user_topic)

        print("[LaTeXWriter] Đang viết Experimental Setup & Results...")
        experiments = self._write_experiments(experiment_design, metrics_data, user_topic=user_topic)

        print("[LaTeXWriter] Đang viết Discussion & Limitations...")
        discussion = self._write_discussion(problem_info, method_info, methodology, metrics_data, user_topic=user_topic)

        print("[LaTeXWriter] Đang viết Conclusion...")
        conclusion = self._write_conclusion(problem_info, method_info, methodology, metrics_data, user_topic=user_topic)

        # 5. Ráp vào template và làm sạch mã LaTeX
        tex_content = tex_content.replace("<<TITLE>>", title.strip())
        tex_content = tex_content.replace("<<ABSTRACT>>", self._sanitize_section(abstract, "Abstract"))
        tex_content = tex_content.replace("<<INTRODUCTION>>", self._sanitize_section(intro, "Introduction"))
        tex_content = tex_content.replace("<<RELATED_WORK>>", self._sanitize_section(related_work, "Related Work"))
        tex_content = tex_content.replace("<<BACKGROUND_AND_FORMULATION>>", self._sanitize_section(formulation, "Problem Formulation & Theoretical Background"))
        tex_content = tex_content.replace("<<METHODOLOGY>>", self._sanitize_section(methodology, "Proposed Methodology"))
        tex_content = tex_content.replace("<<EXPERIMENTS_AND_RESULTS>>", self._sanitize_section(experiments, "Experimental Setup & Empirical Evaluation"))
        tex_content = tex_content.replace("<<DISCUSSION>>", self._sanitize_section(discussion, "Discussion, Limitations & Future Work"))
        tex_content = tex_content.replace("<<CONCLUSION>>", self._sanitize_section(conclusion, "Conclusion"))

        tex_path = os.path.join(latex_dir, "paper.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)

        print(f"[LaTeXWriter] Đã hoàn thành mã nguồn LaTeX tại: {tex_path}")
        print(f"[LaTeXWriter] Tiêu đề bài báo chính thức: '{title.strip()}'")

        # 6. Biên dịch tự động sang PDF
        pdf_path = self.compile_pdf(latex_dir)
        return pdf_path, title.strip()

    def compile_pdf(self, latex_dir: str) -> Optional[str]:
        """Biên dịch mã nguồn LaTeX sang PDF bằng pdflatex và bibtex."""
        print("[LaTeXWriter] Đang biên dịch mã nguồn LaTeX sang PDF...")
        cmd_seq = [
            ["pdflatex", "-interaction=nonstopmode", "--enable-installer", "paper.tex"],
            ["bibtex", "paper"],
            ["pdflatex", "-interaction=nonstopmode", "--enable-installer", "paper.tex"],
            ["pdflatex", "-interaction=nonstopmode", "--enable-installer", "paper.tex"],
        ]

        for cmd in cmd_seq:
            try:
                subprocess.run(
                    cmd,
                    cwd=latex_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=180,
                    check=False
                )
            except Exception as e:
                print(f"[LaTeXWriter] Cảnh báo trong lệnh {' '.join(cmd)}: {e}")

        pdf_file = os.path.join(latex_dir, "paper.pdf")
        if os.path.exists(pdf_file):
            print(f"[LaTeXWriter] BIÊN DỊCH PDF THÀNH CÔNG: {pdf_file}")
            return pdf_file
        else:
            print("[LaTeXWriter] Không thể tạo file PDF tự động (Kiểm tra log trong thư mục latex).")
            return None

    def _sanitize_section(self, text: str, parent_title: str = "") -> str:
        """Làm sạch và chuẩn hóa nội dung LaTeX sinh ra bởi LLM."""
        if not text:
            return ""
        # 1. Loại bỏ markdown code blocks nếu LLM bọc ```latex ... ```
        text = re.sub(r"^```(?:latex|tex)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

        # 2. Xóa các câu rò rỉ layout constraints / prompt instructions
        leaked_patterns = [
            r"(?i)[^\.\n]*to ensure (?:strict )?compliance with (?:the )?(?:layout|formatting|column|narrow-column|page) constraints[^\.\n]*[\.\n]?",
            r"(?i)[^\.\n]*due to (?:two-column|narrow-column|column|layout) constraints[^\.\n]*[\.\n]?",
            r"(?i)[^\.\n]*in compliance with (?:two-column|narrow-column|layout) format[^\.\n]*[\.\n]?",
            r"(?i)[^\.\n]*we format the (?:equation|formulation) to fit (?:narrow )?columns[^\.\n]*[\.\n]?",
            r"(?i)[^\.\n]*as requested by (?:the )?layout constraints[^\.\n]*[\.\n]?",
            r"(?i)[^\.\n]*strictly adherence to layout constraints[^\.\n]*[\.\n]?",
        ]
        for pat in leaked_patterns:
            text = re.sub(pat, "", text)

        # 3. Xử lý triệt để trùng lặp tiêu đề cha-con
        if parent_title:
            norm_parent = re.sub(r"[^a-zA-Z0-9]", "", parent_title).lower()

            def clean_header(match):
                tag = match.group(1) # 'section' or 'subsection'
                h_title = match.group(2).strip()
                norm_h = re.sub(r"[^a-zA-Z0-9]", "", h_title).lower()
                # Nếu tiêu đề con giống hoặc trùng lặp với tiêu đề cha (vd: 'Proposed Methodology' lặp lại)
                if norm_parent in norm_h or norm_h in norm_parent or (len(norm_h) > 6 and norm_h[:12] == norm_parent[:12]):
                    return ""
                # Xóa số thứ tự thủ công (vd: '5.1 ' hay '5.2 ') để LaTeX tự động đánh số
                clean_title = re.sub(r"^\d+(\.\d+)*\s*", "", h_title).strip()
                return f"\\subsection{{{clean_title}}}"

            text = re.sub(r"\\(section|subsection)\*?\{([^}]+)\}", clean_header, text)
        else:
            text = re.sub(r"\\section\*?\{([^}]+)\}", r"\\subsection{\1}", text)

        # 4. Thay thế các ký tự HTML entity như &amp; thành \&
        text = text.replace("&amp;", r"\&")

        # 5. Đảm bảo các bảng tabular không bị tràn cột (Overfull \hbox)
        def wrap_tabular_callback(match):
            start = match.start()
            preceding = text[max(0, start - 60):start]
            if "\\resizebox" in preceding:
                return match.group(0)
            return f"\\resizebox{{\\columnwidth}}{{!}}{{\n{match.group(0)}\n}}"

        text = re.sub(r"\\begin\{tabular\}[\s\S]*?\\end\{tabular\}", wrap_tabular_callback, text)

        # 6. Loại bỏ hoàn toàn lỗi cú pháp lồng môi trường \begin{section} / \end{section}
        text = re.sub(r"\\begin\{section\}\*?(\{[^}]*\})?", "", text)
        text = re.sub(r"\\end\{section\}", "", text)

        # 7. Chuyển đổi tiêu đề dạng Markdown (###, ##, #) thành \subsection hoặc \subsubsection
        text = re.sub(r"^###\s*(.+)$", r"\\subsubsection{\1}", text, flags=re.MULTILINE)
        text = re.sub(r"^##\s*(.+)$", r"\\subsection{\1}", text, flags=re.MULTILINE)
        text = re.sub(r"^#\s*(.+)$", r"\\subsection{\1}", text, flags=re.MULTILINE)

        # 8. Escape ký tự # đứng đơn lẻ (macro parameter character)
        text = re.sub(r"(?<!\\)#", r"\#", text)

        return text.strip()

    def _write_title_and_abstract(
        self,
        problem_info: Dict[str, Any],
        method_info: Dict[str, Any],
        user_topic: str = ""
    ) -> tuple:
        prompt = f"""Generate an academic conference paper Title and Abstract for this research.

MANDATORY USER TOPIC ANCHOR:
"{user_topic}"

Method: {method_info.get('method_name')} ({method_info.get('abbreviation')})
Problem: {problem_info.get('problem')}
Rationale: {problem_info.get('rationale')}

CRITICAL TITLE REQUIREMENTS:
1. The Title MUST strictly, explicitly incorporate the core concepts/keywords of "{user_topic}".
2. You can be creative and academic (e.g. including the method name, specific mathematical framework, or target problem), BUT "{user_topic}" MUST remain the central theme.
3. NEVER drift into unrelated topics or omit the user's core concepts!

Format response as:
TITLE: <Academic Paper Title explicitly featuring {user_topic}>
ABSTRACT: <Continuous professional abstract paragraph, 150-250 words, motivation, methodology, theoretical and empirical results>
"""
        res = self.llm.generate(prompt)
        title = ""
        abstract = f"This paper presents a novel approach to advancing scientific discovery on {user_topic}."

        for line in res.split("\n"):
            if line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
            elif line.startswith("ABSTRACT:"):
                abstract = line.replace("ABSTRACT:", "").strip()

        # Title Guardrail: Verify that title contains key concepts of user_topic
        if user_topic:
            topic_keywords = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", user_topic) if w.lower() not in STOPWORDS]
            title_lower = title.lower()
            matched = [w for w in topic_keywords if w in title_lower]
            if not title or (topic_keywords and len(matched) == 0):
                method_name = method_info.get("method_name") or method_info.get("abbreviation") or "DeepScientist"
                title = f"{method_name}: Advancing {user_topic}"

        if not title:
            title = method_info.get("method_name", f"Autonomous AI Discovery: {user_topic}")

        return title, abstract

    def _write_introduction(
        self,
        problem_info: Dict[str, Any],
        method_info: Dict[str, Any],
        cite_keys: str,
        user_topic: str = ""
    ) -> str:
        prompt = f"""Write a comprehensive, publication-grade Introduction section in LaTeX for an AI conference paper.

MANDATORY CORE TOPIC:
"{user_topic}"

Research Problem: {problem_info.get('problem')}
Motivation: {problem_info.get('rationale')}
Method: {method_info.get('method_name')} ({method_info.get('abbreviation')})
Core Intuition: {method_info.get('intuition')}

Available Citation Keys (Use \\cite{{...}} accurately):
{cite_keys}

Requirements:
- 3 to 4 well-structured paragraphs.
- Must be strictly centered on investigating and solving "{user_topic}".
- Clearly state the problem, why current paradigms fall short, and introduce our proposed method.
- End with an explicit bulleted list of key contributions using \\begin{{itemize}} ... \\end{{itemize}}.
- Do NOT include \\section{{Introduction}}. Provide ONLY the LaTeX body text.
"""
        return self.llm.generate(prompt)

    def _write_related_work(
        self,
        cite_keys: str,
        problem_info: Dict[str, Any],
        user_topic: str = ""
    ) -> str:
        prompt = f"""Write the Related Work section in LaTeX for our paper.

MANDATORY CORE TOPIC:
"{user_topic}"

Problem Domain: {problem_info.get('problem')}

You MUST ground your discussion by citing the following available papers using \\cite{{key}}:
{cite_keys}

Requirements:
- Organize into 2-3 logical sub-themes directly related to "{user_topic}".
- Compare and contrast prior approaches, highlighting why our work differs and solves unaddressed gaps.
- Provide ONLY the LaTeX body text (no \\section{{Related Work}}).
"""
        return self.llm.generate(prompt)

    def _write_formulation(
        self,
        problem_info: Dict[str, Any],
        method_info: Dict[str, Any],
        user_topic: str = ""
    ) -> str:
        prompt = f"""Write the Problem Formulation and Theoretical Background section in LaTeX.

MANDATORY CORE TOPIC:
"{user_topic}"

Problem: {problem_info.get('problem')}
Intuition: {method_info.get('intuition')}
Math Formulation details: {method_info.get('mathematical_formulation')}

Requirements:
- Ground the theoretical formulation strictly in the context of "{user_topic}".
- Introduce formal mathematical definitions, variable notations, and domain constraints.
- CRITICAL TWO-COLUMN MATH LAYOUT CONSTRAINT:
  The paper uses a two-column layout with narrow column width (\\columnwidth ~ 3.25 inches).
  Single-line equations longer than 45 characters will overflow into the adjacent column and overlap text.
  You MUST format long mathematical expressions across multiple lines using:
  \\begin{{equation}}
  \\begin{{aligned}}
  \\min_{{\\theta}} \\mathcal{{L}}(\\theta) &= \\text{{term 1}} \\\\
  &\\quad + \\text{{term 2}}
  \\end{{aligned}}
  \\end{{equation}}
  NEVER write long, unbroken equations on a single line!
  NEVER mention layout constraints or formatting rules in the academic text itself.
- Do NOT include \\section{{...}} tags in your response (use only \\subsection{{...}} or \\subsubsection{{...}}).
- Provide ONLY LaTeX body text.
"""
        return self.llm.generate(prompt)

    def _write_methodology(
        self,
        method_info: Dict[str, Any],
        user_topic: str = ""
    ) -> str:
        prompt = f"""Write the Proposed Methodology section in LaTeX.

MANDATORY CORE TOPIC:
"{user_topic}"

Method Name: {method_info.get('method_name')} ({method_info.get('abbreviation')})
Formulation: {method_info.get('mathematical_formulation')}
Procedure: {json.dumps(method_info.get('algorithmic_procedure', []))}
Theoretical Justification: {method_info.get('theoretical_justification')}

Requirements:
- The methodology must be tailored to advancing "{user_topic}".
- Provide an in-depth, rigorous breakdown of the architecture, objective functions, and optimization algorithm.
- CRITICAL TWO-COLUMN MATH LAYOUT CONSTRAINT:
  The paper uses a two-column layout with narrow column width.
  You MUST break long equations across multiple lines using \\begin{{equation}}\\begin{{aligned}} ... &= ... \\\\ &\\quad + ... \\end{{aligned}}\\end{{equation}}.
  NO individual equation line may exceed 45 characters.
  NEVER mention layout constraints or formatting rules in the academic text itself.
- Do NOT include \\section{{...}} tags in your response (use only \\subsection{{...}} or \\subsubsection{{...}}).
- Provide ONLY LaTeX body text.
"""
        return self.llm.generate(prompt)

    def _write_experiments(
        self,
        experiment_design: Dict[str, Any],
        metrics_data: Dict[str, Any],
        user_topic: str = ""
    ) -> str:
        prompt = f"""Write the Experimental Setup & Empirical Evaluation section in LaTeX.

MANDATORY CORE TOPIC:
"{user_topic}"

Baselines: {json.dumps(experiment_design.get('baselines', []))}
Metrics Measured: {json.dumps(experiment_design.get('metrics', []))}
Actual Empirical Results Obtained: {json.dumps(metrics_data)}

Requirements:
- Discuss the experimental environment, dataset/benchmarks, and training setup specifically evaluated for "{user_topic}".
- You MUST reference all 3 generated figures:
  1. Convergence Dynamics:
  \\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=0.95\\linewidth]{{figure_1.png}}
  \\caption{{Training and validation convergence dynamics across epochs.}}
  \\label{{fig:convergence}}
  \\end{{figure}}

  2. Quantitative Benchmark vs Baselines:
  \\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=0.95\\linewidth]{{figure_2.png}}
  \\caption{{Empirical benchmark comparison of the proposed method against competitive baselines.}}
  \\label{{fig:comparison}}
  \\end{{figure}}

  3. Ablation & Sensitivity Analysis:
  \\begin{{figure}}[htbp]
  \\centering
  \\includegraphics[width=0.95\\linewidth]{{figure_3.png}}
  \\caption{{Ablation study and parameter sensitivity / error distribution analysis.}}
  \\label{{fig:ablation}}
  \\end{{figure}}

- Include an academic LaTeX table comparing the quantitative numbers:
  \\begin{{table}}[htbp]
  \\centering
  \\caption{{Quantitative performance comparison on empirical benchmark metrics.}}
  \\label{{tab:benchmark}}
  \\resizebox{{\\columnwidth}}{{!}}{{
  \\begin{{tabular}}{{lcccc}}
  \\toprule
  \\textbf{{Method}} & ... \\\\
  \\midrule
  ... \\\\
  \\bottomrule
  \\end{{tabular}}
  }}
  \\end{{table}}
- Provide detailed analysis of why the proposed method outperformed baselines based strictly on the empirical data.
- Provide ONLY LaTeX body text.
"""
        return self.llm.generate(prompt)

    def _write_discussion(
        self, 
        problem_info: Dict[str, Any], 
        method_info: Dict[str, Any],
        methodology: str = "",
        metrics_data: Optional[Dict[str, Any]] = None,
        user_topic: str = ""
    ) -> str:
        prompt = f"""Write the Discussion, Limitations & Future Work section in LaTeX.

MANDATORY CORE TOPIC:
"{user_topic}"

Method Name: {method_info.get('method_name')}
Problem: {problem_info.get('problem')}
Actual Method Details developed in this paper:
{methodology[:1200]}
Actual Empirical Results Obtained: {json.dumps(metrics_data or {})}

Requirements:
- Critically discuss practical implications for "{user_topic}", computational efficiency, and boundary conditions.
- Explicitly discuss limitations and exciting avenues for future research.
- STRICT GROUNDING CONSTRAINT: You MUST strictly base your discussion ONLY on the actual method and empirical results shown above. NEVER invent or hallucinate unrelated mathematical tools that were not part of the actual proposed methodology.
- Provide ONLY LaTeX body text.
"""
        return self.llm.generate(prompt)

    def _write_conclusion(
        self, 
        problem_info: Dict[str, Any], 
        method_info: Dict[str, Any],
        methodology: str = "",
        metrics_data: Optional[Dict[str, Any]] = None,
        user_topic: str = ""
    ) -> str:
        prompt = f"""Write a concise, high-impact Conclusion section in LaTeX summarizing the major achievements of {method_info.get('method_name')} and its scientific contribution to "{user_topic}".

Actual Method Details developed in this paper:
{methodology[:1200]}
Actual Empirical Results: {json.dumps(metrics_data or {})}

Requirements:
- Summarize ONLY the actual technical contributions and empirical outcomes established in this paper for "{user_topic}".
- STRICT GROUNDING CONSTRAINT: Do NOT hallucinate or introduce external concepts not present in the proposed methodology.
- Provide ONLY LaTeX body text. NEVER write \\begin{{section}} or \\end{{section}}.
"""
        return self.llm.generate(prompt)
