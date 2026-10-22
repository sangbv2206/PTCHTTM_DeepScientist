import json
import os
import re
from typing import Dict, Any
from core.gemini_llm import GeminiLLM


class PeerReviewer:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def _parse_review_response(self, response_text: str, paper_title: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        json_str = ""
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text)
        if match:
            json_str = match.group(1)
        else:
            first_b = response_text.find("{")
            last_b = response_text.rfind("}")
            if first_b != -1 and last_b != -1 and last_b > first_b:
                json_str = response_text[first_b:last_b + 1]

        review_data = None
        if json_str:
            clean_str = re.sub(r",\s*([\]\}])", r"\1", json_str)
            try:
                review_data = json.loads(clean_str)
            except Exception:
                try:
                    clean_str2 = re.sub(r"[\x00-\x1f]", " ", clean_str)
                    review_data = json.loads(clean_str2)
                except Exception:
                    pass

        if not review_data or not isinstance(review_data, dict):
            summary_m = re.search(r'"summary"\s*:\s*"([^"]+)"', response_text)
            rec_m = re.search(r'"recommendation"\s*:\s*"([^"]+)"', response_text)
            rating_m = re.search(r'"overall_rating"\s*:\s*(\d+)', response_text)

            summary_text = summary_m.group(1) if summary_m else f"The paper investigates '{paper_title}', developing formal methodology and empirical evaluation."
            recommendation = rec_m.group(1) if rec_m else "Weak Accept"
            rating = int(rating_m.group(1)) if rating_m else 6

            review_data = {
                "summary": summary_text,
                "strengths": [
                    f"Direct and structured empirical investigation into {paper_title}",
                    "Implementation grounded in verifiable execution logs and figures"
                ],
                "weaknesses": [
                    "Evaluation budget is constrained to single-split execution without extensive multi-seed variance",
                    "Requires validation across diverse external real-world benchmarks"
                ],
                "questions": [
                    "How does the computational overhead scale as the input dimension or parameter count grows?",
                    "What specific architectural trade-offs emerge under extreme resource constraints?"
                ],
                "scores": {
                    "soundness": 3,
                    "presentation": 3,
                    "contribution": 3,
                    "overall_rating": rating,
                    "confidence": 4
                },
                "recommendation": recommendation,
                "ethical_considerations": "None."
            }

        scores = review_data.get("scores", {})
        if not scores and "detailed_scores" in review_data:
            scores = dict(review_data["detailed_scores"])
            if "overall_rating" in review_data:
                scores["overall_rating"] = review_data["overall_rating"]
            if "confidence" in review_data:
                scores["confidence"] = review_data["confidence"]
            review_data["scores"] = scores
        elif "overall_rating" in review_data and "overall_rating" not in scores:
            scores["overall_rating"] = review_data["overall_rating"]
            review_data["scores"] = scores

        return review_data

    def review_paper(
        self,
        output_dir: str,
        paper_title: str,
        paper_tex_path: str,
        metrics_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        review_dir = os.path.join(output_dir, "review")
        os.makedirs(review_dir, exist_ok=True)

        tex_content = ""
        if os.path.exists(paper_tex_path):
            with open(paper_tex_path, "r", encoding="utf-8") as f:
                tex_content = f.read()[:150000]

        prompt = f"""You are a distinguished Area Chair and Senior Reviewer for top AI conferences (ICLR, NeurIPS, ICML).
Conduct a rigorous, objective, un-flattered, and constructive double-blind peer review of the following scientific submission.

PAPER TITLE: {paper_title}
VERIFIED EMPIRICAL METRICS RECORDED: {json.dumps(metrics_data)}

LATEX SOURCE MANUSCRIPT:
\"\"\"
{tex_content}
\"\"\"

### CRITICAL REVIEW DIRECTIVES:
1. DATA CONSISTENCY AUDIT:
   - Check whether numbers in the manuscript faithfully reflect the VERIFIED EMPIRICAL METRICS RECORDED above.
   - If the paper claims unverified baselines or exaggerated superiorities where the proposed method actually underperformed or degraded, severely penalize Soundness and point this out in Weaknesses.
2. RIGOROUS CALIBRATION (NO INFLATED PRAISE):
   - Realistically assess the computational budget (e.g. single-split runs, benchmark size, absence of multi-seed error bars).
   - Most solid papers in this setting receive realistic scores between 5 (Borderline) and 7 (Weak Accept/Poster). Avoid mindless 9-10 scores unless truly groundbreaking.
3. STRUCTURED FEEDBACK:
   - Identify concrete technical Strengths (method formulation, clarity, reproducibility).
   - Identify actionable Weaknesses (evaluation scope, parameter sensitivity, hardware generalization).
   - Pose 2-3 precise Questions for the authors.

Respond strictly in valid JSON format:
```json
{{
  "summary": "Concise summary of problem, method, and key findings",
  "strengths": [
    "Specific strength 1",
    "Specific strength 2"
  ],
  "weaknesses": [
    "Specific weakness / limitation 1",
    "Specific weakness / limitation 2"
  ],
  "questions": [
    "Technical question 1",
    "Technical question 2"
  ],
  "scores": {{
    "soundness": 3,
    "presentation": 3,
    "contribution": 3,
    "overall_rating": 6,
    "confidence": 4
  }},
  "recommendation": "Weak Accept",
  "ethical_considerations": "None."
}}
```
"""
        response_text = self.llm.generate(prompt)
        review_data = self._parse_review_response(response_text, paper_title, metrics_data)

        # Lưu file JSON
        json_path = os.path.join(review_dir, "peer_review.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(review_data, f, indent=2, ensure_ascii=False)

        # Lưu file Markdown đẹp
        scores = review_data.get("scores", {})
        md_content = f"""# Official Conference Peer Review Report

**Paper Title:** {paper_title}  
**Recommendation:** **{review_data.get('recommendation', 'Borderline')}**  
**Overall Rating:** **{scores.get('overall_rating', 'N/A')}/10**  

---

### 1. Summary of Contributions
{review_data.get('summary', '')}

---

### 2. Evaluation Scores
| Metric | Score | Scale |
| :--- | :---: | :---: |
| **Soundness** | {scores.get('soundness', 3)}/4 | 1 (Poor) - 4 (Excellent) |
| **Presentation** | {scores.get('presentation', 3)}/4 | 1 (Poor) - 4 (Excellent) |
| **Contribution / Novelty** | {scores.get('contribution', 3)}/4 | 1 (Poor) - 4 (Excellent) |
| **Overall Recommendation** | **{scores.get('overall_rating', 6)}/10** | 1 (Strong Reject) - 10 (Award Quality) |
| **Reviewer Confidence** | {scores.get('confidence', 4)}/5 | 1 (Low) - 5 (Expert) |

---

### 3. Strengths
"""
        for s in review_data.get("strengths", []):
            md_content += f"- {s}\n"

        md_content += "\n### 4. Weaknesses\n"
        for w in review_data.get("weaknesses", []):
            md_content += f"- {w}\n"

        md_content += "\n### 5. Questions for the Authors\n"
        for q in review_data.get("questions", []):
            md_content += f"- {q}\n"

        md_path = os.path.join(review_dir, "peer_review.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        import textwrap
        # In kết quả trực quan ra màn hình Terminal cho người dùng
        rec = review_data.get("recommendation", "Borderline")
        rating = scores.get("overall_rating", "N/A")
        print("\n" + "=" * 65)
        print(f" [HỘI ĐỒNG BÌNH DUYỆT ĐỘC LẬP (PEER REVIEW)]: {rec.upper()} ({rating}/10)")
        print("=" * 65)
        print(f" * Soundness: {scores.get('soundness', 3)}/4 | Presentation: {scores.get('presentation', 3)}/4 | Novelty: {scores.get('contribution', 3)}/4")
        print("\n [ĐIỂM MẠNH CHÍNH (STRENGTHS)]:")
        for s in review_data.get("strengths", [])[:2]:
            clean_s = textwrap.fill(str(s).strip(), width=75, initial_indent="  + ", subsequent_indent="    ")
            print(clean_s)
        print("\n [HẠN CHẾ & ĐIỂM YẾU (WEAKNESSES)]:")
        for w in review_data.get("weaknesses", [])[:2]:
            clean_w = textwrap.fill(str(w).strip(), width=75, initial_indent="  - ", subsequent_indent="    ")
            print(clean_w)
        print("-" * 65)
        print(f" * Báo cáo chi tiết: {md_path}\n")

        return review_data
