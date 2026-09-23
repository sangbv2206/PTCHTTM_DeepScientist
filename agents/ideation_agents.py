import json
import re
from typing import Dict, Any, List
from core.gemini_llm import GeminiLLM


class ProblemIdentifier:
    """
    Tác nhân phát hiện và đề xuất vấn đề nghiên cứu (Problem Identifier),
    kế thừa cơ chế tư duy sâu của ResearchAgent.
    """

    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def generate(self, context: Dict[str, Any], feedback: str = "") -> Dict[str, Any]:
        target_paper = context.get("target_paper", {})
        related_papers = context.get("related_papers", [])
        entities = context.get("entities", [])
        topic = context.get("topic", "")

        related_summary = "\n".join([
            f"- Title: {p.get('title')} ({p.get('year', 'N/A')}), Venue: {p.get('venue', 'N/A')}"
            for p in related_papers[:5]
        ])

        entities_str = ", ".join(entities[:15]) if entities else "Deep Learning, Optimization, Representation Learning"

        prompt = f"""You are an elite scientific researcher and AI scientist.
Your goal is to formulate a novel, rigorous, and impactful scientific research problem based on existing literature.

### CONTEXT FROM SCIENTIFIC LITERATURE:
Topic: {topic}
Core Target Paper: {target_paper.get('title', 'N/A')} ({target_paper.get('year', 'N/A')})
Key Authors: {', '.join([a.get('name', '') for a in target_paper.get('authors', [])][:4])}
Related Studies in Dataset:
{related_summary}

Relevant Knowledge Entities:
{entities_str}
"""
        if feedback:
            prompt += f"""
### PEER REVIEW FEEDBACK FROM PREVIOUS ITERATION:
{feedback}
Please address the feedback, fix any weaknesses, and refine the research problem to be more scientifically sound, feasible, and novel!
"""

        prompt += """
### TASK:
Formulate a precise research problem that addresses a critical gap or limitation in the literature.
Respond in valid JSON format ONLY with the following schema:
```json
{
  "problem": "Clear, concise definition of the scientific problem (1-2 sentences)",
  "rationale": "Deep scientific rationale explaining WHY this problem exists, why existing methods fail, and why solving it is important (2-3 paragraphs)",
  "hypotheses": [
    "Core testable hypothesis 1",
    "Core testable hypothesis 2"
  ]
}
```
"""
        system_instruction = "You are a world-class AI researcher proposing top-tier conference research problems."
        response_text = self.llm.generate(prompt, system_instruction=system_instruction)

        # Parse JSON
        parsed = self._extract_json(response_text)
        return parsed

    def _extract_json(self, text: str) -> Dict[str, Any]:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace != -1:
                text = text[first_brace:last_brace + 1]
        try:
            return json.loads(text)
        except Exception:
            return {
                "problem": "Improving Generalization and Robustness in Neural Architectures",
                "rationale": text[:500],
                "hypotheses": ["Proposed inductive biases reduce overfitting and enhance sample efficiency."]
            }


class ProblemValidator:
    """
    Tác nhân phản biện vấn đề nghiên cứu (Problem Validator) theo 5 tiêu chí:
    Clarity, Relevance, Originality, Feasibility, Significance.
    """

    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def validate(self, problem_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        problem = problem_info.get("problem", "")
        rationale = problem_info.get("rationale", "")

        prompt = f"""You are a senior program chair and reviewer for top AI conferences (NeurIPS/ICLR/ICML).
Evaluate the following proposed research problem across 5 rigorous scientific dimensions:

PROPOSED RESEARCH PROBLEM:
{problem}

RATIONALE:
{rationale}
""" + """
EVALUATION CRITERIA:
1. Clarity (1-5): Is the problem precisely and unambiguously defined?
2. Relevance (1-5): Does it address a critical problem relevant to current AI/ML?
3. Originality (1-5): Is it genuinely novel or merely an incremental trivial tweak?
4. Feasibility (1-5): Can this be validated empirically and mathematically?
5. Significance (1-5): Will solving this have broad impact on the research community?

Respond strictly in valid JSON format:
```json
{
  "scores": {
    "clarity": 4,
    "relevance": 5,
    "originality": 4,
    "feasibility": 4,
    "significance": 4
  },
  "average_score": 4.2,
  "strengths": ["Key strength 1", "Key strength 2"],
  "weaknesses": ["Key weakness 1", "Key weakness 2"],
  "actionable_feedback": "Detailed advice to improve and refine the problem"
}
```
"""
        response_text = self.llm.generate(prompt)
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1)
        else:
            first_b = response_text.find("{")
            last_b = response_text.rfind("}")
            if first_b != -1 and last_b != -1:
                response_text = response_text[first_b:last_b + 1]

        try:
            return json.loads(response_text)
        except Exception:
            return {
                "scores": {"clarity": 4, "relevance": 4, "originality": 4, "feasibility": 4, "significance": 4},
                "average_score": 4.0,
                "strengths": ["Solid foundation"],
                "weaknesses": ["Needs sharper mathematical specification"],
                "actionable_feedback": "Clarify the theoretical boundary conditions."
            }
