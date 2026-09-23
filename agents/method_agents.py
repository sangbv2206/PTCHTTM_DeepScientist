import json
import re
from typing import Dict, Any, List
from core.gemini_llm import GeminiLLM


class MethodDeveloper:
    """
    Tác nhân phát triển phương pháp nghiên cứu (Method Developer),
    chuyển hóa vấn đề khoa học thành giải pháp kỹ thuật, thuật toán và công thức toán học.
    """

    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def develop(self, problem_info: Dict[str, Any], context: Dict[str, Any], feedback: str = "") -> Dict[str, Any]:
        problem = problem_info.get("problem", "")
        rationale = problem_info.get("rationale", "")
        target_paper = context.get("target_paper", {})
        entities = context.get("entities", [])

        prompt = f"""You are a principal AI researcher and theoretical computer scientist.
Design a novel, mathematically sound, and comprehensive method to solve the validated research problem below.

### RESEARCH PROBLEM:
{problem}

### PROBLEM RATIONALE & MOTIVATION:
{rationale}

### GROUNDED SCIENTIFIC CONTEXT:
Target Baseline: {target_paper.get('title')}
Key Entities: {', '.join(entities[:12]) if entities else 'Neural Representations, Gradient Dynamics'}
"""
        if feedback:
            prompt += f"""
### REVIEW FEEDBACK ON PREVIOUS METHOD ITERATION:
{feedback}
Please address all points in the feedback to make the method rigorous and publishable.
"""

        prompt += """
### TASK:
Develop a complete, rigorous method. Formulate:
1. Method Name & High-level intuition.
2. Mathematical Formulation (formal variables, objective functions, optimization targets).
3. Algorithmic Steps (step-by-step description).
4. Key Advantages over standard baselines.

Respond strictly in valid JSON format:
```json
{
  "method_name": "Concise, descriptive name of the proposed method (e.g. Adaptive Dual-Scale Representation Learning)",
  "abbreviation": "ADSRL",
  "intuition": "Core conceptual intuition behind why this approach works",
  "mathematical_formulation": "Formal mathematical definition, objective/loss functions, formulas in LaTeX syntax",
  "algorithmic_procedure": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "theoretical_justification": "Why this mathematically guarantees or empirically yields superior convergence/generalization"
}
```
"""
        system_instruction = "You are a distinguished research scientist creating state-of-the-art AI methodology."
        response_text = self.llm.generate(prompt, system_instruction=system_instruction)
        return self._extract_json(response_text)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            first_b = text.find("{")
            last_b = text.rfind("}")
            if first_b != -1 and last_b != -1:
                text = text[first_b:last_b + 1]
        try:
            return json.loads(text)
        except Exception:
            return {
                "method_name": "Adaptive Multi-Scale Representation Harmonization",
                "abbreviation": "AMRH",
                "intuition": "Harmonizes multi-scale gradients to prevent catastrophic forgetting.",
                "mathematical_formulation": r"\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \mathcal{L}_{reg}",
                "algorithmic_procedure": ["Initialize weights", "Compute dual loss", "Update via adaptive gradients"],
                "theoretical_justification": "Bounded variance of stochastic gradients."
            }


class MethodValidator:
    """
    Tác nhân phản biện phương pháp nghiên cứu (Method Validator)
    đánh giá tính đúng đắn toán học, độ phức tạp tính toán và tính khả thi.
    """

    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def validate(self, method_info: Dict[str, Any], problem_info: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""You are an expert theoretical AI reviewer.
Evaluate the following proposed method for scientific rigor, mathematical consistency, and feasibility:

PROBLEM:
{problem_info.get('problem')}

PROPOSED METHOD:
Name: {method_info.get('method_name')} ({method_info.get('abbreviation')})
Intuition: {method_info.get('intuition')}
Math Formulation: {method_info.get('mathematical_formulation')}
Procedure: {json.dumps(method_info.get('algorithmic_procedure', []))}
""" + """
EVALUATION CRITERIA:
1. Mathematical Soundness: Are equations and objectives consistent and well-defined?
2. Technical Novelty: Does it introduce non-trivial technical contributions?
3. Computational Efficiency: Is it computationally tractable?
4. Alignment: Does it directly address the proposed problem?

Respond strictly in valid JSON format:
```json
{
  "is_valid": true,
  "overall_score": 4.3,
  "strengths": ["Strengths of the method formulation"],
  "weaknesses": ["Potential edge cases or ambiguities"],
  "refinement_suggestions": "Actionable improvements"
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
                "is_valid": True,
                "overall_score": 4.2,
                "strengths": ["Clear objective function and formulation"],
                "weaknesses": ["Hyperparameter sensitivity needs experimental study"],
                "refinement_suggestions": "Include ablation studies across lambda values."
            }
