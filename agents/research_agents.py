import json
import re
from typing import Dict, Any, List, Optional
from core.gemini_llm import GeminiLLM


def extract_json_safely(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        raw_json = match.group(1)
    else:
        first_b = text.find("{")
        last_b = text.rfind("}")
        if first_b != -1 and last_b != -1:
            raw_json = text[first_b:last_b + 1]
        else:
            raw_json = ""
    try:
        return json.loads(raw_json)
    except Exception:
        return default


def safe_float(val: Any, default: float = 4.0) -> float:
    """Chuyển đổi an toàn giá trị điểm từ LLM sang float, loại bỏ hoàn toàn nguy cơ lỗi format specifier."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        match = re.search(r"[-+]?\d*\.?\d+", val)
        if match:
            try:
                return float(match.group(0))
            except Exception:
                pass
    return default


def sanitize_validator_output(data: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
    """Đảm bảo mọi trường điểm số trong validator luôn là kiểu float chuẩn."""
    if not isinstance(data, dict):
        data = default.copy()

    scores = data.get("scores")
    if not isinstance(scores, dict):
        scores = default.get("scores", {}).copy()

    cleaned_scores = {}
    for k, v in scores.items():
        cleaned_scores[k] = safe_float(v, 4.0)
    data["scores"] = cleaned_scores

    if "average_score" in data and data["average_score"] is not None:
        data["average_score"] = safe_float(data["average_score"], 4.0)
    elif cleaned_scores:
        data["average_score"] = float(sum(cleaned_scores.values()) / len(cleaned_scores))
    else:
        data["average_score"] = 4.0

    if "feedbacks" not in data or not data["feedbacks"]:
        data["feedbacks"] = default.get("feedbacks", "Proceed with caution.")

    return data


# =========================================================================
# 1. TẦNG 1: PROBLEM FORMULATION (ProblemIdentifier + ProblemValidator)
# =========================================================================

class ProblemIdentifier:
    """Tác nhân xác định bài toán khoa học nguyên bản của ResearchAgent."""
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def run(self, context: Dict[str, Any], feedback: str = "") -> Dict[str, Any]:
        user_topic = context.get("user_topic") or context.get("topic") or ""
        target_paper = context.get("paper", {})
        references = context.get("references", [])
        entities = context.get("entities", [])

        ref_str = "\n".join([f"- {r.get('title')} ({r.get('year', 'N/A')})" for r in references[:5]])
        ent_str = ", ".join(entities[:20])

        prompt = f"""You are an elite AI research scientist whose primary goal is to formulate a key, novel, and rigorous scientific research problem.

### MANDATORY RESEARCH ANCHOR (USER REQUEST):
The user explicitly requested to conduct research on the following topic:
"{user_topic}"

CRITICAL INSTRUCTION:
Your formulated research problem MUST directly, unambiguously, and fundamentally investigate "{user_topic}".
The target reference study and citations below provide academic background, technical inspiration, or baseline methods.
You MUST NEVER abandon, replace, or drift away from "{user_topic}" toward the unrelated topic of the reference papers!

### TARGET REFERENCE STUDY:
Title: {target_paper.get('title')}
Abstract: {target_paper.get('abstract', 'N/A')}

### RELATED STUDIES (CITATIONS):
{ref_str}

### SCIENTIFIC KNOWLEDGE ENTITIES:
{ent_str}
"""
        if feedback:
            prompt += f"\n### REVIEW FEEDBACK FROM PREVIOUS VALIDATION:\n{feedback}\nPlease refine the problem accordingly.\n"

        prompt += """
### TASK:
Formulate a rigorous research problem that is original, clear, feasible, relevant, and strictly centered on the Mandatory User Research Anchor.
Respond strictly in JSON format:
```json
{
  "problem": "Clear statement of the research problem directly addressing the user topic (1-2 sentences)",
  "problem_rationale": "Comprehensive explanation of why this problem matters, what prior literature misses, and the core scientific gap (2-3 paragraphs)"
}
```
"""
        res = self.llm.generate(prompt)
        default = {
            "problem": f"Investigating and Advancing {user_topic}",
            "problem_rationale": f"Current literature lacks a unified, rigorous approach to solving key challenges in {user_topic}."
        }
        return extract_json_safely(res, default)


class ProblemValidator:
    """Tác nhân phản biện bài toán khoa học qua 6 tiêu chí: Topic Fidelity, Clarity, Relevance, Originality, Feasibility, Significance."""
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_topic = context.get("user_topic") or context.get("topic") or ""
        problem = context.get("problem", "")
        rationale = context.get("problem_rationale", "")

        prompt = f"""You are an elite scientific reviewer assessing a proposed research problem across 6 dimensions.

MANDATORY USER TOPIC ANCHOR:
"{user_topic}"

PROPOSED PROBLEM:
{problem}

RATIONALE:
{rationale}

EVALUATION CRITERIA:
1. Topic Fidelity (1-5): Does the proposed problem directly address "{user_topic}"? If the problem has drifted into an unrelated topic (e.g. hijacked by the seed paper), you MUST give Topic Fidelity <= 2.5!
2. Clarity (1-5): Is the problem precisely defined?
3. Relevance (1-5): Is it grounded in and important to the literature?
4. Originality (1-5): Is it a non-trivial, novel scientific inquiry?
5. Feasibility (1-5): Can it be practically studied and tested?
6. Significance (1-5): Will solving it advance the field?
""" + """
Respond strictly in JSON format:
```json
{
  "scores": {
    "topic_fidelity": 4.8,
    "clarity": 4.5,
    "relevance": 4.5,
    "originality": 4.0,
    "feasibility": 4.5,
    "significance": 4.2
  },
  "average_score": 4.41,
  "feedbacks": "Constructive critique highlighting specific improvements needed"
}
```
"""
        res = self.llm.generate(prompt)
        default = {
            "scores": {"topic_fidelity": 4.0, "clarity": 4.0, "relevance": 4.0, "originality": 4.0, "feasibility": 4.0, "significance": 4.0},
            "average_score": 4.0,
            "feedbacks": "Solid formulation, consider tightening the theoretical scope."
        }
        parsed = extract_json_safely(res, default)
        return sanitize_validator_output(parsed, default)


class MethodDeveloper:
    """Tác nhân xây dựng phương pháp giải quyết nguyên bản của ResearchAgent."""
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def run(self, context: Dict[str, Any], feedback: str = "") -> Dict[str, Any]:
        user_topic = context.get("user_topic") or context.get("topic") or ""
        problem = context.get("problem", "")
        rationale = context.get("problem_rationale", "")
        entities = context.get("entities", [])
        template_name = context.get("template_name", "PyTorch Model")

        prompt = f"""You are an elite AI research scientist developing a novel, mathematically rigorous method.

### MANDATORY RESEARCH ANCHOR (USER TOPIC):
"{user_topic}"

RESEARCH PROBLEM:
{problem}

RATIONALE:
{rationale}

RELEVANT SCIENTIFIC ENTITIES:
{', '.join(entities[:15])}

EXECUTION TEMPLATE CONSTRAINT:
The method will be implemented and trained within the '{template_name}' PyTorch architecture.

CRITICAL REQUIREMENT:
The method name, formulation, and algorithmic steps MUST directly and specifically solve the problem for "{user_topic}".
Do NOT drift into unrelated physical, biological, or mathematical domains!
"""
        if feedback:
            prompt += f"\n### PREVIOUS METHOD VALIDATION FEEDBACK:\n{feedback}\nPlease refine the method to resolve these issues.\n"

        prompt += """
### TASK:
Develop a complete method specifying:
1. Method Name (Academic, professional, clearly reflecting the user topic and technical innovation).
2. Method Abbreviation (3-6 capital letters).
3. Formal Mathematical Formulation (variables, objective function, loss components).
4. Algorithmic Procedure (step-by-step).
5. Theoretical Rationale explaining why it solves the problem.

Respond strictly in JSON format:
```json
{
  "method_name": "Full descriptive name of the proposed method",
  "method_abbr": "ABBR",
  "method": "Step-by-step algorithmic description of how the method operates",
  "mathematical_formulation": "Formal LaTeX equations and loss functions: e.g. \\mathcal{L} = \\mathcal{L}_{task} + \\alpha \\mathcal{L}_{reg}",
  "method_rationale": "Theoretical justification of how this method mathematically overcomes the baseline limitations"
}
```
"""
        res = self.llm.generate(prompt)
        default = {
            "method_name": f"Adaptive Framework for {user_topic}",
            "method_abbr": "AFU",
            "method": "Harmonizes representation gradients using adaptive layer-wise normalization.",
            "mathematical_formulation": r"\mathcal{L} = \mathcal{L}_{task} + \lambda \mathcal{L}_{reg}",
            "method_rationale": f"Directly optimizes the objective for {user_topic}."
        }
        return extract_json_safely(res, default)


class MethodValidator:
    """Tác nhân phản biện phương pháp qua 6 tiêu chí: Topic Fidelity, Clarity, Soundness, Novelty, Feasibility, Effectiveness."""
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_topic = context.get("user_topic") or context.get("topic") or ""
        method = context.get("method", "")
        formulation = context.get("mathematical_formulation", "")
        rationale = context.get("method_rationale", "")

        prompt = f"""You are a senior algorithmic reviewer assessing a proposed scientific methodology across 6 dimensions.

MANDATORY USER TOPIC ANCHOR:
"{user_topic}"

PROPOSED METHOD:
{method}

FORMULATION:
{formulation}

RATIONALE:
{rationale}

EVALUATION CRITERIA:
1. Topic Fidelity (1-5): Does the method directly target and solve "{user_topic}"? If it invents unrelated concepts, score <= 2.5!
2. Clarity (1-5): Is the procedure clear and unambiguous?
3. Soundness (1-5): Is the mathematical reasoning theoretically sound?
4. Novelty (1-5): Does it introduce meaningful technical novelty?
5. Feasibility (1-5): Can it be realistically implemented in PyTorch?
6. Effectiveness (1-5): Is it well-suited to solve the underlying problem?
""" + """
Respond strictly in JSON format:
```json
{
  "scores": {
    "topic_fidelity": 4.8,
    "clarity": 4.5,
    "soundness": 4.2,
    "novelty": 4.0,
    "feasibility": 4.8,
    "effectiveness": 4.3
  },
  "average_score": 4.43,
  "feedbacks": "Constructive critique to improve mathematical rigor"
}
```
"""
        res = self.llm.generate(prompt)
        default = {
            "scores": {"topic_fidelity": 4.0, "clarity": 4.0, "soundness": 4.0, "novelty": 4.0, "feasibility": 4.0, "effectiveness": 4.0},
            "average_score": 4.0,
            "feedbacks": "The methodology is sound. Ensure hyperparameters are tuned carefully."
        }
        parsed = extract_json_safely(res, default)
        return sanitize_validator_output(parsed, default)


# =========================================================================
# 3. TẦNG 3: EXPERIMENT DESIGN (ExperimentDesigner + ExperimentValidator)
# =========================================================================

class ExperimentDesigner:
    """Tác nhân thiết kế thực nghiệm nguyên bản của ResearchAgent."""
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def run(self, context: Dict[str, Any], feedback: str = "") -> Dict[str, Any]:
        user_topic = context.get("user_topic") or context.get("topic") or ""
        problem = context.get("problem", "")
        method_name = context.get("method_name", "")
        method = context.get("method", "")
        formulation = context.get("mathematical_formulation", "")

        prompt = f"""You are an elite empirical AI scientist designing rigorous experiments for a top conference paper.

CORE TOPIC: {user_topic}
PROBLEM: {problem}
PROPOSED METHOD: {method_name}
METHOD PROCEDURE: {method}
MATH: {formulation}

CRITICAL: Design baselines, evaluation metrics, and ablation studies that directly measure performance on '{user_topic}'.
"""
        if feedback:
            prompt += f"\n### PREVIOUS EXPERIMENT VALIDATION FEEDBACK:\n{feedback}\nPlease improve the experimental design.\n"

        prompt += """
### TASK:
Design a comprehensive experiment protocol including:
1. Hypotheses to test.
2. Baselines (vanilla baseline, competitive baseline, proposed method).
3. Quantitative Evaluation Metrics.
4. Ablation study configurations.

Respond strictly in JSON format:
```json
{
  "experiment": "Detailed description of the experimental protocol, datasets, and training procedure",
  "experiment_rationale": "Justification of why these specific baselines, metrics, and ablations rigorously validate the method",
  "baselines": ["Vanilla-Baseline", "Competitive-SOTA", "Proposed-Method"],
  "metrics": ["Accuracy (%)", "Loss", "Convergence Steps", "F1 Score"],
  "ablations": ["w/o Regularization", "w/o Adaptive Weighting"]
}
```
"""
        res = self.llm.generate(prompt)
        default = {
            "experiment": "Train models on graph benchmark, evaluate test accuracy and loss curves over 50 epochs.",
            "experiment_rationale": "Directly isolates the contribution of the proposed architectural modifications.",
            "baselines": ["Standard Model", "Ablated Variant", "Proposed Method"],
            "metrics": ["Accuracy (%)", "Validation Loss", "Training Epochs"],
            "ablations": ["Without Auxiliary Loss"]
        }
        return extract_json_safely(res, default)


class ExperimentValidator:
    """Tác nhân phản biện thực nghiệm qua 5 tiêu chí: Clarity, Validity, Robustness, Feasibility, Reproducibility."""
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        experiment = context.get("experiment", "")
        rationale = context.get("experiment_rationale", "")

        prompt = f"""You are a senior meta-reviewer evaluating an experimental design across 5 dimensions:
1. Clarity (1-5): Is the protocol clear, unambiguous, and fully specified?
2. Validity (1-5): Do the metrics and tests truly validate the core hypotheses?
3. Robustness (1-5): Does the design protect against trivial random seed variations?
4. Feasibility (1-5): Can the experiments be executed efficiently in PyTorch?
5. Reproducibility (1-5): Can another researcher reproduce the setup from the description?

EXPERIMENTAL PROTOCOL:
{experiment}

RATIONALE:
{rationale}
""" + """
Respond strictly in JSON format:
```json
{
  "scores": {
    "clarity": 4.5,
    "validity": 4.4,
    "robustness": 4.2,
    "feasibility": 4.7,
    "reproducibility": 4.3
  },
  "average_score": 4.42,
  "feedbacks": "Thorough experimental setup, recommended to include standard deviation across runs."
}
```
"""
        res = self.llm.generate(prompt)
        default = {
            "scores": {"clarity": 4.0, "validity": 4.0, "robustness": 4.0, "feasibility": 4.0, "reproducibility": 4.0},
            "average_score": 4.0,
            "feedbacks": "Experiment protocol is well structured."
        }
        parsed = extract_json_safely(res, default)
        return sanitize_validator_output(parsed, default)
