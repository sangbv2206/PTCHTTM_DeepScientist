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
        try:
            # Escape raw LaTeX backslashes (e.g. \mathcal, \frac) that break JSON parsing
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw_json)
            fixed = re.sub(r',\s*([\]\}])', r'\1', fixed)
            return json.loads(fixed)
        except Exception:
            return default


def safe_float(val: Any, default: float = 4.0) -> float:
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


def seed_paper_block(context: Dict[str, Any], include_results: bool = False) -> str:
    """Khối mô tả bài báo seed (nếu người dùng cung cấp PDF) để chèn vào prompt của các Agent."""
    digest = context.get("seed_digest")
    if not digest:
        return ""
    from core.paper_reader import render_seed_digest
    mode = context.get("research_mode", "innovate")
    header = (
        "### SEED PAPER (REPRODUCTION TARGET) - reconstruct this work faithfully:\n"
        if mode == "reproduce"
        else "### SEED PAPER (state of the art to build upon and compare against):\n"
    )
    return header + render_seed_digest(digest, include_results=include_results) + "\n"


def execution_constraint_text(context: Dict[str, Any]) -> str:
    template_name = context.get("template_name", "")
    if context.get("execution_mode") == "zero_shot" or not template_name:
        return ("The method will be implemented from scratch as a standalone, offline Python experiment "
                "(numpy / scikit-learn / PyTorch) that runs on a laptop CPU.")
    return f"The method will be implemented and trained within the '{template_name}' PyTorch architecture."


class ProblemIdentifier:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def run(self, context: Dict[str, Any], feedback: str = "") -> Dict[str, Any]:
        user_topic = context.get("user_topic") or context.get("topic") or ""
        target_paper = context.get("paper", {})
        references = context.get("references", [])
        entities = context.get("entities", [])
        research_mode = context.get("research_mode", "empirical")

        ref_str = "\n".join([f"- {r.get('title')} ({r.get('year', 'N/A')})" for r in references[:5]])
        ent_str = ", ".join(entities[:20])

        if research_mode == "empirical":
            role_desc = (
                "You are an elite AI research scientist designing a rigorous empirical and comparative benchmark study. "
                "Do NOT invent a new algorithm; formulate clear comparative Research Questions (RQs) evaluating existing paradigms."
            )
            task_desc = """Formulate a rigorous comparative empirical investigation strictly centered on the Mandatory User Research Anchor:
1. State the core comparative research problem and 3-4 concrete Research Questions (e.g., accuracy vs. resource trade-offs, sensitivity to difficulty/parameters, failure modes).
2. Explain why a controlled, head-to-head empirical evaluation of existing paradigms on this topic is scientifically needed.
MANDATORY LANGUAGE: All outputs (problem, problem_rationale) MUST be written strictly in professional academic English. Under no circumstances should any non-English words be used.
Respond strictly in JSON format:
```json
{
  "problem": "Clear statement of the comparative empirical research problem and core research questions (1-3 sentences)",
  "problem_rationale": "Comprehensive explanation of why comparing existing methods under identical conditions matters and what trade-offs remain unclear (2-3 paragraphs)",
  "research_questions": ["RQ1: ...", "RQ2: ...", "RQ3: ..."]
}
```"""
        else:
            role_desc = "You are an elite AI research scientist whose primary goal is to formulate a key, novel, and rigorous scientific research problem."
            task_desc = """Formulate a rigorous research problem that is original, clear, feasible, relevant, and strictly centered on the Mandatory User Research Anchor.
MANDATORY LANGUAGE: All outputs (problem, problem_rationale) MUST be written strictly in professional academic English. Under no circumstances should any non-English words be used.
Respond strictly in JSON format:
```json
{
  "problem": "Clear statement of the research problem directly addressing the user topic (1-2 sentences)",
  "problem_rationale": "Comprehensive explanation of why this problem matters, what prior literature misses, and the core scientific gap (2-3 paragraphs)"
}
```"""

        prompt = f"""{role_desc}

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

{seed_paper_block(context)}"""
        if feedback:
            prompt += f"\n### REVIEW FEEDBACK FROM PREVIOUS VALIDATION:\n{feedback}\nPlease refine the problem accordingly.\n"

        prompt += f"\n### TASK:\n{task_desc}\n"
        res = self.llm.generate(prompt)
        default = {
            "problem": f"Comparative Empirical Evaluation of Methods for {user_topic}" if research_mode == "empirical" else f"Investigating and Advancing {user_topic}",
            "problem_rationale": f"Current literature lacks a unified, controlled empirical comparison across key trade-offs in {user_topic}."
        }
        return extract_json_safely(res, default)


class ProblemValidator:
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
{execution_constraint_text(context)}

{seed_paper_block(context)}
CRITICAL REQUIREMENT:
The method name, formulation, and algorithmic steps MUST directly and specifically solve the problem for "{user_topic}".
Do NOT drift into unrelated physical, biological, or mathematical domains!
"""
        if context.get("research_mode") == "reproduce" and context.get("seed_digest"):
            prompt += (
                "\nREPRODUCTION MODE: Do NOT invent a new method. Your 'method' must be a faithful, step-by-step "
                "reconstruction of the SEED PAPER's method (keep its original name and abbreviation), and the "
                "formulation must formalize exactly those steps.\n"
            )
        if feedback:
            prompt += f"\n### PREVIOUS METHOD VALIDATION FEEDBACK:\n{feedback}\nPlease refine the method to resolve these issues.\n"

        prompt += """
### TASK:
Develop a complete method specifying:
1. Method Name (Academic, professional, clearly reflecting the user topic and technical innovation in English).
2. Method Abbreviation (3-6 capital letters, strictly derived from the capitalized initials of the English Method Name).
3. Formal Mathematical Formulation (variables, objective function, loss components).
4. Algorithmic Procedure (step-by-step).
5. Theoretical Rationale explaining why it solves the problem.

MANDATORY LANGUAGE: All fields (method_name, method_abbr, method, mathematical_formulation, method_rationale) MUST be written strictly in professional academic English. Under NO circumstances should any non-English words be used.

Respond strictly in JSON format:
```json
{
  "method_name": "Full descriptive English name of the proposed method",
  "method_abbr": "ABBR",
  "method": "Step-by-step algorithmic description of how the method operates",
  "mathematical_formulation": "Formal LaTeX equations and loss functions: e.g. \\mathcal{L} = \\mathcal{L}_{task} + \\alpha \\mathcal{L}_{reg}",
  "method_rationale": "Theoretical justification of how this method mathematically overcomes the baseline limitations"
}
```
"""
        res = self.llm.generate(prompt)
        safe_topic = re.sub(r'[^\x00-\x7F]+', ' ', user_topic).strip() or "Scientific Representation Learning"
        safe_topic = re.sub(r'\s+', ' ', safe_topic)
        full_name = f"Adaptive Framework for {safe_topic}"
        words = re.findall(r"[A-Za-z0-9]+", full_name)
        topic_abbr = "".join([w[0].upper() for w in words if w.lower() not in ["and", "for", "with", "of", "in", "the", "a"]])[:6] or "ADAPT"
        default = {
            "method_name": full_name,
            "method_abbr": topic_abbr,
            "method": "Harmonizes representation gradients using adaptive layer-wise normalization.",
            "mathematical_formulation": r"\mathcal{L} = \mathcal{L}_{task} + \lambda \mathcal{L}_{reg}",
            "method_rationale": f"Directly optimizes the objective for {safe_topic}."
        }
        return extract_json_safely(res, default)


class MethodValidator:
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


class MethodSelector:
    """Selects 2-4 established, published methods from the literature for fair empirical comparison (no invented method)."""

    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_topic = context.get("user_topic") or context.get("topic") or ""
        problem = context.get("problem", "")
        target_paper = context.get("paper", {})
        references = context.get("references", [])
        entities = context.get("entities", [])
        ref_str = "\n".join([f"- {r.get('title')} ({r.get('year', 'N/A')})" for r in references[:8]])

        prompt = f"""You are a senior AI researcher organizing a fair, controlled comparative benchmark study.
DO NOT invent a fake "novel" method or fake acronym. Instead, select 3-4 ESTABLISHED, published methods/paradigms from the literature for "{user_topic}" to evaluate head-to-head.

### TOPIC: "{user_topic}"
### PROBLEM & RESEARCH QUESTIONS:
{problem}

### TARGET PAPER:
Title: {target_paper.get('title')}
Abstract: {target_paper.get('abstract', 'N/A')}

### RELATED LITERATURE:
{ref_str}

### KEY CONCEPTS:
{', '.join(entities[:15])}

{seed_paper_block(context)}

### TASK:
Select 3-4 distinct, real existing paradigms/baselines appropriate for this topic (for example, from simple classical baseline up to the target paradigm described in the topic/seed paper).
MANDATORY: All text MUST be in formal academic English. Never invent non-existent methods or claim "Ours".

Respond strictly in JSON:
```json
{{
  "study_title_phrase": "Comparative Benchmark of <Paradigms> for <Task>",
  "study_abbr": "BENCH",
  "evaluated_methods": [
    {{
      "name": "Published/Standard Method 1",
      "abbr": "M1",
      "role": "baseline",
      "mechanism": "Concise technical explanation of how it works",
      "strengths_hypothesized": "Where it is expected to excel",
      "limitations_hypothesized": "Expected failure modes"
    }},
    {{
      "name": "Primary Paradigm Under Study (e.g. from topic/target paper)",
      "abbr": "M2",
      "role": "focal_paradigm",
      "mechanism": "Concise technical explanation of how it works",
      "strengths_hypothesized": "Where it is expected to excel",
      "limitations_hypothesized": "Expected cost/failure modes"
    }}
  ],
  "method_summary": "Cohesive overview of the evaluated paradigms and how they differ algorithmically (2 paragraphs)",
  "mathematical_formulation": "Formal LaTeX definitions of the evaluated retrieval/prediction scoring rules and evaluation metrics",
  "selection_rationale": "Why these methods span the core architectural spectrum of the topic"
}}
```"""
        res = self.llm.generate(prompt)
        default = {
            "study_title_phrase": f"Empirical Benchmark for {user_topic}",
            "study_abbr": "BENCH",
            "evaluated_methods": [
                {"name": "Standard Baseline", "abbr": "STD", "role": "baseline", "mechanism": "Standard single-step baseline."},
                {"name": "Focal Paradigm", "abbr": "FOCAL", "role": "focal_paradigm", "mechanism": f"Core paradigm investigated in {user_topic}."},
            ],
            "method_summary": f"We systematically evaluate established paradigms addressing {user_topic} under identical data and compute budgets.",
            "mathematical_formulation": r"S(x, y) = \mathrm{eval}(f_{\theta}(x), y)",
            "selection_rationale": "Spans simple classical baselines through structured multi-step paradigms."
        }
        parsed = extract_json_safely(res, default)
        methods = parsed.get("evaluated_methods") or default["evaluated_methods"]
        method_names_list = [m.get("name", "Method") for m in methods if isinstance(m, dict)]
        focal = next((m for m in methods if m.get("role") == "focal_paradigm"), methods[-1])
        return {
            "evaluated_methods": methods,
            "baselines": method_names_list,
            "method_name": parsed.get("study_title_phrase", default["study_title_phrase"]),
            "method_abbr": (parsed.get("study_abbr") or "BENCH")[:8],
            "focal_method_name": focal.get("name", method_names_list[-1]),
            "focal_method_abbr": focal.get("abbr", "FOCAL"),
            "method": parsed.get("method_summary", default["method_summary"]),
            "mathematical_formulation": parsed.get("mathematical_formulation", default["mathematical_formulation"]),
            "method_rationale": parsed.get("selection_rationale", default["selection_rationale"]),
        }


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
        research_mode = context.get("research_mode", "empirical")
        rqs = context.get("research_questions", [])
        evaluated_methods = context.get("evaluated_methods", [])

        if research_mode == "empirical":
            methods_ctx = json.dumps(evaluated_methods, indent=2) if evaluated_methods else method
            header_ctx = f"""You are an elite empirical AI scientist designing a rigorous comparative benchmark protocol.

CORE TOPIC: {user_topic}
RESEARCH PROBLEM & QUESTIONS: {problem}
RESEARCH QUESTIONS: {json.dumps(rqs)}
EVALUATED METHODS (ALL EXISTING PARADIGMS):
{methods_ctx}
METHOD OVERVIEW: {method}
FORMALISM: {formulation}

CRITICAL: Design experiments, domain-authentic metrics, sensitivity sweeps, and DYNAMIC figures (2 to 5 figures tailored strictly to the RQs — never hardcode a fixed number) that evaluate these existing methods fairly on '{user_topic}'.
HARDWARE-FEASIBLE SCOPE (MANDATORY): All experiments MUST be genuinely executable on a single laptop CPU + budgeted Gemini API calls on ONE public dataset. NEVER design experiments requiring 7B+ LLM fine-tuning, external unacquired benchmarks (e.g., LiveCodeBench, ImageNet-22k), or unexecuted closed models (GPT-4o, Claude)."""
        else:
            header_ctx = f"""You are an elite empirical AI scientist designing rigorous experiments for a top conference paper.

CORE TOPIC: {user_topic}
PROBLEM: {problem}
PROPOSED METHOD: {method_name}
METHOD PROCEDURE: {method}
MATH: {formulation}

CRITICAL: Design baselines, evaluation metrics, ablation studies, and DYNAMIC figures (tailored strictly to the task) that directly measure performance on '{user_topic}'."""

        prompt = header_ctx + f"\n\n{seed_paper_block(context)}"
        if feedback:
            prompt += f"\n### PREVIOUS EXPERIMENT VALIDATION FEEDBACK:\n{feedback}\nPlease improve the experimental design.\n"

        prompt += """
### TASK:
Design a comprehensive experiment protocol including:
1. Research Questions / Hypotheses to test.
2. Evaluated Methods / Baselines (exact list of methods to run head-to-head).
3. Quantitative Evaluation Metrics authentic to this domain.
4. Ablation or sensitivity sweeps (e.g., varying top-k, dataset difficulty/hops, noise level, or sample budget).
5. Dynamic Figure Plan (`figure_specifications`): choose 2 to 5 figures whose plot types and axes directly answer the RQs for THIS specific topic (never force unrelated plot types).

Respond strictly in JSON format:
```json
{
  "experiment": "Detailed description of the experimental protocol, dataset usage, and evaluation sweeps",
  "experiment_rationale": "Justification of why these specific methods, metrics, sweeps, and figures rigorously answer the RQs",
  "baselines": ["Method 1", "Method 2", "Method 3"],
  "metrics": ["Metric 1", "Metric 2", "Latency (ms)"],
  "ablations": ["Parameter sensitivity sweep", "Difficulty/complexity breakdown"],
  "figure_specifications": [
    {"id": "figure_1", "title": "...", "description": "...", "plot_type": "grouped_bar|line|scatter|heatmap|boxplot"}
  ]
}
```
"""
        res = self.llm.generate(prompt)
        domain_metrics = context.get("metrics") or []
        if not domain_metrics:
            topic_str = str(context.get("user_topic", "")).lower()
            if any(k in topic_str for k in ["forecast", "series", "stage", "regression"]):
                domain_metrics = ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R^2 Score"]
            elif any(k in topic_str for k in ["graph", "node", "classification"]):
                domain_metrics = ["Node Classification Accuracy (%)", "Cross-Entropy Loss"]
            elif any(k in topic_str for k in ["trial", "causal", "survival", "hazard"]):
                domain_metrics = ["C-Index", "Brier Score", "Average Treatment Effect (ATE)"]
            else:
                domain_metrics = ["Evaluation Metric", "Validation Loss", "Convergence Steps"]

        default = {
            "experiment": f"Train models on standardized benchmark for {context.get('user_topic', 'task')}, evaluate generalization metrics and convergence.",
            "experiment_rationale": "Directly isolates the contribution of the proposed architectural modifications against competitive baselines.",
            "baselines": ["Standard Baseline", "Proposed Method"],
            "metrics": domain_metrics,
            "ablations": ["Without Regularization Component"]
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
