import json
import os
import re
import subprocess
import sys
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from core.gemini_llm import GeminiLLM

def extract_json_from_text(text: str) -> Dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        target = match.group(1)
    else:
        first_b = text.find("{")
        last_b = text.rfind("}")
        target = text[first_b:last_b + 1] if (first_b != -1 and last_b != -1) else ""
    return json.loads(target)

class ExperimentDesigner:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def design(self, problem_info: Dict[str, Any], method_info: Dict[str, Any]) -> Dict[str, Any]:
        prob_desc = problem_info.get("problem", "AI Research Problem")
        method_name = method_info.get("method_name", "Proposed Method")
        abbr = method_info.get("abbreviation") or method_info.get("method_abbr") or "Ours"
        formulation = method_info.get("mathematical_formulation", "")
        procedure = method_info.get("algorithmic_procedure", [])

        prompt = f"""You are an elite empirical AI scientist designing an experiment protocol for a top-tier paper.

### RESEARCH TOPIC & METHOD:
Problem / Domain: {prob_desc}
Proposed Method: {method_name} ({abbr})
Mathematical Formulation: {formulation}
Procedure: {json.dumps(procedure)}

### TASK:
Design a rigorous, realistic, and executable experimental setup tailored strictly to this specific domain and method.
Do NOT force deep learning epoch concepts if the topic is non-iterative (e.g. graph algorithms, statistics, causal inference, optimization).

Specify:
1. Core Empirical Hypotheses to validate.
2. Meaningful Baselines for comparison (Vanilla Baseline, Competitive SOTA, Proposed Method).
3. Quantitative Evaluation Metrics that make sense for this problem.
4. Essential Visualizations (2 to 4 figures depending on complexity, e.g. primary benchmark comparison, algorithm dynamics, ablation/parameter sensitivity, error residual distribution).
5. Lightweight Simulation Specification (must run in < 30 seconds on a standard machine using synthetic/benchmark data).

### RESPONSE FORMAT (Valid JSON only):
```json
{{
  "hypotheses": ["H1: ...", "H2: ..."],
  "baselines": ["Standard Baseline", "Competitive Model", "{abbr}"],
  "metrics": ["Metric_1", "Metric_2"],
  "figure_specifications": [
    {{"id": "figure_1", "title": "Performance Comparison", "description": "Quantitative benchmark comparison against baselines", "plot_type": "bar"}},
    {{"id": "figure_2", "title": "Empirical Dynamics", "description": "Convergence or progress trajectory over steps/iterations", "plot_type": "line"}},
    {{"id": "figure_3", "title": "Ablation and Sensitivity", "description": "Component contribution or parameter sensitivity analysis", "plot_type": "line/bar/scatter"}}
  ],
  "simulation_specification": "Description of the lightweight evaluation setup"
}}
```
"""
        response_text = self.llm.generate(prompt)
        try:
            data = extract_json_from_text(response_text)
            if "figure_descriptions" not in data and "figure_specifications" in data:
                data["figure_descriptions"] = {
                    fig.get("id", f"figure_{idx+1}"): fig.get("description", fig.get("title", ""))
                    for idx, fig in enumerate(data.get("figure_specifications", []))
                }
            return data
        except Exception as e:
            print(f"[ExperimentDesigner] WARNING: Failed to parse LLM JSON ({e}). Falling back to context-aware defaults.")
            return {
                "hypotheses": [f"{abbr} outperforms standard baselines on efficiency and primary objective."],
                "baselines": [f"Baseline-{prob_desc[:25]}", f"{abbr} (Proposed)"],
                "metrics": ["Primary Objective Score", "Relative Improvement (%)", "Runtime (s)"],
                "figure_specifications": [
                    {"id": "figure_1", "title": "Performance Comparison", "description": "Comparison against baselines", "plot_type": "bar"},
                    {"id": "figure_2", "title": "Empirical Dynamics", "description": "Convergence or parameter sensitivity", "plot_type": "line"},
                    {"id": "figure_3", "title": "Ablation Analysis", "description": "Component contribution analysis", "plot_type": "bar"}
                ],
                "figure_descriptions": {
                    "figure_1": "Comparison against baselines",
                    "figure_2": "Convergence or parameter sensitivity",
                    "figure_3": "Component contribution analysis"
                },
                "simulation_specification": f"Lightweight synthetic simulation evaluating {prob_desc} under controlled parameters."
            }

class CoderAgent:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def generate_code(
        self,
        problem_info: Dict[str, Any],
        method_info: Dict[str, Any],
        experiment_design: Dict[str, Any],
        error_context: str = ""
    ) -> str:
        prompt = f"""You are an expert scientific Python engineer.
Write a clean, self-contained Python script named `run_experiment.py`.

### RESEARCH CONTEXT:
Problem: {problem_info.get('problem')}
Method: {method_info.get('method_name')} ({method_info.get('abbreviation') or method_info.get('method_abbr', '')})
Baselines: {json.dumps(experiment_design.get('baselines', []))}
Metrics: {json.dumps(experiment_design.get('metrics', []))}
Desired Figures: {json.dumps(experiment_design.get('figure_specifications') or experiment_design.get('figure_descriptions', []))}

### STRICT REQUIREMENTS:
1. **Lightweight & Fast**: Must run in < 30 seconds on CPU. Use synthetic/reproducible data.
2. **Domain-Appropriate Execution**:
   - If iterative (e.g. neural nets, RL): track progress over iterations.
   - If non-iterative (e.g. tree models, optimization, causal, graph): compute actual comparative metrics directly. Do NOT invent fake epoch loops if unnecessary.
3. **Mandatory Outputs in Current Directory**:
   - `metrics.json`: Dictionary containing numerical results for all baselines and metrics.
   - `notes.txt`: Brief summary of scientific takeaways.
   - Publication-grade figures: Save Matplotlib plots with DPI 300 (`figure_1.png`, `figure_2.png`, `figure_3.png`, etc.) matching the experimental design.
4. **Headless Execution**: Use `import matplotlib; matplotlib.use('Agg')` at the very top.
"""
        if error_context:
            prompt += f"\n### FIX RUNTIME ERROR:\n{error_context}\nPlease fix the traceback and return the complete corrected script.\n"

        prompt += "\nReturn ONLY the executable Python script inside a single ```python ... ``` block."
        res = self.llm.generate(prompt)
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", res, re.DOTALL)
        return match.group(1).strip() if match else res.strip()

    def modify_template_code(
        self,
        base_code: str,
        method_info: Dict[str, Any],
        experiment_design: Dict[str, Any],
        error_context: str = ""
    ) -> str:
        prompt = f"""You are an elite PyTorch engineer upgrading a baseline experiment script with a novel scientific method.

### BASELINE CODE:
```python
{base_code}
```

### NOVEL METHOD TO IMPLEMENT:
Name: {method_info.get('method_name')} ({method_info.get('method_abbr', '')})
Mathematical Formulation: {method_info.get('mathematical_formulation', '')}
Procedure: {method_info.get('method', '')}

### INSTRUCTIONS:
1. Integrate the proposed method into the architecture or training logic.
2. Keep the script executable with `--out_dir=run_0` / `--out_dir=run_1`.
3. Save `final_info.json` and `history.json` into `--out_dir`.
4. Ensure fast execution and zero syntax errors.
5. Hardware Agnostic: Support both CPU and CUDA safely. Use `device = "cuda" if torch.cuda.is_available() else "cpu"`. NEVER hardcode `device = "cuda"` without checking `torch.cuda.is_available()`.
"""
        if error_context:
            prompt += f"\n### RUNTIME ERROR TO FIX:\n{error_context}\n"
        prompt += "\nProvide the updated Python code inside a single ```python ... ``` block."
        res = self.llm.generate(prompt)
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", res, re.DOTALL)
        return match.group(1).strip() if match else res.strip()

    def adapt_or_generate_plot_code(
        self,
        base_plot_code: str,
        problem_info: Dict[str, Any],
        method_info: Dict[str, Any],
        experiment_design: Dict[str, Any],
        metrics_data: Dict[str, Any]
    ) -> str:
        prompt = f"""You are an elite scientific data visualization expert and Python engineer.
Review and write a complete, standalone `plot.py` script that reads REAL EXPERIMENT DATA (`run_1/history.json`, `run_0/history.json`, `metrics.json`) and generates publication-grade figures (`figure_1.png`, `figure_2.png`, etc.).

### RESEARCH CONTEXT:
Topic / Problem: {problem_info.get('problem', '')}
Proposed Method: {method_info.get('method_name', '')} ({method_info.get('method_abbr', '')})
Baselines: {json.dumps(experiment_design.get('baselines', []))}
Metrics to Report: {json.dumps(experiment_design.get('metrics', []))}
Empirical Metrics Obtained: {json.dumps(metrics_data)}

### EXISTING TEMPLATE PLOT CODE (Reference):
```python
{base_plot_code}
```

### STRICT DATA-GROUNDING & PUBLICATION REQUIREMENTS:
1. **NO FAKE OR HARDCODED ABLATION MATH**:
   - STRICT PROHIBITION: You MUST NEVER invent fake ablation scores using hardcoded subtraction.
   - If true multi-run ablation data is not logged in history.json, design figures around REAL domain-specific data: residual distributions, actual activations, feature importance, or prediction trajectory overlays.
2. **DOMAIN-AWARE VISUALIZATIONS**:
   - Tailor plots to {problem_info.get('problem', '')}.
   - Do NOT assume every model has training epochs if not found in history logs.
3. **CONVERGENCE & DYNAMIC EPOCH DERIVATION**:
   - If `train_loss` is in `history.json`, always derive epochs safely:
     `epochs = list(range(1, len(history.get("train_loss", [])) + 1))`
4. **TECHNICAL CONSTRAINTS**:
   - `import matplotlib; matplotlib.use("Agg")` at the top.
   - Save figures as `figure_1.png`, `figure_2.png`, `figure_3.png`, etc. with `dpi=300` and `plt.tight_layout()`.

Provide ONLY the executable Python script inside a single ```python ... ``` block.
"""
        try:
            res = self.llm.generate(prompt)
            match = re.search(r"```(?:python)?\s*(.*?)\s*```", res, re.DOTALL)
            return match.group(1).strip() if match else res.strip()
        except Exception as e:
            print(f"[CoderAgent] Warning during plot adaptation: {e}")
            return base_plot_code


class ExperimentRunner:
    def __init__(self, coder: CoderAgent, max_fix_attempts: int = 3):
        self.coder = coder
        self.max_fix_attempts = max_fix_attempts

    def run(
        self,
        work_dir: str,
        problem_info: Dict[str, Any],
        method_info: Dict[str, Any],
        experiment_design: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        os.makedirs(work_dir, exist_ok=True)
        # Copy generic budgeted LLM helper into work_dir if an API key is configured
        helper_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "exp_llm_helper.py")
        if os.path.exists(helper_src) and (os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEYS") or os.getenv("GOOGLE_API_KEY")):
            import shutil
            shutil.copy2(helper_src, os.path.join(work_dir, "llm_client.py"))
        script_path = os.path.join(work_dir, "run_experiment.py")

        error_context = ""
        for attempt in range(self.max_fix_attempts):
            print(f"[ExperimentRunner] Generating experiment code (Attempt {attempt + 1}/{self.max_fix_attempts})...")
            code = self.coder.generate_code(
                problem_info=problem_info,
                method_info=method_info,
                experiment_design=experiment_design,
                error_context=error_context
            )

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            print(f"[ExperimentRunner] Executing {script_path}...")
            try:
                result = subprocess.run(
                    [sys.executable, "run_experiment.py"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    metrics_path = os.path.join(work_dir, "metrics.json")
                    has_metrics = os.path.exists(metrics_path)
                    generated_figures = [f for f in os.listdir(work_dir) if f.endswith(".png")]

                    if has_metrics and len(generated_figures) >= 1:
                        print(f"[ExperimentRunner] Experiment SUCCESS: metrics.json and {len(generated_figures)} figures generated ({generated_figures}).")
                        metrics_data = {}
                        try:
                            with open(metrics_path, "r", encoding="utf-8") as mf:
                                metrics_data = json.load(mf)
                        except Exception:
                            pass
                        return True, result.stdout, metrics_data
                    else:
                        error_context = (
                            f"Script ran with exit code 0 but missing files. "
                            f"has_metrics={has_metrics}, generated_figures_count={len(generated_figures)}. "
                            f"Stdout: {result.stdout}"
                        )
                else:
                    error_context = f"Exit code {result.returncode}.\nStderr: {result.stderr}\nStdout: {result.stdout}"
                    print(f"[ExperimentRunner] Execution failed: {result.stderr[:200]}...")

            except subprocess.TimeoutExpired:
                error_context = "Script execution timed out after 120 seconds. Make the simulation lighter and faster!"
            except Exception as e:
                error_context = f"Execution exception: {str(e)}"

        return False, error_context, {}


# =========================================================================
# ZERO-SHOT EXPERIMENT SYNTHESIS (Template-free)
# Dùng khi kho template không có "phòng thí nghiệm" phù hợp với đề tài.
# =========================================================================

ZERO_SHOT_OUTPUT_CONTRACT = """
### OUTPUT CONTRACT (the pipeline reads these files; all paths relative to the current working directory):
1. `run_0/final_info.json`: flat JSON dict {metric_name: float} for the PRIMARY BASELINE.
2. `run_1/final_info.json`: flat JSON dict {metric_name: float} for the PROPOSED METHOD (same metric keys as run_0).
3. `all_results.json`: {method_label: {metric_name: float}} for EVERY evaluated method
   (all baselines, the proposed method, and ablation variants). Use human-readable labels.
4. `run_0/history.json` and `run_1/history.json`: dict of lists with per-step / per-iteration / per-sample-bucket traces
   that are meaningful for this task (e.g. accuracy per hop count, cost per iteration, loss per epoch).
5. High-resolution, publication-grade figures saved as `figure_1.png`, `figure_2.png`, ... `figure_N.png` (dpi=300, tight_layout).
   The number of figures N is DYNAMIC: generate as many figures as needed to comprehensively reflect the empirical validation
   (matching all quantitative charts from the seed paper if provided, or covering all core hypotheses, ablations, parameter sensitivity,
   and resource trade-offs). At least 2 figures are required.
   Each figure MUST be built ONLY from numbers computed in this execution. Choose plot types that suit the task.
6. `experiment_config.json` with keys:
   "dataset_info" (string: what data was used and how it was built),
   "data_source" ("synthetic" or "real:<name>"),
   "hyperparameters" (list of [name, value] pairs actually used in the code),
   "method_labels" ({"run_0": "...", "run_1": "..."}),
   "figure_captions" ({"figure_1.png": "...", "figure_2.png": "...", ... for ALL generated figures}).
"""

ZERO_SHOT_INTEGRITY_RULES = """
### SCIENTIFIC INTEGRITY RULES (violations make the paper invalid):
- NEVER hardcode, copy, or tune metric values to match any reported numbers. All metrics MUST be computed by running the methods.
- The baseline and the proposed method MUST be evaluated on exactly the same data with exactly the same evaluation function.
- NEVER add artificial bonuses, noise reductions, or special cases that favor the proposed method.
- Fix all random seeds (numpy, random, torch if used) for reproducibility.
- If the proposed method does not beat the baseline, report that honestly; do not change the evaluation to make it win.
"""


class ZeroShotExperimentRunner:
    """
    Sinh – chạy – tự sửa một kịch bản thực nghiệm độc lập (`run_experiment.py`) cho đề tài bất kỳ,
    tuân thủ hợp đồng đầu ra mà pipeline (ClaimLedger, UniversalPlotter, LaTeXWriter) đọc được.
    """

    def __init__(self, llm: GeminiLLM, max_attempts: int = 3, timeout_sec: int = 300):
        self.llm = llm
        self.max_attempts = max_attempts
        self.timeout_sec = timeout_sec

    def _build_prompt(
        self,
        context: Dict[str, Any],
        seed_block: str,
        reference_code: str,
        previous_code: str,
        error_context: str,
        data_manifest: Optional[Dict[str, Any]] = None,
    ) -> str:
        research_mode = context.get("research_mode", "empirical")
        if research_mode == "reproduce":
            mode_text = (
                "REPRODUCTION MODE: `run_1` must faithfully implement the seed paper's method following its algorithm steps. "
                "`run_0` and additional entries in `all_results.json` are the seed paper's baselines."
            )
        elif research_mode == "empirical":
            focal_name = context.get("focal_method_name") or (context.get("baselines") or ["Focal Method"])[-1]
            base_name = (context.get("baselines") or ["Standard Baseline"])[0]
            mode_text = (
                f"EMPIRICAL COMPARATIVE STUDY MODE (NO invented novel method): You are conducting a fair, controlled benchmark "
                f"of established existing methods from the literature.\n"
                f"- `run_0/final_info.json` = primary baseline ({base_name}).\n"
                f"- `run_1/final_info.json` = primary focal paradigm ({focal_name}).\n"
                f"- `all_results.json` = ALL evaluated methods (including any additional baselines/variants) with their exact published names."
            )
        else:
            mode_text = (
                "INNOVATION MODE: `run_1` is the proposed method described below; `run_0` is the primary baseline, "
                "and `all_results.json` includes all baselines and ablations."
            )

        data_manifest = data_manifest or {}
        if data_manifest.get("local_path"):
            data_block = f"""### VERIFIED REAL DATASET AVAILABLE ON DISK (MANDATORY TO USE)
- Source: `{data_manifest.get('data_source')}` ({data_manifest.get('dataset_name')})
- File path (relative to cwd): `{data_manifest.get('local_path')}` (JSON Lines, 1 JSON object per line)
- Number of rows: {data_manifest.get('num_samples')}
- Columns: {json.dumps(data_manifest.get('columns', []))}
- Sample rows (truncated):
```json
{json.dumps(data_manifest.get('sample_preview', []), indent=2, ensure_ascii=False)}
```
- MANDATORY DATA RULE: Load ALL rows from `{data_manifest.get('local_path')}` for evaluation! Set `"data_source": "{data_manifest.get('data_source')}"` and `"dataset_info": "{data_manifest.get('dataset_name')}"` in `experiment_config.json`. NEVER ignore this file or replace it with toy hardcoded examples."""
        else:
            data_block = (
                "### DATASET INSTRUCTION\n"
                f"- Data manifest note: {data_manifest.get('disclosure_note', 'No external dataset file found.')}\n"
                "- Check if any file exists under `./data/dataset.*`. Otherwise build a structured benchmark matching the task, "
                "and honestly set `\"data_source\": \"synthetic\"` in `experiment_config.json`."
            )

        fig_specs = context.get("figure_specifications") or []
        fig_plan_str = json.dumps(fig_specs, indent=2) if fig_specs else "Dynamically choose 2-5 figures matching the research questions."

        prompt = f"""You are an expert research engineer. Write ONE self-contained Python script `run_experiment.py`
that empirically evaluates the methods on real benchmark data (if provided) and generates verified metrics & figures.

### RESEARCH CONTEXT
Topic: {context.get('user_topic') or context.get('topic')}
Problem / Research Questions: {context.get('problem', '')}
Research Questions: {json.dumps(context.get('research_questions', []))}
Study / Method Overview: {context.get('method_name', '')} ({context.get('method_abbr', '')})
Evaluated Methods Spec: {json.dumps(context.get('evaluated_methods', []), indent=2) if context.get('evaluated_methods') else context.get('method', '')}
Formulation: {context.get('mathematical_formulation', '')}
Planned Methods / Baselines: {json.dumps(context.get('baselines', []))}
Planned Metrics: {json.dumps(context.get('metrics', []))}
Planned Sweeps / Ablations: {json.dumps(context.get('ablations', []))}

### MODE
{mode_text}

{data_block}
"""
        if seed_block:
            prompt += f"\n### SEED PAPER DIGEST (reference for algorithm structure, baselines, metrics, and figures where feasible)\n{seed_block}\n"

        has_llm_helper = bool(
            os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEYS") or os.getenv("GOOGLE_API_KEY")
        )
        llm_engine_rule = (
            "- EXECUTION ENGINE (TOPIC-ADAPTIVE — CHOOSE WHAT FITS THE TASK):\n"
            "  * For LLM-dependent tasks (e.g. RAG, QA, multi-hop reasoning, text summarization, prompting, agent workflows): "
            "`llm_client.py` is available right in the working directory (`from llm_client import call_llm`). "
            "It returns `{'text': str, 'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int, 'latency_ms': float}` "
            "using the configured Gemini model with automatic rate-limiting, caching, and a hard call budget (`max_calls=120`). "
            "Use `SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)` (or `TfidfVectorizer`) for retrieval, and call `call_llm(...)` "
            "for genuine decomposition / reasoning / concise answer synthesis on a clean, statistically meaningful evaluation slice (`N >= 12`, ideally `N=12-16` samples for primary comparison; NEVER use tiny `N < 8` slices like `N=4`!). "
            "REUSE per-sample evaluation traces from the main loop to compute category/difficulty/hop breakdown figures instead of wasting extra LLM calls, and reserve extra calls for 1-2 parameter sweeps (e.g., `N=6-8` samples across `top_k=[2, 4, 6]`), keeping total calls <= 115. Measure REAL tokens and latency from `call_llm`! "
            "ONLY evaluate and name the actual model/backbone you executed (never fabricate rows/columns for unexecuted models like GPT-4o, Claude, or Llama by scaling Gemini's scores!).\n"
            "  * For non-LLM ML/DL tasks (Vision, Tabular, Time-Series, RL, Classical Graph/Optimization): do NOT import `llm_client`; run real `torch`, `scikit-learn`, or `scipy` models directly on the full dataset (or `N >= 50`)."
            if has_llm_helper
            else "- NO external LLM API key configured: use `SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)`, `torch`, `scikit-learn`, `scipy`, or deterministic algorithms."
        )

        prompt += f"""
### EXECUTION CONSTRAINTS
- Must finish in under {self.timeout_sec // 60} minutes on a laptop CPU (cap evaluation slice appropriately and report exact N evaluated in `dataset_info`).
- NO extra network downloads (pip/huggingface downloads) inside `run_experiment.py`. Allowed libraries ONLY: Python stdlib, numpy, scipy, pandas, scikit-learn, torch, matplotlib, `sentence_transformers` (`local_files_only=True`), and `llm_client` (when applicable above). NEVER import `seaborn` or uninstalled packages!
{llm_engine_rule}
- CRITICAL EVALUATION ACCURACY CHECK:
  * Inspect the exact structure of the ground-truth answers in `{data_manifest.get('local_path') or 'the dataset'}`!
  * For QA tasks (e.g., HotpotQA/SQuAD), answers are short entities or yes/no phrases. Instruct `call_llm` (or extractor) to return ONLY the exact short answer phrase (e.g. "Answer with ONLY the exact short entity, date, or yes/no, with no explanation"), strip trailing punctuation, and compute Exact Match (EM) and token-level F1 properly. Print 3 sample (question, gold_answer, predicted_answer) tuples to stdout.
  * STRICT ZERO-LEAKAGE & METHOD DIFFERENTIATION RULE:
    - Prediction/retrieval methods MUST NEVER receive, read, or inspect `sample["answer"]`, `gold`, or `ground_truth` during inference. Never use `random.random() < p` to simulate accuracy, failures, or metric differences!
    - EVERY evaluated method MUST execute its OWN genuinely distinct algorithm and produce distinct retrieved contexts / features / prompts / predictions (e.g., for RAG: Standard RAG retrieves `top_k=2` chunks via single-shot TF-IDF/MiniLM and makes 1 LLM call; GraphRAG expands an entity/title co-occurrence graph across passages; Iterative/Triplet RAG decomposes the query into sub-queries/triplets via a 1st LLM call, performs a 2nd retrieval hop, and synthesizes the final answer!). NEVER implement 3-4 methods that all just run identical `TfidfVectorizer(top_k=3)` + 1 `call_llm` with slightly reworded prompt prefixes!
    - For category/difficulty/hop breakdown figures: group samples by REAL metadata fields present in `data/dataset.jsonl` (e.g., `level`, `type`, `category`, `difficulty`) with `>= 5` samples per group (NEVER arbitrarily slice `eval_samples[:2]` and `eval_samples[2:]`!).
  * STRICT ANTI-FAKE-SWEEP, ANTI-HARDCODE & NO-SYNTHETIC-DISTRIBUTION RULE (MANDATORY FOR ALL TOPICS):
    - NEVER use `np.random.normal`, `np.random.uniform`, `np.random.beta`, `random.gauss`, or `random.uniform` to generate or simulate evaluation metrics, accuracies, TPR/TNR, latencies, or pass rates! All metrics MUST come from REAL execution (`call_llm`, `subprocess` running Python code/tests with `timeout=2`, `sklearn`, `torch`, etc.) on the actual samples in `data/dataset.jsonl`.
    - For code / test-case tasks: if `data/dataset.jsonl` contains `solutions` / `incorrect_solutions`, inspect the `language` array (e.g., Python3 = 3 or `'PYTHON'`/`'PYTHON3'`) and actually execute generated test inputs against real Python solutions via `subprocess.run([sys.executable, "-c", ...], input=..., timeout=2)` to measure genuine TPR and TNR. IMPORTANT: Use `call_llm(...)` to actually generate new test inputs from the problem description (for iterative/feedback methods, pass execution feedback to `call_llm` to refine tests), and evaluate ALL generated test inputs (never slice `test_inputs[:3]` in a way that drops the newly generated tests!).
    - NEVER compute sweep/ablation/difficulty curves using arithmetic scaling formulas (e.g., `acc * decay` or `max(0.1, base * factor)`). Every point on every sweep/curve MUST come from actually running the methods on that parameter value or data slice!
    - NEVER hardcode metric/failure arrays (e.g., `failures = [12.5, 8.2, 5.1, 4.0]`) to plot figures. Every plotted number MUST be computed from actual execution!
    - NEVER report or plot unexecuted external models/datasets (e.g. Qwen2.5-7B fine-tuning, LiveCodeBench, GPT-4o, Claude). Only plot what THIS script actually ran!
  * NEVER let all performance metrics collapse to 0.0 across all methods!
- DYNAMIC PUBLICATION FIGURES SPECIFICATION:
  * Planned figures for this topic's RQs:
{fig_plan_str}
  * Prefer multi-point empirical curves/breakdowns (e.g., parameter sensitivity sweep like `top_k` or `budget`, real dataset category/difficulty breakdown, counted failure modes, accuracy-vs-token/latency trade-offs) over single isolated scatter dots.
  * Generate `figure_1.png`, `figure_2.png`, ... dynamically (2 to 5 figures depending on the topic).
  * LAYOUT HYGIENE & INTUITIVE ANNOTATIONS (MANDATORY FOR ALL PLOTS):
    - For bar charts: add numeric value labels directly on top of bars (`ax.bar_label(container, fmt="%.1f", padding=3, fontsize=8)`).
    - For scatter / trade-off plots: add `ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6 + (idx % 3) * 12 - 12), fontsize=8)` directly next to each method point, staggering `xytext` offsets if multiple points have close/identical coordinates so text labels NEVER overlap each other!
    - NEVER let legends overlap plotted data markers, bars, or labels! Place legends OUTSIDE the axes area (`ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True)` or `bbox_to_anchor=(1.02, 1), loc="upper left"`) and save with `plt.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0.25)` so titles and y-axis tick labels are NEVER clipped.
    - For bubble/scatter plots with variable marker sizes (`s=...`), ALWAYS pass `markerscale=0.35` (or fixed-size legend handles) to `ax.legend(...)` so legend icons stay normal-sized and never collide inside the legend box!
    - Add `ax.set_title(..., pad=12)`, `ax.margins(0.15)`, and call `plt.tight_layout()` before saving.
- Use `import matplotlib; matplotlib.use("Agg")` at the very top.
- Print a clear summary table to stdout at the end.
{ZERO_SHOT_OUTPUT_CONTRACT}
{ZERO_SHOT_INTEGRITY_RULES}
"""
        if reference_code:
            prompt += (
                "\n### REFERENCE SCRIPT FROM A PREVIOUSLY SUCCESSFUL, RELATED EXPERIMENT\n"
                "Reuse useful patterns where appropriate, but strictly follow the current dataset and mode.\n"
                f"```python\n{reference_code[:15000]}\n```\n"
            )
        if error_context:
            prompt += (
                "\n### YOUR PREVIOUS ATTEMPT FAILED\n"
                f"```python\n{previous_code[:20000]}\n```\n"
                f"Error / quality-gate violation:\n{error_context[-4000:]}\n"
                "Fix the problem completely and return the corrected script.\n"
            )
        prompt += "\nReturn ONLY the complete script inside a single ```python ... ``` block."
        return prompt

    @staticmethod
    def _extract_code(res: str) -> str:
        blocks = re.findall(r"```(?:python)?\s*(.*?)```", res, re.DOTALL)
        if blocks:
            return max(blocks, key=len).strip()
        return res.strip()

    @staticmethod
    def _check_contract(work_dir: str, data_manifest: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        problems = []
        run_metrics: Dict[str, Dict[str, float]] = {}
        for run in ["run_0", "run_1"]:
            p = os.path.join(work_dir, run, "final_info.json")
            if not os.path.exists(p):
                problems.append(f"missing {run}/final_info.json")
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                numeric = {k: float(v) for k, v in data.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                if not numeric:
                    problems.append(f"{run}/final_info.json has no numeric metrics")
                else:
                    run_metrics[run] = numeric
            except Exception as e:
                problems.append(f"{run}/final_info.json is not valid JSON ({e})")

        # Gate 1: Reject identical or trivially undifferentiated run_0 and run_1 outputs (all topics)
        if "run_0" in run_metrics and "run_1" in run_metrics:
            if run_metrics["run_0"] == run_metrics["run_1"]:
                problems.append("run_0 and run_1 produced 100% identical metrics; methods are not differentiated")
            else:
                shared_perf = [
                    k for k in run_metrics["run_0"]
                    if k in run_metrics["run_1"]
                    and not any(t in k.lower() for t in ("time", "latency", "runtime", "sec", "ms", "fail"))
                ]
                if shared_perf and all(abs(run_metrics["run_0"][k] - run_metrics["run_1"][k]) < 1e-6 for k in shared_perf):
                    problems.append(
                        f"run_0 and run_1 produced IDENTICAL primary performance metrics ({shared_perf})! "
                        "Each method MUST implement its own distinct retrieval/model/prompt logic instead of sharing identical inputs."
                    )

        # Gate 2: Reject degenerate all-zero primary performance metrics
        all_res_path = os.path.join(work_dir, "all_results.json")
        combined_dicts = list(run_metrics.values())
        if os.path.exists(all_res_path):
            try:
                with open(all_res_path, "r", encoding="utf-8") as af:
                    ar = json.load(af)
                if isinstance(ar, dict):
                    for md in ar.values():
                        if isinstance(md, dict):
                            combined_dicts.append({k: float(v) for k, v in md.items() if isinstance(v, (int, float))})
            except Exception:
                pass
        if combined_dicts:
            perf_vals = []
            for md in combined_dicts:
                for k, v in md.items():
                    kl = k.lower()
                    if not any(t in kl for t in ("time", "latency", "token", "cost", "mem", "runtime", "sec", "ms")):
                        perf_vals.append(abs(float(v)))
            if perf_vals and max(perf_vals) < 1e-6:
                problems.append(
                    "ALL primary performance metrics (e.g., EM/F1/Accuracy) are 0.0 across all methods! "
                    "Fix answer extraction / prediction / evaluation logic so methods produce meaningful non-zero scores."
                )

        # Gate 2b: Reject primary-metric collapse across ALL >=3 methods in all_results.json
        if os.path.exists(all_res_path):
            try:
                with open(all_res_path, "r", encoding="utf-8") as af:
                    ar = json.load(af)
                if isinstance(ar, dict) and len(ar) >= 3:
                    method_dicts = [md for md in ar.values() if isinstance(md, dict)]
                    if len(method_dicts) >= 3:
                        core_keys = ("accuracy", "exact match", "f1", "pass@", "pass rate", "tpr", "tnr", "auc", "success", "bleu", "rouge")
                        collapsed = []
                        shared_keys = set.intersection(*(set(md.keys()) for md in method_dicts))
                        for k in sorted(shared_keys):
                            kl = k.lower()
                            if any(ck in kl for ck in core_keys) and not any(t in kl for t in ("time", "latency", "token", "cost", "call", "freq", "fail")):
                                vals = [float(md[k]) for md in method_dicts if isinstance(md.get(k), (int, float)) and not isinstance(md.get(k), bool)]
                                if len(vals) >= 3 and len({round(v, 4) for v in vals}) == 1:
                                    collapsed.append(f"{k}={vals[0]}")
                        if collapsed:
                            problems.append(
                                f"ALL {len(method_dicts)} methods produced IDENTICAL primary accuracy/score metric(s) ({', '.join(collapsed)})! "
                                "This indicates either tiny sample size (e.g. EVAL_N<=4) or undifferentiated method implementations. "
                                "Use N >= 12 samples and ensure each method executes a genuinely distinct retrieval/reasoning/model pipeline."
                            )
            except Exception:
                pass

        # Gate 3: Verify figures exist AND are not blank white images
        figs = sorted([f for f in os.listdir(work_dir) if f.endswith(".png") and os.path.getsize(os.path.join(work_dir, f)) > 1000])
        if len(figs) < 2:
            problems.append(f"only {len(figs)} valid figure(s) generated, need at least 2")
        else:
            try:
                import numpy as np
                import matplotlib.image as mpimg
                for fig in figs:
                    arr = mpimg.imread(os.path.join(work_dir, fig))
                    if float(np.std(arr)) < 0.01:
                        problems.append(f"figure '{fig}' appears nearly blank (std={np.std(arr):.4f})")
            except Exception:
                pass

        # Gate 4: Verify no gold-answer data leakage, fake sweeps, or hardcoded plot arrays (all topics)
        script_path = os.path.join(work_dir, "run_experiment.py")
        if os.path.exists(script_path):
            try:
                import ast
                src_text = open(script_path, "r", encoding="utf-8").read()
                tree = ast.parse(src_text)
                if "random.random()" in src_text and re.search(r'random\.random\(\)\s*[<>]\s*0\.\d+', src_text):
                    problems.append(
                        "SIMULATED PROBABILITY DETECTED (`random.random() < p`): Do NOT simulate predictions, accuracies, or failure counts with random probabilities; compute all metrics and failure counts deterministically from actual execution!"
                    )
                if re.search(r'\b(?:np\.random\.(?:normal|uniform|beta|gamma|binomial|poisson)|random\.(?:gauss|uniform|triangular|normalvariate))\s*\(', src_text):
                    problems.append(
                        "SYNTHETIC DISTRIBUTION SIMULATION DETECTED (`np.random.normal` / `np.random.uniform` / `random.gauss`): "
                        "NEVER fabricate metrics or scores by sampling from random distributions! Compute ALL metrics from REAL execution (call_llm, subprocess code/test execution, sklearn, torch) on data/dataset.jsonl."
                    )
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        cls_src = ast.get_source_segment(src_text, node) or ""
                        if re.search(r'\[["\']answer["\']\]|\["ground_truth"\]|\["label"\]', cls_src):
                            problems.append(
                                f"DATA LEAKAGE DETECTED inside class '{node.name}': methods must NOT access sample['answer'] / ground_truth during inference!"
                            )
                    elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        vname = node.targets[0].id.lower()
                        if vname in ("eval_n", "max_samples", "num_samples", "n_samples", "n_eval") and isinstance(node.value, ast.Constant) and isinstance(node.value.value, int) and not isinstance(node.value.value, bool):
                            if (data_manifest or {}).get("num_samples", 0) >= 10 and node.value.value < 8:
                                problems.append(
                                    f"TINY EVALUATION SAMPLE SIZE DETECTED ({node.targets[0].id}={node.value.value}): "
                                    "Use at least N=12 samples (and >= 5 samples per plotted subgroup) so metrics do not quantize to 0%/50%/100%!"
                                )
                        metric_var_hints = ("failure", "err_rate", "error_rate", "acc_list", "f1_list", "em_list", "scores", "results", "latency_list", "token_list")
                        if any(h in vname for h in metric_var_hints) and isinstance(node.value, ast.List) and len(node.value.elts) >= 3:
                            if all(isinstance(el, ast.Constant) and isinstance(el.value, (int, float)) and not isinstance(el.value, bool) for el in node.value.elts):
                                problems.append(
                                    f"HARDCODED PLOT DATA DETECTED: '{node.targets[0].id}' assigns literal numbers instead of computing them from real runs!"
                                )
                    elif isinstance(node, ast.For):
                        loop_src = ast.get_source_segment(src_text, node) or ""
                        if re.search(r'\*\s*\(\s*1(?:\.0)?\s*[-+]\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\*', loop_src):
                            problems.append(
                                "FAKE SWEEP DETECTED: loop multiplies an evaluated score by a synthetic linear factor `(1.0 - idx * c)` instead of actually varying a real parameter/model!"
                            )
                        elif isinstance(node.target, ast.Name):
                            tname = node.target.id.lower()
                            sweep_hints = ("hop", "noise", "budget", "top_k", "topk", "k_val", "alpha", "lambda", "ratio", "step")
                            if any(h in tname for h in sweep_hints):
                                if re.search(r'\b(decay|penalty|factor)\b|\*\s*\(\s*1\.0\s*-', loop_src) and not re.search(r'\.(run|predict|retrieve|forward|fit|evaluate|retrieve_and_generate|answer|call_llm)\s*\(', loop_src):
                                    problems.append(
                                        f"FAKE SWEEP DETECTED in loop 'for {node.target.id}': computes synthetic curves via arithmetic formulas instead of actually evaluating the methods!"
                                    )
            except Exception:
                pass

        # Gate 5: Verify experiment_config.json and real-data provenance adherence
        cfg_path = os.path.join(work_dir, "experiment_config.json")
        if not os.path.exists(cfg_path):
            problems.append("missing experiment_config.json")
        elif data_manifest and data_manifest.get("local_path"):
            try:
                with open(cfg_path, "r", encoding="utf-8") as cf:
                    cfg = json.load(cf)
                if str(cfg.get("data_source", "")).strip().lower() == "synthetic":
                    problems.append(
                        f"Real dataset '{data_manifest.get('local_path')}' was provided, but experiment_config.json reported data_source='synthetic'. Load and evaluate on '{data_manifest.get('local_path')}'!"
                    )
                # Reject unexecuted external benchmark/model claims in figure captions
                caps_text = json.dumps(cfg.get("figure_captions", {})).lower()
                for forbidden_ext in ("livecodebench", "gpt-4o", "claude 3", "qwen2.5-7b", "llama-3-70b"):
                    if forbidden_ext in caps_text and forbidden_ext not in (open(script_path, "r", encoding="utf-8").read().lower() if os.path.exists(script_path) else ""):
                        problems.append(f"UNEXECUTED EXTERNAL BENCHMARK/MODEL IN FIGURE CAPTION: '{forbidden_ext}' was claimed in figure_captions without being executed!")
            except Exception:
                pass

        return (len(problems) == 0, "; ".join(problems))

    def run(
        self,
        work_dir: str,
        context: Dict[str, Any],
        seed_block: str = "",
        reference_code: str = "",
        env: Dict[str, str] = None,
        data_manifest: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        os.makedirs(work_dir, exist_ok=True)
        # Copy generic budgeted LLM helper into work_dir if an API key is configured
        helper_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "exp_llm_helper.py")
        if os.path.exists(helper_src) and (os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEYS") or os.getenv("GOOGLE_API_KEY")):
            import shutil
            shutil.copy2(helper_src, os.path.join(work_dir, "llm_client.py"))
        script_path = os.path.join(work_dir, "run_experiment.py")
        log_path = os.path.join(work_dir, "zero_shot_attempts.log")
        error_context, code = "", ""

        for attempt in range(1, self.max_attempts + 1):
            print(f"[ZeroShot] Đang sinh mã thực nghiệm (lần {attempt}/{self.max_attempts})...")
            sys.stdout.flush()
            prompt = self._build_prompt(context, seed_block, reference_code, code, error_context, data_manifest=data_manifest)
            code = self._extract_code(self.llm.generate(prompt))
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            print(f"[ZeroShot] Đang chạy run_experiment.py (timeout {self.timeout_sec}s)...")
            sys.stdout.flush()
            try:
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", "run_experiment.py"],
                    cwd=work_dir, capture_output=True, text=True, timeout=self.timeout_sec,
                    env=env, encoding="utf-8", errors="replace",
                )
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\n===== ATTEMPT {attempt} (exit {result.returncode}) =====\n")
                    lf.write("--- STDOUT ---\n" + (result.stdout or "")[-6000:] + "\n")
                    lf.write("--- STDERR ---\n" + (result.stderr or "")[-6000:] + "\n")
                if result.returncode == 0:
                    ok, problems = self._check_contract(work_dir, data_manifest=data_manifest)
                    if ok:
                        print("[ZeroShot] Thực nghiệm chạy thành công và vượt qua toàn bộ cổng kiểm tra chất lượng.")
                        print((result.stdout or "")[-1500:])
                        return True, result.stdout or ""
                    error_context = f"Script exited 0 but failed quality/contract gate: {problems}\nStdout tail:\n{(result.stdout or '')[-2000:]}"
                    print(f"[ZeroShot] Không đạt cổng kiểm tra chất lượng: {problems}")
                else:
                    error_context = f"Exit code {result.returncode}.\nStderr:\n{(result.stderr or '')[-3500:]}"
                    last = (result.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
                    print(f"[ZeroShot] Lỗi khi chạy: {last[0][:200]}")
            except subprocess.TimeoutExpired:
                error_context = f"Script timed out after {self.timeout_sec} seconds. Reduce data size / iterations."
                print("[ZeroShot] Quá thời gian chạy, yêu cầu rút gọn.")
        return False, error_context
