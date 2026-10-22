import json
import os
import re
import subprocess
import sys
from typing import Dict, Any, List, Tuple

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
