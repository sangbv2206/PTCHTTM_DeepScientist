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


class ExperimentDesigner:
    """
    Tác nhân thiết kế thực nghiệm khoa học (Experiment Designer):
    Xác định giả thuyết kiểm định, bộ dữ liệu (dataset/benchmark), baselines và metrics.
    """

    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def design(self, problem_info: Dict[str, Any], method_info: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""You are an elite empirical AI researcher designing experiments for an ICLR/NeurIPS paper.

PROBLEM:
{problem_info.get('problem')}

METHOD:
Name: {method_info.get('method_name')} ({method_info.get('abbreviation')})
Formulation: {method_info.get('mathematical_formulation')}
Procedure: {json.dumps(method_info.get('algorithmic_procedure', []))}
""" + """
### TASK:
Design a rigorous, realistic, and executable experimental setup to validate the method.
Specify:
1. Core Empirical Hypotheses.
2. Baselines for comparison (e.g. Standard Baseline, Variant A, Variant B).
3. Quantitative Evaluation Metrics (e.g. Accuracy, F1, Loss, Latency, Convergence Steps).
4. Ablation Study structure (e.g. w/o key component).
5. Visualizations needed (Figure 1: Convergence/Learning Dynamics, Figure 2: Comparative Performance Benchmark, Figure 3: Ablation Study / Sensitivity Analysis).

Respond strictly in valid JSON format:
```json
{
  "hypotheses": ["H1: ...", "H2: ..."],
  "baselines": ["Baseline-Vanilla", "SOTA-Competitive", "Proposed"],
  "metrics": ["Accuracy (%)", "Loss", "Convergence Epochs"],
  "figure_descriptions": {
    "figure_1": "Learning curves / loss dynamics comparing Proposed method against Baselines over training epochs",
    "figure_2": "Quantitative benchmark comparison of Proposed method vs Baselines on empirical metrics",
    "figure_3": "Ablation study and parameter sensitivity / error distribution analysis"
  },
  "simulation_specification": "Instructions on creating a synthetic or benchmark evaluation environment"
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
                "hypotheses": ["Proposed method achieves lower loss and faster convergence."],
                "baselines": ["Standard Baseline", "Ablated Model", "Proposed Method"],
                "metrics": ["Loss", "Accuracy (%)", "Generalization Score"],
                "figure_descriptions": {
                    "figure_1": "Training and validation convergence curves over epochs",
                    "figure_2": "Performance benchmark comparison vs baselines",
                    "figure_3": "Ablation study and sensitivity analysis"
                },
                "simulation_specification": "Evaluate on multi-dimensional synthetic representations."
            }


class CoderAgent:
    """
    Tác nhân lập trình tự động (Coder Agent, kế thừa triết lý Aider của Sakana AI):
    Tự động viết mã Python thực nghiệm hoàn chỉnh, có thể chạy trực tiếp trên Windows (CPU hoặc GPU),
    xuất file số liệu `metrics.json`, `notes.txt` và biểu đồ trực quan `.png`.
    """

    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def generate_code(
        self,
        problem_info: Dict[str, Any],
        method_info: Dict[str, Any],
        experiment_design: Dict[str, Any],
        error_context: str = ""
    ) -> str:
        prompt = f"""You are an expert Python machine learning engineer and research scientist.
Your task is to write a clean, robust, self-contained Python script named `run_experiment.py`.

### SCIENTIFIC SPECIFICATION:
Problem: {problem_info.get('problem')}
Method: {method_info.get('method_name')} ({method_info.get('abbreviation')})
Procedure: {json.dumps(method_info.get('algorithmic_procedure', []))}
Baselines: {json.dumps(experiment_design.get('baselines', []))}
Metrics: {json.dumps(experiment_design.get('metrics', []))}

### STRICT EXECUTION REQUIREMENTS:
1. **Self-Contained & Lightweight**:
   - Use standard scientific libraries: `numpy`, `torch` (or pure numpy/scipy), `matplotlib`, `json`, `os`.
   - The script must run seamlessly and quickly (< 30 seconds) on standard Windows hardware without external API calls or giant datasets.
   - Use synthetic/benchmark data generator (e.g. multi-class synthetic clusters, noisy regression, or embedded vectors) that mirrors the scientific task.
2. **Execute Baselines vs Proposed Method**:
   - Implement the baseline methods and the proposed method according to the scientific procedure.
   - Train/run iterations and log metrics across epochs/steps.
3. **Mandatory Outputs**:
   The script MUST save the following files in the CURRENT WORKING DIRECTORY:
   - `metrics.json`: JSON dictionary with numerical results (e.g. test metrics for each baseline, final scores, statistics).
   - `notes.txt`: Human-readable summary of experiment findings and numerical takeaways.
   - `figure_1.png`: Clean, publication-quality Matplotlib plot (DPI 300, Seaborn/academic style) showing convergence curves / loss dynamics over epochs from real training logs.
   - `figure_2.png`: Clean, publication-quality Matplotlib plot (DPI 300, bar chart or scatter) showing the comparative evaluation vs baselines using actual empirical test metrics.
   - `figure_3.png`: Clean, publication-quality Matplotlib plot (DPI 300) showing ablation study / parameter sensitivity or error distribution.
4. **No Interactive Windows**:
   - Use `matplotlib.use('Agg')` at the very top so no GUI window pops up.
"""
        if error_context:
            prompt += f"""
### FIX RUNTIME ERROR:
Previous execution crashed with this error:
{error_context}

Please analyze the traceback, fix all syntax and runtime issues, and provide the updated complete script.
"""

        prompt += """
### RESPONSE FORMAT:
Provide the Python code inside a single ```python ... ``` markdown code block. Do NOT include extraneous conversational filler.
"""
        response_text = self.llm.generate(prompt)
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", response_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = response_text.strip()
        return code

    def modify_template_code(
        self,
        base_code: str,
        method_info: Dict[str, Any],
        experiment_design: Dict[str, Any],
        error_context: str = ""
    ) -> str:
        prompt = f"""You are an elite PyTorch engineer and research scientist implementing a proposed AI innovation.

### EXISTING BASELINE EXPERIMENT CODE:
```python
{base_code}
```

### PROPOSED METHOD TO IMPLEMENT (from ResearchAgent):
Method Name: {method_info.get('method_name')} ({method_info.get('method_abbr', '')})
Algorithmic Steps: {method_info.get('method', '')}
Mathematical Formulation: {method_info.get('mathematical_formulation', '')}
Theoretical Rationale: {method_info.get('method_rationale', '')}

### INSTRUCTIONS:
Upgrade the baseline code by integrating the proposed method:
1. Modify the PyTorch neural network architecture or loss function to include the proposed mechanism.
2. Keep the script executable with `--out_dir=run_0` flag.
3. Ensure it runs smoothly and saves `final_info.json` and `history.json` into `--out_dir`.
4. Return the COMPLETE, executable Python script.
"""
        if error_context:
            prompt += f"""
### RUNTIME ERROR TO FIX:
{error_context}
Please fix all errors and return the working script.
"""
        prompt += "\nProvide the updated Python code inside a single ```python ... ``` block."
        response_text = self.llm.generate(prompt)
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response_text.strip()

    def adapt_or_generate_plot_code(
        self,
        base_plot_code: str,
        problem_info: Dict[str, Any],
        method_info: Dict[str, Any],
        experiment_design: Dict[str, Any],
        metrics_data: Dict[str, Any]
    ) -> str:
        """
        Tùy biến hoặc sinh mới script plot.py để đảm bảo figure_1.png, figure_2.png và figure_3.png
        phản ánh CHÍNH XÁC dữ liệu thực nghiệm thật, domain, tên phương pháp, và các metric của đề tài,
        TUYỆT ĐỐI KHÔNG HARDCODE SỐ LIỆU GIẢ HOẶC DÁN NHẦM HÌNH CHÉO DOMAIN.
        """
        try:
            prompt = f"""You are an elite scientific data visualization expert and Python engineer.
Review and write a complete, standalone `plot.py` script that reads REAL EXPERIMENT DATA and generates 3 publication-grade figures (`figure_1.png`, `figure_2.png`, `figure_3.png`).

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
   - STRICT PROHIBITION: You MUST NEVER invent fake ablation scores using hardcoded subtraction like [score, score - 4.5, score - 8.2] or [val, val - 0.042, val - 0.065]!
   - If true multi-run ablation data is not logged in history.json, DO NOT generate a fake ablation bar chart. Instead, design Figure 3 around REAL domain-specific data: residual error distributions, actual activation curves, feature importance, or prediction trajectory overlays!

2. **DOMAIN-AWARE VISUALIZATIONS (Tailored to {problem_info.get('problem', '')})**:
   - For Neural Architectures / SciML / KAN:
     * Figure 1: Convergence trajectory (Train & Val Loss) with clean scientific grid.
     * Figure 2: Ground Truth vs Predicted function approximation overlay $y(x)$ vs $\\hat{{y}}(x)$, or parity scatter plot with $R^2$ annotation.
     * Figure 3: Parameter efficiency (Loss vs Number of Parameters) or Error Residual Distribution (Histogram with fitted KDE).
   - For Causal Inference / Healthcare / Observational Studies:
     * Figure 1: Propensity score overlap or Training convergence.
     * Figure 2: Covariate Balance (Love Plot of ASMD before vs after weighting).
     * Figure 3: Estimated Average Treatment Effect (ATE) with 95% confidence intervals.
   - For Time-Series Forecasting:
     * Figure 1: Training & validation convergence loss.
     * Figure 2: Multi-step forecasting test horizon vs Ground Truth.
     * Figure 3: Residual error distribution across forecast lookback windows.

3. **DYNAMIC DATA INGESTION**:
   - The script must dynamically inspect `run_1/history.json`, `run_0/history.json`, and `run_1/final_info.json` (or `metrics.json`).
   - Use actual recorded keys from `metrics_data`: {json.dumps(metrics_data)}.

4. **CONVERGENCE PLOTTING & EPOCH ALIGNMENT**:
   - `history.json` contains arrays like `"train_loss"` and `"val_loss"`. It does NOT have a separate `"epochs"` key.
   - ALWAYS derive the epochs array from `train_loss`:
     `epochs = list(range(1, len(history.get("train_loss", [])) + 1))`
   - NEVER create an empty or mismatched `epochs` array. Ensure `len(epochs) == len(train_loss)`. If plotting fallback synthetic curves, ensure `len(epochs_list) == len(train_loss)`.

5. **TECHNICAL CONSTRAINTS**:
   - Place `import matplotlib; matplotlib.use("Agg")` at the very top.
   - Save figures as `figure_1.png`, `figure_2.png`, `figure_3.png` with `dpi=300` and `plt.tight_layout()`.
   - Never pop up GUI windows.

Provide ONLY the executable Python script inside a single ```python ... ``` block.
"""
            response_text = self.llm.generate(prompt)
            match = re.search(r"```(?:python)?\s*(.*?)\s*```", response_text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return response_text.strip()
        except Exception as e:
            print(f"[CoderAgent] Lưu ý khi tùy biến plot code: {e}")
            return base_plot_code



class ExperimentRunner:
    """
    Điều phối chạy mã thực nghiệm, tự động kiểm tra lỗi và gọi CoderAgent sửa lỗi (Self-healing loop).
    """

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
            print(f"[ExperimentRunner] Đang sinh mã thực nghiệm (Lần thử {attempt + 1}/{self.max_fix_attempts})...")
            code = self.coder.generate_code(
                problem_info=problem_info,
                method_info=method_info,
                experiment_design=experiment_design,
                error_context=error_context
            )

            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            print(f"[ExperimentRunner] Đang chạy {script_path}...")
            try:
                result = subprocess.run(
                    [sys.executable, "run_experiment.py"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    # Kiểm tra xem các file kết quả đã được sinh ra chưa
                    has_metrics = os.path.exists(os.path.join(work_dir, "metrics.json"))
                    has_fig1 = os.path.exists(os.path.join(work_dir, "figure_1.png"))
                    has_fig2 = os.path.exists(os.path.join(work_dir, "figure_2.png"))
                    has_fig3 = os.path.exists(os.path.join(work_dir, "figure_3.png"))

                    if has_metrics and has_fig1 and has_fig2:
                        print("[ExperimentRunner] Thực nghiệm chạy THÀNH CÔNG! Đã tạo metrics và biểu đồ trực quan.")
                        metrics_data = {}
                        try:
                            with open(os.path.join(work_dir, "metrics.json"), "r", encoding="utf-8") as mf:
                                metrics_data = json.load(mf)
                        except Exception:
                            pass
                        return True, result.stdout, metrics_data
                    else:
                        error_context = f"Script ran with exit code 0 but missing files. has_metrics={has_metrics}, has_fig1={has_fig1}, has_fig2={has_fig2}, has_fig3={has_fig3}. Stdout: {result.stdout}"
                else:
                    error_context = f"Exit code {result.returncode}.\nStderr: {result.stderr}\nStdout: {result.stdout}"
                    print(f"[ExperimentRunner] Lỗi thực thi: {result.stderr[:200]}...")

            except subprocess.TimeoutExpired:
                error_context = "Script execution timed out after 120 seconds. Make the simulation lighter and faster!"
            except Exception as e:
                error_context = f"Execution exception: {str(e)}"

        return False, error_context, {}
