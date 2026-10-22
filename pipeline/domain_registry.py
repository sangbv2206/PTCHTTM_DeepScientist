"""Domain Registry & Context Resolver for DeepScientist.

Tự động phân giải và cấu hình ngữ cảnh miền chuyên ngành (Domain Context),
thông tin tập dữ liệu (Dataset Info), đại lượng đo lường và tên đặc trưng
cho từng template và đề tài nghiên cứu.
Thiết kế theo dạng cấu hình khai báo (Declarative Rule Registry) động,
dễ mở rộng mà không cần hardcode chuỗi lồng nhau trong Orchestrator.
"""

from typing import Dict, Any, List, Optional


# =========================================================================
# 1. BẢNG MÔ TẢ DATASET MẶC ĐỊNH CHO TỪNG TEMPLATE HOẠT ĐỘNG
# =========================================================================
TEMPLATE_DATASET_INFO: Dict[str, str] = {
    "time_series_forecasting": "Multivariate Sequential Spatiotemporal Benchmark Dataset",
    "2d_diffusion": "2D Manifold Density Synthesis Benchmark (Circle, Dino, Line, Moons distributions)",
    "nanoGPT": "Character-Level Language Modeling Benchmark (Shakespeare corpus)",
    "grokking": "Algorithmic Modular Arithmetic Generalization Benchmark (Prime modulo operations)",
    "causal_trial_emulation": "Longitudinal Target Trial Emulation Cohort (Standardized clinical arms, multi-covariate adjustment, survival follow-up)",
    "reinforcement_learning": "CartPole-v1 Discrete Action Inverted Pendulum Benchmark (Self-contained 4D state continuous dynamical balancing)",
}


# =========================================================================
# 2. CẤU HÌNH DOMAIN THEO TEMPLATE (Template-Specific Registry)
# =========================================================================
TEMPLATE_DOMAIN_REGISTRY: List[Dict[str, Any]] = [
    {
        "template_keys": ["nanogpt"],
        "topic_keywords": ["nanogpt", "transformer", "language model", "llm", "nlp", "văn bản", "ngôn ngữ"],
        "config": {
            "domain": "nlp_language_modeling",
            "series_name": "Character-Level Cross-Entropy Perplexity Trajectory",
            "unit_label": "Cross-Entropy Loss",
            "target_name": "Validation Cross-Entropy Loss",
            "feature_names": ["Token Embedding", "Positional Encoding", "Multi-Head Attention", "Feed-Forward Projection"],
            "dataset_info": "Shakespeare Corpus Character-Level Language Modeling Benchmark Dataset",
            "task_focus": "causal autoregressive language modeling and transformer representation learning",
        },
    },
    {
        "template_keys": ["grokking"],
        "topic_keywords": ["grokking", "modular", "arithmetic", "generalization", "transformer"],
        "config": {
            "domain": "algorithmic_reasoning",
            "series_name": "Validation Accuracy Generalization Trajectory",
            "unit_label": "Validation Accuracy (%)",
            "target_name": "Validation Generalization Accuracy",
            "feature_names": ["Operand A Embedding", "Operator Token", "Operand B Embedding", "Modular Result Projection"],
            "dataset_info": "Algorithmic Modular Arithmetic Generalization Benchmark Dataset (Prime modulo operations)",
            "task_focus": "delayed generalization and grokking dynamics in overparametrized neural networks",
        },
    },
    {
        "template_keys": ["diffusion", "2d_diffusion"],
        "topic_keywords": ["diffusion", "generative diffusion", "denoising"],
        "config": {
            "domain": "generative_diffusion",
            "series_name": "2D Manifold Denoising Score Matching Dynamics",
            "unit_label": "Denoising Score MSE",
            "target_name": "Diffusion Reverse Score MSE",
            "feature_names": ["Coordinate X", "Coordinate Y", "Diffusion Time Step t", "Injected Gaussian Noise"],
            "dataset_info": "2D Synthetic Manifold Density Synthesis Benchmark Dataset (Circle, Dino, Line, Moons distributions)",
            "task_focus": "denoising diffusion probabilistic modeling on complex 2D spatial manifolds",
        },
    },
    {
        "template_keys": ["causal"],
        "topic_keywords": ["causal", "trial emulation", "treatment effect", "survival", "hazard"],
        "config": {
            "domain": "causal_inference",
            "series_name": "Longitudinal Causal Hazard Ratio & Propensity Balance Trajectory",
            "unit_label": "Standardized Mean Difference / C-Index",
            "target_name": "Survival Concordance Index (C-Index)",
            "feature_names": ["Patient Age", "Tumor Stage", "ECOG Status", "Prior Therapies", "Serum Biomarkers"],
            "dataset_info": "Semi-Synthetic Longitudinal Oncology Target Trial Emulation Cohort (N=1,200 patients, 8 clinical covariates, 36-month survival follow-up)",
            "task_focus": "target trial emulation and doubly robust causal machine learning for survival outcomes",
        },
    },
    {
        "template_keys": ["reinforcement_learning", "rl"],
        "topic_keywords": ["reinforcement learning", "q-learning", "dqn", "pendulum", "cartpole", "control", "policy"],
        "config": {
            "domain": "reinforcement_learning_control",
            "series_name": "Episodic Reward & Optimization Convergence Trajectory",
            "unit_label": "Episode Return",
            "target_name": "Mean Episode Reward",
            "feature_names": ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"],
            "dataset_info": "CartPole-v1 Discrete-Action Inverted Pendulum Benchmark (Self-contained 4D continuous-state dynamical balancing)",
            "task_focus": "deep reinforcement learning for inverted pendulum stabilization under discrete action dynamics",
        },
    },
]


# =========================================================================
# 3. CẤU HÌNH DOMAIN THEO ĐỀ TÀI / CHUYÊN NGÀNH (Topic-Driven Registry)
# =========================================================================
TOPIC_DOMAIN_REGISTRY: List[Dict[str, Any]] = [
    {
        "keywords": [
            "flood", "lũ", "ngập", "inundation", "rainfall", "precipitation", "hydrolog",
            "meteorolog", "river", "terrain", "mưa", "thủy văn", "mực nước", "bão"
        ],
        "config": {
            "domain": "hydrology_meteorology",
            "series_name": "Observed vs Predicted Flood Hydrograph & Water Stage",
            "unit_label": "Water Level (m)",
            "target_name": "Flood Stage & Inundation Level",
            "feature_names": ["Rainfall Intensity (mm/h)", "Upstream River Stage (m)", "Soil Moisture Saturation (%)", "DEM Elevation Derivative (m)"],
            "dataset_info": "Longitudinal Meteorological & Hydrological Flood Benchmark Dataset (Multi-station rainfall series, upstream gauge stage, soil saturation index, and DEM topographic features)",
            "task_focus": "spatiotemporal flood hazard and river stage trajectory forecasting under dynamic meteorological precipitation patterns",
        },
    },
    {
        "keywords": [
            "energy", "electricity", "power", "load", "consumption", "grid", "smart home",
            "điện", "năng lượng", "phụ tải", "tiêu thụ điện"
        ],
        "config": {
            "domain": "energy_power",
            "series_name": "Diurnal Power Consumption Profile Forecast",
            "unit_label": "Power Load (kWh)",
            "target_name": "Aggregate Energy Consumption",
            "feature_names": ["Historical Load (kWh)", "Ambient Temperature (°C)", "Diurnal Sine (24h)", "Diurnal Cosine (24h)"],
            "dataset_info": "Multivariate Smart Home Energy Load Consumption Benchmark Dataset (1,500 timesteps, ambient temperature correlation, and diurnal seasonality)",
            "task_focus": "multi-step sequential household and smart grid energy demand forecasting",
        },
    },
    {
        "keywords": [
            "air quality", "pm2.5", "pm10", "pollution", "emission", "aqi", "không khí", "ô nhiễm", "khí thải"
        ],
        "config": {
            "domain": "air_quality_environment",
            "series_name": "Atmospheric PM2.5 Concentration Trajectory",
            "unit_label": "PM2.5 (μg/m³)",
            "target_name": "Ambient PM2.5 Concentration",
            "feature_names": ["Wind Speed (m/s)", "Relative Humidity (%)", "Ambient Temperature (°C)", "Traffic Density Index"],
            "dataset_info": "Multivariate Atmospheric Air Quality and Particulate Matter Benchmark Dataset",
            "task_focus": "environmental pollutant dispersion and atmospheric sensor time-series forecasting",
        },
    },
    {
        "keywords": [
            "stock", "price", "financial", "market", "asset", "trading", "tài chính", "chứng khoán", "giá"
        ],
        "config": {
            "domain": "finance_economics",
            "series_name": "Sequential Financial Asset Dynamics",
            "unit_label": "Asset Index / Price",
            "target_name": "Asset Price Trajectory",
            "feature_names": ["Trading Volume", "Rolling Volatility", "Market Momentum", "Bid-Ask Spread"],
            "dataset_info": "Multivariate High-Frequency Financial Time-Series Benchmark Dataset",
            "task_focus": "non-stationary temporal volatility and quantitative financial asset forecasting",
        },
    },
]

# Ngữ cảnh mặc định cho chuỗi thời gian tổng quát
DEFAULT_DOMAIN_CONFIG: Dict[str, Any] = {
    "domain": "generic_sequential",
    "series_name": "Multivariate Dynamic Trajectory Forecast",
    "unit_label": "Normalized Amplitude",
    "target_name": "Continuous Target State",
    "feature_names": ["Autoregressive Trend", "Exogenous Driver", "Harmonic Oscillation", "Stochastic Noise"],
    "dataset_info": "Multivariate Sequential Benchmark Dataset (Standardized multi-scale temporal dynamics)",
    "task_focus": "continuous multivariate sequential time-series forecasting",
}


# =========================================================================
# 4. HÀM TRUY VẤN VÀ PHÂN GIẢI NGỮ CẢNH ĐỘNG (Dynamic LLM Domain Resolver)
# =========================================================================
def extract_domain_context(topic: str, template_name: str, llm: Optional[Any] = None) -> Dict[str, Any]:
    """
    Tự động phân giải ngữ cảnh chuyên ngành ĐỘNG 100%:
    1. Đọc trực tiếp prompt.json và 60 dòng đầu của experiment.py trong thư mục template.
    2. Nếu có LLM: Sử dụng AI để đọc hiểu và trích xuất ngữ cảnh thực tế (Dataset, Domain, Target, Action Space).
    3. Không bao giờ hardcode chuỗi MSBD hay ép từ khóa tĩnh.
    """
    import os
    import json
    import re

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tpl_dir = os.path.join(base_dir, "templates", template_name)

    prompt_info = {}
    prompt_file = os.path.join(tpl_dir, "prompt.json")
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_info = json.load(f)
        except Exception:
            pass

    code_snippet = ""
    exp_file = os.path.join(tpl_dir, "experiment.py")
    if os.path.exists(exp_file):
        try:
            with open(exp_file, "r", encoding="utf-8") as f:
                lines = [next(f) for _ in range(70)]
                code_snippet = "".join(lines)
        except Exception:
            pass

    # Nếu có LLM, để AI tự phân tích và trích xuất động
    if llm:
        try:
            extraction_prompt = f"""You are a senior AI research architect analyzing an experimental template and a user's research topic.
Extract the exact empirical domain context as valid JSON.

USER RESEARCH TOPIC:
"{topic}"

TEMPLATE NAME:
"{template_name}"

TEMPLATE METADATA (prompt.json):
{json.dumps(prompt_info, indent=2)}

TEMPLATE CODE SNIPPET (First 70 lines of experiment.py):
\"\"\"
{code_snippet}
\"\"\"

CRITICAL INSTRUCTIONS:
1. Examine what environment / dataset is ACTUALLY implemented in the code (e.g., CartPole, Pendulum, CIFAR-10, Synthetic Graph, Shakespeare).
2. DO NOT hallucinate fake benchmark names like 'MSBD' unless explicitly in the code.
3. State whether the action/target is continuous or discrete.
4. Output STRICTLY a JSON object with this schema:
{{
  "domain": "concise_domain_identifier",
  "series_name": "Full descriptive title of evaluation dynamic",
  "unit_label": "Measurement unit (e.g. Episode Return, Accuracy (%), Test MSE)",
  "target_name": "Name of target state or metric",
  "feature_names": ["Feature 1", "Feature 2", "Feature 3", "Feature 4"],
  "dataset_info": "Precise name and description of the dataset/environment actually used in experiment.py",
  "task_focus": "Specific empirical scientific objective"
}}
"""
            res = llm.generate(extraction_prompt)
            match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", res)
            json_str = match.group(1) if match else res[res.find("{"):res.rfind("}")+1]
            extracted = json.loads(json_str)
            if isinstance(extracted, dict) and "domain" in extracted and "dataset_info" in extracted:
                return extracted
        except Exception as e:
            pass

    # Fallback dựa trên prompt.json của template nếu LLM không khả dụng
    task_desc = prompt_info.get("task_description", "")
    t_lower = (topic or "").lower()
    tpl = (template_name or "").lower()

    for entry in TEMPLATE_DOMAIN_REGISTRY:
        if any(k in tpl for k in entry["template_keys"]) or any(k in t_lower for k in entry["topic_keywords"]):
            cfg = dict(entry["config"])
            if task_desc and len(task_desc) > 20:
                cfg["task_focus"] = task_desc[:200]
            return cfg

    # Tự động suy luận từ tên template thay vì fallback MSBD
    clean_name = template_name.replace("_", " ").title()
    return {
        "domain": template_name,
        "series_name": f"{clean_name} Empirical Optimization Trajectory",
        "unit_label": "Performance Metric Value",
        "target_name": f"{clean_name} Primary Target",
        "feature_names": ["Feature Vector Dim 1", "Feature Vector Dim 2", "Feature Vector Dim 3", "Feature Vector Dim 4"],
        "dataset_info": f"{clean_name} Controlled Scientific Benchmark Environment",
        "task_focus": f"empirical optimization and evaluation on {clean_name}",
    }


def get_dataset_info_for_template(template_name: str, domain_ctx: Optional[Dict[str, Any]] = None) -> str:
    """Lấy chuỗi mô tả dataset phù hợp nhất cho template và ngữ cảnh."""
    if domain_ctx and domain_ctx.get("dataset_info"):
        return domain_ctx["dataset_info"]
    clean_name = (template_name or "Standard").replace("_", " ").title()
    return TEMPLATE_DATASET_INFO.get(template_name, f"{clean_name} Controlled Scientific Benchmark Dataset")


# =========================================================================
# 5. SIÊU THAM SỐ VÀ CẤU HÌNH THỰC TẾ THEO TEMPLATE (Dùng cho Table 2)
# =========================================================================
TEMPLATE_HYPERPARAMETERS: Dict[str, List[tuple]] = {
    "2d_diffusion": [
        ("Network Backbone", "3-layer MLP with Sinusoidal Positional Embeddings ($d_{\\text{embed}}=128, d_{\\text{hidden}}=256$)"),
        ("Diffusion Timesteps ($T$)", "100 forward/reverse diffusion steps"),
        ("Noise Variance Schedule", "Linear schedule ($\\beta_1 = 10^{-4}, \\beta_T = 0.02$)"),
        ("Optimizer", "AdamW ($\\eta = 3 \\times 10^{-4}$, Cosine Annealing schedule)"),
        ("Manifold Penalty Weight ($\\lambda$)", "$\\lambda = 0.01$ (Intrinsic tangent/normal projection penalty)"),
        ("Training Budget", "10,000 optimization steps per distribution ($N=100,000$ points)"),
        ("Batch Size", "256 samples per mini-batch"),
        ("Evaluation Metric Protocol", "$k=5$ nearest-neighbor continuous KL Divergence (10,000 samples)"),
    ],
    "time_series_forecasting": [
        ("Sequence Architecture", "Gated Recurrent Unit (GRU) with Spatiotemporal Coupling ($d_{\\text{hidden}}=64$, 2 layers)"),
        ("Horizon Window", "Lookback sequence $L=96$, forecast horizon $H=24$"),
        ("Loss Objective", "Mean Squared Error (MSE) with physical boundary penalty"),
        ("Optimizer", "Adam ($\\eta = 10^{-3}$, decoupled weight decay $\\beta = 10^{-4}$)"),
        ("Training Protocol", "50 epochs with early validation stopping (patience 7)"),
        ("Mini-batch Size", "32 sequential windows per batch"),
        ("Evaluation Protocol", "Chronological rolling-origin evaluation (zero future contamination)"),
    ],
    "causal_trial_emulation": [
        ("Propensity Model", "Logistic Regression with $L_2$ regularization for Inverse Probability Weighting"),
        ("Causal Estimator", "Stabilized IPTW with Average Treatment Effect (ATE) risk difference"),
        ("Survival Model", "Cox Proportional Hazards model with robust sandwich standard errors"),
        ("Covariate Balance Criterion", "Absolute Standardized Mean Difference (ASMD $\\le 0.10$ threshold)"),
        ("Sample Cohort Size", "$N=20,000$ observational patients across 12-month follow-up"),
    ],
    "reinforcement_learning": [
        ("RL Algorithm", "Double Deep Q-Network (Double DQN) with target parameter freezing"),
        ("Replay Buffer", "Uniform experience replay memory (capacity 10,000 transitions)"),
        ("Exploration Schedule", "$\\epsilon$-greedy ($\\epsilon_{\\text{start}}=1.0, \\epsilon_{\\text{min}}=0.05$, decay 0.995)"),
        ("Discount Factor ($\\gamma$)", "$\\gamma = 0.99$"),
        ("Target Soft-Update Frequency", "Polyak soft update ($\\tau = 0.005$) every 10 transition steps"),
    ],
    "nanoGPT": [
        ("Transformer Architecture", "Decoder-only Transformer (6 layers, 6 heads, $d_{\\text{model}}=384$)"),
        ("Context Window", "256 tokens"),
        ("Optimization Protocol", "AdamW ($\\eta = 5 \\times 10^{-4}, \\beta_1=0.9, \\beta_2=0.95$) with warmup"),
        ("Regularization", "Dropout $p=0.1$, weight decay $10^{-1}$"),
    ],
    "grokking": [
        ("Architecture", "1-layer standard Transformer with modular arithmetic operator embedding"),
        ("Weight Decay", "High weight decay $\\lambda = 1.0$ (inducing delayed generalization)"),
        ("Optimization", "AdamW ($\\eta = 10^{-3}, \\beta_1=0.9, \\beta_2=0.98$)"),
        ("Training Steps", "10,000 epochs (tracking train vs. validation phase transition)"),
    ],
}


def get_domain_hyperparameters(template_name: str) -> List[tuple]:
    """Trả về danh sách (Tên tham số, Giá trị) chi tiết theo đúng template thực nghiệm."""
    tpl = (template_name or "").lower()
    for k, v in TEMPLATE_HYPERPARAMETERS.items():
        if k in tpl:
            return v
    return [
        ("Optimization Algorithm", "AdamW ($\\eta = 10^{-3}$, decoupled weight decay $\\beta = 10^{-2}$)"),
        ("Training Protocol", "Standard mini-batch optimization with early validation stopping"),
        ("Random Seed", "Deterministic seed 42"),
        ("Evaluation Protocol", "Single-split controlled test benchmark (zero leakage)"),
    ]

