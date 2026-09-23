import json
import os
import re
import shutil
from typing import Dict, List, Optional, Any


class TemplateManager:
    """
    Quản lý toàn bộ kho Template (Template Zoo) của DeepScientist:
    - 16 PyTorch benchmark templates (gnn_graph_learning, recommender_system, causal_trial_emulation,
      reinforcement_learning, mobilenetV3, nanoGPT, 2d_diffusion, time_series_forecasting, etc.)
    - Hỗ trợ định tuyến thông minh song ngữ (Tiếng Việt & Tiếng Anh) qua Semantic LLM Router & Heuristic Keyword Matcher.
    """

    def __init__(self, custom_templates_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.templates_dir = custom_templates_dir or base_dir

    def list_available_templates(self) -> List[str]:
        """Liệt kê tất cả các template đang có trong thư mục templates nội bộ."""
        templates = []
        for name in os.listdir(self.templates_dir):
            p = os.path.join(self.templates_dir, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "experiment.py")):
                templates.append(name)
        return sorted(templates)

    def select_best_template_for_topic(self, topic: str) -> str:
        """
        Chọn template tối ưu bằng Scoring Router đa tiêu chí.

        Thuật toán mới (v2) — thay thế First-Match heuristic:
        - Tính điểm cho TẤT CẢ template dựa trên TEMPLATE_SCORING_RULES.
        - Bigram/trigram đặc trưng có trọng số cao (20-30), unigram phụ có trọng số thấp (5-12).
        - Template có tổng điểm cao nhất được chọn.
        - Loại bỏ hoàn toàn các từ khóa gây nhiễu (single generic words):
          "regression", "data", "user", "classification", "sample", "generative", "predict".
        - Fallback an toàn: "nanoGPT" (tổng quát nhất) khi không có template nào đạt điểm tối thiểu.

        Song ngữ: Hỗ trợ cả tiếng Anh và tiếng Việt qua TEMPLATE_SCORING_RULES.
        """
        topic_lower = topic.lower()
        available = set(self.list_available_templates())

        def match_score(topic_l: str, rules: list) -> float:
            """Tính tổng điểm khớp từ khóa cho topic với danh sách (keyword, weight)."""
            total = 0.0
            for keyword, weight in rules:
                kw = keyword.lower().strip()
                if " " in kw:
                    if kw in topic_l:
                        total += weight
                else:
                    if re.search(r"\b" + re.escape(kw) + r"\b", topic_l):
                        total += weight
            return total

        # Tính điểm cho từng template theo TEMPLATE_SCORING_RULES
        scores: Dict[str, float] = {}
        for template_name, rules in TEMPLATE_SCORING_RULES.items():
            if template_name in available:
                scores[template_name] = match_score(topic_lower, rules)

        # Chọn template có điểm cao nhất
        if scores:
            best_name = max(scores, key=lambda k: scores[k])
            best_score = scores[best_name]
            # Yêu cầu điểm tối thiểu để tránh chọn template không liên quan
            MIN_SCORE = 10.0
            if best_score >= MIN_SCORE:
                return best_name

        # Fallback an toàn: nanoGPT (tổng quát nhất, CoderAgent có thể tùy biến)
        if "nanoGPT" in available:
            return "nanoGPT"
        return available.pop() if available else "time_series_forecasting"

    def get_template_path(self, template_name: str) -> Optional[str]:
        """Lấy đường dẫn thư mục của template."""
        p = os.path.join(self.templates_dir, template_name)
        if os.path.exists(p):
            return p
        return None

    def copy_template_to_dir(self, template_name: str, target_dir: str):
        """Sao chép toàn bộ template vào thư mục thực nghiệm để tiến hành train."""
        src = self.get_template_path(template_name)
        if not src:
            raise FileNotFoundError(f"Template '{template_name}' not found!")

        os.makedirs(target_dir, exist_ok=True)
        for item in os.listdir(src):
            s_item = os.path.join(src, item)
            d_item = os.path.join(target_dir, item)
            if os.path.isdir(s_item):
                shutil.copytree(s_item, d_item, dirs_exist_ok=True)
            else:
                shutil.copy2(s_item, d_item)

    def load_prompt_info(self, template_name: str) -> Dict[str, Any]:
        """Đọc thông tin prompt.json của template."""
        src = self.get_template_path(template_name)
        if not src:
            return {}
        p_path = os.path.join(src, "prompt.json")
        if os.path.exists(p_path):
            with open(p_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "system": f"You are working with the {template_name} PyTorch benchmark codebase.",
            "task_description": "Train and evaluate model performance under proposed architectural enhancements."
        }

    def assess_template_suitability(self, topic: str, llm: Optional[Any] = None) -> Dict[str, Any]:
        """
        Kiểm tra toàn bộ kho template xem có template nào phù hợp với đề tài không.
        Sử dụng LLM Semantic Router để suy luận chuyên sâu nếu có LLM, hoặc fallback sang keyword matching.
        """
        available = self.list_available_templates()

        # 1. Thử dùng LLM Semantic Router nếu có LLM
        if llm:
            try:
                available_meta = {
                    name: TEMPLATE_METADATA.get(name, {"domain": "General PyTorch", "description": "Custom PyTorch benchmark codebase."})
                    for name in available
                }
                prompt = f"""You are an elite scientific AI architect.
User's Research Topic (in Vietnamese or English): "{topic}"

Available PyTorch Experiment Benchmark Templates in the system:
{json.dumps(available_meta, indent=2)}

TASK:
1. Understand the core scientific task and machine learning domain of the research topic.
2. Select the BEST matching template name strictly from the available templates list above.
   - For health impact, clinical trials, cohort studies, hazard analysis, biomass smoke exposure, or causal questions -> select "causal_trial_emulation".
   - For image recognition / CNN / vision -> select "mobilenetV3".
   - For recommender / user-item ranking -> select "recommender_system".
   - For reinforcement learning / control / robotics / decision making -> select "reinforcement_learning".
   - For graph neural networks / network topology -> select "gnn_graph_learning".
   - For sequential time-series / load forecasting / weather / sensors -> select "time_series_forecasting".
   - For language / LLM / NLP / text -> select "nanoGPT".
3. Assign a confidence score from 1 to 10 (1 = completely unrelated/unsupported, 10 = perfect match).
4. If confidence < 6 or if the topic is completely outside the scope of all available templates (e.g. quantum hardware, pure physical simulation), set is_supported = false.
5. Provide concise reasoning and suggestions in Vietnamese.

Respond strictly in JSON format:
```json
{{
  "is_supported": true,
  "confidence": 9,
  "selected_template": "causal_trial_emulation",
  "reasoning": "Đề tài thuộc lĩnh vực suy luận nhân quả và dịch tễ học quan sát, khớp hoàn hảo với template causal_trial_emulation.",
  "suggestions": "Đánh giá bằng đường cong liều lượng - đáp ứng và chỉ số ATE."
}}
```"""
                res = llm.generate(prompt)
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", res, flags=re.DOTALL)
                raw_json = match.group(1) if match else res
                data = json.loads(raw_json)
                if data.get("selected_template") in available:
                    return data
            except Exception as e:
                print(f"[TemplateManager] LLM routing error: {e}")

        # 2. Heuristic fallback
        best_name = self.select_best_template_for_topic(topic)
        topic_lower = topic.lower()
        meta = TEMPLATE_METADATA.get(best_name, {})
        matched_kw = []
        for k in meta.get("keywords", []):
            k_clean = k.lower().strip()
            if " " in k_clean and k_clean in topic_lower:
                matched_kw.append(k)
            elif re.search(r"\b" + re.escape(k_clean) + r"\b", topic_lower):
                matched_kw.append(k)

        is_supported = len(matched_kw) > 0
        confidence = min(5 + len(matched_kw) * 2, 9) if is_supported else 4

        if is_supported:
            reasoning = f"Đã khớp với miền {meta.get('domain', 'Chung')} qua các từ khóa: {', '.join(matched_kw[:4])}."
            suggestions = "Template có sẵn hỗ trợ đầy đủ baseline thực nghiệm và biểu đồ đối sánh."
        else:
            reasoning = f"Khớp tự động về template phù hợp nhất là '{best_name}'."
            suggestions = "CoderAgent sẽ tự động tùy biến kiến trúc và biểu đồ theo sát chủ đề."

        return {
            "is_supported": is_supported,
            "confidence": confidence,
            "selected_template": best_name,
            "fallback_template": best_name,
            "reasoning": reasoning,
            "suggestions": suggestions
        }


TEMPLATE_METADATA = {
    "causal_trial_emulation": {
        "domain": "Target Trial Emulation & Causal Survival Inference",
        "description": "Clinical target trial emulation, observational epidemiology, biomass smoke/environmental exposure risk, Inverse Probability Treatment Weighting (IPTW), covariate balance (ASMD), survival/hazard modeling, and Average Treatment Effect (ATE) estimation.",
        "keywords": [
            "target trial", "clinical trial", "trial emulation", "causal", "causal inference",
            "propensity", "iptw", "treatment effect", "ate", "hazard", "survival",
            "cardiovascular", "drug", "sglt2", "glp-1", "agonist", "patient", "biomedical",
            "ehr", "observational cohort", "confounding", "biomass", "smoke", "respiratory",
            "thử nghiệm lâm sàng", "thử nghiệm giả lập", "suy luận nhân quả", "nhân quả",
            "khói bếp", "sinh khối", "đường hô hấp", "y tế", "y sinh", "tác động của", "khảo sát tác động"
        ]
    },
    "recommender_system": {
        "domain": "Recommender Systems & Information Retrieval",
        "description": "Matrix factorization, collaborative filtering, neural recommendation, and fairness-aware item ranking.",
        "keywords": [
            "recommend", "recommender", "recommendation", "user", "item", "collaborative filtering", "cf", "fairness", "rating", "ranking",
            "gợi ý", "hệ thống gợi ý", "đề xuất", "lọc cộng tác", "xếp hạng", "người dùng", "sản phẩm"
        ]
    },
    "reinforcement_learning": {
        "domain": "Deep Reinforcement Learning & Continuous Control",
        "description": "Deep Q-Networks (DQN), continuous state-space MDP control, experience replay, target networks, and temporal difference learning.",
        "keywords": [
            "reinforcement learning", "reinforcement", "deep reinforcement", "q-learning",
            "dqn", "policy gradient", "ppo", "actor-critic", "agent", "reward", "markov",
            "mdp", "cartpole", "control", "exploration", "replay buffer", "bellman",
            "học tăng cường", "phần thưởng", "hành động", "tác nhân", "điều khiển robot"
        ]
    },
    "mobilenetV3": {
        "domain": "Computer Vision & CNNs",
        "description": "Deep Convolutional Neural Networks, lightweight image classification, CIFAR benchmark.",
        "keywords": [
            "vision", "image", "conv", "cnn", "mobilenet", "cifar", "classification", "visual",
            "thị giác", "thị giác máy tính", "hình ảnh", "nhận diện ảnh", "phân loại ảnh", "nhận dạng"
        ]
    },
    "gnn_graph_learning": {
        "domain": "Graph Neural Networks & Relational Learning",
        "description": "Semi-supervised node classification on topological graphs, GCN, GAT, message passing.",
        "keywords": [
            "graph", "gnn", "node", "edge", "topolog", "gcn", "gat", "subgraph", "network topology",
            "đồ thị", "mạng đồ thị", "mạng lưới", "mạng xã hội", "liên kết", "nút mạng"
        ]
    },
    "nanoGPT": {
        "domain": "Natural Language Processing & LLMs",
        "description": "Transformers, causal language modeling, generative text, attention mechanisms.",
        "keywords": [
            "transformer", "gpt", "language", "nlp", "llm", "text", "token", "sentiment", "translation",
            "xử lý ngôn ngữ", "ngôn ngữ tự nhiên", "văn bản", "sinh văn bản", "dịch thuật", "mô hình ngôn ngữ"
        ]
    },
    "2d_diffusion": {
        "domain": "Generative Modeling & Diffusion",
        "description": "Denoising Diffusion Probabilistic Models (DDPM), 2D generative image synthesis.",
        "keywords": [
            "diffusion", "generative", "denoising", "sample", "ddpm", "score-based",
            "khuếch tán", "mô hình khuếch tán", "tạo ảnh khuếch tán", "khử nhiễu"
        ]
    },
    "time_series_forecasting": {
        "domain": "Time-Series Forecasting & Sequential Regression",
        "description": "Temporal recurrent neural networks (GRU/LSTM/Linear), multi-step sequential load, weather, and sensor prediction.",
        "keywords": [
            "time series", "timeseries", "forecast", "forecasting", "predict", "prediction", "energy", "consumption", "electricity", "household", "power", "load", "sensor", "temporal", "regression", "smart home", "smart grid", "weather",
            "chuỗi thời gian", "dự báo", "dự đoán", "tiêu thụ điện", "điện năng", "phụ tải", "thời tiết", "mực nước"
        ]
    },
    "seir": {
        "domain": "Epidemiological Dynamics & ODE Simulation",
        "description": "Compartmental disease modeling with differential equations.",
        "keywords": ["epidemic", "seir", "disease", "infection", "virus", "ode", "differential equations", "pandemic", "dịch bệnh", "lây nhiễm"]
    },
    "earthquake-prediction": {
        "domain": "Geophysical Time Series",
        "description": "Seismic time-series signal prediction and laboratory earthquake timing.",
        "keywords": ["earthquake", "seismic", "geophysics", "fault", "động đất", "địa chấn"]
    },
    "grokking": {
        "domain": "Theoretical Deep Learning Dynamics",
        "description": "Delayed generalization, algorithmic dataset memorization, weight decay dynamics.",
        "keywords": ["grok", "grokking", "generalization", "overfitting", "algorithmic learning"]
    },
    "sketch_rnn": {
        "domain": "Sequential Vector Graphics & Drawing",
        "description": "Recurrent VAE for sequential stroke generation.",
        "keywords": ["sketch", "drawing", "stroke", "vector graphics", "rnn vae"]
    },
    "tensorf": {
        "domain": "3D Neural Rendering & Radiance Fields",
        "description": "Tensorial Radiance Fields for 3D novel view synthesis.",
        "keywords": ["nerf", "tensorf", "radiance field", "3d rendering", "view synthesis"]
    },
    "MACE": {
        "domain": "Atomic & Molecular Physics",
        "description": "Higher-order equivariant message passing for molecular force fields.",
        "keywords": ["molecule", "molecular", "interatomic", "atom", "chemistry", "force field", "mace", "phân tử", "nguyên tử"]
    },
    "probes": {
        "domain": "Model Interpretability & Memory Profiling",
        "description": "Linear probing of representation dynamics and GPU allocation.",
        "keywords": ["probe", "probing", "interpretability", "representation analysis"]
    }
}


TEMPLATE_SCORING_RULES: Dict[str, list] = {
    "causal_trial_emulation": [
        ("target trial", 30.0), ("trial emulation", 30.0), ("thử nghiệm lâm sàng", 25.0), ("thử nghiệm giả lập", 25.0),
        ("causal inference", 25.0), ("suy luận nhân quả", 25.0), ("nhân quả", 15.0), ("causal", 15.0),
        ("propensity score", 25.0), ("iptw", 25.0), ("treatment effect", 25.0), ("average treatment effect", 25.0),
        ("survival analysis", 20.0), ("hazard ratio", 20.0), ("observational study", 18.0), ("observational cohort", 20.0),
        ("biomass smoke", 25.0), ("khói sinh khối", 25.0), ("khói bếp", 25.0), ("sglt2", 20.0), ("glp-1", 20.0),
        ("cardiovascular", 15.0), ("respiratory", 15.0), ("confounding", 15.0)
    ],
    "recommender_system": [
        ("recommender system", 30.0), ("recommendation system", 30.0), ("hệ thống gợi ý", 30.0), ("hệ thống đề xuất", 30.0),
        ("collaborative filtering", 25.0), ("lọc cộng tác", 25.0), ("matrix factorization", 25.0),
        ("recommender", 20.0), ("recommendation", 15.0), ("item ranking", 18.0), ("fairness in recommendation", 25.0),
        ("gợi ý sản phẩm", 20.0), ("đề xuất sản phẩm", 20.0)
    ],
    "reinforcement_learning": [
        ("reinforcement learning", 30.0), ("học tăng cường", 30.0), ("deep q-network", 25.0), ("dqn", 20.0),
        ("policy gradient", 25.0), ("ppo", 20.0), ("actor-critic", 25.0), ("markov decision", 25.0), ("mdp", 18.0),
        ("q-learning", 20.0), ("continuous control", 20.0), ("reward function", 18.0), ("cartpole", 20.0),
        ("điều khiển robot", 20.0), ("tác nhân học tăng cường", 20.0), ("experience replay", 20.0)
    ],
    "mobilenetV3": [
        ("image classification", 25.0), ("phân loại ảnh", 25.0), ("nhận diện ảnh", 25.0), ("nhận dạng hình ảnh", 25.0),
        ("computer vision", 25.0), ("thị giác máy tính", 25.0), ("cnn", 18.0), ("convolutional", 18.0),
        ("mobilenet", 25.0), ("cifar", 20.0), ("imagenet", 20.0), ("object detection", 20.0),
        ("phát hiện đối tượng", 20.0), ("visual recognition", 20.0)
    ],
    "gnn_graph_learning": [
        ("graph neural network", 30.0), ("mạng nơ-ron đồ thị", 30.0), ("graph neural", 25.0), ("gnn", 22.0),
        ("graph convolutional", 25.0), ("gcn", 22.0), ("graph attention", 25.0), ("gat", 22.0),
        ("node classification", 22.0), ("link prediction", 22.0), ("phân loại nút", 22.0),
        ("đồ thị tri thức", 18.0), ("knowledge graph", 18.0), ("mạng đồ thị", 20.0), ("graph representation", 20.0),
        ("molecular graph", 20.0), ("relational learning", 20.0), ("graph", 10.0)
    ],
    "nanoGPT": [
        ("language model", 25.0), ("mô hình ngôn ngữ", 25.0), ("large language model", 25.0), ("llm", 20.0),
        ("transformer", 22.0), ("nanogpt", 25.0), ("gpt", 18.0), ("causal language modeling", 25.0),
        ("text generation", 22.0), ("sinh văn bản", 22.0), ("nlp", 18.0), ("natural language", 20.0),
        ("xử lý ngôn ngữ tự nhiên", 22.0), ("attention mechanism", 20.0), ("self-attention", 20.0),
        ("kolmogorov-arnold", 25.0), ("kan", 15.0), ("neural architecture", 15.0)
    ],
    "2d_diffusion": [
        ("diffusion model", 30.0), ("mô hình khuếch tán", 30.0), ("ddpm", 25.0), ("denoising diffusion", 25.0),
        ("score-based generative", 25.0), ("diffusion", 20.0), ("tạo ảnh khuếch tán", 25.0), ("khử nhiễu", 15.0),
        ("generative diffusion", 25.0)
    ],
    "time_series_forecasting": [
        ("time series", 25.0), ("timeseries", 25.0), ("chuỗi thời gian", 25.0),
        ("forecasting", 18.0), ("forecast", 15.0), ("dự báo", 15.0),
        ("power consumption", 20.0), ("energy consumption", 20.0), ("electricity consumption", 20.0),
        ("tiêu thụ điện", 20.0), ("phụ tải điện", 20.0), ("smart grid", 18.0),
        ("weather forecasting", 20.0), ("dự báo thời tiết", 20.0),
        ("temporal prediction", 18.0), ("sequential forecasting", 18.0),
        ("multivariate time series", 25.0), ("load forecast", 20.0),
        ("lstm", 10.0), ("gru", 10.0), ("temporal", 8.0)
    ],
    "seir": [
        ("seir", 30.0), ("epidemic", 25.0), ("epidemiological", 25.0), ("dịch bệnh", 25.0),
        ("infectious disease", 25.0), ("disease transmission", 25.0), ("compartmental model", 25.0),
        ("pandemic", 20.0), ("virus transmission", 20.0), ("lây nhiễm", 18.0), ("ode simulation", 20.0),
        ("differential equation", 18.0)
    ],
    "earthquake-prediction": [
        ("earthquake", 30.0), ("động đất", 30.0), ("seismic", 25.0), ("địa chấn", 25.0),
        ("laboratory earthquake", 25.0), ("acoustic emission", 22.0), ("fault rupture", 22.0), ("geophysical", 20.0)
    ],
    "grokking": [
        ("grokking", 30.0), ("delayed generalization", 25.0), ("algorithmic reasoning", 20.0),
        ("algorithmic dataset", 25.0), ("weight decay dynamics", 22.0), ("generalization phase transition", 25.0),
        ("memorization", 15.0)
    ],
    "sketch_rnn": [
        ("sketch-rnn", 30.0), ("sketch rnn", 30.0), ("quickdraw", 25.0), ("stroke generation", 25.0),
        ("vector graphics", 22.0), ("drawing generation", 22.0), ("sketch generation", 25.0), ("sketch", 15.0)
    ],
    "tensorf": [
        ("tensorf", 30.0), ("nerf", 25.0), ("neural radiance field", 30.0), ("radiance field", 25.0),
        ("novel view synthesis", 25.0), ("3d neural rendering", 25.0), ("3d view synthesis", 25.0)
    ],
    "MACE": [
        ("mace", 30.0), ("molecular force field", 30.0), ("interatomic potential", 25.0),
        ("atomic cluster expansion", 25.0), ("equivariant message passing", 25.0),
        ("molecular dynamics", 22.0), ("lực phân tử", 22.0), ("hóa học tính toán", 20.0), ("interatomic", 20.0)
    ],
    "probes": [
        ("linear probe", 25.0), ("probing representation", 25.0), ("interpretability probe", 25.0),
        ("memory profiling", 22.0), ("representation probing", 25.0), ("activation probing", 22.0)
    ]
}

