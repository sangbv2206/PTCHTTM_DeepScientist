
import os
import sys
import re
import json
import platform
import time
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


def get_friendly_cpu_name() -> str:
    """Thu thập tên thương mại của CPU (ví dụ: 12th Gen Intel Core i5-12500H) thay vì chuỗi thô."""
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            if cpu_name:
                return re.sub(r"\s+", " ", cpu_name)
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            if out:
                return out
    except Exception:
        pass
    return platform.processor() or "CPU"


@dataclass
class EnvironmentInfo:
    """Thu thập tự động 100% từ hệ điều hành và phần cứng thực tế (Điều #2 - Không bịa môi trường)."""
    os_name: str = field(default_factory=lambda: f"{platform.system()} {platform.release()}")
    python_version: str = field(default_factory=lambda: platform.python_version())
    torch_version: str = "N/A"
    device: str = "cpu"
    cuda_available: bool = False
    device_name: str = "CPU"
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 1)
    ram_gb: float = 0.0
    wall_time_sec: float = 0.0
    start_timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    is_sandboxed: bool = False

    @classmethod
    def collect(cls, wall_time_sec: float = 0.0, is_sandboxed: bool = False) -> "EnvironmentInfo":
        torch_ver = "N/A"
        cuda_avail = False
        dev_name = get_friendly_cpu_name()
        device = "cpu"

        try:
            import torch
            torch_ver = torch.__version__
            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                device = "cuda"
                dev_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        ram = 0.0
        try:
            import psutil
            ram = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except Exception:
            pass

        return cls(
            torch_version=torch_ver,
            device=device,
            cuda_available=cuda_avail,
            device_name=dev_name,
            ram_gb=ram,
            wall_time_sec=round(wall_time_sec, 2),
            is_sandboxed=is_sandboxed
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class FigureManifestItem:
    """Mô tả xuất xứ của từng hình ảnh kết xuất từ thực nghiệm (Điều #1, #3)."""
    filename: str                      # Ví dụ: "learning_curve.png"
    figure_type: str                   # "empirical_result" hoặc "architectural_concept"
    caption: str                       # Chú thích hình
    data_source: str                   # Đường dẫn tới tệp dữ liệu gốc (vd: "history.json", "results.json")
    metrics_plotted: List[str] = field(default_factory=list)


@dataclass
class ClaimItem:
    """Khoản mục tuyên bố thực nghiệm (Claim) - Nguồn sự thật duy nhất cho việc phát ngôn (Điều #1, #4, #6)."""
    claim_id: str                      # Mã định danh (vd: "R1", "R2", "C1")
    metric_key: str                    # Tên metric (vd: "test_accuracy", "f1_macro", "loss")
    method_name: str                   # Phương pháp đề xuất (vd: "Ours")
    method_mean: float                 # Giá trị trung bình đạt được
    method_std: float                  # Độ lệch chuẩn (tính từ >= 3 seed)
    baseline_name: str = ""            # Tên baseline so sánh (nếu có)
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    delta: float = 0.0                 # Chênh lệch tuyệt đối
    rel_improvement_pct: float = 0.0   # Tỷ lệ % cải thiện tính bằng code
    direction: str = "higher_is_better" # "higher_is_better" hoặc "lower_is_better"
    is_statistically_significant: bool = False
    source_file: str = "results.json"

    def format_canonical_statement(self) -> str:
        """Sinh câu phát ngôn chuẩn mực, chính xác 100% về mặt toán học."""
        std_str = f" ± {self.method_std:.4f}" if self.method_std > 0 else ""
        if not self.baseline_name:
            return f"{self.method_name} đạt {self.metric_key} là {self.method_mean:.4f}{std_str}"
        
        dir_word = "cao hơn" if self.delta > 0 else "thấp hơn"
        better_word = "vượt trội hơn" if (
            (self.direction == "higher_is_better" and self.delta > 0) or
            (self.direction == "lower_is_better" and self.delta < 0)
        ) else "kém hơn"

        return (
            f"{self.method_name} đạt {self.metric_key} là {self.method_mean:.4f}{std_str}, "
            f"thể hiện hiệu năng {better_word} ({dir_word} {abs(self.delta):.4f}, tương đương {abs(self.rel_improvement_pct):.1f}%) "
            f"so với {self.baseline_name} ({self.baseline_mean:.4f})"
        )


def validate_results_data(data: Dict[str, Any]) -> List[str]:
    """Kiểm tra tính hợp lệ của tệp kết quả chạy thật (hỗ trợ cả Flat và Nested Template dict). Trả về danh sách vi phạm."""
    errors = []
    if not isinstance(data, dict):
        return ["Tệp results.json phải là một JSON Object."]

    metrics = data.get("metrics")
    if not metrics or not isinstance(metrics, dict):
        return ["Thiếu trường 'metrics' hoặc 'metrics' không phải là object chứa kết quả."]

    def check_values(val: Any, path: str):
        if val is None:
            errors.append(f"Metric '{path}' bị giá trị None.")
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            if math.isnan(val) or math.isinf(val) or str(val).lower() in ["nan", "inf", "-inf"]:
                errors.append(f"Metric '{path}' chứa NaN hoặc Inf.")
        elif isinstance(val, list):
            if len(val) == 0:
                errors.append(f"Metric '{path}' là danh sách rỗng.")
            else:
                for idx, item in enumerate(val):
                    if item is None:
                        errors.append(f"Metric '{path}[{idx}]' bị giá trị None.")
                    elif isinstance(item, (int, float)) and not isinstance(item, bool):
                        if math.isnan(item) or math.isinf(item) or str(item).lower() in ["nan", "inf", "-inf"]:
                            errors.append(f"Metric '{path}[{idx}]' chứa phần tử NaN/Inf.")
        elif isinstance(val, dict):
            for sub_k, sub_v in val.items():
                check_values(sub_v, f"{path}.{sub_k}")

    for model_name, m_dict in metrics.items():
        if not isinstance(m_dict, dict):
            errors.append(f"Model '{model_name}' không chứa dictionary metric hợp lệ.")
            continue
        for k, v in m_dict.items():
            check_values(v, f"{model_name}.{k}")

    return errors
