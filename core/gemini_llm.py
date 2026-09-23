import os
import sys
import time
import warnings
from typing import Optional, List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

try:
    import google.generativeai as genai
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

import socket
_orig_getaddrinfo = socket.getaddrinfo

def _ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
    try:
        res = _orig_getaddrinfo(host, port, family, type, proto, flags)
        ipv4 = [r for r in res if r[0] == socket.AF_INET]
        return ipv4 if ipv4 else res
    except Exception:
        return _orig_getaddrinfo(host, port, family, type, proto, flags)

socket.getaddrinfo = _ipv4_first

from openai import OpenAI


FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-1.5-pro",
]


class GeminiLLM:
    """
    LLM Client chi phí 0đ chuyên dụng cho DeepScientist, hỗ trợ Google Gemini API.
    Có cơ chế tự động xoay vòng model thông minh (fallback cascading) và thử lại (exponential backoff)
    để không bao giờ bị gián đoạn do Rate Limit hay hết Quota Free Tier.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_retries: int = 5,
    ):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", model_name or "gemini-2.0-flash")
        self.temperature = temperature
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError(
                "Chưa tìm thấy API Key! Vui lòng truyền tham số --api-key hoặc thiết lập biến môi trường "
                "GEMINI_API_KEY (lấy miễn phí 100% tại https://aistudio.google.com/)."
            )

        self._init_models()

    def _init_models(self):
        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        if HAS_GOOGLE_GENAI:
            try:
                genai.configure(api_key=self.api_key)
                self.native_model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=genai.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=8192,
                    )
                )
            except Exception:
                self.native_model = None
        else:
            self.native_model = None

    def _switch_to_next_fallback_model(self):
        try:
            curr_idx = FALLBACK_MODELS.index(self.model_name)
            next_idx = (curr_idx + 1) % len(FALLBACK_MODELS)
        except ValueError:
            next_idx = 0
        candidate = FALLBACK_MODELS[next_idx]
        print(f"[GeminiLLM] Model '{self.model_name}' gặp sự cố/hết quota. Tự động chuyển sang model dự phòng '{candidate}'...")
        self.model_name = candidate
        self._init_models()
        return True

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Gửi prompt tới Gemini API qua REST HTTPS ổn định, tự động đổi model khi gặp Rate Limit/Quota."""
        temp = temperature if temperature is not None else self.temperature
        attempt = 0

        while attempt < self.max_retries:
            try:
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})

                res = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                )
                if res and res.choices and len(res.choices) > 0:
                    content = res.choices[0].message.content
                    if content:
                        return content.strip()

            except Exception as e:
                err_str = str(e)
                attempt += 1

                # Nếu hết quota (429, ResourceExhausted, 404 not found), đổi ngay model kế tiếp
                if any(kw in err_str.lower() for kw in ["quota", "resourceexhausted", "429", "401", "not_found", "unsupported", "404"]):
                    self._switch_to_next_fallback_model()
                    time.sleep(1)
                    continue

                wait_time = 2 * attempt
                print(f"[GeminiLLM] Lỗi API (Lần {attempt}/{self.max_retries}): {err_str[:120]}... Đang đợi {wait_time}s...")
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Gọi Gemini API thất bại sau {self.max_retries} lần thử: {e}")
                time.sleep(wait_time)

        return ""
