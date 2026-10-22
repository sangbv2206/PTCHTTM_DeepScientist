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
    "gemini-1.5-pro",
]


class GeminiLLM:
    """
    LLM Client chi phí 0đ chuyên dụng cho DeepScientist, hỗ trợ Google Gemini API.
    - Điều tiết tốc độ gọi (Pacing: tối thiểu 4.0s giữa các request) để không chạm trần RPM.
    - Khi gặp 429 (Rate Limit): Áp dụng Exponential Backoff trên cùng một model trước.
    - Cố định tính nhất quán của model trong một bài báo.
    - Chuẩn bảo đảm chất lượng (Fail-Closed): Ném ngoại lệ khi hết số lần thử, không bao giờ trả về chuỗi rỗng.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_retries: int = 5,
        min_request_interval: float = 4.0,
    ):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        raw_keys = api_key or os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self.api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        self.current_key_idx = 0
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.model_name = os.getenv("GEMINI_MODEL", model_name or "gemini-2.0-flash")
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_request_interval = min_request_interval
        self._last_request_time = 0.0

        if not self.api_key:
            raise ValueError(
                "Chưa tìm thấy API Key! Vui lòng truyền tham số --api-key hoặc thiết lập biến môi trường "
                "GEMINI_API_KEY hoặc GEMINI_API_KEYS (lấy miễn phí 100% tại https://aistudio.google.com/)."
            )

        print(f"[GeminiLLM] Đã khởi tạo LLM Client với {len(self.api_keys)} API Keys, model gốc: '{self.model_name}' (Pacing: {self.min_request_interval}s/req)")
        self._init_models()

    def _init_models(self):
        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def _rotate_api_key(self) -> bool:
        """Xoay vòng sang API Key kế tiếp nếu có nhiều hơn 1 key được cấu hình."""
        if len(self.api_keys) > 1:
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            self.api_key = self.api_keys[self.current_key_idx]
            masked_key = self.api_key[:6] + "..." + self.api_key[-4:] if len(self.api_key) > 10 else "***"
            print(f"\n[GeminiLLM] TỰ ĐỘNG XOAY VÒNG KEY: Chuyển sang API Key #{self.current_key_idx + 1} ({masked_key}) để giải phóng hạn ngạch...")
            self._init_models()
            return True
        return False

    def _switch_to_next_fallback_model(self):
        try:
            curr_idx = FALLBACK_MODELS.index(self.model_name)
            next_idx = (curr_idx + 1) % len(FALLBACK_MODELS)
        except ValueError:
            next_idx = 0
        candidate = FALLBACK_MODELS[next_idx]
        print(f"\n[GeminiLLM] CẢNH BÁO CHUYỂN MODEL: Model '{self.model_name}' đã cạn kiệt Daily Quota hoặc không hỗ trợ. Chuyển sang: '{candidate}'...")
        self.model_name = candidate
        self._init_models()
        return True

    def _wait_pacing(self):
        """Giữ khoảng cách tối thiểu giữa các lệnh gọi để không bị vượt quá RPM của Free Tier."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Gửi prompt tới Gemini API với Pacing và Exponential Backoff thông minh."""
        temp = temperature if temperature is not None else self.temperature
        attempt = 0
        rpm_429_count = 0

        while attempt < self.max_retries:
            self._wait_pacing()
            try:
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})

                res = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=8192,
                )
                if res and res.choices and len(res.choices) > 0:
                    content = res.choices[0].message.content
                    if content and content.strip():
                        return content.strip()

                raise ValueError(f"Gemini API trả về nội dung rỗng (Model: {self.model_name})")

            except Exception as e:
                err_str = str(e).lower()
                attempt += 1

                # 1. Nếu model bị deprecated / không tìm thấy (404/not_found)
                if any(kw in err_str for kw in ["not_found", "unsupported", "404", "invalid_argument"]):
                    print(f"[GeminiLLM] Model '{self.model_name}' không tồn tại hoặc không hỗ trợ ({e}). Chuyển model dự phòng...")
                    self._switch_to_next_fallback_model()
                    time.sleep(1)
                    continue

                # 2. Nếu là Rate Limit / Quota (429, resource_exhausted)
                if any(kw in err_str for kw in ["quota", "resourceexhausted", "resource_exhausted", "429"]):
                    # Ưu tiên xoay vòng ngay sang API Key kế tiếp nếu có cấu hình nhiều key
                    if self._rotate_api_key():
                        attempt = 0
                        rpm_429_count = 0
                        time.sleep(1)
                        continue

                    rpm_429_count += 1
                    # Nếu mới gặp 1-2 lần -> khả năng cao là chạm trần RPM, chỉ cần backoff nghỉ 10s-20s rồi thử lại cùng model!
                    if rpm_429_count <= 2:
                        wait_time = 12 * rpm_429_count
                        print(f"[GeminiLLM] Chạm ngưỡng RPM của Free Tier ({self.model_name}). Nghỉ {wait_time}s rồi tiếp tục...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Nếu đã backoff 2 lần mà vẫn 429 -> đã hết Daily Quota (RPD) của model này -> mới chuyển model!
                        print(f"[GeminiLLM] Model '{self.model_name}' liên tục 429 (nghi hết Daily Quota RPD). Đổi sang model kế tiếp...")
                        self._switch_to_next_fallback_model()
                        attempt = max(0, attempt - 1)
                        rpm_429_count = 0
                        time.sleep(2)
                        continue

                # 3. Lỗi mạng hoặc server thông thường
                wait_time = 2 * attempt
                print(f"[GeminiLLM] Lỗi API (Lần {attempt}/{self.max_retries}): {str(e)[:120]}... Đang đợi {wait_time}s...")
                if attempt >= self.max_retries:
                    raise RuntimeError(f"[GeminiLLM] Thất bại sau {self.max_retries} lần thử: {e}")
                time.sleep(wait_time)

        raise RuntimeError(f"[GeminiLLM] Không thể sinh phản hồi từ API sau {self.max_retries} lần thử (Model: {self.model_name})")

