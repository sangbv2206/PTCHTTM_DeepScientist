"""
Lightweight, budget-capped LLM helper copied into zero-shot experiment working directories
as `llm_client.py` for topics requiring genuine LLM reasoning (e.g., RAG, QA, Agents, NLP).
Reads GEMINI_API_KEY / GEMINI_API_KEYS / GEMINI_MODEL from environment variables.
"""
import hashlib
import os
import socket
import time
from typing import Any, Dict

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
    try:
        res = _orig_getaddrinfo(host, port, family, type, proto, flags)
        ipv4 = [r for r in res if r[0] == socket.AF_INET]
        return ipv4 if ipv4 else res
    except Exception:
        return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_first


class BudgetedLLMClient:
    def __init__(self, max_calls: int = 120, min_interval: float = 0.5):
        raw = (
            os.getenv("GEMINI_API_KEYS")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        )
        self.api_keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not self.api_keys:
            raise RuntimeError("No GEMINI_API_KEY / GOOGLE_API_KEY available in environment.")
        self.key_idx = 0
        self.exhausted_keys = set()
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.max_calls = max_calls
        self.min_interval = min_interval
        self.calls_made = 0
        self.last_call = 0.0
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._init_client()

    def _init_client(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_keys[self.key_idx],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def call(self, prompt: str, system: str = "You are a concise, accurate assistant.", max_tokens: int = 128, temperature: float = 0.0) -> Dict[str, Any]:
        cache_key = hashlib.sha256(f"{self.model}||{system}||{prompt}".encode("utf-8")).hexdigest()
        if cache_key in self.cache:
            cached = dict(self.cache[cache_key])
            cached["cached"] = True
            return cached

        if self.calls_made >= self.max_calls:
            return {
                "text": "",
                "prompt_tokens": max(1, len(prompt.split())),
                "completion_tokens": 0,
                "total_tokens": max(1, len(prompt.split())),
                "latency_ms": 0.0,
                "budget_exceeded": True,
                "cached": False,
            }

        wait = self.min_interval - (time.time() - self.last_call)
        if wait > 0:
            time.sleep(wait)

        last_err = None
        for attempt in range(4):
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.last_call = time.time()
                self.calls_made += 1
                text = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                ptok = int(getattr(usage, "prompt_tokens", 0) or max(1, int(len(prompt.split()) * 1.3)))
                ctok = int(getattr(usage, "completion_tokens", 0) or max(1, int(len(text.split()) * 1.3)))
                out = {
                    "text": text,
                    "prompt_tokens": ptok,
                    "completion_tokens": ctok,
                    "total_tokens": ptok + ctok,
                    "latency_ms": latency_ms,
                    "cached": False,
                }
                self.cache[cache_key] = out
                return out
            except Exception as e:
                last_err = e
                err_s = str(e).lower()
                if "429" in err_s or "rate" in err_s or "quota" in err_s or "resource_exhausted" in err_s:
                    self.exhausted_keys.add(self.key_idx)
                    print(f"[BudgetedLLMClient] CẢNH BÁO HẠN MỨC: API Key #{self.key_idx + 1} đã chạm hạn mức (429/Quota).")
                    next_idx = next((i for i in range(len(self.api_keys)) if i not in self.exhausted_keys), None)
                    if next_idx is not None:
                        self.key_idx = next_idx
                        print(f"[BudgetedLLMClient] Chuyển sang API Key #{self.key_idx + 1}...")
                        self._init_client()
                        continue
                    msg = f"[BudgetedLLMClient] DỪNG NGAY: Tất cả {len(self.api_keys)} API Key đều đã chạm hạn mức (429/Quota)!"
                    print(msg)
                    raise RuntimeError(msg)
                else:
                    time.sleep(1.0)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")


_DEFAULT_CLIENT = None


def call_llm(prompt: str, system: str = "You are a concise, accurate assistant.", max_tokens: int = 128, temperature: float = 0.0, max_calls: int = 80) -> Dict[str, Any]:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = BudgetedLLMClient(max_calls=max_calls)
    return _DEFAULT_CLIENT.call(prompt, system=system, max_tokens=max_tokens, temperature=temperature)

