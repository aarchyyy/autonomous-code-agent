"""
LLM Backend Module: Supports live OpenAI-compatible endpoints (cloud APIs and local
Token Factory / Ollama / vLLM / Kunlun Gateway servers) as well as a clearly labeled
Deterministic Pipeline Smoke Simulator for offline CI/harness verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


HARDWARE_PRESETS: Dict[str, Dict[str, str]] = {
    "kunlun-gateway": {
        "label": "Kunlun GPT-Factory One / AI Agent Gateway",
        "base_url": "http://127.0.0.1:8000/v1",
        "default_model": "baize-omni-sense",
    },
    "ollama": {
        "label": "Ollama Local (AMD Ryzen AI MAX+ 395 / Local)",
        "base_url": "http://127.0.0.1:11434/v1",
        "default_model": "qwen2.5-coder:14b",
    },
    "vllm": {
        "label": "vLLM / ROCm OpenAI-Compatible Local Server",
        "base_url": "http://127.0.0.1:8000/v1",
        "default_model": "Qwen/Qwen2.5-Coder-32B-Instruct",
    },
    "dashscope": {
        "label": "Alibaba Cloud DashScope (Qwen)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-coder-plus",
    },
    "deepseek": {
        "label": "DeepSeek API",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "openai": {
        "label": "OpenAI API",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
}


@dataclass
class AgentTurn:
    thought: str
    code: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    model: str = ""
    provider: str = ""
    endpoint: str = ""
    is_mock: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    response_sha256: str = ""


class BaseLLM(ABC):
    is_mock: bool = False
    supports_tools: bool = True
    backend_label: str = "BaseLLM"
    model: str = "unknown"

    @abstractmethod
    def generate_patch(self, prompt: str, system_prompt: str) -> Tuple[str, str]:
        """Single-turn compatibility interface returning (thought, code)."""

    def chat_step(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentTurn:
        """Multi-turn interface supporting native or text-protocol tool calls."""
        system_prompt = ""
        user_parts: List[str] = []
        for m in messages:
            if m.get("role") == "system":
                system_prompt = str(m.get("content", ""))
            else:
                user_parts.append(f"[{m.get('role', 'user').upper()}]\n{m.get('content', '')}")
        prompt = "\n\n".join(user_parts)
        t0 = time.perf_counter()
        thought, code = self.generate_patch(prompt=prompt, system_prompt=system_prompt)
        dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        raw = f"[THOUGHT]\n{thought}\n\n[CODE]\n```python\n{code}\n```"
        return AgentTurn(
            thought=thought,
            code=code,
            raw_text=raw,
            model=getattr(self, "model", "unknown"),
            provider=self.backend_label,
            is_mock=self.is_mock,
            latency_ms=dt_ms,
            response_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )


def _normalize_completions_url(base_url: str) -> str:
    clean = base_url.strip().rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return f"{clean}/chat/completions"


def _normalize_models_url(base_url: str) -> str:
    clean = base_url.strip().rstrip("/")
    if clean.endswith("/chat/completions"):
        clean = clean[: -len("/chat/completions")]
    if clean.endswith("/models"):
        return clean
    return f"{clean}/models"


def probe_llm_endpoint(
    base_url: str,
    api_key: Optional[str] = None,
    timeout: int = 8,
) -> Dict[str, Any]:
    """Probes an OpenAI-compatible `/v1/models` endpoint for connectivity & model discovery."""
    url = _normalize_models_url(base_url)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
        models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        return {
            "ok": True,
            "url": url,
            "models": models,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "models": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "error": str(exc),
        }


def _parse_text_response(raw_text: str) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
    text = raw_text or ""

    # Preserve <think> content if no explicit [THOUGHT] block is provided
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    thought = ""
    if "[THOUGHT]" in cleaned_text:
        after = cleaned_text.split("[THOUGHT]", 1)[1]
        for marker in ("[TOOL_CALL]", "[CODE]", "```"):
            if marker in after:
                after = after.split(marker, 1)[0]
        thought = after.strip()
    elif think_match:
        thought = think_match.group(1).strip()

    # Parse text-based tool calls: [TOOL_CALL] {"name": ..., "arguments": {...}} [/TOOL_CALL]
    tool_calls: List[Dict[str, Any]] = []
    for idx, m in enumerate(
        re.finditer(r"\[TOOL_CALL\]\s*(\{.*?\})\s*(?:\[/TOOL_CALL\]|$)", cleaned_text, re.DOTALL)
    ):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and obj.get("name"):
                args = obj.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append({
                    "id": obj.get("id") or f"text_tc_{idx + 1}",
                    "name": obj["name"],
                    "arguments": args if isinstance(args, dict) else {},
                })
        except Exception:
            continue

    code: Optional[str] = None
    code_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", cleaned_text, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    elif not tool_calls and "[CODE]" in cleaned_text:
        code = cleaned_text.split("[CODE]", 1)[1].strip()

    if not thought:
        # Take leading prose before any fenced block or tool tag
        lead = re.split(r"```|\[TOOL_CALL\]|\[CODE\]", cleaned_text, maxsplit=1)[0].strip()
        thought = lead or "Analyzed feedback and prepared next action."

    return thought, code, tool_calls


class LiveLLMClient(BaseLLM):
    """
    HTTP client connecting to any OpenAI-compatible endpoint:
    Kunlun GPT-Factory One, Ollama, vLLM, OpenAI, DeepSeek, DashScope, SiliconFlow, etc.
    """

    is_mock = False
    supports_tools = True
    backend_label = "Live OpenAI-Compatible LLM"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 90,
        max_retries: int = 2,
        enable_native_tools: bool = True,
    ):
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or "EMPTY"
        )
        raw_base = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("KUNLUN_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.base_url = raw_base
        self.endpoint = _normalize_completions_url(raw_base)
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_native_tools = enable_native_tools
        self.supports_tools = enable_native_tools

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = json.dumps(payload).encode("utf-8")
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                self.endpoint,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as err:
                err_body = ""
                try:
                    err_body = err.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                finally:
                    try:
                        err.close()
                    except Exception:
                        pass
                # Retry only transient server/rate errors
                if err.code in (429, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(0.8 * (2 ** attempt))
                    last_err = RuntimeError(f"HTTP {err.code}: {err_body[:400]}")
                    continue
                raise RuntimeError(f"LLM HTTP {err.code} from {self.endpoint}: {err_body[:500]}") from err
            except urllib.error.URLError as err:
                last_err = err
                if attempt < self.max_retries:
                    time.sleep(0.6 * (2 ** attempt))
                    continue
                raise RuntimeError(f"LLM connection failed ({self.endpoint}): {err}") from err

        raise RuntimeError(f"LLM request failed after retries: {last_err}")

    def chat_step(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentTurn:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        use_tools = bool(tools and self.enable_native_tools)
        if use_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        t0 = time.perf_counter()
        try:
            data = self._post_chat(payload)
        except RuntimeError as err:
            # Some local OpenAI-compatible servers reject the `tools` parameter with HTTP 400.
            if use_tools and "HTTP 400" in str(err):
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
                self.enable_native_tools = False
                data = self._post_chat(payload)
            else:
                raise
        dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        raw_content = msg.get("content") or ""
        if isinstance(raw_content, list):
            raw_content = "\n".join(
                part.get("text", "") for part in raw_content if isinstance(part, dict)
            )

        native_tool_calls: List[Dict[str, Any]] = []
        for idx, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                parsed_args = {}
            native_tool_calls.append({
                "id": tc.get("id") or f"call_{idx + 1}",
                "name": fn.get("name", ""),
                "arguments": parsed_args if isinstance(parsed_args, dict) else {},
            })

        thought, code, text_tool_calls = _parse_text_response(raw_content)
        tool_calls = native_tool_calls or text_tool_calls
        usage = data.get("usage") or {}
        digest = hashlib.sha256(
            json.dumps({"content": raw_content, "tool_calls": tool_calls}, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return AgentTurn(
            thought=thought,
            code=code,
            tool_calls=tool_calls,
            raw_text=raw_content,
            model=data.get("model") or self.model,
            provider=self.backend_label,
            endpoint=self.endpoint,
            is_mock=False,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=dt_ms,
            response_sha256=digest,
        )

    def generate_patch(self, prompt: str, system_prompt: str) -> Tuple[str, str]:
        turn = self.chat_step(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tools=None,
        )
        return turn.thought, (turn.code or turn.raw_text.strip())

    def _extract_thought_and_code(self, raw_text: str) -> Tuple[str, str]:
        thought, code, _ = _parse_text_response(raw_text)
        return thought, (code or raw_text.strip())


class DeterministicMockLLM(BaseLLM):
    """
    Deterministic Pipeline Smoke Simulator (NOT real LLM reasoning).

    Used strictly for zero-dependency offline verification of the execution loop,
    AST validation, sandbox isolation, diff generation, checkpointing, and WebUI
    streaming. For evaluating autonomous reasoning on unseen problems, use LiveLLMClient.
    """

    is_mock = True
    supports_tools = True
    backend_label = "Deterministic Pipeline Smoke Mock (NOT Real LLM)"
    model = "deterministic-pipeline-mock-v2"

    def __init__(self):
        self._react_step_counter = 0

    def chat_step(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentTurn:
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        lowered = joined.lower()

        # Multi-file Case 4 smoke sequence using genuine ReAct tool calls
        if tools and ("scheduler.py" in lowered or "dispatch_batch" in lowered or "precision.py" in lowered):
            executed_ids = {
                m.get("tool_call_id")
                for m in messages
                if m.get("role") == "tool" and m.get("tool_call_id")
            }

            t0 = time.perf_counter()
            if "mock_1" not in executed_ids:
                tc = [{"id": "mock_1", "name": "read_file", "arguments": {"filepath": "scheduler.py", "start_line": 1, "end_line": 120}}]
                thought = "[SMOKE MOCK] Inspecting scheduler.py to locate priority queue ordering and capacity bounds."
            elif "mock_2" not in executed_ids:
                tc = [{"id": "mock_2", "name": "read_file", "arguments": {"filepath": "precision.py", "start_line": 1, "end_line": 120}}]
                thought = "[SMOKE MOCK] Inspecting precision.py to inspect score summation and NaN/Inf validation."
            elif "mock_3" not in executed_ids:
                new_prec = (
                    "import math\n\n"
                    "def weighted_score(metrics, weights):\n"
                    "    if len(metrics) != len(weights):\n"
                    "        raise ValueError('metrics and weights must have equal length')\n"
                    "    terms = []\n"
                    "    for m, w in zip(metrics, weights):\n"
                    "        if not (math.isfinite(m) and math.isfinite(w)):\n"
                    "            raise ValueError('non-finite metric or weight detected')\n"
                    "        terms.append(float(m) * float(w))\n"
                    "    return math.fsum(terms)\n"
                )
                tc = [{"id": "mock_3", "name": "write_file", "arguments": {"filepath": "precision.py", "content": new_prec}}]
                thought = "[SMOKE MOCK] Fixing precision.py: adding length/finiteness checks and math.fsum compensated summation."
            elif "mock_4" not in executed_ids:
                new_sched = (
                    "from precision import weighted_score\n\n"
                    "def dispatch_batch(tasks, max_tokens):\n"
                    "    if max_tokens < 0:\n"
                    "        raise ValueError('max_tokens must be non-negative')\n"
                    "    valid = []\n"
                    "    for t in tasks:\n"
                    "        if t.get('tokens', 0) <= 0:\n"
                    "            raise ValueError('task token count must be positive')\n"
                    "        score = weighted_score(t['metrics'], t['weights'])\n"
                    "        valid.append((score, t))\n"
                    "    valid.sort(key=lambda item: (-item[0], item[1]['id']))\n"
                    "    selected = []\n"
                    "    used = 0\n"
                    "    for _, t in valid:\n"
                    "        if used + t['tokens'] <= max_tokens:\n"
                    "            selected.append(dict(t))\n"
                    "            used += t['tokens']\n"
                    "    return selected\n"
                )
                tc = [{"id": "mock_4", "name": "write_file", "arguments": {"filepath": "scheduler.py", "content": new_sched}}]
                thought = "[SMOKE MOCK] Fixing scheduler.py: sorting by descending score then id, enforcing token budget and non-negative validation."
            else:
                tc = [{"id": "mock_5", "name": "run_tests", "arguments": {}}]
                thought = "[SMOKE MOCK] Running TDD gate after patching both precision.py and scheduler.py."

            return AgentTurn(
                thought=thought,
                tool_calls=tc,
                raw_text=thought,
                model=self.model,
                provider=self.backend_label,
                is_mock=True,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                response_sha256=hashlib.sha256(thought.encode("utf-8")).hexdigest(),
            )

        return super().chat_step(messages, tools=None)

    def generate_patch(self, prompt: str, system_prompt: str) -> Tuple[str, str]:
        lower = prompt.lower()

        # Case 1: SpMM CSR Boundary Offset, Negative Index Guard & Compensated Summation
        if "spmm_csr_operator" in lower or "col_indices" in lower:
            thought = (
                "[SMOKE MOCK] Root Cause: CSR SpMM did not validate row_ptr monotonicity/length, "
                "lacked negative/out-of-range column index guards (Python dense_b[-1] silent wrap-around), "
                "and used naive floating accumulation without finiteness checks. "
                "Fix Strategy: Validate CSR bounds (raising ValueError/IndexError on invalid pointers/indices), "
                "fast-skip empty rows (start == end), and use math.fsum for exact accumulation."
            )
            code = '''# Patched Kernel: SpMM CSR with Strict Bounds & Compensated Accumulation
import math


def spmm_csr_operator(row_ptr, col_indices, values, dense_b, num_rows, num_cols):
    """
    Sparse Matrix-Matrix Multiplication (C = A_csr * B_dense).
    Enforces CSR structure invariants, negative/out-of-range index guards,
    empty-row fast paths, and math.fsum accumulation.
    """
    if num_rows < 0 or num_cols < 0:
        raise ValueError("Matrix dimensions must be non-negative")
    if len(row_ptr) != num_rows + 1:
        raise ValueError(f"row_ptr length {len(row_ptr)} != num_rows + 1 ({num_rows + 1})")
    if len(col_indices) != len(values):
        raise ValueError("col_indices and values length mismatch")
    if row_ptr and row_ptr[0] != 0:
        raise ValueError("row_ptr must start at 0")

    num_b_rows = len(dense_b)
    for row_b in dense_b:
        if len(row_b) != num_cols:
            raise ValueError("dense_b column dimension mismatch")

    result = [[0.0 for _ in range(num_cols)] for _ in range(num_rows)]

    for r in range(num_rows):
        start_idx = row_ptr[r]
        end_idx = row_ptr[r + 1]
        if start_idx < 0 or end_idx < start_idx or end_idx > len(col_indices):
            raise IndexError(f"Invalid CSR row_ptr slice [{start_idx}, {end_idx}) at row {r}")
        if start_idx == end_idx:
            continue

        col_terms = [[] for _ in range(num_cols)]
        for k_idx in range(start_idx, end_idx):
            c_idx = col_indices[k_idx]
            val = values[k_idx]
            if not math.isfinite(val):
                raise ValueError("Non-finite sparse matrix value")
            if not (0 <= c_idx < num_b_rows):
                raise IndexError(f"Column index {c_idx} out of bounds for dense_b with {num_b_rows} rows")
            row_b = dense_b[c_idx]
            for j in range(num_cols):
                b_val = row_b[j]
                if not math.isfinite(b_val):
                    raise ValueError("Non-finite dense matrix value")
                col_terms[j].append(val * b_val)

        for j in range(num_cols):
            result[r][j] = math.fsum(col_terms[j])

    return result
'''
            return thought, code

        # Case 2: IEEE-754 Float32 Vector Scale Kernel (Real float32 semantics, NO round(..., 6))
        if "vector_scale_kernel" in lower:
            thought = (
                "[SMOKE MOCK] Root Cause: Vector scale kernel omitted scalar `scale` multiplication, "
                "did not validate shape mismatch or non-finite inputs, and must preserve IEEE-754 float32 "
                "precision without destructive decimal round(..., 6) truncation on sub-micro magnitudes. "
                "Fix Strategy: Validate equal lengths and finiteness, compute float32-rounded (a + b) * scale "
                "via struct float32 packing for bit-exact IEEE-754 float32 alignment."
            )
            code = '''# Patched Vector Scale Operator with IEEE-754 Float32 Alignment
import math
import struct


def _to_f32(x: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(x)))[0]


def vector_scale_kernel(tensor_a, tensor_b, scale=1.0):
    """
    Element-wise fused vector addition and scaling aligned to IEEE-754 float32 arithmetic.
    """
    if len(tensor_a) != len(tensor_b):
        raise ValueError("Tensor shape mismatch between input vectors")
    if not math.isfinite(scale):
        raise ValueError("Non-finite scale factor")

    f_scale = _to_f32(scale)
    out = []
    for a, b in zip(tensor_a, tensor_b):
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("Non-finite element in input vector")
        s = _to_f32(_to_f32(a) + _to_f32(b))
        out.append(_to_f32(s * f_scale))
    return out
'''
            return thought, code

        # Case 3: GQA KV-Cache Broadcast with Deep Buffer Isolation & Validation
        if "gqa_kv_broadcast_kernel" in lower:
            thought = (
                "[SMOKE MOCK] Root Cause: GQA broadcast returned kv_tensor unchanged without validating "
                "head divisibility, checking tensor shape [batch, num_kv_heads, seq_len, head_dim], or "
                "isolating expanded head buffers from aliased list references. "
                "Fix Strategy: Validate positive head counts and divisibility (num_query_heads % num_kv_heads == 0), "
                "check per-batch KV head counts, and deep-copy sequence/head vectors during repeat-interleave."
            )
            code = '''# Patched GQA KV-Cache Repeat-Interleave Broadcast Kernel
def gqa_kv_broadcast_kernel(kv_tensor, num_query_heads=12, num_kv_heads=2):
    """
    GQA KV Cache repeat-interleave broadcast expansion.
    Expands kv_tensor of shape [batch, num_kv_heads, seq_len, head_dim]
    to [batch, num_query_heads, seq_len, head_dim] with independent head buffers.
    """
    if num_kv_heads <= 0 or num_query_heads <= 0:
        raise ValueError("Head counts must be positive integers")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    group_size = num_query_heads // num_kv_heads
    broadcasted = []

    for b_idx, batch_item in enumerate(kv_tensor):
        if len(batch_item) != num_kv_heads:
            raise ValueError(
                f"Batch {b_idx} has {len(batch_item)} KV heads, expected {num_kv_heads}"
            )
        batch_heads = []
        for kv_head in batch_item:
            for _ in range(group_size):
                # Copy inner [seq_len][head_dim] slices so downstream mutations do not alias across heads
                batch_heads.append([list(seq_vec) for seq_vec in kv_head])
        broadcasted.append(batch_heads)

    return broadcasted
'''
            return thought, code

        thought = (
            "[SMOKE MOCK] Unseen task prompt received in Deterministic Pipeline Smoke Mode. "
            "Offline smoke mock only replays fixture patches for built-in benchmark signatures; "
            "configure a Live LLM endpoint (Kunlun GPT-Factory One / Ollama / OpenAI / DeepSeek) "
            "for autonomous reasoning on arbitrary tasks."
        )
        code = "# Unchanged: Offline smoke mock does not synthesize code for unseen tasks.\n"
        return thought, code


def create_llm_backend(
    mode: str = "auto",
    preset: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLM:
    """
    Factory selecting a live OpenAI-compatible client or the offline deterministic smoke mock.

    Args:
        mode: 'auto' (live if key/endpoint/preset provided, else mock), 'live', or 'mock'.
        preset: Optional key from HARDWARE_PRESETS (e.g., 'kunlun-gateway', 'ollama', 'vllm').
        base_url: Explicit OpenAI-compatible base URL.
        api_key: Optional API key (local endpoints like Ollama/vLLM/GPT-Factory One do not require a real key).
        model: Model identifier.
    """
    normalized_mode = (mode or "auto").strip().lower()
    if normalized_mode == "mock":
        return DeterministicMockLLM()

    preset_cfg = HARDWARE_PRESETS.get(preset) if preset else None
    resolved_key = (
        api_key
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
    )
    resolved_base = (
        base_url
        or os.getenv("LLM_BASE_URL")
        or os.getenv("KUNLUN_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or (preset_cfg["base_url"] if preset_cfg else None)
    )
    resolved_model = (
        model
        or os.getenv("LLM_MODEL")
        or (preset_cfg["default_model"] if preset_cfg else None)
    )

    has_live_config = bool(resolved_key or resolved_base or preset_cfg)
    if normalized_mode == "live":
        if not has_live_config:
            raise ValueError(
                "Live LLM mode requested, but no API key, base_url, or preset was provided. "
                "Set LLM_BASE_URL / OPENAI_API_KEY or pass --base-url / --preset."
            )
        return LiveLLMClient(
            api_key=resolved_key or "EMPTY",
            base_url=resolved_base or "https://api.openai.com/v1",
            model=resolved_model or "gpt-4o-mini",
        )

    if has_live_config:
        return LiveLLMClient(
            api_key=resolved_key or "EMPTY",
            base_url=resolved_base or "https://api.openai.com/v1",
            model=resolved_model or "gpt-4o-mini",
        )

    return DeterministicMockLLM()
