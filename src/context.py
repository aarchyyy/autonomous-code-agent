"""
Sliding-Window Context Compactor & Token Budget Estimator for Long-Horizon ReAct Loops.

Prevents multi-file repository exploration from overflowing local 8K-32K LLM context
windows (e.g., Ollama / vLLM / Kunlun Gateway) while strictly preserving:
1. System prompt (`messages[0]`) and initial task specification (`messages[1]`).
2. Structural pairing between `assistant(tool_calls)` and `role="tool"` responses.
3. Full verbatim fidelity of the most recent K conversation messages.
4. Auditable structured summaries of older bulky `read_file` / `run_tests` observations.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """Heuristic token estimator (~3.6 chars/token + tool call schema overhead)."""
    content = str(msg.get("content") or "")
    chars = len(content)
    if msg.get("tool_calls"):
        chars += len(json.dumps(msg["tool_calls"], ensure_ascii=False))
    return max(4, int(chars / 3.6) + 6)


def estimate_conversation_tokens(messages: List[Dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def _summarize_tool_payload(raw_text: str, max_chars: int = 260) -> str:
    """Compresses a bulky JSON tool observation into a deterministic compact digest."""
    if len(raw_text) <= max_chars:
        return raw_text

    # Handle text-protocol wrapper `[TOOL_RESULT name=...]\n{...}`
    prefix = ""
    json_candidate = raw_text
    if raw_text.startswith("[TOOL_RESULT ") and "\n" in raw_text:
        first_line, rest = raw_text.split("\n", 1)
        prefix = first_line + "\n"
        json_candidate = rest

    stripped = json_candidate.replace("\n... [truncated]", "").strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            if "content" in obj and "filepath" in obj:
                compact = {
                    "ok": obj.get("ok", True),
                    "compacted": True,
                    "filepath": obj.get("filepath"),
                    "lines_read": f"{obj.get('start_line', 1)}..{obj.get('end_line', '?')}/{obj.get('total_lines', '?')}",
                    "note": "Stale read_file output compacted; re-read if exact lines are still needed.",
                }
                return prefix + json.dumps(compact, ensure_ascii=False)
            if "test_result" in obj and isinstance(obj["test_result"], dict):
                tr = obj["test_result"]
                compact = {
                    "ok": obj.get("ok", True),
                    "compacted": True,
                    "summary": tr.get("summary"),
                    "tests_run": tr.get("tests_run"),
                    "passed_count": tr.get("passed_count"),
                    "failures": tr.get("failures"),
                    "errors": tr.get("errors"),
                    "error_signature": tr.get("error_signature"),
                }
                return prefix + json.dumps(compact, ensure_ascii=False)
            if "diff" in obj and "filepath" in obj:
                diff_lines = len(str(obj.get("diff") or "").splitlines())
                compact = {
                    "ok": obj.get("ok", True),
                    "compacted": True,
                    "filepath": obj.get("filepath"),
                    "diff_lines": diff_lines,
                    "sha256_after": str(obj.get("sha256_after") or "")[:12],
                }
                return prefix + json.dumps(compact, ensure_ascii=False)
    except Exception:
        pass

    head = raw_text[: max_chars - 60].rstrip()
    return f"{head} ... [compacted {len(raw_text)} chars]"


class ContextWindowManager:
    """Manages deterministic sliding-window observation compaction for ReAct agents."""

    def __init__(
        self,
        max_context_tokens: int = 8000,
        keep_recent_messages: int = 6,
        min_compact_chars: int = 280,
    ):
        self.max_context_tokens = max(512, int(max_context_tokens))
        self.keep_recent_messages = max(2, int(keep_recent_messages))
        self.min_compact_chars = max(120, int(min_compact_chars))
        self.compactions_performed = 0
        self.tokens_reclaimed = 0

    def compact_if_needed(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        before_tokens = estimate_conversation_tokens(messages)
        if before_tokens <= self.max_context_tokens or len(messages) <= 2 + self.keep_recent_messages:
            return messages, {
                "before_tokens": before_tokens,
                "after_tokens": before_tokens,
                "reclaimed_tokens": 0,
                "compacted": 0,
            }

        cutoff = len(messages) - self.keep_recent_messages
        # Never split an assistant(tool_calls) turn from its subsequent tool messages
        while cutoff > 2 and messages[cutoff].get("role") == "tool":
            cutoff -= 1

        updated: List[Dict[str, Any]] = []
        changed = False
        for idx, msg in enumerate(messages):
            if idx < 2 or idx >= cutoff:
                updated.append(msg)
                continue

            role = msg.get("role", "")
            content = str(msg.get("content") or "")
            is_tool_obs = role == "tool" or (role == "user" and content.startswith("[TOOL_RESULT "))
            if is_tool_obs and len(content) >= self.min_compact_chars and '"compacted": true' not in content:
                new_msg = dict(msg)
                new_msg["content"] = _summarize_tool_payload(content, max_chars=self.min_compact_chars)
                updated.append(new_msg)
                changed = True
            else:
                updated.append(msg)

        after_tokens = estimate_conversation_tokens(updated)
        reclaimed = max(0, before_tokens - after_tokens)
        if changed and reclaimed > 0:
            self.compactions_performed += 1
            self.tokens_reclaimed += reclaimed

        return updated, {
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "reclaimed_tokens": reclaimed,
            "compacted": 1 if (changed and reclaimed > 0) else 0,
        }
