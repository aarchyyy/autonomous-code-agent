"""
Tools and Execution Sandbox Interface for Autonomous Code Synthesis & Verification Agent.
Provides workspace-confined I/O, AST syntax/policy validation, unified diff computation,
SHA-256 test-gate integrity verification, and ReAct tool dispatch.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .sandbox import (
    ASTPolicyError,
    ASTPolicyInspector,
    IntegrityMonitor,
    IntegrityViolationError,
    IsolatedTestRunner,
    PathGuard,
    PathSecurityError,
    SandboxPolicy,
    sha256_text,
)

__all__ = [
    "ToolBox",
    "PathSecurityError",
    "IntegrityViolationError",
    "ASTPolicyError",
    "TOOL_DEFINITIONS",
    "format_tools_for_prompt",
]

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List non-hidden files inside the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {"type": "string", "description": "Relative subdirectory (default '.')"},
                    "max_entries": {"type": "integer", "description": "Max entries to return (default 150)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read lines from an allowed file in the workspace (hidden holdout files are inaccessible).",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Relative path inside workspace"},
                    "start_line": {"type": "integer", "description": "1-based start line (default 1)"},
                    "end_line": {"type": "integer", "description": "1-based inclusive end line (default 200)"},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a regex or substring across visible workspace files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex or text pattern to search"},
                    "subdir": {"type": "string", "description": "Relative subdirectory (default '.')"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": "Apply a targeted search-and-replace patch to an authorized writable file (AST pre-checked).",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Target writable file path"},
                    "old_text": {"type": "string", "description": "Exact existing snippet to replace (must match uniquely)"},
                    "new_text": {"type": "string", "description": "Replacement snippet"},
                },
                "required": ["filepath", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Overwrite an authorized writable Python file with complete validated source code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Target writable file path"},
                    "content": {"type": "string", "description": "Complete Python source code"},
                },
                "required": ["filepath", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the immutable TDD test gate in the isolated ephemeral sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_file": {"type": "string", "description": "Optional specific test file or directory"},
                },
            },
        },
    },
]


def format_tools_for_prompt() -> str:
    lines = ["Available Sandbox Tools (call via JSON block `[TOOL_CALL] {\"name\": ..., \"arguments\": {...}} [/TOOL_CALL]`):"]
    for item in TOOL_DEFINITIONS:
        fn = item["function"]
        props = fn["parameters"].get("properties", {})
        args_desc = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
        lines.append(f"- {fn['name']}({args_desc}): {fn['description']}")
    return "\n".join(lines)


class ToolBox:
    def __init__(
        self,
        workspace_root: str = ".",
        writable_files: Optional[Iterable[str]] = None,
        protected_files: Optional[Iterable[str]] = None,
        hidden_files: Optional[Iterable[str]] = None,
        strict_ast: bool = False,
        use_ephemeral_copy: bool = True,
        default_test_file: Optional[str] = None,
    ):
        self.workspace_root = os.path.realpath(workspace_root)
        self.policy = SandboxPolicy(
            workspace_root=self.workspace_root,
            writable_files=set(writable_files) if writable_files is not None else None,
            protected_files=set(protected_files or ()),
            hidden_files=set(hidden_files or ()),
            block_dangerous_ast=strict_ast,
            use_ephemeral_copy=use_ephemeral_copy,
        ).normalized()
        self.guard = PathGuard(
            workspace_root=self.policy.workspace_root,
            writable_files=self.policy.writable_files,
            protected_files=self.policy.protected_files,
            hidden_files=self.policy.hidden_files,
        )
        self.runner = IsolatedTestRunner(self.policy)
        monitored = set(self.policy.protected_files) | set(self.policy.hidden_files)
        self.integrity = IntegrityMonitor(self.policy.workspace_root, monitored)
        self.default_test_file = default_test_file

    def _resolve_path(self, filepath: str, for_write: bool = False) -> str:
        if for_write:
            abs_path, _ = self.guard.check_write(filepath)
        else:
            abs_path, _ = self.guard.check_read(filepath)
        return abs_path

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        return self.integrity.verify()

    def read_source(self, filepath: str) -> str:
        """Safely read code from inside workspace, blocking escapes and hidden holdouts."""
        abs_path, _ = self.guard.check_read(filepath)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Source file not found: {filepath}")
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()

    def write_source(self, filepath: str, content: str) -> None:
        """Safely write code inside workspace subject to write allowlist and protected-file ACL."""
        abs_path, rel = self.guard.check_write(filepath)
        if rel.endswith(".py") and self.policy.block_dangerous_ast:
            ok, msg = self.validate_syntax(content, strict=True)
            if not ok:
                raise ASTPolicyError(msg)
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{abs_path}.acs_tmp_{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, abs_path)

    def validate_syntax(self, code_str: str, strict: Optional[bool] = None) -> Tuple[bool, str]:
        """Validate Python syntax using AST parsing (and strict static security policy if enabled)."""
        use_strict = self.policy.block_dangerous_ast if strict is None else strict
        return ASTPolicyInspector.inspect(code_str, strict=use_strict)

    def list_files(self, subdir: str = ".", max_entries: int = 150) -> List[str]:
        abs_dir, _ = self.guard.check_read(subdir, allow_dir=True)
        if not os.path.isdir(abs_dir):
            raise FileNotFoundError(f"Directory not found: {subdir}")

        results: List[str] = []
        skip_dirs = {".git", "__pycache__", ".pytest_cache", ".acs_runs", ".venv", "venv", "node_modules"}
        for root, dirs, files in os.walk(abs_dir):
            dirs[:] = sorted(d for d in dirs if d not in skip_dirs)
            for fname in sorted(files):
                full = os.path.join(root, fname)
                try:
                    _, rel = self.guard.check_read(full)
                except PathSecurityError:
                    continue
                results.append(rel)
                if len(results) >= max_entries:
                    return results
        return results

    def read_file_lines(self, filepath: str, start_line: int = 1, end_line: int = 200) -> Dict[str, Any]:
        content = self.read_source(filepath)
        lines = content.splitlines()
        total = len(lines)
        start = max(1, int(start_line))
        end = min(total, max(start, int(end_line)))
        numbered = [f"{i}: {lines[i - 1]}" for i in range(start, end + 1)]
        return {
            "filepath": filepath,
            "total_lines": total,
            "start_line": start,
            "end_line": end,
            "content": "\n".join(numbered),
        }

    def search_code(self, pattern: str, subdir: str = ".", max_hits: int = 40) -> List[Dict[str, Any]]:
        files = self.list_files(subdir=subdir, max_entries=300)
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))

        hits: List[Dict[str, Any]] = []
        for rel in files:
            try:
                text = self.read_source(rel)
            except (OSError, UnicodeDecodeError, PathSecurityError):
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    hits.append({"file": rel, "line": idx, "text": line[:200]})
                    if len(hits) >= max_hits:
                        return hits
        return hits

    def patch_source(self, filepath: str, old_text: str, new_text: str) -> Dict[str, Any]:
        before = self.read_source(filepath)
        count = before.count(old_text)
        if count == 0:
            raise ValueError(f"Target snippet not found in {filepath}. Read the file to verify exact lines.")
        if count > 1:
            raise ValueError(f"Target snippet matched {count} times in {filepath}; include more context.")

        after = before.replace(old_text, new_text, 1)
        if filepath.endswith(".py"):
            ok, msg = self.validate_syntax(after)
            if not ok:
                raise ASTPolicyError(msg)

        self.write_source(filepath, after)
        diff = self.compute_diff(before, after, filename=filepath)
        return {
            "filepath": filepath,
            "diff": diff,
            "sha256_after": sha256_text(after),
        }

    def run_tdd_gate(
        self,
        test_file: str,
        timeout: int = 15,
        include_hidden: bool = False,
    ) -> Dict[str, Any]:
        """Execute unit tests in the isolated sandbox and verify test-gate file integrity."""
        ok_before, violations_before = self.verify_integrity()
        if not ok_before:
            raise IntegrityViolationError(
                "Pre-gate integrity violation: " + "; ".join(violations_before)
            )

        res = self.runner.run_tests(test_file=test_file, timeout=timeout, include_hidden=include_hidden)

        ok_after, violations_after = self.verify_integrity()
        if not ok_after:
            raise IntegrityViolationError(
                "Post-gate integrity violation: " + "; ".join(violations_after)
            )
        res["integrity_verified"] = True
        return res

    def compute_diff(self, original_code: str, new_code: str, filename: str = "patch.py") -> str:
        """Generate human-readable unified diff between original and patched code."""
        orig_lines = original_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
        return "".join(diff)

    def dispatch_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes an agent ReAct tool call safely and returns structured JSON output."""
        try:
            if name == "list_files":
                files = self.list_files(
                    subdir=arguments.get("subdir", "."),
                    max_entries=int(arguments.get("max_entries", 150)),
                )
                return {"ok": True, "files": files}

            if name == "read_file":
                res = self.read_file_lines(
                    filepath=arguments["filepath"],
                    start_line=int(arguments.get("start_line", 1)),
                    end_line=int(arguments.get("end_line", 200)),
                )
                return {"ok": True, **res}

            if name == "search_code":
                hits = self.search_code(
                    pattern=arguments["pattern"],
                    subdir=arguments.get("subdir", "."),
                )
                return {"ok": True, "hits": hits}

            if name == "patch_file":
                res = self.patch_source(
                    filepath=arguments["filepath"],
                    old_text=arguments["old_text"],
                    new_text=arguments["new_text"],
                )
                return {"ok": True, **res}

            if name == "write_file":
                filepath = arguments["filepath"]
                content = arguments["content"]
                before = ""
                try:
                    before = self.read_source(filepath)
                except FileNotFoundError:
                    pass
                if filepath.endswith(".py"):
                    valid, msg = self.validate_syntax(content)
                    if not valid:
                        return {"ok": False, "error": msg}
                self.write_source(filepath, content)
                diff = self.compute_diff(before, content, filename=filepath)
                return {
                    "ok": True,
                    "filepath": filepath,
                    "diff": diff,
                    "sha256_after": sha256_text(content),
                }

            if name == "run_tests":
                target = arguments.get("test_file") or self.default_test_file
                if not target:
                    return {"ok": False, "error": "No test_file configured."}
                gate = self.run_tdd_gate(target)
                return {"ok": True, "test_result": gate}

            return {"ok": False, "error": f"Unknown tool: {name}"}
        except (PathSecurityError, ASTPolicyError, IntegrityViolationError, FileNotFoundError, ValueError, KeyError) as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        except Exception as e:
            return {"ok": False, "error": f"ToolExecutionError: {e}"}

    @staticmethod
    def format_tool_result_for_llm(result: Dict[str, Any], limit: int = 4000) -> str:
        raw = json.dumps(result, ensure_ascii=False, indent=2)
        if len(raw) > limit:
            return raw[:limit] + "\n... [truncated]"
        return raw
