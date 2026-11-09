"""
Tools and Execution Sandbox for Autonomous Code Synthesis & Verification Agent.
Provides AST syntax validation, sandboxed TDD gate execution, atomic file I/O, and diff analysis.
"""

import os
import sys
import ast
import difflib
import subprocess
from typing import Tuple, Dict, Any, List

class ToolBox:
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)

    def _resolve_path(self, filepath: str) -> str:
        if os.path.isabs(filepath):
            return filepath
        return os.path.join(self.workspace_root, filepath)

    def read_source(self, filepath: str) -> str:
        """Safely read code from workspace."""
        abs_path = self._resolve_path(filepath)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Source file not found: {filepath}")
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()

    def write_source(self, filepath: str, content: str) -> None:
        """Safely write code to workspace with directory creation."""
        abs_path = self._resolve_path(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

    def validate_syntax(self, code_str: str) -> Tuple[bool, str]:
        """
        Validate Python code syntax using AST parsing before writing to disk.
        Prevents broken patches from polluting the environment.
        """
        try:
            ast.parse(code_str)
            return True, "AST syntax check passed."
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}\n  {e.text}"

    def run_tdd_gate(self, test_file: str, timeout: int = 15) -> Dict[str, Any]:
        """
        Execute unit tests in an isolated subprocess with watchdog timeout.
        Returns execution result, pass/fail status, duration, and error traceback.
        """
        abs_test_path = self._resolve_path(test_file)
        if not os.path.exists(abs_test_path):
            return {
                "passed": False,
                "exit_code": -1,
                "summary": "Test file does not exist",
                "output": f"File not found: {test_file}",
                "traceback": ""
            }

        cmd = [sys.executable, "-m", "unittest", abs_test_path]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(abs_test_path) or self.workspace_root
            )
            raw_output = (res.stdout or "") + (res.stderr or "")
            passed = (res.returncode == 0)

            # Extract key failure trace
            traceback_lines = []
            capture = False
            for line in raw_output.splitlines():
                if "FAIL:" in line or "ERROR:" in line or "Traceback" in line:
                    capture = True
                if capture:
                    traceback_lines.append(line)

            tb_snippet = "\n".join(traceback_lines) if traceback_lines else raw_output.strip()

            return {
                "passed": passed,
                "exit_code": res.returncode,
                "summary": "ALL_TESTS_PASS" if passed else "GATE_FAILED",
                "output": raw_output.strip(),
                "traceback": tb_snippet
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "exit_code": 124,
                "summary": "WATCHDOG_TIMEOUT",
                "output": f"Execution timed out after {timeout} seconds (infinite loop protection triggered).",
                "traceback": f"TimeoutExpired: Execution exceeded safety guardrail of {timeout}s."
            }
        except Exception as e:
            return {
                "passed": False,
                "exit_code": -1,
                "summary": "SUBPROCESS_ERROR",
                "output": str(e),
                "traceback": str(e)
            }

    def compute_diff(self, original_code: str, new_code: str, filename: str = "patch.py") -> str:
        """Generate human-readable unified diff between original and patched code."""
        orig_lines = original_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}"
        )
        return "".join(diff)
