"""
Execution Sandbox, Path Containment, AST Policy Guard, and Integrity Verification.

Provides layered isolation for autonomous code synthesis:
1. Canonical PathGuard: Prevents path traversal, symlink escapes, writes outside
   authorized target allowlists, and reads of hidden holdout tests.
2. IntegrityMonitor: SHA-256 verification of protected test files before and after steps.
3. ASTPolicyInspector: Syntax check plus optional static import/call restriction.
4. IsolatedTestRunner: Executes unit tests inside an ephemeral read-only-protected
   staging copy with scrubbed environment variables, isolated Python mode (-I),
   POSIX resource limits (RLIMIT_CPU, RLIMIT_FSIZE, and RLIMIT_AS on Linux),
   and process-group kill on timeout.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class PathSecurityError(PermissionError):
    """Raised when a file operation attempts to escape or violate sandbox path policy."""


class IntegrityViolationError(RuntimeError):
    """Raised when protected or hidden files are modified unexpectedly."""


class ASTPolicyError(ValueError):
    """Raised when code violates static AST security policies."""


_SECRET_ENV_PATTERNS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASS",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
    "SESSION",
    "PRIVATE",
)

_IGNORED_COPY_NAMES = frozenset({
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".acs_runs",
    ".venv",
    "venv",
    "node_modules",
})

_FORBIDDEN_IMPORTS_STRICT = frozenset({
    "os",
    "subprocess",
    "sys",
    "ctypes",
    "socket",
    "shutil",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "pathlib",
    "importlib",
    "multiprocessing",
    "threading",
    "pty",
    "fcntl",
    "resource",
    "signal",
    "tempfile",
    "pickle",
    "marshal",
})

_FORBIDDEN_CALLS_STRICT = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "breakpoint",
})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class SandboxPolicy:
    workspace_root: str
    writable_files: Optional[Set[str]] = None
    protected_files: Set[str] = field(default_factory=set)
    hidden_files: Set[str] = field(default_factory=set)
    block_dangerous_ast: bool = False
    use_ephemeral_copy: bool = True
    cpu_limit_sec: int = 15
    file_size_limit_mb: int = 16
    memory_limit_mb: int = 512

    def normalized(self) -> "SandboxPolicy":
        real_ws = os.path.realpath(self.workspace_root)
        guard = PathGuard(real_ws)

        def _norm_set(paths: Optional[Iterable[str]]) -> Optional[Set[str]]:
            if paths is None:
                return None
            out: Set[str] = set()
            for p in paths:
                _, rel = guard.canonicalize(p)
                out.add(rel)
            return out

        return SandboxPolicy(
            workspace_root=real_ws,
            writable_files=_norm_set(self.writable_files),
            protected_files=_norm_set(self.protected_files) or set(),
            hidden_files=_norm_set(self.hidden_files) or set(),
            block_dangerous_ast=self.block_dangerous_ast,
            use_ephemeral_copy=self.use_ephemeral_copy,
            cpu_limit_sec=self.cpu_limit_sec,
            file_size_limit_mb=self.file_size_limit_mb,
            memory_limit_mb=self.memory_limit_mb,
        )


class PathGuard:
    """Enforces workspace confinement, symlink escape prevention, and ACLs."""

    def __init__(
        self,
        workspace_root: str,
        writable_files: Optional[Set[str]] = None,
        protected_files: Optional[Set[str]] = None,
        hidden_files: Optional[Set[str]] = None,
    ):
        self.workspace_root = os.path.realpath(workspace_root)
        self.writable_files = set(writable_files) if writable_files is not None else None
        self.protected_files = set(protected_files or ())
        self.hidden_files = set(hidden_files or ())

    def canonicalize(self, filepath: str) -> Tuple[str, str]:
        if not filepath or "\x00" in filepath:
            raise PathSecurityError(f"Invalid file path: {filepath!r}")

        candidate = filepath if os.path.isabs(filepath) else os.path.join(self.workspace_root, filepath)
        real_candidate = os.path.realpath(candidate)

        try:
            common = os.path.commonpath([self.workspace_root, real_candidate])
        except ValueError as exc:
            raise PathSecurityError(f"Path outside workspace boundary: {filepath}") from exc

        if common != self.workspace_root:
            raise PathSecurityError(
                f"Access denied: {filepath!r} resolves outside workspace {self.workspace_root!r}"
            )

        rel = os.path.relpath(real_candidate, self.workspace_root).replace(os.sep, "/")
        return real_candidate, rel

    def _is_internal_path(self, rel: str) -> bool:
        parts = rel.split("/")
        return ".git" in parts or ".acs_runs" in parts

    def check_read(self, filepath: str, allow_dir: bool = False) -> Tuple[str, str]:
        real_path, rel = self.canonicalize(filepath)
        if not allow_dir and rel == ".":
            raise PathSecurityError("Cannot read workspace root as a file.")
        if self._is_internal_path(rel):
            raise PathSecurityError(f"Access to internal path is denied: {rel}")
        if rel in self.hidden_files:
            raise PathSecurityError(f"Access denied: holdout file {rel!r} is hidden from agent.")
        return real_path, rel

    def check_write(self, filepath: str) -> Tuple[str, str]:
        real_path, rel = self.canonicalize(filepath)
        if rel == ".":
            raise PathSecurityError("Cannot overwrite workspace root directory.")
        if self._is_internal_path(rel):
            raise PathSecurityError(f"Write to internal path is denied: {rel}")
        if rel in self.hidden_files:
            raise PathSecurityError(f"Write denied: {rel!r} is a hidden holdout gate.")
        if rel in self.protected_files:
            raise PathSecurityError(f"Write denied: {rel!r} is an immutable protected file.")
        if self.writable_files is not None and rel not in self.writable_files:
            allowed = sorted(self.writable_files)
            raise PathSecurityError(
                f"Write denied: {rel!r} is not in writable allowlist {allowed}."
            )
        return real_path, rel


class ASTPolicyInspector:
    """Validates Python syntax and optionally checks static import/call policies."""

    @staticmethod
    def inspect(code_str: str, strict: bool = False) -> Tuple[bool, str]:
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}\n  {e.text}"

        if not strict:
            return True, "AST syntax check passed."

        violations: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in _FORBIDDEN_IMPORTS_STRICT:
                        violations.append(f"line {node.lineno}: forbidden import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in _FORBIDDEN_IMPORTS_STRICT:
                        violations.append(f"line {node.lineno}: forbidden from-import '{node.module}'")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS_STRICT:
                    violations.append(f"line {node.lineno}: forbidden builtin call '{node.func.id}()'")

        if violations:
            return False, "AST security policy violation: " + "; ".join(violations[:5])
        return True, "AST syntax and strict static policy checks passed."


class IntegrityMonitor:
    """Tracks and verifies SHA-256 digests of immutable test and spec files."""

    def __init__(self, workspace_root: str, monitored_files: Iterable[str]):
        self.workspace_root = os.path.realpath(workspace_root)
        self.guard = PathGuard(self.workspace_root)
        self.baselines: Dict[str, Optional[str]] = {}
        for f in monitored_files:
            real_path, rel = self.guard.canonicalize(f)
            self.baselines[rel] = sha256_file(real_path) if os.path.exists(real_path) else None

    def verify(self) -> Tuple[bool, List[str]]:
        violations: List[str] = []
        for rel, expected_hash in self.baselines.items():
            real_path = os.path.join(self.workspace_root, rel)
            if expected_hash is None:
                if os.path.exists(real_path):
                    violations.append(f"Protected file unexpectedly created: {rel}")
                continue
            if not os.path.exists(real_path):
                violations.append(f"Protected file deleted: {rel}")
                continue
            actual = sha256_file(real_path)
            if actual != expected_hash:
                violations.append(
                    f"Protected file modified ({rel}): expected {expected_hash[:12]}, got {actual[:12]}"
                )
        return len(violations) == 0, violations

    def snapshot_hashes(self) -> Dict[str, Optional[str]]:
        return dict(self.baselines)


def _sanitized_subprocess_env() -> Dict[str, str]:
    safe_env: Dict[str, str] = {}
    for key in ("PATH", "LANG", "LC_ALL", "TERM"):
        if key in os.environ:
            safe_env[key] = os.environ[key]
    for key, val in os.environ.items():
        upper = key.upper()
        if any(pat in upper for pat in _SECRET_ENV_PATTERNS):
            continue
        if key.startswith(("LC_", "XDG_")) and key not in safe_env:
            safe_env[key] = val
    safe_env["PYTHONNOUSERSITE"] = "1"
    safe_env["PYTHONDONTWRITEBYTECODE"] = "1"
    return safe_env


def _handle_remove_readonly(func, path, _exc):
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        func(path)
    except OSError:
        pass


_CHILD_RUNNER_BOOTSTRAP = r"""
import os, sys, resource, unittest, io

workdir = sys.argv[1]
test_dir = sys.argv[2]
test_file = sys.argv[3]
cpu_limit = int(sys.argv[4])
fsize_mb = int(sys.argv[5])
mem_mb = int(sys.argv[6])

applied = []
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    target = cpu_limit if hard == resource.RLIM_INFINITY else min(cpu_limit, hard)
    resource.setrlimit(resource.RLIMIT_CPU, (target, hard))
    applied.append(f"RLIMIT_CPU={target}s")
except Exception:
    pass

try:
    fbytes = fsize_mb * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    target = fbytes if hard == resource.RLIM_INFINITY else min(fbytes, hard)
    resource.setrlimit(resource.RLIMIT_FSIZE, (target, hard))
    applied.append(f"RLIMIT_FSIZE={fsize_mb}MB")
except Exception:
    pass

if sys.platform.startswith("linux") and hasattr(resource, "RLIMIT_AS"):
    try:
        mbytes = mem_mb * 1024 * 1024
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        target = mbytes if hard == resource.RLIM_INFINITY else min(mbytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (target, hard))
        applied.append(f"RLIMIT_AS={mem_mb}MB")
    except Exception:
        pass

for p in (test_dir, workdir):
    if p and p not in sys.path:
        sys.path.insert(0, p)

os.chdir(test_dir)
sys.stderr.write(f"[SANDBOX_RLIMITS] {','.join(applied) or 'none'}\n")
sys.stderr.flush()

loader = unittest.TestLoader()
if os.path.isdir(test_file):
    suite = loader.discover(start_dir=test_file)
else:
    suite = loader.discover(start_dir=test_dir, pattern=os.path.basename(test_file))

runner = unittest.TextTestRunner(stream=sys.stderr, verbosity=2)
res = runner.run(suite)
sys.exit(0 if res.wasSuccessful() else 1)
"""


def parse_unittest_stats(raw_output: str) -> Dict[str, Any]:
    """Extracts structured test counts and normalized error signature from unittest output."""
    tests_run = 0
    failures = 0
    errors = 0
    skipped = 0

    m_run = re.search(r"Ran\s+(\d+)\s+tests?\s+in", raw_output)
    if m_run:
        tests_run = int(m_run.group(1))

    m_fail = re.search(r"failures=(\d+)", raw_output)
    if m_fail:
        failures = int(m_fail.group(1))

    m_err = re.search(r"errors=(\d+)", raw_output)
    if m_err:
        errors = int(m_err.group(1))

    m_skip = re.search(r"skipped=(\d+)", raw_output)
    if m_skip:
        skipped = int(m_skip.group(1))

    failed_total = failures + errors
    passed_count = max(0, tests_run - failed_total - skipped)

    sig_lines = []
    for line in raw_output.splitlines():
        if line.startswith(("FAIL:", "ERROR:")) or re.match(
            r"^(AssertionError|ValueError|IndexError|TypeError|KeyError|RuntimeError|ZeroDivisionError|FloatingPointError|SyntaxError|ImportError|ModuleNotFoundError|AttributeError)",
            line.strip(),
        ):
            sig_lines.append(line.strip())
    sig_source = "\n".join(sig_lines) if sig_lines else raw_output.strip()[-400:]
    error_signature = hashlib.sha256(sig_source.encode("utf-8")).hexdigest()[:16]

    return {
        "tests_run": tests_run,
        "passed_count": passed_count,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "error_signature": error_signature,
    }


class IsolatedTestRunner:
    """Executes unit tests in an ephemeral staging copy with scrubbed env & process-group watchdog."""

    def __init__(self, policy: SandboxPolicy):
        self.policy = policy.normalized()

    def _copy_workspace_to_staging(self, staging_dir: str, include_hidden: bool) -> None:
        ws = self.policy.workspace_root
        hidden_set = self.policy.hidden_files

        for root, dirs, files in os.walk(ws):
            dirs[:] = [d for d in dirs if d not in _IGNORED_COPY_NAMES]
            rel_root = os.path.relpath(root, ws).replace(os.sep, "/")
            dest_root = staging_dir if rel_root == "." else os.path.join(staging_dir, rel_root)
            os.makedirs(dest_root, exist_ok=True)

            for fname in files:
                rel_file = fname if rel_root == "." else f"{rel_root}/{fname}"
                if not include_hidden and rel_file in hidden_set:
                    continue
                src_file = os.path.join(root, fname)
                dst_file = os.path.join(dest_root, fname)
                shutil.copy2(src_file, dst_file)
                if rel_file in self.policy.protected_files or rel_file in hidden_set:
                    try:
                        os.chmod(dst_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                    except OSError:
                        pass

    def run_tests(
        self,
        test_file: str,
        timeout: int = 15,
        include_hidden: bool = False,
    ) -> Dict[str, Any]:
        guard = PathGuard(self.policy.workspace_root)
        real_test, rel_test = guard.canonicalize(test_file)

        if not os.path.exists(real_test):
            return {
                "passed": False,
                "exit_code": -1,
                "summary": "TEST_NOT_FOUND",
                "output": f"File not found: {test_file}",
                "traceback": "",
                "sandbox_mode": "ephemeral_isolated" if self.policy.use_ephemeral_copy else "direct_isolated",
                "tests_run": 0,
                "passed_count": 0,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "error_signature": "missing_test",
            }

        if rel_test in self.policy.hidden_files and not include_hidden:
            raise PathSecurityError(f"Hidden holdout test {rel_test!r} cannot run during open TDD phase.")

        tmp_obj = None
        try:
            if self.policy.use_ephemeral_copy:
                tmp_obj = tempfile.TemporaryDirectory(prefix="acs_sbx_")
                exec_ws = tmp_obj.name
                self._copy_workspace_to_staging(exec_ws, include_hidden=include_hidden)
                sandbox_mode = "ephemeral_ro_copy+python_I+rlimit"
            else:
                exec_ws = self.policy.workspace_root
                sandbox_mode = "workspace+python_I+rlimit"

            staged_test = os.path.join(exec_ws, rel_test)
            staged_test_dir = staged_test if os.path.isdir(staged_test) else os.path.dirname(staged_test)

            cmd = [
                sys.executable,
                "-I",
                "-c",
                _CHILD_RUNNER_BOOTSTRAP,
                exec_ws,
                staged_test_dir,
                staged_test,
                str(max(1, int(timeout))),
                str(self.policy.file_size_limit_mb),
                str(self.policy.memory_limit_mb),
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=staged_test_dir,
                env=_sanitized_subprocess_env(),
                start_new_session=True,
            )

            try:
                stdout_data, stderr_data = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    proc.kill()
                stdout_data, stderr_data = proc.communicate()
                return {
                    "passed": False,
                    "exit_code": 124,
                    "summary": "WATCHDOG_TIMEOUT",
                    "output": f"Execution timed out after {timeout} seconds (process group killed).",
                    "traceback": f"TimeoutExpired: Execution exceeded safety guardrail of {timeout}s.",
                    "sandbox_mode": sandbox_mode,
                    "tests_run": 0,
                    "passed_count": 0,
                    "failures": 0,
                    "errors": 1,
                    "skipped": 0,
                    "error_signature": "watchdog_timeout",
                }

            raw_lines = []
            rlimit_info = ""
            for line in ((stdout_data or "") + "\n" + (stderr_data or "")).splitlines():
                if line.startswith("[SANDBOX_RLIMITS]"):
                    rlimit_info = line.replace("[SANDBOX_RLIMITS]", "").strip()
                else:
                    raw_lines.append(line)
            raw_output = "\n".join(raw_lines).strip()
            passed = (proc.returncode == 0)

            traceback_lines = []
            capture = False
            for line in raw_output.splitlines():
                if "FAIL:" in line or "ERROR:" in line or "Traceback" in line or line.startswith("====="):
                    capture = True
                if capture:
                    traceback_lines.append(line)

            tb_snippet = "\n".join(traceback_lines) if traceback_lines else raw_output
            stats = parse_unittest_stats(raw_output)

            return {
                "passed": passed,
                "exit_code": proc.returncode,
                "summary": "ALL_TESTS_PASS" if passed else "GATE_FAILED",
                "output": raw_output,
                "traceback": tb_snippet,
                "sandbox_mode": f"{sandbox_mode}({rlimit_info})" if rlimit_info else sandbox_mode,
                **stats,
            }
        finally:
            if tmp_obj is not None:
                try:
                    for root, dirs, files in os.walk(tmp_obj.name):
                        for fn in files:
                            fp = os.path.join(root, fn)
                            try:
                                os.chmod(fp, stat.S_IRUSR | stat.S_IWUSR)
                            except OSError:
                                pass
                    tmp_obj.cleanup()
                except OSError:
                    pass
