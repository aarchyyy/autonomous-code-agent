"""
Execution Sandbox, Path Containment, AST/C++ Policy Guard, Multi-Language Gate Runners,
and Container/Process Isolation.

Provides layered isolation for autonomous code synthesis:
1. Canonical PathGuard: Prevents path traversal, symlink escapes, writes outside
   authorized target allowlists, and reads of hidden holdout tests.
2. IntegrityMonitor: SHA-256 verification of protected test files before and after steps.
3. ASTPolicyInspector & NativeSyntaxInspector: Python AST syntax + fine-grained allowlist
   static policy checks, plus pre-flight C/C++/HIP (`clang++`/`g++`/`hipcc -fsyntax-only`)
   type/syntax validation before touching workspace files.
4. IsolatedTestRunner:
   - Tier 1 (Process Sandbox): Ephemeral read-only-protected staging copy, scrubbed
     secret environment variables, isolated Python mode (`-I`), POSIX resource limits
     (`RLIMIT_CPU`, `RLIMIT_FSIZE`, and `RLIMIT_AS` on Linux), and process-group kill
     (`os.killpg`) on timeout. Supports Python `unittest`, `pytest`, and native C++17/HIP
     compile-and-test gates.
   - Tier 2 (Rootless/Unprivileged Docker Container Sandbox): Executes tests inside an
     ephemeral `--network none --read-only --cap-drop ALL --security-opt no-new-privileges`
     container (`--user 65534:65534`, `--pids-limit 64`, memory/CPU quotas) with the
     staged workspace bind-mounted `:ro` and deterministic container cleanup on timeout.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class PathSecurityError(PermissionError):
    """Raised when a file operation attempts to escape or violate sandbox path policy."""


class IntegrityViolationError(RuntimeError):
    """Raised when protected or hidden files are modified unexpectedly."""


class ASTPolicyError(ValueError):
    """Raised when code violates static AST or native compiler pre-flight policies."""


CPP_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hip")
CPP_SOURCE_EXTENSIONS = (".c", ".cc", ".cpp", ".cxx", ".hip")
CPP_HEADER_EXTENSIONS = (".h", ".hpp")

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


def detect_cpp_compiler(for_hip: bool = False) -> Optional[str]:
    """Detects an available C++/HIP compiler on PATH."""
    candidates = ("hipcc", "clang++", "g++", "c++") if for_hip else ("clang++", "g++", "c++")
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


def detect_docker_runtime(image: str = "python:3.12-alpine", timeout: int = 5) -> Dict[str, Any]:
    """Checks whether Docker CLI, daemon, and target container image are available locally."""
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return {
            "available": False,
            "daemon_ok": False,
            "image_present": False,
            "image": image,
            "reason": "docker CLI not found on PATH",
        }
    try:
        info = subprocess.run(
            [docker_bin, "info"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if info.returncode != 0:
            return {
                "available": False,
                "daemon_ok": False,
                "image_present": False,
                "image": image,
                "reason": (info.stderr or info.stdout or "docker daemon unreachable").strip()[:240],
            }
        insp = subprocess.run(
            [docker_bin, "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        image_ok = insp.returncode == 0
        return {
            "available": image_ok,
            "daemon_ok": True,
            "image_present": image_ok,
            "image": image,
            "reason": "ready" if image_ok else f"image {image!r} not present locally",
        }
    except Exception as exc:
        return {
            "available": False,
            "daemon_ok": False,
            "image_present": False,
            "image": image,
            "reason": str(exc)[:240],
        }


@dataclass
class SandboxPolicy:
    workspace_root: str
    writable_files: Optional[Set[str]] = None
    protected_files: Set[str] = field(default_factory=set)
    hidden_files: Set[str] = field(default_factory=set)
    block_dangerous_ast: bool = False
    allowed_imports: Set[str] = field(default_factory=set)
    allowed_calls: Set[str] = field(default_factory=set)
    use_ephemeral_copy: bool = True
    sandbox_backend: str = "process"  # "process" | "docker" | "auto"
    docker_image: str = "python:3.12-alpine"
    gate_runner: str = "auto"  # "auto" | "unittest" | "pytest" | "cpp"
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

        backend = (self.sandbox_backend or "process").strip().lower()
        if backend not in ("process", "docker", "auto"):
            raise ValueError(f"Unsupported sandbox_backend: {self.sandbox_backend!r}")

        runner = (self.gate_runner or "auto").strip().lower()
        if runner not in ("auto", "unittest", "pytest", "cpp"):
            raise ValueError(f"Unsupported gate_runner: {self.gate_runner!r}")

        return SandboxPolicy(
            workspace_root=real_ws,
            writable_files=_norm_set(self.writable_files),
            protected_files=_norm_set(self.protected_files) or set(),
            hidden_files=_norm_set(self.hidden_files) or set(),
            block_dangerous_ast=self.block_dangerous_ast,
            allowed_imports={m.split(".")[0].strip() for m in (self.allowed_imports or ()) if m.strip()},
            allowed_calls={c.strip() for c in (self.allowed_calls or ()) if c.strip()},
            use_ephemeral_copy=self.use_ephemeral_copy,
            sandbox_backend=backend,
            docker_image=self.docker_image or "python:3.12-alpine",
            gate_runner=runner,
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
    """Validates Python syntax and checks static import/call policies with optional allowlists."""

    @staticmethod
    def inspect(
        code_str: str,
        strict: bool = False,
        allowed_imports: Optional[Iterable[str]] = None,
        allowed_calls: Optional[Iterable[str]] = None,
    ) -> Tuple[bool, str]:
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}\n  {e.text}"

        if not strict:
            return True, "AST syntax check passed."

        import_allow = {m.split(".")[0].strip() for m in (allowed_imports or ()) if m.strip()}
        call_allow = {c.strip() for c in (allowed_calls or ()) if c.strip()}

        violations: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in _FORBIDDEN_IMPORTS_STRICT and root_mod not in import_allow:
                        violations.append(f"line {node.lineno}: forbidden import '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in _FORBIDDEN_IMPORTS_STRICT and root_mod not in import_allow:
                        violations.append(f"line {node.lineno}: forbidden from-import '{node.module}'")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    fn_id = node.func.id
                    if fn_id in _FORBIDDEN_CALLS_STRICT and fn_id not in call_allow:
                        violations.append(f"line {node.lineno}: forbidden builtin call '{fn_id}()'")

        if violations:
            return False, "AST security policy violation: " + "; ".join(violations[:5])
        return True, "AST syntax and strict static policy checks passed."


class NativeSyntaxInspector:
    """Pre-flight syntax and type validator for C/C++/HIP source & header files."""

    @staticmethod
    def inspect_cpp(
        filepath: str,
        code_str: str,
        workspace_root: Optional[str] = None,
        hidden_files: Optional[Set[str]] = None,
        timeout: int = 10,
    ) -> Tuple[bool, str]:
        ext = os.path.splitext(filepath)[1].lower()
        compiler = detect_cpp_compiler(for_hip=(ext == ".hip"))
        if not compiler:
            return True, "No C++/HIP compiler on PATH; skipped native syntax pre-flight."

        hidden = set(hidden_files or ())
        with tempfile.TemporaryDirectory(prefix="acs_cpp_ast_") as tmp_dir:
            real_tmp = os.path.realpath(tmp_dir)
            if workspace_root and os.path.isdir(workspace_root):
                real_ws = os.path.realpath(workspace_root)
                for root, dirs, files in os.walk(real_ws):
                    dirs[:] = [d for d in dirs if d not in _IGNORED_COPY_NAMES]
                    rel_root = os.path.relpath(root, real_ws).replace(os.sep, "/")
                    dst_root = real_tmp if rel_root == "." else os.path.join(real_tmp, rel_root)
                    os.makedirs(dst_root, exist_ok=True)
                    for fname in files:
                        rel_f = fname if rel_root == "." else f"{rel_root}/{fname}"
                        if rel_f in hidden:
                            continue
                        if os.path.splitext(fname)[1].lower() in CPP_EXTENSIONS:
                            shutil.copy2(os.path.join(root, fname), os.path.join(dst_root, fname))

            clean_rel = filepath.lstrip("/\\")
            candidate_path = os.path.join(real_tmp, clean_rel)
            os.makedirs(os.path.dirname(candidate_path) or real_tmp, exist_ok=True)
            with open(candidate_path, "w", encoding="utf-8") as f:
                f.write(code_str)

            cmd = [compiler, "-std=c++17", "-fsyntax-only", f"-I{real_tmp}"]
            cand_dir = os.path.dirname(candidate_path)
            if cand_dir and cand_dir != real_tmp:
                cmd.append(f"-I{cand_dir}")
            if ext in CPP_HEADER_EXTENSIONS:
                cmd.extend(["-x", "c++-header"])
            elif ext == ".hip" and os.path.basename(compiler) != "hipcc":
                cmd.extend(["-x", "c++"])
            cmd.append(candidate_path)

            try:
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=real_tmp,
                    env=_sanitized_subprocess_env(),
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return False, f"C++ syntax pre-flight timed out after {timeout}s."

            if res.returncode != 0:
                err_msg = (res.stderr or res.stdout or "C++ compilation syntax error").strip()
                err_msg = err_msg.replace(real_tmp + os.sep, "")
                return False, f"C++ Syntax/Type Error:\n{err_msg[:900]}"
            return True, f"Native C++17 syntax check passed ({os.path.basename(compiler)})."


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
    for key in ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR"):
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


_CHILD_RUNNER_BOOTSTRAP = r"""
import os, sys, resource, unittest

workdir = sys.argv[1]
test_dir = sys.argv[2]
test_file = sys.argv[3]
cpu_limit = int(sys.argv[4])
fsize_mb = int(sys.argv[5])
mem_mb = int(sys.argv[6])
runner_kind = sys.argv[7] if len(sys.argv) > 7 else "unittest"

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

if runner_kind == "pytest":
    import pytest
    rc = pytest.main(["-v", "--tb=short", test_file])
    sys.exit(int(rc))

loader = unittest.TestLoader()
if os.path.isdir(test_file):
    suite = loader.discover(start_dir=test_file)
else:
    suite = loader.discover(start_dir=test_dir, pattern=os.path.basename(test_file))

runner = unittest.TextTestRunner(stream=sys.stderr, verbosity=2)
res = runner.run(suite)
sys.exit(0 if res.wasSuccessful() else 1)
"""

_CPP_EXEC_BOOTSTRAP = r"""
import os, sys, resource

binary_path = sys.argv[1]
workdir = sys.argv[2]
cpu_limit = int(sys.argv[3])
fsize_mb = int(sys.argv[4])
mem_mb = int(sys.argv[5])

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

os.chdir(workdir)
sys.stderr.write(f"[SANDBOX_RLIMITS] {','.join(applied) or 'none'}\n")
sys.stderr.flush()
os.execv(binary_path, [binary_path])
"""


def parse_unittest_stats(raw_output: str) -> Dict[str, Any]:
    """Extracts structured test counts & error signature from unittest, pytest, or C++ gate output."""
    tests_run = 0
    failures = 0
    errors = 0
    skipped = 0

    # 1. Explicit C++ / structured summary line: [SUMMARY] tests_run=5 passed=4 failed=1
    m_cpp = re.search(
        r"\[SUMMARY\]\s*tests_run=(\d+)\s+passed=(\d+)\s+failed=(\d+)(?:\s+errors=(\d+))?",
        raw_output,
    )
    if m_cpp:
        tests_run = int(m_cpp.group(1))
        passed_count = int(m_cpp.group(2))
        failures = int(m_cpp.group(3))
        errors = int(m_cpp.group(4) or 0)
    else:
        # 2. Python unittest format: Ran X tests in Ys
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
            passed_count = max(0, tests_run - (failures + errors) - skipped)
        else:
            # 3. Pytest summary line: "== 3 passed, 1 failed in 0.04s =="
            p_pass = re.search(r"(\d+)\s+passed", raw_output)
            p_fail = re.search(r"(\d+)\s+failed", raw_output)
            p_err = re.search(r"(\d+)\s+errors?", raw_output)
            p_skip = re.search(r"(\d+)\s+skipped", raw_output)
            if p_pass or p_fail or p_err:
                passed_count = int(p_pass.group(1)) if p_pass else 0
                failures = int(p_fail.group(1)) if p_fail else 0
                errors = int(p_err.group(1)) if p_err else 0
                skipped = int(p_skip.group(1)) if p_skip else 0
                tests_run = passed_count + failures + errors + skipped
            else:
                passed_count = 0

    sig_lines = []
    for line in raw_output.splitlines():
        stripped = line.strip()
        if (
            line.startswith(("FAIL:", "ERROR:", "[FAIL]", "FAILED "))
            or ": error:" in line
            or re.match(
                r"^(AssertionError|ValueError|IndexError|TypeError|KeyError|RuntimeError|"
                r"ZeroDivisionError|FloatingPointError|SyntaxError|ImportError|"
                r"ModuleNotFoundError|AttributeError|terminate called|libc\+\+abi)",
                stripped,
            )
        ):
            sig_lines.append(stripped)
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
    """Executes unit tests in an ephemeral process or unprivileged Docker container sandbox."""

    def __init__(self, policy: SandboxPolicy):
        self.policy = policy.normalized()

    def _copy_workspace_to_staging(
        self,
        staging_dir: str,
        include_hidden: bool,
        world_readable: bool = False,
    ) -> None:
        ws = self.policy.workspace_root
        hidden_set = self.policy.hidden_files
        if world_readable:
            try:
                os.chmod(staging_dir, 0o755)
            except OSError:
                pass

        for root, dirs, files in os.walk(ws):
            dirs[:] = [d for d in dirs if d not in _IGNORED_COPY_NAMES]
            rel_root = os.path.relpath(root, ws).replace(os.sep, "/")
            dest_root = staging_dir if rel_root == "." else os.path.join(staging_dir, rel_root)
            os.makedirs(dest_root, exist_ok=True)
            if world_readable:
                try:
                    os.chmod(dest_root, 0o755)
                except OSError:
                    pass

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
                elif world_readable:
                    try:
                        os.chmod(dst_file, 0o644)
                    except OSError:
                        pass

    def _resolve_gate_runner(self, rel_test: str) -> str:
        if self.policy.gate_runner != "auto":
            return self.policy.gate_runner
        ext = os.path.splitext(rel_test)[1].lower()
        if ext in CPP_SOURCE_EXTENSIONS:
            return "cpp"
        return "unittest"

    def _select_execution_backend(self, runner_kind: str) -> str:
        # Native C++ compilation runs via host compiler + process rlimit sandbox unless image has g++
        if self.policy.sandbox_backend == "process":
            return "process"
        if self.policy.sandbox_backend == "docker":
            if runner_kind == "cpp":
                # Check if container image is explicitly requested; fall back only if auto
                pass
            return "docker"
        # "auto": prefer docker for Python unittest if daemon & image are ready locally
        if runner_kind == "unittest":
            dkr = detect_docker_runtime(self.policy.docker_image, timeout=3)
            if dkr.get("available"):
                return "docker"
        return "process"

    def _extract_traceback(self, raw_output: str) -> str:
        traceback_lines = []
        capture = False
        for line in raw_output.splitlines():
            if (
                "FAIL:" in line
                or "ERROR:" in line
                or "[FAIL]" in line
                or "Traceback" in line
                or ": error:" in line
                or line.startswith("=====")
            ):
                capture = True
            if capture:
                traceback_lines.append(line)
        return "\n".join(traceback_lines) if traceback_lines else raw_output

    def _build_docker_command(
        self,
        container_name: str,
        real_exec_ws: str,
        rel_test_dir: str,
        rel_staged_test: str,
        timeout: int,
        runner_kind: str,
    ) -> List[str]:
        container_workdir = "/workspace" if rel_test_dir in ("", ".") else f"/workspace/{rel_test_dir}"
        container_test_path = f"/workspace/{rel_staged_test}"
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,size=64m",
            "--cpus",
            "1.0",
            "--memory",
            f"{self.policy.memory_limit_mb}m",
            "--pids-limit",
            "64",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "-e",
            "HOME=/tmp",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTHONNOUSERSITE=1",
            "-v",
            f"{real_exec_ws}:/workspace:ro",
            "-w",
            container_workdir,
            self.policy.docker_image,
            "python3",
            "-I",
            "-c",
            _CHILD_RUNNER_BOOTSTRAP,
            "/workspace",
            container_workdir,
            container_test_path,
            str(max(1, int(timeout))),
            str(self.policy.file_size_limit_mb),
            str(self.policy.memory_limit_mb),
            runner_kind,
        ]
        return cmd

    def _run_in_docker(
        self,
        exec_ws: str,
        rel_test: str,
        timeout: int,
        runner_kind: str,
    ) -> Dict[str, Any]:
        dkr_status = detect_docker_runtime(self.policy.docker_image, timeout=5)
        if not dkr_status.get("available"):
            return {
                "passed": False,
                "exit_code": -2,
                "summary": "DOCKER_UNAVAILABLE",
                "output": f"Docker sandbox requested but unavailable: {dkr_status.get('reason')}",
                "traceback": f"DockerUnavailable: {dkr_status.get('reason')}",
                "sandbox_mode": f"docker_unavailable({self.policy.docker_image})",
                "tests_run": 0,
                "passed_count": 0,
                "failures": 0,
                "errors": 1,
                "skipped": 0,
                "error_signature": "docker_unavailable",
            }

        real_exec_ws = os.path.realpath(exec_ws)
        rel_test_dir = os.path.dirname(rel_test).replace(os.sep, "/")
        container_name = f"acs_sbx_{uuid.uuid4().hex[:12]}"
        cmd = self._build_docker_command(
            container_name=container_name,
            real_exec_ws=real_exec_ws,
            rel_test_dir=rel_test_dir,
            rel_staged_test=rel_test,
            timeout=timeout,
            runner_kind=runner_kind,
        )
        sandbox_mode = f"docker_isolated({self.policy.docker_image},net=none,ro_rootfs,cap_drop=ALL)"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_sanitized_subprocess_env(),
            start_new_session=True,
        )
        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout + 3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=5,
            )
            stdout_data, stderr_data = proc.communicate()
            return {
                "passed": False,
                "exit_code": 124,
                "summary": "WATCHDOG_TIMEOUT",
                "output": f"Docker container execution timed out after {timeout}s (container {container_name} killed).",
                "traceback": f"TimeoutExpired: Container execution exceeded guardrail of {timeout}s.",
                "sandbox_mode": sandbox_mode,
                "tests_run": 0,
                "passed_count": 0,
                "failures": 0,
                "errors": 1,
                "skipped": 0,
                "error_signature": "watchdog_timeout",
            }

        return self._finalize_output(stdout_data, stderr_data, proc.returncode, sandbox_mode)

    def _run_cpp_in_process(
        self,
        exec_ws: str,
        staged_test: str,
        staged_test_dir: str,
        timeout: int,
    ) -> Dict[str, Any]:
        ext = os.path.splitext(staged_test)[1].lower()
        compiler = detect_cpp_compiler(for_hip=(ext == ".hip"))
        if not compiler:
            return {
                "passed": False,
                "exit_code": -1,
                "summary": "COMPILER_NOT_FOUND",
                "output": "No C++/HIP compiler (clang++/g++/hipcc) found on host PATH.",
                "traceback": "CompilerNotFoundError: clang++/g++ required for C++ gate.",
                "sandbox_mode": "cpp_compiler_missing",
                "tests_run": 0,
                "passed_count": 0,
                "failures": 0,
                "errors": 1,
                "skipped": 0,
                "error_signature": "compiler_missing",
            }

        # Collect sibling non-test .cpp/.cc/.c implementation files if present
        extra_sources: List[str] = []
        for fname in sorted(os.listdir(staged_test_dir)):
            full = os.path.join(staged_test_dir, fname)
            if not os.path.isfile(full) or os.path.samefile(full, staged_test):
                continue
            f_ext = os.path.splitext(fname)[1].lower()
            if f_ext in CPP_SOURCE_EXTENSIONS and not fname.startswith(("test_", "holdout_")):
                extra_sources.append(full)

        with tempfile.TemporaryDirectory(prefix="acs_cpp_bin_") as bin_dir:
            bin_path = os.path.join(bin_dir, "gate_runner_bin")
            compile_cmd = [
                compiler,
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror=return-type",
                f"-I{exec_ws}",
                f"-I{staged_test_dir}",
                staged_test,
                *extra_sources,
                "-o",
                bin_path,
            ]
            try:
                comp = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    cwd=staged_test_dir,
                    env=_sanitized_subprocess_env(),
                    timeout=max(10, timeout),
                )
            except subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "exit_code": 124,
                    "summary": "COMPILE_TIMEOUT",
                    "output": f"C++ compilation timed out after {timeout}s.",
                    "traceback": "CompileTimeoutExpired",
                    "sandbox_mode": f"ephemeral_cpp17({os.path.basename(compiler)})",
                    "tests_run": 0,
                    "passed_count": 0,
                    "failures": 0,
                    "errors": 1,
                    "skipped": 0,
                    "error_signature": "compile_timeout",
                }

            if comp.returncode != 0:
                raw_err = ((comp.stdout or "") + "\n" + (comp.stderr or "")).strip()
                raw_err = raw_err.replace(exec_ws + os.sep, "")
                stats = parse_unittest_stats(raw_err)
                stats["errors"] = max(1, stats["errors"])
                return {
                    "passed": False,
                    "exit_code": comp.returncode,
                    "summary": "COMPILE_FAILED",
                    "output": raw_err,
                    "traceback": raw_err,
                    "sandbox_mode": f"ephemeral_cpp17({os.path.basename(compiler)})",
                    **stats,
                }

            run_cmd = [
                sys.executable,
                "-I",
                "-c",
                _CPP_EXEC_BOOTSTRAP,
                bin_path,
                staged_test_dir,
                str(max(1, int(timeout))),
                str(self.policy.file_size_limit_mb),
                str(self.policy.memory_limit_mb),
            ]
            sandbox_mode = f"ephemeral_ro_cpp17({os.path.basename(compiler)})+rlimit"
            proc = subprocess.Popen(
                run_cmd,
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
                    "output": f"C++ binary timed out after {timeout} seconds (process group killed).",
                    "traceback": f"TimeoutExpired: Native binary exceeded safety guardrail of {timeout}s.",
                    "sandbox_mode": sandbox_mode,
                    "tests_run": 0,
                    "passed_count": 0,
                    "failures": 0,
                    "errors": 1,
                    "skipped": 0,
                    "error_signature": "watchdog_timeout",
                }

            return self._finalize_output(stdout_data, stderr_data, proc.returncode, sandbox_mode)

    def _finalize_output(
        self,
        stdout_data: Optional[str],
        stderr_data: Optional[str],
        returncode: int,
        sandbox_mode: str,
    ) -> Dict[str, Any]:
        raw_lines = []
        rlimit_info = ""
        for line in ((stdout_data or "") + "\n" + (stderr_data or "")).splitlines():
            if line.startswith("[SANDBOX_RLIMITS]"):
                rlimit_info = line.replace("[SANDBOX_RLIMITS]", "").strip()
            else:
                raw_lines.append(line)
        raw_output = "\n".join(raw_lines).strip()
        passed = returncode == 0
        tb_snippet = self._extract_traceback(raw_output)
        stats = parse_unittest_stats(raw_output)
        mode_full = f"{sandbox_mode}({rlimit_info})" if rlimit_info else sandbox_mode

        return {
            "passed": passed,
            "exit_code": returncode,
            "summary": "ALL_TESTS_PASS" if passed else "GATE_FAILED",
            "output": raw_output,
            "traceback": tb_snippet,
            "sandbox_mode": mode_full,
            **stats,
        }

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

        runner_kind = self._resolve_gate_runner(rel_test)
        if runner_kind == "pytest" and importlib.util.find_spec("pytest") is None:
            runner_kind = "unittest"

        exec_backend = self._select_execution_backend(runner_kind)

        tmp_obj = None
        try:
            use_copy = self.policy.use_ephemeral_copy or exec_backend == "docker"
            if use_copy:
                tmp_obj = tempfile.TemporaryDirectory(prefix="acs_sbx_")
                exec_ws = tmp_obj.name
                self._copy_workspace_to_staging(
                    exec_ws,
                    include_hidden=include_hidden,
                    world_readable=(exec_backend == "docker"),
                )
                sandbox_mode = "ephemeral_ro_copy+python_I+rlimit"
            else:
                exec_ws = self.policy.workspace_root
                sandbox_mode = "workspace+python_I+rlimit"

            if exec_backend == "docker" and runner_kind in ("unittest", "pytest"):
                return self._run_in_docker(
                    exec_ws=exec_ws,
                    rel_test=rel_test,
                    timeout=timeout,
                    runner_kind=runner_kind,
                )

            staged_test = os.path.join(exec_ws, rel_test)
            staged_test_dir = staged_test if os.path.isdir(staged_test) else os.path.dirname(staged_test)

            if runner_kind == "cpp":
                return self._run_cpp_in_process(
                    exec_ws=exec_ws,
                    staged_test=staged_test,
                    staged_test_dir=staged_test_dir,
                    timeout=timeout,
                )

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
                runner_kind,
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

            return self._finalize_output(stdout_data, stderr_data, proc.returncode, sandbox_mode)
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
