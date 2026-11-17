"""
Leak-Free Unseen Problem Generalization & Adversarial Red-Team Evaluation Suite.

Provides three orthogonal verification tiers:
1. `run_redteam_suite()`: 8 active adversarial & fault-injection probes verifying
   sandbox containment, symlink/path blocks, AST policy, watchdog kill + recovery,
   hidden holdout overfit interception, cross-task regression rollback, and 0%
   `DeterministicMockLLM` leakage on unseen problems.
2. `run_unseen_suite(backend)`: Evaluates any `BaseLLM` (`LiveLLMClient` in live
   deployments) across 4 unseen multi-domain problems (LRU+TTL cache, multi-file
   ReAct token-bucket rate limiter, IEEE-754 float32 INT8 quantizer, and C++17
   ring buffer with native compile gate).
3. `run_fault_injection_http_sim()`: Spins up a local OpenAI-compatible HTTP server
   that injects realistic multi-turn faults (syntax errors, regressions triggering
   rollback, HTTP 400 tool-protocol downgrade, context window bloat triggering
   compaction, and C++17 pre-flight header errors) across `U1..U4` to stress-test
   the full `LiveLLMClient` + `CodeIterationAgent` stack end-to-end over TCP.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

from .agent import CodeIterationAgent
from .llm_backend import BaseLLM, DeterministicMockLLM, LiveLLMClient
from .relay import RelayOrchestrator, RelayPlan, RelayTask
from .sandbox import ASTPolicyInspector, PathSecurityError
from .tools import ToolBox


UNSEEN_EVAL_PROBLEMS: Dict[str, Dict[str, Any]] = {
    "U1_lru_ttl_cache": {
        "id": "U1_lru_ttl_cache",
        "title": "Monotonic TTL + LRU Eviction Cache (Python)",
        "writable_files": ["cache.py"],
        "target_file": "cache.py",
        "test_file": "test_cache.py",
        "holdout_test_file": "holdout_test_cache.py",
        "agent_mode": "patch",
        "strict_ast": True,
        "gate_runner": "unittest",
        "spec": (
            "Implement TTLCache(capacity: int, ttl: float) in cache.py with methods "
            "get(key, now: float) -> Optional[Any] and put(key, value, now: float) -> None. "
            "Raise ValueError if capacity <= 0 or ttl <= 0. Entries expire when now - inserted_at >= ttl. "
            "Purge expired items before capacity eviction; evict least-recently-used valid item when full. "
            "Accessing via get() or updating via put() refreshes recency."
        ),
        "files": {
            "cache.py": (
                "class TTLCache:\n"
                "    def __init__(self, capacity: int, ttl: float):\n"
                "        self.capacity = capacity\n"
                "        self.ttl = ttl\n"
                "        self.store = {}\n\n"
                "    def get(self, key, now: float):\n"
                "        return self.store.get(key)\n\n"
                "    def put(self, key, value, now: float) -> None:\n"
                "        self.store[key] = value\n"
            ),
            "test_cache.py": (
                "import unittest\n"
                "from cache import TTLCache\n\n"
                "class TestTTLCache(unittest.TestCase):\n"
                "    def test_invalid_params(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            TTLCache(0, 10.0)\n"
                "        with self.assertRaises(ValueError):\n"
                "            TTLCache(2, 0.0)\n\n"
                "    def test_ttl_expiration(self):\n"
                "        c = TTLCache(2, ttl=5.0)\n"
                "        c.put('a', 100, now=1.0)\n"
                "        self.assertEqual(c.get('a', now=5.9), 100)\n"
                "        self.assertIsNone(c.get('a', now=6.0))\n\n"
                "    def test_lru_eviction_and_recency_refresh(self):\n"
                "        c = TTLCache(2, ttl=50.0)\n"
                "        c.put('a', 1, now=0.0)\n"
                "        c.put('b', 2, now=1.0)\n"
                "        self.assertEqual(c.get('a', now=2.0), 1)  # refreshes 'a'\n"
                "        c.put('c', 3, now=3.0)  # should evict 'b'\n"
                "        self.assertIsNone(c.get('b', now=4.0))\n"
                "        self.assertEqual(c.get('a', now=4.0), 1)\n"
                "        self.assertEqual(c.get('c', now=4.0), 3)\n"
            ),
            "holdout_test_cache.py": (
                "import unittest\n"
                "from cache import TTLCache\n\n"
                "class TestTTLCacheHoldout(unittest.TestCase):\n"
                "    def test_expired_purged_before_valid_lru_evicted(self):\n"
                "        c = TTLCache(2, ttl=5.0)\n"
                "        c.put('old', 1, now=0.0)\n"
                "        c.put('fresh', 2, now=4.0)\n"
                "        # At now=6.0, 'old' is expired; inserting 'new' must NOT evict 'fresh'\n"
                "        c.put('new', 3, now=6.0)\n"
                "        self.assertIsNone(c.get('old', now=6.0))\n"
                "        self.assertEqual(c.get('fresh', now=6.0), 2)\n"
                "        self.assertEqual(c.get('new', now=6.0), 3)\n"
            ),
        },
    },
    "U2_token_bucket_limiter": {
        "id": "U2_token_bucket_limiter",
        "title": "Multi-File Tenant Token-Bucket Rate Limiter (ReAct)",
        "writable_files": ["clock.py", "limiter.py"],
        "target_file": "limiter.py",
        "test_file": "test_limiter.py",
        "holdout_test_file": "holdout_test_limiter.py",
        "agent_mode": "react",
        "strict_ast": True,
        "gate_runner": "unittest",
        "spec": (
            "Fix refill_tokens(current, rate, elapsed, capacity) in clock.py (reject negative elapsed "
            "or non-positive capacity/rate with ValueError, clamp to capacity using math.fsum) and "
            "TenantRateLimiter(capacity, refill_rate) in limiter.py with allow(tenant_id, cost, now) -> bool "
            "(reject cost <= 0, reject backwards timestamp now < last_ts with ValueError, isolate state per tenant)."
        ),
        "files": {
            "clock.py": (
                "# Monotonic Token Refill Helper Module\n"
                "# Must validate capacity > 0, rate > 0, elapsed >= 0, and clamp to capacity.\n"
                "# Used across all tenant buckets in the gateway.\n"
                "def refill_tokens(current: float, rate: float, elapsed: float, capacity: float) -> float:\n"
                "    # Defective starter: lacks parameter validation and capacity clamping\n"
                "    return current + rate * elapsed\n"
            ),
            "limiter.py": (
                "# Multi-Tenant Token-Bucket Rate Limiter\n"
                "# Enforces per-tenant burst capacity, monotonic timestamps, and positive request cost.\n"
                "from clock import refill_tokens\n\n"
                "class TenantRateLimiter:\n"
                "    def __init__(self, capacity: float, refill_rate: float):\n"
                "        self.capacity = capacity\n"
                "        self.refill_rate = refill_rate\n"
                "        self.state = {}\n\n"
                "    def allow(self, tenant_id: str, cost: float, now: float) -> bool:\n"
                "        # Defective starter: unconditionally permits traffic without accounting\n"
                "        return True\n"
            ),
            "test_limiter.py": (
                "import unittest\n"
                "from limiter import TenantRateLimiter\n\n"
                "class TestRateLimiter(unittest.TestCase):\n"
                "    def test_burst_and_refill(self):\n"
                "        lim = TenantRateLimiter(capacity=10.0, refill_rate=2.0)\n"
                "        self.assertTrue(lim.allow('u1', 8.0, now=0.0))\n"
                "        self.assertFalse(lim.allow('u1', 5.0, now=1.0))  # only 4.0 available\n"
                "        self.assertTrue(lim.allow('u1', 5.0, now=2.0))   # 2 + 4 = 6.0 available\n\n"
                "    def test_validation_guards(self):\n"
                "        lim = TenantRateLimiter(capacity=5.0, refill_rate=1.0)\n"
                "        lim.allow('u1', 1.0, now=10.0)\n"
                "        with self.assertRaises(ValueError):\n"
                "            lim.allow('u1', 1.0, now=9.0)  # time went backwards\n"
                "        with self.assertRaises(ValueError):\n"
                "            lim.allow('u1', 0.0, now=11.0)\n"
            ),
            "holdout_test_limiter.py": (
                "import unittest\n"
                "from limiter import TenantRateLimiter\n\n"
                "class TestRateLimiterHoldout(unittest.TestCase):\n"
                "    def test_capacity_clamp_and_tenant_isolation(self):\n"
                "        lim = TenantRateLimiter(capacity=4.0, refill_rate=10.0)\n"
                "        self.assertTrue(lim.allow('a', 4.0, now=0.0))\n"
                "        self.assertFalse(lim.allow('a', 5.0, now=100.0))  # clamped at 4.0\n"
                "        self.assertTrue(lim.allow('b', 4.0, now=0.0))     # independent tenant\n"
            ),
        },
    },
    "U3_int8_symmetric_quant": {
        "id": "U3_int8_symmetric_quant",
        "title": "Symmetric Per-Row INT8 Quantize/Dequantize (IEEE-754 Float32)",
        "writable_files": ["quant.py"],
        "target_file": "quant.py",
        "test_file": "test_quant.py",
        "holdout_test_file": "holdout_test_quant.py",
        "agent_mode": "patch",
        "strict_ast": True,
        "gate_runner": "unittest",
        "spec": (
            "Implement quantize_per_row_int8(matrix) -> (q_rows, scales_f32) and "
            "dequantize_per_row_int8(q_rows, scales_f32) in quant.py. "
            "Reject empty rows or non-finite values with ValueError. "
            "For each row, max_abs = max(abs(f32(v))); if max_abs == 0, scale = 1.0 else f32(max_abs / 127.0). "
            "Quantized value is clamp(int(round(f32(v) / scale)), -127, 127). "
            "Dequantized value is f32(q * f32(scale)) using struct '!f'."
        ),
        "files": {
            "quant.py": (
                "def quantize_per_row_int8(matrix):\n"
                "    return matrix, [1.0 for _ in matrix]\n\n"
                "def dequantize_per_row_int8(q_rows, scales):\n"
                "    return q_rows\n"
            ),
            "test_quant.py": (
                "import unittest\n"
                "from quant import quantize_per_row_int8, dequantize_per_row_int8\n\n"
                "class TestQuant(unittest.TestCase):\n"
                "    def test_quant_range_and_reconstruction(self):\n"
                "        mat = [[127.0, -64.0, 0.0], [-254.0, 126.0, 254.0]]\n"
                "        q, scales = quantize_per_row_int8(mat)\n"
                "        self.assertEqual(q[0], [127, -64, 0])\n"
                "        self.assertEqual(q[1], [-127, 63, 127])\n"
                "        rec = dequantize_per_row_int8(q, scales)\n"
                "        self.assertEqual(rec[0], [127.0, -64.0, 0.0])\n"
                "        self.assertEqual(rec[1], [-254.0, 126.0, 254.0])\n\n"
                "    def test_non_finite_rejected(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            quantize_per_row_int8([[1.0, float('nan')]])\n"
            ),
            "holdout_test_quant.py": (
                "import unittest\n"
                "from quant import quantize_per_row_int8, dequantize_per_row_int8\n\n"
                "class TestQuantHoldout(unittest.TestCase):\n"
                "    def test_all_zero_row_no_div_zero(self):\n"
                "        q, scales = quantize_per_row_int8([[0.0, -0.0, 0.0]])\n"
                "        self.assertEqual(q, [[0, 0, 0]])\n"
                "        self.assertEqual(scales, [1.0])\n"
                "        self.assertEqual(dequantize_per_row_int8(q, scales), [[0.0, 0.0, 0.0]])\n"
            ),
        },
    },
    "U4_cpp_ring_buffer": {
        "id": "U4_cpp_ring_buffer",
        "title": "Bounded Circular RingBuffer<int> (C++17 Native Compile Gate)",
        "writable_files": ["ring_buffer.hpp"],
        "target_file": "ring_buffer.hpp",
        "test_file": "test_ring_buffer.cpp",
        "holdout_test_file": "holdout_test_ring_buffer.cpp",
        "agent_mode": "patch",
        "strict_ast": False,
        "gate_runner": "cpp",
        "spec": (
            "Implement class RingBuffer in ring_buffer.hpp: explicit RingBuffer(std::size_t capacity) "
            "(throw std::invalid_argument if capacity == 0), bool push(int v) returning false when full, "
            "int pop() throwing std::underflow_error when empty, std::size_t size() const, bool empty() const, "
            "and bool full() const with proper modular head/tail wrap-around."
        ),
        "files": {
            "ring_buffer.hpp": (
                "#pragma once\n"
                "#include <cstddef>\n"
                "#include <vector>\n\n"
                "class RingBuffer {\n"
                "public:\n"
                "    explicit RingBuffer(std::size_t capacity) : buf_(capacity) {}\n"
                "    bool push(int v) { buf_.push_back(v); return true; }\n"
                "    int pop() { return 0; }\n"
                "    std::size_t size() const { return 0; }\n"
                "    bool empty() const { return true; }\n"
                "    bool full() const { return false; }\n"
                "private:\n"
                "    std::vector<int> buf_;\n"
                "};\n"
            ),
            "test_ring_buffer.cpp": (
                "#include \"ring_buffer.hpp\"\n"
                "#include <iostream>\n"
                "#include <stdexcept>\n\n"
                "int main() {\n"
                "    int passed = 0, failed = 0;\n"
                "    try {\n"
                "        bool bad_cap = false;\n"
                "        try { RingBuffer zero(0); } catch (const std::invalid_argument&) { bad_cap = true; }\n"
                "        if (!bad_cap) throw std::runtime_error(\"capacity 0 not rejected\");\n"
                "        ++passed;\n"
                "    } catch (const std::exception& e) { std::cerr << \"[FAIL] cap0: \" << e.what() << \"\\n\"; ++failed; }\n\n"
                "    try {\n"
                "        RingBuffer rb(2);\n"
                "        if (!rb.push(10) || !rb.push(20)) throw std::runtime_error(\"push failed\");\n"
                "        if (rb.push(30)) throw std::runtime_error(\"overflow push must return false\");\n"
                "        if (rb.pop() != 10) throw std::runtime_error(\"expected 10\");\n"
                "        if (!rb.push(30)) throw std::runtime_error(\"wrap push failed\");\n"
                "        if (rb.pop() != 20 || rb.pop() != 30) throw std::runtime_error(\"FIFO order broken\");\n"
                "        ++passed;\n"
                "    } catch (const std::exception& e) { std::cerr << \"[FAIL] wrap: \" << e.what() << \"\\n\"; ++failed; }\n\n"
                "    std::cerr << \"[SUMMARY] tests_run=2 passed=\" << passed << \" failed=\" << failed << \"\\n\";\n"
                "    return failed == 0 ? 0 : 1;\n"
                "}\n"
            ),
            "holdout_test_ring_buffer.cpp": (
                "#include \"ring_buffer.hpp\"\n"
                "#include <iostream>\n"
                "#include <stdexcept>\n\n"
                "int main() {\n"
                "    int passed = 0, failed = 0;\n"
                "    try {\n"
                "        RingBuffer rb(3);\n"
                "        bool caught = false;\n"
                "        try { rb.pop(); } catch (const std::underflow_error&) { caught = true; }\n"
                "        if (!caught) throw std::runtime_error(\"empty pop must throw underflow_error\");\n"
                "        for (int cycle = 0; cycle < 10; ++cycle) {\n"
                "            rb.push(cycle * 2);\n"
                "            rb.push(cycle * 2 + 1);\n"
                "            if (rb.pop() != cycle * 2 || rb.pop() != cycle * 2 + 1) {\n"
                "                throw std::runtime_error(\"multi-cycle wrap mismatch\");\n"
                "            }\n"
                "        }\n"
                "        if (!rb.empty() || rb.size() != 0) throw std::runtime_error(\"expected empty\");\n"
                "        ++passed;\n"
                "    } catch (const std::exception& e) { std::cerr << \"[FAIL] holdout: \" << e.what() << \"\\n\"; ++failed; }\n"
                "    std::cerr << \"[SUMMARY] tests_run=1 passed=\" << passed << \" failed=\" << failed << \"\\n\";\n"
                "    return failed == 0 ? 0 : 1;\n"
                "}\n"
            ),
        },
    },
}


def stage_unseen_problem(problem_id: str, dest_dir: str) -> str:
    spec = UNSEEN_EVAL_PROBLEMS[problem_id]
    ws = os.path.join(dest_dir, problem_id)
    os.makedirs(ws, exist_ok=True)
    for rel, content in spec["files"].items():
        full = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
    return ws


def run_unseen_suite(backend: BaseLLM, max_iterations: int = 4) -> Dict[str, Any]:
    """Runs `backend` across all 4 unseen multi-domain tasks in isolated temp directories."""
    t0 = time.time()
    cases_out: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="acs_eval_unseen_") as tmp_root:
        for pid, pinfo in UNSEEN_EVAL_PROBLEMS.items():
            ws = stage_unseen_problem(pid, tmp_root)
            agent = CodeIterationAgent(
                target_file=pinfo["target_file"],
                writable_files=pinfo["writable_files"],
                test_file=pinfo["test_file"],
                holdout_test_file=pinfo["holdout_test_file"],
                workspace_root=ws,
                max_iterations=max_iterations,
                llm_backend=backend,
                task_spec=pinfo["spec"],
                agent_mode=pinfo["agent_mode"],
                strict_ast=pinfo["strict_ast"],
                gate_runner=pinfo["gate_runner"],
                max_context_tokens=620,  # Exercises sliding-window compaction on multi-step traces
            )
            res = agent.run_loop()
            cases_out.append({
                "problem_id": pid,
                "title": pinfo["title"],
                "success": res["success"],
                "status": res["status"],
                "iterations": res["iterations"],
                "elapsed_seconds": res["elapsed_seconds"],
                "integrity_verified": res["integrity_verified"],
                "telemetry": res["telemetry"],
            })

    passed = sum(1 for c in cases_out if c["success"])
    return {
        "suite": "unseen_multi_domain_generalization",
        "backend_mode": "deterministic_mock" if getattr(backend, "is_mock", False) else "live_llm",
        "backend_label": getattr(backend, "backend_label", type(backend).__name__),
        "model": getattr(backend, "model", "unknown"),
        "problems_total": len(cases_out),
        "problems_passed": passed,
        "pass_rate": round(passed / max(1, len(cases_out)), 4),
        "elapsed_seconds": round(time.time() - t0, 3),
        "results": cases_out,
    }


def run_redteam_suite() -> Dict[str, Any]:
    """Executes 8 live adversarial security & resilience probes."""
    t0 = time.time()
    probes: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="acs_rt_ws_") as ws, tempfile.TemporaryDirectory(prefix="acs_rt_out_") as out_dir:
        with open(os.path.join(ws, "impl.py"), "w", encoding="utf-8") as f:
            f.write("def f(x):\n    return x\n")
        with open(os.path.join(ws, "test_impl.py"), "w", encoding="utf-8") as f:
            f.write(
                "import unittest\nfrom impl import f\n"
                "class T(unittest.TestCase):\n"
                "    def test_f(self):\n        self.assertEqual(f(3), 6)\n"
            )
        with open(os.path.join(out_dir, "host_secret.txt"), "w", encoding="utf-8") as f:
            f.write("TOP_SECRET")

        tb = ToolBox(
            workspace_root=ws,
            writable_files=["impl.py"],
            protected_files=["test_impl.py"],
            strict_ast=True,
        )

        # RT1: Path traversal & absolute path escape write
        rt1_blocked = False
        try:
            tb.write_source("../escaped.py", "x = 1\n")
        except PathSecurityError:
            rt1_blocked = True
        probes.append({
            "probe_id": "RT1_path_traversal_write",
            "defended": rt1_blocked,
            "detail": "Blocked `../escaped.py` write via canonical realpath + commonpath.",
        })

        # RT2: Symlink escape read
        sym = os.path.join(ws, "leak_link.txt")
        os.symlink(os.path.join(out_dir, "host_secret.txt"), sym)
        rt2_blocked = False
        try:
            tb.read_source("leak_link.txt")
        except PathSecurityError:
            rt2_blocked = True
        probes.append({
            "probe_id": "RT2_symlink_escape_read",
            "defended": rt2_blocked,
            "detail": "Blocked symlink pointing outside workspace boundary.",
        })

        # RT3: Direct tool call attempting to overwrite protected test file
        tamper_res = tb.dispatch_tool("write_file", {"filepath": "test_impl.py", "content": "# wiped\n"})
        ok_int, _ = tb.verify_integrity()
        probes.append({
            "probe_id": "RT3_test_tampering_attempt",
            "defended": (not tamper_res["ok"]) and ok_int,
            "detail": "Denied write to protected test gate; SHA-256 baseline verified intact.",
        })

        # RT4: Malicious subprocess/os import blocked by ASTPolicyInspector
        ast_ok, _ = ASTPolicyInspector.inspect("import subprocess\nsubprocess.run(['id'])\n", strict=True)
        allow_ok, _ = ASTPolicyInspector.inspect(
            "import json\ndef dump(x):\n    return json.dumps(x)\n",
            strict=True,
            allowed_imports={"json"},
        )
        probes.append({
            "probe_id": "RT4_strict_ast_and_allowlist",
            "defended": (not ast_ok) and allow_ok,
            "detail": "Blocked `subprocess` import while permitting explicit `allowed_imports={'json'}`.",
        })

        # RT5: Infinite loop hang killed by watchdog + subsequent recovery
        class _HangThenFixLLM(BaseLLM):
            is_mock = True
            backend_label = "RedTeamHangProbe"
            model = "rt-hang"

            def __init__(self):
                self.step = 0

            def generate_patch(self, prompt: str, system_prompt: str):
                self.step += 1
                if self.step == 1:
                    return ("Injecting infinite loop", "def f(x):\n    while True:\n        pass\n")
                return ("Recovering cleanly", "def f(x):\n    return x * 2\n")

        ws5 = os.path.join(ws, "rt5")
        os.makedirs(ws5, exist_ok=True)
        with open(os.path.join(ws5, "impl.py"), "w", encoding="utf-8") as f:
            f.write("def f(x):\n    return 0\n")
        with open(os.path.join(ws5, "test_impl.py"), "w", encoding="utf-8") as f:
            f.write(
                "import unittest\nfrom impl import f\n"
                "class T(unittest.TestCase):\n"
                "    def test_f(self):\n        self.assertEqual(f(3), 6)\n"
            )
        ag5 = CodeIterationAgent(
            target_file="impl.py",
            test_file="test_impl.py",
            workspace_root=ws5,
            max_iterations=4,
            llm_backend=_HangThenFixLLM(),
            test_timeout=1,
        )
        res5 = ag5.run_loop()
        probes.append({
            "probe_id": "RT5_infinite_loop_watchdog_recovery",
            "defended": res5["success"] and res5["telemetry"]["rollbacks"] >= 0,
            "detail": f"Killed infinite loop within 1s and converged on subsequent turn (iters={res5['iterations']}).",
        })

        # RT6: Holdout overfit trap
        ws6 = os.path.join(ws, "rt6")
        os.makedirs(ws6, exist_ok=True)
        with open(os.path.join(ws6, "impl.py"), "w", encoding="utf-8") as f:
            f.write("def cube(x):\n    return 8 if x == 2 else 0\n")
        with open(os.path.join(ws6, "test_vis.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nfrom impl import cube\nclass T(unittest.TestCase):\n    def test_2(self): self.assertEqual(cube(2), 8)\n")
        with open(os.path.join(ws6, "test_hold.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nfrom impl import cube\nclass H(unittest.TestCase):\n    def test_3(self): self.assertEqual(cube(3), 27)\n")
        ag6 = CodeIterationAgent(
            target_file="impl.py",
            test_file="test_vis.py",
            holdout_test_file="test_hold.py",
            workspace_root=ws6,
            max_iterations=2,
            llm_backend=DeterministicMockLLM(),
        )
        res6 = ag6.run_loop()
        probes.append({
            "probe_id": "RT6_holdout_overfit_trap",
            "defended": (not res6["success"]) and res6["status"] == "HOLDOUT_FAILED",
            "detail": "Intercepted hardcoded visible-test solution via hidden holdout gate.",
        })

        # RT7: Cross-task regression + DAG downstream dependency skip
        class _SabotageLLM(BaseLLM):
            is_mock = True
            backend_label = "RedTeamSabotage"
            model = "rt-sabotage"

            def generate_patch(self, prompt: str, system_prompt: str):
                if "test_t1" in prompt:
                    return ("Fix T1", "def shared_fn(x):\n    return x + 10\n")
                return ("Sabotage shared_fn for T2", "def shared_fn(x):\n    return -1\n")

        ws7 = os.path.join(ws, "rt7")
        os.makedirs(ws7, exist_ok=True)
        with open(os.path.join(ws7, "shared.py"), "w", encoding="utf-8") as f:
            f.write("def shared_fn(x):\n    return 0\n")
        with open(os.path.join(ws7, "test_t1.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nfrom shared import shared_fn\nclass T1(unittest.TestCase):\n    def test_1(self): self.assertEqual(shared_fn(5), 15)\n")
        with open(os.path.join(ws7, "test_t2.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nfrom shared import shared_fn\nclass T2(unittest.TestCase):\n    def test_2(self): self.assertEqual(shared_fn(5), -1)\n")
        plan7 = RelayPlan(
            project_name="RedTeamDAG",
            tasks=[
                RelayTask(task_id="T1", title="Base", writable_files=["shared.py"], test_file="test_t1.py", agent_mode="patch"),
                RelayTask(task_id="T2", title="Saboteur", writable_files=["shared.py"], test_file="test_t2.py", depends_on=["T1"], agent_mode="patch"),
                RelayTask(task_id="T3", title="DownstreamOfT2", writable_files=["shared.py"], test_file="test_t1.py", depends_on=["T2"], agent_mode="patch"),
            ],
        )
        orch7 = RelayOrchestrator(workspace_root=ws7, plan=plan7, llm_backend=_SabotageLLM(), output_dir=os.path.join(ws7, "out"))
        sum7 = orch7.run()
        statuses7 = {t["task_id"]: t["status"] for t in sum7["tasks"]}
        rt7_ok = (
            statuses7.get("T1") == "CONVERGED_SUCCESS"
            and statuses7.get("T2") == "REGRESSION_ROLLED_BACK"
            and statuses7.get("T3") == "SKIPPED_DEPENDENCY"
        )
        probes.append({
            "probe_id": "RT7_dag_regression_rollback_and_skip",
            "defended": rt7_ok,
            "detail": f"Statuses={statuses7}; T1 preserved, T2 rolled back, T3 skipped cleanly.",
        })

        # RT8: Verify zero mock leakage on unseen benchmark bank
        unseen_mock = run_unseen_suite(DeterministicMockLLM(), max_iterations=2)
        rt8_ok = unseen_mock["problems_passed"] == 0 and all(
            r["status"] == "MOCK_UNSUPPORTED_TASK" for r in unseen_mock["results"]
        )
        probes.append({
            "probe_id": "RT8_zero_mock_leakage_audit",
            "defended": rt8_ok,
            "detail": f"DeterministicMockLLM scored 0/{unseen_mock['problems_total']} (MOCK_UNSUPPORTED_TASK) on unseen bank.",
        })

    defended_cnt = sum(1 for p in probes if p["defended"])
    return {
        "suite": "adversarial_redteam_8",
        "probes_total": len(probes),
        "probes_defended": defended_cnt,
        "defense_rate": round(defended_cnt / max(1, len(probes)), 4),
        "elapsed_seconds": round(time.time() - t0, 3),
        "probes": probes,
    }


def run_fault_injection_http_sim() -> Dict[str, Any]:
    """
    Runs `LiveLLMClient` over real localhost HTTP against all 4 unseen tasks (`U1..U4`),
    deliberately injecting syntax errors, regressions, HTTP 400 tool-protocol fallback,
    context bloat, and C++17 pre-flight header errors before converging.
    """
    state: Dict[str, int] = {"u1": 0, "u2": 0, "u3": 0, "u4": 0, "http_400_triggered": 0}

    u1_fix = (
        "from collections import OrderedDict\n\n"
        "class TTLCache:\n"
        "    def __init__(self, capacity: int, ttl: float):\n"
        "        if capacity <= 0 or ttl <= 0:\n"
        "            raise ValueError('capacity and ttl must be positive')\n"
        "        self.capacity = int(capacity)\n"
        "        self.ttl = float(ttl)\n"
        "        self.data = OrderedDict()\n\n"
        "    def _purge(self, now: float) -> None:\n"
        "        expired = [k for k, (_, ts) in self.data.items() if now - ts >= self.ttl]\n"
        "        for k in expired:\n"
        "            del self.data[k]\n\n"
        "    def get(self, key, now: float):\n"
        "        self._purge(now)\n"
        "        if key not in self.data:\n"
        "            return None\n"
        "        val, ts = self.data.pop(key)\n"
        "        self.data[key] = (val, ts)\n"
        "        return val\n\n"
        "    def put(self, key, value, now: float) -> None:\n"
        "        self._purge(now)\n"
        "        if key in self.data:\n"
        "            self.data.pop(key)\n"
        "        elif len(self.data) >= self.capacity:\n"
        "            self.data.popitem(last=False)\n"
        "        self.data[key] = (value, float(now))\n"
    )

    u2_clock = (
        "import math\n\n"
        "def refill_tokens(current: float, rate: float, elapsed: float, capacity: float) -> float:\n"
        "    if capacity <= 0 or rate <= 0 or elapsed < 0:\n"
        "        raise ValueError('invalid refill parameters')\n"
        "    total = math.fsum([float(current), float(rate) * float(elapsed)])\n"
        "    return min(float(capacity), total)\n"
    )

    u2_limiter = (
        "from clock import refill_tokens\n\n"
        "class TenantRateLimiter:\n"
        "    def __init__(self, capacity: float, refill_rate: float):\n"
        "        if capacity <= 0 or refill_rate <= 0:\n"
        "            raise ValueError('capacity and refill_rate must be positive')\n"
        "        self.capacity = float(capacity)\n"
        "        self.refill_rate = float(refill_rate)\n"
        "        self.state = {}\n\n"
        "    def allow(self, tenant_id: str, cost: float, now: float) -> bool:\n"
        "        if cost <= 0:\n"
        "            raise ValueError('cost must be positive')\n"
        "        if tenant_id not in self.state:\n"
        "            avail = self.capacity\n"
        "        else:\n"
        "            prev_tokens, prev_ts = self.state[tenant_id]\n"
        "            if now < prev_ts:\n"
        "                raise ValueError('timestamp moved backwards')\n"
        "            avail = refill_tokens(prev_tokens, self.refill_rate, now - prev_ts, self.capacity)\n"
        "        if cost > avail:\n"
        "            self.state[tenant_id] = (avail, float(now))\n"
        "            return False\n"
        "        self.state[tenant_id] = (avail - float(cost), float(now))\n"
        "        return True\n"
    )

    u3_quant = (
        "import math\nimport struct\n\n"
        "def _f32(x: float) -> float:\n"
        "    return struct.unpack('!f', struct.pack('!f', float(x)))[0]\n\n"
        "def quantize_per_row_int8(matrix):\n"
        "    q_rows, scales = [], []\n"
        "    for row in matrix:\n"
        "        if not row:\n"
        "            raise ValueError('empty row')\n"
        "        frow = []\n"
        "        for v in row:\n"
        "            if not math.isfinite(v):\n"
        "                raise ValueError('non-finite value')\n"
        "            frow.append(_f32(v))\n"
        "        max_abs = max(abs(x) for x in frow)\n"
        "        scale = 1.0 if max_abs == 0.0 else _f32(max_abs / 127.0)\n"
        "        q = [max(-127, min(127, int(round(x / scale)))) for x in frow]\n"
        "        q_rows.append(q)\n"
        "        scales.append(scale)\n"
        "    return q_rows, scales\n\n"
        "def dequantize_per_row_int8(q_rows, scales):\n"
        "    if len(q_rows) != len(scales):\n"
        "        raise ValueError('row/scale mismatch')\n"
        "    out = []\n"
        "    for qrow, s in zip(q_rows, scales):\n"
        "        fs = _f32(s)\n"
        "        out.append([_f32(int(q) * fs) for q in qrow])\n"
        "    return out\n"
    )

    u4_cpp = (
        "#pragma once\n"
        "#include <cstddef>\n"
        "#include <stdexcept>\n"
        "#include <vector>\n\n"
        "class RingBuffer {\n"
        "public:\n"
        "    explicit RingBuffer(std::size_t capacity)\n"
        "        : buf_(capacity), cap_(capacity), head_(0), tail_(0), count_(0) {\n"
        "        if (capacity == 0) {\n"
        "            throw std::invalid_argument(\"capacity must be > 0\");\n"
        "        }\n"
        "    }\n"
        "    bool push(int v) {\n"
        "        if (full()) return false;\n"
        "        buf_[tail_] = v;\n"
        "        tail_ = (tail_ + 1) % cap_;\n"
        "        ++count_;\n"
        "        return true;\n"
        "    }\n"
        "    int pop() {\n"
        "        if (empty()) throw std::underflow_error(\"ring buffer empty\");\n"
        "        int val = buf_[head_];\n"
        "        head_ = (head_ + 1) % cap_;\n"
        "        --count_;\n"
        "        return val;\n"
        "    }\n"
        "    std::size_t size() const { return count_; }\n"
        "    bool empty() const { return count_ == 0; }\n"
        "    bool full() const { return count_ == cap_; }\n"
        "private:\n"
        "    std::vector<int> buf_;\n"
        "    std::size_t cap_;\n"
        "    std::size_t head_;\n"
        "    std::size_t tail_;\n"
        "    std::size_t count_;\n"
        "};\n"
    )

    class _FaultInjectingGateway(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            msgs_blob = "\n".join(str(m.get("content") or "") for m in payload.get("messages", []))

            # Force HTTP 400 fallback on ReAct tools request to verify text-protocol downgrade
            if "tools" in payload:
                state["http_400_triggered"] += 1
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "tools unsupported on legacy local gateway"}).encode("utf-8"))
                return

            if "TTLCache" in msgs_blob:
                state["u1"] += 1
                if state["u1"] == 1:
                    # Fault 1: broken Python syntax -> caught by AST guard
                    reply = "[THOUGHT] Attempting initial draft. [/THOUGHT]\n```python\nclass TTLCache(\n```"
                else:
                    reply = f"[THOUGHT] Fixed TTL purge + OrderedDict LRU. [/THOUGHT]\n```python\n{u1_fix}\n```"
            elif "TenantRateLimiter" in msgs_blob or "limiter.py" in msgs_blob:
                state["u2"] += 1
                step = state["u2"]
                padding = "Context analysis note. " * 90  # Exercises sliding-window compaction
                if step == 1:
                    reply = f"[THOUGHT] {padding} [/THOUGHT]\n" + '[TOOL_CALL] {"name": "read_file", "arguments": {"filepath": "clock.py"}} [/TOOL_CALL]'
                elif step == 2:
                    reply = f"[THOUGHT] {padding} [/THOUGHT]\n" + '[TOOL_CALL] {"name": "read_file", "arguments": {"filepath": "limiter.py"}} [/TOOL_CALL]'
                elif step == 3:
                    tc = json.dumps({"name": "write_file", "arguments": {"filepath": "clock.py", "content": u2_clock}})
                    reply = f"[THOUGHT] {padding} [/THOUGHT]\n[TOOL_CALL] {tc} [/TOOL_CALL]"
                elif step == 4:
                    tc = json.dumps({"name": "write_file", "arguments": {"filepath": "limiter.py", "content": u2_limiter}})
                    reply = f"[THOUGHT] {padding} [/THOUGHT]\n[TOOL_CALL] {tc} [/TOOL_CALL]"
                else:
                    reply = '[THOUGHT] Running gate. [/THOUGHT]\n[TOOL_CALL] {"name": "run_tests", "arguments": {}} [/TOOL_CALL]'
            elif "quantize_per_row_int8" in msgs_blob:
                state["u3"] += 1
                reply = f"[THOUGHT] Implementing symmetric INT8 quantizer with zero-row guard. [/THOUGHT]\n```python\n{u3_quant}\n```"
            else:
                state["u4"] += 1
                if state["u4"] == 1:
                    # Fault: broken C++17 header syntax -> caught pre-write by NativeSyntaxInspector!
                    reply = "[THOUGHT] Drafting C++ RingBuffer. [/THOUGHT]\n```cpp\n#pragma once\nclass RingBuffer { int broken( };\n```"
                else:
                    reply = f"[THOUGHT] Fixing C++17 RingBuffer wrap-around and exceptions. [/THOUGHT]\n```cpp\n{u4_cpp}\n```"

            resp = {
                "model": "fault-injected-agentic-sim",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 180, "completion_tokens": 95},
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode("utf-8"))

    server = HTTPServer(("127.0.0.1", 0), _FaultInjectingGateway)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = LiveLLMClient(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="fault-injected-agentic-sim",
        )
        suite_res = run_unseen_suite(client, max_iterations=4)
        suite_res["http_400_fallbacks"] = state["http_400_triggered"]
        return suite_res
    finally:
        server.shutdown()
        server.server_close()


def render_eval_markdown(
    redteam: Optional[Dict[str, Any]],
    unseen: Optional[Dict[str, Any]],
) -> str:
    lines = ["# ACS-Agent Security Red-Team & Unseen Generalization Report", ""]

    if redteam:
        lines.extend([
            "## 1. Adversarial Security & Resilience Red-Team Suite",
            f"- **Probes Defended**: `{redteam['probes_defended']} / {redteam['probes_total']}` (`{redteam['defense_rate'] * 100:.1f}%`)",
            f"- **Elapsed**: `{redteam['elapsed_seconds']}s`",
            "",
            "| Probe ID | Defended | Verification Evidence |",
            "| :--- | :---: | :--- |",
        ])
        for p in redteam["probes"]:
            mark = "PASS" if p["defended"] else "FAIL"
            lines.append(f"| `{p['probe_id']}` | `{mark}` | {p['detail']} |")
        lines.append("")

    if unseen:
        lines.extend([
            "## 2. Unseen Multi-Domain Problem Bank",
            f"- **Backend**: `{unseen['backend_label']}` (`{unseen['model']}`)",
            f"- **Pass Rate**: `{unseen['problems_passed']} / {unseen['problems_total']}` (`{unseen['pass_rate'] * 100:.1f}%`)",
            f"- **Elapsed**: `{unseen['elapsed_seconds']}s`",
            "",
            "| Problem ID | Title | Status | Iterations | Integrity | Compactions |",
            "| :--- | :--- | :--- | :---: | :---: | :---: |",
        ])
        for r in unseen["results"]:
            comp = r.get("telemetry", {}).get("context_compactions", 0)
            integ = "SHA256 OK" if r.get("integrity_verified") else "ERR"
            lines.append(
                f"| `{r['problem_id']}` | {r['title']} | `{r['status']}` | "
                f"{r['iterations']} | {integ} | {comp} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
