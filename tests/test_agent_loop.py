import json
import os
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.agent import CodeIterationAgent
from src.benchmarks_catalog import BENCHMARK_CASES, inject_defective_starter
from src.context import ContextWindowManager
from src.llm_backend import AgentTurn, BaseLLM, DeterministicMockLLM, LiveLLMClient, probe_llm_endpoint


class _RegressingLLM(BaseLLM):
    is_mock = True
    backend_label = "RegressingTestDouble"
    model = "test-double"

    def __init__(self):
        self.calls = 0

    def generate_patch(self, prompt: str, system_prompt: str):
        self.calls += 1
        if self.calls == 1:
            return ("Attempt 1: broke passing test.", "def op(x, y):\n    return 0\n")
        return ("Attempt 2: fixed cleanly.", "def op(x, y):\n    return x * y + 1\n")


class _StallingLLM(BaseLLM):
    is_mock = False
    backend_label = "StallingTestDouble"
    model = "stall-double"

    def generate_patch(self, prompt: str, system_prompt: str):
        return ("Repeating same bad code", "def op(x, y):\n    return x + y\n")


class _StallingReActLLM(BaseLLM):
    is_mock = False
    supports_tools = True
    backend_label = "StallingReActDouble"
    model = "stall-react"

    def generate_patch(self, prompt: str, system_prompt: str):
        return ("noop", "")

    def chat_step(self, messages, tools=None):
        return AgentTurn(
            thought="Reading same file endlessly without progress.",
            tool_calls=[{"id": "tc_dup", "name": "read_file", "arguments": {"filepath": "mod.py"}}],
            raw_text="loop",
            model=self.model,
            provider=self.backend_label,
        )


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.tmp_ctx = tempfile.TemporaryDirectory(prefix="acs_agent_test_")
        self.tmp_dir = self.tmp_ctx.name

    def tearDown(self):
        self.tmp_ctx.cleanup()

    def test_builtin_cases_smoke_convergence(self):
        for cid, info in BENCHMARK_CASES.items():
            ws = os.path.join(self.tmp_dir, cid)
            shutil.copytree(info["workspace"], ws, ignore=shutil.ignore_patterns("__pycache__"))
            inject_defective_starter(cid, workspace_override=ws)

            agent = CodeIterationAgent(
                target_file=info["target"],
                writable_files=info["writable_files"],
                test_file=info["test"],
                holdout_test_file=info.get("holdout"),
                workspace_root=ws,
                max_iterations=4,
                llm_backend=DeterministicMockLLM(),
                agent_mode=info["agent_mode"],
                strict_ast=info["strict_ast"],
                gate_runner=info.get("gate_runner", "auto"),
            )
            result = agent.run_loop()
            self.assertTrue(result["success"], msg=f"{cid} failed: {result}")
            self.assertTrue(result["is_mock"])
            self.assertTrue(result["integrity_verified"])
            self.assertTrue(len(result["unified_patch"]) > 0)

    def test_rollback_on_regression_restores_best_state(self):
        ws = os.path.join(self.tmp_dir, "rollback_ws")
        os.makedirs(ws)
        with open(os.path.join(ws, "mod.py"), "w", encoding="utf-8") as f:
            f.write("def op(x, y):\n    return x * y + (1 if x == 0 else 0)\n")
        with open(os.path.join(ws, "test_mod.py"), "w", encoding="utf-8") as f:
            f.write(
                "import unittest\nfrom mod import op\n"
                "class T(unittest.TestCase):\n"
                "    def test_a(self):\n        self.assertEqual(op(0, 5), 1)\n"
                "    def test_b(self):\n        self.assertEqual(op(2, 3), 7)\n"
            )

        agent = CodeIterationAgent(
            target_file="mod.py",
            test_file="test_mod.py",
            workspace_root=ws,
            max_iterations=3,
            llm_backend=_RegressingLLM(),
        )
        res = agent.run_loop()
        self.assertTrue(res["success"])
        self.assertGreaterEqual(res["telemetry"]["rollbacks"], 1)

    def test_stagnation_detection_halts_early(self):
        ws = os.path.join(self.tmp_dir, "stall_ws")
        os.makedirs(ws)
        with open(os.path.join(ws, "mod.py"), "w", encoding="utf-8") as f:
            f.write("def op(x, y):\n    return x + y\n")
        with open(os.path.join(ws, "test_mod.py"), "w", encoding="utf-8") as f:
            f.write(
                "import unittest\nfrom mod import op\n"
                "class T(unittest.TestCase):\n"
                "    def test_mul(self):\n        self.assertEqual(op(3, 4), 12)\n"
            )
        agent = CodeIterationAgent(
            target_file="mod.py",
            test_file="test_mod.py",
            workspace_root=ws,
            max_iterations=8,
            llm_backend=_StallingLLM(),
        )
        res = agent.run_loop()
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "STALLED")
        self.assertLess(res["iterations"], 8)

    def test_react_stagnation_detection_halts_early(self):
        ws = os.path.join(self.tmp_dir, "stall_react_ws")
        os.makedirs(ws)
        with open(os.path.join(ws, "mod.py"), "w", encoding="utf-8") as f:
            f.write("def op(x):\n    return x\n")
        with open(os.path.join(ws, "test_mod.py"), "w", encoding="utf-8") as f:
            f.write(
                "import unittest\nfrom mod import op\n"
                "class T(unittest.TestCase):\n"
                "    def test_op(self):\n        self.assertEqual(op(2), 5)\n"
            )
        agent = CodeIterationAgent(
            writable_files=["mod.py"],
            test_file="test_mod.py",
            workspace_root=ws,
            max_iterations=6,
            max_tool_steps=15,
            llm_backend=_StallingReActLLM(),
            agent_mode="react",
        )
        res = agent.run_loop()
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "STALLED")
        self.assertLessEqual(res["telemetry"]["llm_calls"], 4)

    def test_context_window_manager_preserves_tool_pairing(self):
        mgr = ContextWindowManager(max_context_tokens=520, keep_recent_messages=2, min_compact_chars=120)
        big_file_obs = json.dumps({
            "ok": True,
            "filepath": "big.py",
            "total_lines": 500,
            "start_line": 1,
            "end_line": 200,
            "content": "\n".join(f"{i}: x_{i} = {i} * 123456" for i in range(1, 160)),
        })
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task brief"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": big_file_obs},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "recent short obs"},
        ]
        compacted, stats = mgr.compact_if_needed(messages)
        self.assertEqual(stats["compacted"], 1)
        self.assertGreater(stats["reclaimed_tokens"], 200)
        self.assertEqual(len(compacted), 6)
        self.assertEqual(compacted[2]["role"], "assistant")
        self.assertEqual(compacted[3]["role"], "tool")
        self.assertIn('"compacted": true', compacted[3]["content"])
        self.assertEqual(compacted[5]["content"], "recent short obs")

    def test_holdout_gate_blocks_overfit_solution(self):
        ws = os.path.join(self.tmp_dir, "holdout_ws")
        os.makedirs(ws)
        with open(os.path.join(ws, "mod.py"), "w", encoding="utf-8") as f:
            f.write("def double(x):\n    return 4\n")
        with open(os.path.join(ws, "test_vis.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nfrom mod import double\nclass T(unittest.TestCase):\n    def test_two(self): self.assertEqual(double(2), 4)\n")
        with open(os.path.join(ws, "test_holdout.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nfrom mod import double\nclass H(unittest.TestCase):\n    def test_five(self): self.assertEqual(double(5), 10)\n")
        agent = CodeIterationAgent(
            target_file="mod.py",
            test_file="test_vis.py",
            holdout_test_file="test_holdout.py",
            workspace_root=ws,
            max_iterations=3,
            llm_backend=DeterministicMockLLM(),
        )
        res = agent.run_loop()
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "HOLDOUT_FAILED")

    def test_live_http_client_react_unseen_problem(self):
        turn_counter = {"n": 0}

        class MockOpenAIServer(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path.endswith("/models"):
                    body = json.dumps({"data": [{"id": "unseen-test-coder"}]}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(404)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(length)
                turn_counter["n"] += 1
                step = turn_counter["n"]

                if step == 1:
                    msg = {
                        "role": "assistant",
                        "content": "<think>Inspecting rate_limiter.py first.</think>",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": json.dumps({"filepath": "rate_limiter.py"})},
                        }],
                    }
                elif step == 2:
                    msg = {
                        "role": "assistant",
                        "content": "Fixing token bucket logic.",
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps({
                                    "filepath": "rate_limiter.py",
                                    "content": (
                                        "class TokenBucket:\n"
                                        "    def __init__(self, cap):\n"
                                        "        self.cap = cap\n"
                                        "        self.tokens = cap\n"
                                        "    def allow(self, cost=1):\n"
                                        "        if cost <= 0 or cost > self.tokens:\n"
                                        "            return False\n"
                                        "        self.tokens -= cost\n"
                                        "        return True\n"
                                    ),
                                }),
                            },
                        }],
                    }
                else:
                    msg = {
                        "role": "assistant",
                        "content": "Running tests.",
                        "tool_calls": [{
                            "id": "call_3",
                            "type": "function",
                            "function": {"name": "run_tests", "arguments": "{}"},
                        }],
                    }

                resp = {
                    "model": "unseen-test-coder",
                    "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 25},
                }
                raw = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)

        server = HTTPServer(("127.0.0.1", 0), MockOpenAIServer)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            ws = os.path.join(self.tmp_dir, "unseen_ws")
            os.makedirs(ws)
            with open(os.path.join(ws, "rate_limiter.py"), "w", encoding="utf-8") as f:
                f.write("class TokenBucket:\n    def __init__(self, cap):\n        self.tokens = 0\n    def allow(self, cost=1):\n        return True\n")
            with open(os.path.join(ws, "test_rate_limiter.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import unittest\nfrom rate_limiter import TokenBucket\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_bucket(self):\n"
                    "        b = TokenBucket(3)\n"
                    "        self.assertTrue(b.allow(2))\n"
                    "        self.assertFalse(b.allow(2))\n"
                )

            client = LiveLLMClient(base_url=f"http://127.0.0.1:{port}/v1", model="unseen-test-coder")
            probe = probe_llm_endpoint(f"http://127.0.0.1:{port}/v1")
            self.assertTrue(probe["ok"])
            agent = CodeIterationAgent(
                writable_files=["rate_limiter.py"],
                test_file="test_rate_limiter.py",
                workspace_root=ws,
                max_iterations=3,
                llm_backend=client,
                agent_mode="react",
            )
            res = agent.run_loop()
            self.assertTrue(res["success"], msg=res)
            self.assertFalse(res["is_mock"])
            self.assertEqual(res["backend_mode"], "live_llm")
        finally:
            server.shutdown()
            server.server_close()

    def test_live_http_client_falls_back_to_text_react_on_http_400(self):
        turn_counter = {"n": 0}

        class LegacyLocalServer(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                req_json = json.loads(self.rfile.read(length).decode("utf-8"))
                if "tools" in req_json:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "tools parameter unsupported"}).encode("utf-8"))
                    return

                turn_counter["n"] += 1
                if turn_counter["n"] == 1:
                    content = (
                        "[THOUGHT] Patching clamp helper cleanly. [/THOUGHT]\n"
                        "[TOOL_CALL] "
                        + json.dumps({
                            "name": "write_file",
                            "arguments": {
                                "filepath": "clamp.py",
                                "content": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
                            },
                        })
                        + " [/TOOL_CALL]"
                    )
                else:
                    content = '[THOUGHT] Verifying. [/THOUGHT]\n[TOOL_CALL] {"name": "run_tests", "arguments": {}} [/TOOL_CALL]'

                resp = {
                    "model": "local-legacy-coder",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 15},
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))

        server = HTTPServer(("127.0.0.1", 0), LegacyLocalServer)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            ws = os.path.join(self.tmp_dir, "legacy_ws")
            os.makedirs(ws)
            with open(os.path.join(ws, "clamp.py"), "w", encoding="utf-8") as f:
                f.write("def clamp(x, lo, hi):\n    return x\n")
            with open(os.path.join(ws, "test_clamp.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import unittest\nfrom clamp import clamp\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_clamp(self):\n"
                    "        self.assertEqual(clamp(15, 0, 10), 10)\n"
                    "        self.assertEqual(clamp(-5, 0, 10), 0)\n"
                )
            client = LiveLLMClient(base_url=f"http://127.0.0.1:{port}/v1", model="local-legacy-coder")
            agent = CodeIterationAgent(
                writable_files=["clamp.py"],
                test_file="test_clamp.py",
                workspace_root=ws,
                max_iterations=3,
                llm_backend=client,
                agent_mode="react",
            )
            res = agent.run_loop()
            self.assertTrue(res["success"], msg=res)
            self.assertFalse(client.enable_native_tools)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
