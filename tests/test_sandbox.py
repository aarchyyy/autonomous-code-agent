import os
import tempfile
import unittest

from src.sandbox import (
    ASTPolicyInspector,
    IntegrityMonitor,
    IntegrityViolationError,
    IsolatedTestRunner,
    NativeSyntaxInspector,
    PathSecurityError,
    SandboxPolicy,
    detect_cpp_compiler,
    detect_docker_runtime,
)
from src.tools import ToolBox


class TestSandboxSecurity(unittest.TestCase):
    def setUp(self):
        self.tmp_ctx = tempfile.TemporaryDirectory(prefix="acs_test_sbx_")
        self.ws = self.tmp_ctx.name
        self.outside_ctx = tempfile.TemporaryDirectory(prefix="acs_test_out_")
        self.outside_dir = self.outside_ctx.name

        with open(os.path.join(self.ws, "target.py"), "w", encoding="utf-8") as f:
            f.write("def add(a, b):\n    return a + b\n")
        with open(os.path.join(self.ws, "test_target.py"), "w", encoding="utf-8") as f:
            f.write(
                "import unittest\nfrom target import add\n"
                "class T(unittest.TestCase):\n"
                "    def test_ok(self):\n        self.assertEqual(add(1, 2), 3)\n"
            )
        with open(os.path.join(self.ws, "holdout.py"), "w", encoding="utf-8") as f:
            f.write("SECRET_HOLDOUT = 123\n")
        with open(os.path.join(self.outside_dir, "secret.txt"), "w", encoding="utf-8") as f:
            f.write("OUTSIDE_SECRET")

        self.tools = ToolBox(
            workspace_root=self.ws,
            writable_files=["target.py"],
            protected_files=["test_target.py"],
            hidden_files=["holdout.py"],
            strict_ast=True,
            use_ephemeral_copy=True,
        )

    def tearDown(self):
        self.tmp_ctx.cleanup()
        self.outside_ctx.cleanup()

    def test_blocks_path_traversal_and_absolute_escape(self):
        with self.assertRaises(PathSecurityError):
            self.tools.read_source("../outside.txt")
        with self.assertRaises(PathSecurityError):
            self.tools.read_source(os.path.join(self.outside_dir, "secret.txt"))
        with self.assertRaises(PathSecurityError):
            self.tools.write_source(os.path.join(self.outside_dir, "pwn.py"), "x = 1\n")

    def test_blocks_symlink_escape(self):
        link_path = os.path.join(self.ws, "escape_link.txt")
        os.symlink(os.path.join(self.outside_dir, "secret.txt"), link_path)
        with self.assertRaises(PathSecurityError):
            self.tools.read_source("escape_link.txt")

        dir_link = os.path.join(self.ws, "escape_dir")
        os.symlink(self.outside_dir, dir_link)
        with self.assertRaises(PathSecurityError):
            self.tools.write_source("escape_dir/new_pwn.py", "x = 1\n")

    def test_blocks_protected_and_non_allowlisted_writes(self):
        with self.assertRaises(PathSecurityError):
            self.tools.write_source("test_target.py", "# tampered\n")
        with self.assertRaises(PathSecurityError):
            self.tools.write_source("other.py", "x = 1\n")
        with self.assertRaises(PathSecurityError):
            self.tools.read_source("holdout.py")
        self.assertNotIn("holdout.py", self.tools.list_files())

    def test_strict_ast_blocks_dangerous_imports_and_calls(self):
        ok_os, msg_os = ASTPolicyInspector.inspect("import os\nos.system('echo hi')\n", strict=True)
        self.assertFalse(ok_os)
        self.assertIn("forbidden import", msg_os)

        ok_eval, msg_eval = ASTPolicyInspector.inspect("x = eval('1+1')\n", strict=True)
        self.assertFalse(ok_eval)
        self.assertIn("forbidden builtin call", msg_eval)

    def test_strict_ast_fine_grained_allowlist(self):
        code = "import ctypes\nimport pathlib\nx = 1\n"
        ok_blocked, _ = ASTPolicyInspector.inspect(code, strict=True)
        self.assertFalse(ok_blocked)

        ok_allowed, _ = ASTPolicyInspector.inspect(
            code,
            strict=True,
            allowed_imports={"ctypes", "pathlib"},
        )
        self.assertTrue(ok_allowed)

        ok_still_blocks_subproc, _ = ASTPolicyInspector.inspect(
            "import ctypes, subprocess\n",
            strict=True,
            allowed_imports={"ctypes"},
        )
        self.assertFalse(ok_still_blocks_subproc)

    def test_ephemeral_sandbox_scrubs_secrets_and_isolates_cwd_writes(self):
        old_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-super-secret-do-not-leak"
        try:
            with open(os.path.join(self.ws, "test_env_leak.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import os, unittest\n"
                    "class T(unittest.TestCase):\n"
                    "    def test_isolation(self):\n"
                    "        self.assertNotIn('OPENAI_API_KEY', os.environ)\n"
                    "        with open('ephemeral_only.txt', 'w') as fp:\n"
                    "            fp.write('side_effect')\n"
                )
            tb = ToolBox(
                workspace_root=self.ws,
                writable_files=["target.py"],
                protected_files=["test_env_leak.py"],
                hidden_files=["holdout.py"],
                use_ephemeral_copy=True,
            )
            res = tb.run_tdd_gate("test_env_leak.py")
            self.assertTrue(res["passed"], msg=res["output"])
            self.assertFalse(os.path.exists(os.path.join(self.ws, "ephemeral_only.txt")))
        finally:
            if old_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_integrity_monitor_detects_external_tampering(self):
        mon = IntegrityMonitor(self.ws, ["test_target.py"])
        self.assertTrue(mon.verify()[0])
        with open(os.path.join(self.ws, "test_target.py"), "a", encoding="utf-8") as f:
            f.write("\n# tampered externally\n")
        ok, violations = mon.verify()
        self.assertFalse(ok)
        self.assertEqual(len(violations), 1)
        with self.assertRaises(IntegrityViolationError):
            self.tools.run_tdd_gate("test_target.py")

    def test_watchdog_kills_infinite_loop(self):
        with open(os.path.join(self.ws, "test_hang.py"), "w", encoding="utf-8") as f:
            f.write(
                "import unittest\n"
                "class T(unittest.TestCase):\n"
                "    def test_loop(self):\n"
                "        while True:\n            pass\n"
            )
        tb = ToolBox(workspace_root=self.ws, use_ephemeral_copy=True)
        res = tb.run_tdd_gate("test_hang.py", timeout=1)
        self.assertFalse(res["passed"])
        self.assertEqual(res["summary"], "WATCHDOG_TIMEOUT")
        self.assertEqual(res["exit_code"], 124)

    def test_cpp17_syntax_inspector_and_native_compile_gate(self):
        if not detect_cpp_compiler():
            self.skipTest("No C++ compiler available on PATH")

        ok_good, _ = NativeSyntaxInspector.inspect_cpp(
            "op.hpp",
            "#pragma once\ninline int sq(int x) { return x * x; }\n",
            workspace_root=self.ws,
        )
        self.assertTrue(ok_good)

        ok_bad, err_bad = NativeSyntaxInspector.inspect_cpp(
            "op.hpp",
            "#pragma once\ninline int sq(int x { return x; }\n",
            workspace_root=self.ws,
        )
        self.assertFalse(ok_bad)
        self.assertIn("Error", err_bad)

        cpp_ws = os.path.join(self.ws, "cpp_gate_ws")
        os.makedirs(cpp_ws)
        with open(os.path.join(cpp_ws, "op.hpp"), "w", encoding="utf-8") as f:
            f.write("#pragma once\ninline int sq(int x) { return x * x; }\n")
        with open(os.path.join(cpp_ws, "test_op.cpp"), "w", encoding="utf-8") as f:
            f.write(
                '#include "op.hpp"\n#include <iostream>\n'
                "int main() {\n"
                "    bool ok = (sq(4) == 16);\n"
                '    std::cerr << "[SUMMARY] tests_run=1 passed=" << (ok ? 1 : 0) << " failed=" << (ok ? 0 : 1) << "\\n";\n'
                "    return ok ? 0 : 1;\n"
                "}\n"
            )
        tb = ToolBox(workspace_root=cpp_ws, writable_files=["op.hpp"], protected_files=["test_op.cpp"])
        gate = tb.run_tdd_gate("test_op.cpp")
        self.assertTrue(gate["passed"], msg=gate["output"])
        self.assertEqual(gate["passed_count"], 1)

    def test_docker_container_isolated_sandbox(self):
        runner = IsolatedTestRunner(
            SandboxPolicy(
                workspace_root=self.ws,
                protected_files={"test_target.py"},
                sandbox_backend="docker",
                docker_image="python:3.12-alpine",
            )
        )
        cmd = runner._build_docker_command(
            container_name="acs_sbx_unit",
            real_exec_ws=self.ws,
            rel_test_dir="",
            rel_staged_test="test_target.py",
            timeout=10,
            runner_kind="unittest",
        )
        self.assertIn("--network", cmd)
        self.assertIn("none", cmd)
        self.assertIn("--read-only", cmd)
        self.assertIn("--cap-drop", cmd)
        self.assertIn("ALL", cmd)
        self.assertIn("65534:65534", cmd)

        dkr = detect_docker_runtime("python:3.12-alpine")
        if dkr.get("available"):
            with open(os.path.join(self.ws, "test_dkr_ro.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import os, unittest\n"
                    "from target import add\n"
                    "class TDkr(unittest.TestCase):\n"
                    "    def test_readonly_mount_and_math(self):\n"
                    "        self.assertEqual(add(20, 22), 42)\n"
                    "        self.assertNotIn('OPENAI_API_KEY', os.environ)\n"
                    "        with self.assertRaises(OSError):\n"
                    "            open('/workspace/illegal_write.txt', 'w').write('x')\n"
                )
            tb_dkr = ToolBox(
                workspace_root=self.ws,
                writable_files=["target.py"],
                protected_files=["test_dkr_ro.py"],
                sandbox_backend="docker",
                docker_image="python:3.12-alpine",
            )
            res = tb_dkr.run_tdd_gate("test_dkr_ro.py", timeout=15)
            self.assertTrue(res["passed"], msg=res["output"])
            self.assertIn("docker_isolated", res["sandbox_mode"])


if __name__ == "__main__":
    unittest.main()
