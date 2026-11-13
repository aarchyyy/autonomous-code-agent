import json
import os
import tempfile
import unittest

from src.benchmarks_catalog import stage_benchmarks_copy
from src.llm_backend import BaseLLM, DeterministicMockLLM
from src.relay import RelayOrchestrator, RelayPlan, RelayTask


class _RegressionBreakingLLM(BaseLLM):
    """Completes T1 cleanly, then breaks T1's shared module during T2."""

    is_mock = True
    backend_label = "RegressionBreaker"
    model = "reg-breaker"

    def generate_patch(self, prompt: str, system_prompt: str):
        if "test_t1" in prompt:
            return ("Fix T1", "def f1(x):\n    return x + 1\n")
        return ("Break T1 while fixing T2", "def f1(x):\n    return 999\n")


class TestNightRelay(unittest.TestCase):
    def setUp(self):
        self.tmp_ctx = tempfile.TemporaryDirectory(prefix="acs_relay_test_")
        self.ws = self.tmp_ctx.name

    def tearDown(self):
        self.tmp_ctx.cleanup()

    def test_builtin_night_relay_plan_generates_handover_and_ledger(self):
        stage_benchmarks_copy(self.ws)
        plan_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "benchmarks/night_relay_plan.json",
        )
        plan = RelayPlan.from_json_file(plan_path)
        out_dir = os.path.join(self.ws, "out_reports")

        orch = RelayOrchestrator(
            workspace_root=self.ws,
            plan=plan,
            llm_backend=DeterministicMockLLM(),
            output_dir=out_dir,
            enable_git_branch=False,
        )
        summary = orch.run()
        self.assertTrue(summary["all_succeeded"], msg=summary)
        self.assertEqual(summary["tasks_succeeded"], 4)
        self.assertTrue(os.path.exists(summary["handover_report_path"]))
        self.assertTrue(os.path.exists(summary["ledger_path"]))

        with open(summary["handover_report_path"], "r", encoding="utf-8") as f:
            md = f.read()
        self.assertIn("Offline Smoke Mock", md)
        self.assertIn("T4_multi_file_scheduler", md)

        with open(summary["ledger_path"], "r", encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
        self.assertGreaterEqual(len(events), 6)

    def test_cross_task_regression_triggers_rollback(self):
        with open(os.path.join(self.ws, "shared.py"), "w", encoding="utf-8") as f:
            f.write("def f1(x):\n    return x\n")
        with open(os.path.join(self.ws, "helper.py"), "w", encoding="utf-8") as f:
            f.write("def f2(x):\n    return x\n")
        with open(os.path.join(self.ws, "test_t1.py"), "w", encoding="utf-8") as f:
            f.write(
                "import unittest\nfrom shared import f1\n"
                "class T1(unittest.TestCase):\n"
                "    def test_t1(self):\n        self.assertEqual(f1(2), 3)\n"
            )
        with open(os.path.join(self.ws, "test_t2.py"), "w", encoding="utf-8") as f:
            f.write(
                "import unittest\nfrom shared import f1\n"
                "class T2(unittest.TestCase):\n"
                "    def test_t2(self):\n        self.assertEqual(f1(5), 999)\n"
            )

        plan = RelayPlan(
            project_name="Regression Check Project",
            tasks=[
                RelayTask(
                    task_id="T1",
                    title="Implement f1",
                    writable_files=["shared.py"],
                    test_file="test_t1.py",
                    agent_mode="patch",
                ),
                RelayTask(
                    task_id="T2",
                    title="Conflicting overwrite of shared.py",
                    writable_files=["shared.py"],
                    test_file="test_t2.py",
                    agent_mode="patch",
                ),
            ],
        )
        orch = RelayOrchestrator(
            workspace_root=self.ws,
            plan=plan,
            llm_backend=_RegressionBreakingLLM(),
            output_dir=os.path.join(self.ws, "out"),
        )
        summary = orch.run()
        self.assertFalse(summary["all_succeeded"])
        self.assertEqual(summary["tasks"][0]["status"], "CONVERGED_SUCCESS")
        self.assertEqual(summary["tasks"][1]["status"], "REGRESSION_ROLLED_BACK")
        # Verify shared.py was rolled back to T1's passing version
        with open(os.path.join(self.ws, "shared.py"), "r", encoding="utf-8") as f:
            self.assertIn("return x + 1", f.read())


if __name__ == "__main__":
    unittest.main()
