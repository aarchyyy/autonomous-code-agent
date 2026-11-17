import json
import os
import tempfile
import unittest

from src.benchmarks_catalog import stage_benchmarks_copy
from src.llm_backend import AgentTurn, BaseLLM, DeterministicMockLLM
from src.relay import RelayOrchestrator, RelayPlan, RelayTask


class _RegressionBreakingLLM(BaseLLM):
    is_mock = True
    backend_label = "RegressionBreaker"
    model = "reg-breaker"

    def generate_patch(self, prompt: str, system_prompt: str):
        if "test_t1" in prompt:
            return ("Fix T1", "def f1(x):\n    return x + 1\n")
        return ("Break T1 while fixing T2", "def f1(x):\n    return 999\n")


class _SelectiveFixLLM(BaseLLM):
    """Fails T1, succeeds independent T3, and tracks calls for resume & token budget tests."""

    is_mock = False
    backend_label = "SelectiveFixDouble"
    model = "selective-fix"

    def __init__(self, fail_t1: bool = False):
        self.fail_t1 = fail_t1
        self.calls = 0

    def generate_patch(self, prompt: str, system_prompt: str):
        self.calls += 1
        if "test_a" in prompt:
            if self.fail_t1:
                return ("Still broken", "def a(x):\n    return -999\n")
            return ("Fix A", "def a(x):\n    return x + 1\n")
        if "test_b" in prompt:
            return ("Fix B", "def b(x):\n    return x * 2\n")
        return ("Fix C", "def c(x):\n    return x * 3\n")

    def chat_step(self, messages, tools=None):
        turn = super().chat_step(messages, tools=tools)
        turn.prompt_tokens = 400
        turn.completion_tokens = 200
        return turn


class TestNightRelay(unittest.TestCase):
    def setUp(self):
        self.tmp_ctx = tempfile.TemporaryDirectory(prefix="acs_relay_test_")
        self.ws = self.tmp_ctx.name

    def tearDown(self):
        self.tmp_ctx.cleanup()

    def test_builtin_night_relay_plan_generates_handover_patch_and_ledger(self):
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
        self.assertEqual(summary["tasks_succeeded"], 5)
        self.assertTrue(os.path.exists(summary["handover_report_path"]))
        self.assertTrue(os.path.exists(summary["ledger_path"]))
        self.assertTrue(os.path.exists(summary["patch_bundle_path"]))
        self.assertIn("tiled_csr_spmv_f32", summary["patch_bundle"])

        with open(summary["handover_report_path"], "r", encoding="utf-8") as f:
            md = f.read()
        self.assertIn("Offline Smoke Mock", md)
        self.assertIn("T5_cpp_csr_spmv", md)

    def test_cross_task_regression_triggers_rollback(self):
        with open(os.path.join(self.ws, "shared.py"), "w", encoding="utf-8") as f:
            f.write("def f1(x):\n    return x\n")
        with open(os.path.join(self.ws, "test_t1.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nfrom shared import f1\nclass T1(unittest.TestCase):\n    def test_t1(self): self.assertEqual(f1(2), 3)\n")
        with open(os.path.join(self.ws, "test_t2.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nfrom shared import f1\nclass T2(unittest.TestCase):\n    def test_t2(self): self.assertEqual(f1(5), 999)\n")

        plan = RelayPlan(
            project_name="Regression Check Project",
            tasks=[
                RelayTask(task_id="T1", title="Implement f1", writable_files=["shared.py"], test_file="test_t1.py", agent_mode="patch"),
                RelayTask(task_id="T2", title="Conflicting overwrite", writable_files=["shared.py"], test_file="test_t2.py", agent_mode="patch"),
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
        with open(os.path.join(self.ws, "shared.py"), "r", encoding="utf-8") as f:
            self.assertIn("return x + 1", f.read())

    def test_dag_cycle_detection_and_branch_isolation(self):
        cyclic = RelayPlan(
            project_name="Cyclic",
            tasks=[
                RelayTask(task_id="A", title="A", writable_files=["a.py"], test_file="ta.py", depends_on=["B"]),
                RelayTask(task_id="B", title="B", writable_files=["b.py"], test_file="tb.py", depends_on=["A"]),
            ],
        )
        with self.assertRaises(ValueError):
            cyclic.validate_dag()

        for mod in ("a", "b", "c"):
            with open(os.path.join(self.ws, f"{mod}.py"), "w", encoding="utf-8") as f:
                f.write(f"def {mod}(x):\n    return 0\n")
        with open(os.path.join(self.ws, "test_a.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nfrom a import a\nclass T(unittest.TestCase):\n    def test_a(self): self.assertEqual(a(2), 3)\n")
        with open(os.path.join(self.ws, "test_b.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nfrom b import b\nclass T(unittest.TestCase):\n    def test_b(self): self.assertEqual(b(2), 4)\n")
        with open(os.path.join(self.ws, "test_c.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\nfrom c import c\nclass T(unittest.TestCase):\n    def test_c(self): self.assertEqual(c(2), 6)\n")

        dag_plan = RelayPlan(
            project_name="BranchIsolation",
            max_consecutive_failures=5,
            tasks=[
                RelayTask(task_id="T1", title="Failing root", writable_files=["a.py"], test_file="test_a.py", max_iterations=2, agent_mode="patch"),
                RelayTask(task_id="T2", title="Depends on T1", writable_files=["b.py"], test_file="test_b.py", depends_on=["T1"], agent_mode="patch"),
                RelayTask(task_id="T3", title="Independent branch", writable_files=["c.py"], test_file="test_c.py", depends_on=[], agent_mode="patch"),
            ],
        )
        orch = RelayOrchestrator(
            workspace_root=self.ws,
            plan=dag_plan,
            llm_backend=_SelectiveFixLLM(fail_t1=True),
            output_dir=os.path.join(self.ws, "dag_out"),
        )
        res = orch.run()
        by_id = {t["task_id"]: t for t in res["tasks"]}
        self.assertFalse(by_id["T1"]["success"])
        self.assertEqual(by_id["T2"]["status"], "SKIPPED_DEPENDENCY")
        self.assertEqual(by_id["T3"]["status"], "CONVERGED_SUCCESS")

    def test_crash_checkpoint_resume_reuses_verified_tasks(self):
        for mod, mul in (("a", "+ 1"), ("b", "* 2"), ("c", "* 3")):
            with open(os.path.join(self.ws, f"{mod}.py"), "w", encoding="utf-8") as f:
                f.write(f"def {mod}(x):\n    return 0\n")
            with open(os.path.join(self.ws, f"test_{mod}.py"), "w", encoding="utf-8") as f:
                f.write(f"import unittest\nfrom {mod} import {mod}\nclass T(unittest.TestCase):\n    def test_{mod}(self): self.assertEqual({mod}(2), 2 {mul})\n")

        # Run 1: token budget allows T1 & T2 (600 tokens each = 1200) and skips T3!
        plan1 = RelayPlan(
            project_name="InterruptedRun",
            max_total_tokens=1000,  # T1 uses 600 (<1000), T2 uses 600 (-> 1200 >= 1000), T3 skipped!
            tasks=[
                RelayTask(task_id="T1", title="Task A", writable_files=["a.py"], test_file="test_a.py", agent_mode="patch"),
                RelayTask(task_id="T2", title="Task B", writable_files=["b.py"], test_file="test_b.py", depends_on=["T1"], agent_mode="patch"),
                RelayTask(task_id="T3", title="Task C", writable_files=["c.py"], test_file="test_c.py", depends_on=["T2"], agent_mode="patch"),
            ],
        )
        run1_dir = os.path.join(self.ws, "run1")
        orch1 = RelayOrchestrator(workspace_root=self.ws, plan=plan1, llm_backend=_SelectiveFixLLM(), output_dir=run1_dir)
        s1 = orch1.run()
        self.assertEqual(s1["tasks"][2]["status"], "SKIPPED_TOKEN_BUDGET")

        # Run 2: Resume from run1_dir with unlimited token budget -> T1 & T2 re-verified with 0 LLM calls, T3 fixed!
        plan2 = RelayPlan(project_name="ResumedRun", max_total_tokens=0, tasks=plan1.tasks)
        llm2 = _SelectiveFixLLM()
        orch2 = RelayOrchestrator(
            workspace_root=self.ws,
            plan=plan2,
            llm_backend=llm2,
            output_dir=os.path.join(self.ws, "run2"),
            resume_from_dir=run1_dir,
        )
        s2 = orch2.run()
        self.assertTrue(s2["all_succeeded"])
        self.assertEqual(s2["tasks"][0]["status"], "RESUMED_VERIFIED")
        self.assertEqual(s2["tasks"][1]["status"], "RESUMED_VERIFIED")
        self.assertEqual(s2["tasks"][2]["status"], "CONVERGED_SUCCESS")
        self.assertEqual(llm2.calls, 1)


if __name__ == "__main__":
    unittest.main()
