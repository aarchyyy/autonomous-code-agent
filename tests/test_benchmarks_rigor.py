import os
import shutil
import tempfile
import unittest

from src.benchmarks_catalog import BENCHMARK_CASES, inject_defective_starter
from src.tools import ToolBox


class TestBenchmarksRigor(unittest.TestCase):
    def setUp(self):
        self.tmp_ctx = tempfile.TemporaryDirectory(prefix="acs_bench_rigor_")
        self.tmp = self.tmp_ctx.name

    def tearDown(self):
        self.tmp_ctx.cleanup()

    def test_starter_defects_fail_and_round6_hack_fails_case2(self):
        for cid, info in BENCHMARK_CASES.items():
            ws = os.path.join(self.tmp, cid)
            shutil.copytree(info["workspace"], ws, ignore=shutil.ignore_patterns("__pycache__"))

            # Patched repo state must pass both visible & holdout tests
            tb_pass = ToolBox(workspace_root=ws, use_ephemeral_copy=True)
            res_pass = tb_pass.run_tdd_gate(info["test"])
            self.assertTrue(res_pass["passed"], msg=f"{cid} patched code failed: {res_pass['output']}")
            if info.get("holdout"):
                h_pass = tb_pass.run_tdd_gate(info["holdout"], include_hidden=True)
                self.assertTrue(h_pass["passed"], msg=f"{cid} holdout failed: {h_pass['output']}")

            # Starter defective code must fail
            inject_defective_starter(cid, workspace_override=ws)
            res_fail = tb_pass.run_tdd_gate(info["test"])
            self.assertFalse(res_fail["passed"], msg=f"{cid} starter unexpectedly passed!")

        # Explicitly verify that naive `round((a + b) * scale, 6)` fails Case 2's IEEE-754 precision gate
        ws2 = os.path.join(self.tmp, "case2")
        with open(os.path.join(ws2, "kernel.py"), "w", encoding="utf-8") as f:
            f.write(
                "def vector_scale_kernel(tensor_a, tensor_b, scale=1.0):\n"
                "    if len(tensor_a) != len(tensor_b):\n"
                "        raise ValueError('mismatch')\n"
                "    return [round((a + b) * scale, 6) for a, b in zip(tensor_a, tensor_b)]\n"
            )
        tb2 = ToolBox(workspace_root=ws2, use_ephemeral_copy=True)
        res_round6 = tb2.run_tdd_gate("test_kernel.py")
        self.assertFalse(res_round6["passed"], msg="round(..., 6) hack should fail IEEE-754 float32 test!")


if __name__ == "__main__":
    unittest.main()
