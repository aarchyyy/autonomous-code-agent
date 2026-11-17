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

    def test_starter_defects_fail_and_cheat_patches_rejected(self):
        for cid, info in BENCHMARK_CASES.items():
            ws = os.path.join(self.tmp, cid)
            shutil.copytree(info["workspace"], ws, ignore=shutil.ignore_patterns("__pycache__"))

            # Patched repo state must pass both visible & holdout tests
            tb_pass = ToolBox(
                workspace_root=ws,
                use_ephemeral_copy=True,
                gate_runner=info.get("gate_runner", "auto"),
            )
            res_pass = tb_pass.run_tdd_gate(info["test"])
            self.assertTrue(res_pass["passed"], msg=f"{cid} patched code failed: {res_pass['output']}")
            if info.get("holdout"):
                h_pass = tb_pass.run_tdd_gate(info["holdout"], include_hidden=True)
                self.assertTrue(h_pass["passed"], msg=f"{cid} holdout failed: {h_pass['output']}")

            # Starter defective code must fail
            inject_defective_starter(cid, workspace_override=ws)
            res_fail = tb_pass.run_tdd_gate(info["test"])
            self.assertFalse(res_fail["passed"], msg=f"{cid} starter unexpectedly passed!")

        # Verify naive `round((a + b) * scale, 6)` fails Case 2's IEEE-754 precision gate
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

        # Verify naive single-precision float32 accumulator fails Case 5's C++17 holdout cancellation check
        ws5 = os.path.join(self.tmp, "case5")
        with open(os.path.join(ws5, "kernel.hpp"), "w", encoding="utf-8") as f:
            f.write(
                "#pragma once\n#include <cmath>\n#include <stdexcept>\n#include <vector>\n"
                "inline std::vector<float> tiled_csr_spmv_f32(\n"
                "    const std::vector<int>& row_ptr, const std::vector<int>& col_indices,\n"
                "    const std::vector<float>& values, const std::vector<float>& x,\n"
                "    int num_rows, int num_cols) {\n"
                "    if (num_rows < 0 || num_cols < 0 || (int)row_ptr.size() != num_rows + 1\n"
                "        || col_indices.size() != values.size() || (int)x.size() != num_cols) throw std::invalid_argument(\"bad\");\n"
                "    std::vector<float> y(num_rows, 0.0f);\n"
                "    for (int r = 0; r < num_rows; ++r) {\n"
                "        int s = row_ptr[r], e = row_ptr[r+1];\n"
                "        if (s < 0 || e < s || e > (int)values.size()) throw std::out_of_range(\"rp\");\n"
                "        float acc = 0.0f;  // Naive float32 loses 1.0f in (1e8f + 1.0f - 1e8f)\n"
                "        for (int k = s; k < e; ++k) {\n"
                "            if (!std::isfinite(values[k]) || !std::isfinite(x[0])) throw std::invalid_argument(\"nan\");\n"
                "            if (col_indices[k] < 0 || col_indices[k] >= num_cols) throw std::out_of_range(\"col\");\n"
                "            acc += values[k] * x[col_indices[k]];\n"
                "        }\n"
                "        y[r] = acc;\n"
                "    }\n"
                "    return y;\n"
                "}\n"
            )
        tb5 = ToolBox(workspace_root=ws5, use_ephemeral_copy=True, gate_runner="cpp")
        vis5 = tb5.run_tdd_gate("test_kernel.cpp")
        self.assertTrue(vis5["passed"], msg=vis5["output"])
        hold5 = tb5.run_tdd_gate("holdout_test_kernel.cpp", include_hidden=True)
        self.assertFalse(hold5["passed"], msg="Naive float32 accumulator must fail C++17 cancellation holdout!")


if __name__ == "__main__":
    unittest.main()
