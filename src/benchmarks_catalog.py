"""
Shared Catalog of Built-in Benchmark Cases and Defective Starter States.
Used by CLI demo (`run_demo.py`), Night Relay CLI (`run_relay.py`), and WebUI (`run_web.py`).
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BENCHMARK_CASES: Dict[str, Dict[str, Any]] = {
    "case1": {
        "id": "case1",
        "name": "Case 1: SpMM CSR Operator (Bounds, Negative-Index & math.fsum)",
        "short_name": "Case 1: SpMM CSR Operator",
        "subdir": "benchmarks/case1_spmm_boundary",
        "workspace": os.path.join(BASE_DIR, "benchmarks/case1_spmm_boundary"),
        "writable_files": ["kernel.py"],
        "target": "kernel.py",
        "test": "test_kernel.py",
        "holdout": None,
        "agent_mode": "patch",
        "strict_ast": True,
        "spec": (
            "Implement spmm_csr_operator(row_ptr, col_indices, values, dense_b, num_rows, num_cols). "
            "Validate CSR lengths/monotonicity, reject negative or out-of-range column indices, "
            "reject non-finite numbers, fast-skip empty rows, and accumulate with math.fsum."
        ),
        "initial_files": {
            "kernel.py": (
                "# Defective Initial Implementation: CSR Dynamic Sparse Matrix Multiplication\n"
                "def spmm_csr_operator(row_ptr, col_indices, values, dense_b, num_rows, num_cols):\n"
                "    result = [[0.0 for _ in range(num_cols)] for _ in range(num_rows)]\n"
                "    for r in range(num_rows):\n"
                "        start = row_ptr[r]\n"
                "        end = row_ptr[r + 1]\n"
                "        for k in range(start, end):\n"
                "            c = col_indices[k]\n"
                "            val = values[k]\n"
                "            for j in range(num_cols):\n"
                "                result[r][j] += val * dense_b[c][j]\n"
                "    return result\n"
            )
        },
    },
    "case2": {
        "id": "case2",
        "name": "Case 2: Vector Scale Operator (IEEE-754 Float32 0-ULP & Finiteness)",
        "short_name": "Case 2: Vector Scale (IEEE-754 f32)",
        "subdir": "benchmarks/case2_vector_scale",
        "workspace": os.path.join(BASE_DIR, "benchmarks/case2_vector_scale"),
        "writable_files": ["kernel.py"],
        "target": "kernel.py",
        "test": "test_kernel.py",
        "holdout": None,
        "agent_mode": "patch",
        "strict_ast": True,
        "spec": (
            "Implement vector_scale_kernel(tensor_a, tensor_b, scale=1.0) computing IEEE-754 float32 "
            "(a + b) * scale without round(..., 6) truncation; raise ValueError on length mismatch or NaN/Inf."
        ),
        "initial_files": {
            "kernel.py": (
                "# Defective Initial Implementation: Vector Scale Operator\n"
                "def vector_scale_kernel(tensor_a, tensor_b, scale=1.0):\n"
                "    # BUG: Omitted scale factor, shape check, finiteness check, and IEEE-754 float32 alignment\n"
                "    return [round(a + b, 6) for a, b in zip(tensor_a, tensor_b)]\n"
            )
        },
    },
    "case3": {
        "id": "case3",
        "name": "Case 3: GQA KV-Cache Broadcast (Shape Validation & Buffer Isolation)",
        "short_name": "Case 3: GQA KV-Cache Broadcast",
        "subdir": "benchmarks/case3_kv_cache_broadcast",
        "workspace": os.path.join(BASE_DIR, "benchmarks/case3_kv_cache_broadcast"),
        "writable_files": ["kernel.py"],
        "target": "kernel.py",
        "test": "test_kernel.py",
        "holdout": None,
        "agent_mode": "patch",
        "strict_ast": True,
        "spec": (
            "Implement gqa_kv_broadcast_kernel(kv_tensor, num_query_heads=12, num_kv_heads=2) "
            "expanding [batch, num_kv_heads, seq_len, head_dim] to num_query_heads with independent "
            "head buffers and divisibility/dimension checks."
        ),
        "initial_files": {
            "kernel.py": (
                "# Defective Initial Implementation: GQA KV-Cache Broadcast\n"
                "def gqa_kv_broadcast_kernel(kv_tensor, num_query_heads=12, num_kv_heads=2):\n"
                "    # BUG: Fails to repeat-interleave KV heads and isolate head buffers\n"
                "    return kv_tensor\n"
            )
        },
    },
    "case4": {
        "id": "case4",
        "name": "Case 4: Multi-File Token Batch Scheduler (ReAct + Hidden Holdout Gate)",
        "short_name": "Case 4: Multi-File Scheduler (ReAct + Holdout)",
        "subdir": "benchmarks/case4_relay_scheduler",
        "workspace": os.path.join(BASE_DIR, "benchmarks/case4_relay_scheduler"),
        "writable_files": ["precision.py", "scheduler.py"],
        "target": "scheduler.py",
        "test": "test_scheduler.py",
        "holdout": "holdout_test_scheduler.py",
        "agent_mode": "react",
        "strict_ast": True,
        "spec": (
            "Fix weighted_score(metrics, weights) in precision.py (length check, finiteness check, "
            "math.fsum compensated summation) and dispatch_batch(tasks, max_tokens) in scheduler.py "
            "(validate max_tokens >= 0 and task tokens > 0, sort by descending score then ascending id, "
            "greedy pack within max_tokens, return shallow copies so caller dicts are not mutated)."
        ),
        "initial_files": {
            "precision.py": (
                "# Defective Initial Implementation: Weighted Score\n"
                "def weighted_score(metrics, weights):\n"
                "    # BUG: Missing length check, finiteness check, and compensated sum\n"
                "    return sum(m * w for m, w in zip(metrics, weights))\n"
            ),
            "scheduler.py": (
                "# Defective Initial Implementation: Token Batch Scheduler\n"
                "from precision import weighted_score\n\n"
                "def dispatch_batch(tasks, max_tokens):\n"
                "    # BUG: Ignores score ordering, tie-breaking, and validation\n"
                "    return list(tasks)\n"
            ),
        },
    },
}


def inject_defective_starter(case_id: str, workspace_override: str = None) -> str:
    """Writes defective initial code for `case_id` into workspace and returns target workspace path."""
    case = BENCHMARK_CASES[case_id]
    ws = os.path.realpath(workspace_override or case["workspace"])
    for rel_path, content in case["initial_files"].items():
        abs_path = os.path.join(ws, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
    return ws


def stage_benchmarks_copy(dest_root: str) -> str:
    """Copies benchmarks into a temporary directory and injects defective starter files for clean relay demos."""
    src_benchmarks = os.path.join(BASE_DIR, "benchmarks")
    dst_benchmarks = os.path.join(dest_root, "benchmarks")
    if os.path.exists(dst_benchmarks):
        shutil.rmtree(dst_benchmarks)
    shutil.copytree(
        src_benchmarks,
        dst_benchmarks,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for cid, cinfo in BENCHMARK_CASES.items():
        sub_ws = os.path.join(dest_root, cinfo["subdir"])
        inject_defective_starter(cid, workspace_override=sub_ws)
    return dest_root
