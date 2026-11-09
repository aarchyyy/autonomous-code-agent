#!/usr/bin/env python3
"""
CLI Demo Entrypoint for Autonomous Code Synthesis & Verification Agent.
Runs the complete multi-round self-correction loop across benchmark cases.
"""

import os
import sys
import argparse
import time

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.agent import CodeIterationAgent
from src.llm_backend import create_llm_backend

# ANSI terminal formatting
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_RED = "\033[31m"
C_BLUE = "\033[34m"
C_CYAN = "\033[36m"
C_YELLOW = "\033[33m"
C_MAGENTA = "\033[35m"

CASES = {
    "case1": {
        "name": "Case 1: SpMM CSR Operator (Unaligned Offsets & Zero-NNZ Guard)",
        "workspace": os.path.join(BASE_DIR, "benchmarks/case1_spmm_boundary"),
        "target": "kernel.py",
        "test": "test_kernel.py",
        "initial_code": """# Defective Initial Implementation: CSR Dynamic Sparse Matrix Multiplication
def spmm_csr_operator(row_ptr, col_indices, values, dense_b, num_rows, num_cols):
    result = [[0.0 for _ in range(num_cols)] for _ in range(num_rows)]
    for r in range(num_rows):
        start = row_ptr[r]
        end = row_ptr[r + 1]
        for k in range(start, end):
            c = col_indices[k]
            val = values[k]
            for j in range(num_cols):
                result[r][j] += val * dense_b[c][j]
    return result
"""
    },
    "case2": {
        "name": "Case 2: Vector Scale Operator (0-ULP Float Precision & Boundary)",
        "workspace": os.path.join(BASE_DIR, "benchmarks/case2_vector_scale"),
        "target": "kernel.py",
        "test": "test_kernel.py",
        "initial_code": """# Defective Initial Implementation: Vector Scale Operator
def vector_scale_kernel(tensor_a, tensor_b, scale=1.0):
    # BUG: Forgot scale multiplication and forgot shape check
    return [a + b for a, b in zip(tensor_a, tensor_b)]
"""
    },
    "case3": {
        "name": "Case 3: GQA KV-Cache Repeat-Interleave Stride Broadcast",
        "workspace": os.path.join(BASE_DIR, "benchmarks/case3_kv_cache_broadcast"),
        "target": "kernel.py",
        "test": "test_kernel.py",
        "initial_code": """# Defective Initial Implementation: GQA KV-Cache Broadcast
def gqa_kv_broadcast_kernel(kv_tensor, num_query_heads=12, num_kv_heads=2):
    # BUG: Fails to repeat-interleave KV heads along the head dimension
    return kv_tensor
"""
    }
}

def print_banner():
    banner = f"""
{C_CYAN}{C_BOLD}================================================================================
   🤖 AUTONOMOUS CODE SYNTHESIS & VERIFICATION AGENT (ACS-Agent)
   Night-Relay TDD Closed-Loop Autonomous Evolution System
   Aligned with: AMD GCN/CDNA • ROCm • 0-ULP Gatekeeper • ReAct Self-Correction
================================================================================{C_RESET}
"""
    print(banner)

def run_benchmark_case(case_id: str, case_info: dict):
    print(f"\n{C_MAGENTA}{C_BOLD}▶ Running {case_info['name']}...{C_RESET}")
    print(f"  Target File : {case_info['target']}")
    print(f"  Test Gate   : {case_info['test']}")
    print(f"  Workspace   : {case_info['workspace']}")

    # 1. Reset target to defective initial code
    target_path = os.path.join(case_info["workspace"], case_info["target"])
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(case_info["initial_code"])

    # 2. Instantiate Agent
    backend = create_llm_backend()
    backend_mode = "Live LLM API" if os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") else "Deterministic Mock Engine (Offline Mode)"
    print(f"  Backend     : {C_YELLOW}{backend_mode}{C_RESET}")

    agent = CodeIterationAgent(
        target_file=case_info["target"],
        test_file=case_info["test"],
        workspace_root=case_info["workspace"],
        max_iterations=4,
        llm_backend=backend
    )

    def on_step(step):
        it = step["iteration"]
        status = step["status"]
        if status == "CONVERGED_SUCCESS":
            print(f"  {C_GREEN}✔ [Iteration {it}] TDD Gatekeeper: 100% PASS! No further action required.{C_RESET}")
        else:
            print(f"  {C_RED}✖ [Iteration {it}] TDD Gatekeeper: FAILED. Intercepting exception stack...{C_RESET}")
            trace = step["test_result"].get("traceback", "").strip().splitlines()
            short_trace = trace[-1] if trace else "Assertion Failure"
            print(f"    {C_YELLOW}→ Traceback: {short_trace}{C_RESET}")
            
            print(f"    {C_BLUE}→ Agent Thought: {step.get('agent_thought')}{C_RESET}")
            diff_lines = step.get("diff", "").splitlines()
            diff_summary = [line for line in diff_lines if line.startswith("+") or line.startswith("-")]
            if diff_summary:
                print(f"    {C_CYAN}→ Patch Applied ({len(diff_summary)} diff lines){C_RESET}")

    result = agent.run_loop(callback=on_step)

    # Final summary for case
    if result["success"]:
        print(f"\n  {C_GREEN}{C_BOLD}🎉 RESULT: CASE CONVERGED! Finished in {result['iterations']} iterations ({result['elapsed_seconds']}s){C_RESET}")
    else:
        print(f"\n  {C_RED}{C_BOLD}⚠️ RESULT: FAILED to converge within budget.{C_RESET}")
    print("-" * 80)
    return result

def main():
    parser = argparse.ArgumentParser(description="Autonomous Code Agent Benchmark Runner")
    parser.add_argument("--case", choices=["case1", "case2", "case3", "all"], default="all",
                        help="Select which benchmark case to run (default: all)")
    args = parser.parse_args()

    print_banner()

    start_all = time.time()
    results = {}

    if args.case == "all":
        for cid, cinfo in CASES.items():
            results[cid] = run_benchmark_case(cid, cinfo)
    else:
        results[args.case] = run_benchmark_case(args.case, CASES[args.case])

    total_time = round(time.time() - start_all, 2)
    all_passed = all(r["success"] for r in results.values())

    print(f"\n{C_BOLD}======================= OVERALL BENCHMARK SUMMARY ======================={C_RESET}")
    for cid, r in results.items():
        status_str = f"{C_GREEN}PASS (100%){C_RESET}" if r["success"] else f"{C_RED}FAIL{C_RESET}"
        print(f"  • {CASES[cid]['name']:<65} : {status_str} in {r['iterations']} iter ({r['elapsed_seconds']}s)")
    print(f"--------------------------------------------------------------------------------")
    print(f"  Total Wall-Clock Time: {total_time}s")
    if all_passed:
        print(f"  {C_GREEN}{C_BOLD}🌟 PROOF OF WORK VERIFIED: All benchmarks converged with zero human intervention!{C_RESET}")
    print(f"{C_BOLD}================================================================================{C_RESET}\n")

if __name__ == "__main__":
    main()
