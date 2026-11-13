#!/usr/bin/env python3
"""
CLI Entrypoint for Autonomous Code Synthesis & Verification Agent (ACS-Agent).
Supports built-in benchmark verification (Cases 1-4) as well as custom user workspaces.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.agent import CodeIterationAgent
from src.benchmarks_catalog import BENCHMARK_CASES, inject_defective_starter
from src.llm_backend import HARDWARE_PRESETS, create_llm_backend, probe_llm_endpoint

C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_RED = "\033[31m"
C_BLUE = "\033[34m"
C_CYAN = "\033[36m"
C_YELLOW = "\033[33m"
C_MAGENTA = "\033[35m"


def print_banner(backend) -> None:
    is_mock = getattr(backend, "is_mock", False)
    mode_banner = (
        f"{C_YELLOW}OFFLINE PIPELINE SMOKE MOCK (verifies loop/sandbox/rollback — NOT real LLM reasoning){C_RESET}"
        if is_mock
        else f"{C_GREEN}LIVE LLM REASONING MODE ({backend.backend_label} / {backend.model}){C_RESET}"
    )
    banner = f"""
{C_CYAN}{C_BOLD}================================================================================
   AUTONOMOUS CODE SYNTHESIS & VERIFICATION AGENT (ACS-Agent)
   24h Carbon-Silicon Night-Relay TDD Verification System
   Sandbox: Ephemeral R/O Copy + PathGuard + SHA-256 Gate + Process-Group Watchdog
================================================================================{C_RESET}
  Active Backend : {mode_banner}
"""
    print(banner)


def _on_step_printer(step: dict) -> None:
    it = step.get("iteration", 1)
    status = step.get("status", "")
    rolled_back = step.get("rolled_back", False)
    if rolled_back:
        print(f"  {C_YELLOW}↺ [Iteration {it}] Regression detected — rolled back to best checkpoint.{C_RESET}")

    if status == "CONVERGED_SUCCESS":
        print(f"  {C_GREEN}✔ [Iteration {it}] TDD Gatekeeper: PASS ({step.get('sandbox_mode', 'sandbox')}).{C_RESET}")
        if step.get("holdout_result"):
            hr = step["holdout_result"]
            h_col = C_GREEN if hr["passed"] else C_RED
            print(f"    {h_col}→ Holdout Gate: {hr['summary']} ({hr['passed_count']}/{hr['tests_run']} passed){C_RESET}")
        return

    if status == "HOLDOUT_FAILED":
        print(f"  {C_RED}✖ [Iteration {it}] Visible tests passed, but HIDDEN HOLDOUT GATE FAILED!{C_RESET}")
        return

    test_res = step.get("test_result") or {}
    print(
        f"  {C_RED}✖ [Iteration {it}] TDD Gatekeeper: {test_res.get('summary', 'FAILED')} "
        f"({test_res.get('passed_count', 0)}/{test_res.get('tests_run', 0)} passed){C_RESET}"
    )
    trace = (test_res.get("traceback") or "").strip().splitlines()
    if trace:
        print(f"    {C_YELLOW}→ Traceback: {trace[-1]}{C_RESET}")
    if step.get("agent_thought"):
        print(f"    {C_BLUE}→ Agent Thought: {step['agent_thought']}{C_RESET}")
    if step.get("tool_trace"):
        tools_used = " -> ".join(f"{t['tool']}({'ok' if t.get('ok') else 'err'})" for t in step["tool_trace"])
        print(f"    {C_MAGENTA}→ Tool Trace: {tools_used}{C_RESET}")
    diff_lines = [
        line for line in (step.get("diff") or "").splitlines()
        if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---"))
    ]
    if diff_lines:
        print(f"    {C_CYAN}→ Patch Applied ({len(diff_lines)} changed lines, AST valid={step.get('ast_valid')}){C_RESET}")


def run_benchmark_case(case_id: str, backend, max_iterations: int = 4) -> dict:
    case_info = BENCHMARK_CASES[case_id]
    print(f"\n{C_MAGENTA}{C_BOLD}▶ Running {case_info['name']}...{C_RESET}")
    print(f"  Writable    : {', '.join(case_info['writable_files'])}")
    print(f"  Test Gate   : {case_info['test']}" + (f" (+ hidden holdout: {case_info['holdout']})" if case_info.get("holdout") else ""))
    print(f"  Workspace   : {case_info['workspace']}")

    inject_defective_starter(case_id)

    agent = CodeIterationAgent(
        target_file=case_info["target"],
        writable_files=case_info["writable_files"],
        test_file=case_info["test"],
        holdout_test_file=case_info.get("holdout"),
        workspace_root=case_info["workspace"],
        max_iterations=max_iterations,
        llm_backend=backend,
        task_spec=case_info.get("spec"),
        agent_mode=case_info.get("agent_mode", "auto"),
        strict_ast=case_info.get("strict_ast", False),
    )

    result = agent.run_loop(callback=_on_step_printer)

    if result["success"]:
        print(
            f"\n  {C_GREEN}{C_BOLD}✔ CONVERGED in {result['iterations']} round(s) "
            f"({result['elapsed_seconds']}s) | SHA-256 Integrity={result['integrity_verified']}{C_RESET}"
        )
    else:
        print(f"\n  {C_RED}{C_BOLD}✖ DID NOT CONVERGE ({result['status']}): {result['final_message']}{C_RESET}")
    print("-" * 80)
    return result


def run_custom_task(args, backend) -> dict:
    writable = args.target or ["kernel.py"]
    print(f"\n{C_MAGENTA}{C_BOLD}▶ Running Custom Workspace Task...{C_RESET}")
    print(f"  Workspace : {os.path.realpath(args.workspace)}")
    print(f"  Writable  : {', '.join(writable)}")
    print(f"  Test Gate : {args.test}")

    agent = CodeIterationAgent(
        target_file=writable[0],
        writable_files=writable,
        test_file=args.test,
        holdout_test_file=args.holdout,
        workspace_root=args.workspace,
        max_iterations=args.max_iter,
        llm_backend=backend,
        task_spec=args.spec,
        agent_mode="react" if (args.react or len(writable) > 1) else "auto",
        strict_ast=args.strict_sandbox,
    )
    result = agent.run_loop(callback=_on_step_printer)
    print(f"\n  Status: {result['status']} ({result['elapsed_seconds']}s) | {result['final_message']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="ACS-Agent Benchmark & Custom Workspace Runner")
    parser.add_argument("--case", choices=["case1", "case2", "case3", "case4", "all"], default="all",
                        help="Built-in benchmark case to run (default: all)")
    parser.add_argument("--mode", choices=["auto", "live", "mock"], default="auto",
                        help="LLM backend mode: auto, live, or deterministic smoke mock")
    parser.add_argument("--preset", choices=sorted(HARDWARE_PRESETS.keys()), default=None,
                        help="Hardware/provider endpoint preset (e.g. kunlun-gateway, ollama, vllm)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key (optional for local endpoints)")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--probe", action="store_true", help="Probe --base-url /models endpoint and exit")
    parser.add_argument("--max-iter", type=int, default=4, help="Max TDD iterations per task")
    # Custom workspace options
    parser.add_argument("--workspace", default=None, help="Custom project workspace directory")
    parser.add_argument("--target", action="append", default=None, help="Writable target file(s) relative to workspace")
    parser.add_argument("--test", default="test_kernel.py", help="Test file relative to workspace")
    parser.add_argument("--holdout", default=None, help="Optional hidden holdout test file relative to workspace")
    parser.add_argument("--spec", default=None, help="Optional task specification")
    parser.add_argument("--react", action="store_true", help="Force multi-tool ReAct loop")
    parser.add_argument("--strict-sandbox", action="store_true", help="Enable strict AST import/call guardrail")
    parser.add_argument("--report-json", default=None, help="Write structured run summary JSON to file")
    args = parser.parse_args()

    if args.probe:
        url = args.base_url or (HARDWARE_PRESETS[args.preset]["base_url"] if args.preset else "http://127.0.0.1:8000/v1")
        probe = probe_llm_endpoint(url, api_key=args.api_key)
        print(json.dumps(probe, ensure_ascii=False, indent=2))
        return 0 if probe["ok"] else 1

    backend = create_llm_backend(
        mode=args.mode,
        preset=args.preset,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )
    print_banner(backend)

    start_all = time.time()
    if args.workspace:
        custom_res = run_custom_task(args, backend)
        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as f:
                json.dump(custom_res, f, ensure_ascii=False, indent=2)
        return 0 if custom_res["success"] else 1

    case_ids = list(BENCHMARK_CASES.keys()) if args.case == "all" else [args.case]
    results = {cid: run_benchmark_case(cid, backend, max_iterations=args.max_iter) for cid in case_ids}

    total_time = round(time.time() - start_all, 2)
    all_passed = all(r["success"] for r in results.values())

    print(f"\n{C_BOLD}======================= OVERALL BENCHMARK SUMMARY ======================={C_RESET}")
    for cid, r in results.items():
        status_str = f"{C_GREEN}PASS{C_RESET}" if r["success"] else f"{C_RED}{r['status']}{C_RESET}"
        mode_tag = "MOCK" if r["is_mock"] else f"LIVE:{r['model']}"
        print(f"  • {BENCHMARK_CASES[cid]['short_name']:<48} : {status_str} [{mode_tag}] in {r['iterations']} iter ({r['elapsed_seconds']}s)")
    print("-" * 80)
    print(f"  Total Wall-Clock Time : {total_time}s")
    if backend.is_mock:
        print(f"  {C_YELLOW}Note: Ran with Offline Deterministic Mock. Use --mode live for real LLM verification.{C_RESET}")
    elif all_passed:
        print(f"  {C_GREEN}{C_BOLD}All benchmarks converged with Live LLM verification and SHA-256 gate integrity.{C_RESET}")
    print(f"{C_BOLD}================================================================================{C_RESET}\n")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump({"results": results, "total_seconds": total_time}, f, ensure_ascii=False, indent=2)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
