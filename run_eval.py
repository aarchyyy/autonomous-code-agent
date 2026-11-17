#!/usr/bin/env python3
"""
CLI Runner for ACS-Agent Adversarial Red-Team & Unseen Generalization Evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.evaluator import (
    render_eval_markdown,
    run_fault_injection_http_sim,
    run_redteam_suite,
    run_unseen_suite,
)
from src.llm_backend import HARDWARE_PRESETS, create_llm_backend


def main() -> int:
    parser = argparse.ArgumentParser(description="ACS-Agent Red-Team & Unseen Generalization Evaluator")
    parser.add_argument(
        "--mode",
        choices=["redteam", "fault-sim", "live", "all-offline"],
        default="all-offline",
        help="Evaluation mode: redteam (8 security probes), fault-sim (local HTTP fault-injected unseen eval), live (real LLM unseen eval), or all-offline (redteam + fault-sim)",
    )
    parser.add_argument("--preset", choices=sorted(HARDWARE_PRESETS.keys()), default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-iter", type=int, default=4)
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, ".acs_runs", "latest_eval"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    redteam_res = None
    unseen_res = None

    if args.mode in ("redteam", "all-offline"):
        print("▶ Running 8-Scenario Adversarial Security & Resilience Red-Team Suite...")
        redteam_res = run_redteam_suite()
        for p in redteam_res["probes"]:
            mark = "✔" if p["defended"] else "✖"
            print(f"  {mark} {p['probe_id']:<38} : {p['detail']}")
        print(f"  Red-Team Result: {redteam_res['probes_defended']}/{redteam_res['probes_total']} defended ({redteam_res['elapsed_seconds']}s)\n")

    if args.mode in ("fault-sim", "all-offline"):
        print("▶ Running Multi-Fault Local HTTP Agentic Stress Simulation on Unseen Problem Bank (U1..U4)...")
        unseen_res = run_fault_injection_http_sim()
        for r in unseen_res["results"]:
            mark = "✔" if r["success"] else "✖"
            comp = r["telemetry"].get("context_compactions", 0)
            print(f"  {mark} {r['problem_id']:<28} : {r['status']} in {r['iterations']} iter ({r['elapsed_seconds']}s, compactions={comp})")
        print(f"  Fault-Sim Result: {unseen_res['problems_passed']}/{unseen_res['problems_total']} converged ({unseen_res['elapsed_seconds']}s)\n")

    if args.mode == "live":
        backend = create_llm_backend(
            mode="live",
            preset=args.preset,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
        )
        print(f"▶ Running Live LLM Generalization Eval ({backend.backend_label} / {backend.model}) on Unseen Bank...")
        unseen_res = run_unseen_suite(backend, max_iterations=args.max_iter)
        for r in unseen_res["results"]:
            mark = "✔" if r["success"] else "✖"
            print(f"  {mark} {r['problem_id']:<28} : {r['status']} in {r['iterations']} iter ({r['elapsed_seconds']}s)")
        print(f"  Live Eval Result: {unseen_res['problems_passed']}/{unseen_res['problems_total']} converged ({unseen_res['elapsed_seconds']}s)\n")

    md = render_eval_markdown(redteam_res, unseen_res)
    md_path = os.path.join(args.output_dir, "EVAL_REPORT.md")
    json_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"redteam": redteam_res, "unseen": unseen_res}, f, ensure_ascii=False, indent=2)

    print(f"  Report saved to : {md_path}")
    ok_rt = True if redteam_res is None else (redteam_res["probes_defended"] == redteam_res["probes_total"])
    ok_un = True if unseen_res is None else (unseen_res["problems_passed"] == unseen_res["problems_total"])
    return 0 if (ok_rt and ok_un) else 1


if __name__ == "__main__":
    raise SystemExit(main())
