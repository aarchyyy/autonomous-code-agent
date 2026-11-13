#!/usr/bin/env python3
"""
24h Carbon-Silicon Night Relay CLI Runner.
Executes a multi-task autonomous development plan (`relay_plan.json`) unattended,
enforces cross-task regression gates, optionally commits milestones on an isolated
Git relay branch, and emits `HANDOVER_REPORT.md` + `audit_events.jsonl`.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.benchmarks_catalog import stage_benchmarks_copy
from src.llm_backend import HARDWARE_PRESETS, create_llm_backend
from src.relay import RelayOrchestrator, RelayPlan


def main() -> int:
    parser = argparse.ArgumentParser(description="ACS-Agent 24h Night Relay Runner")
    parser.add_argument(
        "--plan",
        default=os.path.join(BASE_DIR, "benchmarks/night_relay_plan.json"),
        help="Path to relay_plan.json",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Target workspace directory (omit when running built-in demo plan to stage in temp dir)",
    )
    parser.add_argument("--wall-budget", type=int, default=None, help="Override max_total_seconds budget")
    parser.add_argument("--mode", choices=["auto", "live", "mock"], default="auto")
    parser.add_argument("--preset", choices=sorted(HARDWARE_PRESETS.keys()), default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--git-branch", action="store_true", help="Create isolated acs-relay/<ts> branch & commit milestones")
    parser.add_argument("--output-dir", default=None, help="Directory for HANDOVER_REPORT.md and audit_events.jsonl")
    args = parser.parse_args()

    plan = RelayPlan.from_json_file(args.plan)
    if args.wall_budget is not None:
        plan.max_total_seconds = args.wall_budget

    backend = create_llm_backend(
        mode=args.mode,
        preset=args.preset,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
    )

    tmp_stage = None
    if args.workspace:
        workspace_root = os.path.realpath(args.workspace)
    else:
        tmp_stage = tempfile.TemporaryDirectory(prefix="acs_relay_stage_")
        workspace_root = stage_benchmarks_copy(tmp_stage.name)

    default_out = args.output_dir or os.path.join(BASE_DIR, ".acs_runs", "latest_relay")

    def _on_event(ev: dict) -> None:
        kind = ev.get("event")
        if kind == "task_started":
            print(f"\n[{ev['index']}/{ev['total']}] ▶ Starting {ev['task_id']}: {ev['title']}")
        elif kind == "task_step":
            st = ev["step"]
            print(f"    iter={st.get('iteration')} status={st.get('status')} rollback={st.get('rolled_back', False)}")
        elif kind == "task_finished":
            mark = "✔" if ev.get("success") else "✖"
            print(f"  {mark} Finished {ev['task_id']} -> {ev['status']} ({ev['iterations']} round(s))")

    try:
        print("=" * 80)
        print(f"  ACS-Agent 24h Carbon-Silicon Night Relay | Plan: {plan.project_name}")
        print(f"  Backend: {backend.backend_label} ({backend.model})")
        print(f"  Workspace: {workspace_root}")
        print("=" * 80)

        orchestrator = RelayOrchestrator(
            workspace_root=workspace_root,
            plan=plan,
            llm_backend=backend,
            output_dir=default_out,
            enable_git_branch=args.git_branch,
        )
        summary = orchestrator.run(event_callback=_on_event)

        print("\n" + "=" * 80)
        print(f"  NIGHT RELAY COMPLETE: {summary['tasks_succeeded']} / {summary['tasks_total']} succeeded in {summary['elapsed_seconds']}s")
        print(f"  Handover Report : {summary['handover_report_path']}")
        print(f"  Audit Ledger    : {summary['ledger_path']}")
        print("=" * 80)
        return 0 if summary["all_succeeded"] else 1
    finally:
        if tmp_stage is not None:
            tmp_stage.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
