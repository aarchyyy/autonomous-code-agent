#!/usr/bin/env python3
"""
24h Carbon-Silicon Night Relay CLI Runner (DAG + Crash Resume + Consolidated Patch).
Executes a multi-task autonomous development plan (`relay_plan.json`) unattended,
validates DAG dependencies, supports crash-safe `--resume`, enforces cross-task
regression gates, and emits `HANDOVER_REPORT.md` + `night_relay_changes.patch`.
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
    parser.add_argument("--token-budget", type=int, default=None, help="Optional max_total_tokens cap")
    parser.add_argument("--sandbox", choices=["process", "docker", "auto"], default=None)
    parser.add_argument("--docker-image", default=None)
    parser.add_argument("--resume", default=None, help="Previous run output dir to resume verified tasks from")
    parser.add_argument("--mode", choices=["auto", "live", "mock"], default="auto")
    parser.add_argument("--preset", choices=sorted(HARDWARE_PRESETS.keys()), default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--git-branch", action="store_true", help="Create isolated acs-relay/<ts> branch & commit milestones")
    parser.add_argument("--output-dir", default=None, help="Directory for HANDOVER_REPORT.md and artifacts")
    args = parser.parse_args()

    plan = RelayPlan.from_json_file(args.plan)
    if args.wall_budget is not None:
        plan.max_total_seconds = args.wall_budget
    if args.token_budget is not None:
        plan.max_total_tokens = args.token_budget
    if args.sandbox is not None:
        plan.sandbox_backend = args.sandbox
    if args.docker_image is not None:
        plan.docker_image = args.docker_image

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
            deps = ev.get("depends_on") or []
            dep_str = f" (depends_on={deps})" if deps else ""
            print(f"\n[{ev['index']}/{ev['total']}] ▶ Starting {ev['task_id']}: {ev['title']}{dep_str}")
        elif kind == "task_step":
            st = ev["step"]
            print(f"    iter={st.get('iteration')} status={st.get('status')} rollback={st.get('rolled_back', False)}")
        elif kind == "task_finished":
            mark = "✔" if ev.get("success") else "✖"
            print(f"  {mark} Finished {ev['task_id']} -> {ev['status']} ({ev['iterations']} round(s))")

    try:
        print("=" * 80)
        print(f"  ACS-Agent 24h Carbon-Silicon Night Relay | Plan: {plan.project_name}")
        print(f"  Backend : {backend.backend_label} ({backend.model})")
        print(f"  Sandbox : {plan.sandbox_backend} | Workspace: {workspace_root}")
        if args.resume:
            print(f"  Resume  : {os.path.realpath(args.resume)}")
        print("=" * 80)

        orchestrator = RelayOrchestrator(
            workspace_root=workspace_root,
            plan=plan,
            llm_backend=backend,
            output_dir=default_out,
            enable_git_branch=args.git_branch,
            resume_from_dir=args.resume,
        )
        summary = orchestrator.run(event_callback=_on_event)

        print("\n" + "=" * 80)
        print(f"  NIGHT RELAY COMPLETE: {summary['tasks_succeeded']} / {summary['tasks_total']} succeeded in {summary['elapsed_seconds']}s")
        print(f"  Handover Report : {summary['handover_report_path']}")
        print(f"  Patch Bundle    : {summary['patch_bundle_path']}")
        print(f"  Audit Ledger    : {summary['ledger_path']}")
        print("=" * 80)
        return 0 if summary["all_succeeded"] else 1
    finally:
        if tmp_stage is not None:
            tmp_stage.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
