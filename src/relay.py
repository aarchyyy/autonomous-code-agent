"""
24h Carbon-Silicon Night Relay Orchestrator (DAG + Crash Resume + Morning Handover).

Manages unattended multi-task autonomous development queues:
1. Validates acyclic DAG task dependencies (`depends_on`) and executes tasks in valid
   topological order; marks downstream tasks `SKIPPED_DEPENDENCY` if an upstream
   dependency fails while letting independent DAG branches continue.
2. Enforces wall-clock (`max_total_seconds`), token (`max_total_tokens`), and
   consecutive-failure circuit-breaker budgets.
3. Supports Crash-Safe Checkpoint Resume (`resume_from_dir`): re-verifies SHA-256
   digests, visible TDD gates, and hidden holdouts of previously converged tasks
   (`RESUMED_VERIFIED`) with zero redundant LLM calls.
4. Enforces cross-task regression verification across all completed tasks and rolls
   back offending changes (`REGRESSION_ROLLED_BACK`).
5. Emits `HANDOVER_REPORT.md`, `handover_summary.json`, `audit_events.jsonl`, and
   `night_relay_changes.patch` (`git apply`-ready bundle).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from .agent import CodeIterationAgent
from .checkpoint import FileSnapshotManager, GitRelayManager
from .llm_backend import BaseLLM, create_llm_backend
from .tools import ToolBox


@dataclass
class RelayTask:
    task_id: str
    title: str
    writable_files: List[str]
    test_file: str
    holdout_test_file: Optional[str] = None
    protected_files: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    allowed_imports: List[str] = field(default_factory=list)
    spec: str = ""
    max_iterations: int = 4
    agent_mode: str = "auto"
    strict_ast: bool = False
    gate_runner: str = "auto"
    subdir: str = "."

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelayTask":
        writable = data.get("writable_files")
        if writable is None and data.get("target_file"):
            writable = [data["target_file"]]
        return cls(
            task_id=str(data["task_id"]),
            title=str(data.get("title") or data["task_id"]),
            writable_files=list(writable or []),
            test_file=str(data["test_file"]),
            holdout_test_file=data.get("holdout_test_file"),
            protected_files=list(data.get("protected_files") or []),
            depends_on=list(data.get("depends_on") or []),
            allowed_imports=list(data.get("allowed_imports") or []),
            spec=str(data.get("spec") or ""),
            max_iterations=int(data.get("max_iterations", 4)),
            agent_mode=str(data.get("agent_mode", "auto")),
            strict_ast=bool(data.get("strict_ast", False)),
            gate_runner=str(data.get("gate_runner", "auto")),
            subdir=str(data.get("subdir", ".")),
        )


@dataclass
class RelayPlan:
    project_name: str
    tasks: List[RelayTask]
    max_total_seconds: int = 3600
    max_total_tokens: int = 0  # 0 = unlimited
    max_consecutive_failures: int = 3
    use_ephemeral_copy: bool = True
    sandbox_backend: str = "process"
    docker_image: str = "python:3.12-alpine"

    def validate_dag(self) -> List[RelayTask]:
        """Validates unique task IDs, known dependencies, and acyclic topological order."""
        by_id: Dict[str, RelayTask] = {}
        for t in self.tasks:
            if t.task_id in by_id:
                raise ValueError(f"Duplicate task_id in RelayPlan: {t.task_id!r}")
            by_id[t.task_id] = t

        for t in self.tasks:
            for dep in t.depends_on:
                if dep not in by_id:
                    raise ValueError(f"Task {t.task_id!r} depends on unknown task {dep!r}")
                if dep == t.task_id:
                    raise ValueError(f"Task {t.task_id!r} cannot depend on itself")

        # Kahn's algorithm preserving declaration order tie-breaking
        in_deg: Dict[str, int] = {t.task_id: len(t.depends_on) for t in self.tasks}
        dependents: Dict[str, List[str]] = {t.task_id: [] for t in self.tasks}
        order_index = {t.task_id: i for i, t in enumerate(self.tasks)}
        for t in self.tasks:
            for dep in t.depends_on:
                dependents[dep].append(t.task_id)

        ready = [t.task_id for t in self.tasks if in_deg[t.task_id] == 0]
        ordered: List[RelayTask] = []
        while ready:
            ready.sort(key=lambda tid: order_index[tid])
            curr_id = ready.pop(0)
            ordered.append(by_id[curr_id])
            for nxt in dependents[curr_id]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    ready.append(nxt)

        if len(ordered) != len(self.tasks):
            raise ValueError("Cycle detected in RelayPlan task dependencies (`depends_on`).")
        return ordered

    @classmethod
    def from_json_file(cls, path: str) -> "RelayPlan":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            project_name=str(data.get("project_name") or "ACS-Night-Relay"),
            tasks=[RelayTask.from_dict(t) for t in data.get("tasks", [])],
            max_total_seconds=int(data.get("max_total_seconds", 3600)),
            max_total_tokens=int(data.get("max_total_tokens", 0)),
            max_consecutive_failures=int(data.get("max_consecutive_failures", 3)),
            use_ephemeral_copy=bool(data.get("use_ephemeral_copy", True)),
            sandbox_backend=str(data.get("sandbox_backend", "process")),
            docker_image=str(data.get("docker_image", "python:3.12-alpine")),
        )


class RelayOrchestrator:
    """Runs an unattended multi-task Carbon-Silicon Night Relay DAG queue."""

    def __init__(
        self,
        workspace_root: str,
        plan: RelayPlan,
        llm_backend: Optional[BaseLLM] = None,
        output_dir: Optional[str] = None,
        enable_git_branch: bool = False,
        resume_from_dir: Optional[str] = None,
    ):
        self.workspace_root = os.path.realpath(workspace_root)
        self.plan = plan
        self.ordered_tasks = self.plan.validate_dag()
        self.llm = llm_backend or create_llm_backend()
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.output_dir = os.path.realpath(
            output_dir or os.path.join(self.workspace_root, ".acs_runs", self.run_id)
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.ledger_path = os.path.join(self.output_dir, "audit_events.jsonl")
        self.enable_git_branch = enable_git_branch
        self.git = GitRelayManager(self.workspace_root) if enable_git_branch else None
        self.resume_from_dir = os.path.realpath(resume_from_dir) if resume_from_dir else None

    def _append_ledger(self, event: Dict[str, Any]) -> None:
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _task_paths(self, task: RelayTask) -> tuple[str, List[str], str, Optional[str], List[str]]:
        if task.subdir and task.subdir != ".":
            task_ws = os.path.realpath(os.path.join(self.workspace_root, task.subdir))
        else:
            task_ws = self.workspace_root
        return (
            task_ws,
            task.writable_files,
            task.test_file,
            task.holdout_test_file,
            task.protected_files,
        )

    def _build_task_toolbox(self, task: RelayTask) -> ToolBox:
        task_ws, writable, test_file, holdout, protected = self._task_paths(task)
        prot = set(protected) | {test_file}
        hidden = {holdout} if holdout else set()
        return ToolBox(
            workspace_root=task_ws,
            writable_files=writable,
            protected_files=prot,
            hidden_files=hidden,
            strict_ast=task.strict_ast,
            allowed_imports=task.allowed_imports,
            use_ephemeral_copy=self.plan.use_ephemeral_copy,
            sandbox_backend=self.plan.sandbox_backend,
            docker_image=self.plan.docker_image,
            gate_runner=task.gate_runner,
            default_test_file=test_file,
        )

    def _load_resume_succeeded_ids(self) -> Set[str]:
        if not self.resume_from_dir:
            return set()
        summary_file = os.path.join(self.resume_from_dir, "handover_summary.json")
        if not os.path.exists(summary_file):
            return set()
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                prior = json.load(f)
            return {
                str(t["task_id"])
                for t in prior.get("tasks", [])
                if t.get("success") is True
            }
        except Exception:
            return set()

    def _verify_resumed_task(self, task: RelayTask) -> Optional[Dict[str, Any]]:
        """Re-verifies a previously completed task before trusting it on resume."""
        tb = self._build_task_toolbox(task)
        gate = tb.run_tdd_gate(task.test_file)
        if not gate["passed"]:
            return None
        holdout_res = None
        if task.holdout_test_file:
            holdout_res = tb.run_tdd_gate(task.holdout_test_file, include_hidden=True)
            if not holdout_res["passed"]:
                return None
        ok_int, _ = tb.verify_integrity()
        if not ok_int:
            return None
        return {
            "gate": gate,
            "holdout_result": holdout_res,
            "hashes": tb.integrity.snapshot_hashes(),
        }

    def _run_regression_gates(
        self,
        completed_tasks: List[RelayTask],
    ) -> Dict[str, Any]:
        failures: List[Dict[str, Any]] = []
        for prev in completed_tasks:
            tb = self._build_task_toolbox(prev)
            gate = tb.run_tdd_gate(prev.test_file)
            if not gate["passed"]:
                failures.append({
                    "task_id": prev.task_id,
                    "test_file": prev.test_file,
                    "summary": gate["summary"],
                    "output": gate["output"][-600:],
                })
                continue
            if prev.holdout_test_file:
                h_gate = tb.run_tdd_gate(prev.holdout_test_file, include_hidden=True)
                if not h_gate["passed"]:
                    failures.append({
                        "task_id": prev.task_id,
                        "test_file": prev.holdout_test_file,
                        "summary": f"HOLDOUT_{h_gate['summary']}",
                        "output": h_gate["output"][-600:],
                    })
        return {
            "checked_tasks": len(completed_tasks),
            "passed": len(failures) == 0,
            "failures": failures,
        }

    def run(
        self,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        started_at = time.time()
        started_iso = datetime.now(timezone.utc).isoformat()
        relay_branch: Optional[str] = None
        if self.git and self.git.is_available:
            relay_branch = self.git.start_relay_branch(self.run_id)

        resume_candidates = self._load_resume_succeeded_ids()

        self._append_ledger({
            "event": "relay_started",
            "run_id": self.run_id,
            "project_name": self.plan.project_name,
            "backend_mode": "deterministic_mock" if self.llm.is_mock else "live_llm",
            "model": getattr(self.llm, "model", "unknown"),
            "sandbox_backend": self.plan.sandbox_backend,
            "relay_branch": relay_branch,
            "resumed_from": self.resume_from_dir,
            "total_tasks": len(self.ordered_tasks),
        })

        task_reports: List[Dict[str, Any]] = []
        completed_tasks: List[RelayTask] = []
        succeeded_ids: Set[str] = set()
        consolidated_patches: List[str] = []
        consecutive_failures = 0
        cumulative_tokens = 0
        halted_reason: Optional[str] = None

        for idx, task in enumerate(self.ordered_tasks, start=1):
            # 1. Global budget & circuit breaker checks
            elapsed_total = time.time() - started_at
            if elapsed_total >= self.plan.max_total_seconds:
                halted_reason = f"Wall-clock budget exceeded ({self.plan.max_total_seconds}s)."
                task_reports.append({
                    "task_id": task.task_id,
                    "title": task.title,
                    "depends_on": task.depends_on,
                    "status": "SKIPPED_BUDGET",
                    "success": False,
                    "iterations": 0,
                    "elapsed_seconds": 0.0,
                    "final_message": halted_reason,
                })
                continue

            if self.plan.max_total_tokens > 0 and cumulative_tokens >= self.plan.max_total_tokens:
                halted_reason = f"Token budget exceeded ({cumulative_tokens} >= {self.plan.max_total_tokens})."
                task_reports.append({
                    "task_id": task.task_id,
                    "title": task.title,
                    "depends_on": task.depends_on,
                    "status": "SKIPPED_TOKEN_BUDGET",
                    "success": False,
                    "iterations": 0,
                    "elapsed_seconds": 0.0,
                    "final_message": halted_reason,
                })
                continue

            if consecutive_failures >= self.plan.max_consecutive_failures:
                halted_reason = f"Circuit breaker tripped after {consecutive_failures} consecutive task failures."
                task_reports.append({
                    "task_id": task.task_id,
                    "title": task.title,
                    "depends_on": task.depends_on,
                    "status": "SKIPPED_CIRCUIT_BREAKER",
                    "success": False,
                    "iterations": 0,
                    "elapsed_seconds": 0.0,
                    "final_message": halted_reason,
                })
                continue

            # 2. DAG upstream dependency check (independent branches still proceed!)
            unmet_deps = [d for d in task.depends_on if d not in succeeded_ids]
            if unmet_deps:
                skip_msg = f"Skipped because upstream dependency failed or did not converge: {unmet_deps}"
                dep_report = {
                    "task_id": task.task_id,
                    "title": task.title,
                    "subdir": task.subdir,
                    "depends_on": task.depends_on,
                    "writable_files": task.writable_files,
                    "test_file": task.test_file,
                    "holdout_test_file": task.holdout_test_file,
                    "status": "SKIPPED_DEPENDENCY",
                    "success": False,
                    "iterations": 0,
                    "elapsed_seconds": 0.0,
                    "final_message": skip_msg,
                    "integrity_verified": True,
                }
                task_reports.append(dep_report)
                ev = {
                    "event": "task_finished",
                    "task_id": task.task_id,
                    "status": "SKIPPED_DEPENDENCY",
                    "success": False,
                    "iterations": 0,
                    "git_commit": None,
                }
                self._append_ledger(ev)
                if event_callback:
                    event_callback({**ev, "task_report": dep_report})
                continue

            # 3. Crash-safe Checkpoint Resume check
            if task.task_id in resume_candidates:
                t0_res = time.time()
                verified = self._verify_resumed_task(task)
                if verified is not None:
                    completed_tasks.append(task)
                    succeeded_ids.add(task.task_id)
                    res_report = {
                        "task_id": task.task_id,
                        "title": task.title,
                        "subdir": task.subdir,
                        "depends_on": task.depends_on,
                        "writable_files": task.writable_files,
                        "test_file": task.test_file,
                        "holdout_test_file": task.holdout_test_file,
                        "status": "RESUMED_VERIFIED",
                        "success": True,
                        "iterations": 0,
                        "elapsed_seconds": round(time.time() - t0_res, 3),
                        "final_message": "Re-verified from prior checkpoint (0 LLM calls).",
                        "integrity_verified": True,
                        "test_file_hashes": verified["hashes"],
                        "holdout_result": verified["holdout_result"],
                        "regression_result": None,
                        "telemetry": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "rollbacks": 0},
                        "git_commit": None,
                        "diffs": [],
                    }
                    task_reports.append(res_report)
                    done_ev = {
                        "event": "task_finished",
                        "task_id": task.task_id,
                        "status": "RESUMED_VERIFIED",
                        "success": True,
                        "iterations": 0,
                        "git_commit": None,
                    }
                    self._append_ledger(done_ev)
                    if event_callback:
                        event_callback({**done_ev, "task_report": res_report})
                    continue

            task_ws, writable, test_file, holdout, protected = self._task_paths(task)
            task_snapshot = FileSnapshotManager(task_ws, writable)
            task_snapshot.save_checkpoint("pre_task")

            start_ev = {
                "event": "task_started",
                "index": idx,
                "total": len(self.ordered_tasks),
                "task_id": task.task_id,
                "title": task.title,
                "depends_on": task.depends_on,
            }
            self._append_ledger(start_ev)
            if event_callback:
                event_callback(start_ev)

            def _forward_step(step_payload: Dict[str, Any]) -> None:
                ev = {
                    "event": "task_step",
                    "task_id": task.task_id,
                    "step": step_payload,
                }
                self._append_ledger({
                    "event": "task_step",
                    "task_id": task.task_id,
                    "iteration": step_payload.get("iteration"),
                    "status": step_payload.get("status"),
                    "rolled_back": step_payload.get("rolled_back", False),
                    "response_sha256": step_payload.get("response_sha256"),
                })
                if event_callback:
                    event_callback(ev)

            agent = CodeIterationAgent(
                target_file=writable[0] if writable else None,
                writable_files=writable,
                test_file=test_file,
                protected_files=protected,
                holdout_test_file=holdout,
                workspace_root=task_ws,
                max_iterations=task.max_iterations,
                llm_backend=self.llm,
                task_spec=task.spec or None,
                agent_mode=task.agent_mode,
                strict_ast=task.strict_ast,
                allowed_imports=task.allowed_imports,
                use_ephemeral_copy=self.plan.use_ephemeral_copy,
                sandbox_backend=self.plan.sandbox_backend,
                docker_image=self.plan.docker_image,
                gate_runner=task.gate_runner,
            )

            res = agent.run_loop(callback=_forward_step)
            tel = res.get("telemetry", {})
            cumulative_tokens += int(tel.get("prompt_tokens", 0)) + int(tel.get("completion_tokens", 0))

            regression_res: Optional[Dict[str, Any]] = None
            commit_sha: Optional[str] = None
            final_status = res["status"]
            final_success = bool(res["success"])

            if final_success:
                regression_res = self._run_regression_gates(completed_tasks)
                if not regression_res["passed"]:
                    task_snapshot.restore_checkpoint("pre_task")
                    final_success = False
                    final_status = "REGRESSION_ROLLED_BACK"
                    res["final_message"] = (
                        f"Task passed local gate, but broke {len(regression_res['failures'])} "
                        "previously completed task(s); automatically rolled back."
                    )
                else:
                    completed_tasks.append(task)
                    succeeded_ids.add(task.task_id)
                    prefix = "" if task.subdir in ("", ".") else task.subdir
                    task_patch = task_snapshot.export_unified_patch("pre_task", path_prefix=prefix)
                    if task_patch:
                        consolidated_patches.append(task_patch)
                    if self.git and self.git.is_available:
                        commit_files = [
                            os.path.relpath(os.path.join(task_ws, wf), self.workspace_root)
                            for wf in writable
                        ]
                        commit_sha = self.git.commit_milestone(
                            commit_files,
                            f"acs-relay({task.task_id}): verified {task.title}",
                        )

            if final_success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            diffs = [
                step.get("diff", "")
                for step in res.get("history", [])
                if step.get("diff")
            ]

            task_report = {
                "task_id": task.task_id,
                "title": task.title,
                "subdir": task.subdir,
                "depends_on": task.depends_on,
                "writable_files": writable,
                "test_file": test_file,
                "holdout_test_file": holdout,
                "status": final_status,
                "success": final_success,
                "iterations": res["iterations"],
                "elapsed_seconds": res["elapsed_seconds"],
                "final_message": res["final_message"],
                "integrity_verified": res["integrity_verified"],
                "test_file_hashes": res["test_file_hashes"],
                "holdout_result": res.get("holdout_result"),
                "regression_result": regression_res,
                "telemetry": tel,
                "git_commit": commit_sha,
                "diffs": diffs,
            }
            task_reports.append(task_report)

            done_ev = {
                "event": "task_finished",
                "task_id": task.task_id,
                "status": final_status,
                "success": final_success,
                "iterations": res["iterations"],
                "git_commit": commit_sha,
            }
            self._append_ledger(done_ev)
            if event_callback:
                event_callback({**done_ev, "task_report": task_report})

        total_elapsed = round(time.time() - started_at, 3)
        succeeded_count = sum(1 for r in task_reports if r["success"])
        total_llm_calls = sum(r.get("telemetry", {}).get("llm_calls", 0) for r in task_reports)
        total_prompt_tokens = sum(r.get("telemetry", {}).get("prompt_tokens", 0) for r in task_reports)
        total_completion_tokens = sum(r.get("telemetry", {}).get("completion_tokens", 0) for r in task_reports)
        total_compactions = sum(r.get("telemetry", {}).get("context_compactions", 0) for r in task_reports)
        total_reclaimed = sum(r.get("telemetry", {}).get("tokens_reclaimed", 0) for r in task_reports)

        patch_bundle_text = "\n".join(consolidated_patches)
        patch_bundle_path = os.path.join(self.output_dir, "night_relay_changes.patch")
        with open(patch_bundle_path, "w", encoding="utf-8") as f:
            f.write(patch_bundle_text)

        summary = {
            "run_id": self.run_id,
            "project_name": self.plan.project_name,
            "started_at_utc": started_iso,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": total_elapsed,
            "backend_mode": "deterministic_mock" if self.llm.is_mock else "live_llm",
            "is_mock": bool(self.llm.is_mock),
            "backend_label": getattr(self.llm, "backend_label", type(self.llm).__name__),
            "model": getattr(self.llm, "model", "unknown"),
            "sandbox_backend": self.plan.sandbox_backend,
            "relay_branch": relay_branch,
            "resumed_from": self.resume_from_dir,
            "tasks_total": len(self.ordered_tasks),
            "tasks_succeeded": succeeded_count,
            "all_succeeded": succeeded_count == len(self.ordered_tasks),
            "halted_reason": halted_reason,
            "telemetry": {
                "llm_calls": total_llm_calls,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "context_compactions": total_compactions,
                "tokens_reclaimed": total_reclaimed,
            },
            "output_dir": self.output_dir,
            "ledger_path": self.ledger_path,
            "patch_bundle_path": patch_bundle_path,
            "patch_bundle": patch_bundle_text,
            "tasks": task_reports,
        }

        summary_json_path = os.path.join(self.output_dir, "handover_summary.json")
        report_md_path = os.path.join(self.output_dir, "HANDOVER_REPORT.md")
        summary["summary_json_path"] = summary_json_path
        summary["handover_report_path"] = report_md_path

        report_md = self.render_handover_markdown(summary)
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(report_md)

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        summary["handover_markdown"] = report_md

        self._append_ledger({
            "event": "relay_finished",
            "run_id": self.run_id,
            "tasks_succeeded": succeeded_count,
            "tasks_total": len(self.ordered_tasks),
            "elapsed_seconds": total_elapsed,
            "handover_report_path": report_md_path,
            "patch_bundle_path": patch_bundle_path,
        })
        return summary

    @staticmethod
    def render_handover_markdown(summary: Dict[str, Any]) -> str:
        mode_notice = (
            "> **Verification Scope Notice (Offline Smoke Mock)**: This run used `DeterministicMockLLM`. "
            "It verifies the pipeline mechanics (DAG scheduler, sandbox, AST/C++ guard, rollback, regression gate, "
            "checkpoint resume, reporting), NOT autonomous LLM reasoning on unseen problems."
            if summary.get("is_mock")
            else f"> **Live LLM Verification**: Executed via `{summary.get('backend_label')}` (`{summary.get('model')}`)."
        )

        lines = [
            f"# Carbon-Silicon Night Relay Handover Report — {summary['project_name']}",
            "",
            mode_notice,
            "",
            "## 1. Shift Summary",
            f"- **Run ID**: `{summary['run_id']}`",
            f"- **Started (UTC)**: `{summary['started_at_utc']}`",
            f"- **Finished (UTC)**: `{summary['finished_at_utc']}`",
            f"- **Wall-Clock Duration**: `{summary['elapsed_seconds']}s`",
            f"- **Backend Mode**: `{summary['backend_mode']}` (`{summary['model']}`)",
            f"- **Sandbox Tier**: `{summary.get('sandbox_backend', 'process')}`",
            f"- **Git Relay Branch**: `{summary.get('relay_branch') or 'not enabled'}`",
            f"- **Resumed From**: `{summary.get('resumed_from') or 'fresh start'}`",
            f"- **Tasks Converged**: `{summary['tasks_succeeded']} / {summary['tasks_total']}`",
            f"- **LLM Calls / Tokens**: `{summary['telemetry']['llm_calls']}` calls | "
            f"`{summary['telemetry']['prompt_tokens']}` prompt + `{summary['telemetry']['completion_tokens']}` completion tokens "
            f"(reclaimed `{summary['telemetry'].get('tokens_reclaimed', 0)}` via `{summary['telemetry'].get('context_compactions', 0)}` compactions)",
            "",
        ]
        if summary.get("halted_reason"):
            lines.extend([
                f"> **Early Halt Notice**: {summary['halted_reason']}",
                "",
            ])

        lines.extend([
            "## 2. DAG Task Results & Integrity Verification",
            "",
            "| Task ID | Depends On | Title | Status | Iterations | Integrity | Holdout | Regression | Commit |",
            "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ])

        needs_human: List[Dict[str, Any]] = []
        for t in summary.get("tasks", []):
            deps_str = ", ".join(f"`{d}`" for d in (t.get("depends_on") or [])) or "-"
            integrity_str = "SHA256 OK" if t.get("integrity_verified", True) else "VIOLATION"
            holdout = t.get("holdout_result")
            holdout_str = "-" if holdout is None else ("PASS" if holdout.get("passed") else "FAIL")
            reg = t.get("regression_result")
            reg_str = "-" if reg is None else (f"PASS ({reg.get('checked_tasks', 0)})" if reg.get("passed") else "ROLLED BACK")
            commit_str = f"`{t['git_commit']}`" if t.get("git_commit") else "-"
            lines.append(
                f"| `{t['task_id']}` | {deps_str} | {t['title']} | `{t['status']}` | {t.get('iterations', 0)} | "
                f"{integrity_str} | {holdout_str} | {reg_str} | {commit_str} |"
            )
            if not t.get("success"):
                needs_human.append(t)

        lines.extend(["", "## 3. Morning Action Items for Human Engineers"])
        if not needs_human:
            lines.append("- All queued DAG tasks passed visible TDD gates, holdouts (where configured), and cross-task regression checks.")
            lines.append("- Apply `night_relay_changes.patch` or merge the relay branch after manual inspection.")
        else:
            for t in needs_human:
                lines.append(f"- **`{t['task_id']}` ({t['status']})**: {t.get('final_message', 'Requires human review.')}")

        lines.extend(["", "## 4. Artifact Ledger"])
        lines.append(f"- Audit JSONL: `{summary.get('ledger_path')}`")
        lines.append(f"- Machine Summary JSON: `{summary.get('summary_json_path', '')}`")
        lines.append(f"- Consolidated Patch Bundle: `{summary.get('patch_bundle_path', '')}`")
        return "\n".join(lines) + "\n"
