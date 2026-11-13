"""
24h Carbon-Silicon Night Relay Orchestrator & Morning Handover Reporter.

Manages unattended multi-task autonomous development queues:
1. Executes ordered engineering tasks within wall-clock and failure-circuit budgets.
2. Enforces cross-task regression verification across previously completed tasks.
3. Snapshots & rolls back failing/regressing task states; optionally commits
   verified milestones onto an isolated `acs-relay/<run_id>` Git branch.
4. Appends an immutable JSONL audit ledger and generates a Morning Handover Report
   (`HANDOVER_REPORT.md` + `handover_summary.json`) for human engineers.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

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
    spec: str = ""
    max_iterations: int = 4
    agent_mode: str = "auto"
    strict_ast: bool = False
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
            spec=str(data.get("spec") or ""),
            max_iterations=int(data.get("max_iterations", 4)),
            agent_mode=str(data.get("agent_mode", "auto")),
            strict_ast=bool(data.get("strict_ast", False)),
            subdir=str(data.get("subdir", ".")),
        )


@dataclass
class RelayPlan:
    project_name: str
    tasks: List[RelayTask]
    max_total_seconds: int = 3600
    max_consecutive_failures: int = 3
    use_ephemeral_copy: bool = True

    @classmethod
    def from_json_file(cls, path: str) -> "RelayPlan":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            project_name=str(data.get("project_name") or "ACS-Night-Relay"),
            tasks=[RelayTask.from_dict(t) for t in data.get("tasks", [])],
            max_total_seconds=int(data.get("max_total_seconds", 3600)),
            max_consecutive_failures=int(data.get("max_consecutive_failures", 3)),
            use_ephemeral_copy=bool(data.get("use_ephemeral_copy", True)),
        )


class RelayOrchestrator:
    """Runs an unattended multi-task Carbon-Silicon Night Relay queue."""

    def __init__(
        self,
        workspace_root: str,
        plan: RelayPlan,
        llm_backend: Optional[BaseLLM] = None,
        output_dir: Optional[str] = None,
        enable_git_branch: bool = False,
    ):
        self.workspace_root = os.path.realpath(workspace_root)
        self.plan = plan
        self.llm = llm_backend or create_llm_backend()
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.output_dir = os.path.realpath(
            output_dir or os.path.join(self.workspace_root, ".acs_runs", self.run_id)
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.ledger_path = os.path.join(self.output_dir, "audit_events.jsonl")
        self.enable_git_branch = enable_git_branch
        self.git = GitRelayManager(self.workspace_root) if enable_git_branch else None

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

    def _run_regression_gates(
        self,
        completed_tasks: List[RelayTask],
    ) -> Dict[str, Any]:
        failures: List[Dict[str, Any]] = []
        for prev in completed_tasks:
            task_ws, writable, test_file, holdout, protected = self._task_paths(prev)
            prot = set(protected) | {test_file}
            hidden = {holdout} if holdout else set()
            tb = ToolBox(
                workspace_root=task_ws,
                writable_files=writable,
                protected_files=prot,
                hidden_files=hidden,
                use_ephemeral_copy=self.plan.use_ephemeral_copy,
            )
            gate = tb.run_tdd_gate(test_file)
            if not gate["passed"]:
                failures.append({
                    "task_id": prev.task_id,
                    "test_file": test_file,
                    "summary": gate["summary"],
                    "output": gate["output"][-600:],
                })
                continue
            if holdout:
                h_gate = tb.run_tdd_gate(holdout, include_hidden=True)
                if not h_gate["passed"]:
                    failures.append({
                        "task_id": prev.task_id,
                        "test_file": holdout,
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

        self._append_ledger({
            "event": "relay_started",
            "run_id": self.run_id,
            "project_name": self.plan.project_name,
            "backend_mode": "deterministic_mock" if self.llm.is_mock else "live_llm",
            "model": getattr(self.llm, "model", "unknown"),
            "relay_branch": relay_branch,
            "total_tasks": len(self.plan.tasks),
        })

        task_reports: List[Dict[str, Any]] = []
        completed_tasks: List[RelayTask] = []
        consecutive_failures = 0
        halted_reason: Optional[str] = None

        for idx, task in enumerate(self.plan.tasks, start=1):
            elapsed_total = time.time() - started_at
            if elapsed_total >= self.plan.max_total_seconds:
                halted_reason = f"Wall-clock budget exceeded ({self.plan.max_total_seconds}s)."
                task_reports.append({
                    "task_id": task.task_id,
                    "title": task.title,
                    "status": "SKIPPED_BUDGET",
                    "success": False,
                    "message": halted_reason,
                })
                break

            if consecutive_failures >= self.plan.max_consecutive_failures:
                halted_reason = f"Circuit breaker tripped after {consecutive_failures} consecutive task failures."
                task_reports.append({
                    "task_id": task.task_id,
                    "title": task.title,
                    "status": "SKIPPED_CIRCUIT_BREAKER",
                    "success": False,
                    "message": halted_reason,
                })
                break

            task_ws, writable, test_file, holdout, protected = self._task_paths(task)
            task_snapshot = FileSnapshotManager(task_ws, writable)
            task_snapshot.save_checkpoint("pre_task")

            start_ev = {
                "event": "task_started",
                "index": idx,
                "total": len(self.plan.tasks),
                "task_id": task.task_id,
                "title": task.title,
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
                use_ephemeral_copy=self.plan.use_ephemeral_copy,
            )

            res = agent.run_loop(callback=_forward_step)
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
                if final_status not in ("REGRESSION_ROLLED_BACK", "HOLDOUT_FAILED"):
                    # Keep best partial progress only if it didn't regress; otherwise already restored
                    pass

            diffs = [
                step.get("diff", "")
                for step in res.get("history", [])
                if step.get("diff")
            ]

            task_report = {
                "task_id": task.task_id,
                "title": task.title,
                "subdir": task.subdir,
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
                "telemetry": res.get("telemetry", {}),
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
            "relay_branch": relay_branch,
            "tasks_total": len(self.plan.tasks),
            "tasks_succeeded": succeeded_count,
            "all_succeeded": succeeded_count == len(self.plan.tasks),
            "halted_reason": halted_reason,
            "telemetry": {
                "llm_calls": total_llm_calls,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            },
            "output_dir": self.output_dir,
            "ledger_path": self.ledger_path,
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
            "tasks_total": len(self.plan.tasks),
            "elapsed_seconds": total_elapsed,
            "handover_report_path": report_md_path,
        })
        return summary

    @staticmethod
    def render_handover_markdown(summary: Dict[str, Any]) -> str:
        mode_notice = (
            "> **Verification Scope Notice (Offline Smoke Mock)**: This run used `DeterministicMockLLM`. "
            "It verifies the pipeline mechanics (sandbox, AST guard, rollback, regression gate, reporting), "
            "NOT autonomous LLM reasoning on unseen problems."
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
            f"- **Git Relay Branch**: `{summary.get('relay_branch') or 'not enabled'}`",
            f"- **Tasks Converged**: `{summary['tasks_succeeded']} / {summary['tasks_total']}`",
            f"- **LLM Calls / Tokens**: `{summary['telemetry']['llm_calls']}` calls | "
            f"`{summary['telemetry']['prompt_tokens']}` prompt + `{summary['telemetry']['completion_tokens']}` completion tokens",
            "",
        ]
        if summary.get("halted_reason"):
            lines.extend([
                f"> **Early Halt Notice**: {summary['halted_reason']}",
                "",
            ])

        lines.extend([
            "## 2. Task Results & Integrity Verification",
            "",
            "| Task ID | Title | Status | Iterations | Integrity | Holdout | Regression | Commit |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ])

        needs_human: List[Dict[str, Any]] = []
        for t in summary.get("tasks", []):
            integrity_str = "SHA256 OK" if t.get("integrity_verified", True) else "VIOLATION"
            holdout = t.get("holdout_result")
            holdout_str = "-" if holdout is None else ("PASS" if holdout.get("passed") else "FAIL")
            reg = t.get("regression_result")
            reg_str = "-" if reg is None else (f"PASS ({reg.get('checked_tasks', 0)})" if reg.get("passed") else "ROLLED BACK")
            commit_str = f"`{t['git_commit']}`" if t.get("git_commit") else "-"
            lines.append(
                f"| `{t['task_id']}` | {t['title']} | `{t['status']}` | {t.get('iterations', 0)} | "
                f"{integrity_str} | {holdout_str} | {reg_str} | {commit_str} |"
            )
            if not t.get("success"):
                needs_human.append(t)

        lines.extend(["", "## 3. Morning Action Items for Human Engineers"])
        if not needs_human:
            lines.append("- All queued night-relay tasks passed visible TDD gates, holdouts (where configured), and cross-task regression checks.")
            lines.append("- Review generated diffs or merge the relay branch after manual inspection.")
        else:
            for t in needs_human:
                lines.append(f"- **`{t['task_id']}` ({t['status']})**: {t.get('final_message', 'Requires human review.')}")

        lines.extend(["", "## 4. Artifact Ledger"])
        lines.append(f"- Audit JSONL: `{summary.get('ledger_path')}`")
        lines.append(f"- Machine Summary JSON: `{summary.get('summary_json_path', '')}`")
        return "\n".join(lines) + "\n"
