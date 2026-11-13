"""
Core Agent Module: Orchestrates ReAct reasoning, multi-turn TDD gatekeeper feedback,
AST/sandbox validation, snapshot checkpointing & regression rollback, anti-stagnation
detection, and optional hidden holdout verification.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .checkpoint import FileSnapshotManager
from .llm_backend import AgentTurn, BaseLLM, create_llm_backend
from .prompts import (
    REACT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_error_feedback_prompt,
    build_react_task_prompt,
)
from .sandbox import IntegrityViolationError, PathSecurityError
from .tools import TOOL_DEFINITIONS, ToolBox


def _score_test_result(test_result: Dict[str, Any]) -> Tuple[int, int, int]:
    """Higher tuple is better: (passed_flag, passed_count, -failed_count)."""
    passed_flag = 1 if test_result.get("passed") else 0
    passed_cnt = int(test_result.get("passed_count") or 0)
    failed_cnt = int(test_result.get("failures") or 0) + int(test_result.get("errors") or 0)
    return (passed_flag, passed_cnt, -failed_cnt)


class CodeIterationAgent:
    def __init__(
        self,
        target_file: Optional[str] = None,
        test_file: str = "test_kernel.py",
        workspace_root: str = ".",
        max_iterations: int = 4,
        llm_backend: Optional[BaseLLM] = None,
        writable_files: Optional[List[str]] = None,
        protected_files: Optional[List[str]] = None,
        holdout_test_file: Optional[str] = None,
        task_spec: Optional[str] = None,
        agent_mode: str = "auto",
        strict_ast: bool = False,
        use_ephemeral_copy: bool = True,
        rollback_on_regression: bool = True,
        max_tool_steps: int = 18,
        test_timeout: int = 15,
    ):
        if writable_files is not None:
            resolved_writable = list(writable_files)
        elif target_file is not None:
            resolved_writable = [target_file]
        else:
            raise ValueError("Either target_file or writable_files must be provided.")

        self.target_file = target_file or resolved_writable[0]
        self.writable_files = resolved_writable
        self.test_file = test_file
        self.holdout_test_file = holdout_test_file
        self.workspace_root = workspace_root
        self.max_iterations = max(1, int(max_iterations))
        self.max_tool_steps = max(4, int(max_tool_steps))
        self.task_spec = task_spec
        self.agent_mode = agent_mode
        self.rollback_on_regression = rollback_on_regression
        self.test_timeout = test_timeout

        protected = set(protected_files or ())
        protected.add(test_file)
        hidden = {holdout_test_file} if holdout_test_file else set()

        self.tools = ToolBox(
            workspace_root=workspace_root,
            writable_files=self.writable_files,
            protected_files=protected,
            hidden_files=hidden,
            strict_ast=strict_ast,
            use_ephemeral_copy=use_ephemeral_copy,
            default_test_file=test_file,
        )
        self.llm: BaseLLM = llm_backend or create_llm_backend()
        self.snapshots = FileSnapshotManager(
            workspace_root=self.tools.workspace_root,
            tracked_files=self.writable_files,
        )
        self.history: List[Dict[str, Any]] = []
        self._telemetry = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_latency_ms": 0.0,
            "rollbacks": 0,
        }

    def _record_turn_telemetry(self, turn: AgentTurn) -> None:
        self._telemetry["llm_calls"] += 1
        self._telemetry["prompt_tokens"] += int(turn.prompt_tokens or 0)
        self._telemetry["completion_tokens"] += int(turn.completion_tokens or 0)
        self._telemetry["llm_latency_ms"] = round(
            self._telemetry["llm_latency_ms"] + float(turn.latency_ms or 0.0), 2
        )

    def _run_holdout_if_configured(self) -> Optional[Dict[str, Any]]:
        if not self.holdout_test_file:
            return None
        return self.tools.run_tdd_gate(
            self.holdout_test_file,
            timeout=self.test_timeout,
            include_hidden=True,
        )

    def _build_final_result(
        self,
        success: bool,
        iterations: int,
        start_time: float,
        final_message: str,
        holdout_result: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        elapsed = round(time.time() - start_time, 3)
        ok_integrity, violations = self.tools.verify_integrity()
        return {
            "success": success,
            "status": status or ("CONVERGED_SUCCESS" if success else "FAILED"),
            "iterations": iterations,
            "elapsed_seconds": elapsed,
            "history": self.history,
            "final_message": final_message,
            "is_mock": bool(getattr(self.llm, "is_mock", False)),
            "backend_mode": "deterministic_mock" if getattr(self.llm, "is_mock", False) else "live_llm",
            "backend_label": getattr(self.llm, "backend_label", type(self.llm).__name__),
            "model": getattr(self.llm, "model", "unknown"),
            "integrity_verified": ok_integrity,
            "integrity_violations": violations,
            "test_file_hashes": self.tools.integrity.snapshot_hashes(),
            "holdout_result": holdout_result,
            "telemetry": dict(self._telemetry),
        }

    def _emit(self, callback: Optional[Callable[[Dict[str, Any]], None]], payload: Dict[str, Any]) -> None:
        if callback:
            callback(payload)

    def execute_step(self, iteration: int) -> Dict[str, Any]:
        """Executes a single patch-mode iteration step (kept compatible with single-step callers)."""
        step_log: Dict[str, Any] = {
            "iteration": iteration,
            "timestamp": time.time(),
            "target_file": self.target_file,
            "test_file": self.test_file,
            "is_mock": bool(getattr(self.llm, "is_mock", False)),
        }

        test_result = self.tools.run_tdd_gate(self.test_file, timeout=self.test_timeout)
        step_log["test_result"] = test_result
        step_log["sandbox_mode"] = test_result.get("sandbox_mode")

        if test_result["passed"]:
            step_log["status"] = "CONVERGED_SUCCESS"
            step_log["message"] = f"TDD Gate passed on iteration {iteration}."
            return step_log

        current_source = self.tools.read_source(self.target_file)
        step_log["before_code"] = current_source

        prompt = build_error_feedback_prompt(
            target_file=self.target_file,
            current_source=current_source,
            test_file=self.test_file,
            test_output=test_result["traceback"] or test_result["output"],
            iteration=iteration,
            max_iterations=self.max_iterations,
            task_spec=self.task_spec,
        )

        turn = self.llm.chat_step(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tools=None,
        )
        self._record_turn_telemetry(turn)
        thought = turn.thought
        patched_code = turn.code or turn.raw_text.strip()

        step_log["agent_thought"] = thought
        step_log["patched_code"] = patched_code
        step_log["response_sha256"] = turn.response_sha256
        step_log["latency_ms"] = turn.latency_ms

        valid_syntax, syntax_msg = self.tools.validate_syntax(patched_code)
        step_log["ast_valid"] = valid_syntax
        step_log["ast_message"] = syntax_msg

        if not valid_syntax:
            step_log["status"] = "SYNTAX_ERROR"
            step_log["message"] = f"AST rejection: {syntax_msg}"
            return step_log

        if patched_code.strip() == current_source.strip():
            step_log["status"] = "NO_OP_PATCH"
            step_log["message"] = "Model produced an identical patch (no changes)."
            return step_log

        diff_str = self.tools.compute_diff(current_source, patched_code, filename=self.target_file)
        step_log["diff"] = diff_str

        self.tools.write_source(self.target_file, patched_code)
        step_log["status"] = "PATCH_APPLIED"
        step_log["message"] = "Patch safely applied to target file."
        return step_log

    def _run_patch_loop(
        self,
        start_time: float,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        previous_attempts: List[str] = []
        seen_state_errors: Dict[Tuple[str, str], int] = {}
        last_error_sig: Optional[str] = None
        consecutive_same_error = 0

        for it in range(1, self.max_iterations + 1):
            try:
                test_result = self.tools.run_tdd_gate(self.test_file, timeout=self.test_timeout)
            except IntegrityViolationError as exc:
                self.snapshots.restore_best()
                return self._build_final_result(
                    success=False,
                    iterations=it,
                    start_time=start_time,
                    final_message=f"Integrity violation detected: {exc}",
                    status="INTEGRITY_VIOLATION",
                )

            score = _score_test_result(test_result)
            rolled_back = False

            if it == 1:
                self.snapshots.save_checkpoint("initial", score=score)
            else:
                best = self.snapshots.best_checkpoint
                if self.rollback_on_regression and best is not None and score < best.score:
                    self.snapshots.restore_best()
                    self._telemetry["rollbacks"] += 1
                    rolled_back = True
                else:
                    self.snapshots.update_best_if_better(f"iter_{it}", score=score)

            if test_result["passed"]:
                holdout_res = self._run_holdout_if_configured()
                step_log = {
                    "iteration": it,
                    "timestamp": time.time(),
                    "target_file": self.target_file,
                    "test_file": self.test_file,
                    "test_result": test_result,
                    "holdout_result": holdout_res,
                    "sandbox_mode": test_result.get("sandbox_mode"),
                    "is_mock": bool(getattr(self.llm, "is_mock", False)),
                    "rolled_back": rolled_back,
                }
                if holdout_res is not None and not holdout_res["passed"]:
                    step_log["status"] = "HOLDOUT_FAILED"
                    step_log["message"] = "Visible TDD gate passed, but hidden holdout tests failed."
                    self.history.append(step_log)
                    self._emit(callback, step_log)
                    return self._build_final_result(
                        success=False,
                        iterations=it,
                        start_time=start_time,
                        final_message=step_log["message"],
                        holdout_result=holdout_res,
                        status="HOLDOUT_FAILED",
                    )

                step_log["status"] = "CONVERGED_SUCCESS"
                step_log["message"] = f"TDD Gate passed on iteration {it}."
                self.history.append(step_log)
                self._emit(callback, step_log)
                return self._build_final_result(
                    success=True,
                    iterations=it,
                    start_time=start_time,
                    final_message=f"Converged in {it} iteration(s) ({round(time.time() - start_time, 2)}s).",
                    holdout_result=holdout_res,
                    status="CONVERGED_SUCCESS",
                )

            # Anti-stagnation tracking
            err_sig = test_result.get("error_signature", "")
            state_hash = self.snapshots.current_hash()
            pair = (state_hash, err_sig)
            seen_state_errors[pair] = seen_state_errors.get(pair, 0) + 1
            if err_sig == last_error_sig:
                consecutive_same_error += 1
            else:
                consecutive_same_error = 1
                last_error_sig = err_sig

            stagnation_warning: Optional[str] = None
            if rolled_back:
                stagnation_warning = (
                    "Your previous patch reduced the number of passing tests and was rolled back. "
                    "Analyze why the previous change regressed and take a different approach."
                )
            elif seen_state_errors[pair] >= 2 or consecutive_same_error >= 2:
                stagnation_warning = (
                    f"Same failure signature ({err_sig}) occurred {consecutive_same_error} times. "
                    "Your previous hypothesis did not resolve the underlying cause; reconsider invariants and edge cases."
                )

            if seen_state_errors[pair] >= 3:
                step_log = {
                    "iteration": it,
                    "timestamp": time.time(),
                    "target_file": self.target_file,
                    "test_file": self.test_file,
                    "test_result": test_result,
                    "status": "STALLED",
                    "message": "Halted early due to repeated identical state and error signature.",
                    "rolled_back": rolled_back,
                }
                self.history.append(step_log)
                self._emit(callback, step_log)
                return self._build_final_result(
                    success=False,
                    iterations=it,
                    start_time=start_time,
                    final_message="Agent stalled on repeated identical failure state; human review advised.",
                    status="STALLED",
                )

            current_source = self.tools.read_source(self.target_file)
            prompt = build_error_feedback_prompt(
                target_file=self.target_file,
                current_source=current_source,
                test_file=self.test_file,
                test_output=test_result["traceback"] or test_result["output"],
                iteration=it,
                max_iterations=self.max_iterations,
                task_spec=self.task_spec,
                stagnation_warning=stagnation_warning,
                previous_attempts_summary=previous_attempts,
            )

            turn = self.llm.chat_step(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tools=None,
            )
            self._record_turn_telemetry(turn)
            thought = turn.thought
            patched_code = turn.code or turn.raw_text.strip()

            valid_syntax, syntax_msg = self.tools.validate_syntax(patched_code)
            step_log: Dict[str, Any] = {
                "iteration": it,
                "timestamp": time.time(),
                "target_file": self.target_file,
                "test_file": self.test_file,
                "test_result": test_result,
                "sandbox_mode": test_result.get("sandbox_mode"),
                "before_code": current_source,
                "agent_thought": thought,
                "patched_code": patched_code,
                "ast_valid": valid_syntax,
                "ast_message": syntax_msg,
                "response_sha256": turn.response_sha256,
                "latency_ms": turn.latency_ms,
                "is_mock": bool(getattr(self.llm, "is_mock", False)),
                "rolled_back": rolled_back,
            }

            if not valid_syntax:
                step_log["status"] = "SYNTAX_ERROR"
                step_log["message"] = f"AST rejection: {syntax_msg}"
                previous_attempts.append(f"Iter {it}: Rejected by AST guard ({syntax_msg.splitlines()[0]}).")
            elif patched_code.strip() == current_source.strip():
                step_log["status"] = "NO_OP_PATCH"
                step_log["message"] = "Patch produced no changes to target file."
                previous_attempts.append(f"Iter {it}: No-op patch (code unchanged).")
            else:
                diff_str = self.tools.compute_diff(current_source, patched_code, filename=self.target_file)
                self.tools.write_source(self.target_file, patched_code)
                step_log["diff"] = diff_str
                step_log["status"] = "PATCH_APPLIED"
                step_log["message"] = "Patch safely applied to target file."
                previous_attempts.append(
                    f"Iter {it}: Applied patch ({len(diff_str.splitlines())} diff lines) after {test_result['summary']}."
                )

            self.history.append(step_log)
            self._emit(callback, step_log)

            if step_log["status"] == "NO_OP_PATCH" and getattr(self.llm, "is_mock", False):
                return self._build_final_result(
                    success=False,
                    iterations=it,
                    start_time=start_time,
                    final_message="Offline smoke mock cannot synthesize code for unseen tasks.",
                    status="MOCK_UNSUPPORTED_TASK",
                )

        # Post-final gate check
        final_gate = self.tools.run_tdd_gate(self.test_file, timeout=self.test_timeout)
        final_score = _score_test_result(final_gate)
        if final_gate["passed"]:
            holdout_res = self._run_holdout_if_configured()
            if holdout_res is not None and not holdout_res["passed"]:
                return self._build_final_result(
                    success=False,
                    iterations=self.max_iterations,
                    start_time=start_time,
                    final_message="Visible TDD gate passed after final patch, but hidden holdout tests failed.",
                    holdout_result=holdout_res,
                    status="HOLDOUT_FAILED",
                )
            return self._build_final_result(
                success=True,
                iterations=self.max_iterations,
                start_time=start_time,
                final_message=f"All tests passed after final patch verification ({round(time.time() - start_time, 2)}s).",
                holdout_result=holdout_res,
                status="CONVERGED_SUCCESS",
            )

        best = self.snapshots.best_checkpoint
        if self.rollback_on_regression and best is not None and final_score < best.score:
            self.snapshots.restore_best()
            self._telemetry["rollbacks"] += 1

        return self._build_final_result(
            success=False,
            iterations=self.max_iterations,
            start_time=start_time,
            final_message=f"Reached max iterations ({self.max_iterations}) without full convergence.",
            status="MAX_ITERATIONS_REACHED",
        )

    def _run_react_loop(
        self,
        start_time: float,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        baseline = self.tools.run_tdd_gate(self.test_file, timeout=self.test_timeout)
        best_score = _score_test_result(baseline)
        self.snapshots.save_checkpoint("initial", score=best_score)

        baseline_log = {
            "iteration": 1,
            "timestamp": time.time(),
            "target_file": ", ".join(self.writable_files),
            "test_file": self.test_file,
            "test_result": baseline,
            "sandbox_mode": baseline.get("sandbox_mode"),
            "is_mock": bool(getattr(self.llm, "is_mock", False)),
            "tool_trace": [],
        }

        if baseline["passed"]:
            holdout_res = self._run_holdout_if_configured()
            if holdout_res is not None and not holdout_res["passed"]:
                baseline_log["status"] = "HOLDOUT_FAILED"
                baseline_log["message"] = "Baseline visible tests passed, but hidden holdout failed."
                self.history.append(baseline_log)
                self._emit(callback, baseline_log)
                return self._build_final_result(
                    success=False,
                    iterations=1,
                    start_time=start_time,
                    final_message=baseline_log["message"],
                    holdout_result=holdout_res,
                    status="HOLDOUT_FAILED",
                )
            baseline_log["status"] = "CONVERGED_SUCCESS"
            baseline_log["message"] = "Baseline TDD gate already passed."
            self.history.append(baseline_log)
            self._emit(callback, baseline_log)
            return self._build_final_result(
                success=True,
                iterations=1,
                start_time=start_time,
                final_message="All tests already passed.",
                holdout_result=holdout_res,
                status="CONVERGED_SUCCESS",
            )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_react_task_prompt(
                    writable_files=self.writable_files,
                    test_file=self.test_file,
                    initial_test_output=baseline["traceback"] or baseline["output"],
                    task_spec=self.task_spec,
                ),
            },
        ]

        test_round = 1
        pending_diffs: List[str] = []
        pending_thoughts: List[str] = []
        pending_tools: List[Dict[str, Any]] = []
        last_test_result = baseline
        empty_turns = 0

        for step_idx in range(1, self.max_tool_steps + 1):
            turn = self.llm.chat_step(messages=messages, tools=TOOL_DEFINITIONS)
            self._record_turn_telemetry(turn)
            if turn.thought:
                pending_thoughts.append(turn.thought)

            if turn.tool_calls:
                empty_turns = 0
                use_native = bool(getattr(self.llm, "supports_tools", False) and getattr(self.llm, "enable_native_tools", True))
                if use_native:
                    messages.append({
                        "role": "assistant",
                        "content": turn.raw_text or "",
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                                },
                            }
                            for tc in turn.tool_calls
                        ],
                    })
                else:
                    messages.append({"role": "assistant", "content": turn.raw_text or turn.thought})

                for tc in turn.tool_calls:
                    name = tc["name"]
                    args = tc.get("arguments") or {}
                    res = self.tools.dispatch_tool(name, args)
                    pending_tools.append({"tool": name, "arguments": args, "ok": res.get("ok")})

                    if res.get("ok") and res.get("diff"):
                        pending_diffs.append(res["diff"])

                    if name == "run_tests" and res.get("ok"):
                        test_result = res["test_result"]
                        score = _score_test_result(test_result)
                        rolled_back = False
                        if self.rollback_on_regression and score < best_score:
                            self.snapshots.restore_best()
                            self._telemetry["rollbacks"] += 1
                            rolled_back = True
                        else:
                            if score >= best_score:
                                best_score = score
                                self.snapshots.save_checkpoint(f"react_{test_round}", score=score)

                        if test_result["passed"]:
                            if pending_diffs or pending_thoughts or pending_tools:
                                patch_entry = {
                                    "iteration": test_round,
                                    "timestamp": time.time(),
                                    "target_file": ", ".join(self.writable_files),
                                    "test_file": self.test_file,
                                    "test_result": last_test_result,
                                    "sandbox_mode": test_result.get("sandbox_mode"),
                                    "agent_thought": "\n".join(pending_thoughts),
                                    "diff": "\n".join(pending_diffs),
                                    "tool_trace": list(pending_tools),
                                    "ast_valid": True,
                                    "ast_message": "Validated per-write AST checks.",
                                    "is_mock": bool(getattr(self.llm, "is_mock", False)),
                                    "rolled_back": False,
                                    "status": "PATCH_APPLIED",
                                    "message": "ReAct tool edits applied.",
                                }
                                self.history.append(patch_entry)
                                self._emit(callback, patch_entry)

                            holdout_res = self._run_holdout_if_configured()
                            converged_entry = {
                                "iteration": test_round + (1 if (pending_diffs or pending_thoughts or pending_tools) else 0),
                                "timestamp": time.time(),
                                "target_file": ", ".join(self.writable_files),
                                "test_file": self.test_file,
                                "test_result": test_result,
                                "holdout_result": holdout_res,
                                "sandbox_mode": test_result.get("sandbox_mode"),
                                "is_mock": bool(getattr(self.llm, "is_mock", False)),
                                "rolled_back": False,
                                "status": "HOLDOUT_FAILED" if (holdout_res is not None and not holdout_res["passed"]) else "CONVERGED_SUCCESS",
                                "message": "Visible TDD gate passed, but hidden holdout tests failed." if (holdout_res is not None and not holdout_res["passed"]) else f"TDD Gate passed on round {test_round}.",
                            }
                            self.history.append(converged_entry)
                            self._emit(callback, converged_entry)

                            if holdout_res is not None and not holdout_res["passed"]:
                                return self._build_final_result(
                                    success=False,
                                    iterations=converged_entry["iteration"],
                                    start_time=start_time,
                                    final_message="Visible TDD gate passed, but hidden holdout tests failed.",
                                    holdout_result=holdout_res,
                                    status="HOLDOUT_FAILED",
                                )
                            return self._build_final_result(
                                success=True,
                                iterations=converged_entry["iteration"],
                                start_time=start_time,
                                final_message=f"Converged via ReAct tools in {test_round} test round(s) ({step_idx} tool steps).",
                                holdout_result=holdout_res,
                                status="CONVERGED_SUCCESS",
                            )

                        step_entry = {
                            "iteration": test_round,
                            "timestamp": time.time(),
                            "target_file": ", ".join(self.writable_files),
                            "test_file": self.test_file,
                            "test_result": last_test_result,
                            "post_test_result": test_result,
                            "sandbox_mode": test_result.get("sandbox_mode"),
                            "agent_thought": "\n".join(pending_thoughts),
                            "diff": "\n".join(pending_diffs),
                            "tool_trace": list(pending_tools),
                            "ast_valid": True,
                            "ast_message": "Validated per-write AST checks.",
                            "is_mock": bool(getattr(self.llm, "is_mock", False)),
                            "rolled_back": rolled_back,
                            "status": "PATCH_APPLIED",
                            "message": "ReAct cycle completed; tests still failing.",
                        }
                        self.history.append(step_entry)
                        self._emit(callback, step_entry)

                        last_test_result = test_result
                        test_round += 1
                        pending_diffs.clear()
                        pending_thoughts.clear()
                        pending_tools.clear()

                    formatted = self.tools.format_tool_result_for_llm(res)
                    if use_native:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": formatted,
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"[TOOL_RESULT name={name}]\n{formatted}",
                        })

                if test_round > self.max_iterations:
                    break
            elif turn.code and len(self.writable_files) == 1:
                empty_turns = 0
                res = self.tools.dispatch_tool("write_file", {
                    "filepath": self.writable_files[0],
                    "content": turn.code,
                })
                pending_tools.append({"tool": "write_file", "arguments": {"filepath": self.writable_files[0]}, "ok": res.get("ok")})
                if res.get("diff"):
                    pending_diffs.append(res["diff"])
                messages.append({"role": "assistant", "content": turn.raw_text or turn.thought})
                messages.append({
                    "role": "user",
                    "content": f"Applied write_file result: {json.dumps(res, ensure_ascii=False)}. Now call `run_tests`.",
                })
            else:
                empty_turns += 1
                messages.append({"role": "assistant", "content": turn.raw_text or ""})
                messages.append({
                    "role": "user",
                    "content": "Please invoke one of the available tools (`read_file`, `patch_file`, `write_file`, `run_tests`).",
                })
                if empty_turns >= 2:
                    break

        # Final verification if edits were made after last run_tests
        final_gate = self.tools.run_tdd_gate(self.test_file, timeout=self.test_timeout)
        final_score = _score_test_result(final_gate)
        if final_gate["passed"]:
            holdout_res = self._run_holdout_if_configured()
            if holdout_res is not None and not holdout_res["passed"]:
                return self._build_final_result(
                    success=False,
                    iterations=test_round,
                    start_time=start_time,
                    final_message="Visible tests passed, but hidden holdout failed.",
                    holdout_result=holdout_res,
                    status="HOLDOUT_FAILED",
                )
            return self._build_final_result(
                success=True,
                iterations=test_round,
                start_time=start_time,
                final_message="All tests passed on final verification.",
                holdout_result=holdout_res,
                status="CONVERGED_SUCCESS",
            )

        if self.rollback_on_regression and final_score < best_score:
            self.snapshots.restore_best()
            self._telemetry["rollbacks"] += 1

        if not self.history:
            baseline_log["agent_thought"] = "\n".join(pending_thoughts)
            baseline_log["diff"] = "\n".join(pending_diffs)
            baseline_log["tool_trace"] = pending_tools
            baseline_log["status"] = "MAX_ITERATIONS_REACHED"
            self.history.append(baseline_log)

        return self._build_final_result(
            success=False,
            iterations=min(test_round, self.max_iterations),
            start_time=start_time,
            final_message="Reached ReAct step budget without full convergence.",
            status="MAX_ITERATIONS_REACHED",
        )

    def run_loop(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """Runs the autonomous iteration loop using either ReAct tool use or patch mode."""
        self.history.clear()
        self._telemetry = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_latency_ms": 0.0,
            "rollbacks": 0,
        }
        start_time = time.time()

        use_react = (
            self.agent_mode == "react"
            or (self.agent_mode == "auto" and len(self.writable_files) > 1)
        )
        if use_react:
            return self._run_react_loop(start_time=start_time, callback=callback)
        return self._run_patch_loop(start_time=start_time, callback=callback)
