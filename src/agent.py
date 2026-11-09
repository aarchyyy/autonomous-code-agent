"""
Core Agent Module: Orchestrates the ReAct reasoning, TDD gatekeeper feedback,
AST validation, patch application, and self-correction loop.
"""

import time
from typing import Dict, Any, List, Optional, Callable
from .tools import ToolBox
from .prompts import SYSTEM_PROMPT, build_error_feedback_prompt
from .llm_backend import BaseLLM, create_llm_backend

class CodeIterationAgent:
    def __init__(
        self,
        target_file: str,
        test_file: str,
        workspace_root: str = ".",
        max_iterations: int = 4,
        llm_backend: Optional[BaseLLM] = None
    ):
        self.target_file = target_file
        self.test_file = test_file
        self.workspace_root = workspace_root
        self.max_iterations = max_iterations
        self.tools = ToolBox(workspace_root=workspace_root)
        self.llm = llm_backend or create_llm_backend()
        self.history: List[Dict[str, Any]] = []

    def execute_step(self, iteration: int) -> Dict[str, Any]:
        """
        Executes a single iteration step of the TDD self-correction cycle.
        """
        step_log: Dict[str, Any] = {
            "iteration": iteration,
            "timestamp": time.time(),
            "target_file": self.target_file,
            "test_file": self.test_file
        }

        # Step 1: Run TDD gatekeeper
        test_result = self.tools.run_tdd_gate(self.test_file)
        step_log["test_result"] = test_result

        if test_result["passed"]:
            step_log["status"] = "CONVERGED_SUCCESS"
            step_log["message"] = f"TDD Gate passed 100% on iteration {iteration}."
            return step_log

        # Step 2: Extract current code and formulate prompt
        current_source = self.tools.read_source(self.target_file)
        step_log["before_code"] = current_source
        
        prompt = build_error_feedback_prompt(
            target_file=self.target_file,
            current_source=current_source,
            test_file=self.test_file,
            test_output=test_result["traceback"] or test_result["output"],
            iteration=iteration,
            max_iterations=self.max_iterations
        )

        # Step 3: LLM ReAct Reflection & Patch Synthesis
        thought, patched_code = self.llm.generate_patch(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT
        )
        step_log["agent_thought"] = thought
        step_log["patched_code"] = patched_code

        # Step 4: AST Syntax Pre-flight Guardrail
        valid_syntax, syntax_msg = self.tools.validate_syntax(patched_code)
        step_log["ast_valid"] = valid_syntax
        step_log["ast_message"] = syntax_msg

        if not valid_syntax:
            step_log["status"] = "SYNTAX_ERROR"
            step_log["message"] = f"AST rejection: {syntax_msg}"
            return step_log

        # Step 5: Diff computation & Atomic Patch write
        diff_str = self.tools.compute_diff(
            current_source,
            patched_code,
            filename=self.target_file
        )
        step_log["diff"] = diff_str

        self.tools.write_source(self.target_file, patched_code)
        step_log["status"] = "PATCH_APPLIED"
        step_log["message"] = "Patch safely applied to source code."

        return step_log

    def run_loop(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Runs the full autonomous relay loop until convergence or max iterations reached.
        """
        self.history.clear()
        start_time = time.time()

        for it in range(1, self.max_iterations + 1):
            step_res = self.execute_step(it)
            self.history.append(step_res)

            if callback:
                callback(step_res)

            if step_res["status"] == "CONVERGED_SUCCESS":
                elapsed = time.time() - start_time
                return {
                    "success": True,
                    "iterations": it,
                    "elapsed_seconds": round(elapsed, 3),
                    "history": self.history,
                    "final_message": f"Successfully converged in {it} iteration(s) ({round(elapsed, 2)}s)."
                }

        # Check post-final gate status
        final_gate = self.tools.run_tdd_gate(self.test_file)
        elapsed = time.time() - start_time
        if final_gate["passed"]:
            return {
                "success": True,
                "iterations": self.max_iterations,
                "elapsed_seconds": round(elapsed, 3),
                "history": self.history,
                "final_message": f"All tests passed after final patch verification ({round(elapsed, 2)}s)."
            }

        return {
            "success": False,
            "iterations": self.max_iterations,
            "elapsed_seconds": round(elapsed, 3),
            "history": self.history,
            "final_message": f"Reached max iterations ({self.max_iterations}) without full convergence. Human review advised."
        }
