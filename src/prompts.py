"""
Prompt Engineering & Context Templates for Autonomous Code Synthesis & Verification Agent.
Encapsulates TDD gatekeeper rules, multi-turn failure context, anti-stagnation hints,
and ReAct tool-use instructions.
"""

from __future__ import annotations

from typing import List, Optional

from .tools import format_tools_for_prompt

SYSTEM_PROMPT = """You are an Autonomous Code Synthesis & Systems Engineer operating inside a TDD night-relay loop.

Your mission is to inspect failing tests, diagnose root causes, and patch the authorized target implementation until all tests pass without human intervention.

OPERATIONAL RULES:
1. TEST IMMUTABILITY: Test files are immutable ground-truth specifications. NEVER attempt to modify or bypass tests.
2. ROOT CAUSE DIAGNOSIS: Analyze exact tracebacks, boundary conditions (negative indices, empty inputs, shape mismatches, NaN/Inf, mutability/aliasing), and type invariants.
3. NUMERICAL PRECISION RIGOR: Never replace true floating-point arithmetic with arbitrary `round(..., 6)` decimal rounding. Use `math.fsum`, IEEE-754 `float32`/`float64` semantics, and `math.isfinite` checks as required.
4. SYNTAX & POLICY COMPLIANCE: Output valid Python code with no hidden side effects.
5. RESPONSE FORMAT:
   [THOUGHT]: Concise root cause analysis and planned code fix.
   [CODE]: Complete updated Python implementation for the target file inside ```python ... ``` fences.
"""

REACT_SYSTEM_PROMPT = f"""You are an Autonomous Software Engineer executing an unattended Night-Relay TDD workflow.

You operate inside a restricted sandbox with explicit write allowlists and immutable test gates.

OPERATIONAL RULES:
1. NEVER attempt to modify test files or access paths outside the project workspace.
2. Inspect the codebase before guessing: read relevant source modules and visible test files to understand cross-file contracts.
3. Prefer minimal, surgical `patch_file` edits (or `write_file` when rewriting a small module cleanly).
4. After applying changes, ALWAYS call `run_tests` to verify against the isolated sandbox gate.
5. Numerical rigor: Do NOT use `round(..., 6)` to mask floating-point behavior; use exact IEEE-754 / `math.fsum` / `math.isfinite` logic.
6. If native tool calling is unavailable, invoke exactly one tool per turn using:
   [THOUGHT] your reasoning [/THOUGHT]
   [TOOL_CALL] {{"name": "tool_name", "arguments": {{...}}}} [/TOOL_CALL]

{format_tools_for_prompt()}
"""


def build_error_feedback_prompt(
    target_file: str,
    current_source: str,
    test_file: str,
    test_output: str,
    iteration: int,
    max_iterations: int,
    task_spec: Optional[str] = None,
    stagnation_warning: Optional[str] = None,
    previous_attempts_summary: Optional[List[str]] = None,
) -> str:
    """Constructs prompt incorporating current source, traceback, and prior attempt history."""
    sections: List[str] = [
        f"### TDD GATEKEEPER REPORT (Iteration {iteration} of {max_iterations})",
        f"Target File to Patch: `{target_file}`",
        f"Test Gate Executed: `{test_file}`",
    ]
    if task_spec:
        sections.append(f"#### Task Specification:\n{task_spec}")

    sections.append(f"#### Current Source Code (`{target_file}`):\n```python\n{current_source}\n```")
    sections.append(f"#### Test Execution Output & Traceback:\n```text\n{test_output}\n```")

    if previous_attempts_summary:
        hist_text = "\n".join(f"- {item}" for item in previous_attempts_summary[-3:])
        sections.append(f"#### Previous Attempts Summary (Do NOT repeat failed strategies):\n{hist_text}")

    if stagnation_warning:
        sections.append(f"#### ⚠️ STAGNATION ALERT:\n{stagnation_warning}")

    sections.append(
        "#### Instructions:\n"
        f"1. Diagnose why the assertions or boundary checks failed.\n"
        f"2. Provide [THOUGHT] explaining the root cause and fix.\n"
        f"3. Provide the complete corrected `{target_file}` inside ```python ... ``` under [CODE]."
    )
    return "\n\n".join(sections)


def build_react_task_prompt(
    writable_files: List[str],
    test_file: str,
    initial_test_output: str,
    task_spec: Optional[str] = None,
) -> str:
    parts: List[str] = [
        "### NIGHT-RELAY AUTONOMOUS TASK BRIEF",
        f"Authorized Writable Files: {', '.join(f'`{f}`' for f in writable_files)}",
        f"Immutable TDD Gate: `{test_file}`",
    ]
    if task_spec:
        parts.append(f"#### Specification:\n{task_spec}")
    parts.append(f"#### Baseline TDD Gate Output (currently failing):\n```text\n{initial_test_output}\n```")
    parts.append(
        "Inspect the workspace files using `list_files` / `read_file`, fix the implementation in the "
        "authorized writable files, and verify with `run_tests` until all tests pass."
    )
    return "\n\n".join(parts)
