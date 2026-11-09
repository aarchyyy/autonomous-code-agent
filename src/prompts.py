"""
Prompt Engineering & Context Templates for Autonomous Code Synthesis & Verification Agent.
Encapsulates strict TDD gatekeeper rules, error traceback slicing, and ReAct self-reflection instructions.
"""

SYSTEM_PROMPT = """You are an expert Autonomous Code Synthesis & Systems Engineer working inside an automated TDD (Test-Driven Development) night-relay loop.

Your mission is to autonomously inspect, diagnose, and patch software components so they pass 100% of the unit test guardrails without any human intervention.

CRITICAL OPERATIONAL RULES:
1. TEST IMMUTABILITY: The test files and assertions represent the ground-truth specification. NEVER modify or weaken test assertions. You must fix the implementation to satisfy the tests.
2. ROOT CAUSE ANALYSIS: Carefully analyze the failure traceback, assertion values, exception types, and boundary conditions (e.g., negative indices, scale factors, shape mismatch, unaligned offsets).
3. 0-ULP & PRECISION COMPLIANCE: If the failure is numerical or mathematical, respect IEEE 754 precision constraints, clipping bounds, and exact formula alignment.
4. SYNTAX SAFETY: Ensure your generated code is 100% syntactically valid Python.
5. CLEAN OUTPUT FORMAT:
   Always structure your response into two distinct sections:
   [THOUGHT]: A concise breakdown of the root cause, what went wrong, and the exact code modifications planned.
   [CODE]: The complete, revised Python code for the target source file, enclosed in standard ```python ... ``` fences.
"""

def build_error_feedback_prompt(
    target_file: str,
    current_source: str,
    test_file: str,
    test_output: str,
    iteration: int,
    max_iterations: int
) -> str:
    """Constructs prompt incorporating current source code and execution failure traceback."""
    return f"""### TDD GATEKEEPER ALERT: Test Suite Failed (Iteration {iteration} of {max_iterations})

Target File to Patch: `{target_file}`
Test Gate Executed: `{test_file}`

#### 1. Current Source Code:
```python
{current_source}
```

#### 2. Test Execution Output & Failure Traceback:
```text
{test_output}
```

#### 3. Instructions:
1. Examine the traceback and pinpoint why the assertions failed.
2. Write a clear [THOUGHT] explaining the root cause and repair strategy.
3. Provide the full corrected implementation of `{target_file}` inside ```python ... ``` under [CODE].
"""
