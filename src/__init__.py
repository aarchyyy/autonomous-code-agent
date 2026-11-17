"""
Autonomous Code Synthesis & Verification Agent (ACS-Agent) — Complete Edition
"""
from .agent import CodeIterationAgent
from .checkpoint import FileSnapshotManager, GitRelayManager
from .context import ContextWindowManager, estimate_conversation_tokens
from .llm_backend import (
    HARDWARE_PRESETS,
    AgentTurn,
    BaseLLM,
    DeterministicMockLLM,
    LiveLLMClient,
    create_llm_backend,
    probe_llm_endpoint,
)
from .relay import RelayOrchestrator, RelayPlan, RelayTask
from .sandbox import (
    ASTPolicyError,
    ASTPolicyInspector,
    IntegrityMonitor,
    IntegrityViolationError,
    IsolatedTestRunner,
    NativeSyntaxInspector,
    PathGuard,
    PathSecurityError,
    SandboxPolicy,
    detect_cpp_compiler,
    detect_docker_runtime,
)
from .tools import TOOL_DEFINITIONS, ToolBox

__all__ = [
    "CodeIterationAgent",
    "ToolBox",
    "TOOL_DEFINITIONS",
    "ContextWindowManager",
    "estimate_conversation_tokens",
    "BaseLLM",
    "LiveLLMClient",
    "DeterministicMockLLM",
    "AgentTurn",
    "HARDWARE_PRESETS",
    "create_llm_backend",
    "probe_llm_endpoint",
    "RelayOrchestrator",
    "RelayPlan",
    "RelayTask",
    "FileSnapshotManager",
    "GitRelayManager",
    "SandboxPolicy",
    "PathGuard",
    "PathSecurityError",
    "IntegrityMonitor",
    "IntegrityViolationError",
    "ASTPolicyInspector",
    "NativeSyntaxInspector",
    "ASTPolicyError",
    "IsolatedTestRunner",
    "detect_cpp_compiler",
    "detect_docker_runtime",
]
