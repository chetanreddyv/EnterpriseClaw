"""Typed error taxonomy for orchestration-level fault handling."""


class AgentError(Exception):
    """Base exception for all agent faults."""


class InfrastructureError(AgentError):
    """Fatal system fault that the LLM cannot recover from in the current run."""


class ToolExecutionError(AgentError):
    """Recoverable tool failure that the LLM may recover from."""
