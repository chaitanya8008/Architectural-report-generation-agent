"""
AcoustiQ agent package.

This package provides a retrieval-augmented generation (RAG) agent for
acoustic consulting queries. It uses hybrid search (dense + sparse + RRF)
with Cohere cross-encoder reranking over a Qdrant vector store.

Structure:
- core.py: Configuration and utilities
- agent.py: ReAct agent with MCP retrieval tools
- ../mcp_server/: Standalone MCP retrieval server (tools + retrieval logic)
"""

from __future__ import annotations

from .core import AgentConfig, load_config


def build_agent(*args, **kwargs):
    """Lazy import graph builder to avoid hard dependency during retrieval-only scripts."""
    from .agent import build_agent as _build_agent

    return _build_agent(*args, **kwargs)


__all__ = [
    "AgentConfig",
    "load_config",
    "build_agent",
]
