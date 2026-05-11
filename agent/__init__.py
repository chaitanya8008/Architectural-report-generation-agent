"""
AcoustiQ agent package.

Provides two ReAct agent modes for acoustic consulting:
  - Fast Mode (build_agent): Single agent for Q&A
  - Pro Mode (build_pro_agent): Boss agent dispatching 8 specialist sub-agents

Both use hybrid search (dense + sparse + RRF) with Cohere cross-encoder
reranking over Qdrant, via the shared HybridRetriever in mcp_server/retrieval.py.

Structure:
- core.py: Configuration and response parsing utilities
- agent.py: Fast Mode agent, Pro Mode boss + sub-agent builder
- ../report_gen/personas.py: Sub-agent persona prompts
- ../mcp_server/retrieval.py: HybridRetriever (shared retrieval engine)
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
