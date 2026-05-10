# report_gen/orchestrator.py
"""
Pro Mode Orchestrator — thin wrapper around the Boss ReAct agent.

The Boss agent is a single ReAct agent with sub-agent tools.
Each sub-agent (architect, HVAC, auditor, etc.) is exposed as a tool
that internally runs its own ReAct loop with shared retriever components.

The Boss's natural reasoning loop handles planning, delegation, 
cross-referencing, and report writing — no StateGraph routing needed.
"""

from typing import Any

from agent.agent import build_pro_agent


def build_report_graph(cfg: Any, checkpointer: Any = None):
    """Build the Pro Mode agent — a Boss ReAct agent with sub-agent tools."""
    return build_pro_agent(cfg, checkpointer=checkpointer)
