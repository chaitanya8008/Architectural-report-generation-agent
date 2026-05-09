# report_gen/orchestrator.py
import os
import json
from typing import Any, List, Dict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .state import ReportState
from .personas import BOSS_PROMPT, ARCHITECTURAL_SCOUT_PROMPT, CONSISTENCY_AUDITOR_PROMPT
from agent.agent import build_agent

def build_report_graph(cfg: Any, checkpointer: Any = None):
    # ── Initialize LLM ──
    # Ensure we use the same Vertex AI config as agent.py
    llm = ChatGoogleGenerativeAI(
        model=cfg.chat_model,
        location="us-central1",
        project=os.environ.get("GCP_PROJECT_ID"),
        vertexai=True,
        temperature=0
    )

    # ── Worker Factory ──
    # Create specialists on demand
    def get_worker(persona_prompt: str):
        return build_agent(cfg, system_prompt=persona_prompt)

    # ── Node 1: The Boss Agent (Planner) ──
    def boss_node(state: ReportState):
        prompt = BOSS_PROMPT + f"\n\nCurrent Report Sections: {list(state['sections'].keys())}"
        msg = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Decide the next step.")])
        # Simple sequence for prototype
        if "Architectural" not in state['sections']:
            return {"next_agent": "architect", "current_task": "Map all wall types."}
        if "SafetySweep" not in state['sections']:
            return {"next_agent": "auditor", "current_task": "Run Tool C."}
        return {"next_agent": "FINISH"}

    # ── Node 2: The Architect Node ──
    def architect_node(state: ReportState):
        worker = get_worker(ARCHITECTURAL_SCOUT_PROMPT)
        result = worker.invoke({"messages": [HumanMessage(content=state["current_task"])]})
        last_msg = result["messages"][-1].content
        return {
            "sections": {"Architectural": last_msg}, 
            "completed_steps": ["arch_scout"]
        }

    # ── Node 3: The Auditor Node ──
    def auditor_node(state: ReportState):
        worker = get_worker(CONSISTENCY_AUDITOR_PROMPT)
        # Note: In a real run, we'd pass seen_chunk_ids here
        result = worker.invoke({"messages": [HumanMessage(content="Run a safety sweep on the project.")]})
        last_msg = result["messages"][-1].content
        return {
            "sections": {"SafetySweep": last_msg}, 
            "completed_steps": ["safety_sweep"]
        }

    # ── Build Graph ──
    workflow = StateGraph(ReportState)
    workflow.add_node("boss", boss_node)
    workflow.add_node("architect", architect_node)
    workflow.add_node("auditor", auditor_node)

    workflow.set_entry_point("boss")
    
    # Conditional logic
    def router(state):
        if state["next_agent"] == "architect": return "architect"
        if state["next_agent"] == "auditor": return "auditor"
        return "end"

    workflow.add_conditional_edges("boss", router, {"architect": "architect", "auditor": "auditor", "end": END})
    workflow.add_edge("architect", "boss")
    workflow.add_edge("auditor", "boss")

    return workflow.compile()
