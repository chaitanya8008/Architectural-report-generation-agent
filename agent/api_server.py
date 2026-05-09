import os
import json
import asyncio
import sys
from typing import AsyncGenerator, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

# ─── Resolve paths ────────────────────────────────────────────────────────────
_AGENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from core import load_config
from agent import build_agent
from report_gen.orchestrator import build_report_graph
from langchain_core.messages import HumanMessage, AIMessageChunk

# ─── Load Environment ─────────────────────────────────────────────────────────
load_dotenv(_PROJECT_ROOT / ".env", override=True)

app = FastAPI(title="AcoustiQ Pro API")

# Enable CORS for React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from langgraph.checkpoint.memory import MemorySaver

# ─── Global Agent Instances ───────────────────────────────────────────────────
cfg = load_config()
checkpointer = MemorySaver()

# FAST MODE: Standard ReAct Agent
fast_graph = build_agent(cfg, checkpointer=checkpointer)

# PRO MODE: Multi-Agent Orchestrator
pro_graph = build_report_graph(cfg, checkpointer=checkpointer)

@app.get("/stats")
async def stats():
    # In a real app, you'd query Qdrant/DB for these. 
    return {
        "chunks": 113798,
        "latency": "0.43s",
        "model": cfg.chat_model,
        "status": "Ready",
        "index_date": "2026-05-09"
    }

@app.post("/chat")
async def chat(request: Request):
    """
    Unified chat endpoint with Mode selection (Fast/Pro).
    """
    body = await request.json()
    prompt = body.get("message", "")
    mode = body.get("mode", "fast") # Default to fast
    thread_id = body.get("thread_id", "default-web-user")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Choose which graph to run
    active_graph = pro_graph if mode == "pro" else fast_graph
    
    async def event_generator() -> AsyncGenerator:
        try:
            if mode == "pro":
                # Pro mode (LangGraph Orchestrator)
                # We stream updates so the user sees which agent is active
                async for event in active_graph.astream(
                    {"project_id": "current", "revision_label": "all", "sections": {}, "ledger": {}, "deficiencies": [], "completed_steps": []}, 
                    config=config,
                    stream_mode="updates"
                ):
                    node_name = list(event.keys())[0]
                    node_data = event[node_name]
                    
                    if node_name == "boss":
                        yield {"event": "status", "data": f"Boss Agent: {node_data.get('current_task')}"}
                    elif node_name == "architect":
                        yield {"event": "status", "data": "Architect Scout is mapping wall schedules..."}
                    elif node_name == "auditor":
                        yield {"event": "status", "data": "Safety Auditor is running the sweep..."}
                    
                    # If the node produced a report section, send it as text
                    if "sections" in node_data:
                        for sec_name, content in node_data["sections"].items():
                            yield {"event": "text", "data": f"\n\n### {sec_name}\n{content}\n"}

            else:
                # Fast mode (Standard ReAct Agent)
                async for chunk in active_graph.astream(
                    {"messages": [HumanMessage(content=prompt)]}, 
                    config=config, 
                    stream_mode="messages"
                ):
                    msg = chunk[0] if isinstance(chunk, tuple) else chunk
                    
                    if isinstance(msg, AIMessageChunk):
                        if hasattr(msg, "content"):
                            content = msg.content
                            if isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict):
                                        if part.get("type") == "thought":
                                            yield {"event": "thought", "data": part.get("thought", "")}
                                        elif part.get("type") == "text":
                                            yield {"event": "text", "data": part.get("text", "")}
                                    elif isinstance(part, str):
                                        yield {"event": "text", "data": part}
                            elif isinstance(content, str) and content:
                                yield {"event": "text", "data": content}
                        
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield {"event": "tool", "data": json.dumps({"name": tc["name"], "id": tc["id"]})}
                    
                    from langchain_core.messages import ToolMessage
                    if isinstance(msg, ToolMessage):
                        import re
                        content_str = msg.content
                        if "**[1]**" in content_str:
                            hits = re.split(r'\*\*\[\d+\]\*\*', content_str)
                            for i in range(1, len(hits)):
                                hit_content = hits[i].split("---")[0].strip()
                                yield {
                                    "event": "source",
                                    "data": json.dumps({"id": f"{msg.tool_call_id}_{i}", "content": hit_content[:1500]})
                                }
                        else:
                            yield {
                                "event": "source",
                                "data": json.dumps({"id": msg.tool_call_id, "content": msg.content[:1500]})
                            }
                             
        except Exception as e:
            yield {"event": "error", "data": str(e)}
        finally:
            yield {"event": "done", "data": "[DONE]"}
 
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    # Start on 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
