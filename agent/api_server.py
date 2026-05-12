import os
import json
import asyncio
import sys
from typing import AsyncGenerator, Any
from pathlib import Path

# Unicode safety for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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

# PRO MODE: Boss ReAct Agent with Sub-Agent Tools
# Shared retriever is initialized ONCE here — no per-request rebuilds
pro_graph = build_report_graph(cfg, checkpointer=checkpointer)

@app.get("/stats")
async def stats():
    return {
        "chunks": 113798,
        "latency": "0.43s",
        "model": cfg.chat_model,
        "status": "Ready",
        "index_date": "2026-05-09"
    }

@app.get("/projects")
async def list_projects():
    """Auto-discover available projects by scanning the filter_registries directory."""
    from pathlib import Path
    registries_dir = Path(__file__).resolve().parent.parent / "filter_registries"
    projects = []
    if registries_dir.exists():
        for folder in sorted(registries_dir.iterdir()):
            if folder.is_dir():
                project_id = folder.name
                # Convert folder name to display name: project_dayton_hotel -> Dayton Hotel
                display_name = project_id.replace("project_", "").replace("_", " ").title()
                projects.append({"project_id": project_id, "display_name": display_name})
    return {"projects": projects}

@app.post("/chat")
async def chat(request: Request):
    """
    Unified chat endpoint with Mode selection (Fast/Pro).
    Both modes now use the same ReAct streaming — Pro mode just has
    additional sub-agent tools available to the Boss.
    """
    body = await request.json()
    prompt = body.get("message", "")
    mode = body.get("mode", "fast")
    thread_id = body.get("thread_id", "default-web-user")
    project_id = body.get("project_id", cfg.project_id)
    config = {"configurable": {"thread_id": thread_id, "project_id": project_id}}
    
    # Choose which graph to run
    active_graph = pro_graph if mode == "pro" else fast_graph
    
    # Pro mode gets a fresh thread for each request (no state leakage)
    if mode == "pro":
        import uuid
        config["configurable"]["thread_id"] = str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator:
        try:
            # Both Fast and Pro mode stream identically via ReAct message stream
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
                            tool_name = tc["name"]
                            # Emit friendly status for sub-agent dispatches
                            if tool_name.startswith("run_"):
                                agent_name = tool_name.replace("run_", "").replace("_", " ").title()
                                yield {"event": "status", "data": f"{agent_name} Agent is working..."}
                            yield {"event": "tool", "data": json.dumps({"name": tool_name, "id": tc["id"]})}
                
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
                                "data": json.dumps({"id": f"{msg.tool_call_id}_{i}", "content": hit_content})
                            }
                    else:
                        yield {
                            "event": "source",
                            "data": json.dumps({"id": msg.tool_call_id, "content": msg.content})
                        }
                             
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"DEBUG ERROR in event_generator: {error_details}")
            yield {"event": "text", "data": f"\n\n**System Error:** {str(e)}\n"}
            yield {"event": "error", "data": str(e)}
        finally:
            yield {"event": "done", "data": "[DONE]"}
 
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
