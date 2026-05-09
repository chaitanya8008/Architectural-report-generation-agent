"""
AcoustiQ Live CLI Agent.

Usage:
    cd agent
    python run_agent.py            # interactive CLI
    python run_agent.py --batch    # run 10 evaluation questions and save to JSON
"""
import os
import sys
import io
import json
import time
import threading
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# pyrefly: ignore [missing-import]
from pathlib import Path
from dotenv import load_dotenv

# ── Resolve paths ────────────────────────────────────────────────────────────
_AGENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENT_DIR.parent

load_dotenv(_PROJECT_ROOT / ".env")

from core import load_config
from agent import build_agent
import argparse

# ── Argument Parsing ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--batch", action="store_true", help="Run evaluation questions")
parser.add_argument("--query", type=str, help="Run a single query")
args = parser.parse_args()

batch_mode = args.batch
single_query = args.query


# ── Context Directory ───────────────────────────────────────────────────────
CONTEXT_DIR = os.path.join(os.path.dirname(__file__), "context")
if not os.path.exists(CONTEXT_DIR):
    os.makedirs(CONTEXT_DIR)


def sanitize_filename(text):
    """Clean a string to be a safe filename."""
    s = re.sub(r'[^\w\s-]', '', text).strip().lower()
    s = re.sub(r'[-\s]+', '_', s)
    return s[:100]


# ── Live Timer ───────────────────────────────────────────────────────────────
class LiveTimer:
    """Displays a live-updating elapsed timer on the current line."""
    
    def __init__(self, prefix=""):
        self.prefix = prefix
        self._running = False
        self._thread = None
        self._start_time = None
    
    def start(self):
        self._start_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
    
    def stop(self, tokens=None):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        elapsed = time.time() - self._start_time
        msg = f"\r{self.prefix} done in {elapsed:.2f}s"
        if tokens:
            ti = tokens.get("input_tokens") or tokens.get("prompt_tokens")
            to = tokens.get("output_tokens") or tokens.get("completion_tokens")
            if ti is not None and to is not None:
                msg += f" {DIM}[Tokens: 📥 {ti:,} | 📤 {to:,}]{RESET}"
        sys.stdout.write(f"{msg}\n")
        sys.stdout.flush()
        return elapsed
    
    def _tick(self):
        while self._running:
            elapsed = time.time() - self._start_time
            # Only tick for 'Thinking' phase, tools print their own progress
            if "Thinking" in self.prefix:
                sys.stdout.write(f"\r{self.prefix} {elapsed:.1f}s ...")
                sys.stdout.flush()
            time.sleep(0.1)


# ── ANSI color helpers ───────────────────────────────────────────────────────
DIM = "\033[2m"
CYAN = "\033[36m"
RESET = "\033[0m"

# ── Run a single query and return the answer text ────────────────────────────
def run_query(agent_graph, query, config):
    """Stream a query through the agent, print live output, return answer text."""
    initial_state = {"messages": [("user", query)]}
    
    has_streamed_text = False
    tool_timer = None
    think_timer = None
    tool_call_count = 0
    collected_text = []
    
    def stop_think_timer(tokens=None):
        nonlocal think_timer
        if think_timer:
            think_timer.stop(tokens=tokens)
            think_timer = None
    
    def stop_tool_timer():
        nonlocal tool_timer
        if tool_timer:
            tool_timer.stop()
            tool_timer = None
    
    # Setup context logging
    q_slug = sanitize_filename(query)
    context_file = os.path.join(CONTEXT_DIR, f"{q_slug}.txt")
    with open(context_file, "w", encoding="utf-8") as f:
        f.write(f"USER QUERY: {query}\n")
        f.write("="*80 + "\n\n")

    step_logged = 0

    # Start LLM thinking timer
    think_timer = LiveTimer(prefix="  🧠 Thinking")
    think_timer.start()
    
    for chunk_type, chunk_data in agent_graph.stream(initial_state, config=config, stream_mode=["messages", "values"]):
        if chunk_type == "messages":
            msg, metadata = chunk_data
            if msg.type != "ai" or not msg.content:
                continue
            
            content = msg.content
            
            # Gemini with include_thoughts=True sends:
            #   1. list[{"type":"thinking","text":""}] during thinking phase
            #   2. str("actual response text") for the response
            
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "text")
                        btext = block.get("text", "")
                        
                        if btype == "thinking":
                            # Thinking phase — timer is already running, nothing to print
                            # (Gemini doesn't expose thinking text via API)
                            continue
                        
                        if btype == "text" and btext:
                            stop_think_timer()
                            if not has_streamed_text:
                                print("\n🤖 AcoustiQ: ", end="", flush=True)
                            has_streamed_text = True
                            collected_text.append(btext)
                            print(btext, end="", flush=True)
                    
                    elif isinstance(block, str) and block:
                        stop_think_timer()
                        if not has_streamed_text:
                            print("\n🤖 AcoustiQ: ", end="", flush=True)
                        has_streamed_text = True
                        collected_text.append(block)
                        print(block, end="", flush=True)
            
            elif isinstance(content, str) and content:
                stop_think_timer()
                if not has_streamed_text:
                    print("\n🤖 AcoustiQ: ", end="", flush=True)
                has_streamed_text = True
                collected_text.append(content)
                print(content, end="", flush=True)
                    
        if chunk_type == "values":
            state = chunk_data
            messages = state["messages"]
            message = messages[-1]
            
            # Log the full prompt context for the NEXT step
            # We log whenever the state values update, as that's what the LLM sees
            step_logged += 1
            with open(context_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n[PROMPT STEP {step_logged}]\n")
                f.write("-" * 40 + "\n")
                for m in messages:
                    role = m.type
                    content = m.content
                    if role == "ai" and hasattr(m, "tool_calls") and m.tool_calls:
                        content = f"TOOL CALLS: {json.dumps(m.tool_calls, indent=2)}"
                    f.write(f"ROLE: {role}\nCONTENT: {content}\n\n")
                f.write("-" * 40 + "\n")

            # Extract usage metadata for token reporting
            usage = None
            if hasattr(message, "usage_metadata") and message.usage_metadata:
                usage = message.usage_metadata
            elif hasattr(message, "response_metadata") and "usage" in message.response_metadata:
                usage = message.response_metadata["usage"]

            if message.type == "ai":
                if hasattr(message, "tool_calls") and message.tool_calls:
                    stop_think_timer(tokens=usage)
                    
                    if has_streamed_text:
                        print()
                    
                    for tc in message.tool_calls:
                        tool_call_count += 1
                        print(f"  🛠️  [{tool_call_count}] {tc['name']}({tc['args']})", flush=True)
                    
                    tool_timer = LiveTimer(prefix=f"  ⏱️  Tool execution")
                    tool_timer.start()
                    has_streamed_text = False
                    
                else:
                    # Final AI response (fallback if messages stream didn't fire)
                    if not has_streamed_text:
                        stop_think_timer(tokens=usage)
                        print("\n🤖 AcoustiQ: ", end="", flush=True)
                        answer_raw = message.content
                        if isinstance(answer_raw, list):
                            answer_text = "".join(
                                block.get("text", "") if isinstance(block, dict) and block.get("type") == "text"
                                else str(block) if isinstance(block, str) else ""
                                for block in answer_raw
                            )
                        else:
                            answer_text = str(answer_raw)
                        collected_text.append(answer_text)
                        print(answer_text, end="", flush=True)
                    has_streamed_text = False
                    
            elif message.type == "tool":
                stop_tool_timer()
                think_timer = LiveTimer(prefix="  🧠 Thinking")
                think_timer.start()
    
    # Cleanup
    stop_think_timer()
    stop_tool_timer()
    
    print()  # Final newline
    return "".join(collected_text)


# ── Batch Evaluation Questions ───────────────────────────────────────────────
BATCH_QUESTIONS = [
    "What is the composition of partition detail KS/KA from drawing A8.01?",
    "What floor assembly is specified for guestroom corridors on levels 2, 3, and 4?",
    "What is the floor assembly at the level 5 mechanical zone and what is missing from the 50% CD drawings compared to the 100% DD set?",
    "What partition type is used between the guestroom and elevator, staircase, and pantry, and what are its key components?",
    "What is the construction type for levels 2 and above, and what acoustic limitation does this introduce?",
    "What resilient flooring product is specified in the gym and where exactly does it apply?",
    "What partition is proposed between the wine library and lobby, and what does the acoustic recommendation say about it?",
    "What are the NC criteria for the event space and the spa?",
    "What vibration isolator is specified for FCU-1.3 through FCU-1.10?",
    "What is the partition recommendation between the office and office spaces at level 1, and why is the proposed partition type insufficient?",
]


# ── Setup ────────────────────────────────────────────────────────────────────
cfg = load_config()

from langgraph.checkpoint.memory import MemorySaver

print("Building ReAct agent with MCP retrieval tools...")
memory = MemorySaver()
agent_graph = build_agent(cfg, checkpointer=memory)


# ── Batch Mode ───────────────────────────────────────────────────────────────
if batch_mode:
    print("\n" + "=" * 80)
    print(f"  📋 Running {len(BATCH_QUESTIONS)} evaluation questions")
    print("=" * 80)
    
    results = {}
    total_start = time.time()
    output_file = "agent_results.json"
    
    for i, query in enumerate(BATCH_QUESTIONS, 1):
        # Each question gets its own thread_id so there's no conversation bleed
        config = {"configurable": {"thread_id": f"batch_q{i}"}}
        
        print(f"\n{'─' * 80}")
        print(f"  📝 Question {i}/{len(BATCH_QUESTIONS)}")
        print(f"  🗣️  {query}")
        print(f"{'─' * 80}")
        
        # Retry with backoff for rate limit errors
        answer = None
        for attempt in range(3):
            try:
                answer = run_query(agent_graph, query, config)
                break
            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in exc_str or "resource_exhausted" in exc_str or "rate" in exc_str:
                    wait = 60 * (attempt + 1)
                    print(f"\n  ⚠️  Rate limited (attempt {attempt+1}/3). Waiting {wait}s...")
                    time.sleep(wait)
                    # Reset thread_id so it retries fresh
                    config = {"configurable": {"thread_id": f"batch_q{i}_retry{attempt+1}"}}
                else:
                    print(f"\n  ❌ Error: {exc}")
                    answer = f"ERROR: {exc}"
                    break
        
        if answer is None:
            answer = "ERROR: All retries exhausted (rate limited)"
        
        results[query] = answer
        
        # Save partial results after each question
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 80)
    print(f"  ✅ All {len(BATCH_QUESTIONS)} questions processed in {total_elapsed:.1f}s")
    print(f"  📄 Results saved to {output_file}")
    print("=" * 80)


# ── Single Query Mode ────────────────────────────────────────────────────────
elif single_query:
    config = {"configurable": {"thread_id": "single_q_1"}}
    run_query(agent_graph, single_query, config)


# ── Interactive Mode ─────────────────────────────────────────────────────────
else:
    config = {"configurable": {"thread_id": "cli_session_1"}}
    
    print("\n" + "=" * 80)
    print("  🤖 AcoustiQ Live CLI Agent")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 80)
    
    while True:
        try:
            query = input("\n🗣️  You: ")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break
            
        query = query.strip()
        if query.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
        if not query:
            continue
            
        run_query(agent_graph, query, config)