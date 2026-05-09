"""
AcoustiQ — Premium Streamlit Frontend
Inspired by Claude's conversational UI: clean typography, animated thinking
indicators, robust session state, and smooth token streaming.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from core import AgentConfig, load_config
from agent import build_agent

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    BaseCallbackHandler = object

import threading
import queue


# ─── Constants ────────────────────────────────────────────────────────────────

_THINKING_PHRASES = [
    "Reading project documents…",
    "Searching acoustic specifications…",
    "Cross-referencing partition tables…",
    "Checking STC and IIC ratings…",
    "Reviewing wall assemblies…",
    "Analyzing floor-ceiling details…",
    "Scanning HVAC noise criteria…",
    "Identifying relevant sections…",
    "Comparing design variants…",
    "Pulling construction notes…",
    "Verifying isolation requirements…",
    "Composing answer…",
]


# ─── Streaming callback ──────────────────────────────────────────────────────

class QueueStreamHandler(BaseCallbackHandler):
    """Routes LLM tokens to a thread-safe queue for real-time streaming."""

    def __init__(self, q: queue.Queue):
        self.q = q
        self.is_streaming_answer = False

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        try:
            msg_content = str(messages).lower()
            if "acoustic consultant" in msg_content or "sourcing rules" in msg_content:
                self.is_streaming_answer = True
                self.q.put("__STATUS__:ANSWERING")
            elif "break down" in msg_content or "sub-questions" in msg_content:
                self.q.put("__STATUS__:PLANNING")
            elif "reviewing an answer" in msg_content or "completeness" in msg_content:
                self.q.put("__STATUS__:VALIDATING")
            else:
                self.is_streaming_answer = False
        except Exception:
            self.is_streaming_answer = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if self.is_streaming_answer:
            self.q.put(token)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        pass

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        pass

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name") if serialized else None
        if tool_name == "search_documents":
            inputs = kwargs.get("inputs", {})
            query = inputs.get("query", "")
            if query:
                self.q.put(f"__STATUS__:SEARCHING:{query}")
            else:
                self.q.put("__STATUS__:SEARCHING:project documents")
        elif tool_name == "get_sheet_contents":
            inputs = kwargs.get("inputs", {})
            sheet = inputs.get("sheet_number", "")
            self.q.put(f"__STATUS__:SEARCHING:sheet {sheet}")
        elif tool_name == "list_available_filters":
            self.q.put("__STATUS__:SEARCHING:available filters")


# ─── Checkpointers ─────────────────────────────────────────────────────────────

from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

# ─── Backend init (cached) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _init_backend() -> tuple[Any, AgentConfig]:
    load_dotenv()
    cfg = load_config()

    checkpointer = MemorySaver() # Fallback
    if HAS_POSTGRES and cfg.db_uri:
        try:
            pool = ConnectionPool(cfg.db_uri, max_size=10, kwargs={"autocommit": True})
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
        except Exception as e:
            print(f"[INIT] Postgres checkpointer failed: {e}. Falling back to MemorySaver.")
            checkpointer = MemorySaver()

    graph = build_agent(cfg, checkpointer=checkpointer)
    return graph, cfg


# ─── CSS / Styling ───────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Typography ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

/* ── Layout ──────────────────────────────────────────────────────────── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 6rem !important;
    max-width: 820px;
}

/* Hide Streamlit's default header/footer chrome */
#MainMenu, header[data-testid="stHeader"], footer {
    visibility: hidden;
    height: 0;
}

/* ── Brand header ───────────────────────────────────────────────────── */
.aq-header {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.25rem 0 1.2rem 0;
    border-bottom: 1px solid rgba(127,127,127,0.15);
    margin-bottom: 1.5rem;
}
.aq-header .logo {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #6C63FF 0%, #48B4FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.aq-header .tagline {
    font-size: 0.82rem;
    opacity: 0.5;
    font-weight: 400;
    margin-left: 0.1rem;
}

/* ── Chat bubbles ───────────────────────────────────────────────────── */
.stChatMessage {
    border: none !important;
    border-radius: 1rem !important;
    padding: 1rem 1.35rem !important;
    font-size: 0.97rem !important;
    line-height: 1.72 !important;
    margin-bottom: 0.6rem !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

/* User bubble */
.stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(108, 99, 255, 0.06) !important;
    border: 1px solid rgba(108, 99, 255, 0.12) !important;
}

/* Assistant bubble */
.stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: transparent !important;
}

/* ── Chat input ─────────────────────────────────────────────────────── */
.stChatInputContainer {
    padding-bottom: 1rem;
}
.stChatInputContainer textarea {
    border-radius: 1rem !important;
    border: 1.5px solid rgba(127,127,127,0.2) !important;
    padding: 0.85rem 1.1rem !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stChatInputContainer textarea:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.1) !important;
}

/* ── Thinking indicator ─────────────────────────────────────────────── */
@keyframes aq-shimmer {
    0%   { opacity: 0.45; }
    50%  { opacity: 1; }
    100% { opacity: 0.45; }
}
@keyframes aq-pulse-dot {
    0%, 80%, 100% { transform: scale(0); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}
.aq-thinking {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.7rem 0;
    font-size: 0.88rem;
    font-weight: 500;
    color: #6C63FF;
}
.aq-thinking .dots {
    display: inline-flex;
    gap: 3px;
}
.aq-thinking .dots span {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #6C63FF;
    display: inline-block;
    animation: aq-pulse-dot 1.4s infinite ease-in-out both;
}
.aq-thinking .dots span:nth-child(1) { animation-delay: -0.32s; }
.aq-thinking .dots span:nth-child(2) { animation-delay: -0.16s; }
.aq-thinking .dots span:nth-child(3) { animation-delay: 0s; }
.aq-thinking .label {
    animation: aq-shimmer 2.2s ease-in-out infinite;
}

/* ── Sources expander ───────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: rgba(127,127,127,0.7) !important;
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.02);
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 0.6rem;
    font-weight: 500;
    border: 1.5px solid rgba(108,99,255,0.25);
    color: #6C63FF;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(108,99,255,0.08);
    border-color: #6C63FF;
}

/* ── Scrollbar styling ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(127,127,127,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(127,127,127,0.35); }

/* ── Markdown inside chat ───────────────────────────────────────────── */
.stChatMessage h1, .stChatMessage h2, .stChatMessage h3 {
    margin-top: 1rem !important;
    margin-bottom: 0.3rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}
.stChatMessage ul, .stChatMessage ol {
    padding-left: 1.3rem !important;
}
.stChatMessage li {
    margin-bottom: 0.25rem !important;
}
.stChatMessage strong {
    font-weight: 600;
}
</style>
"""


def _thinking_html(phrase: str) -> str:
    return f"""<div class="aq-thinking">
        <div class="dots"><span></span><span></span><span></span></div>
        <span class="label">{phrase}</span>
    </div>"""


# ─── Main app ────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="AcoustiQ · AI Acoustic Consultant",
        page_icon="🔊",
        layout="centered",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # Brand header
    st.markdown(
        '<div class="aq-header">'
        '  <span class="logo">AcoustiQ</span>'
        '  <span class="tagline">AI Acoustic Consultant</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("#### Project")
        st.caption(os.environ.get("DAYTON_PROJECT_ID", "VA_1040_025_DAYTON_HOTEL"))
        st.markdown("---")
        if st.button("✦  New conversation", key="new_convo"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    # ── Session state (robust) ────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # ── Backend ───────────────────────────────────────────────────────────
    try:
        graph, cfg = _init_backend()
    except Exception as exc:
        st.error(f"Backend initialization failed: {exc}")
        st.stop()

    # ── Render history ────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask about the project acoustics…")
    if not prompt:
        return

    # Immediately persist the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Assistant response ────────────────────────────────────────────────
    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        answer_placeholder = st.empty()

        # Start the animated thinking indicator
        thinking_placeholder.markdown(
            _thinking_html(_THINKING_PHRASES[0]),
            unsafe_allow_html=True,
        )

        try:
            q: queue.Queue = queue.Queue()
            handler = QueueStreamHandler(q)
            config = {
                "configurable": {"thread_id": st.session_state.thread_id},
                "callbacks": [handler],
            }
            state = {"messages": [("user", prompt)]}
            result_box: dict[str, Any] = {}

            def _run_graph():
                try:
                    result_box["data"] = graph.invoke(state, config=config)
                except Exception as e:
                    result_box["error"] = e
                finally:
                    q.put(None)  # sentinel — always unblock the consumer

            thread = threading.Thread(target=_run_graph, daemon=True)
            thread.start()

            # ── Token consumer with animated thinking ─────────────────
            streamed_tokens: list[str] = []
            phrase_idx = 0
            last_phrase_time = time.time()
            answering = False

            while True:
                # Cycle thinking phrases every ~1.8s while waiting
                if not answering:
                    now = time.time()
                    if now - last_phrase_time > 1.8:
                        phrase_idx = (phrase_idx + 1) % len(_THINKING_PHRASES)
                        thinking_placeholder.markdown(
                            _thinking_html(_THINKING_PHRASES[phrase_idx]),
                            unsafe_allow_html=True,
                        )
                        last_phrase_time = now

                # Non-blocking read with short timeout so we can cycle phrases
                try:
                    token = q.get(timeout=0.15)
                except queue.Empty:
                    continue

                if token is None:
                    break

                # Handle status signals
                if isinstance(token, str) and token.startswith("__STATUS__:"):
                    parts = token.split(":", 2)
                    status = parts[1]
                    if status == "ANSWERING":
                        answering = True
                        thinking_placeholder.empty()
                    elif status == "SEARCHING":
                        query = parts[2] if len(parts) > 2 else "project documents"
                        msg = f"Searching for '{query}'…"
                        thinking_placeholder.markdown(
                            _thinking_html(msg),
                            unsafe_allow_html=True,
                        )
                    continue

                # Real answer token
                if isinstance(token, str):
                    streamed_tokens.append(token)
                    answer_placeholder.markdown("".join(streamed_tokens) + "▌")

            # Final render (remove cursor)
            full_answer = "".join(streamed_tokens)
            if full_answer:
                answer_placeholder.markdown(full_answer)
            thinking_placeholder.empty()

            thread.join(timeout=5)

            result = result_box.get("data", {})
            messages = result.get("messages", [])
            final_answer = messages[-1].content if messages else full_answer
            
            # ── Sources ───────────────────────────────────────────────
            # Citations are inline in the agent's response.

        except Exception as exc:
            thinking_placeholder.empty()
            final_answer = f"⚠️ Something went wrong: {exc}"
            st.error(final_answer)

        # ── Persist assistant message (always) ────────────────────────
        st.session_state.messages.append({"role": "assistant", "content": final_answer})


if __name__ == "__main__":
    main()
