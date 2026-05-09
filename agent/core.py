"""
Core types, configuration, schemas, and utilities for the acoustic consulting agent.

This module consolidates:
- agent/config.py
- agent/schemas.py
- agent/retrieval/utils.py
- agent/retrieval/response_parser.py
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, Annotated
import operator


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class AgentConfig:
    """Agent configuration."""

    project_id: str
    collection: str
    google_api_key: str | None
    embedding_model: str
    chat_model: str
    max_total_hits: int
    db_uri: str

def _get_project_root() -> Path:
    """Auto-detect project root directory."""
    current_file = Path(__file__).resolve()
    for parent in [current_file.parent, *current_file.parents]:
        if (parent / "agent").exists() and (parent / "agent").is_dir():
            return parent
    return Path.cwd()


def _load_env_file(base: Path) -> None:
    """Load environment variables from a local .env file if present."""
    env_path = base / ".env"
    if not env_path.exists():
        return

    try:
        with env_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                # ALWAYS override values with those in the .env file.
                os.environ[key] = value
    except Exception:
        # Config loading should remain resilient; missing .env parsing is non-fatal.
        return


def load_config() -> AgentConfig:
    """Load agent configuration from environment variables."""
    base = _get_project_root()

    # Auto-load .env for local development workflows.
    _load_env_file(base)

    # Allow GEMINI_API_KEY to act as a fallback source for GOOGLE_API_KEY.
    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ.setdefault("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))

    return AgentConfig(
        project_id=os.getenv("DAYTON_PROJECT_ID", "VA_1040_025_DAYTON_HOTEL"),

        collection=os.getenv("DAYTON_COLLECTION", "VAVA"),

        google_api_key=os.getenv("GOOGLE_API_KEY"),

        embedding_model=os.getenv(
            "DAYTON_EMBED_MODEL", "models/text-embedding-004"
        ),

        chat_model=os.getenv("DAYTON_CHAT_MODEL", "gemini-2.5-flash"),
    
        max_total_hits=int(os.getenv("DAYTON_MAX_TOTAL_HITS", "140")),
        db_uri=os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:5432/va_agent"),
    )




# ============================================================================
# Response Parsing Utilities
# ============================================================================


def extract_text_from_gemini_response(response_content: Any) -> str:
    """
    Extract text content from various Gemini API response formats.

    Args:
        response_content: The content from response.content

    Returns:
        Extracted text as a string

    Raises:
        ValueError: If text cannot be extracted
    """
    # Case 1: Already a plain string
    if isinstance(response_content, str):
        return response_content.strip()

    # Case 2: List of content blocks (most common Gemini format)
    if isinstance(response_content, list):
        if not response_content:
            raise ValueError("Empty response list")

        # Extract text from all text blocks
        text_parts = []
        for block in response_content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    text_parts.append(block["text"])
                elif "text" in block:
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)

        if text_parts:
            return "\n".join(text_parts).strip()

        # Fallback: try to stringify the first item
        return str(response_content[0]).strip()

    # Case 3: Dict with type/text fields
    if isinstance(response_content, dict):
        if response_content.get("type") == "text" and "text" in response_content:
            return response_content["text"].strip()
        if "text" in response_content:
            return response_content["text"].strip()
        # Fallback: stringify the dict
        return str(response_content).strip()

    # Case 4: Other types - convert to string
    return str(response_content).strip()


def parse_json_from_response(response_content: Any) -> dict[str, Any]:
    """
    Parse JSON from Gemini response, handling various formats.

    Args:
        response_content: The content from response.content

    Returns:
        Parsed JSON as a dict

    Raises:
        ValueError: If JSON cannot be parsed
    """
    text = extract_text_from_gemini_response(response_content)

    # Try to parse as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                json_str = text[start:end].strip()
                return json.loads(json_str)
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                json_str = text[start:end].strip()
                return json.loads(json_str)

        # Re-raise the original error
        raise ValueError(f"Failed to parse JSON from response: {e}") from e
