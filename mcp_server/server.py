"""
AcoustiQ MCP Retrieval Server

Standalone MCP server exposing document retrieval tools over stdio transport.
Used by the AcoustiQ agent (and any MCP-compatible client) to search project
documents with full filter support.

Tools:
    - list_available_filters: Discover what filters exist for a project
    - search_documents: Hybrid semantic + keyword search with filtering
    - get_sheet_contents: Retrieve all chunks for a specific sheet

Usage:
    python mcp_server/server.py          # stdio transport (default)
    python mcp_server/server.py --sse    # SSE transport (persistent server)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ── Load environment ─────────────────────────────────────────────────────────

# Find the project root (.env lives there)
_SERVER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVER_DIR.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)


# ── MCP server instance ─────────────────────────────────────────────────────

mcp = FastMCP(
    "AcoustiQ Retrieval",
    instructions=(
        "Provides hybrid semantic + keyword search over architectural project "
        "documents with rich metadata filtering. Supports AEC (Architecture, "
        "Engineering, Construction) document types including drawings, specs, "
        "schedules, and acoustic reports."
    ),
)


# ── Lazy initialization ─────────────────────────────────────────────────────
# We initialize these on first use so the server starts quickly and so
# we can report errors through MCP rather than crashing at import time.

_retriever = None
_filter_registry = None


def _get_retriever():
    """Lazy-init the HybridRetriever."""
    global _retriever
    if _retriever is not None:
        return _retriever

    from retrieval import HybridRetriever

    qdrant_path = str(_PROJECT_ROOT / "qdrant_data")
    collection_name = os.environ.get("QDRANT_COLLECTION", "VAVA")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "models/text-embedding-004")
    vertex_location = os.environ.get("VERTEX_LOCATION", "asia-south1")
    gcp_project_id = os.environ.get("GCP_PROJECT_ID")

    print(f"[MCP] Initializing retrieval engine...", file=sys.stderr)
    print(f"  Qdrant path: {qdrant_path}", file=sys.stderr)
    print(f"  Collection: {collection_name}", file=sys.stderr)

    _retriever = HybridRetriever(
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
        vertex_location=vertex_location,
        gcp_project_id=gcp_project_id,
    )
    print(f"[MCP] Retrieval engine ready.", file=sys.stderr)
    return _retriever


def _get_filter_registry() -> dict:
    """Load filter registry JSON."""
    global _filter_registry
    if _filter_registry is not None:
        return _filter_registry

    # Try multiple locations
    candidates = [
        _PROJECT_ROOT / "filter_registry.json",
        _PROJECT_ROOT / "tmp_runs" / "complete_run" / "filter_registry.json",
    ]

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                _filter_registry = json.load(f)
            print(f"[MCP] Loaded filter registry from {path}", file=sys.stderr)
            return _filter_registry

    # If no registry file, return a helpful error structure
    _filter_registry = {
        "available_filters": {},
        "total_chunks": 0,
        "error": "No filter_registry.json found. Run push_to_qdrant.py to generate it.",
    }
    return _filter_registry


# ── Validation helpers ───────────────────────────────────────────────────────

def _validate_filter_values(
    field: str, values: list[str], registry: dict
) -> list[str]:
    """Check requested filter values against registry. Return warnings."""
    available = registry.get("available_filters", {})
    valid_values = available.get(field, [])
    if not valid_values:
        return []  # Field not in registry, skip validation

    warnings = []
    for v in values:
        if v not in valid_values:
            warnings.append(
                f"WARNING: '{v}' is not a known value for '{field}'. "
                f"Valid values: {valid_values[:10]}"
            )
    return warnings


def _format_hit(hit: dict, rank: int) -> str:
    """Format a single hit as readable markdown."""
    source = hit.get("source_path", "Unknown")
    section = hit.get("section_title", "Unknown")
    revision = hit.get("revision_label", "")
    score = hit.get("score", 0)
    text = hit.get("text", "")[:800]

    header = f"**[{rank}]** `{source}`"
    if revision:
        header += f" | Rev: `{revision}`"
    header += f" | Score: {score:.3f}"
    header += f"\n**Section:** {section}"

    return f"{header}\n\n{text}"


# ============================================================================
# MCP TOOLS
# ============================================================================


@mcp.tool()
def list_available_filters(project_id: str = "") -> str:
    """
    Returns all filterable fields and their valid values for the loaded project.

    Call this FIRST before search_documents to understand what filters you can
    apply. This helps you narrow searches to specific disciplines, revision
    stages, chunk types, etc.

    Args:
        project_id: Optional project ID to scope (usually auto-detected)

    Returns:
        JSON listing all filterable fields with their available values
    """
    registry = _get_filter_registry()

    if "error" in registry:
        return json.dumps(registry, indent=2)

    # Format for readability
    output = {
        "total_chunks": registry.get("total_chunks", 0),
        "generated_at": registry.get("generated_at", "unknown"),
        "filterable_fields": {},
    }

    available = registry.get("available_filters", {})
    for field, values in available.items():
        if isinstance(values, list) and len(values) <= 50:
            output["filterable_fields"][field] = values
        elif isinstance(values, list):
            output["filterable_fields"][field] = {
                "count": len(values),
                "sample": values[:20],
                "note": f"{len(values)} total values — showing first 20",
            }
        else:
            output["filterable_fields"][field] = values

    return json.dumps(output, indent=2, ensure_ascii=False)


@mcp.tool()
def search_documents(
    query: str,
    revision_label: str = "",
    discipline: str = "",
    chunk_type: str = "",
    sheet_number: str = "",
    entity_type: str = "",
    rating_type: str = "",
    is_current_revision: bool = False,
    page_class: str = "",
    limit: int = 20,
) -> str:
    """
    Hybrid semantic + keyword search over project documents.

    Combines dense embeddings (Google Vertex AI) with sparse BM42 keyword
    matching via Reciprocal Rank Fusion, followed by Cohere cross-encoder
    reranking for maximum precision.

    IMPORTANT: Filter values must match exactly what list_available_filters
    reports. Multiple values for the same filter should be comma-separated.

    Args:
        query: Natural language search query (required)
        revision_label: Filter by design stage. Comma-separated for multiple.
            Examples: "25% CD", "50% CD,75% CD"
        discipline: Filter by discipline. Comma-separated for multiple.
            Examples: "architectural", "mechanical,electrical"
        chunk_type: Filter by chunk type. Comma-separated for multiple.
            Examples: "acoustic_rating", "entity,table_row"
        sheet_number: Filter by specific sheet. Example: "A2.00"
        entity_type: Filter by entity type. Example: "room"
        rating_type: Filter by acoustic rating type. Example: "STC"
        is_current_revision: If true, only show latest revision chunks
        page_class: Filter by page class. Example: "drawing", "schedule"
        limit: Max results to return (default 20, max 60)

    Returns:
        Formatted search results with source citations and metadata
    """
    if not query or not query.strip():
        return "ERROR: query parameter is required and cannot be empty."

    limit = max(1, min(limit, 60))

    # Build filters dict from parameters
    filters: dict = {}
    warnings: list[str] = []
    registry = _get_filter_registry()

    # Parse comma-separated list filters
    def _parse_list(value: str, field_name: str) -> list[str] | None:
        if not value or not value.strip():
            return None
        items = [v.strip() for v in value.split(",") if v.strip()]
        if items:
            w = _validate_filter_values(field_name, items, registry)
            warnings.extend(w)
            return items
        return None

    rl = _parse_list(revision_label, "revision_label")
    if rl:
        filters["revision_label"] = rl

    disc = _parse_list(discipline, "discipline")
    if disc:
        filters["discipline"] = disc

    ct = _parse_list(chunk_type, "chunk_type")
    if ct:
        filters["chunk_type"] = ct

    if sheet_number and sheet_number.strip():
        filters["sheet_number"] = sheet_number.strip()

    if entity_type and entity_type.strip():
        filters["entity_type"] = entity_type.strip()

    if rating_type and rating_type.strip():
        filters["rating_type"] = rating_type.strip()

    if is_current_revision:
        filters["is_current_revision"] = True

    pc = _parse_list(page_class, "page_class")
    if pc:
        filters["page_class"] = pc

    # Execute search
    try:
        retriever = _get_retriever()
        result = retriever.search(query=query, filters=filters, k=limit)
    except Exception as e:
        return f"ERROR: Search failed: {e}"

    hits = result["hits"]
    stats = result["stats"]

    # Format output
    parts = []

    if warnings:
        parts.append("### ⚠️ Warnings\n" + "\n".join(f"- {w}" for w in warnings))

    parts.append(
        f"### Search Results ({len(hits)} hits)\n"
        f"**Query:** {query}\n"
        f"**Filters:** {json.dumps(filters) if filters else 'none'}\n"
        f"**Reranker:** {'Cohere' if stats.get('reranker_used') else 'disabled'}\n"
    )

    if not hits:
        parts.append("_No results found. Try broadening your search or removing filters._")
    else:
        for i, hit in enumerate(hits, 1):
            parts.append(_format_hit(hit, i))
            parts.append("---")

    return "\n\n".join(parts)


@mcp.tool()
def get_sheet_contents(
    sheet_number: str,
    revision_label: str = "",
) -> str:
    """
    Retrieve ALL chunks for a specific sheet number. Use when the user asks
    'what is on sheet X?' or wants comprehensive sheet-level information.

    This does NOT use vector search — it fetches every chunk from the sheet
    via direct payload filtering, giving complete coverage.

    Args:
        sheet_number: The sheet identifier (e.g. "A2.00", "M1.01")
        revision_label: Optional revision filter (e.g. "25% CD").
            If empty, returns chunks from all revisions.

    Returns:
        All chunks from the specified sheet, grouped by type
    """
    if not sheet_number or not sheet_number.strip():
        return "ERROR: sheet_number is required."

    filters = {"sheet_number": sheet_number.strip()}
    if revision_label and revision_label.strip():
        filters["revision_label"] = revision_label.strip()

    try:
        retriever = _get_retriever()
        chunks = retriever.scroll_all(filters=filters, limit=500)
    except Exception as e:
        return f"ERROR: Failed to retrieve sheet contents: {e}"

    if not chunks:
        return f"No chunks found for sheet {sheet_number}."

    # Group by chunk_type for organized output
    by_type: dict[str, list] = {}
    for chunk in chunks:
        ct = chunk.get("chunk_type", "other")
        by_type.setdefault(ct, []).append(chunk)

    parts = [
        f"## Sheet {sheet_number} Contents",
        f"**Total chunks:** {len(chunks)}",
        f"**Revision filter:** {revision_label or 'all revisions'}",
        f"**Chunk types found:** {', '.join(sorted(by_type.keys()))}",
        "",
    ]

    # Output in logical order: summary first, then entities, then tables, etc.
    type_order = [
        "page_summary", "entity", "drawing_block", "table_block", "table_row",
        "notes_block", "legend_block", "acoustic_rating", "acoustic_assembly",
        "room_acoustic_requirement", "equipment_noise", "keynote",
        "cross_reference", "entities_summary", "cross_refs_summary",
        "merged_context", "rendering_block",
    ]

    seen_types = set()
    for ct in type_order:
        if ct in by_type:
            seen_types.add(ct)
            parts.append(f"### {ct} ({len(by_type[ct])} chunks)")
            for chunk in by_type[ct]:
                text = chunk.get("text", "")[:600]
                rev = chunk.get("revision_label", "")
                disc = chunk.get("discipline", "")
                meta_parts = [x for x in [rev, disc] if x]
                meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
                parts.append(f"- {text}{meta}")
            parts.append("")

    # Any remaining types not in the order list
    for ct, type_chunks in by_type.items():
        if ct not in seen_types:
            parts.append(f"### {ct} ({len(type_chunks)} chunks)")
            for chunk in type_chunks:
                text = chunk.get("text", "")[:600]
                parts.append(f"- {text}")
            parts.append("")

    return "\n".join(parts)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AcoustiQ MCP Retrieval Server")
    parser.add_argument("--sse", action="store_true", help="Use SSE transport instead of stdio")
    args = parser.parse_args()

    if args.sse:
        mcp.run(transport="sse")
    else:
        mcp.run()
