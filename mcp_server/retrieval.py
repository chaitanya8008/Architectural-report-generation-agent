"""
Retrieval engine for the AcoustiQ MCP server.

Provides hybrid search (dense + sparse + RRF fusion) with Cohere reranking
over Qdrant vector store. This is the single source of truth for all
retrieval logic used by both the MCP server and the agent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Optional Dependencies
# ============================================================================
try:
    import cohere

    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client import models as qmodels

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

try:
    from fastembed import SparseTextEmbedding

    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    HAS_DENSE = True
except ImportError:
    HAS_DENSE = False


# ============================================================================
# Hit Normalization
# ============================================================================

def _parse_metadata_value(value: Any) -> Any:
    """Parse JSON strings in metadata values."""
    if isinstance(value, str):
        s = value.strip()
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("{") and s.endswith("}")
        ):
            try:
                return json.loads(s)
            except Exception:
                return value
    return value


def normalize_hit(raw: dict[str, Any], score: float) -> dict[str, Any]:
    """Normalize a raw Qdrant payload into a standard retrieval hit format."""
    # Build source_path from source_file_name (the field that actually exists)
    source_file = str(raw.get("source_file_name", ""))
    sheet_number = str(raw.get("sheet_number", ""))
    source_path = source_file
    if sheet_number and sheet_number not in source_path:
        source_path = f"{source_file} :: {sheet_number}"

    # Build section_title from available fields
    sheet_title = str(raw.get("sheet_title", ""))
    discipline = str(raw.get("discipline", ""))
    chunk_type = str(raw.get("chunk_type", ""))
    section_parts = []
    if sheet_number:
        section_parts.append(sheet_number)
    if sheet_title:
        section_parts.append(sheet_title)
    elif discipline:
        section_parts.append(discipline)
    if chunk_type:
        section_parts.append(f"[{chunk_type}]")
    section_title = " | ".join(section_parts) if section_parts else "Unknown"

    # Extract text fields
    text = str(raw.get("text", ""))
    text_to_embed = str(raw.get("text_to_embed", ""))
    original_content = str(raw.get("original_content", ""))
    contextual_summary = str(raw.get("contextual_summary", ""))

    if not text_to_embed:
        text_to_embed = text
    if not original_content:
        original_content = text

    result: dict[str, Any] = {
        "chunk_id": str(raw.get("chunk_id", "")),
        "score": float(score),
        "text": text,
        "text_to_embed": text_to_embed,
        "contextual_summary": contextual_summary,
        "original_content": original_content,
        "source_path": source_path,
        "section_title": section_title,
        "source_file_name": source_file,
        "sheet_number": sheet_number,
        "revision_label": str(raw.get("revision_label", "")),
        "discipline": discipline,
        "chunk_type": chunk_type,
        "page_number": raw.get("page_number", 0),
    }

    # Pass through extra metadata
    _known_keys = set(result.keys()) | {"text_to_embed"}
    for key, value in raw.items():
        if key not in _known_keys and value is not None:
            result[key] = value

    return result


# ============================================================================
# Cohere Reranking
# ============================================================================

def _get_full_chunk_text(chunk: dict[str, Any]) -> str:
    """Extract full chunk text for reranking — no compression."""
    contextual_summary = chunk.get("contextual_summary", "")
    original_content = chunk.get("original_content", "")
    text = chunk.get("text", "")
    if contextual_summary and original_content:
        return f"{contextual_summary}\n{original_content}"
    return original_content or text


def rerank_chunks_cohere(
    query: str,
    chunks: list[dict[str, Any]],
    cohere_client: Any,
    top_k: int = 20,
    rerank_cap: int = 60,
) -> list[dict[str, Any]]:
    """Rerank chunks using Cohere cross-encoder model."""
    if not chunks:
        return []

    for attempt in range(3):
        try:
            t_start = time.time()
            chunks_to_rerank = chunks[:rerank_cap]
            print(f"    🎯 Reranking {len(chunks_to_rerank)} candidates...", end="", flush=True, file=sys.stderr)
            
            documents = [_get_full_chunk_text(chunk) for chunk in chunks_to_rerank]

            response = cohere_client.rerank(
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
                model="rerank-english-v3.0",
            )

            reranked = []
            for result in response.results:
                idx = result.index
                if idx < 0 or idx >= len(chunks_to_rerank):
                    continue
                chunk = dict(chunks_to_rerank[idx])
                chunk["rerank_score"] = result.relevance_score
                chunk["score"] = result.relevance_score
                reranked.append(chunk)

            reranked.sort(key=lambda x: x.get("score", 0), reverse=True)
            elapsed = time.time() - t_start
            print(f" done in {elapsed:.2f}s", file=sys.stderr)
            return reranked[:top_k]

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait_time = 60
                print(f"\n  ⚠️  Cohere rate limit reached. Waiting {wait_time}s before retry...", file=sys.stderr)
                time.sleep(wait_time)
                continue
            
            logger.exception("Cohere reranking failed", extra={"query": query})
            raise RuntimeError(f"Cohere reranking failed: {e}") from e
    
    raise RuntimeError("Cohere reranking failed after all retries (rate limit).")


# ============================================================================
# Hybrid Retriever
# ============================================================================

class HybridRetriever:
    """
    Unified hybrid retrieval engine using Qdrant.

    Search flow:
        embed queries (dense + sparse)
        → Qdrant prefetch (dense + sparse, parallel server-side)
        → Qdrant RRF fusion
        → Cohere cross-encoder reranking
        → normalized hits
    """

    def __init__(
        self,
        qdrant_url: str | None = None,
        qdrant_path: str | None = None,
        collection_name: str = "VAVA",
        embedding_model: str = "models/text-embedding-004",
        vertex_location: str = "asia-south1",
        gcp_project_id: str | None = None,
    ):
        self.collection_name = collection_name
        self._init_clients(
            qdrant_url, qdrant_path, embedding_model, vertex_location, gcp_project_id
        )

    def _init_clients(
        self,
        qdrant_url: str | None,
        qdrant_path: str | None,
        embedding_model: str,
        vertex_location: str,
        gcp_project_id: str | None,
    ) -> None:
        """Initialize Qdrant, dense embedder, sparse embedder, and Cohere."""
        if not HAS_QDRANT:
            raise RuntimeError("qdrant-client not installed. Run: pip install qdrant-client")
        if not HAS_FASTEMBED:
            raise RuntimeError("fastembed not installed. Run: pip install fastembed")
        if not HAS_DENSE:
            raise RuntimeError("langchain-google-genai not installed.")

        if qdrant_url:
            self.qdrant = QdrantClient(url=qdrant_url)
            print(f"  ✓ Qdrant client initialized (URL: {qdrant_url})", file=sys.stderr)
        elif qdrant_path:
            self.qdrant = QdrantClient(path=qdrant_path)
            print(f"  ✓ Qdrant client initialized (Path: {qdrant_path})", file=sys.stderr)
        else:
            raise ValueError("Either qdrant_url or qdrant_path must be provided.")

        self.dense_embedder = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            location=vertex_location,
            project=gcp_project_id or os.environ.get("GCP_PROJECT_ID"),
            vertexai=True,
            task_type="retrieval_query",
        )
        print("  ✓ Dense embedder (Vertex AI) initialized", file=sys.stderr)

        self.sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm42-all-minilm-l6-v2-attentions"
        )
        print("  ✓ Sparse embedder (BM42) initialized", file=sys.stderr)

        # Cohere reranker — optional but strongly recommended
        self.cohere_client = None
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        if HAS_COHERE and cohere_api_key:
            try:
                self.cohere_client = cohere.Client(api_key=cohere_api_key)
                print("  ✓ Cohere reranker initialized", file=sys.stderr)
            except Exception as e:
                print(f"  ⚠  Cohere initialization failed: {e}", file=sys.stderr)
        else:
            print("  ⚠  Cohere reranker unavailable — set COHERE_API_KEY to enable", file=sys.stderr)

    def build_qdrant_filter(self, filters: dict[str, Any]) -> Any | None:
        """Convert filter dict → Qdrant Filter object using correct field names."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, list):
                if len(value) == 1:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value[0]))
                    )
                elif len(value) > 1:
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
            elif isinstance(value, bool):
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        if not conditions:
            return None

        return Filter(must=conditions)

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        k: int = 20,
        max_candidates: int = 140,
    ) -> dict[str, Any]:
        """
        Execute hybrid search: Qdrant RRF fusion → Cohere rerank.
        """
        from qdrant_client.models import (
            Prefetch,
            FusionQuery,
            Fusion,
            SparseVector as QSparseVector,
        )

        qdrant_filter = self.build_qdrant_filter(filters or {})

        # Auto-generate BM25 query: strip punctuation from semantic query
        bm25_query = " ".join(re.sub(r'[^a-zA-Z0-9 ]', '', query).split())

        # Over-fetch so Cohere has enough candidates (Increased from k*5 for stability)
        fetch_k = min(k * 5, max_candidates)
        rerank_cap = min(fetch_k, 100)

        # ── Embed both queries ──
        t_embed = time.time()
        dense_vector = self.dense_embedder.embed_query(query)

        sparse_result = list(self.sparse_model.embed([bm25_query]))[0]
        sparse_vector = QSparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist(),
        )

        # ── Qdrant hybrid search (server-side RRF) ──
        t_search = time.time()
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=dense_vector, using="dense", limit=fetch_k, filter=qdrant_filter),
                Prefetch(query=sparse_vector, using="sparse", limit=fetch_k, filter=qdrant_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=fetch_k,
            with_payload=True,
        )
        search_elapsed = time.time() - t_search

        # ── Normalize hits ──
        hits: list[dict[str, Any]] = []
        for point in results.points:
            payload = dict(point.payload)
            hit = normalize_hit(payload, float(point.score))
            hits.append(hit)

        # ── Cohere reranking ──
        if self.cohere_client and hits:
            hits = rerank_chunks_cohere(
                query=query,
                chunks=hits,
                cohere_client=self.cohere_client,
                top_k=k,
                rerank_cap=rerank_cap,
            )
        else:
            hits = hits[:k]

        return {
            "hits": hits,
            "stats": {
                "query": query,
                "bm25_query": bm25_query,
                "k": k,
                "fetch_k": fetch_k,
                "returned": len(hits),
                "qdrant_candidates": len(results.points),
                "filters": filters or {},
                "reranker_used": self.cohere_client is not None,
                "search_elapsed_s": round(search_elapsed, 2),
            },
        }

    def scroll_all(
        self, filters: dict[str, Any], limit: int = 2000
    ) -> list[dict[str, Any]]:
        """Retrieve ALL chunks matching filters (no vector search)."""
        qdrant_filter = self.build_qdrant_filter(filters)

        all_points: list[dict[str, Any]] = []
        offset = None
        while True:
            results, next_offset = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=min(limit - len(all_points), 100),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in results:
                all_points.append(normalize_hit(dict(pt.payload), 1.0))
            if next_offset is None or len(all_points) >= limit:
                break
            offset = next_offset

        return all_points
