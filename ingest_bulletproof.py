"""
Production PDF drawing-package ingestor.

This ingestor focuses on extracting and storing architectural / engineering
drawing-package PDFs with strong revision metadata, mixed-page support, richer
block extraction, relational storage, and vector chunks that future retrieval
can filter cleanly.
"""

from __future__ import annotations

import atexit
import argparse
import concurrent.futures
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import threading
import time
import uuid as uuid_lib
from collections import Counter, defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

try:
    from pdf2image import convert_from_path, pdfinfo_from_path

    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False
    convert_from_path = None
    pdfinfo_from_path = None
    print("WARNING: pdf2image not installed. Run: pip install pdf2image")

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = Any
    print("WARNING: Pillow not installed. Run: pip install pillow")

load_dotenv(override=True)

try:
    import psycopg

    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False
    print("WARNING: psycopg not installed. Run: pip install psycopg[binary,pool]")

try:
    from google import genai
    from google.genai import types

    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False
    print("WARNING: google-genai not installed. Run: pip install google-genai")

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    HAS_VECTOR = True
except ImportError:
    HAS_VECTOR = False
    print("WARNING: langchain-google-genai not installed.")

try:
    from qdrant_client import QdrantClient
    from qdrant_client import models as qmodels

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    print("WARNING: qdrant-client not installed. Run: pip install qdrant-client")

try:
    from fastembed import SparseTextEmbedding

    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False
    print("WARNING: fastembed not installed. Run: pip install fastembed")

try:
    from pypdf import PdfReader

    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("WARNING: pypdf not installed. Run: pip install pypdf")

HAS_PSUTIL = importlib.util.find_spec("psutil") is not None

PROJECT_CONTEXTS = {
    "DAYTON_HOTEL": (
        "This is the Dayton Hotel, an 8-story hospitality project. "
        "Pay special attention to room labels, wall types, acoustic ratings, "
        "legends, schedules, and cross-sheet references."
    ),
    "MERCY_HOSPITAL": (
        "This is Mercy Hospital, a 200-bed acute care facility. "
        "Pay special attention to room functions, equipment identifiers, "
        "pressure relationships, HVAC labels, and code / legend text."
    ),
}

SUPPORTED_PAGE_CLASSES = {
    "drawing",
    "mixed",
    "schedule",
    "legend",
    "notes",
    "sheet_index",
    "cover_or_rendering",
    "text_heavy",
}

SUPPORTED_BLOCK_TYPES = {
    "drawing_view",
    "detail_view",
    "table",
    "schedule",
    "legend",
    "keynote_block",
    "general_notes",
    "title_block",
    "photo_or_rendering",
    "text_region",
}

SUPPORTED_APPROX_REGIONS = {
    "full_page",
    "top_band",
    "bottom_band",
    "left_band",
    "right_band",
    "upper_left",
    "upper_center",
    "upper_right",
    "center_left",
    "center",
    "center_right",
    "lower_left",
    "lower_center",
    "lower_right",
}
DENSE_VECTOR_DIM = 768

DEFAULT_LLM_MODEL = os.getenv("INGEST_LLM_MODEL", "gemini-2.5-flash")
DEFAULT_FAST_MODEL = os.getenv("INGEST_FAST_MODEL", "gemini-2.5-flash")
DEFAULT_JSON_REPAIR_MODEL = os.getenv("INGEST_JSON_REPAIR_MODEL", DEFAULT_LLM_MODEL)
DEFAULT_EMBEDDING_MODEL = os.getenv("INGEST_EMBEDDING_MODEL", "models/text-embedding-004")
DEFAULT_GENAI_PROVIDER = (os.getenv("INGEST_GENAI_PROVIDER", "vertexai") or "vertexai").strip().lower()
DEFAULT_VERTEX_LOCATION = os.getenv("INGEST_VERTEX_LOCATION", "us-central1")
LLM_RATE_LIMIT_RETRY_COUNT = max(0, int(os.getenv("INGEST_LLM_RATE_LIMIT_RETRY_COUNT", "1")))
LLM_RATE_LIMIT_COOLDOWN_SECONDS = max(0, int(os.getenv("INGEST_LLM_RATE_LIMIT_COOLDOWN_SECONDS", "120")))
DEFAULT_PARALLEL_VISION_MODEL_POOL = (
    os.getenv(
        "INGEST_PARALLEL_VISION_MODEL_POOL",
        "gemini-2.5-flash",
    )
    or "gemini-2.5-flash"
)

SUPPORTED_EXTRACTION_STRATEGIES = {
    "vision_universal",
    "native_only",
}

DEFAULT_EXTRACTION_STRATEGY_BY_CLASS = {
    "drawing": "vision_universal",
    "drawing_pdf": "vision_universal",
    "mixed_pdf": "vision_universal",
    "submittal": "native_only",
    "report": "native_only",
    "specification": "native_only",
    "narrative": "native_only",
    "text_native": "native_only",
    "unknown": "vision_universal",
    "*": "vision_universal",
}

QUARANTINE_DIR_NAME = "_quarantine"

# Keep chunk text small enough for stable embeddings and higher retrieval precision.
CHUNK_TARGET_CHARS = 900
CHUNK_MIN_CHARS = 250
CHUNK_MAX_CHARS = 1400
CHUNK_OVERLAP_CHARS = 150
TINY_CHUNK_MIN_CHARS = 120
ALLOW_TINY_CHUNK_TYPES = {
    "entity",
    "acoustic_rating",
    "cross_reference",
    "keynote",
    "legend_item",
}
MERGE_COMPATIBLE_TYPES = {
    "block_note",
    "general_notes",
    "page_summary",
    "legend_context",
    "entities_summary",
    "cross_refs_summary",
}

# Backward-compatible aliases for helper call sites/tests.
MAX_EMBED_TEXT_CHARS = CHUNK_MAX_CHARS
EMBED_TEXT_OVERLAP_CHARS = CHUNK_OVERLAP_CHARS
MAX_STRUCTURED_PAYLOAD_CHARS = 1800

HIGH_VALUE_ENTITY_TYPES = {
    "room",
    "space",
    "equipment",
    "door",
    "window",
    "wall_assembly",
    "wall",
    "partition",
    "partition_type",
    "wall_type",
    "floor_assembly",
    "ceiling_assembly",
    "column",
    "beam",
    "duct",
    "pipe",
    "fixture",
}

KEYWORD_PAYLOAD_FIELDS = [
    "project_id",
    "revision_id",
    "revision_label",
    "document_id",
    "document_family_id",
    "document_class",
    "chunk_type",
    "block_type",
    "discipline",
    "primary_discipline",
    "drawing_type",
    "sheet_number",
    "entity_type",
    "entity_id",
    "entity_name",
    "page_class",
    "source_file_name",
    "retrieval_scope",
    "extraction_strategy",
    "table_title",
    "target_sheet_number",
    "reference_text",
    "ref_type",
    "rating_type",
    "rating_value",
    "assembly_id",
    "room_id",
    "equipment_id",
    "keynote_number",
    "extraction_source",
]

BOOLEAN_PAYLOAD_FIELDS = [
    "is_current_revision",
    "has_drawings",
    "has_tables",
    "has_legends",
    "has_notes",
    "has_renderings",
    "has_photos",
    "page_contains_multiple_content_types",
    "resolved",
    "quarantined",
    "low_confidence_extraction",
]

INTEGER_PAYLOAD_FIELDS = [
    "revision_sequence",
    "page_number",
    "block_index",
    "row_index",
    "table_index",
]

REVISION_RULES = [
    (re.compile(r"\b25\b.*\bSD\b|\bSD\b.*\b25\b", re.IGNORECASE), "25% SD", 25),
    (re.compile(r"\b25\b.*\bDD\b|\bDD\b.*\b25\b", re.IGNORECASE), "25% DD", 25),
    (re.compile(r"\b25\b.*\bCD\b|\bCD\b.*\b25\b", re.IGNORECASE), "25% CD", 25),
    (re.compile(r"\b50\b.*\bSD\b|\bSD\b.*\b50\b", re.IGNORECASE), "50% SD", 50),
    (re.compile(r"\b50\b.*\bDD\b|\bDD\b.*\b50\b", re.IGNORECASE), "50% DD", 50),
    (re.compile(r"\b50\b.*\bCD\b|\bCD\b.*\b50\b", re.IGNORECASE), "50% CD", 50),
    (re.compile(r"\b75\b.*\bSD\b|\bSD\b.*\b75\b", re.IGNORECASE), "75% SD", 75),
    (re.compile(r"\b75\b.*\bDD\b|\bDD\b.*\b75\b", re.IGNORECASE), "75% DD", 75),
    (re.compile(r"\b75\b.*\bCD\b|\bCD\b.*\b75\b", re.IGNORECASE), "75% CD", 75),
    (re.compile(r"\b100\b.*\bSD\b|\bSD\b.*\b100\b", re.IGNORECASE), "100% SD", 100),
    (re.compile(r"\b100\b.*\bDD\b|\bDD\b.*\b100\b", re.IGNORECASE), "100% DD", 100),
    (re.compile(r"\b100\b.*\bCD\b|\bCD\b.*\b100\b", re.IGNORECASE), "100% CD", 100),
    (re.compile(r"\bIFC\b", re.IGNORECASE), "IFC", 1000),
    (re.compile(r"\bPERMIT\b", re.IGNORECASE), "PERMIT", 900),
    (re.compile(r"\bBID\b", re.IGNORECASE), "BID", 850),
    (re.compile(r"\bBOD\b", re.IGNORECASE), "BOD", 10),
    (re.compile(r"\bDD\b", re.IGNORECASE), "DD", 500),
    (re.compile(r"\bCD\b", re.IGNORECASE), "CD", 600),
    (re.compile(r"\bSD\b", re.IGNORECASE), "SD", 400),
]

UNIVERSAL_EXTRACTION_PROMPT = """\
You are an expert at reading architectural, engineering, and construction
(AEC) package pages, including drawings, schedules, legends, notes, cut sheets,
specification-style pages, cover pages, and renderings.

Analyze this single PDF page image and extract everything you can read.

Return ONLY valid JSON with exactly this shape:

{
  "header": {
    "sheet_number": "sheet identifier",
    "sheet_title": "full sheet title",
    "discipline": "architectural | structural | mechanical | electrical | plumbing | civil | landscape | fire_protection | interior | other",
    "drawing_type": "short free-text label",
    "scale": "drawing scale if visible",
    "title_block_revision": "revision from title block if visible",
    "title_block_date": "date from title block if visible"
  },
    "page_class": "drawing | mixed | schedule | legend | notes | sheet_index | cover_or_rendering | text_heavy",
  "page_summary": "3-10 paragraphs of exhaustive prose describing the page",
  "content_blocks": [
    {
      "block_index": 0,
      "block_type": "drawing_view | detail_view | table | schedule | legend | keynote_block | general_notes | title_block | photo_or_rendering | text_region",
      "block_label": "short descriptive label",
      "approx_region": "one of: full_page, top_band, bottom_band, left_band, right_band, upper_left, upper_center, upper_right, center_left, center, center_right, lower_left, lower_center, lower_right",
    "bbox_norm": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
      "text": "all readable text or descriptive summary for this block",
      "structured_payload": {},
      "extraction_source": "vision",
      "confidence": 0.0
    }
  ],
  "entities": [
    {
      "entity_type": "free-text entity type",
      "entity_id": "identifier or mark",
      "entity_name": "descriptive name",
      "attributes": {},
      "location_on_drawing": "brief location text",
      "page_block_index": 0
    }
  ],
  "cross_references": [
    {
      "reference_text": "3/A7.01",
      "type": "detail_callout | section_cut | schedule_ref | spec_ref | sheet_ref | note_ref",
      "context": "what this reference is about",
      "target_sheet_number": "best normalized target if possible",
      "page_block_index": 0
    }
  ],
  "tables_on_sheet": [
    {
      "table_title": "Door Schedule",
      "columns": ["COL1", "COL2"],
      "rows": [{"COL1": "value", "COL2": "value"}],
      "notes": "notes text",
      "page_block_index": 0
    }
  ],
  "keynotes_or_legends": [
    {
      "number": "1",
      "text": "legend or keynote text",
      "page_block_index": 0,
      "block_type": "legend"
    }
  ],
  "acoustic_data": {
    "ratings": [
      {
        "rating_type": "STC | IIC | NIC | NRC | SAA | CAC | RC | NC | NCB | RT60 | dBA | sones | sound_power | other",
        "value": "rating value",
        "unit": "unit if applicable",
        "source_text": "exact visible text",
        "applies_to": "wall, floor, ceiling, room, equipment, door, window, detail, assembly, other",
        "assembly_id": "wall/floor/ceiling/door/window type if visible",
        "room_id": "room number/name if visible",
        "equipment_id": "equipment tag if visible",
        "location_on_drawing": "where it appears",
        "page_block_index": 0,
        "confidence": 0.0
      }
    ],
    "assemblies": [
      {
        "assembly_id": "partition/wall/floor/ceiling type",
        "assembly_type": "wall | floor | ceiling | roof | door | window | other",
        "description": "visible description",
        "layers": ["layer descriptions if visible"],
        "rated_stc": "value if visible",
        "rated_iic": "value if visible",
        "rated_nrc": "value if visible",
        "fire_rating": "value if visible",
        "source_text": "exact text",
        "page_block_index": 0,
        "confidence": 0.0
      }
    ],
    "equipment_noise": [
      {
        "equipment_id": "equipment tag",
        "equipment_type": "AHU, RTU, fan, pump, generator, diffuser, grille, etc.",
        "noise_metric": "dBA | NC | RC | sones | sound_power | octave_band",
        "value": "visible value",
        "location": "room/roof/plan location",
        "source_text": "exact visible text",
        "page_block_index": 0,
        "confidence": 0.0
      }
    ],
    "room_acoustic_requirements": [
      {
        "room_id": "room number",
        "room_name": "room name",
        "requirement_type": "NC | RC | RT60 | STC adjacency | privacy | vibration | other",
        "requirement_value": "visible requirement",
        "source_text": "exact visible text",
        "page_block_index": 0,
        "confidence": 0.0
      }
    ]
  },
  "page_flags": {
    "has_drawings": true,
    "has_tables": false,
    "has_legends": false,
    "has_notes": false,
    "has_renderings": false,
    "page_contains_multiple_content_types": false
  }
}

Rules:
1. Be exhaustive. Capture drawings, schedules, legends, notes, title block
   metadata, cross-references, and image/rendering content.
2. A single page may contain multiple content types. Emit separate blocks.
3. If a value is unreadable, use ILLEGIBLE. If partial, use PARTIAL: ...
4. Do not hallucinate. If something is not visible, leave it empty or ILLEGIBLE.
5. Use the block most closely associated with each entity / table / legend
   when filling page_block_index.
6. Confidence must be between 0 and 1.
7. Include bbox_norm when you can estimate a tight block boundary. If unsure,
    keep approx_region and omit bbox_norm.
8. Acoustic engineering priority: extract all visible STC, IIC, NIC, NRC,
   SAA, CAC, NC, RC, NCB, RT60, dBA, sones, sound power levels,
   octave-band data, vibration criteria, wall/floor/ceiling assembly ratings,
   door/window acoustic ratings, equipment noise values, and room acoustic
   criteria. Do not infer values. Only extract visible data.
"""

VERIFICATION_PROMPT = """\
Look at the page again and compare it against the extraction summary supplied in
the user content. Check for missing drawings, tables, legends, note blocks,
title block details, cross-references, or identifiers. Return ONLY valid JSON:

{
  "missing_items": [
    {
      "type": "entity | table | legend | notes | title_block | cross_reference | block",
      "text": "what was missed",
      "location": "where it appears",
      "page_block_index": 0
    }
  ],
  "extraction_complete": true
}
"""

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    project_name TEXT,
    project_number TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_revisions (
    revision_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id),
    revision_label TEXT NOT NULL,
    revision_sequence INTEGER NOT NULL,
    revision_source TEXT NOT NULL,
    revision_date DATE,
    revision_confidence DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE(project_id, revision_source)
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id),
    revision_id TEXT REFERENCES project_revisions(revision_id),
    document_family_id TEXT NOT NULL,
    document_class TEXT NOT NULL,
    source_relative_path TEXT NOT NULL,
    source_file_name TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    issue_date DATE,
    page_count INTEGER,
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE(project_id, revision_id, source_relative_path)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id),
    started_at TIMESTAMP DEFAULT now(),
    completed_at TIMESTAMP,
    status TEXT NOT NULL,
    stats JSONB
);

CREATE TABLE IF NOT EXISTS sheets (
    id SERIAL PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id),
    revision_id TEXT REFERENCES project_revisions(revision_id),
    document_id TEXT REFERENCES documents(document_id),
    document_family_id TEXT NOT NULL,
    source_relative_path TEXT NOT NULL,
    source_file_name TEXT NOT NULL,
    document_class TEXT NOT NULL,
    sheet_number TEXT NOT NULL,
    sheet_title TEXT,
    discipline TEXT,
    drawing_type TEXT,
    scale TEXT,
    page_number INTEGER,
    page_class TEXT,
    image_path TEXT,
    page_summary TEXT,
    title_block_revision TEXT,
    title_block_date TEXT,
    has_drawings BOOLEAN DEFAULT FALSE,
    has_tables BOOLEAN DEFAULT FALSE,
    has_legends BOOLEAN DEFAULT FALSE,
    has_notes BOOLEAN DEFAULT FALSE,
    has_renderings BOOLEAN DEFAULT FALSE,
    page_contains_multiple_content_types BOOLEAN DEFAULT FALSE,
    raw_extraction JSONB,
    UNIQUE(document_id, page_number)
);

CREATE TABLE IF NOT EXISTS page_blocks (
    id SERIAL PRIMARY KEY,
    sheet_id INTEGER REFERENCES sheets(id) ON DELETE CASCADE,
    block_index INTEGER NOT NULL,
    block_type TEXT NOT NULL,
    block_label TEXT,
    text TEXT,
    structured_payload JSONB,
    approx_region TEXT,
    crop_image_path TEXT,
    extraction_source TEXT,
    confidence DOUBLE PRECISION,
    UNIQUE(sheet_id, block_index)
);

CREATE TABLE IF NOT EXISTS media_assets (
    id SERIAL PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id),
    revision_id TEXT REFERENCES project_revisions(revision_id),
    document_id TEXT REFERENCES documents(document_id),
    sheet_id INTEGER REFERENCES sheets(id) ON DELETE CASCADE,
    page_block_id INTEGER REFERENCES page_blocks(id) ON DELETE CASCADE,
    asset_type TEXT NOT NULL,
    asset_path TEXT NOT NULL,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    sheet_id INTEGER REFERENCES sheets(id) ON DELETE CASCADE,
    page_block_id INTEGER REFERENCES page_blocks(id) ON DELETE SET NULL,
    project_id TEXT REFERENCES projects(project_id),
    revision_id TEXT REFERENCES project_revisions(revision_id),
    document_id TEXT REFERENCES documents(document_id),
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    entity_name TEXT,
    attributes JSONB NOT NULL,
    location_on_drawing TEXT,
    UNIQUE(sheet_id, page_block_id, entity_type, entity_id)
);

CREATE TABLE IF NOT EXISTS cross_references (
    id SERIAL PRIMARY KEY,
    sheet_id INTEGER REFERENCES sheets(id) ON DELETE CASCADE,
    page_block_id INTEGER REFERENCES page_blocks(id) ON DELETE SET NULL,
    project_id TEXT REFERENCES projects(project_id),
    revision_id TEXT REFERENCES project_revisions(revision_id),
    document_id TEXT REFERENCES documents(document_id),
    reference_text TEXT NOT NULL,
    ref_type TEXT,
    context TEXT,
    target_sheet_number TEXT,
    resolved BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS tables_on_sheet (
    id SERIAL PRIMARY KEY,
    sheet_id INTEGER REFERENCES sheets(id) ON DELETE CASCADE,
    page_block_id INTEGER REFERENCES page_blocks(id) ON DELETE SET NULL,
    table_title TEXT,
    columns JSONB,
    rows JSONB,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS keynotes (
    id SERIAL PRIMARY KEY,
    sheet_id INTEGER REFERENCES sheets(id) ON DELETE CASCADE,
    page_block_id INTEGER REFERENCES page_blocks(id) ON DELETE SET NULL,
    keynote_number TEXT,
    keynote_text TEXT,
    block_type TEXT
);

CREATE TABLE IF NOT EXISTS acoustic_facts (
    id SERIAL PRIMARY KEY,
    sheet_id INTEGER REFERENCES sheets(id) ON DELETE CASCADE,
    page_block_id INTEGER REFERENCES page_blocks(id) ON DELETE SET NULL,
    project_id TEXT REFERENCES projects(project_id),
    revision_id TEXT REFERENCES project_revisions(revision_id),
    document_id TEXT REFERENCES documents(document_id),
    fact_type TEXT NOT NULL,
    subject_id TEXT,
    subject_type TEXT,
    metric TEXT,
    value TEXT,
    unit TEXT,
    source_text TEXT,
    location_on_drawing TEXT,
    confidence DOUBLE PRECISION,
    attributes JSONB
);

ALTER TABLE project_revisions ADD COLUMN IF NOT EXISTS revision_date DATE;
ALTER TABLE project_revisions ADD COLUMN IF NOT EXISTS revision_confidence DOUBLE PRECISION;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS issue_date DATE;

CREATE INDEX IF NOT EXISTS idx_revisions_project ON project_revisions(project_id);
CREATE INDEX IF NOT EXISTS idx_documents_project_revision ON documents(project_id, revision_id);
CREATE INDEX IF NOT EXISTS idx_sheets_doc_page ON sheets(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_sheets_revision ON sheets(revision_id);
CREATE INDEX IF NOT EXISTS idx_sheets_page_class ON sheets(page_class);
CREATE INDEX IF NOT EXISTS idx_blocks_sheet ON page_blocks(sheet_id);
CREATE INDEX IF NOT EXISTS idx_blocks_type ON page_blocks(block_type);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_project ON entities(project_id);
CREATE INDEX IF NOT EXISTS idx_entities_attrs ON entities USING gin(attributes);
CREATE INDEX IF NOT EXISTS idx_cross_refs_target ON cross_references(target_sheet_number);
CREATE INDEX IF NOT EXISTS idx_cross_refs_revision ON cross_references(revision_id);
CREATE INDEX IF NOT EXISTS idx_tables_sheet ON tables_on_sheet(sheet_id);
CREATE INDEX IF NOT EXISTS idx_acoustic_facts_project ON acoustic_facts(project_id);
CREATE INDEX IF NOT EXISTS idx_acoustic_facts_revision ON acoustic_facts(revision_id);
CREATE INDEX IF NOT EXISTS idx_acoustic_facts_metric ON acoustic_facts(metric);
CREATE INDEX IF NOT EXISTS idx_acoustic_facts_subject ON acoustic_facts(subject_id);
CREATE INDEX IF NOT EXISTS idx_acoustic_facts_attrs ON acoustic_facts USING gin(attributes);
"""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return normalized.lower() or "unknown"


def safe_json_loads(
    value: str,
    default: Any,
    *,
    debug_label: str = "",
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Any:
    cleaned = (value or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception as exc:
        if diagnostics is not None:
            diagnostics["parse_fallback"] = True
            diagnostics["parse_error"] = str(exc)
        if cleaned:
            preview = normalize_space(cleaned)[:180]
            label = normalize_space(debug_label)
            label_suffix = f" [{label}]" if label else ""
            print(
                f"WARNING: failed to parse JSON response; using fallback{label_suffix}. "
                f"Error: {exc}; preview: {preview}"
            )
        return default


def write_quarantine_record(
    output_path: Path,
    document_meta: dict,
    page_num: int,
    reason: str,
    raw_response: str = "",
    exception: str = "",
    image_path: str = "",
    model: str = "",
    prompt_name: str = "",
) -> str:
    payload = {
        "project_id": document_meta.get("project_id", ""),
        "revision_id": document_meta.get("revision_id", ""),
        "document_id": document_meta.get("document_id", ""),
        "source_relative_path": document_meta.get("source_relative_path", ""),
        "source_file_name": document_meta.get("source_file_name", ""),
        "page_number": page_num,
        "reason": reason,
        "raw_response": (raw_response or "")[:20000],
        "exception": exception,
        "image_path": image_path,
        "model": model,
        "prompt_name": prompt_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = (
        output_path
        / QUARANTINE_DIR_NAME
        / normalize_space(str(document_meta.get("document_id", "")) or "unknown_document")
        / f"page_{int(page_num):04d}.json"
    )
    write_json(path, payload)
    return normalize_storage_path(path)


def try_parse_llm_json(text: str, debug_label: str = "") -> Tuple[Optional[dict], dict]:
    diagnostics: Dict[str, Any] = {}
    parsed = safe_json_loads(
        text,
        None,
        debug_label=debug_label,
        diagnostics=diagnostics,
    )
    if parsed is None or not isinstance(parsed, dict):
        return None, {
            "parse_fallback": 1,
            "parse_error": normalize_space(str(diagnostics.get("parse_error", ""))),
        }
    return parsed, {"parse_fallback": 0, "parse_error": ""}


def parse_llm_json_or_quarantine(
    text: str,
    output_path: Optional[Path],
    document_meta: Optional[dict],
    page_num: int,
    image_path: str,
    model: str,
    prompt_name: str,
    debug_label: str = "",
) -> Tuple[Optional[dict], dict]:
    parsed, diagnostics = try_parse_llm_json(text, debug_label=debug_label)
    if parsed is not None:
        return parsed, {"parse_fallback": 0, "quarantined": False, "parse_error": ""}

    quarantine_path = ""
    if output_path is not None and document_meta:
        quarantine_path = write_quarantine_record(
            output_path=output_path,
            document_meta=document_meta,
            page_num=page_num,
            reason="json_parse_failed",
            raw_response=text,
            image_path=image_path,
            model=model,
            prompt_name=prompt_name,
        )

    return None, {
        "parse_fallback": 1,
        "quarantined": bool(quarantine_path),
        "quarantine_path": quarantine_path,
        "parse_error": diagnostics.get("parse_error", ""),
    }


class TeeStream:
    def __init__(self, primary: Any, mirror: Any):
        self.primary = primary
        self.mirror = mirror

    def write(self, data: Any) -> int:
        text = data if isinstance(data, str) else str(data)
        primary_text = text
        try:
            self.primary.write(primary_text)
        except UnicodeEncodeError:
            encoding = getattr(self.primary, "encoding", None) or "utf-8"
            primary_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
            self.primary.write(primary_text)
        self.mirror.write(text)
        return len(primary_text)

    def flush(self) -> None:
        self.primary.flush()
        self.mirror.flush()

    def isatty(self) -> bool:
        if hasattr(self.primary, "isatty"):
            try:
                return bool(self.primary.isatty())
            except Exception:
                return False
        return False

    def fileno(self) -> int:
        if hasattr(self.primary, "fileno"):
            return self.primary.fileno()
        raise OSError("Stream does not expose file descriptor")

    @property
    def encoding(self) -> str:
        return getattr(self.primary, "encoding", "utf-8")


def enable_terminal_log_capture(log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)

    def _cleanup() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        try:
            if sys.stdout is not original_stdout:
                sys.stdout = original_stdout
            if sys.stderr is not original_stderr:
                sys.stderr = original_stderr
        except Exception:
            pass

        try:
            log_handle.close()
        except Exception:
            pass

    atexit.register(_cleanup)
    return normalize_storage_path(log_path)


def is_rate_limited_error(exc: Exception) -> bool:
    message = normalize_space(str(exc)).lower()
    markers = ["429", "resource_exhausted", "rate limit", "quota"]
    return any(marker in message for marker in markers)


def is_unsupported_model_error(exc: Exception) -> bool:
    message = normalize_space(str(exc)).lower()
    markers = [
        "404",
        "not found",
        "unsupported model",
        "unknown model",
        "invalid model",
        "model is not supported",
    ]
    return any(marker in message for marker in markers)


def parse_parallel_model_pool(raw_value: str, fallback_model: str = DEFAULT_LLM_MODEL) -> List[str]:
    models = [normalize_space(part) for part in (raw_value or "").split(",")]
    models = [model for model in models if model]
    if not models:
        return [normalize_space(fallback_model) or DEFAULT_LLM_MODEL]
    seen = set()
    ordered: List[str] = []
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        ordered.append(model)
    return ordered


def resolve_effective_verify_mode(verify_mode: str, parallel_vision: bool) -> str:
    return "off" if parallel_vision else verify_mode


def log_parallel_event(message: str) -> None:
    tqdm.write(message)


def image_to_jpeg_bytes(image: object, quality: int = 95) -> bytes:
    image_bytes_buffer = io.BytesIO()
    image.save(image_bytes_buffer, format="JPEG", quality=quality)
    return image_bytes_buffer.getvalue()


def read_binary_file(path: Path) -> bytes:
    return path.read_bytes()


def get_process_rss_bytes() -> int:
    if not HAS_PSUTIL:
        return 0
    try:
        psutil_module = importlib.import_module("psutil")
        return int(psutil_module.Process(os.getpid()).memory_info().rss)
    except Exception:
        return 0


def close_image(image: Any) -> None:
    if image is None:
        return
    close_fn = getattr(image, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def estimate_image_raw_bytes(image: Any) -> int:
    if image is None:
        return 0
    try:
        width, height = image.size
        band_count = len(image.getbands())
        return max(0, int(width)) * max(0, int(height)) * max(1, int(band_count))
    except Exception:
        return 0


class InFlightPageTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._active_by_stage: Dict[str, int] = defaultdict(int)
        self._max_active_by_stage: Dict[str, int] = defaultdict(int)
        self._active_bytes = 0
        self._max_active_bytes = 0

    def enter(self, stage_name: str, approx_bytes: int) -> Tuple[str, int]:
        with self._lock:
            self._active_by_stage[stage_name] += 1
            self._max_active_by_stage[stage_name] = max(
                self._max_active_by_stage.get(stage_name, 0),
                self._active_by_stage[stage_name],
            )
            self._active_bytes += max(0, int(approx_bytes))
            self._max_active_bytes = max(self._max_active_bytes, self._active_bytes)
        return stage_name, max(0, int(approx_bytes))

    def exit(self, token: Tuple[str, int]) -> None:
        stage_name, approx_bytes = token
        with self._lock:
            if self._active_by_stage.get(stage_name, 0) > 0:
                self._active_by_stage[stage_name] -= 1
            self._active_bytes = max(0, self._active_bytes - max(0, int(approx_bytes)))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "max_workers_by_stage": {key: self._max_active_by_stage[key] for key in sorted(self._max_active_by_stage)},
                "max_inflight_page_bytes": self._max_active_bytes,
            }


class RoundRobinPageScheduler:
    def __init__(self, doc_states: List[dict]):
        self._queue: deque[dict] = deque(
            state for state in doc_states if int(state.get("page_count", 0) or 0) > 0 and int(state.get("next_page", 1) or 1) <= int(state.get("page_count", 0) or 0)
        )

    def has_pending(self) -> bool:
        return bool(self._queue)

    def next_submission(self) -> Optional[Tuple[dict, int]]:
        if not self._queue:
            return None
        state = self._queue.popleft()
        page_num = int(state.get("next_page", 1) or 1)
        state["next_page"] = page_num + 1
        if int(state.get("next_page", 0) or 0) <= int(state.get("page_count", 0) or 0):
            self._queue.append(state)
        return state, page_num


class VisionModelPool:
    def __init__(
        self,
        models: List[str],
        cooldown_seconds: int,
        *,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.models = [model for model in models if normalize_space(model)]
        self.cooldown_seconds = max(0, int(cooldown_seconds))
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._lock = threading.Lock()
        self._cooldown_until: Dict[str, float] = {}
        self._disabled_models: set[str] = set()
        self._stats = {
            "retry_count": 0,
            "failover_count": 0,
            "cooldown_wait_count": 0,
            "cooldown_wait_seconds": 0.0,
            "per_model": {
                model: {
                    "attempts": 0,
                    "successes": 0,
                    "rate_limit_errors": 0,
                    "cooldowns_started": 0,
                    "hard_disables": 0,
                }
                for model in self.models
            },
        }

    def _ensure_model_stats(self, model: str) -> dict:
        if model not in self._stats["per_model"]:
            self._stats["per_model"][model] = {
                "attempts": 0,
                "successes": 0,
                "rate_limit_errors": 0,
                "cooldowns_started": 0,
                "hard_disables": 0,
            }
        return self._stats["per_model"][model]

    def _available_models(self) -> List[str]:
        now = self._time_fn()
        ordered: List[str] = []
        for model in self.models:
            if model in self._disabled_models:
                continue
            if self._cooldown_until.get(model, 0.0) > now:
                continue
            ordered.append(model)
        return ordered

    def _next_wait_seconds(self) -> float:
        now = self._time_fn()
        waits = [
            max(0.0, until - now)
            for model, until in self._cooldown_until.items()
            if model not in self._disabled_models and until > now
        ]
        return min(waits) if waits else 0.0

    def call_generate_content(
        self,
        client: Any,
        *,
        contents: List[Any],
        config: Any,
        operation_name: str,
        page_label: str = "",
    ) -> Tuple[Any, dict]:
        if not self.models:
            raise RuntimeError("Vision model pool is empty")

        attempt_number = 0
        failovers_for_request = 0
        while True:
            with self._lock:
                ordered_models = self._available_models()
                wait_seconds = self._next_wait_seconds() if not ordered_models else 0.0

            if not ordered_models:
                with self._lock:
                    self._stats["cooldown_wait_count"] += 1
                    self._stats["cooldown_wait_seconds"] += float(wait_seconds)
                detail = f"{operation_name} {page_label}".strip()
                log_parallel_event(
                    f"[MODEL_POOL] all models cooling down for {detail}; waiting {round(wait_seconds, 2)}s"
                )
                if wait_seconds > 0:
                    self._sleep_fn(wait_seconds)
                continue

            for model in ordered_models:
                attempt_number += 1
                with self._lock:
                    self._ensure_model_stats(model)["attempts"] += 1
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    with self._lock:
                        self._ensure_model_stats(model)["successes"] += 1
                    if failovers_for_request > 0:
                        detail = f"{operation_name} {page_label}".strip()
                        log_parallel_event(
                            f"[MODEL_POOL] recovered {detail} on model={model} after failovers={failovers_for_request}"
                        )
                    return response, {
                        "model": model,
                        "attempts": attempt_number,
                        "failovers": failovers_for_request,
                    }
                except Exception as exc:
                    if is_rate_limited_error(exc):
                        with self._lock:
                            self._ensure_model_stats(model)["rate_limit_errors"] += 1
                            self._ensure_model_stats(model)["cooldowns_started"] += 1
                            self._cooldown_until[model] = self._time_fn() + self.cooldown_seconds
                            self._stats["retry_count"] += 1
                            self._stats["failover_count"] += 1
                        failovers_for_request += 1
                        detail = f"{operation_name} {page_label}".strip()
                        log_parallel_event(
                            f"[MODEL_POOL] rate limited {detail} on model={model}; cooling down {self.cooldown_seconds}s and failing over"
                        )
                        continue
                    if is_unsupported_model_error(exc):
                        with self._lock:
                            self._disabled_models.add(model)
                            self._ensure_model_stats(model)["hard_disables"] += 1
                            self._stats["retry_count"] += 1
                            self._stats["failover_count"] += 1
                        failovers_for_request += 1
                        detail = f"{operation_name} {page_label}".strip()
                        log_parallel_event(
                            f"[MODEL_POOL] disabling unsupported model={model} during {detail}: {exc}"
                        )
                        continue
                    raise

    def snapshot(self) -> dict:
        with self._lock:
            per_model = {
                model: dict(self._stats["per_model"].get(model, {}))
                for model in sorted(self._stats["per_model"])
            }
            return {
                "models": list(self.models),
                "disabled_models": sorted(self._disabled_models),
                "retry_count": self._stats["retry_count"],
                "failover_count": self._stats["failover_count"],
                "cooldown_wait_count": self._stats["cooldown_wait_count"],
                "cooldown_wait_seconds": round(float(self._stats["cooldown_wait_seconds"]), 2),
                "per_model": per_model,
            }


class PageImageCacheManager:
    def __init__(self, cache_root: Path, rerender_invalid_cache: bool = False):
        self.cache_root = cache_root
        self.rerender_invalid_cache = bool(rerender_invalid_cache)
        self._lock = threading.Lock()
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "invalid_cache_entries": 0,
            "trusted_existing_cache_entries": 0,
            "rerenders": 0,
            "pages_rendered": 0,
            "render_failures": 0,
            "bytes_written": 0,
            "max_cached_jpeg_bytes": 0,
        }

    def page_image_path(self, manifest: dict, page_num: int) -> Path:
        return self.cache_root / manifest["revision_id"] / manifest["document_id"] / f"page_{page_num:04d}.jpg"

    def page_meta_path(self, manifest: dict, page_num: int) -> Path:
        return self.cache_root / manifest["revision_id"] / manifest["document_id"] / f"page_{page_num:04d}.meta.json"

    def inspect_cached_page(self, manifest: dict, page_num: int, dpi: int) -> dict:
        page_path = self.page_image_path(manifest, page_num)
        meta_path = self.page_meta_path(manifest, page_num)
        if not page_path.exists():
            return {"status": "missing", "page_path": page_path, "meta_path": meta_path}

        expected = {
            "file_sha256": manifest.get("file_sha256", ""),
            "page_number": int(page_num),
            "dpi": int(dpi),
        }
        if meta_path.exists():
            meta = safe_json_loads(meta_path.read_text(encoding="utf-8"), {})
            if (
                isinstance(meta, dict)
                and normalize_space(str(meta.get("file_sha256", ""))) == expected["file_sha256"]
                and int(meta.get("page_number", 0) or 0) == expected["page_number"]
                and int(meta.get("dpi", 0) or 0) == expected["dpi"]
            ):
                return {
                    "status": "hit",
                    "page_path": page_path,
                    "meta_path": meta_path,
                    "file_size": page_path.stat().st_size,
                    "metadata": meta,
                }

        if self.rerender_invalid_cache:
            return {"status": "invalid_rerender", "page_path": page_path, "meta_path": meta_path}
        return {
            "status": "invalid_trusted",
            "page_path": page_path,
            "meta_path": meta_path,
            "file_size": page_path.stat().st_size,
        }

    def _record_cache_stat(self, key: str, increment: int = 1) -> None:
        with self._lock:
            self._stats[key] += increment

    def ensure_page_image(
        self,
        root_path: Path,
        manifest: dict,
        page_num: int,
        dpi: int,
        *,
        inflight_tracker: Optional[InFlightPageTracker] = None,
        record_stats: bool = True,
    ) -> dict:
        inspection = self.inspect_cached_page(manifest, page_num, dpi)
        status = inspection["status"]
        if status == "hit":
            if record_stats:
                self._record_cache_stat("cache_hits")
                with self._lock:
                    self._stats["max_cached_jpeg_bytes"] = max(
                        self._stats["max_cached_jpeg_bytes"],
                        int(inspection.get("file_size", 0) or 0),
                    )
            return inspection
        if status == "invalid_trusted":
            if record_stats:
                self._record_cache_stat("invalid_cache_entries")
                self._record_cache_stat("trusted_existing_cache_entries")
                with self._lock:
                    self._stats["max_cached_jpeg_bytes"] = max(
                        self._stats["max_cached_jpeg_bytes"],
                        int(inspection.get("file_size", 0) or 0),
                    )
            return inspection

        if record_stats:
            self._record_cache_stat("cache_misses")
            if status == "invalid_rerender":
                self._record_cache_stat("invalid_cache_entries")
                self._record_cache_stat("rerenders")

        document_path = root_path / manifest["source_relative_path"]
        tracker_token = None
        page_image = None
        try:
            page_image = convert_from_path(
                str(document_path),
                dpi=dpi,
                first_page=page_num,
                last_page=page_num,
            )[0]
            raw_bytes = estimate_image_raw_bytes(page_image)
            if inflight_tracker:
                tracker_token = inflight_tracker.enter("render", raw_bytes)
            page_path = self.page_image_path(manifest, page_num)
            meta_path = self.page_meta_path(manifest, page_num)
            saved_path = save_page_image(page_image, page_path)
            file_size = page_path.stat().st_size if page_path.exists() else 0
            meta_payload = {
                "file_sha256": manifest.get("file_sha256", ""),
                "page_number": int(page_num),
                "dpi": int(dpi),
                "document_id": manifest.get("document_id", ""),
                "revision_id": manifest.get("revision_id", ""),
                "source_relative_path": manifest.get("source_relative_path", ""),
                "width_px": int(getattr(page_image, "size", (0, 0))[0] or 0),
                "height_px": int(getattr(page_image, "size", (0, 0))[1] or 0),
                "raw_image_bytes": raw_bytes,
                "jpeg_file_bytes": file_size,
            }
            write_json(meta_path, meta_payload)
            if record_stats:
                with self._lock:
                    self._stats["pages_rendered"] += 1
                    self._stats["bytes_written"] += file_size
                    self._stats["max_cached_jpeg_bytes"] = max(self._stats["max_cached_jpeg_bytes"], file_size)
            return {
                "status": "rendered",
                "page_path": Path(saved_path),
                "meta_path": meta_path,
                "file_size": file_size,
                "metadata": meta_payload,
            }
        except Exception:
            if record_stats:
                self._record_cache_stat("render_failures")
            raise
        finally:
            if tracker_token:
                inflight_tracker.exit(tracker_token)
            close_image(page_image)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "cache_hits": self._stats["cache_hits"],
                "cache_misses": self._stats["cache_misses"],
                "invalid_cache_entries": self._stats["invalid_cache_entries"],
                "trusted_existing_cache_entries": self._stats["trusted_existing_cache_entries"],
                "rerenders": self._stats["rerenders"],
                "pages_rendered": self._stats["pages_rendered"],
                "render_failures": self._stats["render_failures"],
                "bytes_written": self._stats["bytes_written"],
                "max_cached_jpeg_bytes": self._stats["max_cached_jpeg_bytes"],
            }


def call_generate_content_with_rate_limit_retry(
    client: Any,
    *,
    model: str,
    contents: List[Any],
    config: Any,
    operation_name: str,
) -> Any:
    max_attempts = 1 + LLM_RATE_LIMIT_RETRY_COUNT

    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            if attempt < max_attempts and is_rate_limited_error(exc):
                print(
                    f"WARNING: {operation_name} hit rate limits on attempt {attempt}/{max_attempts}; "
                    f"waiting {LLM_RATE_LIMIT_COOLDOWN_SECONDS}s before retrying same request"
                )
                if LLM_RATE_LIMIT_COOLDOWN_SECONDS > 0:
                    time.sleep(LLM_RATE_LIMIT_COOLDOWN_SECONDS)
                continue
            raise


def load_text_file(path_text: str) -> str:
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        print(f"WARNING: context file not found: {path_text}")
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"WARNING: failed reading context file {path_text}: {exc}")
        return ""


def normalize_storage_path(path: Path) -> str:
    return path.as_posix()


def parse_extractor_route_overrides(raw_json: str) -> Dict[str, str]:
    text = normalize_space(raw_json)
    if not text:
        return {}
    parsed = safe_json_loads(text, {})
    if not isinstance(parsed, dict):
        print("WARNING: extractor route overrides must be a JSON object; ignoring.")
        return {}

    routes: Dict[str, str] = {}
    for key, value in parsed.items():
        normalized_class = normalize_space(str(key)).lower()
        normalized_strategy = normalize_space(str(value)).lower().replace("-", "_")
        if not normalized_class:
            continue
        if normalized_strategy not in SUPPORTED_EXTRACTION_STRATEGIES:
            print(f"WARNING: unsupported extraction strategy '{value}' for class '{key}'; ignoring.")
            continue
        routes[normalized_class] = normalized_strategy
    return routes


def resolve_extraction_strategy(document_class: str, route_overrides: Dict[str, str]) -> str:
    normalized_class = normalize_space(document_class).lower() or "unknown"
    strategy = (
        route_overrides.get(normalized_class)
        or route_overrides.get("*")
        or DEFAULT_EXTRACTION_STRATEGY_BY_CLASS.get(normalized_class)
        or DEFAULT_EXTRACTION_STRATEGY_BY_CLASS.get("*", "vision_universal")
    )
    strategy = normalize_space(strategy).lower().replace("-", "_")
    if strategy not in SUPPORTED_EXTRACTION_STRATEGIES:
        return "vision_universal"
    return strategy


def resolve_page_extraction_strategy(
    document_meta: dict,
    native_context: dict,
    genai_client: Any,
    extractor_route_overrides: Dict[str, str],
    native_vision_fallback: bool,
    native_vision_fallback_max_lines: int,
    native_vision_fallback_max_chars: int,
) -> Tuple[str, str]:
    base_extraction_strategy = resolve_extraction_strategy(document_meta.get("document_class", "unknown"), extractor_route_overrides)
    extraction_strategy = base_extraction_strategy
    if (
        base_extraction_strategy == "native_only"
        and bool(native_vision_fallback)
        and bool(genai_client)
        and should_native_page_fallback_to_vision(
            native_context,
            max_lines=native_vision_fallback_max_lines,
            max_chars=native_vision_fallback_max_chars,
        )
    ):
        extraction_strategy = "vision_universal"
    return base_extraction_strategy, extraction_strategy


def get_project_context(
    project_id: str,
    explicit_context: str = "",
    context_file: str = "",
    enable_builtin_context: bool = False,
) -> str:
    direct = (explicit_context or "").strip()
    if direct:
        return direct

    if context_file:
        file_context = load_text_file(context_file)
        if file_context:
            return file_context

    if not enable_builtin_context:
        return ""
    return PROJECT_CONTEXTS.get(project_id, "")


def extract_revision_source(parts: Iterable[str]) -> str:
    matches: List[str] = []
    for part in parts:
        part_text = part.replace("_", " ").replace("-", " ")
        for pattern, _, _ in REVISION_RULES:
            if pattern.search(part_text):
                matches.append(part)
                break
    if matches:
        return matches[-1]
    return "UNSPECIFIED"


def normalize_revision_searchable_text(parts: Iterable[str]) -> str:
    raw = " / ".join(str(part) for part in parts)
    normalized = raw.replace("_", " ").replace("-", " ")
    return normalize_space(normalized)


def fallback_revision_source(parts: Iterable[str]) -> str:
    for part in reversed(tuple(parts)):
        normalized = normalize_space(str(part))
        if normalized:
            return normalized
    return "UNSPECIFIED"


def parse_date_candidate(year: int, month: int, day: int) -> Optional[date]:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def infer_date_from_text(value: str) -> Tuple[str, float, str]:
    text = normalize_space(value)
    if not text:
        return "", 0.0, ""

    for match in re.finditer(r"\b(20\d{2})[._\-/](\d{1,2})[._\-/](\d{1,2})\b", text):
        parsed = parse_date_candidate(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if parsed:
            return parsed.isoformat(), 0.95, match.group(0)

    for match in re.finditer(r"\b(\d{1,2})[._\-/](\d{1,2})[._\-/](\d{2,4})\b", text):
        year = int(match.group(3))
        if year < 100:
            year += 2000
        parsed = parse_date_candidate(year, int(match.group(1)), int(match.group(2)))
        if parsed:
            return parsed.isoformat(), 0.7, match.group(0)

    return "", 0.0, ""


def infer_revision_date_metadata(relative_path: Path) -> Tuple[str, float, str]:
    parent_text = normalize_revision_searchable_text(relative_path.parts[:-1])
    all_text = normalize_revision_searchable_text(relative_path.parts)
    revision_date, confidence, source = infer_date_from_text(parent_text)
    if revision_date:
        return revision_date, confidence, source
    return infer_date_from_text(all_text)


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return date.min


def revision_label_sort_value(label: str) -> str:
    normalized = normalize_space(label).upper()
    stage_rank = 0
    if "SD" in normalized:
        stage_rank = 10
    if "DD" in normalized:
        stage_rank = 20
    if "CD" in normalized:
        stage_rank = 30
    if "BID" in normalized:
        stage_rank = 40
    if "PERMIT" in normalized:
        stage_rank = 50
    if "IFC" in normalized:
        stage_rank = 60
    return f"{stage_rank:03d}:{normalized}"


def infer_revision_metadata(relative_path: Path) -> Tuple[str, int, str]:
    parent_parts = relative_path.parts[:-1]
    source_parts = parent_parts or relative_path.parts
    revision_source = extract_revision_source(source_parts)
    if revision_source == "UNSPECIFIED":
        revision_source = fallback_revision_source(source_parts)

    search_targets = [
        normalize_revision_searchable_text(source_parts),
        normalize_revision_searchable_text(relative_path.parts),
    ]

    # Prefer folder-level revision tokens over filename tokens.
    for searchable in search_targets:
        for pattern, label, sequence in REVISION_RULES:
            if pattern.search(searchable):
                return label, sequence, revision_source

    return "UNSPECIFIED", 0, revision_source


def strip_revision_tokens(name: str) -> str:
    cleaned = name
    cleaned = re.sub(r"\b(25|50|75|100)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(SD|DD|CD|BOD|IFC|PERMIT|BID|REV\d+)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    return normalize_space(cleaned)


def infer_document_family_id(path: Path) -> str:
    family_name = strip_revision_tokens(path.stem)
    return slugify(family_name)

def infer_document_class(relative_path: Path) -> str:
    searchable = " ".join(relative_path.parts).lower()
    stem = relative_path.stem.lower()
    filename = relative_path.name.lower()

    if re.search(r"(^|_)text(_|\.|$)", filename):
        return "text_native"

    # TODO: replace this heuristic router with a learned document router.
    if re.search(r"\b(specs?|specifications?|division\s*\d{1,2})\b", searchable):
        return "specification"
    if re.search(r"\b(report|assessment|study|analysis|summary|memo)\b", searchable):
        return "report"
    if re.search(r"\b(narrative|basis\s+of\s+design|design\s+narrative)\b", searchable):
        return "narrative"
    if re.search(r"\b(submittal|shop\s+drawing|cut\s*sheet|product\s+data)\b", searchable):
        return "submittal"
    if re.search(r"\b(mixed|combined|package)\b", searchable):
        return "mixed_pdf"
    if re.search(r"\b(sheet|plan|elevation|section|detail|legend|schedule|reflected\s+ceiling)\b", searchable):
        return "drawing"
    if re.search(r"^[a-z]{1,3}\d+(?:\.\d+)?[a-z]?$", stem, re.IGNORECASE):
        return "drawing"
    return "unknown"


def infer_retrieval_scope(document_class: str) -> str:
    normalized = normalize_space(document_class).lower()
    if normalized in {"drawing", "drawing_pdf"}:
        return "drawing_only"
    if normalized in {"report", "specification", "narrative", "submittal", "text_native"}:
        return "report_only"
    return "all"


def bool_payload(value: Any) -> bool:
    return bool(value)


def should_native_page_fallback_to_vision(native_context: dict, max_lines: int, max_chars: int) -> bool:
    line_limit = max(0, int(max_lines))
    char_limit = max(0, int(max_chars))
    line_count = int(native_context.get("line_count", 0) or 0)
    text = normalize_space(str(native_context.get("text", "")))
    char_count = len(text)
    candidate_tables = native_context.get("candidate_tables", [])
    return line_count <= line_limit and char_count <= char_limit and not candidate_tables


def should_parallel_render_native_page(extraction_strategy: str, save_crops: bool) -> bool:
    return extraction_strategy != "native_only" or bool(save_crops)


def counter_to_sorted_dict(counter: Counter) -> Dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def empty_phase_usage() -> Dict[str, Dict[str, int]]:
    return {
        "classify": {"prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "calls": 0},
        "extract": {"prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "calls": 0},
        "verify": {"prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0, "calls": 0},
    }


def accumulate_phase_usage(target: Dict[str, Dict[str, int]], source: Dict[str, Dict[str, int]]) -> None:
    for phase_name, phase_totals in target.items():
        phase_values = source.get(phase_name, {}) if isinstance(source, dict) else {}
        for key in ["prompt_tokens", "output_tokens", "thinking_tokens", "calls"]:
            phase_totals[key] += int(phase_values.get(key, 0) or 0)


def log_page_phase(
    enabled: bool,
    source_file_name: str,
    page_num: int,
    phase_name: str,
    detail: str = "",
) -> None:
    if not enabled:
        return
    prefix = f"[{source_file_name} | page {page_num:04d}]"
    message = f"{prefix} {phase_name}"
    if detail:
        message += f" | {detail}"
    tqdm.write(message)


def clamp_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.5
    return max(0.0, min(1.0, numeric))


def ensure_supported_page_class(value: str) -> str:
    value = normalize_space(value).lower().replace(" ", "_")
    return value if value in SUPPORTED_PAGE_CLASSES else "drawing"


def ensure_supported_block_type(value: str) -> str:
    value = normalize_space(value).lower()
    return value if value in SUPPORTED_BLOCK_TYPES else "text_region"


def ensure_supported_region(value: str) -> str:
    value = normalize_space(value).lower()
    return value if value in SUPPORTED_APPROX_REGIONS else "full_page"


def truncate_text(value: str, max_chars: int) -> str:
    text = value or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars].rstrip()} ... [truncated {omitted} chars]"


SPLIT_BOUNDARY_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def is_word_character(value: str) -> bool:
    return value.isalnum() or value == "_"


def preferred_split_index(text: str, max_chars: int) -> int:
    if len(text) <= max_chars:
        return len(text)

    window = text[: max_chars + 1]
    min_index = max(1, int(max_chars * 0.6))

    for separator in SPLIT_BOUNDARY_SEPARATORS:
        idx = window.rfind(separator, min_index)
        if idx != -1:
            return idx + len(separator)

    for separator in SPLIT_BOUNDARY_SEPARATORS:
        idx = window.rfind(separator, 1)
        if idx != -1:
            return idx + len(separator)

    return max_chars


def advance_split_start(text: str, start_index: int) -> int:
    idx = max(0, min(start_index, len(text)))
    if 0 < idx < len(text):
        while idx < len(text) and is_word_character(text[idx - 1]) and is_word_character(text[idx]):
            idx += 1
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def split_long_text(
    text: str,
    max_chars: int = MAX_EMBED_TEXT_CHARS,
    overlap_chars: int = EMBED_TEXT_OVERLAP_CHARS,
) -> List[str]:
    content = (text or "").strip()
    if not content:
        return []
    if max_chars <= 0 or len(content) <= max_chars:
        return [content]

    overlap_chars = max(0, min(overlap_chars, max_chars // 2))
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", content) if part.strip()]
    if not paragraphs:
        paragraphs = [content]

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            if overlap_chars:
                overlap = current[-overlap_chars:].strip()
                current = f"{overlap}\n\n{paragraph}".strip() if overlap else paragraph
            else:
                current = paragraph
        else:
            current = paragraph

        while len(current) > max_chars:
            split_at = preferred_split_index(current, max_chars)
            split_at = max(1, min(split_at, len(current) - 1))
            window = current[:split_at].strip()
            if not window:
                split_at = min(max_chars, len(current) - 1)
                window = current[:split_at].strip()
            if window:
                chunks.append(window)
            if overlap_chars:
                effective_overlap = min(overlap_chars, max(0, split_at - 1))
                next_start = split_at - effective_overlap
                next_start = advance_split_start(current, next_start)
            else:
                next_start = split_at

            next_current = current[next_start:].strip()
            if not next_current or next_current == current:
                next_current = current[split_at:].strip()
            current = next_current

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def build_split_chunks(
    base_chunk: dict,
    text: str,
    max_chars: int = MAX_EMBED_TEXT_CHARS,
    overlap_chars: int = EMBED_TEXT_OVERLAP_CHARS,
) -> List[dict]:
    pieces = split_long_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
    if not pieces:
        return []

    total = len(pieces)
    chunk_list: List[dict] = []
    for idx, piece in enumerate(pieces):
        chunk = dict(base_chunk)
        chunk["text"] = piece
        chunk["text_to_embed"] = piece
        if total > 1:
            chunk["chunk_sub_id"] = idx
            chunk["chunk_sub_count"] = total
        chunk_list.append(chunk)
    return chunk_list


def normalized_entity_type(entity: dict) -> str:
    return normalize_space(str(entity.get("entity_type", ""))).lower() or "unknown"


def is_high_value_entity(entity: dict) -> bool:
    return normalized_entity_type(entity) in HIGH_VALUE_ENTITY_TYPES


def clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def normalize_bbox_candidate(raw_bbox: Any, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    values: Optional[Tuple[float, float, float, float]] = None
    if isinstance(raw_bbox, dict):
        if all(key in raw_bbox for key in ["x0", "y0", "x1", "y1"]):
            values = (
                to_float(raw_bbox.get("x0")),
                to_float(raw_bbox.get("y0")),
                to_float(raw_bbox.get("x1")),
                to_float(raw_bbox.get("y1")),
            )
        elif all(key in raw_bbox for key in ["left", "top", "right", "bottom"]):
            values = (
                to_float(raw_bbox.get("left")),
                to_float(raw_bbox.get("top")),
                to_float(raw_bbox.get("right")),
                to_float(raw_bbox.get("bottom")),
            )
        elif all(key in raw_bbox for key in ["x", "y", "w", "h"]):
            x = to_float(raw_bbox.get("x"))
            y = to_float(raw_bbox.get("y"))
            w = to_float(raw_bbox.get("w"))
            h = to_float(raw_bbox.get("h"))
            if None not in {x, y, w, h}:
                values = (x, y, x + w, y + h)
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        values = (to_float(raw_bbox[0]), to_float(raw_bbox[1]), to_float(raw_bbox[2]), to_float(raw_bbox[3]))

    if not values or any(item is None for item in values):
        return None

    x0, y0, x1, y1 = values
    use_pixels = any(abs(value) > 1.5 for value in [x0, y0, x1, y1])
    if use_pixels:
        if width <= 0 or height <= 0:
            return None
        x0 /= width
        x1 /= width
        y0 /= height
        y1 /= height

    x0 = clamp_unit_interval(x0)
    y0 = clamp_unit_interval(y0)
    x1 = clamp_unit_interval(x1)
    y1 = clamp_unit_interval(y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def extract_block_bbox_norm(block: dict, image: object) -> Optional[Tuple[float, float, float, float]]:
    width, height = image.size
    candidates = [
        block.get("bbox_norm"),
        block.get("bbox"),
        block.get("bounding_box"),
        block.get("box"),
    ]
    payload = block.get("structured_payload", {})
    if isinstance(payload, dict):
        candidates.extend(
            [
                payload.get("bbox_norm"),
                payload.get("bbox"),
                payload.get("bounding_box"),
                payload.get("box"),
                payload.get("region_bbox"),
                payload.get("crop_box"),
            ]
        )

    for candidate in candidates:
        bbox = normalize_bbox_candidate(candidate, width, height)
        if bbox:
            return bbox
    return None


def normalize_pixel_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return (x0, y0, x1, y1)


def apply_padding_to_box(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    if padding_ratio <= 0:
        return normalize_pixel_box(box, width, height)

    x0, y0, x1, y1 = box
    pad_x = max(2, int((x1 - x0) * padding_ratio))
    pad_y = max(2, int((y1 - y0) * padding_ratio))
    padded = (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)
    return normalize_pixel_box(padded, width, height)


def split_region_box(
    box: Tuple[int, int, int, int],
    slot_index: int,
    slot_count: int,
    region: str,
) -> Tuple[int, int, int, int]:
    if slot_count <= 1:
        return box

    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    slot_index = max(0, min(slot_index, slot_count - 1))

    if region in {"top_band", "bottom_band"}:
        step = width / slot_count
        sx0 = int(x0 + (slot_index * step))
        sx1 = int(x0 + ((slot_index + 1) * step))
        return (sx0, y0, max(sx0 + 1, sx1), y1)

    if region in {"left_band", "right_band"}:
        step = height / slot_count
        sy0 = int(y0 + (slot_index * step))
        sy1 = int(y0 + ((slot_index + 1) * step))
        return (x0, sy0, x1, max(sy0 + 1, sy1))

    cols = max(1, int(math.ceil(math.sqrt(slot_count))))
    rows = max(1, int(math.ceil(slot_count / cols)))
    col = slot_index % cols
    row = slot_index // cols
    cell_w = width / cols
    cell_h = height / rows
    sx0 = int(x0 + (col * cell_w))
    sx1 = int(x0 + ((col + 1) * cell_w))
    sy0 = int(y0 + (row * cell_h))
    sy1 = int(y0 + ((row + 1) * cell_h))
    return (sx0, sy0, max(sx0 + 1, sx1), max(sy0 + 1, sy1))


def box_to_norm(box: Tuple[int, int, int, int], width: int, height: int) -> Dict[str, float]:
    x0, y0, x1, y1 = box
    return {
        "x0": round(x0 / width, 6),
        "y0": round(y0 / height, 6),
        "x1": round(x1 / width, 6),
        "y1": round(y1 / height, 6),
    }


def region_to_box(image: object, region: str) -> Tuple[int, int, int, int]:
    width, height = image.size
    regions = {
        "full_page": (0.0, 0.0, 1.0, 1.0),
        "top_band": (0.0, 0.0, 1.0, 0.25),
        "bottom_band": (0.0, 0.75, 1.0, 1.0),
        "left_band": (0.0, 0.0, 0.33, 1.0),
        "right_band": (0.67, 0.0, 1.0, 1.0),
        "upper_left": (0.0, 0.0, 0.5, 0.5),
        "upper_center": (0.25, 0.0, 0.75, 0.5),
        "upper_right": (0.5, 0.0, 1.0, 0.5),
        "center_left": (0.0, 0.25, 0.5, 0.75),
        "center": (0.2, 0.2, 0.8, 0.8),
        "center_right": (0.5, 0.25, 1.0, 0.75),
        "lower_left": (0.0, 0.5, 0.5, 1.0),
        "lower_center": (0.25, 0.5, 0.75, 1.0),
        "lower_right": (0.5, 0.5, 1.0, 1.0),
    }
    x0, y0, x1, y1 = regions.get(region, regions["full_page"])
    return (
        max(0, int(width * x0)),
        max(0, int(height * y0)),
        min(width, int(width * x1)),
        min(height, int(height * y1)),
    )


def save_page_image(image: object, page_path: Path) -> str:
    page_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(page_path, format="JPEG", quality=95)
    return normalize_storage_path(page_path)


def save_block_crop(
    image: object,
    crop_path: Path,
    block: dict,
    approx_region: str,
    save_crops: bool,
    slot_index: int = 0,
    slot_count: int = 1,
    crop_padding_ratio: float = 0.02,
) -> Tuple[str, dict]:
    if not save_crops:
        return "", {}

    width, height = image.size
    bbox_norm = extract_block_bbox_norm(block, image)
    strategy = "region_template"
    if bbox_norm:
        x0, y0, x1, y1 = bbox_norm
        box = (int(width * x0), int(height * y0), int(width * x1), int(height * y1))
        box = apply_padding_to_box(box, width, height, crop_padding_ratio)
        strategy = "bbox_norm"
    else:
        base_region = ensure_supported_region(approx_region)
        box = region_to_box(image, base_region)
        box = split_region_box(box, slot_index, slot_count, base_region)
        box = apply_padding_to_box(box, width, height, crop_padding_ratio * 0.6)
        if slot_count > 1:
            strategy = "region_subdivision"

    box = normalize_pixel_box(box, width, height)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).save(crop_path, format="JPEG", quality=92)
    metadata = {
        "strategy": strategy,
        "slot_index": slot_index,
        "slot_count": slot_count,
        "box_px": {"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]},
        "box_norm": box_to_norm(box, width, height),
    }
    return normalize_storage_path(crop_path), metadata


def extract_native_text(page: Any) -> str:
    if not page:
        return ""
    try:
        text = page.extract_text(extraction_mode="layout")
        if text:
            return text
    except TypeError:
        pass
    except Exception:
        pass
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def candidate_table_rows(line: str) -> Optional[List[str]]:
    if "\t" in line:
        cells = [normalize_space(part) for part in line.split("\t")]
    else:
        cells = [normalize_space(part) for part in re.split(r"\s{2,}", line)]
    cells = [cell for cell in cells if cell]
    return cells if len(cells) >= 3 else None


def detect_candidate_tables(native_text: str) -> List[dict]:
    tables: List[dict] = []
    current_rows: List[List[str]] = []
    table_index = 0

    for raw_line in native_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current_rows:
                table_index += 1
                tables.append(
                    {
                        "table_title": f"Native Candidate Table {table_index}",
                        "columns": current_rows[0],
                        "rows": [
                            {
                                current_rows[0][idx]: row[idx] if idx < len(row) else ""
                                for idx in range(len(current_rows[0]))
                            }
                            for row in current_rows[1:]
                        ],
                        "row_count": max(0, len(current_rows) - 1),
                        "preview_lines": [" | ".join(row) for row in current_rows[:8]],
                    }
                )
                current_rows = []
            continue

        cells = candidate_table_rows(line)
        if cells:
            current_rows.append(cells)
        elif current_rows:
            table_index += 1
            tables.append(
                {
                    "table_title": f"Native Candidate Table {table_index}",
                    "columns": current_rows[0],
                    "rows": [
                        {
                            current_rows[0][idx]: row[idx] if idx < len(row) else ""
                            for idx in range(len(current_rows[0]))
                        }
                        for row in current_rows[1:]
                    ],
                    "row_count": max(0, len(current_rows) - 1),
                    "preview_lines": [" | ".join(row) for row in current_rows[:8]],
                }
            )
            current_rows = []

    if current_rows:
        table_index += 1
        tables.append(
            {
                "table_title": f"Native Candidate Table {table_index}",
                "columns": current_rows[0],
                "rows": [
                    {
                        current_rows[0][idx]: row[idx] if idx < len(row) else ""
                        for idx in range(len(current_rows[0]))
                    }
                    for row in current_rows[1:]
                ],
                "row_count": max(0, len(current_rows) - 1),
                "preview_lines": [" | ".join(row) for row in current_rows[:8]],
            }
        )

    return tables


def build_native_page_context(page: Any, enabled: bool) -> dict:
    if not enabled or not HAS_PYPDF:
        return {"text": "", "candidate_tables": [], "line_count": 0}
    text = extract_native_text(page)
    text = text.replace("\x00", " ")
    text = text.strip()
    return {
        "text": text,
        "candidate_tables": detect_candidate_tables(text),
        "line_count": len([line for line in text.splitlines() if line.strip()]),
    }


def native_context_snippet(native_context: dict, max_chars: int = 3500) -> str:
    snippet = native_context.get("text", "")[:max_chars]
    payload = {
        "native_text_excerpt": snippet,
        "candidate_tables": native_context.get("candidate_tables", [])[:3],
        "line_count": native_context.get("line_count", 0),
    }
    return json.dumps(payload, ensure_ascii=False)


def heuristic_header_from_text(native_text: str) -> dict:
    lines = [normalize_space(line) for line in native_text.splitlines() if normalize_space(line)]
    joined = "\n".join(lines[:40])
    sheet_match = re.search(r"\b([A-Z]{1,3}\d+(?:\.\d+)?[A-Z]?)\b", joined)
    title = lines[0] if lines else "Unknown"
    discipline = "architectural"
    if re.search(r"\bMECH|DUCT|HVAC|CFM\b", joined, re.IGNORECASE):
        discipline = "mechanical"
    elif re.search(r"\bELEC|PANEL|CIRCUIT|LIGHTING\b", joined, re.IGNORECASE):
        discipline = "electrical"
    elif re.search(r"\bPLUMB|SANITARY|WATER\b", joined, re.IGNORECASE):
        discipline = "plumbing"
    elif re.search(r"\bSTRUCT|COLUMN|BEAM\b", joined, re.IGNORECASE):
        discipline = "structural"
    return {
        "sheet_number": sheet_match.group(1) if sheet_match else "Unknown",
        "sheet_title": title,
        "discipline": discipline,
        "drawing_type": "drawing",
        "scale": "",
        "title_block_revision": "",
        "title_block_date": "",
        "project_name": "",
        "project_number": "",
    }


def heuristic_page_class(native_context: dict) -> Tuple[str, dict]:
    text = native_context.get("text", "")
    upper = text.upper()
    has_tables = "SCHEDULE" in upper or len(native_context.get("candidate_tables", [])) > 0
    has_legends = "LEGEND" in upper or "ABBREVIATION" in upper
    has_notes = "GENERAL NOTES" in upper or "NOTE:" in upper or "NOTES" in upper
    has_renderings = "PERSPECTIVE" in upper or "RENDERING" in upper
    has_drawings = bool(re.search(r"\bROOM\b|\bGRID\b|\bDETAIL\b|\bSECTION\b|\bELEVATION\b", upper))

    page_class = "drawing"
    if has_tables and has_legends:
        page_class = "mixed"
    elif has_tables and not has_drawings:
        page_class = "schedule"
    elif has_legends and not has_drawings:
        page_class = "legend"
    elif has_notes and not has_drawings and not has_tables:
        page_class = "notes"
    elif "SHEET INDEX" in upper or "INDEX OF DRAWINGS" in upper:
        page_class = "sheet_index"
    elif has_renderings and not has_drawings:
        page_class = "cover_or_rendering"
    elif native_context.get("line_count", 0) > 60 and not has_drawings:
        page_class = "text_heavy"

    flags = {
        "has_drawings": has_drawings or page_class in {"drawing", "mixed"},
        "has_tables": has_tables,
        "has_legends": has_legends,
        "has_notes": has_notes,
        "has_renderings": has_renderings,
    }
    flags["page_contains_multiple_content_types"] = (
        sum(1 for key in ["has_drawings", "has_tables", "has_legends", "has_notes", "has_renderings"] if flags[key]) > 1
    )
    return page_class, flags


def heuristic_extraction(
    classification: dict,
    native_context: dict,
) -> dict:
    header = dict(classification.get("header", {}))
    page_summary = native_context.get("text", "")[:4000] or "ILLEGIBLE"
    blocks: List[dict] = []
    if page_summary:
        blocks.append(
            {
                "block_index": 0,
                "block_type": "text_region",
                "block_label": "Native text extract",
                "approx_region": "full_page",
                "text": page_summary,
                "structured_payload": {
                    "source": "native_pdf_text",
                    "candidate_tables": native_context.get("candidate_tables", []),
                },
                "extraction_source": "native",
                "confidence": 0.5,
            }
        )

    tables = []
    for idx, table in enumerate(native_context.get("candidate_tables", []), start=1):
        table_payload = {
            "table_title": table.get("table_title", f"Native Candidate Table {idx}"),
            "columns": table.get("columns", []),
            "rows": table.get("rows", []),
            "notes": "",
            "page_block_index": idx,
        }
        tables.append(table_payload)
        blocks.append(
            {
                "block_index": idx,
                "block_type": "table",
                "block_label": table_payload["table_title"],
                "approx_region": "center",
                "text": "\n".join(table.get("preview_lines", [])),
                "structured_payload": table_payload,
                "extraction_source": "native",
                "confidence": 0.4,
            }
        )

    return {
        "header": header,
        "page_summary": page_summary,
        "content_blocks": blocks,
        "entities": [],
        "cross_references": [],
        "tables_on_sheet": tables,
        "keynotes_or_legends": [],
        "acoustic_data": normalize_acoustic_data({}),
        "page_flags": classification.get("page_flags", {}),
    }


def merge_classification_with_extraction(classification: dict, raw_extraction: dict) -> dict:
    merged = {
        "header": dict(classification.get("header", {})),
        "page_class": ensure_supported_page_class(classification.get("page_class", "drawing")),
        "page_flags": normalize_page_flags(classification.get("page_flags", {}), classification.get("page_flags", {})),
    }

    if not isinstance(raw_extraction, dict):
        return merged

    raw_header = raw_extraction.get("header", {})
    if isinstance(raw_header, dict):
        merged["header"].update(raw_header)

    raw_page_class = normalize_space(str(raw_extraction.get("page_class", ""))).lower().replace("-", "_")
    if raw_page_class in SUPPORTED_PAGE_CLASSES:
        merged["page_class"] = raw_page_class

    merged["page_flags"] = normalize_page_flags(
        raw_extraction.get("page_flags", {}),
        merged["page_flags"],
    )
    return merged


def extract_page_universal(
    image_bytes: bytes,
    classification: dict,
    native_context: dict,
    client: Any,
    llm_model: str,
    project_context: str = "",
    model_pool: Optional[VisionModelPool] = None,
    page_label: str = "",
    output_path: Optional[Path] = None,
    document_meta: Optional[dict] = None,
    page_num: int = 0,
    image_path: str = "",
) -> Tuple[dict, dict]:
    default_result = heuristic_extraction(classification, native_context)
    usage = {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "parse_fallback": 0,
        "parse_error": "",
        "quarantined": False,
        "quarantine_path": "",
        "low_confidence_extraction": False,
        "model": normalize_space(llm_model) or DEFAULT_LLM_MODEL,
        "attempts": 0,
        "failovers": 0,
        "exception": "",
    }

    if not client:
        return default_result, usage

    system_instruction = UNIVERSAL_EXTRACTION_PROMPT
    if project_context:
        system_instruction = f"{project_context}\n\n{UNIVERSAL_EXTRACTION_PROMPT}"

    try:
        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part(
                text=(
                    "Classification context:\n"
                    f"{json.dumps(classification, ensure_ascii=False)}\n\n"
                    "Native support context:\n"
                    f"{native_context_snippet(native_context)}"
                )
            ),
        ]
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.0,
        )

        def absorb_response_usage(response_obj: Any) -> None:
            if getattr(response_obj, "usage_metadata", None):
                usage["prompt_tokens"] += getattr(response_obj.usage_metadata, "prompt_token_count", 0) or 0
                usage["output_tokens"] += getattr(response_obj.usage_metadata, "candidates_token_count", 0) or 0
                usage["thinking_tokens"] += getattr(response_obj.usage_metadata, "thoughts_token_count", 0) or 0

        def direct_call(model: str, call_contents: List[Any], operation_name: str) -> Any:
            return call_generate_content_with_rate_limit_retry(
                client,
                model=model,
                contents=call_contents,
                config=config,
                operation_name=operation_name,
            )

        model_meta = {
            "model": normalize_space(llm_model) or DEFAULT_LLM_MODEL,
            "attempts": 1,
            "failovers": 0,
        }
        if model_pool:
            response, model_meta = model_pool.call_generate_content(
                client,
                contents=contents,
                config=config,
                operation_name="extraction",
                page_label=page_label,
            )
        else:
            response = direct_call(llm_model, contents, "extraction")

        absorb_response_usage(response)
        usage["model"] = normalize_space(model_meta.get("model", usage.get("model", ""))) or usage["model"]
        usage["attempts"] = int(model_meta.get("attempts", 1) or 1)
        usage["failovers"] = int(model_meta.get("failovers", 0) or 0)

        parsed, parse_info = try_parse_llm_json(getattr(response, "text", "") or "", debug_label=page_label)
        if parsed is not None:
            return parsed, usage

        usage["parse_fallback"] = 1
        usage["parse_error"] = parse_info.get("parse_error", "")

        repair_instruction = types.Part(
            text=(
                "Your previous response was invalid JSON. Return ONLY valid JSON. "
                "Do not include markdown, prose, comments, or code fences. "
                "Use the required extraction schema exactly."
            )
        )
        repair_contents = contents + [
            types.Part(text=f"Invalid previous response:\n{(getattr(response, 'text', '') or '')[:12000]}"),
            repair_instruction,
        ]

        retry_models: List[str] = []
        first_model = normalize_space(str(usage.get("model", ""))) or normalize_space(llm_model) or DEFAULT_LLM_MODEL
        retry_models.append(first_model)
        repair_model = normalize_space(DEFAULT_JSON_REPAIR_MODEL) or DEFAULT_LLM_MODEL
        if repair_model not in retry_models:
            retry_models.append(repair_model)

        raw_retry_text = ""
        for retry_index, retry_model in enumerate(retry_models, start=1):
            try:
                retry_response = direct_call(retry_model, repair_contents, "extraction_json_repair")
                usage["attempts"] += 1
                usage["model"] = retry_model
                absorb_response_usage(retry_response)
                raw_retry_text = getattr(retry_response, "text", "") or ""
                parsed, parse_info = try_parse_llm_json(raw_retry_text, debug_label=f"{page_label} repair {retry_index}")
                if parsed is not None:
                    usage["parse_error"] = ""
                    return parsed, usage
                usage["parse_error"] = parse_info.get("parse_error", "") or usage["parse_error"]
            except Exception as retry_exc:
                usage["exception"] = str(retry_exc)
                print(f"WARNING: extraction JSON repair failed ({page_label or 'unknown_page'}): {retry_exc}")

        _, quarantine_info = parse_llm_json_or_quarantine(
            raw_retry_text or getattr(response, "text", "") or "",
            output_path=output_path,
            document_meta=document_meta,
            page_num=page_num,
            image_path=image_path,
            model=normalize_space(str(usage.get("model", ""))) or first_model,
            prompt_name="universal_extraction",
            debug_label=page_label,
        )
        usage.update({key: quarantine_info.get(key, usage.get(key)) for key in ["quarantined", "quarantine_path", "parse_error"]})
        usage["low_confidence_extraction"] = True
        fallback = dict(default_result)
        fallback["ingestion_diagnostics"] = {
            "parse_fallback": 1,
            "quarantined": bool(usage.get("quarantined", False)),
            "quarantine_path": usage.get("quarantine_path", ""),
            "parse_error": usage.get("parse_error", ""),
            "low_confidence_extraction": True,
        }
        return fallback, usage
    except Exception as exc:
        usage["exception"] = str(exc)
        usage["low_confidence_extraction"] = True
        print(f"WARNING: extraction failed ({page_label or 'unknown_page'}): {exc}")
        fallback = dict(default_result)
        quarantine_path = ""
        if output_path is not None and document_meta:
            try:
                quarantine_path = write_quarantine_record(
                    output_path=output_path,
                    document_meta=document_meta,
                    page_num=page_num,
                    reason="extraction_exception",
                    exception=str(exc),
                    image_path=image_path,
                    model=normalize_space(llm_model) or DEFAULT_LLM_MODEL,
                    prompt_name="universal_extraction",
                )
            except Exception as quarantine_exc:
                print(f"WARNING: failed to write quarantine record ({page_label or 'unknown_page'}): {quarantine_exc}")
        usage["quarantined"] = bool(quarantine_path)
        usage["quarantine_path"] = quarantine_path
        fallback["ingestion_diagnostics"] = {
            "parse_fallback": int(usage.get("parse_fallback", 0) or 0),
            "quarantined": bool(quarantine_path),
            "quarantine_path": quarantine_path,
            "exception": str(exc),
            "low_confidence_extraction": True,
        }
        return fallback, usage


def verify_extraction(
    image_bytes: bytes,
    previous_summary: str,
    client: Any,
    llm_model: str,
) -> Tuple[dict, dict]:
    usage = {"prompt_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    if not client or not previous_summary:
        return {"missing_items": [], "extraction_complete": True}, usage

    try:
        response = call_generate_content_with_rate_limit_retry(
            client,
            model=llm_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                types.Part(text=previous_summary),
            ],
            config=types.GenerateContentConfig(
                system_instruction=VERIFICATION_PROMPT,
                response_mime_type="application/json",
                temperature=0.0,
            ),
            operation_name="verification",
        )

        if getattr(response, "usage_metadata", None):
            usage["prompt_tokens"] = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            usage["output_tokens"] = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            usage["thinking_tokens"] = getattr(response.usage_metadata, "thoughts_token_count", 0) or 0

        return safe_json_loads(getattr(response, "text", "") or "", {"missing_items": [], "extraction_complete": True}), usage
    except Exception as exc:
        print(f"WARNING: verification failed: {exc}")
        return {"missing_items": [], "extraction_complete": True}, usage


def should_verify_page(
    verify_mode: str,
    classification: dict,
    extraction: dict,
) -> bool:
    if verify_mode == "always":
        return True
    if verify_mode == "off":
        return False

    page_class = classification.get("page_class", "drawing")
    header = extraction.get("header", {})
    thin_extraction = (
        len(extraction.get("content_blocks", [])) <= 1
        and not extraction.get("entities")
        and not extraction.get("tables_on_sheet")
        and not extraction.get("keynotes_or_legends")
    )
    missing_key_header = any(
        not normalize_space(str(header.get(field, "")))
        or header.get(field) == "Unknown"
        for field in ["sheet_number", "sheet_title", "discipline"]
    )
    return page_class == "mixed" or thin_extraction or missing_key_header


def add_verification_findings(
    extraction: dict,
    verification_result: dict,
) -> dict:
    missing_items = [item for item in verification_result.get("missing_items", []) if isinstance(item, dict)]
    extraction["verification_findings"] = missing_items
    if not missing_items:
        return extraction

    page_summary = extraction.get("page_summary", "")
    page_summary += "\n\nVerification findings:\n"

    for item in missing_items:
        item_text = item.get("text", "ILLEGIBLE")
        page_summary += f"- {item_text}\n"

    extraction["page_summary"] = page_summary.strip()
    return extraction


def normalize_block(block: dict, fallback_index: int) -> dict:
    payload = block.get("structured_payload", {})
    if not isinstance(payload, dict):
        payload = {"raw": payload}
    return {
        "block_index": int(block.get("block_index", fallback_index)),
        "block_type": ensure_supported_block_type(block.get("block_type", "text_region")),
        "block_label": normalize_space(block.get("block_label", "")) or f"Block {fallback_index}",
        "approx_region": ensure_supported_region(block.get("approx_region", "full_page")),
        "bbox_norm": block.get("bbox_norm"),
        "text": block.get("text", "") or "",
        "structured_payload": payload,
        "extraction_source": normalize_space(block.get("extraction_source", "")) or "vision",
        "confidence": clamp_confidence(block.get("confidence", 0.5)),
    }


def normalize_page_flags(flags: dict, fallback: dict) -> dict:
    merged = dict(fallback)
    if isinstance(flags, dict):
        for key in ["has_drawings", "has_tables", "has_legends", "has_notes", "has_renderings", "has_photos"]:
            if key in flags:
                merged[key] = bool_payload(flags[key])
    merged["has_photos"] = bool_payload(merged.get("has_photos", merged.get("has_renderings", False)))
    merged["page_contains_multiple_content_types"] = (
        sum(1 for key in ["has_drawings", "has_tables", "has_legends", "has_notes", "has_renderings"] if merged.get(key)) > 1
    )
    return merged


def enrich_blocks_from_tables_and_keynotes(extraction: dict) -> None:
    existing_indices = {block["block_index"] for block in extraction.get("content_blocks", [])}
    next_index = (max(existing_indices) + 1) if existing_indices else 0

    for table in extraction.get("tables_on_sheet", []):
        if table.get("page_block_index") in existing_indices:
            continue
        preview_rows = table.get("rows", [])[:5]
        preview_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in preview_rows)
        extraction.setdefault("content_blocks", []).append(
            {
                "block_index": next_index,
                "block_type": "table",
                "block_label": normalize_space(table.get("table_title", "")) or f"Table {next_index}",
                "approx_region": "center",
                "text": preview_text,
                "structured_payload": table,
                "extraction_source": "derived",
                "confidence": 0.6,
            }
        )
        table["page_block_index"] = next_index
        existing_indices.add(next_index)
        next_index += 1

    if extraction.get("keynotes_or_legends"):
        missing_legends = [
            item
            for item in extraction["keynotes_or_legends"]
            if item.get("page_block_index") not in existing_indices
        ]
        if missing_legends:
            legend_text = "\n".join(
                f"{item.get('number', '')}: {item.get('text', '')}".strip(": ")
                for item in missing_legends
            )
            block_type = ensure_supported_block_type(missing_legends[0].get("block_type", "legend"))
            extraction.setdefault("content_blocks", []).append(
                {
                    "block_index": next_index,
                    "block_type": block_type,
                    "block_label": "Legend / Keynotes",
                    "approx_region": "right_band",
                    "text": legend_text,
                    "structured_payload": {"items": missing_legends},
                    "extraction_source": "derived",
                    "confidence": 0.6,
                }
            )
            for item in missing_legends:
                item["page_block_index"] = next_index
            existing_indices.add(next_index)
            next_index += 1


def normalize_acoustic_data(raw: dict) -> dict:
    acoustic = raw.get("acoustic_data", {}) if isinstance(raw, dict) else {}
    if not isinstance(acoustic, dict):
        acoustic = {}
    return {
        "ratings": [item for item in acoustic.get("ratings", []) if isinstance(item, dict)],
        "assemblies": [item for item in acoustic.get("assemblies", []) if isinstance(item, dict)],
        "equipment_noise": [item for item in acoustic.get("equipment_noise", []) if isinstance(item, dict)],
        "room_acoustic_requirements": [
            item for item in acoustic.get("room_acoustic_requirements", []) if isinstance(item, dict)
        ],
    }


def normalize_extraction_output(
    raw_extraction: dict,
    classification: dict,
    native_context: dict,
) -> dict:
    normalized = heuristic_extraction(classification, native_context)
    if not isinstance(raw_extraction, dict):
        enrich_blocks_from_tables_and_keynotes(normalized)
        return normalized

    header = dict(classification.get("header", {}))
    header.update(raw_extraction.get("header", {}) if isinstance(raw_extraction.get("header"), dict) else {})
    normalized["header"] = header
    normalized["page_summary"] = raw_extraction.get("page_summary") or raw_extraction.get("description") or normalized["page_summary"]
    raw_blocks = [
        normalize_block(block, idx)
        for idx, block in enumerate(raw_extraction.get("content_blocks", []))
        if isinstance(block, dict)
    ]
    block_index_map: Dict[int, int] = {}
    normalized_blocks = []
    for new_index, block in enumerate(raw_blocks):
        old_index = int(block.get("block_index", new_index))
        block_index_map[old_index] = new_index
        remapped = dict(block)
        remapped["block_index"] = new_index
        normalized_blocks.append(remapped)
    normalized["content_blocks"] = normalized_blocks
    normalized["entities"] = [entity for entity in raw_extraction.get("entities", []) if isinstance(entity, dict)]
    normalized["cross_references"] = [
        xref for xref in raw_extraction.get("cross_references", []) if isinstance(xref, dict)
    ]
    normalized["tables_on_sheet"] = [
        table for table in raw_extraction.get("tables_on_sheet", []) if isinstance(table, dict)
    ]
    normalized["keynotes_or_legends"] = [
        item for item in raw_extraction.get("keynotes_or_legends", []) if isinstance(item, dict)
    ]
    normalized["acoustic_data"] = normalize_acoustic_data(raw_extraction)
    if isinstance(raw_extraction.get("ingestion_diagnostics"), dict):
        normalized["ingestion_diagnostics"] = dict(raw_extraction.get("ingestion_diagnostics", {}))
    normalized["page_flags"] = normalize_page_flags(
        raw_extraction.get("page_flags", {}),
        classification.get("page_flags", {}),
    )
    for entity in normalized["entities"]:
        if entity.get("page_block_index") in block_index_map:
            entity["page_block_index"] = block_index_map[entity["page_block_index"]]
    for xref in normalized["cross_references"]:
        if xref.get("page_block_index") in block_index_map:
            xref["page_block_index"] = block_index_map[xref["page_block_index"]]
    for table in normalized["tables_on_sheet"]:
        if table.get("page_block_index") in block_index_map:
            table["page_block_index"] = block_index_map[table["page_block_index"]]
    for item in normalized["keynotes_or_legends"]:
        if item.get("page_block_index") in block_index_map:
            item["page_block_index"] = block_index_map[item["page_block_index"]]
    acoustic_data = normalized.get("acoustic_data", {})
    for collection_name in ["ratings", "assemblies", "equipment_noise", "room_acoustic_requirements"]:
        for item in acoustic_data.get(collection_name, []):
            if item.get("page_block_index") in block_index_map:
                item["page_block_index"] = block_index_map[item["page_block_index"]]
    enrich_blocks_from_tables_and_keynotes(normalized)
    return normalized


def block_lookup(blocks: List[dict]) -> Dict[int, dict]:
    return {int(block.get("block_index", idx)): block for idx, block in enumerate(blocks)}


def assign_region_slots_for_blocks(blocks: List[dict], page_image: object) -> Dict[int, Tuple[int, int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, block in enumerate(blocks):
        block_index = int(block.get("block_index", idx))
        if extract_block_bbox_norm(block, page_image):
            continue
        approx_region = ensure_supported_region(block.get("approx_region", "full_page"))
        grouped[approx_region].append(block_index)

    slots: Dict[int, Tuple[int, int]] = {}
    for _, indices in grouped.items():
        ordered = sorted(set(indices))
        total = len(ordered)
        for slot_index, block_index in enumerate(ordered):
            slots[block_index] = (slot_index, total)
    return slots


def process_drawing_page(
    document_meta: dict,
    page_num: int,
    page_image: Optional[object],
    native_context: dict,
    genai_client: Any,
    output_path: Path,
    save_crops: bool,
    verify_mode: str,
    project_context: str,
    llm_model: str,
    extractor_route_overrides: Dict[str, str],
    crop_padding_ratio: float,
    native_vision_fallback: bool,
    native_vision_fallback_max_lines: int,
    native_vision_fallback_max_chars: int,
    log_page_phases: bool,
    *,
    existing_page_image_path: Optional[Path] = None,
    existing_image_bytes: Optional[bytes] = None,
    save_page_image_to_disk: bool = True,
    model_pool: Optional[VisionModelPool] = None,
    extraction_strategy_override: Optional[str] = None,
) -> Tuple[dict, dict, List[dict], int, dict]:
    source_file_name = document_meta.get("source_file_name", "unknown.pdf")
    log_page_phase(log_page_phases, source_file_name, page_num, "phase:start")
    llm_model = normalize_space(llm_model) or DEFAULT_LLM_MODEL

    total_usage = {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "phase_usage": empty_phase_usage(),
    }
    base_extraction_strategy, resolved_extraction_strategy = resolve_page_extraction_strategy(
        document_meta,
        native_context,
        genai_client,
        extractor_route_overrides,
        native_vision_fallback,
        native_vision_fallback_max_lines,
        native_vision_fallback_max_chars,
    )
    extraction_strategy = normalize_space(extraction_strategy_override or resolved_extraction_strategy) or resolved_extraction_strategy
    image_bytes = existing_image_bytes or b""
    if extraction_strategy != "native_only" and not image_bytes:
        if page_image is not None:
            image_bytes = image_to_jpeg_bytes(page_image)
        elif existing_page_image_path and Path(existing_page_image_path).exists():
            image_bytes = read_binary_file(Path(existing_page_image_path))
        else:
            raise RuntimeError(
                f"vision extraction requires an image source for {source_file_name} page {page_num}"
            )

    if base_extraction_strategy == "native_only" and extraction_strategy == "vision_universal":
        log_page_phase(
            log_page_phases,
            source_file_name,
            page_num,
            "phase:strategy:fallback_to_vision",
            (
                f"line_count={native_context.get('line_count', 0)} "
                f"text_chars={len(normalize_space(str(native_context.get('text', ''))))} "
                f"max_lines={max(0, native_vision_fallback_max_lines)} "
                f"max_chars={max(0, native_vision_fallback_max_chars)}"
            ),
        )

    log_page_phase(
        log_page_phases,
        source_file_name,
        page_num,
        "phase:strategy",
        (
            f"document_class={document_meta.get('document_class', 'unknown')} "
            f"base_strategy={base_extraction_strategy} strategy={extraction_strategy}"
        ),
    )

    heuristic_class, heuristic_flags = heuristic_page_class(native_context)
    classification = {
        "header": heuristic_header_from_text(native_context.get("text", "")),
        "page_class": heuristic_class,
        "page_flags": heuristic_flags,
    }
    raw_extraction = heuristic_extraction(classification, native_context)

    if extraction_strategy != "native_only" and genai_client:
        log_page_phase(log_page_phases, source_file_name, page_num, "phase:extract:start")
        raw_extraction, extraction_usage = extract_page_universal(
            image_bytes,
            classification,
            native_context,
            genai_client,
            llm_model,
            project_context,
            model_pool=model_pool,
            page_label=f"{source_file_name} page {page_num:04d}",
            output_path=output_path,
            document_meta=document_meta,
            page_num=page_num,
            image_path=normalize_storage_path(existing_page_image_path) if existing_page_image_path else "",
        )
        total_usage["prompt_tokens"] += extraction_usage.get("prompt_tokens", 0)
        total_usage["output_tokens"] += extraction_usage.get("output_tokens", 0)
        total_usage["thinking_tokens"] += extraction_usage.get("thinking_tokens", 0)
        total_usage["phase_usage"]["extract"]["prompt_tokens"] += extraction_usage.get("prompt_tokens", 0)
        total_usage["phase_usage"]["extract"]["output_tokens"] += extraction_usage.get("output_tokens", 0)
        total_usage["phase_usage"]["extract"]["thinking_tokens"] += extraction_usage.get("thinking_tokens", 0)
        total_usage["phase_usage"]["extract"]["calls"] += 1
        classification = merge_classification_with_extraction(classification, raw_extraction)
        log_page_phase(
            log_page_phases,
            source_file_name,
            page_num,
            "phase:extract:done",
            (
                f"prompt={extraction_usage.get('prompt_tokens', 0)} "
                f"output={extraction_usage.get('output_tokens', 0)} "
                f"model={normalize_space(str(extraction_usage.get('model', llm_model)))} "
                f"failovers={int(extraction_usage.get('failovers', 0) or 0)} "
                f"parse_fallback={int(extraction_usage.get('parse_fallback', 0) or 0)} "
                f"page_class={classification.get('page_class', 'drawing')}"
            ),
        )
    else:
        log_page_phase(log_page_phases, source_file_name, page_num, "phase:classify_extract:skipped", "native_only or no_client")

    log_page_phase(log_page_phases, source_file_name, page_num, "phase:normalize:start")
    extraction = normalize_extraction_output(raw_extraction, classification, native_context)
    log_page_phase(
        log_page_phases,
        source_file_name,
        page_num,
        "phase:normalize:done",
        (
            f"blocks={len(extraction.get('content_blocks', []))} "
            f"entities={len(extraction.get('entities', []))} "
            f"xrefs={len(extraction.get('cross_references', []))}"
        ),
    )

    should_verify = extraction_strategy != "native_only" and should_verify_page(verify_mode, classification, extraction) and bool(genai_client)
    if should_verify:
        log_page_phase(log_page_phases, source_file_name, page_num, "phase:verify:start")
        verification, verification_usage = verify_extraction(
            image_bytes,
            extraction.get("page_summary", "")[:2500],
            genai_client,
            llm_model,
        )
        total_usage["prompt_tokens"] += verification_usage.get("prompt_tokens", 0)
        total_usage["output_tokens"] += verification_usage.get("output_tokens", 0)
        total_usage["thinking_tokens"] += verification_usage.get("thinking_tokens", 0)
        total_usage["phase_usage"]["verify"]["prompt_tokens"] += verification_usage.get("prompt_tokens", 0)
        total_usage["phase_usage"]["verify"]["output_tokens"] += verification_usage.get("output_tokens", 0)
        total_usage["phase_usage"]["verify"]["thinking_tokens"] += verification_usage.get("thinking_tokens", 0)
        total_usage["phase_usage"]["verify"]["calls"] += 1

        extraction = add_verification_findings(extraction, verification)
        extraction["verification"] = verification
        log_page_phase(
            log_page_phases,
            source_file_name,
            page_num,
            "phase:verify:done",
            (
                f"prompt={verification_usage.get('prompt_tokens', 0)} "
                f"output={verification_usage.get('output_tokens', 0)} "
                f"findings={len(verification.get('missing_items', []))}"
            ),
        )
    else:
        verify_reason = "native_only" if extraction_strategy == "native_only" else "policy_skip_or_no_client"
        log_page_phase(log_page_phases, source_file_name, page_num, "phase:verify:skipped", verify_reason)

    page_image_saved = ""
    if existing_page_image_path:
        page_image_saved = normalize_storage_path(Path(existing_page_image_path))
        log_page_phase(log_page_phases, source_file_name, page_num, "phase:save_page_image", "reused_cached_image")
    elif save_page_image_to_disk and page_image is not None:
        log_page_phase(log_page_phases, source_file_name, page_num, "phase:save_page_image")
        page_image_path = output_path / "images" / document_meta["revision_id"] / document_meta["document_id"] / f"page_{page_num:04d}.jpg"
        page_image_saved = save_page_image(page_image, page_image_path)
    else:
        log_page_phase(log_page_phases, source_file_name, page_num, "phase:save_page_image:skipped", "no_page_image_source")

    header = extraction["header"]
    ingestion_diagnostics = extraction.get("ingestion_diagnostics", {}) if isinstance(extraction.get("ingestion_diagnostics"), dict) else {}
    header.update(
        {
            "project_id": document_meta["project_id"],
            "revision_id": document_meta["revision_id"],
            "revision_label": document_meta["revision_label"],
            "revision_sequence": document_meta["revision_sequence"],
            "is_current_revision": bool_payload(document_meta.get("is_current_revision", False)),
            "document_id": document_meta["document_id"],
            "document_family_id": document_meta["document_family_id"],
            "document_class": document_meta["document_class"],
            "retrieval_scope": document_meta.get("retrieval_scope", "all"),
            "source_pdf": document_meta["source_file_name"],
            "source_relative_path": document_meta["source_relative_path"],
            "page_number": page_num,
            "image_path": page_image_saved,
            "page_class": ensure_supported_page_class(classification.get("page_class", "")),
            "extraction_strategy": extraction_strategy,
            "quarantined": bool_payload(ingestion_diagnostics.get("quarantined", False)),
            "quarantine_path": ingestion_diagnostics.get("quarantine_path", ""),
            "low_confidence_extraction": bool_payload(ingestion_diagnostics.get("low_confidence_extraction", False)),
        }
    )

    blocks = []
    crop_root = output_path / "crops" / document_meta["revision_id"] / document_meta["document_id"]
    crop_image = page_image
    crop_image_opened = False
    if save_crops and crop_image is None and page_image_saved and HAS_PIL:
        try:
            crop_image = Image.open(Path(page_image_saved))
            crop_image_opened = True
        except Exception as exc:
            log_page_phase(log_page_phases, source_file_name, page_num, "phase:block_crops:skipped", f"open_failed={exc}")

    log_page_phase(log_page_phases, source_file_name, page_num, "phase:block_crops:start", f"block_count={len(extraction.get('content_blocks', []))}")
    region_slots = assign_region_slots_for_blocks(extraction.get("content_blocks", []), crop_image) if crop_image is not None else {}
    for block in extraction.get("content_blocks", []):
        saved_block = dict(block)
        saved_block["crop_image_path"] = ""
        if crop_image is not None:
            block_index = int(block.get("block_index", 0))
            approx_region = ensure_supported_region(block.get("approx_region", "full_page"))
            crop_path = crop_root / f"page_{page_num:04d}_block_{block_index:03d}.jpg"
            slot_index, slot_count = region_slots.get(block_index, (0, 1))
            crop_image_path, crop_meta = save_block_crop(
                crop_image,
                crop_path,
                block,
                approx_region,
                save_crops,
                slot_index=slot_index,
                slot_count=slot_count,
                crop_padding_ratio=crop_padding_ratio,
            )
            saved_block["crop_image_path"] = crop_image_path
            if crop_meta:
                payload = saved_block.get("structured_payload", {})
                if not isinstance(payload, dict):
                    payload = {"raw": payload}
                payload["crop_meta"] = crop_meta
                saved_block["structured_payload"] = payload
        blocks.append(saved_block)

    if crop_image_opened:
        close_image(crop_image)
    log_page_phase(log_page_phases, source_file_name, page_num, "phase:block_crops:done", f"saved_blocks={len(blocks)}")

    extraction["content_blocks"] = blocks
    log_page_phase(
        log_page_phases,
        source_file_name,
        page_num,
        "phase:complete",
        f"prompt={total_usage['prompt_tokens']} output={total_usage['output_tokens']} thinking={total_usage['thinking_tokens']}",
    )
    return header, extraction, blocks, len(image_bytes), total_usage


def finalize_processed_page(
    manifest: dict,
    page_num: int,
    page_native_context: dict,
    header: dict,
    extraction: dict,
    page_blocks: List[dict],
    log_page_phases: bool,
) -> dict:
    source_file_name = manifest.get("source_file_name", "unknown.pdf")
    fallback_page_class, fallback_flags = heuristic_page_class(page_native_context)
    page_flags = normalize_page_flags(extraction.get("page_flags", {}), fallback_flags)
    header.update(page_flags)
    extraction["page_flags"] = page_flags
    if header.get("page_class") not in SUPPORTED_PAGE_CLASSES:
        header["page_class"] = fallback_page_class

    record = {"header": header, "extraction": extraction}
    block_records = [
        {
            **block,
            "project_id": manifest["project_id"],
            "revision_id": manifest["revision_id"],
            "revision_label": manifest["revision_label"],
            "document_id": manifest["document_id"],
            "document_family_id": manifest["document_family_id"],
            "source_file_name": manifest["source_file_name"],
            "sheet_number": header.get("sheet_number", ""),
            "page_number": page_num,
        }
        for block in page_blocks
    ]

    page_class_counter: Counter = Counter()
    block_type_counter: Counter = Counter()
    page_class_counter[header["page_class"]] += 1
    for block in page_blocks:
        block_type_counter[block.get("block_type", "text_region")] += 1

    blocks_by_index = block_lookup(page_blocks)
    log_page_phase(log_page_phases, source_file_name, page_num, "phase:chunking:start")
    semantic_units = build_semantic_units(header, extraction, blocks_by_index)
    chunks = semantic_units_to_chunks(header, semantic_units)
    log_page_phase(log_page_phases, source_file_name, page_num, "phase:chunking:done")

    return {
        "page_num": page_num,
        "record": record,
        "block_records": block_records,
        "chunks": chunks,
        "page_classes": counter_to_sorted_dict(page_class_counter),
        "block_types": counter_to_sorted_dict(block_type_counter),
    }


def build_document_outputs_from_page_results(page_results_by_num: Dict[int, dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    chunks: List[dict] = []
    page_records: List[dict] = []
    block_records: List[dict] = []
    for page_num in sorted(page_results_by_num):
        page_result = page_results_by_num[page_num]
        page_records.append(page_result["record"])
        block_records.extend(page_result["block_records"])
        chunks.extend(page_result["chunks"])
    return chunks, page_records, block_records


def write_document_checkpoint(
    manifest: dict,
    page_results_by_num: Dict[int, dict],
    page_count: int,
    usage_summary: dict,
    checkpoint_root: Optional[Path],
) -> None:
    if not checkpoint_root:
        return
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    doc_tag = manifest["document_id"]
    chunks, page_records, block_records = build_document_outputs_from_page_results(page_results_by_num)
    write_json(checkpoint_root / f"{doc_tag}_chunks.partial.json", chunks)
    write_json(checkpoint_root / f"{doc_tag}_extractions.partial.json", page_records)
    write_json(checkpoint_root / f"{doc_tag}_blocks.partial.json", block_records)
    write_json(
        checkpoint_root / f"{doc_tag}_stats.partial.json",
        {
            "document_id": manifest["document_id"],
            "source_file_name": manifest["source_file_name"],
            "pages_processed": len(page_records),
            "page_count": page_count,
            "page_classes": counter_to_sorted_dict(Counter(usage_summary.get("page_classes", {}))),
            "block_types": counter_to_sorted_dict(Counter(usage_summary.get("block_types", {}))),
            "total_prompt_tokens": usage_summary.get("total_prompt_tokens", 0),
            "total_output_tokens": usage_summary.get("total_output_tokens", 0),
            "total_tokens": usage_summary.get("total_tokens", 0),
            "token_usage_by_phase": usage_summary.get("phase_usage", empty_phase_usage()),
        },
    )


def render_table_overview_as_text(table: dict) -> str:
    if not isinstance(table, dict):
        return ""
    table_title = normalize_space(str(table.get("table_title", ""))) or "Untitled table"
    columns = [normalize_space(str(item)) for item in table.get("columns", []) if normalize_space(str(item))]
    rows = table.get("rows", [])
    row_count = len(rows) if isinstance(rows, list) else 0
    notes = normalize_space(str(table.get("notes", "")))
    parts = [f"{table_title} table."]
    if columns:
        parts.append(f"Columns: {', '.join(columns)}.")
    parts.append(f"Row count: {row_count}.")
    if notes:
        parts.append(f"Notes: {notes}")
    return " ".join(parts)


def block_text(block: dict) -> str:
    block_type = ensure_supported_block_type(block.get("block_type", "text_region"))
    text = normalize_space(block.get("text", ""))

    payload_text = ""
    if block_type in {"table", "schedule"} and isinstance(block.get("structured_payload"), dict):
        payload_text = render_table_overview_as_text(block.get("structured_payload", {}))
    elif block_type in {"legend", "keynote_block"} and block.get("structured_payload"):
        payload_text = json.dumps(block["structured_payload"], ensure_ascii=False)

    if text and payload_text:
        return f"{text}\n\nStructured payload: {payload_text}"
    return text or payload_text


def base_chunk_metadata(header: dict) -> dict:
    return {
        "project_id": header.get("project_id", ""),
        "revision_id": header.get("revision_id", ""),
        "revision_label": header.get("revision_label", ""),
        "revision_sequence": header.get("revision_sequence", 0),
        "is_current_revision": bool_payload(header.get("is_current_revision", False)),
        "document_id": header.get("document_id", ""),
        "document_family_id": header.get("document_family_id", ""),
        "document_class": header.get("document_class", "drawing_pdf"),
        "retrieval_scope": header.get("retrieval_scope", "all"),
        "extraction_strategy": header.get("extraction_strategy", "vision_universal"),
        "discipline": header.get("discipline", ""),
        "primary_discipline": header.get("discipline", ""),
        "drawing_type": header.get("drawing_type", ""),
        "sheet_number": header.get("sheet_number", ""),
        "page_number": header.get("page_number", 0),
        "source_file_name": header.get("source_pdf", ""),
        "page_class": header.get("page_class", ""),
        "has_drawings": header.get("has_drawings", False),
        "has_tables": header.get("has_tables", False),
        "has_legends": header.get("has_legends", False),
        "has_notes": header.get("has_notes", False),
        "has_renderings": header.get("has_renderings", False),
        "has_photos": header.get("has_photos", header.get("has_renderings", False)),
        "page_contains_multiple_content_types": header.get("page_contains_multiple_content_types", False),
        "image_path": header.get("image_path", ""),
        "quarantined": bool_payload(header.get("quarantined", False)),
        "quarantine_path": header.get("quarantine_path", ""),
        "low_confidence_extraction": bool_payload(header.get("low_confidence_extraction", False)),
    }


def make_semantic_unit(
    unit_type: str,
    text: str,
    metadata: dict,
    priority: str = "normal",
    atomic: bool = True,
) -> dict:
    return {
        "unit_type": unit_type,
        "text": normalize_space(text),
        "metadata": dict(metadata or {}),
        "priority": priority,
        "atomic": bool(atomic),
    }


def render_table_row_as_text(header: dict, table: dict, row: dict, row_index: int) -> str:
    sheet = header.get("sheet_number", "Unknown")
    table_title = normalize_space(str(table.get("table_title", ""))) or "Untitled table"
    parts = []
    for key, value in row.items():
        key_text = normalize_space(str(key))
        value_text = normalize_space(str(value))
        if key_text and value_text:
            parts.append(f"{key_text}: {value_text}")
    row_text = "; ".join(parts) or "No readable row values"
    return f"Table row from {table_title} on sheet {sheet}. Row {row_index + 1}: {row_text}."


def build_page_summary_units(header: dict, extraction: dict) -> List[dict]:
    text = extraction.get("page_summary", "")
    if not normalize_space(str(text)):
        return []
    metadata = base_chunk_metadata(header)
    metadata.update({"chunk_type": "page_summary", "extraction_source": "synthesized_page_summary"})
    return [
        make_semantic_unit(
            "page_summary",
            f"Sheet {header.get('sheet_number', 'Unknown')}: {text}",
            metadata,
            priority="normal",
            atomic=False,
        )
    ]


def build_block_units(header: dict, extraction: dict, blocks_by_index: Dict[int, dict]) -> List[dict]:
    _ = extraction
    units: List[dict] = []
    metadata = base_chunk_metadata(header)
    for block in blocks_by_index.values():
        text = block_text(block)
        if not text:
            continue
        block_type = block.get("block_type", "text_region")
        chunk_type = block_type_to_chunk_type(block_type)
        unit_type = "general_notes" if chunk_type == "notes_block" else "block_note"
        if chunk_type == "table_block":
            unit_type = "table_context"
        elif chunk_type == "legend_block":
            unit_type = "legend_context"
        block_metadata = {
            **metadata,
            "chunk_type": chunk_type,
            "block_type": block_type,
            "block_index": block.get("block_index"),
            "block_label": block.get("block_label", ""),
            "table_title": (
                block.get("structured_payload", {}).get("table_title", "")
                if isinstance(block.get("structured_payload"), dict)
                else ""
            )
            if block_type in {"table", "schedule"}
            else "",
            "approx_region": block.get("approx_region", "full_page"),
            "crop_image_path": block.get("crop_image_path", ""),
            "extraction_source": block.get("extraction_source", ""),
            "confidence": block.get("confidence", 0.5),
        }
        units.append(make_semantic_unit(unit_type, text, block_metadata, priority="normal", atomic=False))
    return units


def build_entities_summary_units(header: dict, extraction: dict) -> List[dict]:
    entities = [
        entity
        for entity in extraction.get("entities", [])
        if isinstance(entity, dict)
    ]
    if not entities:
        return []
    type_counts: Counter = Counter(normalized_entity_type(entity) for entity in entities)
    count_line = ", ".join(f"{etype}: {count}" for etype, count in sorted(type_counts.items()))
    lines = [
        f"Entities summary for sheet {header.get('sheet_number', 'Unknown')}.",
        f"Total entities: {len(entities)}.",
        f"Entity type counts: {count_line}",
    ]
    for entity in entities:
        lines.append(f"- {build_entity_text(header, entity)}")
    metadata = base_chunk_metadata(header)
    metadata.update(
        {
            "chunk_type": "entities_summary",
            "entity_count": len(entities),
            "extraction_source": "synthesized_entities_summary",
        }
    )
    return [make_semantic_unit("entities_summary", "\n".join(lines), metadata, atomic=False)]


def build_entity_units(header: dict, extraction: dict, blocks_by_index: Dict[int, dict]) -> List[dict]:
    units: List[dict] = []
    metadata = base_chunk_metadata(header)
    for entity_index, entity in enumerate(extraction.get("entities", [])):
        if not isinstance(entity, dict) or not is_high_value_entity(entity):
            continue
        block_index = entity.get("page_block_index")
        block = blocks_by_index.get(block_index) if block_index is not None else None
        text = build_entity_text(header, entity, block)
        entity_metadata = {
            **metadata,
            "chunk_type": "entity",
            "block_type": block.get("block_type", "") if block else "",
            "block_index": block_index,
            "entity_index": entity_index,
            "entity_type": entity.get("entity_type", ""),
            "entity_id": entity.get("entity_id", ""),
            "entity_name": entity.get("entity_name", ""),
            "location_on_drawing": entity.get("location_on_drawing", ""),
            "crop_image_path": block.get("crop_image_path", "") if block else "",
            "extraction_source": block.get("extraction_source", "") if block else "",
            "entity_priority": "high_value",
        }
        units.append(make_semantic_unit("entity", text, entity_metadata, priority="high", atomic=True))
    return units


def build_table_row_units(header: dict, extraction: dict, blocks_by_index: Dict[int, dict]) -> List[dict]:
    units: List[dict] = []
    metadata = base_chunk_metadata(header)
    for table_index, table in enumerate(extraction.get("tables_on_sheet", [])):
        if not isinstance(table, dict):
            continue
        rows = table.get("rows", [])
        if not isinstance(rows, list):
            continue
        block_index = table.get("page_block_index")
        block = blocks_by_index.get(block_index) if block_index is not None else None
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            text = render_table_row_as_text(header, table, row, row_index)
            row_metadata = {
                **metadata,
                "chunk_type": "table_row",
                "table_title": table.get("table_title", ""),
                "table_index": table_index,
                "row_index": row_index,
                "page_block_index": block_index,
                "block_index": block_index,
                "block_type": block.get("block_type", "") if block else "",
                "crop_image_path": block.get("crop_image_path", "") if block else "",
                "extraction_source": "table_row_renderer",
            }
            units.append(make_semantic_unit("table_row", text, row_metadata, priority="high", atomic=True))
    return units


def build_keynote_units(header: dict, extraction: dict, blocks_by_index: Dict[int, dict]) -> List[dict]:
    units: List[dict] = []
    metadata = base_chunk_metadata(header)
    for item_index, item in enumerate(extraction.get("keynotes_or_legends", [])):
        if not isinstance(item, dict):
            continue
        number = normalize_space(str(item.get("number", "")))
        item_text = normalize_space(str(item.get("text", "")))
        if not item_text:
            continue
        block_index = item.get("page_block_index")
        block = blocks_by_index.get(block_index) if block_index is not None else None
        rendered = f"Keynote or legend item {number} on sheet {header.get('sheet_number', 'Unknown')}: {item_text}"
        keynote_metadata = {
            **metadata,
            "chunk_type": "keynote",
            "keynote_number": number,
            "keynote_index": item_index,
            "block_type": item.get("block_type", "legend"),
            "page_block_index": block_index,
            "block_index": block_index,
            "crop_image_path": block.get("crop_image_path", "") if block else "",
            "extraction_source": "keynote_renderer",
        }
        units.append(make_semantic_unit("keynote", rendered, keynote_metadata, priority="high", atomic=True))
    return units


def render_acoustic_rating(header: dict, rating: dict) -> str:
    sheet = header.get("sheet_number", "Unknown")
    return (
        f"Acoustic rating on sheet {sheet}: "
        f"{rating.get('rating_type', 'rating')} {rating.get('value', '')} {rating.get('unit', '')}. "
        f"Applies to {rating.get('applies_to', 'unknown subject')}. "
        f"Assembly: {rating.get('assembly_id', '')}. "
        f"Room: {rating.get('room_id', '')}. "
        f"Equipment: {rating.get('equipment_id', '')}. "
        f"Location: {rating.get('location_on_drawing', '')}. "
        f"Source text: {rating.get('source_text', '')}."
    )


def render_acoustic_assembly(header: dict, assembly: dict) -> str:
    layers = assembly.get("layers", [])
    layer_text = "; ".join(normalize_space(str(item)) for item in layers if normalize_space(str(item))) if isinstance(layers, list) else ""
    ratings = []
    for key, label in [("rated_stc", "STC"), ("rated_iic", "IIC"), ("rated_nrc", "NRC"), ("fire_rating", "fire rating")]:
        value = normalize_space(str(assembly.get(key, "")))
        if value:
            ratings.append(f"{label}: {value}")
    return (
        f"Acoustic assembly on sheet {header.get('sheet_number', 'Unknown')}: "
        f"{assembly.get('assembly_id', '')} ({assembly.get('assembly_type', 'assembly')}). "
        f"Description: {assembly.get('description', '')}. "
        f"{'; '.join(ratings)}. "
        f"Layers: {layer_text}. "
        f"Source text: {assembly.get('source_text', '')}."
    )


def render_equipment_noise(header: dict, item: dict) -> str:
    return (
        f"Equipment noise on sheet {header.get('sheet_number', 'Unknown')}: "
        f"{item.get('equipment_id', '')} {item.get('equipment_type', '')} has "
        f"{item.get('noise_metric', 'noise metric')} {item.get('value', '')}. "
        f"Location: {item.get('location', '')}. Source text: {item.get('source_text', '')}."
    )


def render_room_acoustic_requirement(header: dict, item: dict) -> str:
    room = normalize_space(" ".join(str(item.get(key, "")) for key in ["room_id", "room_name"]))
    return (
        f"Room acoustic requirement on sheet {header.get('sheet_number', 'Unknown')}: "
        f"{room or 'unknown room'} requires {item.get('requirement_type', 'requirement')} "
        f"{item.get('requirement_value', '')}. Source text: {item.get('source_text', '')}."
    )


def build_acoustic_units(header: dict, extraction: dict, blocks_by_index: Dict[int, dict]) -> List[dict]:
    units: List[dict] = []
    metadata = base_chunk_metadata(header)
    acoustic = normalize_acoustic_data(extraction)

    for idx, rating in enumerate(acoustic.get("ratings", [])):
        block_index = rating.get("page_block_index")
        block = blocks_by_index.get(block_index) if block_index is not None else None
        rating_metadata = {
            **metadata,
            "chunk_type": "acoustic_rating",
            "rating_type": rating.get("rating_type", ""),
            "rating_value": rating.get("value", ""),
            "applies_to": rating.get("applies_to", ""),
            "assembly_id": rating.get("assembly_id", ""),
            "room_id": rating.get("room_id", ""),
            "equipment_id": rating.get("equipment_id", ""),
            "acoustic_fact_index": idx,
            "page_block_index": block_index,
            "block_index": block_index,
            "block_type": block.get("block_type", "") if block else "",
            "crop_image_path": block.get("crop_image_path", "") if block else "",
            "extraction_source": "acoustic_extractor",
        }
        units.append(make_semantic_unit("acoustic_rating", render_acoustic_rating(header, rating), rating_metadata, priority="critical", atomic=True))

    for idx, assembly in enumerate(acoustic.get("assemblies", [])):
        block_index = assembly.get("page_block_index")
        assembly_metadata = {
            **metadata,
            "chunk_type": "acoustic_assembly",
            "assembly_id": assembly.get("assembly_id", ""),
            "assembly_type": assembly.get("assembly_type", ""),
            "rating_type": "assembly",
            "rating_value": assembly.get("rated_stc") or assembly.get("rated_iic") or assembly.get("rated_nrc") or "",
            "page_block_index": block_index,
            "block_index": block_index,
            "extraction_source": "acoustic_extractor",
        }
        units.append(make_semantic_unit("acoustic_assembly", render_acoustic_assembly(header, assembly), assembly_metadata, priority="critical", atomic=True))

    for idx, item in enumerate(acoustic.get("equipment_noise", [])):
        block_index = item.get("page_block_index")
        equipment_metadata = {
            **metadata,
            "chunk_type": "equipment_noise",
            "equipment_id": item.get("equipment_id", ""),
            "rating_type": item.get("noise_metric", ""),
            "rating_value": item.get("value", ""),
            "page_block_index": block_index,
            "block_index": block_index,
            "extraction_source": "acoustic_extractor",
        }
        units.append(make_semantic_unit("equipment_noise", render_equipment_noise(header, item), equipment_metadata, priority="critical", atomic=True))

    for idx, item in enumerate(acoustic.get("room_acoustic_requirements", [])):
        block_index = item.get("page_block_index")
        room_metadata = {
            **metadata,
            "chunk_type": "room_acoustic_requirement",
            "room_id": item.get("room_id", ""),
            "rating_type": item.get("requirement_type", ""),
            "rating_value": item.get("requirement_value", ""),
            "page_block_index": block_index,
            "block_index": block_index,
            "extraction_source": "acoustic_extractor",
        }
        units.append(make_semantic_unit("room_acoustic_requirement", render_room_acoustic_requirement(header, item), room_metadata, priority="critical", atomic=True))

    return units


def build_cross_reference_units(header: dict, extraction: dict, blocks_by_index: Dict[int, dict]) -> List[dict]:
    units: List[dict] = []
    xrefs = [xref for xref in extraction.get("cross_references", []) if isinstance(xref, dict)]
    if not xrefs:
        return units
    metadata = base_chunk_metadata(header)

    lines = [
        f"Cross-reference summary for sheet {header.get('sheet_number', 'Unknown')}.",
        f"Total cross references: {len(xrefs)}.",
    ]
    for xref in xrefs:
        reference_text = normalize_space(xref.get("reference_text", "")) or "ILLEGIBLE"
        target = normalize_space(xref.get("target_sheet_number", "")) or "UNSPECIFIED"
        ref_type = normalize_space(xref.get("type", "")) or "unknown"
        context = normalize_space(xref.get("context", ""))
        line = f"- {reference_text} -> {target} ({ref_type})"
        if context:
            line += f": {context}"
        lines.append(line)

    summary_metadata = {
        **metadata,
        "chunk_type": "cross_refs_summary",
        "cross_reference_count": len(xrefs),
        "extraction_source": "synthesized_cross_refs_summary",
    }
    units.append(make_semantic_unit("cross_refs_summary", "\n".join(lines), summary_metadata, priority="normal", atomic=False))

    for idx, xref in enumerate(xrefs):
        block_index = xref.get("page_block_index")
        block = blocks_by_index.get(block_index) if block_index is not None else None
        ref_type = xref.get("ref_type") or xref.get("type", "")
        text = (
            f"Cross-reference on sheet {header.get('sheet_number', 'Unknown')}: "
            f"{xref.get('reference_text', 'ILLEGIBLE')} of type {ref_type or 'unknown'} "
            f"points to sheet {xref.get('target_sheet_number', 'UNSPECIFIED')}. "
            f"Context: {xref.get('context', '')}."
        )
        xref_metadata = {
            **metadata,
            "chunk_type": "cross_reference",
            "reference_text": xref.get("reference_text", ""),
            "ref_type": ref_type,
            "target_sheet_number": xref.get("target_sheet_number", ""),
            "cross_reference_index": idx,
            "page_block_index": block_index,
            "block_index": block_index,
            "block_type": block.get("block_type", "") if block else "",
            "crop_image_path": block.get("crop_image_path", "") if block else "",
            "resolved": bool_payload(xref.get("resolved", False)),
            "extraction_source": "cross_reference_renderer",
        }
        units.append(make_semantic_unit("cross_reference", text, xref_metadata, priority="high", atomic=True))
    return units


def merge_units(units: List[dict]) -> dict:
    if not units:
        return make_semantic_unit("merged_context", "", {}, atomic=False)
    metadata = dict(units[0].get("metadata", {}))
    unit_types = [unit.get("unit_type", "") for unit in units]
    chunk_types = [unit.get("metadata", {}).get("chunk_type", unit.get("unit_type", "")) for unit in units]
    metadata["chunk_type"] = chunk_types[0] if len(set(chunk_types)) == 1 else "merged_context"
    metadata["merged_unit_types"] = sorted(set(unit_types))
    metadata["merged_unit_count"] = len(units)
    metadata["extraction_source"] = "semantic_unit_merge"
    text = "\n\n".join(unit.get("text", "") for unit in units if unit.get("text"))
    return make_semantic_unit(metadata["chunk_type"], text, metadata, priority="normal", atomic=False)


def normalize_semantic_units_to_sized_units(units: List[dict]) -> List[dict]:
    sized: List[dict] = []
    pending: List[dict] = []

    def flush_pending() -> None:
        nonlocal pending
        if pending:
            sized.append(merge_units(pending))
            pending = []

    for unit in units:
        text = unit.get("text", "")
        if not text:
            continue
        unit_type = unit.get("unit_type", "")
        text_len = len(text)

        if unit_type in ALLOW_TINY_CHUNK_TYPES:
            flush_pending()
            sized.append(unit)
            continue

        if text_len < CHUNK_MIN_CHARS and unit_type in MERGE_COMPATIBLE_TYPES:
            pending.append(unit)
            merged_len = sum(len(item.get("text", "")) for item in pending)
            if merged_len >= CHUNK_TARGET_CHARS:
                flush_pending()
            continue

        flush_pending()
        sized.append(unit)

    flush_pending()
    return sized


def semantic_units_to_chunks(header: dict, units: List[dict]) -> List[dict]:
    chunks: List[dict] = []
    source_file_name = header.get("source_file_name", "")
    for unit in normalize_semantic_units_to_sized_units(units):
        text = unit.get("text", "")
        if not text:
            continue
        
        # Add filename context to text to ensure retrieval finds technical terms in filenames (like "PS-4W")
        if source_file_name and source_file_name not in text:
            text = f"[{source_file_name}] {text}"
            
        metadata = dict(unit.get("metadata", {}))
        metadata.setdefault("chunk_type", unit.get("unit_type", "semantic_unit"))
        metadata["semantic_unit_type"] = unit.get("unit_type", "")
        metadata["semantic_priority"] = unit.get("priority", "normal")
        overlap = 0 if unit.get("atomic", True) else CHUNK_OVERLAP_CHARS
        chunks.extend(build_split_chunks(metadata, text, max_chars=CHUNK_MAX_CHARS, overlap_chars=overlap))
    return dedupe_chunks(chunks)


def dedupe_chunks(chunks: List[dict]) -> List[dict]:
    seen = set()
    deduped: List[dict] = []
    for chunk in chunks:
        chunk_id = make_deterministic_chunk_id(chunk)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(chunk)
    return deduped


def build_semantic_units(header: dict, extraction: dict, blocks_by_index: Dict[int, dict]) -> List[dict]:
    units: List[dict] = []
    units.extend(build_page_summary_units(header, extraction))
    units.extend(build_block_units(header, extraction, blocks_by_index))
    units.extend(build_table_row_units(header, extraction, blocks_by_index))
    units.extend(build_keynote_units(header, extraction, blocks_by_index))
    units.extend(build_entities_summary_units(header, extraction))
    units.extend(build_entity_units(header, extraction, blocks_by_index))
    units.extend(build_acoustic_units(header, extraction, blocks_by_index))
    units.extend(build_cross_reference_units(header, extraction, blocks_by_index))
    return units


def summarize_chunk_sizes(chunks: List[dict]) -> dict:
    lengths = sorted(len(chunk.get("text_to_embed") or chunk.get("text") or "") for chunk in chunks)
    if not lengths:
        return {}

    def percentile(p: int) -> int:
        idx = int((p / 100) * (len(lengths) - 1))
        return lengths[idx]

    return {
        "count": len(lengths),
        "min": lengths[0],
        "p25": percentile(25),
        "p50": percentile(50),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
        "max": lengths[-1],
        "under_250_chars": sum(1 for value in lengths if value < 250),
        "over_1400_chars": sum(1 for value in lengths if value > 1400),
    }


def block_type_to_chunk_type(block_type: str) -> str:
    mapping = {
        "drawing_view": "drawing_block",
        "detail_view": "drawing_block",
        "table": "table_block",
        "schedule": "table_block",
        "legend": "legend_block",
        "keynote_block": "legend_block",
        "general_notes": "notes_block",
        "title_block": "notes_block",
        "photo_or_rendering": "rendering_block",
        "text_region": "notes_block",
    }
    return mapping.get(block_type, "notes_block")


def build_entity_text(header: dict, entity: dict, block: Optional[dict] = None) -> str:
    attr_lines = []
    attributes = entity.get("attributes", {}) if isinstance(entity.get("attributes"), dict) else {}
    for key, value in attributes.items():
        label = key.replace("_", " ").title()
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        attr_lines.append(f"{label}: {str(value)}")

    text = f"{entity.get('entity_id', 'ILLEGIBLE')}"
    if entity.get("entity_name"):
        text += f" ({entity['entity_name']})"
    text += f" is a {entity.get('entity_type', 'unknown')} on sheet {header.get('sheet_number', 'Unknown')}."
    if entity.get("location_on_drawing"):
        text += f" Location: {entity['location_on_drawing']}."
    if block:
        text += f" Block: {block.get('block_label', '')}."
    if attr_lines:
        text += " " + " | ".join(attr_lines)
    return text


def build_cross_reference_text(item: dict) -> str:
    text = f"Cross-reference {item.get('reference_text', 'ILLEGIBLE')}"
    ref_type = item.get("ref_type") or item.get("type")
    if ref_type:
        text += f" ({ref_type})"
    if item.get("context"):
        text += f": {item['context']}"
    if bool_payload(item.get("resolved", False)):
        text += " [RESOLVED]"
    else:
        text += " [UNRESOLVED]"
        unresolved_reason = normalize_space(str(item.get("unresolved_reason", "")))
        if unresolved_reason:
            text += f" ({unresolved_reason})"
    if item.get("resolution_scope"):
        text += f" [scope={item.get('resolution_scope')}]"
    return text


def sync_cross_reference_chunk(chunk: dict, resolver: "CrossReferenceResolver") -> None:
    resolver.annotate_reference(chunk.get("revision_id", ""), chunk)
    text = build_cross_reference_text(chunk)
    chunk["text"] = text
    chunk["text_to_embed"] = text


def make_deterministic_chunk_id(chunk: dict) -> str:
    revision_id = chunk.get("revision_id", "")
    document_id = chunk.get("document_id", "")
    page_number = chunk.get("page_number", 0)
    chunk_type = chunk.get("chunk_type", "")
    parts = [revision_id, document_id, str(page_number), chunk_type]
    if chunk.get("chunk_sub_id") is not None:
        parts.extend(["sub", str(chunk.get("chunk_sub_id")), str(chunk.get("chunk_sub_count", ""))])

    if chunk_type == "entity":
        entity_text_key = normalize_space(str(chunk.get("text_to_embed") or chunk.get("text") or ""))[:300]
        parts.extend(
            [
                chunk.get("entity_type", ""),
                chunk.get("entity_id", ""),
                str(chunk.get("block_index", "")),
                str(chunk.get("entity_index", "")),
                chunk.get("entity_name", ""),
                chunk.get("location_on_drawing", ""),
                entity_text_key,
            ]
        )
    elif chunk_type == "table_row":
        parts.extend(
            [
                str(chunk.get("table_index", "")),
                str(chunk.get("row_index", "")),
                chunk.get("table_title", ""),
                normalize_space(str(chunk.get("text_to_embed") or chunk.get("text") or ""))[:300],
            ]
        )
    elif chunk_type == "keynote":
        parts.extend(
            [
                str(chunk.get("keynote_index", "")),
                chunk.get("keynote_number", ""),
                str(chunk.get("block_index", "")),
                normalize_space(str(chunk.get("text_to_embed") or chunk.get("text") or ""))[:300],
            ]
        )
    elif chunk_type in {"acoustic_rating", "acoustic_assembly", "equipment_noise", "room_acoustic_requirement"}:
        parts.extend(
            [
                str(chunk.get("acoustic_fact_index", "")),
                chunk.get("rating_type", ""),
                chunk.get("rating_value", ""),
                chunk.get("assembly_id", ""),
                chunk.get("room_id", ""),
                chunk.get("equipment_id", ""),
                str(chunk.get("block_index", "")),
                normalize_space(str(chunk.get("text_to_embed") or chunk.get("text") or ""))[:300],
            ]
        )
    elif chunk_type == "cross_reference":
        parts.extend(
            [
                str(chunk.get("cross_reference_index", "")),
                chunk.get("reference_text", ""),
                chunk.get("target_sheet_number", ""),
                chunk.get("ref_type", ""),
                str(chunk.get("block_index", "")),
                normalize_space(str(chunk.get("text_to_embed") or chunk.get("text") or ""))[:300],
            ]
        )
    elif chunk_type in {"entities_summary", "cross_refs_summary"}:
        parts.extend([str(chunk.get("entity_count", "")), str(chunk.get("cross_reference_count", ""))])
    elif chunk_type == "page_summary":
        pass
    elif chunk_type in {"drawing_block", "table_block", "legend_block", "notes_block", "rendering_block"}:
        parts.extend([str(chunk.get("block_index", "")), chunk.get("block_type", "")])
    else:
        parts.extend([str(chunk.get("block_index", "")), normalize_space(chunk.get("text", ""))[:300]])
    safe_parts = ["" if value is None else str(value) for value in parts]
    return sha256_text("||".join(safe_parts))


def iter_acoustic_fact_records(extraction: dict) -> List[dict]:
    records: List[dict] = []
    acoustic = normalize_acoustic_data(extraction)

    for item in acoustic.get("ratings", []):
        records.append(
            {
                "fact_type": "rating",
                "subject_id": item.get("assembly_id") or item.get("room_id") or item.get("equipment_id") or "",
                "subject_type": item.get("applies_to", ""),
                "metric": item.get("rating_type", ""),
                "value": item.get("value", ""),
                "unit": item.get("unit", ""),
                "source_text": item.get("source_text", ""),
                "location_on_drawing": item.get("location_on_drawing", ""),
                "confidence": clamp_confidence(item.get("confidence", 0.5)),
                "page_block_index": item.get("page_block_index"),
                "attributes": item,
            }
        )

    for item in acoustic.get("assemblies", []):
        rating_values = {
            key: item.get(key, "")
            for key in ["rated_stc", "rated_iic", "rated_nrc", "fire_rating"]
            if normalize_space(str(item.get(key, "")))
        }
        records.append(
            {
                "fact_type": "assembly",
                "subject_id": item.get("assembly_id", ""),
                "subject_type": item.get("assembly_type", ""),
                "metric": "assembly_rating",
                "value": json.dumps(rating_values, ensure_ascii=False) if rating_values else "",
                "unit": "",
                "source_text": item.get("source_text", ""),
                "location_on_drawing": "",
                "confidence": clamp_confidence(item.get("confidence", 0.5)),
                "page_block_index": item.get("page_block_index"),
                "attributes": item,
            }
        )

    for item in acoustic.get("equipment_noise", []):
        records.append(
            {
                "fact_type": "equipment_noise",
                "subject_id": item.get("equipment_id", ""),
                "subject_type": item.get("equipment_type", ""),
                "metric": item.get("noise_metric", ""),
                "value": item.get("value", ""),
                "unit": "",
                "source_text": item.get("source_text", ""),
                "location_on_drawing": item.get("location", ""),
                "confidence": clamp_confidence(item.get("confidence", 0.5)),
                "page_block_index": item.get("page_block_index"),
                "attributes": item,
            }
        )

    for item in acoustic.get("room_acoustic_requirements", []):
        records.append(
            {
                "fact_type": "room_requirement",
                "subject_id": item.get("room_id", ""),
                "subject_type": "room",
                "metric": item.get("requirement_type", ""),
                "value": item.get("requirement_value", ""),
                "unit": "",
                "source_text": item.get("source_text", ""),
                "location_on_drawing": item.get("room_name", ""),
                "confidence": clamp_confidence(item.get("confidence", 0.5)),
                "page_block_index": item.get("page_block_index"),
                "attributes": item,
            }
        )

    return records


class DatabaseManager:
    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        self.conn = None

    def connect(self):
        if not HAS_PSYCOPG:
            raise RuntimeError("psycopg not available")
        self.conn = psycopg.connect(self.conn_string)
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()

    def init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute(DB_SCHEMA)
        self.conn.commit()

    def insert_project(self, project_id: str, project_name: str = "", project_number: str = ""):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO projects (project_id, project_name, project_number)
                VALUES (%s, %s, %s)
                ON CONFLICT (project_id) DO UPDATE
                SET project_name = EXCLUDED.project_name,
                    project_number = EXCLUDED.project_number
                """,
                (project_id, project_name, project_number),
            )
        self.conn.commit()

    def start_ingestion_run(self, project_id: str) -> str:
        run_id = str(uuid_lib.uuid4())
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_runs (run_id, project_id, status)
                VALUES (%s, %s, %s)
                """,
                (run_id, project_id, "running"),
            )
        self.conn.commit()
        return run_id

    def finish_ingestion_run(self, run_id: str, status: str, stats: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_runs
                SET completed_at = now(), status = %s, stats = %s
                WHERE run_id = %s
                """,
                (status, json.dumps(stats), run_id),
            )
        self.conn.commit()

    def upsert_revision(self, manifest: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO project_revisions (
                    revision_id, project_id, revision_label, revision_sequence, revision_source,
                    revision_date, revision_confidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (revision_id) DO UPDATE
                SET revision_label = EXCLUDED.revision_label,
                    revision_sequence = EXCLUDED.revision_sequence,
                    revision_source = EXCLUDED.revision_source,
                    revision_date = EXCLUDED.revision_date,
                    revision_confidence = EXCLUDED.revision_confidence
                """,
                (
                    manifest["revision_id"],
                    manifest["project_id"],
                    manifest["revision_label"],
                    manifest["revision_sequence"],
                    manifest["revision_source"],
                    manifest.get("revision_date") or None,
                    manifest.get("revision_confidence", 0.0),
                ),
            )

    def upsert_document(self, manifest: dict, page_count: int):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (
                    document_id, project_id, revision_id, document_family_id, document_class,
                    source_relative_path, source_file_name, file_sha256, issue_date, page_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE
                SET page_count = EXCLUDED.page_count,
                    file_sha256 = EXCLUDED.file_sha256,
                    issue_date = EXCLUDED.issue_date
                """,
                (
                    manifest["document_id"],
                    manifest["project_id"],
                    manifest["revision_id"],
                    manifest["document_family_id"],
                    manifest["document_class"],
                    manifest["source_relative_path"],
                    manifest["source_file_name"],
                    manifest["file_sha256"],
                    manifest.get("issue_date") or None,
                    page_count,
                ),
            )

    def insert_sheet(self, header: dict, extraction: dict) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sheets (
                    project_id, revision_id, document_id, document_family_id, source_relative_path,
                    source_file_name, document_class, sheet_number, sheet_title, discipline,
                    drawing_type, scale, page_number, page_class, image_path, page_summary,
                    title_block_revision, title_block_date, has_drawings, has_tables,
                    has_legends, has_notes, has_renderings, page_contains_multiple_content_types,
                    raw_extraction
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (document_id, page_number) DO UPDATE
                SET sheet_number = EXCLUDED.sheet_number,
                    sheet_title = EXCLUDED.sheet_title,
                    discipline = EXCLUDED.discipline,
                    drawing_type = EXCLUDED.drawing_type,
                    scale = EXCLUDED.scale,
                    page_class = EXCLUDED.page_class,
                    image_path = EXCLUDED.image_path,
                    page_summary = EXCLUDED.page_summary,
                    title_block_revision = EXCLUDED.title_block_revision,
                    title_block_date = EXCLUDED.title_block_date,
                    has_drawings = EXCLUDED.has_drawings,
                    has_tables = EXCLUDED.has_tables,
                    has_legends = EXCLUDED.has_legends,
                    has_notes = EXCLUDED.has_notes,
                    has_renderings = EXCLUDED.has_renderings,
                    page_contains_multiple_content_types = EXCLUDED.page_contains_multiple_content_types,
                    raw_extraction = EXCLUDED.raw_extraction
                RETURNING id
                """,
                (
                    header["project_id"],
                    header["revision_id"],
                    header["document_id"],
                    header["document_family_id"],
                    header["source_relative_path"],
                    header["source_pdf"],
                    header["document_class"],
                    header.get("sheet_number", ""),
                    header.get("sheet_title", ""),
                    header.get("discipline", ""),
                    header.get("drawing_type", ""),
                    header.get("scale", ""),
                    header.get("page_number", 0),
                    header.get("page_class", ""),
                    header.get("image_path", ""),
                    extraction.get("page_summary", ""),
                    header.get("title_block_revision", ""),
                    header.get("title_block_date", ""),
                    bool_payload(header.get("has_drawings", False)),
                    bool_payload(header.get("has_tables", False)),
                    bool_payload(header.get("has_legends", False)),
                    bool_payload(header.get("has_notes", False)),
                    bool_payload(header.get("has_renderings", False)),
                    bool_payload(header.get("page_contains_multiple_content_types", False)),
                    json.dumps(extraction),
                ),
            )
            return cur.fetchone()[0]

    def insert_media_asset(
        self,
        header: dict,
        sheet_id: int,
        asset_type: str,
        asset_path: str,
        page_block_id: Optional[int] = None,
        metadata: Optional[dict] = None,
    ):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO media_assets (
                    project_id, revision_id, document_id, sheet_id, page_block_id, asset_type, asset_path, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    header["project_id"],
                    header["revision_id"],
                    header["document_id"],
                    sheet_id,
                    page_block_id,
                    asset_type,
                    asset_path,
                    json.dumps(metadata or {}),
                ),
            )

    def insert_page_block(self, sheet_id: int, block: dict) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO page_blocks (
                    sheet_id, block_index, block_type, block_label, text, structured_payload,
                    approx_region, crop_image_path, extraction_source, confidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (sheet_id, block_index) DO UPDATE
                SET block_type = EXCLUDED.block_type,
                    block_label = EXCLUDED.block_label,
                    text = EXCLUDED.text,
                    structured_payload = EXCLUDED.structured_payload,
                    approx_region = EXCLUDED.approx_region,
                    crop_image_path = EXCLUDED.crop_image_path,
                    extraction_source = EXCLUDED.extraction_source,
                    confidence = EXCLUDED.confidence
                RETURNING id
                """,
                (
                    sheet_id,
                    block["block_index"],
                    block["block_type"],
                    block.get("block_label", ""),
                    block.get("text", ""),
                    json.dumps(block.get("structured_payload", {})),
                    block.get("approx_region", "full_page"),
                    block.get("crop_image_path", ""),
                    block.get("extraction_source", ""),
                    block.get("confidence", 0.5),
                ),
            )
            return cur.fetchone()[0]

    def clear_sheet_children(self, sheet_id: int):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM media_assets WHERE sheet_id = %s", (sheet_id,))
            cur.execute("DELETE FROM entities WHERE sheet_id = %s", (sheet_id,))
            cur.execute("DELETE FROM cross_references WHERE sheet_id = %s", (sheet_id,))
            cur.execute("DELETE FROM tables_on_sheet WHERE sheet_id = %s", (sheet_id,))
            cur.execute("DELETE FROM keynotes WHERE sheet_id = %s", (sheet_id,))
            cur.execute("DELETE FROM acoustic_facts WHERE sheet_id = %s", (sheet_id,))
            cur.execute("DELETE FROM page_blocks WHERE sheet_id = %s", (sheet_id,))

    def insert_entity(self, header: dict, sheet_id: int, page_block_id: Optional[int], entity: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entities (
                    sheet_id, page_block_id, project_id, revision_id, document_id,
                    entity_type, entity_id, entity_name, attributes, location_on_drawing
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    sheet_id,
                    page_block_id,
                    header["project_id"],
                    header["revision_id"],
                    header["document_id"],
                    entity.get("entity_type", ""),
                    entity.get("entity_id", ""),
                    entity.get("entity_name", ""),
                    json.dumps(entity.get("attributes", {})),
                    entity.get("location_on_drawing", ""),
                ),
            )

    def insert_cross_reference(self, header: dict, sheet_id: int, page_block_id: Optional[int], xref: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cross_references (
                    sheet_id, page_block_id, project_id, revision_id, document_id,
                    reference_text, ref_type, context, target_sheet_number, resolved
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    sheet_id,
                    page_block_id,
                    header["project_id"],
                    header["revision_id"],
                    header["document_id"],
                    xref.get("reference_text", ""),
                    xref.get("type", ""),
                    xref.get("context", ""),
                    xref.get("target_sheet_number", ""),
                    bool_payload(xref.get("resolved", False)),
                ),
            )

    def insert_table(self, sheet_id: int, page_block_id: Optional[int], table: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tables_on_sheet (sheet_id, page_block_id, table_title, columns, rows, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    sheet_id,
                    page_block_id,
                    table.get("table_title", ""),
                    json.dumps(table.get("columns", [])),
                    json.dumps(table.get("rows", [])),
                    table.get("notes", ""),
                ),
            )

    def insert_keynote(self, sheet_id: int, page_block_id: Optional[int], item: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO keynotes (sheet_id, page_block_id, keynote_number, keynote_text, block_type)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    sheet_id,
                    page_block_id,
                    item.get("number", ""),
                    item.get("text", ""),
                    item.get("block_type", "legend"),
                ),
            )

    def insert_acoustic_fact(self, header: dict, sheet_id: int, page_block_id: Optional[int], fact: dict):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO acoustic_facts (
                    sheet_id, page_block_id, project_id, revision_id, document_id,
                    fact_type, subject_id, subject_type, metric, value, unit,
                    source_text, location_on_drawing, confidence, attributes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    sheet_id,
                    page_block_id,
                    header["project_id"],
                    header["revision_id"],
                    header["document_id"],
                    fact.get("fact_type", ""),
                    fact.get("subject_id", ""),
                    fact.get("subject_type", ""),
                    fact.get("metric", ""),
                    fact.get("value", ""),
                    fact.get("unit", ""),
                    fact.get("source_text", ""),
                    fact.get("location_on_drawing", ""),
                    fact.get("confidence", 0.5),
                    json.dumps(fact.get("attributes", {})),
                ),
            )

    def insert_page_record(self, manifest: dict, header: dict, extraction: dict):
        with self.conn.transaction():
            self.upsert_revision(manifest)
            self.upsert_document(manifest, manifest.get("page_count", 0))
            sheet_id = self.insert_sheet(header, extraction)
            self.clear_sheet_children(sheet_id)
            self.insert_media_asset(header, sheet_id, "page_image", header.get("image_path", ""), metadata={"page_number": header.get("page_number")})

            page_block_ids: Dict[int, int] = {}
            for block in extraction.get("content_blocks", []):
                block_id = self.insert_page_block(sheet_id, block)
                page_block_ids[int(block["block_index"])] = block_id
                if block.get("crop_image_path"):
                    self.insert_media_asset(
                        header,
                        sheet_id,
                        "block_crop",
                        block["crop_image_path"],
                        page_block_id=block_id,
                        metadata={"block_type": block.get("block_type"), "block_index": block.get("block_index")},
                    )

            for entity in extraction.get("entities", []):
                page_block_id = page_block_ids.get(entity.get("page_block_index"))
                self.insert_entity(header, sheet_id, page_block_id, entity)

            for xref in extraction.get("cross_references", []):
                page_block_id = page_block_ids.get(xref.get("page_block_index"))
                self.insert_cross_reference(header, sheet_id, page_block_id, xref)

            for table in extraction.get("tables_on_sheet", []):
                page_block_id = page_block_ids.get(table.get("page_block_index"))
                self.insert_table(sheet_id, page_block_id, table)

            for item in extraction.get("keynotes_or_legends", []):
                page_block_id = page_block_ids.get(item.get("page_block_index"))
                self.insert_keynote(sheet_id, page_block_id, item)

            for fact in iter_acoustic_fact_records(extraction):
                page_block_id = page_block_ids.get(fact.get("page_block_index"))
                self.insert_acoustic_fact(header, sheet_id, page_block_id, fact)

    def mark_cross_references_resolved(self, revision_id: str):
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE cross_references cr
                SET resolved = TRUE
                WHERE cr.revision_id = %s
                  AND cr.target_sheet_number IN (
                      SELECT sheet_number
                      FROM sheets
                      WHERE revision_id = %s
                  )
                """,
                (revision_id, revision_id),
            )
        self.conn.commit()


class CrossReferenceResolver:
    def __init__(self, resolution_scope: str = "revision_run"):
        self.sheet_numbers_by_revision: Dict[str, set] = defaultdict(set)
        self.revision_coverage_by_id: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"processed_pages": 0, "expected_pages": 0}
        )
        self.resolution_scope = normalize_space(resolution_scope).lower() or "revision_run"

    def add_sheet(self, revision_id: str, sheet_number: str):
        normalized_sheet = self.normalize_sheet_ref(sheet_number)
        if normalized_sheet:
            self.sheet_numbers_by_revision[revision_id].add(normalized_sheet)

    def set_revision_coverage(self, revision_id: str, processed_pages: int, expected_pages: int):
        self.revision_coverage_by_id[revision_id] = {
            "processed_pages": max(0, int(processed_pages)),
            "expected_pages": max(0, int(expected_pages)),
        }

    def resolve_xref(self, revision_id: str, target: str) -> bool:
        normalized = self.normalize_sheet_ref(target)
        return bool(normalized) and normalized in self.sheet_numbers_by_revision.get(revision_id, set())

    def normalize_sheet_ref(self, ref: str) -> str:
        ref = normalize_space(ref)
        ref = re.sub(r"^See\s+(Sheet\s+)?", "", ref, flags=re.IGNORECASE)
        ref = re.sub(r"^Detail\s+", "", ref, flags=re.IGNORECASE)
        ref = re.sub(r"^Section\s+", "", ref, flags=re.IGNORECASE)
        ref = re.sub(r"^\d+\s*/\s*", "", ref)
        trailing_detail_match = re.match(r"^([A-Za-z]{1,3}\d+(?:\.\d+)?[A-Za-z0-9\-]*)\s*/\s*\d+[A-Za-z]?$", ref)
        if trailing_detail_match:
            ref = trailing_detail_match.group(1)
        return ref

    def build_resolution_metadata(self, revision_id: str, normalized_target: str) -> dict:
        known_sheets = self.sheet_numbers_by_revision.get(revision_id, set())
        coverage = self.revision_coverage_by_id.get(revision_id, {"processed_pages": 0, "expected_pages": 0})
        processed_pages = int(coverage.get("processed_pages", 0))
        expected_pages = int(coverage.get("expected_pages", 0))
        revision_complete = bool(expected_pages) and processed_pages >= expected_pages
        resolved = bool(normalized_target) and normalized_target in known_sheets

        unresolved_reason = ""
        if not normalized_target:
            unresolved_reason = "empty_or_unreadable_target"
        elif not resolved:
            unresolved_reason = "target_not_in_scope"
            if expected_pages > 0 and not revision_complete:
                unresolved_reason = "target_not_in_partial_scope"

        return {
            "resolved": resolved,
            "resolution_scope": self.resolution_scope,
            "known_sheet_count": len(known_sheets),
            "resolution_pages_processed": processed_pages,
            "resolution_pages_expected": expected_pages,
            "revision_complete": revision_complete,
            "unresolved_reason": unresolved_reason,
        }

    def annotate_reference(self, revision_id: str, reference_payload: dict):
        target = reference_payload.get("target_sheet_number") or reference_payload.get("reference_text", "")
        normalized_target = self.normalize_sheet_ref(target)
        if normalized_target:
            reference_payload["target_sheet_number"] = normalized_target
        reference_payload.update(self.build_resolution_metadata(revision_id, normalized_target))

    def resolve_all(self, page_records: List[dict]) -> List[dict]:
        for record in page_records:
            revision_id = record.get("header", {}).get("revision_id", "")
            for xref in record.get("extraction", {}).get("cross_references", []):
                if not isinstance(xref, dict):
                    continue
                self.annotate_reference(revision_id, xref)
        return page_records


def discover_documents(root_path: Path, project_id: str) -> dict:
    manifests = []
    revision_counter: Counter = Counter()

    for pdf_path in sorted(root_path.rglob("*.pdf")):
        relative_path = pdf_path.relative_to(root_path)
        revision_label, revision_sequence, revision_source = infer_revision_metadata(relative_path)
        revision_date, revision_confidence, revision_date_source = infer_revision_date_metadata(relative_path)
        revision_key = f"{project_id}::{revision_source}"
        revision_id = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_URL, revision_key))
        document_family_id = infer_document_family_id(pdf_path)
        document_id = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_URL, f"{project_id}::{revision_id}::{relative_path.as_posix()}"))
        document_class = infer_document_class(relative_path)
        manifest = {
            "project_id": project_id,
            "revision_id": revision_id,
            "revision_label": revision_label,
            "revision_sequence": revision_sequence,
            "revision_source": revision_source,
            "revision_date": revision_date,
            "revision_confidence": revision_confidence,
            "revision_date_source": revision_date_source,
            "document_id": document_id,
            "document_family_id": document_family_id,
            "source_relative_path": relative_path.as_posix(),
            "source_file_name": pdf_path.name,
            "file_sha256": sha256_file(pdf_path),
            "issue_date": revision_date,
            "document_class": document_class,
            "retrieval_scope": infer_retrieval_scope(document_class),
        }
        manifests.append(manifest)
        revision_counter[revision_id] += 1

    revisions = {}
    for manifest in manifests:
        revisions[manifest["revision_id"]] = {
            "revision_id": manifest["revision_id"],
            "revision_label": manifest["revision_label"],
            "revision_sequence": manifest["revision_sequence"],
            "revision_source": manifest["revision_source"],
            "revision_date": manifest.get("revision_date", ""),
            "revision_confidence": manifest.get("revision_confidence", 0.0),
            "revision_date_source": manifest.get("revision_date_source", ""),
            "document_count": revision_counter[manifest["revision_id"]],
        }

    latest_revision_sort_key = max(
        (
            (
                item["revision_sequence"],
                parse_iso_date(item.get("revision_date", "")),
                revision_label_sort_value(item.get("revision_label", "")),
            )
            for item in revisions.values()
        ),
        default=(0, date.min, ""),
    )
    current_revision_ids = {
        item["revision_id"]
        for item in revisions.values()
        if (
            item["revision_sequence"],
            parse_iso_date(item.get("revision_date", "")),
            revision_label_sort_value(item.get("revision_label", "")),
        )
        == latest_revision_sort_key
    }
    for revision in revisions.values():
        revision["is_current_revision"] = revision["revision_id"] in current_revision_ids
    for manifest in manifests:
        manifest["is_current_revision"] = manifest["revision_id"] in current_revision_ids

    return {
        "project_id": project_id,
        "root_path": str(root_path),
        "document_count": len(manifests),
        "documents": manifests,
        "revisions": sorted(
            revisions.values(),
            key=lambda item: (
                item["revision_sequence"],
                parse_iso_date(item.get("revision_date", "")),
                revision_label_sort_value(item["revision_label"]),
                item["revision_id"],
            ),
        ),
    }


def process_document(
    root_path: Path,
    manifest: dict,
    genai_client: Any,
    output_path: Path,
    dpi: int,
    verify_mode: str,
    save_crops: bool,
    native_text_enabled: bool,
    project_context: str,
    llm_model: str,
    extractor_route_overrides: Dict[str, str],
    crop_padding_ratio: float,
    native_vision_fallback: bool,
    native_vision_fallback_max_lines: int,
    native_vision_fallback_max_chars: int,
    log_page_phases: bool,
    checkpoint_root: Optional[Path] = None,
    checkpoint_every: int = 0,
) -> Tuple[List[dict], List[dict], List[dict], Dict[str, Any]]:
    document_path = root_path / manifest["source_relative_path"]

    if not HAS_PDF2IMAGE or not HAS_PIL or pdfinfo_from_path is None:
        return [], [], [], {
            "error": "missing_render_dependencies",
            "pages_processed": 0,
            "page_classes": {},
            "block_types": {},
        }

    try:
        pdf_info = pdfinfo_from_path(str(document_path))
        page_count = int(pdf_info["Pages"])
    except Exception as exc:
        return [], [], [], {"error": str(exc), "pages_processed": 0, "page_classes": {}, "block_types": {}}

    manifest["page_count"] = page_count

    pdf_reader = None
    if native_text_enabled and HAS_PYPDF:
        try:
            pdf_reader = PdfReader(str(document_path))
        except Exception as exc:
            print(f"WARNING: could not open {document_path} for native text extraction: {exc}")

    chunks: List[dict] = []
    page_records: List[dict] = []
    block_records: List[dict] = []
    total_usage = {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "phase_usage": empty_phase_usage(),
    }
    total_image_bytes = 0
    page_class_counter: Counter = Counter()
    block_type_counter: Counter = Counter()

    for page_num in tqdm(range(1, page_count + 1), desc=f"Processing {manifest['source_file_name']}"):
        page_image = None
        try:
            log_page_phase(log_page_phases, manifest["source_file_name"], page_num, "phase:render:start")
            page_image = convert_from_path(
                str(document_path),
                dpi=dpi,
                first_page=page_num,
                last_page=page_num,
            )[0]
            log_page_phase(log_page_phases, manifest["source_file_name"], page_num, "phase:render:done")

            log_page_phase(log_page_phases, manifest["source_file_name"], page_num, "phase:native_context:start")
            page_native_context = build_native_page_context(
                pdf_reader.pages[page_num - 1] if pdf_reader and len(pdf_reader.pages) >= page_num else None,
                native_text_enabled,
            )
            log_page_phase(
                log_page_phases,
                manifest["source_file_name"],
                page_num,
                "phase:native_context:done",
                (
                    f"lines={page_native_context.get('line_count', 0)} "
                    f"candidate_tables={len(page_native_context.get('candidate_tables', []))}"
                ),
            )

            header, extraction, page_blocks, image_bytes, usage = process_drawing_page(
                manifest,
                page_num,
                page_image,
                page_native_context,
                genai_client,
                output_path,
                save_crops,
                verify_mode,
                project_context,
                llm_model,
                extractor_route_overrides,
                crop_padding_ratio,
                native_vision_fallback,
                native_vision_fallback_max_lines,
                native_vision_fallback_max_chars,
                log_page_phases,
            )
            page_result = finalize_processed_page(
                manifest,
                page_num,
                page_native_context,
                header,
                extraction,
                page_blocks,
                log_page_phases,
            )
            page_records.append(page_result["record"])
            block_records.extend(page_result["block_records"])
            chunks.extend(page_result["chunks"])

            page_class_counter.update(page_result["page_classes"])
            block_type_counter.update(page_result["block_types"])

            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["output_tokens"] += usage.get("output_tokens", 0)
            total_usage["thinking_tokens"] += usage.get("thinking_tokens", 0)
            accumulate_phase_usage(total_usage["phase_usage"], usage.get("phase_usage", {}))
            total_image_bytes += image_bytes

            if checkpoint_root and checkpoint_every > 0 and len(page_records) % checkpoint_every == 0:
                page_results_map = {
                    int(item["record"]["header"].get("page_number", idx + 1)): item
                    for idx, item in enumerate(
                        [
                            {
                                "record": page_record,
                                "block_records": [block for block in block_records if block.get("page_number") == page_record["header"].get("page_number")],
                                "chunks": [chunk for chunk in chunks if chunk.get("page_number") == page_record["header"].get("page_number")],
                            }
                            for page_record in page_records
                        ]
                    )
                }
                write_document_checkpoint(
                    manifest,
                    page_results_map,
                    page_count,
                    {
                        "page_classes": counter_to_sorted_dict(page_class_counter),
                        "block_types": counter_to_sorted_dict(block_type_counter),
                        "total_prompt_tokens": total_usage["prompt_tokens"],
                        "total_output_tokens": total_usage["output_tokens"],
                        "total_thinking_tokens": total_usage["thinking_tokens"],
                        "total_tokens": total_usage["prompt_tokens"] + total_usage["output_tokens"] + total_usage["thinking_tokens"],
                        "phase_usage": total_usage["phase_usage"],
                    },
                    checkpoint_root,
                )
        except Exception as exc:
            tqdm.write(f"WARNING: failed page {page_num} of {document_path}: {exc}")
        finally:
            close_image(page_image)

    if checkpoint_root and page_records:
        page_results_map = {
            int(item["record"]["header"].get("page_number", idx + 1)): item
            for idx, item in enumerate(
                [
                    {
                        "record": page_record,
                        "block_records": [block for block in block_records if block.get("page_number") == page_record["header"].get("page_number")],
                        "chunks": [chunk for chunk in chunks if chunk.get("page_number") == page_record["header"].get("page_number")],
                    }
                    for page_record in page_records
                ]
            )
        }
        write_document_checkpoint(
            manifest,
            page_results_map,
            page_count,
            {
                "page_classes": counter_to_sorted_dict(page_class_counter),
                "block_types": counter_to_sorted_dict(block_type_counter),
                "total_prompt_tokens": total_usage["prompt_tokens"],
                "total_output_tokens": total_usage["output_tokens"],
                "total_thinking_tokens": total_usage["thinking_tokens"],
                "total_tokens": total_usage["prompt_tokens"] + total_usage["output_tokens"] + total_usage["thinking_tokens"],
                "phase_usage": total_usage["phase_usage"],
            },
            checkpoint_root,
        )

    return (
        chunks,
        page_records,
        block_records,
        {
            "page_count": page_count,
            "pages_processed": len(page_records),
            "total_image_bytes": total_image_bytes,
            "total_prompt_tokens": total_usage["prompt_tokens"],
            "total_output_tokens": total_usage["output_tokens"],
            "total_thinking_tokens": total_usage["thinking_tokens"],
            "total_tokens": total_usage["prompt_tokens"] + total_usage["output_tokens"] + total_usage["thinking_tokens"],
            "phase_usage": total_usage["phase_usage"],
            "page_classes": counter_to_sorted_dict(page_class_counter),
            "block_types": counter_to_sorted_dict(block_type_counter),
        },
    )


def get_document_page_count(document_path: Path) -> int:
    if not HAS_PDF2IMAGE or pdfinfo_from_path is None:
        raise RuntimeError("missing_render_dependencies")
    pdf_info = pdfinfo_from_path(str(document_path))
    return int(pdf_info["Pages"])


def classify_parallel_document_stage(
    document_path: Path,
    manifest: dict,
    page_count: int,
    genai_client: Any,
    extractor_route_overrides: Dict[str, str],
    native_text_enabled: bool,
    native_vision_fallback: bool,
    native_vision_fallback_max_lines: int,
    native_vision_fallback_max_chars: int,
) -> dict:
    base_strategy = resolve_extraction_strategy(manifest.get("document_class", "unknown"), extractor_route_overrides)
    if base_strategy != "native_only":
        return {
            "stage": "vision",
            "base_strategy": base_strategy,
            "fallback_triggered": False,
            "fallback_page": 0,
            "reason": f"base_strategy={base_strategy}",
        }

    if not native_vision_fallback or not genai_client:
        return {
            "stage": "native",
            "base_strategy": base_strategy,
            "fallback_triggered": False,
            "fallback_page": 0,
            "reason": "native_only_no_fallback",
        }

    if not native_text_enabled or not HAS_PYPDF:
        return {
            "stage": "vision",
            "base_strategy": base_strategy,
            "fallback_triggered": True,
            "fallback_page": 1 if page_count > 0 else 0,
            "reason": "native_context_unavailable",
        }

    pdf_reader = None
    try:
        pdf_reader = PdfReader(str(document_path))
    except Exception as exc:
        print(f"WARNING: could not open {document_path} for preflight native-context scan: {exc}")
        return {
            "stage": "vision",
            "base_strategy": base_strategy,
            "fallback_triggered": True,
            "fallback_page": 1 if page_count > 0 else 0,
            "reason": "native_context_scan_failed",
        }

    page_total = len(pdf_reader.pages) if pdf_reader else 0
    for page_num in range(1, max(0, int(page_count)) + 1):
        native_context = build_native_page_context(
            pdf_reader.pages[page_num - 1] if pdf_reader and page_total >= page_num else None,
            native_text_enabled,
        )
        if should_native_page_fallback_to_vision(
            native_context,
            max_lines=native_vision_fallback_max_lines,
            max_chars=native_vision_fallback_max_chars,
        ):
            return {
                "stage": "vision",
                "base_strategy": base_strategy,
                "fallback_triggered": True,
                "fallback_page": page_num,
                "reason": "native_page_low_signal",
            }

    return {
        "stage": "native",
        "base_strategy": base_strategy,
        "fallback_triggered": False,
        "fallback_page": 0,
        "reason": "native_page_signal_ok",
    }


def empty_document_result_usage(page_count: int) -> Dict[str, Any]:
    return {
        "page_count": max(0, int(page_count)),
        "pages_processed": 0,
        "total_image_bytes": 0,
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
        "total_thinking_tokens": 0,
        "total_tokens": 0,
        "phase_usage": empty_phase_usage(),
        "page_classes": {},
        "block_types": {},
        "max_process_rss_bytes": 0,
    }


def build_document_usage_snapshot_from_state(state: dict) -> Dict[str, Any]:
    total_prompt_tokens = int(state.get("total_prompt_tokens", 0) or 0)
    total_output_tokens = int(state.get("total_output_tokens", 0) or 0)
    total_thinking_tokens = int(state.get("total_thinking_tokens", 0) or 0)
    return {
        "page_count": int(state.get("page_count", 0) or 0),
        "pages_processed": len(state.get("page_results", {})),
        "total_image_bytes": int(state.get("total_image_bytes", 0) or 0),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "total_thinking_tokens": total_thinking_tokens,
        "total_tokens": total_prompt_tokens + total_output_tokens + total_thinking_tokens,
        "phase_usage": state.get("phase_usage", empty_phase_usage()),
        "page_classes": counter_to_sorted_dict(state.get("page_class_counter", Counter())),
        "block_types": counter_to_sorted_dict(state.get("block_type_counter", Counter())),
        "max_process_rss_bytes": int(state.get("max_process_rss_bytes", 0) or 0),
    }


def process_cached_vision_page(
    root_path: Path,
    manifest: dict,
    page_num: int,
    page_native_context: dict,
    genai_client: Any,
    output_path: Path,
    cache_manager: PageImageCacheManager,
    inflight_tracker: Optional[InFlightPageTracker],
    save_crops: bool,
    verify_mode: str,
    project_context: str,
    llm_model: str,
    extractor_route_overrides: Dict[str, str],
    crop_padding_ratio: float,
    native_vision_fallback: bool,
    native_vision_fallback_max_lines: int,
    native_vision_fallback_max_chars: int,
    log_page_phases: bool,
    dpi: int,
    model_pool: VisionModelPool,
) -> dict:
    cache_entry = cache_manager.ensure_page_image(
        root_path,
        manifest,
        page_num,
        dpi,
        inflight_tracker=inflight_tracker,
        record_stats=False,
    )
    image_path = Path(cache_entry["page_path"])
    _, page_extraction_strategy = resolve_page_extraction_strategy(
        manifest,
        page_native_context,
        genai_client,
        extractor_route_overrides,
        native_vision_fallback,
        native_vision_fallback_max_lines,
        native_vision_fallback_max_chars,
    )
    image_bytes = None
    llm_token = None
    if page_extraction_strategy != "native_only":
        image_bytes = read_binary_file(image_path)
        llm_token = inflight_tracker.enter("llm", len(image_bytes)) if inflight_tracker else None
    try:
        header, extraction, page_blocks, image_bytes_len, usage = process_drawing_page(
            manifest,
            page_num,
            None,
            page_native_context,
            genai_client,
            output_path,
            save_crops,
            verify_mode,
            project_context,
            llm_model,
            extractor_route_overrides,
            crop_padding_ratio,
            native_vision_fallback,
            native_vision_fallback_max_lines,
            native_vision_fallback_max_chars,
            log_page_phases,
            existing_page_image_path=image_path,
            existing_image_bytes=image_bytes,
            save_page_image_to_disk=False,
            model_pool=model_pool,
            extraction_strategy_override=page_extraction_strategy,
        )
    finally:
        if llm_token:
            inflight_tracker.exit(llm_token)
    page_result = finalize_processed_page(
        manifest,
        page_num,
        page_native_context,
        header,
        extraction,
        page_blocks,
        log_page_phases,
    )
    return {
        "page_result": page_result,
        "usage": usage,
        "image_bytes": image_bytes_len,
        "cache_status": cache_entry.get("status", ""),
        "page_extraction_strategy": page_extraction_strategy,
    }


def process_document_parallel_native(
    root_path: Path,
    manifest: dict,
    genai_client: Any,
    output_path: Path,
    cache_manager: PageImageCacheManager,
    inflight_tracker: Optional[InFlightPageTracker],
    model_pool: VisionModelPool,
    dpi: int,
    verify_mode: str,
    save_crops: bool,
    native_text_enabled: bool,
    project_context: str,
    llm_model: str,
    extractor_route_overrides: Dict[str, str],
    crop_padding_ratio: float,
    native_vision_fallback: bool,
    native_vision_fallback_max_lines: int,
    native_vision_fallback_max_chars: int,
    log_page_phases: bool,
    checkpoint_root: Optional[Path] = None,
    checkpoint_every: int = 0,
) -> Tuple[List[dict], List[dict], List[dict], Dict[str, Any]]:
    document_path = root_path / manifest["source_relative_path"]
    try:
        page_count = get_document_page_count(document_path)
    except Exception as exc:
        return [], [], [], {"error": str(exc), **empty_document_result_usage(0)}

    manifest["page_count"] = page_count
    pdf_reader = None
    if native_text_enabled and HAS_PYPDF:
        try:
            pdf_reader = PdfReader(str(document_path))
        except Exception as exc:
            print(f"WARNING: could not open {document_path} for native text extraction: {exc}")

    page_results_by_num: Dict[int, dict] = {}
    doc_state = {
        "page_count": page_count,
        "page_results": page_results_by_num,
        "total_image_bytes": 0,
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
        "phase_usage": empty_phase_usage(),
        "page_class_counter": Counter(),
        "block_type_counter": Counter(),
        "max_process_rss_bytes": get_process_rss_bytes(),
    }

    for page_num in tqdm(range(1, page_count + 1), desc=f"Processing {manifest['source_file_name']}"):
        existing_page_path = None
        image_bytes = None
        llm_token = None
        try:
            log_page_phase(log_page_phases, manifest["source_file_name"], page_num, "phase:native_context:start")
            page_native_context = build_native_page_context(
                pdf_reader.pages[page_num - 1] if pdf_reader and len(pdf_reader.pages) >= page_num else None,
                native_text_enabled,
            )
            log_page_phase(
                log_page_phases,
                manifest["source_file_name"],
                page_num,
                "phase:native_context:done",
                (
                    f"lines={page_native_context.get('line_count', 0)} "
                    f"candidate_tables={len(page_native_context.get('candidate_tables', []))}"
                ),
            )

            _, page_extraction_strategy = resolve_page_extraction_strategy(
                manifest,
                page_native_context,
                genai_client,
                extractor_route_overrides,
                native_vision_fallback,
                native_vision_fallback_max_lines,
                native_vision_fallback_max_chars,
            )
            if should_parallel_render_native_page(page_extraction_strategy, save_crops):
                cache_entry = cache_manager.ensure_page_image(
                    root_path,
                    manifest,
                    page_num,
                    dpi,
                    inflight_tracker=inflight_tracker,
                )
                existing_page_path = Path(cache_entry["page_path"])
                if page_extraction_strategy != "native_only":
                    image_bytes = read_binary_file(existing_page_path)
                    if inflight_tracker:
                        llm_token = inflight_tracker.enter("llm", len(image_bytes))

            header, extraction, page_blocks, image_bytes_len, usage = process_drawing_page(
                manifest,
                page_num,
                None,
                page_native_context,
                genai_client,
                output_path,
                save_crops,
                verify_mode,
                project_context,
                llm_model,
                extractor_route_overrides,
                crop_padding_ratio,
                native_vision_fallback,
                native_vision_fallback_max_lines,
                native_vision_fallback_max_chars,
                log_page_phases,
                existing_page_image_path=existing_page_path,
                existing_image_bytes=image_bytes,
                save_page_image_to_disk=False,
                model_pool=model_pool,
                extraction_strategy_override=page_extraction_strategy,
            )
            page_result = finalize_processed_page(
                manifest,
                page_num,
                page_native_context,
                header,
                extraction,
                page_blocks,
                log_page_phases,
            )
            page_results_by_num[page_num] = page_result
            doc_state["page_class_counter"].update(page_result["page_classes"])
            doc_state["block_type_counter"].update(page_result["block_types"])
            doc_state["total_image_bytes"] += image_bytes_len
            doc_state["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
            doc_state["total_output_tokens"] += usage.get("output_tokens", 0)
            accumulate_phase_usage(doc_state["phase_usage"], usage.get("phase_usage", {}))
            doc_state["max_process_rss_bytes"] = max(
                int(doc_state.get("max_process_rss_bytes", 0) or 0),
                get_process_rss_bytes(),
            )

            if checkpoint_root and checkpoint_every > 0 and len(page_results_by_num) % checkpoint_every == 0:
                write_document_checkpoint(
                    manifest,
                    page_results_by_num,
                    page_count,
                    build_document_usage_snapshot_from_state(doc_state),
                    checkpoint_root,
                )
        except Exception as exc:
            tqdm.write(f"WARNING: failed page {page_num} of {document_path}: {exc}")
        finally:
            image_bytes = None
            if llm_token:
                inflight_tracker.exit(llm_token)

    if checkpoint_root and page_results_by_num:
        write_document_checkpoint(
            manifest,
            page_results_by_num,
            page_count,
            build_document_usage_snapshot_from_state(doc_state),
            checkpoint_root,
        )

    chunks, page_records, block_records = build_document_outputs_from_page_results(page_results_by_num)
    return chunks, page_records, block_records, build_document_usage_snapshot_from_state(doc_state)


def process_documents_parallel_vision(
    root_path: Path,
    manifests: List[dict],
    genai_client: Any,
    output_path: Path,
    page_image_cache_root: Path,
    dpi: int,
    save_crops: bool,
    native_text_enabled: bool,
    project_context: str,
    llm_model: str,
    extractor_route_overrides: Dict[str, str],
    crop_padding_ratio: float,
    native_vision_fallback: bool,
    native_vision_fallback_max_lines: int,
    native_vision_fallback_max_chars: int,
    log_page_phases: bool,
    checkpoint_root: Optional[Path],
    checkpoint_every: int,
    render_max_workers: int,
    llm_max_workers: int,
    vision_model_pool: List[str],
    vision_model_cooldown_seconds: int,
    rerender_invalid_cache: bool,
) -> Tuple[Dict[str, Tuple[List[dict], List[dict], List[dict], Dict[str, Any]]], dict]:
    if not HAS_PDF2IMAGE or not HAS_PIL or pdfinfo_from_path is None:
        raise RuntimeError("missing_render_dependencies")

    cache_manager = PageImageCacheManager(page_image_cache_root, rerender_invalid_cache=rerender_invalid_cache)
    inflight_tracker = InFlightPageTracker()
    model_pool = VisionModelPool(vision_model_pool, vision_model_cooldown_seconds)

    doc_outputs_by_id: Dict[str, Tuple[List[dict], List[dict], List[dict], Dict[str, Any]]] = {}
    vision_manifests: List[dict] = []
    native_manifests: List[dict] = []
    preflight_stats = {
        "documents": len(manifests),
        "vision_documents": 0,
        "native_documents": 0,
        "native_docs_promoted_to_vision": 0,
        "page_count_errors": 0,
    }
    max_process_rss = get_process_rss_bytes()

    log_parallel_event(f"[PARALLEL] preflight stage start docs={len(manifests)}")
    for manifest in manifests:
        document_path = root_path / manifest["source_relative_path"]
        try:
            page_count = get_document_page_count(document_path)
        except Exception as exc:
            preflight_stats["page_count_errors"] += 1
            log_parallel_event(
                f"[PARALLEL][PREFLIGHT] skip file={manifest['source_file_name']} reason=page_count_error error={exc}"
            )
            doc_outputs_by_id[manifest["document_id"]] = ([], [], [], {"error": str(exc), **empty_document_result_usage(0)})
            continue

        manifest["page_count"] = page_count
        stage_decision = classify_parallel_document_stage(
            document_path,
            manifest,
            page_count,
            genai_client,
            extractor_route_overrides,
            native_text_enabled,
            native_vision_fallback,
            native_vision_fallback_max_lines,
            native_vision_fallback_max_chars,
        )
        if stage_decision["stage"] == "vision":
            vision_manifests.append(manifest)
            preflight_stats["vision_documents"] += 1
            if stage_decision.get("base_strategy") == "native_only" and stage_decision.get("fallback_triggered"):
                preflight_stats["native_docs_promoted_to_vision"] += 1
        else:
            native_manifests.append(manifest)
            preflight_stats["native_documents"] += 1

        log_parallel_event(
            (
                f"[PARALLEL][PREFLIGHT] file={manifest['source_file_name']} stage={stage_decision['stage']} "
                f"base={stage_decision.get('base_strategy', 'unknown')} reason={stage_decision.get('reason', '')} "
                f"fallback_page={int(stage_decision.get('fallback_page', 0) or 0)}"
            )
        )
        max_process_rss = max(max_process_rss, get_process_rss_bytes())
    log_parallel_event(
        (
            f"[PARALLEL] preflight stage complete vision_docs={len(vision_manifests)} "
            f"native_docs={len(native_manifests)} promoted_native_docs={preflight_stats['native_docs_promoted_to_vision']}"
        )
    )

    render_jobs = [
        (manifest, page_num)
        for manifest in vision_manifests
        for page_num in range(1, max(0, int(manifest.get("page_count", 0) or 0)) + 1)
    ]
    render_stage_stats = {
        "documents": len(vision_manifests),
        "target_pages": len(render_jobs),
        "submitted": 0,
        "completed": 0,
        "failed": 0,
        "max_workers": max(1, render_max_workers),
    }
    log_parallel_event(
        (
            f"[PARALLEL] render stage start docs={render_stage_stats['documents']} "
            f"pages={render_stage_stats['target_pages']} workers={render_stage_stats['max_workers']}"
        )
    )
    if render_jobs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, render_max_workers)) as executor:
            future_map = {}
            for manifest, page_num in render_jobs:
                future = executor.submit(
                    cache_manager.ensure_page_image,
                    root_path,
                    manifest,
                    page_num,
                    dpi,
                    inflight_tracker=inflight_tracker,
                )
                future_map[future] = (manifest, page_num)
                render_stage_stats["submitted"] += 1
            for future in concurrent.futures.as_completed(future_map):
                manifest, page_num = future_map[future]
                try:
                    result = future.result()
                    render_stage_stats["completed"] += 1
                    log_parallel_event(
                        (
                            f"[PARALLEL][RENDER] complete file={manifest['source_file_name']} "
                            f"page={page_num:04d} status={result.get('status', 'unknown')}"
                        )
                    )
                except Exception as exc:
                    render_stage_stats["failed"] += 1
                    log_parallel_event(
                        f"[PARALLEL][RENDER] failed file={manifest['source_file_name']} page={page_num:04d} error={exc}"
                    )
                max_process_rss = max(max_process_rss, get_process_rss_bytes())
    render_stage_cache_snapshot = cache_manager.snapshot()
    log_parallel_event("[PARALLEL] render stage complete")

    doc_states: List[dict] = []
    for manifest in vision_manifests:
        document_path = root_path / manifest["source_relative_path"]
        pdf_reader = None
        if native_text_enabled and HAS_PYPDF:
            try:
                pdf_reader = PdfReader(str(document_path))
            except Exception as exc:
                print(f"WARNING: could not open {document_path} for native text extraction: {exc}")
        doc_states.append(
            {
                "manifest": manifest,
                "page_count": int(manifest.get("page_count", 0) or 0),
                "pdf_reader": pdf_reader,
                "next_page": 1,
                "page_results": {},
                "total_image_bytes": 0,
                "total_prompt_tokens": 0,
                "total_output_tokens": 0,
                "phase_usage": empty_phase_usage(),
                "page_class_counter": Counter(),
                "block_type_counter": Counter(),
                "max_process_rss_bytes": get_process_rss_bytes(),
            }
        )

    llm_stage_stats = {
        "documents": len(doc_states),
        "submitted": 0,
        "completed": 0,
        "failed": 0,
        "pages_vision_strategy": 0,
        "pages_native_strategy": 0,
        "max_workers": max(1, llm_max_workers),
    }
    llm_diagnostic_events: List[dict] = []
    llm_page_task_failures: List[dict] = []
    scheduler = RoundRobinPageScheduler(doc_states)
    log_parallel_event(
        (
            f"[PARALLEL] vision llm stage start docs={llm_stage_stats['documents']} "
            f"workers={llm_stage_stats['max_workers']}"
        )
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, llm_max_workers)) as executor:
        future_map: Dict[Any, Tuple[dict, int]] = {}

        def submit_more() -> None:
            while len(future_map) < max(1, llm_max_workers) and scheduler.has_pending():
                next_item = scheduler.next_submission()
                if not next_item:
                    break
                state, page_num = next_item
                manifest = state["manifest"]
                pdf_reader = state.get("pdf_reader")
                page_total = len(pdf_reader.pages) if pdf_reader else 0
                page_native_context = build_native_page_context(
                    pdf_reader.pages[page_num - 1] if pdf_reader and page_total >= page_num else None,
                    native_text_enabled,
                )
                llm_stage_stats["submitted"] += 1
                log_parallel_event(
                    f"[PARALLEL][SCHED] submit file={manifest['source_file_name']} page={page_num:04d}"
                )
                future = executor.submit(
                    process_cached_vision_page,
                    root_path,
                    manifest,
                    page_num,
                    page_native_context,
                    genai_client,
                    output_path,
                    cache_manager,
                    inflight_tracker,
                    save_crops,
                    "off",
                    project_context,
                    llm_model,
                    extractor_route_overrides,
                    crop_padding_ratio,
                    native_vision_fallback,
                    native_vision_fallback_max_lines,
                    native_vision_fallback_max_chars,
                    log_page_phases,
                    dpi,
                    model_pool,
                )
                future_map[future] = (state, page_num)
                if int(state.get("next_page", 0) or 0) <= int(state.get("page_count", 0) or 0):
                    log_parallel_event(
                        (
                            f"[PARALLEL][SCHED] requeue file={manifest['source_file_name']} "
                            f"next_page={int(state.get('next_page', 0) or 0):04d}"
                        )
                    )

        submit_more()
        while future_map:
            done, _ = concurrent.futures.wait(
                list(future_map.keys()),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                state, page_num = future_map.pop(future)
                manifest = state["manifest"]
                try:
                    result = future.result()
                    llm_stage_stats["completed"] += 1
                    strategy = normalize_space(str(result.get("page_extraction_strategy", ""))) or "vision_universal"
                    if strategy == "native_only":
                        llm_stage_stats["pages_native_strategy"] += 1
                    else:
                        llm_stage_stats["pages_vision_strategy"] += 1

                    page_result = result["page_result"]
                    state["page_results"][page_num] = page_result
                    state["page_class_counter"].update(page_result["page_classes"])
                    state["block_type_counter"].update(page_result["block_types"])
                    state["total_image_bytes"] += int(result.get("image_bytes", 0) or 0)
                    state["total_prompt_tokens"] += result["usage"].get("prompt_tokens", 0)
                    state["total_output_tokens"] += result["usage"].get("output_tokens", 0)
                    accumulate_phase_usage(state["phase_usage"], result["usage"].get("phase_usage", {}))
                    state["max_process_rss_bytes"] = max(
                        int(state.get("max_process_rss_bytes", 0) or 0),
                        get_process_rss_bytes(),
                    )

                    usage = result.get("usage", {}) if isinstance(result.get("usage"), dict) else {}
                    output_tokens = int(usage.get("output_tokens", 0) or 0)
                    if int(usage.get("parse_fallback", 0) or 0) > 0:
                        parse_event = {
                            "event": "parse_fallback",
                            "source_file_name": manifest.get("source_file_name", ""),
                            "source_relative_path": manifest.get("source_relative_path", ""),
                            "document_id": manifest.get("document_id", ""),
                            "revision_id": manifest.get("revision_id", ""),
                            "page_number": int(page_num),
                            "model": normalize_space(str(usage.get("model", ""))),
                            "attempts": int(usage.get("attempts", 0) or 0),
                            "failovers": int(usage.get("failovers", 0) or 0),
                            "parse_error": normalize_space(str(usage.get("parse_error", ""))),
                            "output_tokens": output_tokens,
                            "cache_status": result.get("cache_status", ""),
                            "strategy": strategy,
                        }
                        llm_diagnostic_events.append(parse_event)
                        log_parallel_event(
                            (
                                f"[PARALLEL][DEBUG] parse_fallback file={manifest.get('source_file_name', '')} "
                                f"page={page_num:04d} model={parse_event['model'] or 'unknown'} "
                                f"output_tokens={output_tokens}"
                            )
                        )

                    extraction_exception = normalize_space(str(usage.get("exception", "")))
                    if extraction_exception:
                        exception_event = {
                            "event": "extract_exception_fallback",
                            "source_file_name": manifest.get("source_file_name", ""),
                            "source_relative_path": manifest.get("source_relative_path", ""),
                            "document_id": manifest.get("document_id", ""),
                            "revision_id": manifest.get("revision_id", ""),
                            "page_number": int(page_num),
                            "model": normalize_space(str(usage.get("model", ""))),
                            "attempts": int(usage.get("attempts", 0) or 0),
                            "failovers": int(usage.get("failovers", 0) or 0),
                            "exception": extraction_exception,
                            "output_tokens": output_tokens,
                            "cache_status": result.get("cache_status", ""),
                            "strategy": strategy,
                        }
                        llm_diagnostic_events.append(exception_event)
                        log_parallel_event(
                            (
                                f"[PARALLEL][DEBUG] extract_exception_fallback file={manifest.get('source_file_name', '')} "
                                f"page={page_num:04d} model={exception_event['model'] or 'unknown'}"
                            )
                        )

                    if output_tokens >= 20000:
                        high_token_event = {
                            "event": "high_output_tokens",
                            "source_file_name": manifest.get("source_file_name", ""),
                            "source_relative_path": manifest.get("source_relative_path", ""),
                            "document_id": manifest.get("document_id", ""),
                            "revision_id": manifest.get("revision_id", ""),
                            "page_number": int(page_num),
                            "model": normalize_space(str(usage.get("model", ""))),
                            "attempts": int(usage.get("attempts", 0) or 0),
                            "failovers": int(usage.get("failovers", 0) or 0),
                            "output_tokens": output_tokens,
                            "cache_status": result.get("cache_status", ""),
                            "strategy": strategy,
                        }
                        llm_diagnostic_events.append(high_token_event)
                        log_parallel_event(
                            (
                                f"[PARALLEL][DEBUG] high_output_tokens file={manifest.get('source_file_name', '')} "
                                f"page={page_num:04d} output_tokens={output_tokens}"
                            )
                        )

                    log_parallel_event(
                        (
                            f"[PARALLEL][SCHED] complete file={manifest['source_file_name']} page={page_num:04d} "
                            f"strategy={strategy} cache={result.get('cache_status', '')}"
                        )
                    )
                    if checkpoint_root and checkpoint_every > 0 and len(state["page_results"]) % checkpoint_every == 0:
                        write_document_checkpoint(
                            manifest,
                            state["page_results"],
                            state["page_count"],
                            build_document_usage_snapshot_from_state(state),
                            checkpoint_root,
                        )
                except Exception as exc:
                    llm_stage_stats["failed"] += 1
                    failed_entry = {
                        "event": "page_task_failed",
                        "source_file_name": manifest.get("source_file_name", ""),
                        "source_relative_path": manifest.get("source_relative_path", ""),
                        "document_id": manifest.get("document_id", ""),
                        "revision_id": manifest.get("revision_id", ""),
                        "page_number": int(page_num),
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    }
                    llm_page_task_failures.append(failed_entry)
                    log_parallel_event(
                        f"[PARALLEL][SCHED] failed file={manifest['source_file_name']} page={page_num:04d} error={exc}"
                    )
                max_process_rss = max(max_process_rss, get_process_rss_bytes())
            submit_more()
    log_parallel_event("[PARALLEL] vision llm stage complete")

    for state in doc_states:
        manifest = state["manifest"]
        if checkpoint_root and state["page_results"]:
            write_document_checkpoint(
                manifest,
                state["page_results"],
                state["page_count"],
                build_document_usage_snapshot_from_state(state),
                checkpoint_root,
            )
        chunks, page_records, block_records = build_document_outputs_from_page_results(state["page_results"])
        usage = build_document_usage_snapshot_from_state(state)
        doc_outputs_by_id[manifest["document_id"]] = (
            chunks,
            page_records,
            block_records,
            usage,
        )
        max_process_rss = max(max_process_rss, int(usage.get("max_process_rss_bytes", 0) or 0))

    native_stage_stats = {
        "documents": len(native_manifests),
        "pages_processed": 0,
        "errors": 0,
    }
    for manifest in native_manifests:
        log_parallel_event(f"[PARALLEL] native stage start file={manifest['source_file_name']}")
        doc_outputs_by_id[manifest["document_id"]] = process_document_parallel_native(
            root_path,
            manifest,
            genai_client,
            output_path,
            cache_manager,
            inflight_tracker,
            model_pool,
            dpi,
            "off",
            save_crops,
            native_text_enabled,
            project_context,
            llm_model,
            extractor_route_overrides,
            crop_padding_ratio,
            native_vision_fallback,
            native_vision_fallback_max_lines,
            native_vision_fallback_max_chars,
            log_page_phases,
            checkpoint_root,
            checkpoint_every,
        )
        usage = doc_outputs_by_id[manifest["document_id"]][3]
        native_stage_stats["pages_processed"] += usage.get("pages_processed", 0)
        if usage.get("error"):
            native_stage_stats["errors"] += 1
        max_process_rss = max(
            max_process_rss,
            int(usage.get("max_process_rss_bytes", 0) or 0),
            get_process_rss_bytes(),
        )
        log_parallel_event(f"[PARALLEL] native stage done file={manifest['source_file_name']}")

    final_cache_snapshot = cache_manager.snapshot()
    memory_snapshot = inflight_tracker.snapshot()
    memory_snapshot["max_process_rss_bytes"] = int(max_process_rss)

    parallel_stats = {
        "mode": "disk_backed_parallel_vision",
        "effective_verify_mode": "off",
        "preflight": preflight_stats,
        "render_stage": {
            **render_stage_stats,
            **render_stage_cache_snapshot,
        },
        "vision_stage": {
            **llm_stage_stats,
            "pages_processed": sum(
                doc_outputs_by_id[state["manifest"]["document_id"]][3].get("pages_processed", 0)
                for state in doc_states
                if state["manifest"]["document_id"] in doc_outputs_by_id
            ),
        },
        "native_stage": native_stage_stats,
        "cache": final_cache_snapshot,
        "model_pool": model_pool.snapshot(),
        "memory": memory_snapshot,
    }

    rerun_targets: List[dict] = []
    seen_targets = set()
    for event in llm_page_task_failures + llm_diagnostic_events:
        source_relative_path = normalize_space(str(event.get("source_relative_path", "")))
        page_number = int(event.get("page_number", 0) or 0)
        if not source_relative_path or page_number <= 0:
            continue
        key = (source_relative_path, page_number)
        if key in seen_targets:
            continue
        seen_targets.add(key)
        rerun_targets.append(
            {
                "source_relative_path": source_relative_path,
                "source_file_name": event.get("source_file_name", ""),
                "document_id": event.get("document_id", ""),
                "revision_id": event.get("revision_id", ""),
                "page_number": page_number,
            }
        )

    parallel_stats["diagnostics"] = {
        "llm_page_task_failure_count": len(llm_page_task_failures),
        "llm_page_task_failures": llm_page_task_failures,
        "llm_diagnostic_event_count": len(llm_diagnostic_events),
        "llm_diagnostic_events": llm_diagnostic_events,
        "rerun_target_count": len(rerun_targets),
        "rerun_targets": rerun_targets,
    }

    return doc_outputs_by_id, parallel_stats


def run_output_consistency_audit(manifest: dict, chunks: List[dict], page_records: List[dict], block_records: List[dict]) -> dict:
    page_keys = {
        (record.get("header", {}).get("document_id", ""), int(record.get("header", {}).get("page_number", 0) or 0))
        for record in page_records
        if isinstance(record, dict)
    }
    duplicate_chunk_ids = 0
    seen_chunk_ids = set()
    orphan_block_count = 0
    for chunk in chunks:
        deterministic_id = make_deterministic_chunk_id(chunk)
        if deterministic_id in seen_chunk_ids:
            duplicate_chunk_ids += 1
        else:
            seen_chunk_ids.add(deterministic_id)
    for block in block_records:
        block_key = (block.get("document_id", ""), int(block.get("page_number", 0) or 0))
        if block_key not in page_keys:
            orphan_block_count += 1

    manifest_expected_pages = sum(int(item.get("page_count", 0) or 0) for item in manifest.get("documents", []))
    pages_processed = len(page_records)
    errors = []
    if duplicate_chunk_ids:
        errors.append(f"duplicate_chunk_ids={duplicate_chunk_ids}")
    if orphan_block_count:
        errors.append(f"orphan_blocks={orphan_block_count}")
    if manifest_expected_pages and pages_processed != manifest_expected_pages:
        errors.append(f"page_count_mismatch expected={manifest_expected_pages} actual={pages_processed}")

    return {
        "passed": not errors,
        "manifest_expected_pages": manifest_expected_pages,
        "pages_processed": pages_processed,
        "chunk_count": len(chunks),
        "block_count": len(block_records),
        "duplicate_chunk_ids": duplicate_chunk_ids,
        "orphan_blocks": orphan_block_count,
        "errors": errors,
    }


def prepare_qdrant_payload(chunk: dict) -> dict:
    payload = {}
    for key, value in chunk.items():
        if key == "text_to_embed" or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            payload[key] = value
        elif isinstance(value, (list, dict)):
            payload[key] = json.dumps(value, ensure_ascii=False)
    return payload


def build_vector_db(
    chunks: List[dict],
    collection_name: str,
    qdrant_path: str,
    embedding_model: str,
    genai_provider: str,
    vertex_location: str,
    gcp_project_id: str,
) -> Dict[str, Any]:
    if not HAS_QDRANT or not HAS_VECTOR or not HAS_FASTEMBED:
        return {"status": "skipped", "reason": "missing_dependencies"}

    print(f"Building Qdrant collection '{collection_name}'...")
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        qdrant = QdrantClient(url=qdrant_url)
        print(f"  Using Qdrant at {qdrant_url}")
    else:
        qdrant = QdrantClient(path=qdrant_path)
        print(f"  Using local Qdrant at {qdrant_path}")
    existing = [collection.name for collection in qdrant.get_collections().collections]
    if collection_name not in existing:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": qmodels.VectorParams(size=DENSE_VECTOR_DIM, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)},
        )
    else:
        print(f"  Reusing existing collection '{collection_name}' for incremental upserts...")

    for field in KEYWORD_PAYLOAD_FIELDS:
        try:
            qdrant.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass
    for field in INTEGER_PAYLOAD_FIELDS:
        try:
            qdrant.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.INTEGER,
            )
        except Exception:
            pass
    for field in BOOLEAN_PAYLOAD_FIELDS:
        try:
            qdrant.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.BOOL,
            )
        except Exception:
            pass

    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")

    provider = normalize_space(genai_provider).lower() or "vertexai"
    if provider == "vertexai":
        dense_embedder = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            location=vertex_location,
            project=gcp_project_id,
            vertexai=True,
        )
    else:
        embedder_kwargs: Dict[str, Any] = {"model": embedding_model}
        google_api_key = os.getenv("GOOGLE_API_KEY", "")
        if google_api_key:
            embedder_kwargs["google_api_key"] = google_api_key
        try:
            dense_embedder = GoogleGenerativeAIEmbeddings(**embedder_kwargs)
        except TypeError:
            # Fallback for library versions that do not accept explicit API-key kwargs.
            dense_embedder = GoogleGenerativeAIEmbeddings(model=embedding_model)

    texts_to_embed = [chunk.get("text_to_embed") or chunk.get("text") or "" for chunk in chunks]
    payloads = [prepare_qdrant_payload(chunk) for chunk in chunks]
    point_ids = [
        str(uuid_lib.uuid5(uuid_lib.NAMESPACE_URL, make_deterministic_chunk_id(chunk)))
        for chunk in chunks
    ]

    print(f"  Generating sparse vectors for {len(chunks)} chunks...")
    sparse_vectors = list(tqdm(sparse_model.embed(texts_to_embed, batch_size=16), total=len(chunks)))

    dense_batch_size = 5
    for batch_start in tqdm(range(0, len(chunks), dense_batch_size), desc="Dense embed + upsert"):
        batch_end = min(batch_start + dense_batch_size, len(chunks))
        batch_texts = texts_to_embed[batch_start:batch_end]
        batch_payloads = payloads[batch_start:batch_end]
        batch_ids = point_ids[batch_start:batch_end]
        batch_sparse = sparse_vectors[batch_start:batch_end]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                dense_vectors = dense_embedder.embed_documents(batch_texts)
                if len(dense_vectors) != len(batch_texts):
                    raise ValueError("Dense embedding count mismatch")
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    tqdm.write(f"\n  Waiting 60s after embedding failure: {exc}")
                    time.sleep(60)
                else:
                    raise

        points = [
            qmodels.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vector,
                    "sparse": qmodels.SparseVector(
                        indices=sparse_vector.indices.tolist(),
                        values=sparse_vector.values.tolist(),
                    ),
                },
                payload=payload,
            )
            for point_id, dense_vector, sparse_vector, payload in zip(
                batch_ids,
                dense_vectors,
                batch_sparse,
                batch_payloads,
            )
        ]
        qdrant.upsert(collection_name=collection_name, points=points)

    final_count = qdrant.get_collection(collection_name).points_count
    print(f"  Qdrant build complete: {final_count} points")
    return {
        "status": "success",
        "collection_name": collection_name,
        "total_documents": len(chunks),
    }


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_cross_reference_resolution(page_records: List[dict]) -> dict:
    total = 0
    resolved = 0
    unresolved_reasons: Counter = Counter()
    partial_scope_refs = 0

    for record in page_records:
        extraction = record.get("extraction", {}) if isinstance(record, dict) else {}
        xrefs = extraction.get("cross_references", []) if isinstance(extraction, dict) else []
        for xref in xrefs:
            if not isinstance(xref, dict):
                continue
            total += 1
            if bool_payload(xref.get("resolved", False)):
                resolved += 1
            else:
                reason = normalize_space(str(xref.get("unresolved_reason", ""))) or "unknown"
                unresolved_reasons[reason] += 1
            if not bool_payload(xref.get("revision_complete", True)):
                partial_scope_refs += 1

    return {
        "total": total,
        "resolved": resolved,
        "resolution_rate_pct": round((resolved / total) * 100, 2) if total else 0.0,
        "partial_scope_references": partial_scope_refs,
        "unresolved_by_reason": counter_to_sorted_dict(unresolved_reasons),
    }


def summarize_ingestion_quality(page_records: List[dict], chunks: List[dict]) -> dict:
    parse_fallback_pages = 0
    quarantined_pages = 0
    low_confidence_pages = 0
    pages_with_no_chunks = 0
    pages_with_no_blocks = 0
    chunk_pages = {
        (chunk.get("document_id", ""), int(chunk.get("page_number", 0) or 0))
        for chunk in chunks
    }
    for record in page_records:
        header = record.get("header", {}) if isinstance(record, dict) else {}
        extraction = record.get("extraction", {}) if isinstance(record, dict) else {}
        diagnostics = extraction.get("ingestion_diagnostics", {}) if isinstance(extraction, dict) else {}
        if bool_payload(diagnostics.get("parse_fallback", False)):
            parse_fallback_pages += 1
        if bool_payload(header.get("quarantined", False)) or bool_payload(diagnostics.get("quarantined", False)):
            quarantined_pages += 1
        if bool_payload(header.get("low_confidence_extraction", False)) or bool_payload(diagnostics.get("low_confidence_extraction", False)):
            low_confidence_pages += 1
        page_key = (header.get("document_id", ""), int(header.get("page_number", 0) or 0))
        if page_key not in chunk_pages:
            pages_with_no_chunks += 1
        if not extraction.get("content_blocks"):
            pages_with_no_blocks += 1
    return {
        "parse_fallback_pages": parse_fallback_pages,
        "quarantined_pages": quarantined_pages,
        "low_confidence_pages": low_confidence_pages,
        "pages_with_no_chunks": pages_with_no_chunks,
        "pages_with_no_blocks": pages_with_no_blocks,
    }


def main():
    parser = argparse.ArgumentParser(description="Production drawing-package PDF ingestor")
    parser.add_argument("--root", required=True, help="Root directory containing PDF drawing packages")
    parser.add_argument("--project-id", required=True, help="Project identifier")
    parser.add_argument("--project-name", default="", help="Project name")
    parser.add_argument("--project-number", default="", help="Project number")
    parser.add_argument("--output", default="ingestion-output", help="Output directory")
    parser.add_argument("--collection", default="VAVA", help="Qdrant collection name")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for page rendering")
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", ""), help="PostgreSQL connection string")
    parser.add_argument(
        "--genai-provider",
        choices=["vertexai", "api_key"],
        default=DEFAULT_GENAI_PROVIDER if DEFAULT_GENAI_PROVIDER in {"vertexai", "api_key"} else "vertexai",
        help="GenAI provider mode",
    )
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Model for classify/extract/verify calls")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model for vector DB")
    parser.add_argument("--vertex-location", default=DEFAULT_VERTEX_LOCATION, help="Vertex AI region")
    parser.add_argument("--skip-db", action="store_true", help="Skip PostgreSQL storage")
    parser.add_argument("--skip-vectordb", action="store_true", help="Skip Qdrant storage")
    parser.add_argument(
        "--verify-mode",
        choices=["auto", "always", "off"],
        default="auto",
        help="Verification pass policy",
    )
    parser.add_argument(
        "--save-crops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save block crop images",
    )
    parser.add_argument(
        "--native-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use native PDF text/layout support when available",
    )
    parser.add_argument(
        "--native-vision-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If native_only has low native text signal, fall back to vision extraction for that page",
    )
    parser.add_argument(
        "--native-vision-fallback-max-lines",
        type=int,
        default=6,
        help="Fallback threshold: native_only pages with line_count <= this value can use vision",
    )
    parser.add_argument(
        "--native-vision-fallback-max-chars",
        type=int,
        default=220,
        help="Fallback threshold: native_only pages with text chars <= this value can use vision",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Write per-document partial checkpoints every N pages (0 disables)",
    )
    parser.add_argument(
        "--project-context",
        default="",
        help="Optional project-specific extraction context text",
    )
    parser.add_argument(
        "--project-context-file",
        default="",
        help="Optional text file with project extraction context",
    )
    parser.add_argument(
        "--enable-builtin-project-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use built-in project context map fallback",
    )
    parser.add_argument(
        "--extractor-routes-json",
        default="",
        help="JSON mapping of document_class to extraction strategy",
    )
    parser.add_argument(
        "--crop-padding-ratio",
        type=float,
        default=0.02,
        help="Padding ratio around block crop boxes",
    )
    parser.add_argument(
        "--log-page-phases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print per-page pipeline phase progress",
    )
    parser.add_argument(
        "--terminal-log-file",
        default="",
        help="Path to continuous terminal output log file (.txt). Defaults to <output>/terminal_output_<timestamp>.txt",
    )
    parser.add_argument(
        "--parallel-vision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable disk-backed page-cache rendering plus parallel vision extraction",
    )
    parser.add_argument(
        "--page-image-cache-dir",
        default="",
        help="Optional shared cache root for rendered page JPEGs. Defaults to <output>/images in parallel mode.",
    )
    parser.add_argument(
        "--render-max-workers",
        type=int,
        default=2,
        help="Max workers for the page-render cache stage in parallel mode",
    )
    parser.add_argument(
        "--llm-max-workers",
        type=int,
        default=12,
        help="Max workers for cached-image LLM extraction in parallel mode",
    )
    parser.add_argument(
        "--vision-model-pool",
        default=DEFAULT_PARALLEL_VISION_MODEL_POOL,
        help="Comma-separated failover pool for parallel vision extraction",
    )
    parser.add_argument(
        "--vision-model-cooldown-seconds",
        type=int,
        default=120,
        help="Cooldown applied to a model after 429/resource exhaustion in parallel mode",
    )
    parser.add_argument(
        "--rerender-invalid-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, rerender cached page JPEGs whose sidecar metadata is missing or mismatched",
    )
    args = parser.parse_args()

    genai_provider = normalize_space(args.genai_provider).lower() or "vertexai"
    extractor_route_overrides = parse_extractor_route_overrides(args.extractor_routes_json)
    effective_verify_mode = resolve_effective_verify_mode(args.verify_mode, args.parallel_vision)
    vision_model_pool = parse_parallel_model_pool(args.vision_model_pool, fallback_model=args.llm_model)

    root_path = Path(args.root)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    page_image_cache_root = (
        Path(args.page_image_cache_dir)
        if normalize_space(args.page_image_cache_dir)
        else output_path / "images"
    )

    terminal_log_target = (
        Path(args.terminal_log_file)
        if normalize_space(args.terminal_log_file)
        else output_path / f"terminal_output_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    )
    terminal_log_path = ""
    try:
        terminal_log_path = enable_terminal_log_capture(terminal_log_target)
    except Exception as exc:
        print(f"WARNING: failed to enable terminal log capture at {terminal_log_target}: {exc}")

    print("=" * 60)
    print("PRODUCTION DRAWING-PACKAGE INGESTOR")
    print("=" * 60)
    print(f"Root: {root_path}")
    print(f"Output: {output_path}")
    print(f"Project: {args.project_id}")
    print(f"DPI: {args.dpi}")
    print(f"GenAI provider: {genai_provider}")
    print(f"LLM model: {args.llm_model}")
    print(f"LLM 429 retries: {LLM_RATE_LIMIT_RETRY_COUNT} (max attempts={1 + LLM_RATE_LIMIT_RETRY_COUNT})")
    print(f"LLM 429 cooldown seconds: {LLM_RATE_LIMIT_COOLDOWN_SECONDS}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Vertex location: {args.vertex_location}")
    print(f"Verify mode: {args.verify_mode}")
    if effective_verify_mode != args.verify_mode:
        print(f"Effective verify mode: {effective_verify_mode} (parallel vision override)")
    print(f"Save crops: {args.save_crops}")
    print(f"Crop padding ratio: {args.crop_padding_ratio}")
    print(f"Log page phases: {args.log_page_phases}")
    print(f"Native text: {args.native_text}")
    print(f"Native-only to vision fallback: {args.native_vision_fallback}")
    print(f"Native-only fallback max lines: {max(0, args.native_vision_fallback_max_lines)}")
    print(f"Native-only fallback max chars: {max(0, args.native_vision_fallback_max_chars)}")
    print(f"Builtin project context: {args.enable_builtin_project_context}")
    print(f"Extractor route overrides: {json.dumps(extractor_route_overrides, ensure_ascii=False)}")
    print(f"Checkpoint every: {args.checkpoint_every} page(s)")
    print(f"Parallel vision: {args.parallel_vision}")
    print(f"Page image cache root: {page_image_cache_root}")
    print(f"Render max workers: {max(1, args.render_max_workers)}")
    print(f"LLM max workers: {max(1, args.llm_max_workers)}")
    print(f"Vision model pool: {json.dumps(vision_model_pool, ensure_ascii=False)}")
    print(f"Vision model cooldown seconds: {max(0, args.vision_model_cooldown_seconds)}")
    print(f"Rerender invalid cache: {args.rerender_invalid_cache}")
    print(f"Terminal log capture: {'enabled' if terminal_log_path else 'disabled'}")
    print(f"Terminal log file: {terminal_log_path or normalize_storage_path(terminal_log_target)}")
    print("=" * 60)

    manifest = discover_documents(root_path, args.project_id)
    manifest_path = output_path / "bulletproof_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Manifest saved -> {manifest_path}")

    genai_client = None
    if HAS_GOOGLE_GENAI:
        try:
            if genai_provider == "api_key":
                google_api_key = os.getenv("GOOGLE_API_KEY", "")
                if not google_api_key:
                    print("WARNING: GOOGLE_API_KEY not set; GenAI client disabled.")
                else:
                    genai_client = genai.Client(api_key=google_api_key)
                    print("Google GenAI API-key client initialized")
            else:
                genai_client = genai.Client(
                    vertexai=True,
                    location=args.vertex_location,
                    project=os.getenv("GCP_PROJECT_ID"),
                )
                print("Vertex AI client initialized")
        except Exception as exc:
            print(f"WARNING: GenAI init failed: {exc}")

    db = None
    run_id = ""
    if not args.skip_db and args.db_url and HAS_PSYCOPG:
        try:
            db = DatabaseManager(args.db_url)
            db.connect()
            db.init_schema()
            db.insert_project(args.project_id, args.project_name, args.project_number)
            run_id = db.start_ingestion_run(args.project_id)
            print("PostgreSQL initialized")
        except Exception as exc:
            print(f"WARNING: PostgreSQL init failed: {exc}")
            db = None

    project_context = get_project_context(
        args.project_id,
        explicit_context=args.project_context,
        context_file=args.project_context_file,
        enable_builtin_context=args.enable_builtin_project_context,
    )

    all_chunks: List[dict] = []
    all_page_records: List[dict] = []
    all_block_records: List[dict] = []
    total_usage = {
        "pages_processed": 0,
        "total_image_bytes": 0,
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
        "phase_usage": empty_phase_usage(),
    }
    page_class_counter: Counter = Counter()
    block_type_counter: Counter = Counter()
    xref_resolver = CrossReferenceResolver(resolution_scope="revision_run")
    revision_coverage: Dict[str, Dict[str, int]] = defaultdict(lambda: {"processed_pages": 0, "expected_pages": 0})

    chunks_path = output_path / "bulletproof_chunks.json"
    pages_path = output_path / "bulletproof_extractions.json"
    blocks_path = output_path / "bulletproof_blocks.json"
    stats_path = output_path / "bulletproof_stats.json"
    consistency_audit_path = output_path / "bulletproof_consistency_audit.json"
    failure_debug_path = output_path / "bulletproof_failure_debug.json"
    vectordb_stats = {"status": "skipped", "reason": "not_started"}
    parallel_stats: Dict[str, Any] = {"mode": "sequential"}
    stats: Dict[str, Any] = {}

    try:
        if args.parallel_vision:
            doc_outputs_by_id, parallel_stats = process_documents_parallel_vision(
                root_path,
                manifest["documents"],
                genai_client,
                output_path,
                page_image_cache_root,
                args.dpi,
                args.save_crops,
                args.native_text,
                project_context,
                args.llm_model,
                extractor_route_overrides,
                max(0.0, args.crop_padding_ratio),
                args.native_vision_fallback,
                max(0, args.native_vision_fallback_max_lines),
                max(0, args.native_vision_fallback_max_chars),
                args.log_page_phases,
                output_path / "_checkpoints",
                max(0, args.checkpoint_every),
                max(1, args.render_max_workers),
                max(1, args.llm_max_workers),
                vision_model_pool,
                max(0, args.vision_model_cooldown_seconds),
                args.rerender_invalid_cache,
            )
            for doc_manifest in manifest["documents"]:
                if doc_manifest["document_id"] not in doc_outputs_by_id:
                    continue
                chunks, page_records, block_records, usage = doc_outputs_by_id[doc_manifest["document_id"]]
                all_chunks.extend(chunks)
                all_page_records.extend(page_records)
                all_block_records.extend(block_records)
                total_usage["pages_processed"] += usage.get("pages_processed", 0)
                total_usage["total_image_bytes"] += usage.get("total_image_bytes", 0)
                total_usage["total_prompt_tokens"] += usage.get("total_prompt_tokens", 0)
                total_usage["total_output_tokens"] += usage.get("total_output_tokens", 0)
                accumulate_phase_usage(total_usage["phase_usage"], usage.get("phase_usage", {}))
                page_class_counter.update(usage.get("page_classes", {}))
                block_type_counter.update(usage.get("block_types", {}))
                revision_coverage[doc_manifest["revision_id"]]["processed_pages"] += usage.get("pages_processed", 0)
                revision_coverage[doc_manifest["revision_id"]]["expected_pages"] += usage.get("page_count", 0)
                for record in page_records:
                    xref_resolver.add_sheet(record["header"]["revision_id"], record["header"].get("sheet_number", ""))
        else:
            for doc_manifest in manifest["documents"]:
                doc_start = time.time()
                print(
                    f"[DOC] start file={doc_manifest['source_file_name']} "
                    f"revision={doc_manifest['revision_label']} "
                    f"document_id={doc_manifest['document_id']}"
                )
                chunks, page_records, block_records, usage = process_document(
                    root_path,
                    doc_manifest,
                    genai_client,
                    output_path,
                    args.dpi,
                    effective_verify_mode,
                    args.save_crops,
                    args.native_text,
                    project_context,
                    args.llm_model,
                    extractor_route_overrides,
                    max(0.0, args.crop_padding_ratio),
                    args.native_vision_fallback,
                    max(0, args.native_vision_fallback_max_lines),
                    max(0, args.native_vision_fallback_max_chars),
                    args.log_page_phases,
                    output_path / "_checkpoints",
                    max(0, args.checkpoint_every),
                )
                doc_elapsed = round(time.time() - doc_start, 2)
                print(
                    f"[DOC] done file={doc_manifest['source_file_name']} elapsed_s={doc_elapsed} "
                    f"pages={usage.get('pages_processed', 0)}/{usage.get('page_count', 0)} "
                    f"chunks={len(chunks)} blocks={len(block_records)} "
                    f"prompt_tokens={usage.get('total_prompt_tokens', 0)} "
                    f"output_tokens={usage.get('total_output_tokens', 0)}"
                )
                all_chunks.extend(chunks)
                all_page_records.extend(page_records)
                all_block_records.extend(block_records)
                total_usage["pages_processed"] += usage.get("pages_processed", 0)
                total_usage["total_image_bytes"] += usage.get("total_image_bytes", 0)
                total_usage["total_prompt_tokens"] += usage.get("total_prompt_tokens", 0)
                total_usage["total_output_tokens"] += usage.get("total_output_tokens", 0)
                accumulate_phase_usage(total_usage["phase_usage"], usage.get("phase_usage", {}))
                page_class_counter.update(usage.get("page_classes", {}))
                block_type_counter.update(usage.get("block_types", {}))
                revision_coverage[doc_manifest["revision_id"]]["processed_pages"] += usage.get("pages_processed", 0)
                revision_coverage[doc_manifest["revision_id"]]["expected_pages"] += usage.get("page_count", 0)

                for record in page_records:
                    xref_resolver.add_sheet(record["header"]["revision_id"], record["header"].get("sheet_number", ""))

        for revision_id, coverage in revision_coverage.items():
            xref_resolver.set_revision_coverage(
                revision_id,
                coverage.get("processed_pages", 0),
                coverage.get("expected_pages", 0),
            )

        all_page_records = xref_resolver.resolve_all(all_page_records)
        cross_reference_stats = summarize_cross_reference_resolution(all_page_records)

        for chunk in all_chunks:
            if chunk.get("chunk_type") == "cross_reference":
                sync_cross_reference_chunk(chunk, xref_resolver)

        if db:
            for record in all_page_records:
                doc_manifest = next(
                    item for item in manifest["documents"] if item["document_id"] == record["header"]["document_id"]
                )
                db.insert_page_record(doc_manifest, record["header"], record["extraction"])
            for revision in manifest["revisions"]:
                db.mark_cross_references_resolved(revision["revision_id"])

        write_json(chunks_path, all_chunks)
        write_json(pages_path, all_page_records)
        write_json(blocks_path, all_block_records)
        consistency_audit = run_output_consistency_audit(manifest, all_chunks, all_page_records, all_block_records)
        write_json(consistency_audit_path, consistency_audit)

        if args.parallel_vision:
            write_json(failure_debug_path, parallel_stats.get("diagnostics", {"message": "no_parallel_diagnostics"}))

        vectordb_stats = {"status": "skipped", "reason": "user_requested"}
        if not args.skip_vectordb:
            qdrant_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_data")
            vectordb_stats = build_vector_db(
                all_chunks,
                args.collection,
                qdrant_path,
                args.embedding_model,
                genai_provider,
                args.vertex_location,
                os.getenv("GCP_PROJECT_ID", ""),
            )

        stats = {
            "project_id": args.project_id,
            "pdfs_processed": len(manifest["documents"]),
            "pages_processed": total_usage["pages_processed"],
            "total_chunks": len(all_chunks),
            "total_blocks": len(all_block_records),
            "image_storage_mb": total_usage["total_image_bytes"] / (1024 * 1024) if total_usage["total_image_bytes"] else 0,
            "token_usage": {
                "prompt_tokens": total_usage["total_prompt_tokens"],
                "output_tokens": total_usage["total_output_tokens"],
                "thinking_tokens": total_usage.get("total_thinking_tokens", 0),
                "total_tokens": total_usage["total_prompt_tokens"] + total_usage["total_output_tokens"] + total_usage.get("total_thinking_tokens", 0),
                "by_phase": total_usage["phase_usage"],
            },
            "counts_by_revision": {
                revision["revision_label"]: revision["document_count"] for revision in manifest["revisions"]
            },
            "counts_by_page_class": counter_to_sorted_dict(page_class_counter),
            "counts_by_block_type": counter_to_sorted_dict(block_type_counter),
            "chunk_size_distribution": summarize_chunk_sizes(all_chunks),
            "quality": summarize_ingestion_quality(all_page_records, all_chunks),
            "cross_reference_resolution": cross_reference_stats,
            "parallel": parallel_stats,
            "consistency_audit": consistency_audit,
            "vectordb": vectordb_stats,
        }
        write_json(stats_path, stats)
    except Exception as exc:
        cross_reference_stats = summarize_cross_reference_resolution(all_page_records)
        consistency_audit = run_output_consistency_audit(manifest, all_chunks, all_page_records, all_block_records)
        stats = {
            "project_id": args.project_id,
            "pdfs_processed": len(manifest["documents"]),
            "pages_processed": total_usage["pages_processed"],
            "total_chunks": len(all_chunks),
            "total_blocks": len(all_block_records),
            "image_storage_mb": total_usage["total_image_bytes"] / (1024 * 1024) if total_usage["total_image_bytes"] else 0,
            "token_usage": {
                "prompt_tokens": total_usage["total_prompt_tokens"],
                "output_tokens": total_usage["total_output_tokens"],
                "thinking_tokens": total_usage.get("total_thinking_tokens", 0),
                "total_tokens": total_usage["total_prompt_tokens"] + total_usage["total_output_tokens"] + total_usage.get("total_thinking_tokens", 0),
                "by_phase": total_usage["phase_usage"],
            },
            "counts_by_revision": {
                revision["revision_label"]: revision["document_count"] for revision in manifest["revisions"]
            },
            "counts_by_page_class": counter_to_sorted_dict(page_class_counter),
            "counts_by_block_type": counter_to_sorted_dict(block_type_counter),
            "chunk_size_distribution": summarize_chunk_sizes(all_chunks),
            "quality": summarize_ingestion_quality(all_page_records, all_chunks),
            "cross_reference_resolution": cross_reference_stats,
            "parallel": parallel_stats,
            "consistency_audit": consistency_audit,
            "vectordb": vectordb_stats,
            "error": str(exc),
        }
        try:
            write_json(consistency_audit_path, consistency_audit)
            write_json(stats_path, stats)
            if args.parallel_vision:
                write_json(failure_debug_path, parallel_stats.get("diagnostics", {"message": "no_parallel_diagnostics"}))
        except Exception:
            pass

        if db and run_id:
            try:
                db.finish_ingestion_run(run_id, "failed", stats)
            except Exception as finish_exc:
                print(f"WARNING: could not finalize failed ingestion run: {finish_exc}")
            finally:
                db.close()

        raise
    else:
        if db and run_id:
            try:
                db.finish_ingestion_run(run_id, "completed", stats)
            except Exception as finish_exc:
                print(f"WARNING: could not finalize ingestion run: {finish_exc}")
            finally:
                db.close()

        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Pages processed: {stats['pages_processed']}")
        print(f"Chunks generated: {stats['total_chunks']}")
        print(f"Blocks captured: {stats['total_blocks']}")
        print(f"Manifest: {manifest_path}")
        print(f"Chunks:   {chunks_path}")
        print(f"Pages:    {pages_path}")
        print(f"Blocks:   {blocks_path}")
        print(f"Stats:    {stats_path}")
        print(f"Audit:    {consistency_audit_path}")
        if args.parallel_vision:
            print(f"Failure debug: {failure_debug_path}")
        if terminal_log_path:
            print(f"Terminal log: {terminal_log_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
