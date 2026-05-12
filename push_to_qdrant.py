import json
import time
import os
import sys
import argparse
import uuid as uuid_lib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from fastembed import SparseTextEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(override=True)

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
NUM_WORKERS = 15
DENSE_BATCH_SIZE = 50          # texts per worker per job
SPARSE_BATCH_SIZE = 256       # fastembed is local, can go bigger
MAX_RETRIES = 5
RETRY_WAIT_SECS = 30

DEFAULT_CHUNKS_PATH = os.path.join(
    "tmp_runs", "complete_run", "bulletproof_chunks.json"
)

# ── Payload index definitions ────────────────────────────────────────
KEYWORD_INDEX_FIELDS = [
    "project_id", "revision_id", "revision_label", "document_id",
    "document_family_id", "document_class", "chunk_type",
    "discipline", "primary_discipline", "drawing_type", "sheet_number",
    "entity_type", "entity_id", "entity_name", "page_class",
    "source_file_name", "retrieval_scope", "extraction_strategy",
    "rating_type", "assembly_id", "room_id", "semantic_unit_type",
    "section_name",
]
BOOLEAN_INDEX_FIELDS = ["is_current_revision", "quarantined", "low_confidence_extraction"]
INTEGER_INDEX_FIELDS = ["page_number", "revision_sequence", "block_index"]

# ── Filter registry fields ───────────────────────────────────────────
FILTERABLE_KEYWORD_FIELDS = [
    "project_id", "revision_label", "discipline", "chunk_type",
    "page_class", "document_class", "sheet_number", "entity_type",
    "drawing_type", "retrieval_scope", "rating_type", "assembly_id",
    "source_file_name", "semantic_unit_type", "section_name",
]
FILTERABLE_BOOLEAN_FIELDS = ["is_current_revision", "quarantined", "low_confidence_extraction"]


def build_filter_registry(chunks_list, output_path):
    """Scan all chunks and emit distinct values for each filterable field."""
    registry = {"available_filters": {}, "total_chunks": len(chunks_list)}
    for field in FILTERABLE_KEYWORD_FIELDS:
        values = sorted(set(
            str(c.get(field, "")) for c in chunks_list if c.get(field)
        ))
        if values:
            registry["available_filters"][field] = values
    for field in FILTERABLE_BOOLEAN_FIELDS:
        registry["available_filters"][field] = [True, False]

    registry["generated_at"] = datetime.now().isoformat()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    print(f"📋 Filter registry saved: {output_path} ({len(registry['available_filters'])} fields)")
    return registry


def create_payload_indexes(qdrant_client, collection):
    """Create payload indexes for fast filtered search."""
    print("📇 Creating payload indexes...")
    created = 0
    for field in KEYWORD_INDEX_FIELDS:
        try:
            qdrant_client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            created += 1
        except Exception:
            pass
    for field in BOOLEAN_INDEX_FIELDS:
        try:
            qdrant_client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.BOOL,
            )
            created += 1
        except Exception:
            pass
    for field in INTEGER_INDEX_FIELDS:
        try:
            qdrant_client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.INTEGER,
            )
            created += 1
        except Exception:
            pass
    total_fields = len(KEYWORD_INDEX_FIELDS) + len(BOOLEAN_INDEX_FIELDS) + len(INTEGER_INDEX_FIELDS)
    print(f"   ✅ Indexes created/verified: {created}/{total_fields} fields")


parser = argparse.ArgumentParser(description="Push chunks to Qdrant – parallel edition")
parser.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS_PATH, help="Path to chunks JSON")
parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel embedding workers")
parser.add_argument("--batch-size", type=int, default=DENSE_BATCH_SIZE, help="Texts per dense-embedding batch")
parser.add_argument("--collection", type=str, default="VAVA", help="Qdrant collection name")
parser.add_argument("--project-id", type=str, required=True, help="Project ID to store in chunk payloads")
parser.add_argument("--replace-project", action="store_true", help="Delete existing points for this project_id before upsert")
parser.add_argument(
    "--repair-indexes", action="store_true",
    help="Add payload indexes to existing collection without re-ingesting."
)
args = parser.parse_args()

# ── Handle --repair-indexes mode ─────────────────────────────────────
if args.repair_indexes:
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        print(f"🔧 Repair mode: Adding payload indexes to '{args.collection}' at {qdrant_url}")
        qdrant = QdrantClient(url=qdrant_url)
    else:
        qdrant_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_data")
        print(f"🔧 Repair mode: Adding payload indexes to '{args.collection}' at {qdrant_path}")
        qdrant = QdrantClient(path=qdrant_path)
    create_payload_indexes(qdrant, args.collection)
    info = qdrant.get_collection(args.collection)
    print(f"   Collection has {info.points_count:,} points, {len(info.payload_schema)} indexed fields")
    qdrant.close()
    print("✅ Repair complete.")
    sys.exit(0)

NUM_WORKERS = args.workers
DENSE_BATCH_SIZE = args.batch_size
collection_name = args.collection
project_id = args.project_id

# ──────────────────────────────────────────────────────────────────────
# 1. Load chunks
# ──────────────────────────────────────────────────────────────────────
print(f"📂 Loading chunks from: {args.chunks} ...")
with open(args.chunks, "r", encoding="utf-8") as f:
    chunks = json.load(f)
total = len(chunks)
for chunk in chunks:
    chunk["project_id"] = project_id
print(f"   → {total:,} chunks loaded.")

# ── Generate filter registry ─────────────────────────────────────────
registry_dir = Path(__file__).resolve().parent / "filter_registries" / project_id
registry_dir.mkdir(parents=True, exist_ok=True)
registry_path = registry_dir / f"{project_id}_registry.json"
legacy_registry_path = registry_dir / f"{project_id}.json"

print(f"📊 Generating filter registry for project '{project_id}'...")
build_filter_registry(chunks, str(registry_path))
build_filter_registry(chunks, str(legacy_registry_path))
print(f"   → Registry saved to {registry_path}")

# ──────────────────────────────────────────────────────────────────────
# 2. Prepare texts / payloads / IDs upfront
# ──────────────────────────────────────────────────────────────────────
print("🔧 Preparing texts, payloads, and UUIDs ...")
texts_to_embed = []
payloads = []
point_ids = []

for i, chunk in enumerate(chunks):
    text = (
        chunk.get("text_to_embed", "")
        or chunk.get("original_content", "")
        or chunk.get("text", "")
    )
    texts_to_embed.append(text)

    # Strip massive text block from payload to save memory
    payload = {k: v for k, v in chunk.items() if k != "text_to_embed"}
    payloads.append(payload)

    chunk_id = chunk.get("chunk_id", f"fallback_id_{i}")
    point_ids.append(str(uuid_lib.uuid5(uuid_lib.NAMESPACE_URL, project_id + chunk_id + str(i))))

# Free the raw chunks list — we have what we need
del chunks

# ──────────────────────────────────────────────────────────────────────
# 3. Dense embeddings FIRST (parallel – Google API, the bottleneck)
# ──────────────────────────────────────────────────────────────────────
print(f"\n🚀 Generating dense vectors with {NUM_WORKERS} workers, batch_size={DENSE_BATCH_SIZE} ...")

all_dense = [None] * total  # pre-allocate slots
dense_lock = Lock()

def embed_dense_batch(batch_idx: int, start: int, end: int):
    """Call Google embedding API for a slice of texts."""
    b_texts = texts_to_embed[start:end]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            embedder = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                location="us-central1",
                project=os.getenv("GCP_PROJECT_ID"),
                vertexai=True,
            )
            b_dense = embedder.embed_documents(b_texts)
            if len(b_dense) != len(b_texts):
                raise ValueError(
                    f"Truncated! Expected {len(b_texts)} vectors, got {len(b_dense)}"
                )
            # Write results into the pre-allocated list (no lock needed for non-overlapping slices)
            for j, vec in enumerate(b_dense):
                all_dense[start + j] = vec
            return len(b_dense)
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_WAIT_SECS * attempt
                print(
                    f"\n  ⏳ Batch {batch_idx}: API error (attempt {attempt}/{MAX_RETRIES}). "
                    f"Waiting {wait}s ... Error: {e}"
                )
                time.sleep(wait)
            else:
                print(f"\n  ❌ Batch {batch_idx}: FAILED after {MAX_RETRIES} attempts. Error: {e}")
                raise

batch_ranges = []
for i, start in enumerate(range(0, total, DENSE_BATCH_SIZE)):
    end = min(start + DENSE_BATCH_SIZE, total)
    batch_ranges.append((i, start, end))

num_batches = len(batch_ranges)
dense_failed = []

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {
        executor.submit(embed_dense_batch, idx, s, e): idx
        for idx, s, e in batch_ranges
    }
    with tqdm(total=total, desc="Dense Embed", unit="chunks") as pbar:
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                count = future.result()
                pbar.update(count)
            except Exception:
                dense_failed.append(batch_idx)
                _, s, e = batch_ranges[batch_idx]
                pbar.update(e - s)
                pbar.set_postfix(failed=len(dense_failed))

if dense_failed:
    print(f"\n⚠️  {len(dense_failed)} dense batches failed — those chunks will be skipped during upsert.")
print("   ✅ Dense vectors done.\n")

# ──────────────────────────────────────────────────────────────────────
# 4. Sparse embeddings (local model – fast, no API limits)
# ──────────────────────────────────────────────────────────────────────
print(f"⚡ Generating sparse vectors for {total:,} chunks (batch_size={SPARSE_BATCH_SIZE}) ...")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
all_sparse = list(
    tqdm(
        sparse_model.embed(texts_to_embed, batch_size=SPARSE_BATCH_SIZE),
        total=total,
        desc="Sparse Embed",
    )
)
print("   ✅ Sparse vectors done.\n")

# ──────────────────────────────────────────────────────────────────────
# 5. Qdrant setup – clean slate
# ──────────────────────────────────────────────────────────────────────
qdrant_url = os.getenv("QDRANT_URL")
if qdrant_url:
    print(f"📡 Connecting to Qdrant at {qdrant_url}...")
    qdrant = QdrantClient(url=qdrant_url)
else:
    qdrant_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_data")
    print(f"📂 Using local Qdrant at {qdrant_path}...")
    qdrant = QdrantClient(path=qdrant_path)

existing = [c.name for c in qdrant.get_collections().collections]
if collection_name not in existing:
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(modifier=qmodels.Modifier.IDF)
        },
    )
else:
    print(f"📦 Appending to existing collection '{collection_name}'")

# ── Create payload indexes for fast filtered search ──────────────────
create_payload_indexes(qdrant, collection_name)

if args.replace_project:
    print(f"🧹 Deleting existing points for project_id='{project_id}'...")
    qdrant.delete(
        collection_name=collection_name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="project_id",
                        match=qmodels.MatchValue(value=project_id),
                    )
                ]
            )
        ),
        wait=True,
    )

# ──────────────────────────────────────────────────────────────────────
# 6. Upsert to Qdrant in batches
# ──────────────────────────────────────────────────────────────────────
UPSERT_BATCH = 200  # bigger batches for local upsert, it's fast
print(f"📤 Upserting to Qdrant in batches of {UPSERT_BATCH} ...")

pushed = 0
skipped = 0

for start in tqdm(range(0, total, UPSERT_BATCH), desc="Upsert", unit="batch"):
    end = min(start + UPSERT_BATCH, total)
    points = []
    for j in range(start, end):
        if all_dense[j] is None:
            skipped += 1
            continue
        points.append(
            qmodels.PointStruct(
                id=point_ids[j],
                vector={
                    "dense": all_dense[j],
                    "sparse": qmodels.SparseVector(
                        indices=all_sparse[j].indices.tolist(),
                        values=all_sparse[j].values.tolist(),
                    ),
                },
                payload=payloads[j],
            )
        )
    if points:
        qdrant.upsert(collection_name=collection_name, points=points)
        pushed += len(points)

# ──────────────────────────────────────────────────────────────────────
# 7. Summary
# ──────────────────────────────────────────────────────────────────────
count = qdrant.get_collection(collection_name).points_count
print("\n" + "=" * 50)
print(f"🎉 DONE!  Qdrant points: {count:,}")
print(f"   Project ID          : {project_id}")
print(f"   Pushed successfully : {pushed:,} / {total:,}")
if skipped:
    print(f"   ⚠️  Skipped (no dense vec): {skipped:,}")
if dense_failed:
    print(f"   ⚠️  Failed dense batches  : {len(dense_failed)}")
print("=" * 50)
