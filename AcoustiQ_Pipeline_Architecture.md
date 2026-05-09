# AcoustiQ Pro: Deep-Dive Pipeline Architecture

This document provides a highly detailed technical breakdown of the AcoustiQ project. It covers the logic, data structures, and reasoning protocols that power the acoustic consulting agent.

---

## 1. The Vision-Extraction Engine (`ingest_bulletproof.py`)
AcoustiQ treats drawing packages as visual data, not just text.

### A. Vision LLM Strategy
We utilize **Gemini 3 Flash** with a specialized `UNIVERSAL_EXTRACTION_PROMPT`. Unlike standard OCR, this model understands the **spatial relationship** of AEC documents.
- **Title Block Extraction**: Captures the absolute "Project Truth" (Project Name, Sheet Number, Issue Date, Revision).
- **Entity Discovery**: Recognizes symbols (bubbles, tags) and maps them to text descriptions.
- **Acoustic Facts**: Specifically looks for strings like "STC", "IIC", and "NC" to populate the `acoustic_facts` data structure.

### B. The JSON Schema
The extraction produces a massive JSON object for every page, including:
- `page_summary`: A 3-10 paragraph prose description used for high-level sheet identification.
- `entities`: Structured lists of rooms, wall types, and equipment.
- `cross_references`: Explicit breadcrumbs like "See 3/A7.01" which are used for recursive search.

---

## 2. Anatomy of a Chunk
The "Chunk" is the fundamental unit of intelligence. Each chunk is a JSON object with two main parts: the **Text** and the **Metadata Payload**.

### A. High-Value Metadata Fields
The payload contains over 20 fields used for **Hard Filtering** during retrieval:
- `discipline`: (e.g., `architectural`, `mechanical`) Limits search to specific engineering silos.
- `document_class`: (e.g., `text_native`, `drawing`) Crucial for the agent to distinguish between "Spec requirements" and "Drawing details."
- `chunk_type`: (e.g., `acoustic_rating`, `keynote`, `table_row`) Allows the agent to look specifically for data in tables vs. general notes.
- `assembly_id`: The exact code for a wall or floor (e.g., `JB`, `FA4`).
- `revision_label`: Tracks the design stage (e.g., `50% CD`, `100% DD`).

---

## 3. Storage & Filter Discovery Mechanics
### A. The Filter Registry (`filter_registry.json`)
Every time you run `push_to_qdrant.py`, the system performs **Filter Discovery**:
1. It iterates through every chunk.
2. It extracts every unique value for fields like `discipline`, `sheet_number`, and `revision_label`.
3. It saves this to `filter_registry.json`.

**Why this matters**: The Agent calls `list_available_filters` at the start of a session. It "learns" that the project has "Architectural" and "Mechanical" drawings but NO "Plumbing" drawings. This prevents it from wasting time on searches that will return zero results.

---

## 4. Hybrid Retrieval & Reranking (`retrieval.py`)
Search is a multi-stage tournament.

### A. Hybrid Search (Dense + Sparse)
We use two search heads simultaneously:
1. **Dense (Semantic)**: Uses Vertex AI to find "concepts." If you search for "noise from fans," it finds "Mechanical equipment sound power levels."
2. **Sparse (Keyword)**: Uses BM42 (similar to BM25) to find "exact matches." If you search for `KS/KA`, it finds that exact string even if the semantic model doesn't understand the assembly code.

### B. RRF Fusion
The two results are fused using **Reciprocal Rank Fusion (RRF)**. This ensures that a chunk that appears in both lists gets a massive boost in ranking.

### C. The Cohere Reranker (The Final Judge)
The top 60 fused results are sent to the **Cohere Rerank-3** cross-encoder. 
- **The Problem**: Vector search is "fuzzy." It might think `STC 50` and `STC 55` are the same thing.
- **The Solution**: The Reranker reads the query and the chunk text *together* to detect subtle technical differences. It ensures that if you asked for a "5th floor rating," a chunk mentioning the "5th floor" moves to position #1, even if it had a lower initial vector score.

---

## 5. Agent Intelligence: The Consultant Protocols
The Agent (`agent.py`) isn't just "chatting"; it follows a strict **Technical Advisory Mandate**.

### A. Variant Collapse Prevention
The AI is instructed: *"Architectural projects often have different specifications depending on the floor level. NEVER report only one value when several are present."* 
- If Chunk 1 says STC 50 and Chunk 2 says STC 55, the agent **must** list both and explain the conditions for each.

### B. The "Parallel Detective" Protocol
The agent is hard-coded to cross-reference:
1. It searches **Drawings** to see what the architect *showed*.
2. It searches **Acoustic Reports** to see what the consultant *recommended*.
3. If they disagree, it flags a **"TECHNICAL DISCREPANCY ALERT."**

### C. Citation & Interactive Sources
Every fact is cited (e.g., `[1]`). In the UI, these aren't just numbers; they are links to the `source` payload in the frontend state. This creates a "Closed Loop" where the user can verify every single claim the AI makes.

---

## 6. Execution Flow Summary
1. **User asks**: "What is the wall rating for the gym?"
2. **Agent thinks**: "I need to check the guestroom adjacency and the acoustic report."
3. **Agent acts**: Calls `search_documents` with `discipline: architectural` AND a second search with `document_class: text_native`.
4. **Retrieval**: Qdrant finds the chunks → RRF blends them → Cohere Reranks the technical matches to the top.
5. **Response**: Agent compares the drawing to the report, cites both, and warns if the gym needs a higher rating than shown.
