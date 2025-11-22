# RAG Ingestion Worker

This project implements a robust RAG (Retrieval-Augmented Generation) ingestion pipeline that fetches research papers (from ArXiv or direct URLs), processes them, and indexes them into a Qdrant vector database.

## Architecture Overview

The system consists of two main components:
1.  **Worker (`worker.py`)**: A Redis-backed worker that listens for ingestion tasks.
2.  **Ingestion Service (`rag_ingestion_service.py`)**: The core logic for searching, downloading, processing, and indexing documents.

### Data Flow
1.  **Task Queue**: A task containing a query (e.g., "Machine Learning" or a specific ArXiv ID) is pushed to a Redis queue (`ingest:queue`).
2.  **Worker Pickup**: `worker.py` pops the task.
3.  **Search/Download**: `rag_ingestion_service.py` searches ArXiv or downloads the URL.
4.  **Processing**: The PDF is converted to text, chunked, and embedded.
5.  **Upsertion**: Embeddings and metadata are upserted into Qdrant.

---

## Deep Dive: Ingestion & Upsertion

### 1. The Worker (`worker.py`)
-   **Redis Loop**: The worker runs an infinite loop using `BRPOP` to block and wait for tasks from `ingest:queue`.
-   **Task Processing**:
    -   It parses the JSON payload to get the `query` and `task_id`.
    -   It calls `process_direct_input` to check if the query is a direct URL or ArXiv ID.
    -   If not, it calls `search_and_run_pipeline` to perform a search.
-   **State Management**: It maintains sets in Redis (`ingest:pending`, `ingest:processing`) to track task status and logs results to `ingest:results`.

### 2. Search & Selection (`rag_ingestion_service.py`)
The `search_and_run_pipeline` function determines what to ingest:
-   **Specific Title Search**: First, it checks if the query matches a specific paper title on ArXiv. If a high-confidence match is found, it selects that single paper.
-   **Topic Search**: If no title match is found, it treats the query as a topic list.
    -   It fetches a large pool of candidates from ArXiv.
    -   It re-ranks them using a weighted score of **Relevance** (search rank) and **Recency** (publication date), heavily favoring new papers (`BETA_RECENCY = 0.7`).
-   **Download**: Selected papers are downloaded to the `./papers` directory.

### 3. The Ingestion Pipeline (`run_ingestion`)
Once a PDF is downloaded, the `run_ingestion` function takes over:

#### A. Document Conversion
-   Uses `docling.document_converter.DocumentConverter` to parse the PDF.
-   This extracts structured text, preserving layout information where possible.

#### B. Chunking
The text is split into manageable pieces for embedding:
1.  **Hybrid Chunking**: Uses `docling.chunking.HybridChunker` to create initial semantic chunks based on document structure (paragraphs, headers).
2.  **Token Splitting**: Each semantic chunk is further split if it exceeds `MAX_TOKENS` (450).
    -   Uses `tiktoken` (cl100k_base) to ensure chunks fit within the embedding model's context window.
    -   Applies an overlap (`OVERLAP_TOKENS = 75`) to maintain context between chunks.

#### C. Metadata & Stable IDs
-   **Stable ID**: A deterministic ID is generated for each chunk using `hashlib.sha1(text)`. This ensures that if the same file is processed again, the IDs remain the same, preventing duplicates in Qdrant (idempotency).
-   **Metadata**: Each chunk is enriched with:
    -   `source`: Filename.
    -   `pages`: Page numbers where the text appears.
    -   `headings`: Section headers associated with the text.
    -   `preview`: A snippet of the text.

#### D. Embedding
-   Uses `sentence-transformers/paraphrase-MiniLM-L3-v2` to convert text chunks into vector embeddings.
-   Embeddings are generated in batches (`EMBED_BATCH_SIZE = 256`) for efficiency.

#### E. Upsertion (Qdrant)
The final step is storing the data in Qdrant:
1.  **Collection Check**: Ensures the collection (`documents_chunks`) exists with the correct vector size.
2.  **Deduplication**: Before processing, it checks if the file is already ingested by querying Qdrant for the `source` filename.
3.  **Batch Upsert**:
    -   Chunks are grouped into batches (`QDRANT_UPSERT_BATCH = 256`).
    -   `client.upsert()` is called to push points (Vector + Payload + ID) to Qdrant.
    -   The `id` used is a UUIDv5 derived from the Stable ID, ensuring consistent addressing.

## Configuration
Key parameters in `rag_ingestion_service.py` allow tuning:
-   `POOL_MULTIPLIER` / `BETA_RECENCY`: Controls how "fresh" the topic search results are.
-   `MAX_TOKENS` / `OVERLAP_TOKENS`: Controls chunk granularity.
-   `QDRANT_UPSERT_BATCH`: Controls write performance to the DB.
