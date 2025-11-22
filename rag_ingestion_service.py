#!/usr/bin/env python3
"""
It attempts to return "latest popular"
papers by pulling a larger candidate pool (sorted by relevance) and re-ranking the
results by combining relevance (result position) with recency (published date).

Behavioral notes:
- We fetch a larger pool (POOL_MULTIPLIER * max_results, min POOL_MIN) from arXiv
  (sorted by relevance), then compute a combined score:
    score = alpha * relevance_score + beta * recency_score
  where relevance_score = 1/(1+rank_index) and recency_score = 1/(1+days_since_published).
- The top `max_results` from that re-ranked list are then downloaded and ingested.

All original downloader and ingestion logic is preserved; only the arXiv search
and selection logic were enhanced. You can tune POOL_MULTIPLIER, POOL_MIN,
ALPHA_RELEVANCE and BETA_RECENCY to bias toward relevance or recency.
"""

import os
import gc
import hashlib
import uuid
import time
import math
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, unquote
import datetime

# --- Import Dependencies ---
import requests
from tqdm.auto import tqdm
from dotenv import load_dotenv
import tiktoken
from sentence_transformers import SentenceTransformer
import ollama
from litellm import completion

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import ResponseHandlingException
from ollama import Client
import httpx

# --- Core docling tools (ensure these are installed) ---
try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
except ImportError:
    print(
        "[ERROR] 'docling' library not found. Please install it: pip install docling",
        flush=True
    )
    exit(1)

# --- NEW: Optional dependency for searching ---
try:
    import arxiv
except ImportError:
    print(
        "[ERROR] 'arxiv' library not found. Please install it: pip install arxiv",
        flush=True
    )
    exit(1)


# ----------------- CONFIG (tweak these) -----------------
# Load .env file
load_dotenv()

# --- Shared Config ---
# Destination for downloads AND source for ingestion
PAPER_DIR = "./papers"

# --- Download Config ---
# *** FIX: Set a reasonable timeout. 0 means wait forever, which can cause hangs. ***
DOWNLOAD_TIMEOUT = 60  # per-request timeout (seconds)

# --- Qdrant Config (prefer environment overrides) ---
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION =  "documents_chunks"
OPEN_ROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# --- Embedding Model Config ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
EMBEDDING_DEVICE = "cpu"  # "cuda" if available

# --- Chunking / Token Config ---
MAX_TOKENS = 450
OVERLAP_TOKENS = 75
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 600
CHUNK_CHAR_OVERLAP = 100
TOKENIZER_NAME = "cl100k_base"

EMBED_BATCH_SIZE = 256
QDRANT_UPSERT_BATCH = 256

MAX_CHUNKS_PER_PDF = 200_000

# --- arXiv selection tuning ---
POOL_MULTIPLIER = 6        # fetch this many * max_results candidates from arXiv
POOL_MIN = 20              # minimum number of candidates to fetch regardless of multiplier
ALPHA_RELEVANCE = 0.6      # weight for relevance (position in arXiv results)
BETA_RECENCY = 0.4         # weight for recency (published date)

# -----------------------------------------------------------------

# --- Basic Guards ---
if OVERLAP_TOKENS >= MAX_TOKENS:
    raise ValueError(
        f"OVERLAP_TOKENS ({OVERLAP_TOKENS}) must be less than MAX_TOKENS ({MAX_TOKENS})"
    )

# --- Regex for Download Helpers ---
INVALID_FILENAME_CHARS = r'<>:"/\\|?*\0'
_filename_strip_re = re.compile(r'[%s]+' % re.escape(INVALID_FILENAME_CHARS))
_title_parse_re = re.compile(
    r'^(?P<url>\S+)(?:\s+(?:Title\s*:\s*)?(?P<title>.+))?$', re.IGNORECASE
)


# =================================================================
#
# DOWNLOADER FUNCTIONS
#
# =================================================================

def sanitize_filename(name: str, max_len: int = 200) -> str:
    """Cleans a string to be a valid filename."""
    name = name.strip()
    name = _filename_strip_re.sub('_', name)
    # collapse spaces
    name = re.sub(r'\s+', ' ', name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name

def filename_from_url(url: str) -> str:
    """Generates a fallback filename from a URL."""
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    if path:
        base = os.path.basename(path)
        if base:
            return sanitize_filename(base)
    # fallback to host+timestamp
    safe_host = sanitize_filename(parsed.hostname or "file")
    return f"{safe_host}_{int(time.time())}.pdf"

def ensure_unique(dest_folder: str, filename: str) -> str:
    """Ensures the final filename is unique in the destination folder."""
    base, ext = os.path.splitext(filename)
    if not ext:
        ext = ".pdf"
    candidate = f"{base}{ext}"
    i = 1
    while os.path.exists(os.path.join(dest_folder, candidate)):
        candidate = f"{base}({i}){ext}"
        i += 1
    return candidate

def parse_input_line(line: str):
    """Parses a line from the input file into (url, title) or None."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    m = _title_parse_re.match(line)
    if not m:
        return None
    url = m.group('url')
    title = m.group('title')
    if title:
        # strip possible "Title :" prefix inside capture
        title = re.sub(r'^\s*Title\s*:\s*', '', title, flags=re.IGNORECASE).strip()
    return url, title

def download_one(
    session: requests.Session,
    url: str,
    title: Optional[str],
    dest_folder: str,
    timeout: int
):
    """Downloads a single file."""
    try:
        # Use a non-zero timeout if specified
        req_timeout = timeout if timeout > 0 else None
        
        with session.get(
            url, stream=True, timeout=req_timeout, allow_redirects=True
        ) as resp:
            resp.raise_for_status()
            # --- Determine filename ---
            filename = ""
            if title:
                name = title
                if not os.path.splitext(name)[1]:
                    name = name + ".pdf"
                elif os.path.splitext(name)[1].lower() != ".pdf":
                    name = os.path.splitext(name)[0] + ".pdf"
                filename = sanitize_filename(name)
            else:
                cd = resp.headers.get("content-disposition", "")
                fname = None
                if cd:
                    # Check for standardized RFC 5987 filename*
                    m = re.search(r'filename\*=.*\'\'(?P<n>[^;]+)', cd)
                    if m:
                        fname = m.group('n')
                    else:
                        # Check for non-standard filename=
                        m2 = re.search(r'filename="?([^";]+)"?', cd)
                        if m2:
                            fname = m2.group(1)
                if fname:
                    filename = sanitize_filename(fname)
                else:
                    # Fallback to URL
                    filename = filename_from_url(resp.url or url)
                    if not filename.lower().endswith(".pdf"):
                        filename = filename + ".pdf"

            # Ensure unique and write to file
            filename = ensure_unique(dest_folder, filename)
            fullpath = os.path.join(dest_folder, filename)
            total = resp.headers.get("content-length")
            total = int(total) if total and total.isdigit() else None

            chunk_size = 1024 * 32
            with open(fullpath, "wb") as f:
                if total:
                    # Show progress for known file sizes
                    with tqdm(
                        total=total, unit="B", unit_scale=True, desc=filename, leave=False
                    ) as pbar:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # Write without progress for unknown sizes
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)

            return {"url": url, "path": fullpath, "size": os.path.getsize(fullpath), "ok": True}
    except Exception as e:
        return {"url": url, "error": str(e), "ok": False}


def run_downloader(
    input_file_path: str, dest_folder: str, timeout: int
) -> List[str]:
    """
    Reads the input file and downloads all specified URLs sequentially.
    Returns a list of paths to successfully downloaded files.
    """
    if not os.path.exists(input_file_path):
        print(f"[ERROR] Input file not found: {input_file_path}", flush=True)
        return []

    os.makedirs(dest_folder, exist_ok=True)

    tasks = []
    with open(input_file_path, "r", encoding="utf-8") as f:
        for raw in f:
            parsed = parse_input_line(raw)
            if parsed:
                tasks.append(parsed)

    if not tasks:
        print("No valid URLs found in input file. Nothing to download.", flush=True)
        return []

    # Setup requests session
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) KaggleDownloader/1.0",
        "Accept": "application/pdf,application/octet-stream,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    })

    print(
        f"\n--- Starting sequential download of {len(tasks)} files -> '{dest_folder}' ---\n",
        flush=True
    )

    succ_files = []
    fail_count = 0
    failed_logs = []

    # Loop sequentially with a TQDM progress bar
    for url, title in tqdm(tasks, desc="Files", unit="file"):
        res = download_one(session, url, title, dest_folder, timeout)
        if res.get("ok"):
            succ_files.append(res["path"])
        else:
            fail_count += 1
            log_msg = f"  - FAILED: {res.get('url')} -> {res.get('error')}"
            failed_logs.append(log_msg)
            # Print failures immediately
            print(log_msg, flush=True)

    # --- Summary ---
    print("\n--- Download Summary ---", flush=True)
    print(f"  Success: {len(succ_files)}", flush=True)
    if succ_files:
        # show top few files
        for r_path in succ_files[:10]:
             try:
                 size_kb = os.path.getsize(r_path) / 1024
                 print(f"    - {os.path.basename(r_path)} ({size_kb:.1f} KB)", flush=True)
             except OSError:
                 print(f"    - {os.path.basename(r_path)} (size unavailable)", flush=True)
        if len(succ_files) > 10:
             print(f"    - ...and {len(succ_files) - 10} more.", flush=True)

    print(f"  Failed:  {fail_count}", flush=True)
    if failed_logs:
        print("  Failed URLs (first 10):", flush=True)
        for log_line in failed_logs[:10]:
            print(f"    {log_line}", flush=True)
        if len(failed_logs) > 10:
             print(f"    - ...and {len(failed_logs) - 10} more errors.", flush=True)

    print(f"\nFiles saved to: {os.path.abspath(dest_folder)}", flush=True)
    return succ_files


# =================================================================
#
# INGESTION FUNCTIONS
#
# =================================================================

# ---------- TOKENIZER WRAPPER ----------
class TokenizerWrapper:
    """Wraps tiktoken for easy encoding, decoding, and length calculation."""
    def __init__(self, encoding_name: str = TOKENIZER_NAME):
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.enc.decode(token_ids)

    def token_len(self, text: str) -> int:
        return len(self.encode(text))


def token_split_text(
    text: str, max_tokens: int, overlap_tokens: int, tokenizer: TokenizerWrapper
) -> List[str]:
    """Splits text into chunks based on token count."""
    if not text or not text.strip():
        return []

    ids = tokenizer.encode(text)
    total = len(ids)
    if total <= max_tokens:
        return [text]

    chunks = []
    start = 0
    step = max(1, max_tokens - overlap_tokens) # Ensure step is at least 1
    while start < total:
        end = min(start + max_tokens, total)
        token_slice = ids[start:end]
        chunk_text = tokenizer.decode(token_slice).strip()
        if not chunk_text:
            # defensive: advance by step if decode returned empty (rare)
            start += step
            continue
        chunks.append(chunk_text)
        if end == total:
            break
        start += step

    return chunks


# ---------- STABLE ID + METADATA HELPERS ----------
def stable_id(filename: str, index: Any, text: str) -> str:
    """Generates a stable, content-addressable ID for a chunk."""
    # Use SHA1 for speed, truncated to 12 chars.
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    base = os.path.splitext(os.path.basename(filename))[0]
    return f"{base}::chunk::{index}::{h}"


def extract_pages_from_chunk_meta(chunk_meta) -> str:
    """Extracts page numbers from docling metadata."""
    page_nos = set()
    try:
        # Access docling's metadata structure safely
        for doc_item in getattr(chunk_meta, "doc_items", []) or []:
            for prov in getattr(doc_item, "prov", []) or []:
                page_nos.add(int(getattr(prov, "page_no", -1)))
    except Exception:
        # Ignore errors during metadata extraction
        pass
    page_nos = sorted([p for p in page_nos if p >= 0])
    return ",".join(map(str, page_nos))


def chunk_metadata(chunk, original_index, sub_index, filename) -> Dict[str, Any]:
    """Creates the payload dictionary for a sub-chunk."""
    pages = extract_pages_from_chunk_meta(getattr(chunk, "meta", {}))

    # Safely extract headings
    headings_val = getattr(getattr(chunk, "meta", None), "headings", None)
    if headings_val is None:
        headings = ""
    elif isinstance(headings_val, (list, tuple, set)):
        headings = ",".join(str(h) for h in headings_val)
    else:
        headings = str(headings_val)

    # Create a short preview
    chunk_text = getattr(chunk, "text", "")
    preview = (chunk_text[:300] + "...") if chunk_text and len(chunk_text) > 300 else chunk_text
    
    return {
        "source": os.path.basename(filename),
        "pages": pages,
        "headings": headings,
        "parent_chunk_index": original_index,
        "subchunk_index": sub_index,
        "preview": preview
    }


# ---------- Embedding generator ----------
class EmbeddingGenerator:
    """Wraps SentenceTransformer for embedding generation."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = EMBEDDING_DEVICE):
        print(f"[INFO] Loading SentenceTransformer '{model_name}' on device='{device}'", flush=True)
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> List[List[float]]:
        if not texts:
            return []
        # show_progress_bar=False to avoid nested progress bars
        embs = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=batch_size
        )
        return embs.tolist()


# ---------- Qdrant helpers ----------
def ensure_qdrant_collection(
    client: QdrantClient, collection_name: str, vector_size: int, distance: str = "Cosine"
):
    """Checks if a collection exists and creates it if not."""
    try:
        existing = client.get_collections().collections
        if any(c.name == collection_name for c in existing):
            # Collection already exists, do nothing
            # print(f"[INFO] Collection '{collection_name}' already exists.") # Optional: for verbose logging
            return
    except Exception:
        # Fallback: try to create anyway
        print("[WARN] Could not list collections. Will attempt to create.", flush=True)
        pass

    # Map distance string to Qdrant's rest.Distance enum
    dist_map = {
        "cosine": rest.Distance.COSINE,
        "dot": rest.Distance.DOT,
        "euclid": rest.Distance.EUCLID
    }
    dist = dist_map.get(distance.lower(), rest.Distance.COSINE)

    print(
        f"[INFO] Creating collection '{collection_name}' vector_size={vector_size} distance={distance}",
        flush=True
    )
    try:
        # *** FIX: Use create_collection, NOT recreate_collection. ***
        # `recreate_collection` DELETES existing data, which is dangerous.
        # `create_collection` will safely fail if it already exists.
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=dist),
        )
        print(f"[INFO] Collection '{collection_name}' created.", flush=True)
    except ResponseHandlingException as e:
        # Handle race condition or if the initial check failed
        e_str = str(e).lower()
        if "already exists" in e_str or "status_code=409" in e_str:
            print(f"[INFO] Collection '{collection_name}' already exists.", flush=True)
        else:
            print(f"[ERROR] Failed to create collection '{collection_name}': {e}", flush=True)
            raise # Re-raise other creation errors
    except Exception as e:
        print(f"[ERROR] Failed to create collection '{collection_name}': {e}", flush=True)
        raise


def qdrant_uuid_from_stable(stable_id_str: str) -> str:
    """
    Deterministically convert stable_id string into a UUID string (UUIDv5).
    This produces a valid Qdrant point id and is repeatable across runs.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_id_str))


def upsert_points(client: QdrantClient, collection_name: str, points: List[rest.PointStruct]):
    """Wraps the Qdrant upsert call with error handling."""
    try:
        client.upsert(collection_name=collection_name, points=points, wait=True)
    except (httpx.ReadTimeout, httpx.ConnectError, ResponseHandlingException, httpx.TransportError) as e:
        print(f"[ERROR] qdrant upsert transport error: {repr(e)}", flush=True)
        raise
    except Exception as e:
        print(f"[ERROR] unexpected exception during qdrant upsert: {repr(e)}", flush=True)
        raise


def get_existing_ids_in_collection(client: QdrantClient, collection_name: str, limit: int = 10000) -> set:
    """
    Best-effort attempt to gather existing ids via scroll. If scroll is not available or fails, returns empty set.
    """
    ids = set()
    try:
        offset = None
        while True:
            resp, next_offset = client.scroll(
                collection_name=collection_name,
                limit=limit,
                offset=offset,
                with_payload=False,
                with_vectors=False
            )
            if not resp:
                break
            
            for record in resp:
                ids.add(str(record.id))
            
            if next_offset is None:
                break
            offset = next_offset
            
    except Exception:
        # unable to fetch ids - return empty set
        print("[WARN] Could not fetch existing IDs from collection. Will proceed with upsert.", flush=True)
        pass
    return ids



def is_file_ingested(client: QdrantClient, collection_name: str, filename: str) -> bool:
    """
    Checks if a file has already been ingested by looking for any point
    with 'source' matching the filename.
    """
    try:
        # Filter for points where 'source' == filename
        # We only need to find 1 to know it's there.
        scroll_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="source",
                    match=rest.MatchValue(value=filename)
                )
            ]
        )
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=1,
            with_payload=False,
            with_vectors=False
        )
        return bool(points)
    except Exception as e:
        print(f"[WARN] Failed to check if file is ingested: {e}", flush=True)
        return False


# ---------- Main ingestion logic ----------
def run_ingestion(pdf_files: List[str]):
    """
    Runs the ingestion pipeline for a specific list of PDF files.
    """
    if not pdf_files:
        print("[INFO] No PDF files provided for ingestion. Skipping.", flush=True)
        return

    print(f"\n--- Starting Ingestion for {len(pdf_files)} PDFs ---", flush=True)
    t0 = time.time()

    # initialize qdrant client
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=(QDRANT_API_KEY.strip() if QDRANT_API_KEY else None),
            prefer_grpc=False,
            timeout=30.0 # Add a reasonable timeout
        )
        client.get_collections()
        print("[INFO] Qdrant client initialized and connected.", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to initialize Qdrant client at {QDRANT_URL}: {e}", flush=True)
        print("[ERROR] Please check QDRANT_URL and QDRANT_API_KEY environment variables.", flush=True)
        return

    # Prepare local tools
    try:
        converter = DocumentConverter()
        chunker = HybridChunker(
            min_chunk_size=MIN_CHUNK_CHARS,
            max_chunk_size=MAX_CHUNK_CHARS,
            overlap=CHUNK_CHAR_OVERLAP
        )
        tokenizer = TokenizerWrapper(encoding_name=TOKENIZER_NAME)
        emb_gen = EmbeddingGenerator(
            model_name=EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize local tools (Converter, Chunker, Tokenizer, or Embedder): {e}", flush=True)
        return

    total_points_processed = 0
    
    # Ensure collection exists before we try to check against it
    # (Though usually we check inside the loop, checking existence once is fine)
    # We'll rely on ensure_qdrant_collection called later or just assume it might exist.
    # If it doesn't exist, is_file_ingested will likely just return False or error (caught).

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        
        # --- DEDUPLICATION CHECK ---
        if is_file_ingested(client, QDRANT_COLLECTION, filename):
            print(f"[INFO] File '{filename}' already ingested. Skipping.", flush=True)
            continue
        # ---------------------------

        print(f"\n--- Processing: {pdf_path} ---", flush=True)
        file_t0 = time.time()
        try:
            dl_doc = converter.convert(pdf_path).document
        except Exception as e:
            print(f"[ERROR] Failed to convert {pdf_path}: {e}", flush=True)
            continue # Skip this file

        initial_chunks = list(chunker.chunk(dl_doc=dl_doc))
        print(f"[INFO] Initial char-chunks: {len(initial_chunks)}", flush=True)

        sub_texts = []
        sub_metas = []
        sub_ids = []

        total_tokens = 0
        for orig_idx, chunk in enumerate(initial_chunks):
            text = getattr(chunk, "text", "") or ""
            text = text.strip()
            if not text:
                continue
            
            total_tokens += tokenizer.token_len(text)

            token_subs = token_split_text(
                text,
                max_tokens=MAX_TOKENS,
                overlap_tokens=OVERLAP_TOKENS,
                tokenizer=tokenizer
            )
            for sub_idx, stext in enumerate(token_subs):
                sid = stable_id(pdf_path, f"{orig_idx}.{sub_idx}", stext)
                meta = chunk_metadata(chunk, orig_idx, sub_idx, pdf_path)
                sub_texts.append(stext)
                sub_metas.append(meta)
                sub_ids.append(sid)

            if len(sub_texts) >= MAX_CHUNKS_PER_PDF:
                print(
                    f"[WARN] reached MAX_CHUNKS_PER_PDF={MAX_CHUNKS_PER_PDF}; stopping further chunking for this file",
                    flush=True
                )
                break

        est_chunks = len(sub_texts)
        est_batches = math.ceil(est_chunks / QDRANT_UPSERT_BATCH) if QDRANT_UPSERT_BATCH > 0 else 0
        print(
            f"[INFO] file={os.path.basename(pdf_path)} est_tokens={total_tokens} est_subchunks={est_chunks} est_upsert_batches={est_batches}",
            flush=True
        )

        if not sub_texts:
            print("[INFO] no subchunks produced for this file; skipping", flush=True)
            continue

        try:
            first_batch_texts = sub_texts[:EMBED_BATCH_SIZE]
            first_embs = emb_gen.embed(first_batch_texts, batch_size=EMBED_BATCH_SIZE)
        except Exception as e:
            print(f"[ERROR] Embedding failed for first batch of {pdf_path}: {e}", flush=True)
            continue # Skip this file

        if not first_embs:
            print("[ERROR] First embeddings empty; skipping file", flush=True)
            continue

        emb_dim = len(first_embs[0])
        ensure_qdrant_collection(
            client, QDRANT_COLLECTION, vector_size=emb_dim, distance="Cosine"
        )
        
        def index_batches(n, batch_size):
            for i in range(0, n, batch_size):
                yield i, min(n, i + batch_size)

        n_chunks = len(sub_texts)
        for i0, i1 in index_batches(n_chunks, QDRANT_UPSERT_BATCH):
            batch_texts = sub_texts[i0:i1]
            batch_metas = sub_metas[i0:i1]
            batch_ids = sub_ids[i0:i1]

            embeddings = []
            try:
                if i0 == 0:
                    embeddings.extend(first_embs)
                    if len(batch_texts) > len(first_embs):
                        remaining_texts = batch_texts[len(first_embs):]
                        remaining_embs = emb_gen.embed(remaining_texts, batch_size=EMBED_BATCH_SIZE)
                        embeddings.extend(remaining_embs)
                else:
                    embs = emb_gen.embed(batch_texts, batch_size=EMBED_BATCH_SIZE)
                    embeddings.extend(embs)
            
            except Exception as e:
                print(f"[ERROR] Embedding failed for batch {i0}-{i1} of {pdf_path}: {e}", flush=True)
                continue 

            points = []
            for sid, vec, meta, text in zip(batch_ids, embeddings, batch_metas, batch_texts):
                payload = dict(meta)
                payload["text"] = text
                payload["_stable_id"] = sid
                point_id = qdrant_uuid_from_stable(sid)
                point = rest.PointStruct(id=str(point_id), vector=vec, payload=payload)
                points.append(point)

            if not points:
                print(f"[WARN] No points generated for batch {i0}-{i1}. Skipping upsert.", flush=True)
                continue

            try:
                upsert_points(client, QDRANT_COLLECTION, points)
                total_points_processed += len(points)
                print(f"[INFO] Upserted items {i0}..{i1} ({len(points)} points).", flush=True)
            except Exception as e:
                print(f"[ERROR] upsert failed for batch {i0}-{i1} of {pdf_path}: {e}", flush=True)
                break 

            del points, embeddings, batch_texts, batch_metas, batch_ids
            gc.collect()

        file_t1 = time.time()
        print(f"[INFO] Finished {os.path.basename(pdf_path)} in {file_t1 - file_t0:.2f}s", flush=True)

    t1 = time.time()
    print(f"\n--- Ingestion Complete ---", flush=True)
    print(f"Total new/updated points processed: {total_points_processed} in {(t1 - t0):.1f}s", flush=True)


# =================================================================
#
# MAIN PIPELINE FUNCTIONS
#
# =================================================================

def run_pipeline(input_file_path: str):
    """
    Runs the full download and ingest pipeline from a given URL file.

    Args:
        input_file_path: Path to the text file containing URLs.
    """
    print(f"--- Starting Pipeline for: {input_file_path} ---", flush=True)
    pipeline_start_time = time.time()
    
    # 1. Download
    downloaded_files = run_downloader(
        input_file_path=input_file_path,
        dest_folder=PAPER_DIR,
        timeout=DOWNLOAD_TIMEOUT
    )
    
    # 2. Ingest
    if downloaded_files:
        run_ingestion(pdf_files=downloaded_files)
    else:
        print(f"\n--- No files downloaded, ingestion skipped. ---", flush=True)
        
    pipeline_end_time = time.time()
    print(
        f"\n--- Pipeline Finished in {pipeline_end_time - pipeline_start_time:.2f}s ---",
        flush=True
    )

# -----------------------------------------------------------------
# NEW: Function to search arXiv and then run the pipeline
# -----------------------------------------------------------------
def search_and_run_pipeline(query: str, max_results: int = 2):
    """
    Searches arXiv for a query, saves results to a temp file,
    and then runs the full download/ingest pipeline.
    
    This function now attempts to fetch a candidate pool (larger than
    max_results) and re-ranks candidates by a combined relevance+recency score
    so the final selection is biased toward "latest popular" papers.
    """

    # Attempt to convert topics into arXiv categories using the model_query helper.
    categories = None
    try:
        categories = model_query(query)
    except Exception:
        # If model_query fails for any reason, we'll just use the raw query string.
        categories = None

    effective_query = categories if categories else query

    print(f"[INFO] Starting arXiv search for query: '{effective_query}'", flush=True)

    # Determine pool size to fetch from arXiv
    pool_size = max(POOL_MIN, int(max_results * POOL_MULTIPLIER))

    try:
        # Search arXiv for relevant papers, fetch a larger pool sorted by relevance
        search = arxiv.Search(
            query=effective_query,
            max_results=pool_size,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        results = list(search.results())

        # --- Fallback: If no results and query was modified (or contains special chars), try raw/cleaned query ---
        if not results:
            print(f"[INFO] No results found for '{effective_query}'. Trying fallback strategies...", flush=True)
            
            # Strategy 1: If we used categories, try the original query
            if effective_query != query:
                print(f"[INFO] Fallback 1: Trying original query '{query}'", flush=True)
                search = arxiv.Search(query=query, max_results=pool_size, sort_by=arxiv.SortCriterion.Relevance, sort_order=arxiv.SortOrder.Descending)
                results = list(search.results())

            # Strategy 2: If still no results, try cleaning the query (remove special chars)
            if not results:
                clean_query = re.sub(r'[^\w\s]', '', query).strip()
                if clean_query != query and clean_query:
                    print(f"[INFO] Fallback 2: Trying cleaned query '{clean_query}'", flush=True)
                    search = arxiv.Search(query=clean_query, max_results=pool_size, sort_by=arxiv.SortCriterion.Relevance, sort_order=arxiv.SortOrder.Descending)
                    results = list(search.results())

            # Strategy 3: Try searching as a title specifically
            if not results:
                print(f"[INFO] Fallback 3: Trying title search for '{query}'", flush=True)
                # arxiv supports ti:title
                title_query = f'ti:"{query}"'
                search = arxiv.Search(query=title_query, max_results=pool_size, sort_by=arxiv.SortCriterion.Relevance, sort_order=arxiv.SortOrder.Descending)
                results = list(search.results())

            # Strategy 4: Split into keywords and search with OR (broad search)
            if not results:
                keywords = query.split()
                if len(keywords) > 1:
                    or_query = " OR ".join(keywords)
                    print(f"[INFO] Fallback 4: Trying broad keyword search '{or_query}'", flush=True)
                    search = arxiv.Search(query=or_query, max_results=pool_size, sort_by=arxiv.SortCriterion.Relevance, sort_order=arxiv.SortOrder.Descending)
                    results = list(search.results())

        if not results:
            msg = f"No results found on arXiv for query: '{query}' (and fallbacks)"
            print(f"[INFO] {msg}", flush=True)
            return msg

        print(f"[INFO] Retrieved {len(results)} candidate papers from arXiv. Re-ranking by recency+relevance.", flush=True)

        # Compute combined score for each candidate
        now = datetime.datetime.now(datetime.UTC)
        scored = []
        for idx, r in enumerate(results):
            # rank-based relevance score (higher for earlier results)
            relevance_score = 1.0 / (1 + idx)
            # days since published
            try:
                published = r.published if getattr(r, 'published', None) else getattr(r, 'updated', None)
                if not published:
                    days = 3650
                else:
                    if isinstance(published, datetime.datetime):
                        delta = now - published
                    else:
                        # fallback: try parsing
                        delta = now - datetime.datetime.strptime(str(published), "%Y-%m-%dT%H:%M:%SZ")
                    days = max(0, delta.days)
            except Exception:
                days = 3650

            recency_score = 1.0 / (1 + days)
            combined = ALPHA_RELEVANCE * relevance_score + BETA_RECENCY * recency_score
            scored.append((combined, idx, r))

        # Sort by combined score (descending) and pick top max_results
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_results]
        selected = [item[2] for item in top]

        print(f"[INFO] Selected top {len(selected)} papers (by combined score). Preparing download list.", flush=True)

        # Prepare lines in the format our downloader expects
        download_tasks = []
        for res in selected:
            # Clean title: remove newlines and excess whitespace
            clean_title = re.sub(r'\s+', ' ', res.title).strip()
            # Format: URL Title : Title Text
            line = f"{res.pdf_url} Title : {clean_title}"
            download_tasks.append(line)

        # Write the discovered URLs to a temporary file
        temp_file_path = "_discovered_links.txt"
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(download_tasks))
            
        print(f"[INFO] Wrote {len(download_tasks)} URLs to {temp_file_path}", flush=True)
        
        # Now, call the original pipeline using this new file
        run_pipeline(input_file_path=temp_file_path)
        
        return f"Successfully ingested {len(download_tasks)} papers for query: '{query}'"

    except Exception as e:
        err_msg = f"An error occurred during the arXiv search: {e}"
        print(f"[ERROR] {err_msg}", flush=True)
        return err_msg



def process_direct_arxiv_request(query: str) -> Optional[str]:
    """
    Checks if the query is a direct arXiv ID or URL.
    If so, downloads and ingests it directly.
    Returns a success message if handled, otherwise None.
    """
    # Regex for arXiv ID (e.g., 2310.12345 or 2310.12345v1)
    # Also matches full URLs like https://arxiv.org/abs/2310.12345
    arxiv_id_pattern = r'(?:arxiv\.org\/(?:abs|pdf)\/)?(\d{4}\.\d{4,5}(?:v\d+)?)'
    
    match = re.search(arxiv_id_pattern, query.strip())
    if not match:
        return None

    arxiv_id = match.group(1)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"[INFO] Detected direct arXiv ID: {arxiv_id}. Downloading from {pdf_url}", flush=True)

    try:
        # Create a session for download
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) CustomDownloader/1.0",
            "Accept": "application/pdf,application/octet-stream,*/*;q=0.9",
        })

        # Download
        # We don't have a title easily available without querying API, so we'll let it fallback to ID
        res = download_one(session, pdf_url, None, PAPER_DIR, DOWNLOAD_TIMEOUT)
        
        if not res.get("ok"):
            return f"Failed to download arXiv ID {arxiv_id}: {res.get('error')}"

        file_path = res["path"]
        print(f"[INFO] Successfully downloaded to {file_path}", flush=True)

        # Ingest
        run_ingestion([file_path])
        
        return f"Successfully ingested arXiv paper: {arxiv_id} ({os.path.basename(file_path)})"

    except Exception as e:
        print(f"[ERROR] Error processing direct arXiv request {arxiv_id}: {e}", flush=True)
        return f"Error processing arXiv ID {arxiv_id}: {str(e)}"


def model_query(query:str)->str:
    
    prompt = f"""
    You are an assistant that converts topic keywords into arXiv category codes.

    Input: a comma-separated list of topic keywords.
    Output: EXACTLY one line containing 5-10 arXiv category codes separated by commas (example: cs.AI, cs.LG, stat.ML).
    Do NOT output JSON, bullet lists, explanation, or any other text—only the comma-separated categories.
    If unsure, choose general categories like cs.LG, cs.AI, cs.CV, stat.ML.

    Topics: {query}
    """.strip()
    
    resp = completion(
        model="openrouter/minimax/minimax-m2:free",
        messages=[{"role":"system","content":"You are a concise conversion assistant."},
                  {"role":"user","content":prompt}],
        temperature=0
    )
    return resp.choices[0].message.content


# --- Script Entry Point ---
if __name__ == "__main__":
    
    # --- CONFIGURE YOUR SEARCH QUERY HERE ---
    
    # The simple query you want to run
    USER_QUERY = "Machine learning ,Deep Learning, Reinforcement Learning , Meta Learning,LLMs"
    # How many papers to fetch from the query
    MAX_PAPERS_TO_FETCH = 5 

    search_and_run_pipeline(
        query=USER_QUERY,
        max_results=MAX_PAPERS_TO_FETCH
    )
