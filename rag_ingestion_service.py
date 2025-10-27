#!/usr/bin/env python3
"""
Single-file script to search for PDFs on arXiv, download them, and ingest into Qdrant.

This script:
1.  Accepts a search query (e.g., "Machine learning").
2.  Uses the `arxiv` library to find papers matching the query.
3.  Writes the discovered PDF URLs and titles to a temporary file.
4.  Passes this file to the downloader and ingestion pipeline.

Usage:
1.  Install dependencies: pip install docling sentence-transformers tiktoken qdrant-client requests tqdm python-dotenv arxiv
2.  Set your QDRANT_URL/QDRANT_API_KEY in a .env file.
3.  Set the `USER_QUERY` variable in the `if __name__ == "__main__":` block below.
4.  Run `python3 your_script_name.py`
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

# --- Import Dependencies ---
import requests
from tqdm.auto import tqdm
from dotenv import load_dotenv
import tiktoken
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import ResponseHandlingException
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
DOWNLOAD_TIMEOUT = 0  # per-request timeout (seconds)

# --- Qdrant Config (prefer environment overrides) ---
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION =  "document_chunks"

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
        with session.get(
            url, stream=True, timeout=timeout, allow_redirects=True
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
            return
    except Exception:
        # Fallback: try to create anyway
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
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=dist),
        )
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
    _ = get_existing_ids_in_collection(client, QDRANT_COLLECTION)

    for pdf_path in pdf_files:
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
def search_and_run_pipeline(query: str, max_results: int = 10):
    """
    Searches arXiv for a query, saves results to a temp file,
    and then runs the full download/ingest pipeline.
    
    Args:
        query: The search term (e.g., "Machine learning").
        max_results: The maximum number of papers to fetch.
    """
    print(f"[INFO] Starting arXiv search for query: '{query}'", flush=True)
    
    try:
        # Search arXiv for relevant papers, sorted by relevance
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = list(search.results())
        
        if not results:
            print(f"[INFO] No results found on arXiv for query: '{query}'", flush=True)
            return

        print(f"[INFO] Found {len(results)} papers. Preparing download list.", flush=True)
        
        # Prepare lines in the format our downloader expects
        # (URL <whitespace> Title : <title>)
        download_tasks = []
        for res in results:
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

    except Exception as e:
        print(f"[ERROR] An error occurred during the arXiv search: {e}", flush=True)


# --- Script Entry Point ---
if __name__ == "__main__":
    
    # --- CONFIGURE YOUR SEARCH QUERY HERE ---
    
    # The simple query you want to run
    USER_QUERY = "Machine learning"
    
    # How many papers to fetch from the query
    MAX_PAPERS_TO_FETCH = 5 
    
    # This is the new main function call
    search_and_run_pipeline(
        query=USER_QUERY,
        max_results=MAX_PAPERS_TO_FETCH
    )