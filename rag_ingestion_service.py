#!/usr/bin/env python3
"""
Multi-Source Paper Ingestion Pipeline
-------------------------------------
Sources:
1. arXiv (via python library)
2. Semantic Scholar (via Graph API)
3. CORE (via API v3)

Logic:
- Fetches candidates from all available sources.
- Normalizes them into a standard object.
- Re-ranks the combined pool based on Relevance (Search Rank) + Recency (Date).
- Downloads the top N papers.
- Ingests them into Qdrant using Docling for parsing and Hybrid Chunking.
"""

import os
import gc
import hashlib
import uuid
import time
import math
import re
import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, unquote
from dataclasses import dataclass

# --- Dependencies ---
import requests
from tqdm.auto import tqdm
from dotenv import load_dotenv
import tiktoken
from sentence_transformers import SentenceTransformer
from litellm import completion

# --- Qdrant ---
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import ResponseHandlingException

# --- Docling & Arxiv ---
try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
except ImportError:
    print("[ERROR] 'docling' library not found. Please install it: pip install docling")
    exit(1)

try:
    import arxiv
except ImportError:
    print("[ERROR] 'arxiv' library not found. Please install it: pip install arxiv")
    exit(1)


# =================================================================
# CONFIGURATION
# =================================================================
load_dotenv()

# --- API KEYS ---
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
CORE_API_KEY = os.environ.get("CORE_API_KEY")

# --- Qdrant ---
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
QDRANT_COLLECTION = "document_chunks"

# --- Ingestion Settings ---
PAPER_DIR = "./papers"
DOWNLOAD_TIMEOUT = 60
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L3-v2'
EMBEDDING_DEVICE = "cpu"  # Change to "cuda" or "mps" if available

# --- Chunking / Token Config ---
# Reduced to 300 to ensure we don't exceed the 512 limit of the embedding model
# (tiktoken counts are often lower than BERT tokenizer counts)
MAX_TOKENS = 300
OVERLAP_TOKENS = 50
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 600
TOKENIZER_NAME = "cl100k_base"
EMBED_BATCH_SIZE = 256
QDRANT_UPSERT_BATCH = 256

# --- Search & Ranking Tuning ---
# How many papers to fetch per source to build the candidate pool
# e.g. if we want 5 papers, we fetch 5 * 6 = 30 from each source to re-rank
POOL_MULTIPLIER = 6  
POOL_MIN = 20
ALPHA_RELEVANCE = 0.4  # Weight for search engine rank
BETA_RECENCY = 0.6     # Weight for how new the paper is


# =================================================================
# DATA STRUCTURES
# =================================================================

@dataclass
class PaperCandidate:
    """Standardized object for papers from any source."""
    title: str
    pdf_url: str
    published_date: Optional[datetime.datetime]
    source: str
    original_rank: int
    citation_count: int = 0  # Quality signal from Semantic Scholar


# =================================================================
# SEARCH MODULES
# =================================================================

def search_arxiv_candidates(query: str, limit: int) -> List[PaperCandidate]:
    """Fetches candidates from ArXiv."""
    print(f"[SEARCH] Querying ArXiv for '{query}'...", flush=True)
    candidates = []
    try:
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        for i, res in enumerate(search.results()):
            candidates.append(PaperCandidate(
                title=res.title,
                pdf_url=res.pdf_url,
                published_date=res.published,
                source="arxiv",
                original_rank=i
            ))
    except Exception as e:
        print(f"[WARN] ArXiv search error: {e}")
    return candidates


def search_semantic_scholar_candidates(query: str, limit: int) -> List[PaperCandidate]:
    """Fetches candidates from Semantic Scholar Graph API."""
    # API key is optional now
    
    print(f"[SEARCH] Querying Semantic Scholar for '{query}'...", flush=True)
    candidates = []
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    # Fetch 2x limit to account for papers without Open Access PDFs
    params = {
        "query": query,
        "limit": limit * 2,
        "fields": "title,openAccessPdf,publicationDate,url,citationCount"
    }
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            idx = 0
            for item in data.get("data", []):
                # Filter: Must have Open Access PDF
                pdf_info = item.get("openAccessPdf")
                if pdf_info and pdf_info.get("url"):
                    # Parse date
                    pub_date = None
                    if item.get("publicationDate"):
                        try:
                            pub_date = datetime.datetime.strptime(item["publicationDate"], "%Y-%m-%d")
                            pub_date = pub_date.replace(tzinfo=datetime.timezone.utc)
                        except: pass
                    
                    candidates.append(PaperCandidate(
                        title=item.get("title"),
                        pdf_url=pdf_info.get("url"),
                        published_date=pub_date,
                        source="semantic_scholar",
                        original_rank=idx,
                        citation_count=item.get("citationCount") or 0
                    ))
                    idx += 1
        else:
            print(f"[WARN] Semantic Scholar Error: {r.status_code}")
    except Exception as e:
        print(f"[WARN] Semantic Scholar exception: {e}")

    return candidates


def search_core_candidates(query: str, limit: int) -> List[PaperCandidate]:
    """Fetches candidates from CORE API."""
    if not CORE_API_KEY:
        print("[INFO] No CORE API Key found. Skipping CORE search.")
        return []

    print(f"[SEARCH] Querying CORE for '{query}'...", flush=True)
    candidates = []
    
    url = "https://api.core.ac.uk/v3/search/works"
    headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
    
    payload = {
        "q": query,
        "limit": limit * 2
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            idx = 0
            for item in data.get("results", []):
                download_url = item.get("downloadUrl")
                # Fallback: check links list
                if not download_url:
                    for link in item.get("links", []):
                        if link.get("type") == "download":
                            download_url = link.get("url")
                            break
                
                if download_url:
                    pub_date = None
                    d_str = item.get("publishedDate")
                    if d_str:
                        try:
                            # Try parsing ISO
                            pub_date = datetime.datetime.fromisoformat(d_str.replace("Z", "+00:00"))
                        except: pass
                    
                    candidates.append(PaperCandidate(
                        title=item.get("title"),
                        pdf_url=download_url,
                        published_date=pub_date,
                        source="core",
                        original_rank=idx
                    ))
                    idx += 1
        else:
            print(f"[WARN] CORE API Error: {r.status_code}")
    except Exception as e:
        print(f"[WARN] CORE exception: {e}")
        
    return candidates


# =================================================================
# UTILS: DOWNLOADER
# =================================================================

INVALID_FILENAME_CHARS = r'<>:"/\\|?*\0'
_filename_strip_re = re.compile(r'[%s]+' % re.escape(INVALID_FILENAME_CHARS))
_title_parse_re = re.compile(r'^(?P<url>\S+)(?:\s+(?:Title\s*:\s*)?(?P<title>.+))?$', re.IGNORECASE)

def sanitize_filename(name: str, max_len: int = 200) -> str:
    name = name.strip()
    name = _filename_strip_re.sub('_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:max_len].rstrip()

def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    if path:
        base = os.path.basename(path)
        if base: return sanitize_filename(base)
    return f"file_{int(time.time())}.pdf"

def ensure_unique(dest_folder: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    # Fix for ArXiv IDs being parsed as extensions (e.g. .19437)
    if not ext or re.match(r'^\.\d+$', ext): 
        ext = ".pdf"
    
    candidate = f"{base}{ext}"
    i = 1
    while os.path.exists(os.path.join(dest_folder, candidate)):
        candidate = f"{base}({i}){ext}"
        i += 1
    return candidate

def download_one(session, url, title, dest_folder, timeout):
    """Downloads a single file to dest_folder."""
    try:
        # Ensure destination exists
        os.makedirs(dest_folder, exist_ok=True)

        with session.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
            resp.raise_for_status()
            
            # Determine Filename
            if title:
                name = title + ".pdf" if not title.lower().endswith(".pdf") else title
                filename = sanitize_filename(name)
            else:
                filename = filename_from_url(resp.url or url)
            
            filename = ensure_unique(dest_folder, filename)
            fullpath = os.path.join(dest_folder, filename)
            
            # Write
            with open(fullpath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=32768):
                    if chunk: f.write(chunk)
            
            return {"url": url, "path": fullpath, "ok": True}
    except Exception as e:
        return {"url": url, "error": str(e), "ok": False}

def run_downloader(input_file_path, dest_folder, timeout):
    """Reads URLs from a file and downloads them."""
    if not os.path.exists(input_file_path): return []
    os.makedirs(dest_folder, exist_ok=True)
    
    tasks = []
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            m = _title_parse_re.match(line.strip())
            if m: tasks.append((m.group('url'), m.group('title')))
    
    succ_files = []
    session = requests.Session()
    # Spoof generic user agent to avoid blocking
    session.headers.update({"User-Agent": "ResearchBot/1.0 (Academic Use)"})
    
    print(f"\n--- Downloading {len(tasks)} papers ---", flush=True)
    for url, title in tqdm(tasks, desc="Download"):
        res = download_one(session, url, title, dest_folder, timeout)
        if res["ok"]: 
            succ_files.append(res["path"])
        else: 
            print(f"  [FAIL] {url} -> {res.get('error')}")
            
    return succ_files


# =================================================================
# UTILS: INGESTION (Docling -> Embedding -> Qdrant)
# =================================================================

class TokenizerWrapper:
    def __init__(self): self.enc = tiktoken.get_encoding(TOKENIZER_NAME)
    def token_len(self, text): return len(self.enc.encode(text))
    def decode(self, ids): return self.enc.decode(ids)
    def encode(self, text): return self.enc.encode(text)

def token_split_text(text, max_tokens, overlap, tokenizer):
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens: return [text]
    chunks = []
    start = 0
    step = max(1, max_tokens - overlap)
    while start < len(ids):
        chunks.append(tokenizer.decode(ids[start:start+max_tokens]))
        start += step
    return chunks

class EmbeddingGenerator:
    def __init__(self): 
        print("[INFO] Loading Embedding Model...", flush=True)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    
    def embed(self, texts, batch_size=32): 
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True).tolist()

def stable_id(filename, index, text):
    """Creates a deterministic ID based on content hash."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{os.path.basename(filename)}::{index}::{h}"

def calculate_file_hash(filepath: str) -> str:
    """Calculates SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def is_document_ingested(client: QdrantClient, collection_name: str, file_hash: str) -> bool:
    """
    Checks if a document with the given hash already exists in the collection.
    """
    try:
        # Filter for points where 'doc_hash' == file_hash
        scroll_filter = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="doc_hash",
                    match=rest.MatchValue(value=file_hash)
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
        # If collection doesn't exist or other error, assume not ingested
        return False

def run_ingestion(pdf_files):
    if not pdf_files: 
        print("[INFO] No files to ingest.", flush=True)
        return
    
    print(f"\n--- Ingesting {len(pdf_files)} PDFs into Qdrant ---", flush=True)
    
    # 1. Setup Qdrant
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)
        try: 
            client.get_collection(QDRANT_COLLECTION)
        except: 
            print(f"[INFO] Creating collection '{QDRANT_COLLECTION}'...")
            client.create_collection(
                QDRANT_COLLECTION, 
                vectors_config=rest.VectorParams(size=384, distance=rest.Distance.COSINE)
            )
            # Create payload index for doc_hash to make deduplication fast
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="doc_hash",
                field_schema=rest.PayloadSchemaType.KEYWORD
            )
    except Exception as e:
        print(f"[ERROR] Could not connect to Qdrant: {e}")
        return

    # 2. Setup Local Models
    try:
        converter = DocumentConverter()
        chunker = HybridChunker(min_chunk_size=MIN_CHUNK_CHARS, max_chunk_size=MAX_CHUNK_CHARS)
        tokenizer = TokenizerWrapper()
        embedder = EmbeddingGenerator()
    except Exception as e:
        print(f"[ERROR] Failed to init local models: {e}")
        return

    # 3. Process Files
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        
        # --- DEDUPLICATION ---
        try:
            file_hash = calculate_file_hash(pdf_path)
            if is_document_ingested(client, QDRANT_COLLECTION, file_hash):
                print(f"[INFO] File '{filename}' (hash: {file_hash[:8]}...) already ingested. Skipping.", flush=True)
                continue
        except Exception as e:
            print(f"[WARN] Deduplication check failed for {filename}: {e}. Proceeding.")
            file_hash = "unknown"
        # ---------------------

        print(f"Processing: {filename}...", flush=True)
        
        try:
            # Convert PDF to text
            doc_res = converter.convert(pdf_path).document
            chunks = list(chunker.chunk(dl_doc=doc_res))
            
            sub_texts, sub_metas, sub_ids = [], [], []
            
            # Post-process chunks (Token splitting)
            for i, c in enumerate(chunks):
                txt = c.text.strip()
                if not txt: continue
                
                # Split large chunks by tokens
                subs = token_split_text(txt, MAX_TOKENS, OVERLAP_TOKENS, tokenizer)
                
                for j, s in enumerate(subs):
                    sub_texts.append(s)
                    sid = stable_id(pdf_path, f"{i}.{j}", s)
                    sub_ids.append(sid)
                    
                    # Metadata
                    meta = {
                        "source": filename,
                        "doc_hash": file_hash, # Store hash for future dedup
                        "parent_chunk": i,
                        "text_preview": s[:150] + "..."
                    }
                    sub_metas.append(meta)

            if not sub_texts:
                print("  [WARN] No text extracted.")
                continue

            # Embed and Upsert in Batches
            total_upserted = 0
            for k in range(0, len(sub_texts), QDRANT_UPSERT_BATCH):
                batch_txt = sub_texts[k:k+QDRANT_UPSERT_BATCH]
                batch_ids = sub_ids[k:k+QDRANT_UPSERT_BATCH]
                batch_meta = sub_metas[k:k+QDRANT_UPSERT_BATCH]
                
                # Create Vectors
                vecs = embedder.embed(batch_txt, batch_size=EMBED_BATCH_SIZE)
                
                # Create Points
                points = []
                for bid, v, m, t in zip(batch_ids, vecs, batch_meta, batch_txt):
                    # Qdrant requires UUID format for IDs
                    uuid_id = str(uuid.uuid5(uuid.NAMESPACE_URL, bid))
                    
                    payload = dict(m)
                    payload["text"] = t
                    payload["_stable_id"] = bid
                    
                    points.append(rest.PointStruct(id=uuid_id, vector=v, payload=payload))
                
                # Upload
                client.upsert(QDRANT_COLLECTION, points=points)
                total_upserted += len(points)
            
            print(f"  [OK] Upserted {total_upserted} chunks.")

        except Exception as e:
            print(f"  [ERROR] Failed to ingest {filename}: {e}")


# =================================================================
# MAIN LOGIC
# =================================================================

def normalize_string(s: str) -> str:
    """Simple normalization for deduplication."""
    if not s: return ""
    return re.sub(r'\W+', '', s.lower())

def model_query(query: str) -> Dict[str, str]:
    """
    Uses LLM to convert topic keywords into:
    1. ArXiv category codes (for ArXiv)
    2. Simplified keywords (for S2/CORE)
    Returns a dict: {"arxiv": "...", "keywords": "..."}
    """
    print(f"[INFO] No results found. Expanding query '{query}' using LLM...", flush=True)
    
    # Suppress LiteLLM logs
    import logging
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    
    prompt = f"""
    You are an assistant that converts topic keywords into search queries.
    Input: {query}
    Output: A JSON object with two keys:
    - "arxiv": 5-10 arXiv category codes separated by commas (e.g. "cs.AI, cs.LG").
    - "keywords": A concise, space-separated string of 3-5 main keywords suitable for semantic search (e.g. "Machine Learning Deep Learning").
    Do NOT output markdown code blocks, just the raw JSON string.
    """
    try:
        resp = completion(
            model="openrouter/openai/gpt-oss-20b:free",
            messages=[{"role":"system","content":"You are a concise conversion assistant."},
                      {"role":"user","content":prompt}],
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        content = resp.choices[0].message.content.strip()
        # Clean up potential markdown code blocks
        content = content.replace("```json", "").replace("```", "").strip()
        
        import json
        data = json.loads(content)
        return {
            "arxiv": data.get("arxiv", ""),
            "keywords": data.get("keywords", query)
        }
    except Exception as e:
        print(f"[WARN] LLM expansion failed: {e}")
        print(f"[INFO] Falling back to simple keyword splitting.")
        return {"arxiv": "", "keywords": query}

def process_direct_input(query: str) -> str:
    """
    Checks if query is a direct URL or ArXiv ID. 
    If so, downloads and ingests immediately.
    Returns a status message string if handled, or None if not a direct input.
    """
    query = query.strip()

    # 0. Special handling for ArXiv URLs/IDs (canonicalize to .pdf)
    # Matches: arxiv.org/pdf/ID, arxiv.org/abs/ID, or just ID
    arxiv_match = re.search(r'(?:arxiv\.org/(?:pdf|abs)/)?(\d{4}\.\d{4,5}(?:v\d+)?)', query)
    # If it looks like an ArXiv ID/URL, treat it as such
    if arxiv_match:
        # Check if it's a full URL but NOT arxiv (unlikely given the regex, but safe)
        # If the user passed a non-arxiv URL that happens to contain an ID pattern, we might want to be careful.
        # But the regex `(?:arxiv\.org...)?` makes it greedy for arxiv prefixes.
        
        # Let's be more specific:
        # 1. Explicit ArXiv URL
        is_arxiv_url = "arxiv.org" in query.lower()
        # 2. Just an ID (no http/https)
        is_plain_id = not re.match(r'^https?://', query)
        
        if is_arxiv_url or is_plain_id:
            arxiv_id = arxiv_match.group(1)
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            print(f"[INFO] Detected ArXiv ID/URL: {arxiv_id} -> {pdf_url}")
            
            session = requests.Session()
            session.headers.update({"User-Agent": "ResearchBot/1.0"})
            res = download_one(session, pdf_url, None, PAPER_DIR, DOWNLOAD_TIMEOUT)
            if res["ok"]:
                run_ingestion([res["path"]])
                print(f"[SUCCESS] Ingested ArXiv Paper: {arxiv_id}")
                return f"Success: Ingested ArXiv paper {arxiv_id}"
            else:
                print(f"[ERROR] Failed to download ArXiv Paper: {res.get('error')}")
                return f"Error: Failed to download ArXiv paper {arxiv_id} - {res.get('error')}"

    # 1. Generic URL (non-ArXiv)
    if re.match(r'^https?://\S+$', query):
        print(f"[INFO] Detected direct URL: {query}")
        session = requests.Session()
        session.headers.update({"User-Agent": "ResearchBot/1.0"})
        res = download_one(session, query, None, PAPER_DIR, DOWNLOAD_TIMEOUT)
        if res["ok"]:
            run_ingestion([res["path"]])
            print(f"[SUCCESS] Ingested direct URL: {query}")
            return f"Success: Ingested direct URL"
        else:
            print(f"[ERROR] Failed to download URL: {res.get('error')}")
            return f"Error: Failed to download URL - {res.get('error')}"

    return None

def unified_search_and_run(query: str, max_results: int = 5) -> str:
    """
    1. Check direct input.
    2. Search (ArXiv, S2, CORE).
    3. If NO results OR Errors -> LLM Expand -> Retry ALL sources.
    4. Deduplicate, Rank, Ingest.
    Returns a status message string.
    """
    # 0. Direct Input
    direct_result = process_direct_input(query)
    if direct_result:
        return direct_result

    print(f"--- Starting Pipeline for Query: '{query}' ---")
    
    # 1. Define pool size
    pool_size = max(POOL_MIN, int(max_results * POOL_MULTIPLIER))
    
    # 2. Gather Candidates
    all_candidates: List[PaperCandidate] = []
    errors_occurred = False
    
    # Helper to run searches safely
    def run_searches(q, limit, boost_arxiv=False):
        cands = []
        s2_ok, core_ok = False, False
        
        # ArXiv - always runs, optionally with boosted limit
        arxiv_limit = limit * 2 if boost_arxiv else limit
        cands.extend(search_arxiv_candidates(q, arxiv_limit))
        
        # Semantic Scholar (catch errors)
        try:
            s2_cands = search_semantic_scholar_candidates(q, limit)
            cands.extend(s2_cands)
            s2_ok = len(s2_cands) > 0
        except Exception as e:
            print(f"[WARN] Semantic Scholar search failed for '{q}': {e}")
            nonlocal errors_occurred
            errors_occurred = True

        # CORE (catch errors)
        try:
            core_cands = search_core_candidates(q, limit)
            cands.extend(core_cands)
            core_ok = len(core_cands) > 0
        except Exception as e:
            print(f"[WARN] CORE search failed for '{q}': {e}")
            errors_occurred = True
        
        # If S2 and CORE both failed/empty, boost ArXiv results
        if not s2_ok and not core_ok and not boost_arxiv:
            print("[INFO] S2/CORE returned no results - expanding ArXiv search...")
            extra_arxiv = search_arxiv_candidates(q, limit)  # Additional round
            cands.extend(extra_arxiv)
            
        return cands

    # Initial Search
    print("[INFO] Attempting Raw Search...")
    all_candidates.extend(run_searches(query, pool_size))
    
    # Fallback: LLM Expansion if (No Results OR Errors)
    if not all_candidates or errors_occurred:
        reason = "No results found" if not all_candidates else "Errors occurred during raw search"
        print(f"[INFO] {reason}. Attempting LLM expansion/refinement...")
        
        expanded_data = model_query(query)
        
        # 1. Retry ArXiv with Categories
        arxiv_q = expanded_data.get("arxiv")
        if arxiv_q:
            print(f"[INFO] Retrying ArXiv with categories: '{arxiv_q}'")
            all_candidates.extend(search_arxiv_candidates(arxiv_q, pool_size))
            
        # 2. Retry S2/CORE with Keywords
        kw_q = expanded_data.get("keywords")
        if kw_q and kw_q != query: # Only retry if different/simplified
            print(f"[INFO] Retrying S2/CORE with simplified keywords: '{kw_q}'")
            
            # S2
            try:
                all_candidates.extend(search_semantic_scholar_candidates(kw_q, pool_size))
            except Exception as e:
                print(f"[WARN] S2 Retry failed: {e}")
                
            # CORE
            try:
                all_candidates.extend(search_core_candidates(kw_q, pool_size))
            except Exception as e:
                print(f"[WARN] CORE Retry failed: {e}")
    
    if not all_candidates:
        print(f"[ERROR] No papers found in any source (even after expansion).")
        return f"No results found for query: '{query}'"

    print(f"\n[INFO] Total raw candidates: {len(all_candidates)}", flush=True)

    # 3. Deduplicate
    # Logic: Keep the first occurrence. 
    # Since we appended lists in order (ArXiv, S2, Core), this prioritizes ArXiv links.
    unique_candidates = []
    seen_titles = set()
    
    for cand in all_candidates:
        norm_title = normalize_string(cand.title)
        if norm_title and norm_title not in seen_titles:
            seen_titles.add(norm_title)
            unique_candidates.append(cand)
    
    print(f"[INFO] Candidates after deduplication: {len(unique_candidates)}", flush=True)

    # 4. Scoring & Re-ranking (Quality + Recency + Relevance)
    now = datetime.datetime.now(datetime.timezone.utc)
    scored_candidates = []
    
    # Weights: balanced for quality research papers
    W_RELEVANCE = 0.30   # Search engine rank
    W_RECENCY = 0.40     # How new the paper is
    W_QUALITY = 0.30     # Citation count (quality signal)

    for cand in unique_candidates:
        # Recency Score - exponential decay with ~1 year half-life
        days_old = 3650  # Default 10 years if date unknown
        if cand.published_date:
            # Ensure timezone awareness
            if cand.published_date.tzinfo is None:
                p_date = cand.published_date.replace(tzinfo=datetime.timezone.utc)
            else:
                p_date = cand.published_date
            
            delta = now - p_date
            days_old = max(0, delta.days)
        
        # Smoother decay: exp(-days/365) gives ~0.37 at 1 year, ~0.14 at 2 years
        recency_score = math.exp(-days_old / 365.0)
        
        # Relevance Score (based on search rank)
        relevance_score = 1.0 / (1 + cand.original_rank)
        
        # Quality Score (log-scaled citations, capped at 10 for normalization)
        # Papers with 0 citations: 0, 10 citations: ~0.24, 100: ~0.46, 1000: ~0.69
        quality_score = min(1.0, math.log(1 + cand.citation_count) / 10.0)
        
        # Final Score
        final_score = (W_RELEVANCE * relevance_score) + (W_RECENCY * recency_score) + (W_QUALITY * quality_score)
        
        scored_candidates.append((final_score, cand))

    # Sort descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Select top N
    selected = [x[1] for x in scored_candidates[:max_results]]
    
    print(f"\n[INFO] Top {len(selected)} Selected Papers:", flush=True)
    download_tasks = []
    
    for p in selected:
        d_str = p.published_date.strftime("%Y-%m-%d") if p.published_date else "Unknown"
        print(f"  [{p.source.upper()}] {p.title[:60]}... (Date: {d_str})", flush=True)
        
        # Format for downloader: URL Title : TitleString
        clean_title = re.sub(r'\s+', ' ', p.title).strip()
        line = f"{p.pdf_url} Title : {clean_title}"
        download_tasks.append(line)

    # 5. Write to temp file
    temp_file = "_discovered_links.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(download_tasks))
    
    # 6-7. Run Pipeline with cleanup guarantee
    try:
        downloaded_files = run_downloader(temp_file, PAPER_DIR, DOWNLOAD_TIMEOUT)
        
        if downloaded_files:
            run_ingestion(downloaded_files)
            print("\n--- Pipeline Complete: Success ---")
            
            # 8. Cleanup Papers
            print(f"[INFO] Cleaning up {len(downloaded_files)} downloaded papers...")
            for fpath in downloaded_files:
                try:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                except Exception as e:
                    print(f"[WARN] Failed to delete {fpath}: {e}")
            print("[INFO] Paper cleanup complete.")
            return f"Success: Ingested {len(downloaded_files)} paper(s) for query '{query}'"
        else:
            print("\n--- Pipeline Complete: No files downloaded ---")
            return f"No results found for query: '{query}' (download failed)"
    finally:
        # Always cleanup temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"[INFO] Cleaned up temporary file: {temp_file}")



if __name__ == "__main__":
    
    # --- ENTER YOUR QUERY HERE ---
    USER_QUERY = "Machine learning ,Deep Learning, Reinforcement Learning , Meta Learning,LLMs,Agentic AI"
    MAX_PAPERS = 5

    unified_search_and_run(USER_QUERY, MAX_PAPERS)