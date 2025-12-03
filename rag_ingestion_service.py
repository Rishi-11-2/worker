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

# --- Chunking ---
MAX_TOKENS = 450
OVERLAP_TOKENS = 75
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
        "fields": "title,openAccessPdf,publicationDate,url"
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
                        original_rank=idx
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

def model_query(query: str) -> str:
    """
    Uses LLM to convert topic keywords into ArXiv category codes.
    Fallback: If LLM fails, returns comma-separated parts of the query.
    """
    print(f"[INFO] No results found. Expanding query '{query}' using LLM...", flush=True)
    
    # Suppress LiteLLM logs
    import logging
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    
    prompt = f"""
    You are an assistant that converts topic keywords into arXiv category codes.
    Input: {query}
    Output: EXACTLY one line containing 5-10 arXiv category codes separated by commas (example: cs.AI, cs.LG, stat.ML).
    Do NOT output JSON, bullet lists, or explanation.
    """
    try:
        # Using the model from user's previous code
        resp = completion(
            model="openrouter/openai/gpt-oss-20b:free",
            messages=[{"role":"system","content":"You are a concise conversion assistant."},
                      {"role":"user","content":prompt}],
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        content = resp.choices[0].message.content.strip()
        if not content: raise ValueError("Empty response from LLM")
        return content
    except Exception as e:
        print(f"[WARN] LLM expansion failed: {e}")
        print(f"[INFO] Falling back to simple keyword splitting.")
        # Fallback: Just use the query parts as keywords
        # e.g. "Machine Learning, Deep Learning" -> "Machine Learning, Deep Learning"
        # This allows the retry logic to run with shorter chunks if the original was too long
        return query

def process_direct_input(query: str) -> bool:
    """
    Checks if query is a direct URL or ArXiv ID. 
    If so, downloads and ingests immediately.
    Returns True if handled.
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
                return True
            else:
                print(f"[ERROR] Failed to download ArXiv Paper: {res.get('error')}")
                return True

    # 1. Generic URL (non-ArXiv)
    if re.match(r'^https?://\S+$', query):
        print(f"[INFO] Detected direct URL: {query}")
        session = requests.Session()
        session.headers.update({"User-Agent": "ResearchBot/1.0"})
        res = download_one(session, query, None, PAPER_DIR, DOWNLOAD_TIMEOUT)
        if res["ok"]:
            run_ingestion([res["path"]])
            print(f"[SUCCESS] Ingested direct URL: {query}")
            return True
        else:
            print(f"[ERROR] Failed to download URL: {res.get('error')}")
            return True # Handled, even if failed

    return False

def unified_search_and_run(query: str, max_results: int = 5):
    """
    1. Check direct input.
    2. Search (ArXiv, S2, CORE).
    3. If NO results OR Errors -> LLM Expand -> Retry ALL sources.
    4. Deduplicate, Rank, Ingest.
    """
    # 0. Direct Input
    if process_direct_input(query):
        return

    print(f"--- Starting Pipeline for Query: '{query}' ---")
    
    # 1. Define pool size
    pool_size = max(POOL_MIN, int(max_results * POOL_MULTIPLIER))
    
    # 2. Gather Candidates
    all_candidates: List[PaperCandidate] = []
    errors_occurred = False
    
    # Helper to run searches safely
    def run_searches(q, limit):
        cands = []
        # ArXiv
        cands.extend(search_arxiv_candidates(q, limit))
        
        # Semantic Scholar (catch errors)
        try:
            s2_cands = search_semantic_scholar_candidates(q, limit)
            if not s2_cands and " " in q and len(q) > 50: # likely a complex query failure
                raise ValueError("Complex query yielded no results or failed")
            cands.extend(s2_cands)
        except Exception as e:
            print(f"[WARN] Semantic Scholar search failed for '{q}': {e}")
            nonlocal errors_occurred
            errors_occurred = True

        # CORE (catch errors)
        try:
            core_cands = search_core_candidates(q, limit)
            if not core_cands and " " in q and len(q) > 50:
                raise ValueError("Complex query yielded no results or failed")
            cands.extend(core_cands)
        except Exception as e:
            print(f"[WARN] CORE search failed for '{q}': {e}")
            errors_occurred = True
            
        return cands

    # Initial Search
    print("[INFO] Attempting Raw Search...")
    all_candidates.extend(run_searches(query, pool_size))
    
    # Fallback: LLM Expansion if (No Results OR Errors)
    # We trigger this if we have 0 candidates, OR if we had errors (likely due to bad query)
    if not all_candidates or errors_occurred:
        reason = "No results found" if not all_candidates else "Errors occurred during raw search"
        print(f"[INFO] {reason}. Attempting LLM expansion/refinement...")
        
        categories = model_query(query)
        if categories:
            print(f"[INFO] LLM suggested categories/keywords: {categories}")
            # Retry ALL sources with the refined categories
            # This helps S2/CORE which might have failed on the long raw query
            print(f"[INFO] Retrying search with refined query: '{categories}'")
            all_candidates.extend(run_searches(categories, pool_size))
    
    if not all_candidates:
        print(f"[ERROR] No papers found in any source (even after expansion).")
        return

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

    # 4. Scoring & Re-ranking
    now = datetime.datetime.now(datetime.timezone.utc)
    scored_candidates = []

    for cand in unique_candidates:
        # Recency Score
        days_old = 3650 # Default 10 years if date unknown
        if cand.published_date:
            # Ensure timezone awareness
            if cand.published_date.tzinfo is None:
                p_date = cand.published_date.replace(tzinfo=datetime.timezone.utc)
            else:
                p_date = cand.published_date
            
            delta = now - p_date
            days_old = max(0, delta.days)
        
        recency_score = 1.0 / (1 + days_old)
        
        # Relevance Score (based on search rank)
        relevance_score = 1.0 / (1 + cand.original_rank)
        
        # Final Score
        final_score = (ALPHA_RELEVANCE * relevance_score) + (BETA_RECENCY * recency_score)
        
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
    
    # 6. Run Pipeline
    downloaded_files = run_downloader(temp_file, PAPER_DIR, DOWNLOAD_TIMEOUT)
    
    if downloaded_files:
        run_ingestion(downloaded_files)
        print("\n--- Pipeline Complete: Success ---")
    else:
        print("\n--- Pipeline Complete: No files downloaded ---")


if __name__ == "__main__":
    
    # --- ENTER YOUR QUERY HERE ---
    USER_QUERY = "Machine learning ,Deep Learning, Reinforcement Learning , Meta Learning,LLMs,Agentic AI"
    MAX_PAPERS = 10

    unified_search_and_run(USER_QUERY, MAX_PAPERS)