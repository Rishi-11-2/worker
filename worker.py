# worker.py
import os
import json
import time
import traceback
import sys
from dotenv import load_dotenv
load_dotenv()

# ---------------- Config ----------------
REDIS_URL = os.getenv('REDIS_URL')
QUEUE_KEY = "ingest:queue"
PENDING_SET = "ingest:pending"
PROCESSING_SET = "ingest:processing"
RESULT_LIST = "ingest:results"
MAX_RESULT_LOG = 200
BRPOP_TIMEOUT = 5

BATCH_MAX_ITEMS = 10

# print(REDIS_URL)
# ---------- Redis client ----------
if not REDIS_URL:
    raise RuntimeError("REDIS_URL environment variable must be set (e.g. redis://:PASS@host:6379/0).")

try:
    import redis
    r = redis.from_url(
        REDIS_URL,
        decode_responses=True,
    )
    r.ping()
    print("[worker] Redis connection successful.", flush=True)
except Exception as e:
    print(f"[worker] Failed to initialize Redis client: {e}", flush=True)
    sys.exit(1)

def try_acquire_lock(lock_key="worker:lock", ttl=600):
    """
    Acquire a lock for this run. Returns True if lock acquired, False if another run holds it.
    TTL should be >= expected max run time in seconds (default: 10 minutes).
    """
    try:
        # SETNX-like with expiry: set key only if not exist, with TTL seconds
        # redis-py >=4 supports set(..., nx=True, ex=ttl)
        ok = r.set(lock_key, "1", nx=True, ex=ttl)
        return bool(ok)
    except Exception as e:
        print("[worker] lock acquire error:", e, flush=True)
        return False

def release_lock(lock_key="worker:lock"):
    try:
        r.delete(lock_key)
    except Exception as e:
        print("[worker] lock release error:", e, flush=True)

# ---------- Ingestion clients & function ----------
try:
    # Import your ingestion function
    from rag_ingestion_service import (
        unified_search_and_run,
        process_direct_input,
    )
except ImportError as e:
    print(f"[worker] Failed to import ingestion modules: {e}", flush=True)
    print("[worker] Make sure rag_ingestion_service.py is in the same directory.", flush=True)
    sys.exit(1)


def _process_payload(payload: str):
    """
    Processes a single payload from the queue.
    """
    try:
        data = json.loads(payload)
    except Exception:
        print("[worker] malformed payload, skipping:", payload, flush=True)
        return
    if not data or "task_id" not in data or "query" not in data:
        print("[worker] bad payload, skipping:", data, flush=True)
        return

    task_id = data["task_id"]
    query = data["query"]

    # Move from pending -> processing (best-effort)
    try:
        r.srem(PENDING_SET, task_id)
        r.sadd(PROCESSING_SET, task_id)
    except Exception as e:
        print(f"[worker] warning updating redis sets: {e}", flush=True)

    print(f"[worker] processing task {task_id} query={query!r}", flush=True)
    status = "failed"
    result_msg = None
    try:
        # 1. Check if it's a direct URL or arXiv ID (returns message string or None)
        direct_result = process_direct_input(query)
        
        if direct_result:
            print(f"[worker] direct ingestion result: {direct_result}", flush=True)
            result_msg = direct_result
            # Check if the result indicates success or error
            if result_msg.startswith("Success"):
                status = "success"
            else:
                status = "failed"
        else:
            # 2. Fallback to search pipeline
            result_msg = unified_search_and_run(query, max_results=5)
            print(f"[worker] ingestion returned: {str(result_msg)[:400]}", flush=True)
            
            if result_msg and result_msg.startswith("Success"):
                status = "success"
            else:
                status = "failed"
    except Exception as e:
        print(f"[worker] ingestion exception for {task_id}: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        status = "failed"

    # Store only the query string
    try:
        r.lpush(RESULT_LIST, query)
        r.ltrim(RESULT_LIST, 0, MAX_RESULT_LOG - 1)
    except Exception as e:
        print("[worker] failed to write result:", e, flush=True)

    # Remove from processing so it can be re-enqueued later if needed
    try:
        r.srem(PROCESSING_SET, task_id)
    except Exception as e:
        print("[worker] warning removing from processing set:", e, flush=True)

    print(f"[worker] finished task {task_id} status={status}", flush=True)


def process_batch_once(max_items=100):
    """
    Non-blocking drain: process up to max_items from the queue and exit.
    Uses r.rpop to match the same L/R semantics as BRPOP in the original loop.
    """
    print(f"[worker] drain mode: up to {max_items} items", flush=True)
    processed = 0
    while processed < max_items:
        try:
            payload = r.rpop(QUEUE_KEY)   # non-blocking pop; adjust to lpop if desired
            if not payload:
                break
            _process_payload(payload)
            processed += 1
        except redis.exceptions.RedisError as re:
            print("[worker] Redis error during drain:", re, flush=True)
            print(traceback.format_exc(), flush=True)
            time.sleep(1)
        except Exception:
            print("[worker] exception during drain loop:", flush=True)
            print(traceback.format_exc(), flush=True)
            time.sleep(1)
    print(f"[worker] drain finished, processed {processed} items", flush=True)


def worker_loop():
    """
    Main worker loop (blocking BRPOP).
    """
    print("[worker] starting loop...", flush=True)
    while True:
        try:
            item = r.brpop(QUEUE_KEY, timeout=BRPOP_TIMEOUT)
            if not item:
                continue
            _, payload = item
            _process_payload(payload)
        except redis.exceptions.RedisError as re:
            print("[worker] Redis error:", re, flush=True)
            print(traceback.format_exc(), flush=True)
            time.sleep(1)
        except Exception as e:
            print("[worker] exception in main loop:", e, flush=True)
            print(traceback.format_exc(), flush=True)
            time.sleep(1)


if __name__ == "__main__":
    print("[worker] Starting worker...", flush=True)

    if try_acquire_lock(lock_key="worker:lock", ttl=600):
        try:
            process_batch_once(max_items=BATCH_MAX_ITEMS)
        finally:
            release_lock("worker:lock")
            print("[worker] Lock released.", flush=True)
    else:
        print("[worker] Another worker is already running. Exiting.", flush=True)
