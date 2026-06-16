"""
Stacking Benjamins RAG API.

Production FastAPI server for question-answering against Stacking Benjamins
guides. Uses HyPE (Hypothetical Prompt Embeddings) for retrieval, Claude Haiku
for synthesis, and Supabase for query logging.

Run locally:
    uvicorn app:app --reload --port 8000

Run in production (Railway uses this command via railway.json):
    gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 600 \\
        --worker-class uvicorn.workers.UvicornWorker --preload
"""

import json
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

import anthropic
import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

from prompts import get_system_prompt, format_user_prompt
from scope_config import (
    SCOPE_CONFIG,
    SOURCE_TO_SCOPE,
    REFUSE_THRESHOLD,
    CHUNKS_FOR_SYNTHESIS,
    REFUSAL_MESSAGE,
    get_upsell_message,
    find_topic_override,
)


# ---------- Environment ----------

load_dotenv()  # Loads .env in local dev; no-op in production (env vars set in Railway)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
PAGE_TOKEN = os.environ["PAGE_TOKEN"]
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DATA_DIR = os.getenv("DATA_DIR", "data")


# ---------- Logging ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sb-rag")


# ---------- Global state (populated at startup) ----------

class AppState:
    embed_model: SentenceTransformer
    anthropic_client: anthropic.Anthropic
    supabase: Client
    hype_records: list
    section_id2meta: list
    hype_index: faiss.IndexFlatIP


state = AppState()


# ---------- Startup ----------

def load_artifacts():
    """Load HyPE artifacts from disk and build the in-memory FAISS index."""
    logger.info("Loading HyPE artifacts from %s/", DATA_DIR)

    with open(f"{DATA_DIR}/hype_records.json", "r", encoding="utf-8") as f:
        state.hype_records = json.load(f)
    logger.info("  hype_records: %d question records", len(state.hype_records))

    with open(f"{DATA_DIR}/section_id2meta.json", "r", encoding="utf-8") as f:
        state.section_id2meta = json.load(f)
    logger.info("  section_id2meta: %d chunks", len(state.section_id2meta))

    emb_matrix = np.load(f"{DATA_DIR}/hype_emb_matrix.npy")
    logger.info("  hype_emb_matrix: shape %s", emb_matrix.shape)

    state.hype_index = faiss.IndexFlatIP(emb_matrix.shape[1])
    state.hype_index.add(emb_matrix)
    logger.info("  FAISS index built: %d vectors, %d dims", state.hype_index.ntotal, emb_matrix.shape[1])


def load_embed_model():
    """Initialize the sentence-transformers model used for query embedding."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading embedding model %s on %s", EMBED_MODEL_NAME, device)
    state.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    # Warm up: trigger the first inference call during startup, not during a request.
    # Some torch+macOS combos segfault on first encode(); better to crash loudly here.
    logger.info("Warming up embedding model with a dummy query...")
    _ = state.embed_model.encode(["warmup query"], normalize_embeddings=True, convert_to_numpy=True)
    logger.info("Embedding model ready")


def init_clients():
    """Set up Anthropic and Supabase clients."""
    state.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    state.supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    logger.info("Anthropic and Supabase clients initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run once at app startup; cleanup on shutdown."""
    logger.info("=" * 60)
    logger.info("Starting Stacking Benjamins RAG API")
    logger.info("=" * 60)
    load_embed_model()
    load_artifacts()
    init_clients()
    logger.info("Startup complete - ready to serve")
    yield
    logger.info("Shutting down")


# ---------- Retrieval ----------

def embed_query(query: str) -> np.ndarray:
    """Embed a single query using bge-small-en-v1.5."""
    emb = state.embed_model.encode(
        [query],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return emb.astype("float32")


def hype_search(query: str, retrieve_k: int = 20) -> list:
    """
    Search by matching the user's query against indexed hypothetical questions.
    Returns chunks (not questions) sorted by best question-match score.
    """
    q_emb = embed_query(query)
    scores, ids = state.hype_index.search(q_emb, retrieve_k * 2)

    # Deduplicate by chunk_id - keep best score per chunk
    seen_chunks = {}
    for score, q_idx in zip(scores[0], ids[0]):
        record = state.hype_records[int(q_idx)]
        chunk_id = record["chunk_id"]
        if chunk_id not in seen_chunks or score > seen_chunks[chunk_id]["match_score"]:
            seen_chunks[chunk_id] = {
                "match_score": float(score),
                "matched_question": record["question"],
            }

    sorted_chunks = sorted(seen_chunks.items(), key=lambda x: x[1]["match_score"], reverse=True)[:retrieve_k]

    results = []
    for chunk_id, match_info in sorted_chunks:
        meta = state.section_id2meta[chunk_id]
        results.append({
            "chunk_id": chunk_id,
            "source": meta["source"],
            "section_heading": meta.get("section_heading", ""),
            "text": meta["text"],
            "match_score": match_info["match_score"],
            "matched_question": match_info["matched_question"],
        })
    return results


def format_chunks_for_synthesis(results: list) -> str:
    """Build a clean labeled chunk listing to send to Haiku."""
    parts = []
    for i, r in enumerate(results, 1):
        source_label = r["source"].replace(".md", "").replace("_", " ")
        heading = r.get("section_heading", "") or "(no heading)"
        parts.append(f"[Passage {i}] from {source_label}, section {heading}\n{r['text']}\n")
    return "\n".join(parts)


# ---------- Synthesis ----------

def synthesize_answer(query: str, chunks: list) -> dict:
    """Call Haiku to generate an answer from the retrieved chunks."""
    chunks_text = format_chunks_for_synthesis(chunks)
    # Compute current year at request time. The prompt tells Haiku what year
    # it is; Haiku infers the source's tax year from the chunks themselves
    # (which explicitly label dollar amounts with "for 2025" etc.) and
    # handles any mismatch. No code changes needed when guides are updated.
    current_year = datetime.now(timezone.utc).year
    system_prompt = get_system_prompt(current_year)
    response = state.anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": format_user_prompt(query, chunks_text)}],
    )
    answer = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    # Haiku 4.5 pricing: $1 input / $5 output per million tokens
    cost = (input_tokens * 1.0 + output_tokens * 5.0) / 1_000_000
    return {
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }


# ---------- Mode A5 routing ----------

def route_query(query: str, scope: str) -> dict:
    """
    Three-outcome routing:
      1. ANSWER  - in-scope retrieval crosses threshold; synthesize via Haiku
      2. UPSELL  - in-scope misses but another guide has a strong match
      3. REFUSE  - nothing in any scope crosses threshold

    Returns a dict ready to serialize as JSON to the client.
    """
    if scope not in SCOPE_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown scope: {scope}")

    config = SCOPE_CONFIG[scope]
    allowed_sources = set(config["allowed_sources"])

    # Topic override check: handle known problem cases where retrieval gives
    # the "wrong" answer for product reasons. E.g., 529 questions always belong
    # to college-guide even though tax-guide has detailed 529 content.
    # See TOPIC_OVERRIDES in scope_config.py.
    override_scope = find_topic_override(query)
    if override_scope is not None and override_scope != scope:
        return {
            "decision": "UPSELL",
            "scope": scope,
            "top_score": 1.0,  # Synthetic - override fired before retrieval
            "answer": get_upsell_message(override_scope),
            "upsell_target": override_scope,
            "upsell_url": SCOPE_CONFIG[override_scope]["url"],
            "sources": [],
            "cost_usd": 0.0,
        }

    # Augment the query with the current year so retrieval prefers chunks
    # labeled with the present-day year over historical ones. Users who ask
    # "this year" or don't specify a year implicitly mean the current year -
    # this nudges HyPE matching toward chunks tagged with that year.
    current_year = datetime.now(timezone.utc).year
    augmented_query = f"{query} ({current_year})"
    all_results = hype_search(augmented_query, retrieve_k=20)

    if not all_results:
        return {
            "decision": "REFUSE",
            "scope": scope,
            "top_score": 0.0,
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "cost_usd": 0.0,
        }

    in_scope = [r for r in all_results if r["source"] in allowed_sources]
    out_scope = [r for r in all_results if r["source"] not in allowed_sources]

    # Step 1: in-scope match good enough? -> ANSWER
    if in_scope and in_scope[0]["match_score"] >= REFUSE_THRESHOLD:
        top_chunks = in_scope[:CHUNKS_FOR_SYNTHESIS]
        synth = synthesize_answer(query, top_chunks)
        return {
            "decision": "ANSWER",
            "scope": scope,
            "top_score": in_scope[0]["match_score"],
            "answer": synth["answer"],
            "sources": [
                {
                    "source": r["source"],
                    "section_heading": r["section_heading"],
                    "match_score": r["match_score"],
                }
                for r in top_chunks
            ],
            "matched_question": in_scope[0]["matched_question"],
            "input_tokens": synth["input_tokens"],
            "output_tokens": synth["output_tokens"],
            "cost_usd": synth["cost_usd"],
        }

    # Step 2: any out-of-scope chunk crosses threshold AND maps to a sellable guide? -> UPSELL
    for r in out_scope:
        if r["match_score"] < REFUSE_THRESHOLD:
            break  # Sorted desc; nothing else will qualify
        target_scope = SOURCE_TO_SCOPE.get(r["source"])
        if target_scope is not None:
            return {
                "decision": "UPSELL",
                "scope": scope,
                "top_score": r["match_score"],
                "answer": get_upsell_message(target_scope),
                "upsell_target": target_scope,
                "upsell_url": SCOPE_CONFIG[target_scope]["url"],
                "sources": [],
                "cost_usd": 0.0,
            }

    # Step 3: REFUSE
    return {
        "decision": "REFUSE",
        "scope": scope,
        "top_score": all_results[0]["match_score"],
        "answer": REFUSAL_MESSAGE,
        "sources": [],
        "cost_usd": 0.0,
    }


# ---------- Logging ----------

def log_query(query: str, scope: str, result: dict, latency_ms: int):
    """Write a row to Supabase. Failures here should not break the user response."""
    try:
        row = {
            "query": query,
            "scope": scope,
            "decision": result["decision"],
            "top_score": result.get("top_score"),
            "answer": result.get("answer"),
            "matched_question": result.get("matched_question"),
            "upsell_target": result.get("upsell_target"),
            "source_chunks": result.get("sources", []),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cost_usd": result.get("cost_usd", 0.0),
            "latency_ms": latency_ms,
        }
        state.supabase.table("query_logs").insert(row).execute()
    except Exception as e:
        # Log to console but don't fail the user request
        logger.error("Failed to log query to Supabase: %s", e)


# ---------- Rate limiting (in-memory) ----------
# Tracks 5 queries/hour/IP. Resets on container restart, which is fine for beta.
# If Railway ever runs multiple replicas, swap to a Supabase-backed counter.

RATE_LIMIT_WINDOW_SECONDS = 3600  # 1 hour
RATE_LIMIT_MAX = 5

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
_rate_limit_lock = Lock()


def check_rate_limit(ip: str) -> tuple[bool, int]:
    """
    Returns (allowed, remaining_seconds_until_next_slot).
    remaining_seconds is 0 when allowed.
    """
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    with _rate_limit_lock:
        timestamps = [t for t in _rate_limit_store[ip] if t > cutoff]
        if len(timestamps) >= RATE_LIMIT_MAX:
            oldest = min(timestamps)
            wait = int(oldest + RATE_LIMIT_WINDOW_SECONDS - now)
            _rate_limit_store[ip] = timestamps
            return False, max(wait, 1)
        timestamps.append(now)
        _rate_limit_store[ip] = timestamps
        return True, 0


# ---------- FastAPI app ----------

app = FastAPI(
    title="Stacking Benjamins RAG API",
    description="Question-answering against Stacking Benjamins guides",
    version="1.5.0",
    lifespan=lifespan,
)

# CORS - allow requests from the Stacking Benjamins domain and the Firebase pages.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.stackingbenjamins.com",
        "https://stackingbenjamins.com",
        "https://sb-ai-chat.web.app",
        "https://sb-ai-chat.firebaseapp.com",
        "http://localhost:3000",  # for local widget testing
        "http://localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    scope: str = Field(..., description="One of: tax-guide, college-guide, workplace-benefits")


@app.get("/health")
def health():
    """Health check for Railway uptime monitoring. Also touches Supabase to
    keep the free-tier project from being paused for inactivity (Supabase
    pauses free projects after 7 days with no database activity)."""
    # Lightweight Supabase ping - keeps the inactivity timer fresh.
    # We just check that we can query the query_logs table; we don't care
    # about the result. Failure is logged but doesn't fail the health check.
    db_alive = False
    try:
        state.supabase.table("query_logs").select("id").limit(1).execute()
        db_alive = True
    except Exception as e:
        logger.warning("Supabase health ping failed: %s", e)
    return {
        "status": "ok",
        "chunks_loaded": len(state.section_id2meta) if hasattr(state, "section_id2meta") else 0,
        "questions_indexed": len(state.hype_records) if hasattr(state, "hype_records") else 0,
        "scopes_available": list(SCOPE_CONFIG.keys()),
        "db_alive": db_alive,
    }


@app.post("/search")
def search(request: SearchRequest, http_request: Request):
    """Main endpoint: take a query + scope, return an answer/upsell/refusal."""
    # Token check - simple shared secret. Stops casual API abuse from anyone who
    # finds the URL but isn't paying. Anyone with a page can still extract the
    # token from JS, but that's fine - the rate limit caps damage.
    token = http_request.headers.get("X-Page-Token", "")
    if token != PAGE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing page token")

    # Rate limit - per IP, 5 per hour
    client_ip = http_request.client.host if http_request.client else "unknown"
    allowed, wait_seconds = check_rate_limit(client_ip)
    if not allowed:
        minutes = max(wait_seconds // 60, 1)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {minutes} minutes.",
        )

    start = time.time()
    try:
        result = route_query(request.query, request.scope)
        latency_ms = int((time.time() - start) * 1000)
        log_query(request.query, request.scope, result, latency_ms)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Search failed for query=%r scope=%r", request.query, request.scope)
        raise HTTPException(status_code=500, detail="Internal error")