"""
Benchmark FastAPI app — wraps PrismCache around a real or mock LLM.

Endpoints:
  POST /query          — semantic cache lookup + LLM fallback
  POST /query/batch    — batch queries
  GET  /metrics        — live cache metrics snapshot
  GET  /health         — liveness probe
  POST /admin/seed     — load seed data into the cache
  POST /admin/reset    — flush the cache (new test run)
  GET  /admin/status   — cache size, mode, config

All requests are traced via Azure Application Insights (when configured).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from prism.cache import PrismCache, HashEmbedder
from benchmark.app.config import get_config
from benchmark.app.telemetry import setup_telemetry, trace_request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local index — the "PrismDriver" in-process vector store
# ---------------------------------------------------------------------------

WRAPPER_URL = os.getenv("PRISM_WRAPPER_URL", "")
TARGET_DIM  = int(os.getenv("PRISM_TARGET_DIM", "64"))
TENANT_ID_D = os.getenv("PRISM_TENANT_ID", "benchmark-app")


class _LocalIndex:
    """
    In-process float32 similarity index.
    Warmed by pulling WAL events from the Server Wrapper (wrapper-sim).
    Represents the local PrismResonance index inside PrismDriver.
    """

    def __init__(self, tenant_id: str, dim: int = 64) -> None:
        self._tenant_id = tenant_id
        self._dim = dim
        self._rows: list[dict] = []
        self._matrix: Optional[np.ndarray] = None
        self._dirty = True
        self._warmed_at: Optional[float] = None
        self._rows_received = 0
        self._query_count = 0
        self._total_latency_ms = 0.0

    def ingest(self, row_id: str, text_repr: str, vector: list[float]) -> None:
        """Feed one WAL event into the local index."""
        self._rows.append({
            "row_id": row_id,
            "text_repr": text_repr,
            "vector": np.array(vector, dtype=np.float32),
        })
        self._dirty = True
        self._rows_received += 1

    def _rebuild(self) -> None:
        if not self._rows:
            self._matrix = None
            return
        self._matrix = np.stack([r["vector"] for r in self._rows])
        self._dirty = False

    def query(self, query_vector: list[float], top_k: int = 5,
              threshold: float = 0.5) -> tuple[list[dict], float]:
        """Sub-millisecond in-process similarity search."""
        t0 = time.perf_counter()
        if self._dirty:
            self._rebuild()
        if self._matrix is None or len(self._rows) == 0:
            return [], 0.0

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return [], 0.0
        q = q / q_norm

        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True) + 1e-8
        scores = (self._matrix / norms) @ q
        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_idx:
            score = float(scores[idx])
            if score < threshold:
                break
            results.append({
                "row_id":    self._rows[idx]["row_id"],
                "text_repr": self._rows[idx]["text_repr"],
                "score":     score,
            })

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._query_count += 1
        self._total_latency_ms += elapsed_ms
        return results, elapsed_ms

    def reset(self) -> int:
        n = len(self._rows)
        self._rows.clear()
        self._matrix = None
        self._dirty = True
        self._warmed_at = None
        self._rows_received = 0
        self._query_count = 0
        self._total_latency_ms = 0.0
        return n

    @property
    def size(self) -> int:
        return len(self._rows)

    @property
    def avg_latency_ms(self) -> float:
        if self._query_count == 0:
            return 0.0
        return self._total_latency_ms / self._query_count


_local_index: _LocalIndex = _LocalIndex(TENANT_ID_D, TARGET_DIM)

# Stats for driver benchmark
_driver_stats = {
    "baseline_queries": 0,
    "baseline_total_ms": 0.0,
    "driver_queries": 0,
    "driver_total_ms": 0.0,
    "warmup_rows": 0,
    "warmup_duration_ms": 0.0,
}

# ---------------------------------------------------------------------------
# Cache singleton
# ---------------------------------------------------------------------------

_cache: PrismCache | None = None


def get_cache() -> PrismCache:
    if _cache is None:
        raise RuntimeError("Cache not initialised — app not started yet.")
    return _cache


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def make_llm_fn(question: str, cfg: Any):
    """Return a zero-arg callable that calls the LLM (or mock)."""
    if cfg.use_mock_llm:
        def mock_llm():
            # Deterministic mock: simulates 50-150ms LLM latency
            time.sleep(0.08)
            return f"[mock] Answer to: {question[:60]}"
        return mock_llm
    else:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.openai_api_key)
        def real_llm():
            resp = client.chat.completions.create(
                model=cfg.llm_model,
                messages=[{"role": "user", "content": question}],
                max_tokens=256,
            )
            return resp.choices[0].message.content
        return real_llm


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cache
    cfg = get_config()
    setup_telemetry(cfg.appinsights_connection_string)

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        embedder = None  # let PrismCache.build() pick SentenceTransformerEmbedder
    except ImportError:
        embedder = HashEmbedder(output_dim=384)
        logger.warning("sentence-transformers not installed — using HashEmbedder")

    _cache = PrismCache.build(
        tenant_id=cfg.tenant_id,
        llm_model=cfg.llm_model,
        similarity_threshold=cfg.similarity_threshold,
        ttl_seconds=cfg.ttl_seconds,
        embedder=embedder,
    )
    logger.info("PrismCache ready (tenant=%s, mock=%s)", cfg.tenant_id, cfg.use_mock_llm)
    yield
    _cache.invalidate_all()
    logger.info("PrismCache shut down.")


app = FastAPI(
    title="PrismLib Benchmark",
    description="Load-test harness for PrismCache semantic LLM cache",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    expected_tokens: int = Field(default=256, ge=1)

class QueryResponse(BaseModel):
    answer:     str
    cache_hit:  bool
    latency_ms: float
    similarity: float | None = None

class BatchQueryRequest(BaseModel):
    questions: list[str] = Field(..., min_items=1, max_items=100)

class BatchQueryResponse(BaseModel):
    results:        list[QueryResponse]
    total_ms:       float
    hits:           int
    misses:         int
    hit_rate_pct:   float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
@trace_request("query")
async def query(req: QueryRequest):
    cfg    = get_config()
    cache  = get_cache()
    t0     = time.monotonic()

    hits_before = cache.get_metrics().total_hits

    answer = await cache.aget_or_call(
        query=req.question,
        call_fn=make_llm_fn(req.question, cfg),
        tokens_in_response=req.expected_tokens,
    )

    latency_ms = (time.monotonic() - t0) * 1000
    was_hit    = cache.get_metrics().total_hits > hits_before

    return QueryResponse(
        answer=str(answer)[:500],
        cache_hit=was_hit,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/query/batch", response_model=BatchQueryResponse)
@trace_request("batch_query")
async def batch_query(req: BatchQueryRequest):
    cfg   = get_config()
    cache = get_cache()
    t0    = time.monotonic()

    tasks = [
        cache.aget_or_call(q, make_llm_fn(q, cfg))
        for q in req.questions
    ]
    answers = await asyncio.gather(*tasks)

    total_ms = (time.monotonic() - t0) * 1000
    m = cache.get_metrics()

    results = [
        QueryResponse(answer=str(a)[:200], cache_hit=False, latency_ms=0)
        for a in answers
    ]
    return BatchQueryResponse(
        results=results,
        total_ms=round(total_ms, 2),
        hits=m.total_hits,
        misses=m.total_misses,
        hit_rate_pct=round(m.hit_rate_pct, 1),
    )


@app.get("/metrics")
async def metrics():
    m = get_cache().get_metrics()
    return {
        "total_queries":          m.total_queries,
        "total_hits":             m.total_hits,
        "total_misses":           m.total_misses,
        "hit_rate_pct":           round(m.hit_rate_pct, 2),
        "avg_hit_latency_ms":     round(m.avg_hit_latency_ms, 2),
        "avg_miss_latency_ms":    round(m.avg_miss_latency_ms, 2),
        "speedup_factor":         round(m.speedup_factor, 1),
        "total_tokens_saved":     m.total_tokens_saved,
        "total_cost_saved_usd":   round(m.total_cost_saved_usd, 4),
        "projected_monthly_usd":  round(m.projected_monthly_savings_usd, 2),
        "cache_size":             get_cache().cache_size,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": get_cache().cache_size}


@app.post("/admin/seed")
async def seed(count: int = 1000):
    """Pre-load N Q&A pairs so the cache is warm before load testing."""
    from benchmark.load.seed_data import get_seed_questions
    cfg   = get_config()
    cache = get_cache()

    questions = get_seed_questions(count)
    loaded = 0
    for q in questions:
        await cache.aget_or_call(q, make_llm_fn(q, cfg))
        loaded += 1

    return {"seeded": loaded, "cache_size": cache.cache_size}


@app.post("/admin/reset")
async def reset():
    evicted = get_cache().invalidate_all()
    return {"evicted": evicted}


@app.get("/admin/status")
async def status():
    cfg = get_config()
    return {
        "tenant_id":           cfg.tenant_id,
        "llm_model":           cfg.llm_model,
        "similarity_threshold": cfg.similarity_threshold,
        "use_mock_llm":        cfg.use_mock_llm,
        "cache_size":          get_cache().cache_size,
    }


# ---------------------------------------------------------------------------
# PrismDriver endpoints
# (baseline = every query hits the wrapper over the network;
#  driver   = every query hits the local in-process index)
# ---------------------------------------------------------------------------

def _make_query_vector(text: str) -> list[float]:
    """Deterministic float32 query vector from text (mirrors wrapper_sim projection)."""
    h = hashlib.sha256(text.encode()).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(TARGET_DIM).astype(np.float32)
    norm = np.linalg.norm(v)
    return (v / norm if norm > 0 else v).tolist()


@app.post("/driver/warmup")
async def driver_warmup(count: int = Query(default=5000, ge=1, le=100000)):
    """
    Pull WAL events from the Server Wrapper and load them into the local index.
    This is what the background Subscribe() loop does in PrismDriver.
    Call once before running the 'with-driver' benchmark phase.
    """
    if not WRAPPER_URL:
        raise HTTPException(502, "PRISM_WRAPPER_URL not set — no wrapper to subscribe to")

    t0 = time.monotonic()
    rows_loaded = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("GET", f"{WRAPPER_URL}/wal/subscribe",
                                 params={"limit": count}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                import json as _json
                event = _json.loads(line)
                _local_index.ingest(
                    row_id=event["row_id"],
                    text_repr=event["text_repr"],
                    vector=event["vector"],
                )
                rows_loaded += 1

    elapsed_ms = (time.monotonic() - t0) * 1000
    _local_index._warmed_at = time.time()
    _driver_stats["warmup_rows"] = rows_loaded
    _driver_stats["warmup_duration_ms"] = elapsed_ms

    logger.info("driver warmup: loaded %d rows in %.1fms", rows_loaded, elapsed_ms)
    return {
        "rows_loaded":  rows_loaded,
        "elapsed_ms":   round(elapsed_ms, 1),
        "index_size":   _local_index.size,
        "throughput_rows_per_s": round(rows_loaded / (elapsed_ms / 1000), 0),
    }


@app.post("/driver/baseline")
async def driver_baseline(text: str = Query(..., min_length=1)):
    """
    BASELINE path: proxy the query to the Server Wrapper over the network.
    Simulates direct DB access — every query pays the network round-trip.
    """
    if not WRAPPER_URL:
        raise HTTPException(502, "PRISM_WRAPPER_URL not set")

    t0 = time.monotonic()
    vector = _make_query_vector(text)

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{WRAPPER_URL}/query",
                                 json={"vector": vector, "top_k": 5, "threshold": 0.5})
        resp.raise_for_status()
        data = resp.json()

    elapsed_ms = (time.monotonic() - t0) * 1000
    _driver_stats["baseline_queries"] += 1
    _driver_stats["baseline_total_ms"] += elapsed_ms

    return {
        "results":    data.get("results", []),
        "elapsed_ms": round(elapsed_ms, 2),
        "source":     "db-node-network",
        "index_size": _local_index.size,
    }


@app.post("/driver/query")
async def driver_query(text: str = Query(..., min_length=1)):
    """
    DRIVER path: query the local in-process PrismResonance index.
    Sub-millisecond — no network hop, data was pushed here by the wrapper.
    """
    vector = _make_query_vector(text)
    results, elapsed_ms = _local_index.query(vector, top_k=5, threshold=0.5)

    _driver_stats["driver_queries"] += 1
    _driver_stats["driver_total_ms"] += elapsed_ms

    return {
        "results":    results,
        "elapsed_ms": round(elapsed_ms, 3),
        "source":     "local-index",
        "index_size": _local_index.size,
    }


@app.get("/driver/metrics")
async def driver_metrics():
    """Comparison metrics: baseline (network) vs driver (local index)."""
    bs_q   = _driver_stats["baseline_queries"]
    drv_q  = _driver_stats["driver_queries"]
    bs_avg = (_driver_stats["baseline_total_ms"] / bs_q) if bs_q else 0
    drv_avg= (_driver_stats["driver_total_ms"] / drv_q)  if drv_q else 0
    speedup= (bs_avg / drv_avg) if drv_avg > 0 else 0

    return {
        "baseline": {
            "queries":        bs_q,
            "avg_latency_ms": round(bs_avg, 2),
            "source":         "db-node-network",
        },
        "driver": {
            "queries":        drv_q,
            "avg_latency_ms": round(drv_avg, 3),
            "source":         "local-index",
        },
        "speedup_factor":    round(speedup, 1),
        "index_size":        _local_index.size,
        "warmup_rows":       _driver_stats["warmup_rows"],
        "warmup_duration_ms": round(_driver_stats["warmup_duration_ms"], 1),
        "wrapper_url":       WRAPPER_URL or "not-configured",
    }


@app.post("/driver/reset")
async def driver_reset():
    """Reset the local index and all driver stats (start a fresh test run)."""
    evicted = _local_index.reset()
    for k in _driver_stats:
        _driver_stats[k] = 0 if isinstance(_driver_stats[k], int) else 0.0
    return {"evicted": evicted}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(
        "benchmark.app.main:app",
        host=cfg.host,
        port=cfg.port,
        workers=1,
        log_level="info",
    )
