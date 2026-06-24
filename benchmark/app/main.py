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
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from prism.cache import PrismCache, HashEmbedder
from benchmark.app.config import get_config
from benchmark.app.telemetry import setup_telemetry, trace_request

logger = logging.getLogger(__name__)

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
