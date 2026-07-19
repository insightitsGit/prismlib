"""
Tests for PrismCache 0.5.0 selective invalidation, tags, hit metadata, and on_hit.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pytest

from prism.cache import (
    HitMeta,
    PrismCache,
    PrismCacheConfig,
    HashEmbedder,
    InMemoryStore,
    SQLiteStore,
)
from prism.cache.store import CacheEntry
from prism.lib.lang import PrismProjector, ProjectionConfig


@pytest.fixture()
def embedder() -> HashEmbedder:
    return HashEmbedder(output_dim=384)


def _make_cache(
    embedder: HashEmbedder,
    store: InMemoryStore | SQLiteStore | None = None,
    *,
    on_hit=None,
    tenant_id: str = "inv-tenant",
) -> PrismCache:
    cfg = PrismCacheConfig(
        tenant_id=tenant_id,
        similarity_threshold=0.99,
        ttl_seconds=3600,
        llm_model="gpt-4o-mini",
    )
    return PrismCache(cfg, embedder, store or InMemoryStore(), on_hit=on_hit)


def _project(tenant_id: str, embedder: HashEmbedder, text: str) -> np.ndarray:
    projector = PrismProjector(ProjectionConfig(tenant_id=tenant_id, target_dim=64))
    return projector.project(embedder.embed(text)).vector


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


# ---------------------------------------------------------------------------
# Vector eviction
# ---------------------------------------------------------------------------


class TestInvalidateWhere:
    def test_evicts_at_threshold_boundary(self, embedder: HashEmbedder) -> None:
        cache = _make_cache(embedder, tenant_id="bound-tenant")
        cache.get_or_call("Question about Alice", lambda: "Alice answer")

        # Retrieve the stored projected vector via store scan
        entries = list(cache._store.iter_entries())
        assert len(entries) == 1
        stored = _unit(entries[0].query_vector)

        # Exact match → cosine 1.0 → must evict at threshold 1.0
        n = cache.invalidate_where(stored, threshold=1.0)
        assert n == 1
        assert cache.cache_size == 0
        assert cache._store.count() == 0

    def test_miss_below_threshold(self, embedder: HashEmbedder) -> None:
        cache = _make_cache(embedder, tenant_id="miss-tenant")
        cache.get_or_call("Question about Alice", lambda: "Alice answer")
        entries = list(cache._store.iter_entries())
        stored = _unit(entries[0].query_vector)

        # Orthogonal-ish probe: flip sign of half the dims to drop cosine
        probe = stored.copy()
        probe[:32] = -probe[:32]
        probe = _unit(probe)
        sim = float(np.dot(stored, probe))
        assert sim < 0.5

        n = cache.invalidate_where(probe, threshold=0.99)
        assert n == 0
        assert cache.cache_size == 1

    def test_sqlite_store_vector_eviction(self, embedder: HashEmbedder) -> None:
        store = SQLiteStore(":memory:")
        cache = _make_cache(embedder, store, tenant_id="sql-vec")
        cache.get_or_call("persist me", lambda: "saved", tags=["t1"])

        entries = list(store.iter_entries())
        assert entries[0].query_vector is not None
        stored = _unit(entries[0].query_vector)

        # New cache instance — empty resonance, vectors only in SQLite
        cache2 = _make_cache(embedder, store, tenant_id="sql-vec")
        assert cache2.cache_size == 0
        n = cache2.invalidate_where(stored, threshold=0.99)
        assert n == 1
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Tag eviction
# ---------------------------------------------------------------------------


class TestInvalidateTags:
    def test_evicts_any_matching_tag(self, embedder: HashEmbedder) -> None:
        cache = _make_cache(embedder)
        cache.get_or_call("q1", lambda: "a1", tags=["person_a", "family"])
        cache.get_or_call("q2", lambda: "a2", tags=["person_b"])
        cache.get_or_call("q3", lambda: "a3", tags=["unrelated"])

        n = cache.invalidate_tags(["person_a", "person_b"])
        assert n == 2
        assert cache.cache_size == 1
        remaining = list(cache._store.iter_entries())
        assert remaining[0].tags == ["unrelated"]

    def test_empty_tags_noop(self, embedder: HashEmbedder) -> None:
        cache = _make_cache(embedder)
        cache.get_or_call("q", lambda: "a", tags=["x"])
        assert cache.invalidate_tags([]) == 0
        assert cache.cache_size == 1


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


class TestPersistenceRoundTrip:
    def test_sqlite_migrates_pre_0_5_schema(self, tmp_path) -> None:
        """Opening a 0.4.0 SQLite file adds tags_json / query_vector columns."""
        import json
        import sqlite3

        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE cache_entries (
                packet_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                tokens_saved INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL DEFAULT ''
            )
            """
        )
        now = time.time()
        conn.execute(
            "INSERT INTO cache_entries VALUES (?,?,?,?,?,?,?,?)",
            ("old1", "q", json.dumps("legacy"), now, now + 3600, 0, 0, "gpt"),
        )
        conn.commit()
        conn.close()

        store = SQLiteStore(str(db_path))
        loaded = store.load("old1")
        assert loaded is not None
        assert loaded.response == "legacy"
        assert loaded.tags == []
        assert loaded.query_vector is None

        vec = np.ones(64, dtype=np.float32)
        store.save(
            CacheEntry(
                "new1",
                "q2",
                "fresh",
                now,
                now + 3600,
                tags=["a"],
                query_vector=vec,
            )
        )
        fresh = store.load("new1")
        assert fresh is not None
        assert fresh.tags == ["a"]
        assert fresh.query_vector is not None
        store.close()

    def test_tags_and_created_at_survive_sqlite(self, embedder: HashEmbedder) -> None:
        store = SQLiteStore(":memory:")
        cache = _make_cache(embedder, store, tenant_id="persist-tags")
        before = time.time()
        cache.get_or_call(
            "Who is Person A?",
            lambda: "brother",
            tags=["person_a", "family"],
        )
        after = time.time()

        entries = list(store.iter_entries())
        assert len(entries) == 1
        e = entries[0]
        assert e.tags == ["person_a", "family"]
        assert before <= e.created_at <= after
        assert e.query_vector is not None
        assert e.query_vector.shape == (64,)
        assert e.model == "gpt-4o-mini"

        # Reload via a fresh SQLiteStore-shaped load
        loaded = store.load(e.packet_id)
        assert loaded is not None
        assert loaded.tags == ["person_a", "family"]
        assert loaded.created_at == e.created_at
        assert loaded.query_vector is not None
        np.testing.assert_allclose(loaded.query_vector, e.query_vector, rtol=1e-5)


# ---------------------------------------------------------------------------
# Hit metadata + on_hit
# ---------------------------------------------------------------------------


class TestHitMeta:
    def test_last_hit_meta_on_hit(self, embedder: HashEmbedder) -> None:
        cache = _make_cache(embedder)
        cache.get_or_call("meta query", lambda: "ans", tags=["t"])
        assert cache.last_hit_meta is None  # miss

        cache.get_or_call("meta query", lambda: "should-not-run")
        meta = cache.last_hit_meta
        assert meta is not None
        assert isinstance(meta, HitMeta)
        assert meta.tags == ["t"]
        assert meta.llm_model == "gpt-4o-mini"
        assert meta.similarity >= 0.99
        assert meta.created_at > 0
        assert meta.packet_id

    def test_on_hit_callback_fires(self, embedder: HashEmbedder) -> None:
        seen: list[HitMeta] = []
        cache = _make_cache(embedder, on_hit=seen.append)
        cache.get_or_call("cb query", lambda: "ans", tags=["x"])
        assert seen == []

        cache.get_or_call("cb query", lambda: "nope")
        assert len(seen) == 1
        assert seen[0].tags == ["x"]
        assert seen[0].similarity >= 0.99

    def test_on_hit_exception_does_not_break_hit(self, embedder: HashEmbedder) -> None:
        def boom(_: HitMeta) -> None:
            raise RuntimeError("callback failed")

        cache = _make_cache(embedder, on_hit=boom)
        cache.get_or_call("safe query", lambda: "ans")
        result = cache.get_or_call("safe query", lambda: "miss")
        assert result == "ans"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrentEviction:
    def test_eviction_during_concurrent_get_or_call(
        self, embedder: HashEmbedder
    ) -> None:
        cache = _make_cache(embedder, tenant_id="conc-tenant")
        # Seed several entries
        for i in range(20):
            cache.get_or_call(f"seed question {i}", lambda i=i: f"ans-{i}", tags=[f"t{i % 3}"])

        errors: list[BaseException] = []
        stop = threading.Event()

        def reader() -> None:
            try:
                while not stop.is_set():
                    cache.get_or_call(
                        "seed question 0",
                        lambda: "ans-0",
                        tags=["t0"],
                    )
            except BaseException as exc:
                errors.append(exc)

        def evictor() -> None:
            try:
                for _ in range(30):
                    cache.invalidate_tags(["t1"])
                    probe = _project("conc-tenant", embedder, "seed question 5")
                    cache.invalidate_where(probe, threshold=0.5)
                    time.sleep(0.001)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=evictor))
        for t in threads:
            t.start()
        time.sleep(0.15)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)

        assert errors == []
        # Cache remains usable
        result = cache.get_or_call("post-evict query", lambda: "ok")
        assert result == "ok"

    def test_parallel_get_or_call_no_exceptions(self, embedder: HashEmbedder) -> None:
        cache = _make_cache(embedder, tenant_id="pool-tenant")
        cache.get_or_call("shared", lambda: "cached", tags=["s"])

        def work(_: int) -> Any:
            return cache.get_or_call("shared", lambda: "miss")

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(work, i) for i in range(40)]
            results = [f.result() for f in as_completed(futures)]
        assert all(r == "cached" for r in results)
