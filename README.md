# PrismLib

**Tensor-native semantic cache and distributed data plane.**

Two products, one mathematical core:

| Product | What it does | Install |
|---------|-------------|---------|
| **PrismCache** | Semantic LLM cache — serve repeated/paraphrased queries from memory instead of paying the LLM again | `pip install prismlib[cache]` |
| **PrismDriver** | Tensor-native DB driver — replaces SQL connection strings; queries hit an in-process vector cache seeded by WAL streaming | `pip install prismlib[fabric]` |

Both use the same core: Johnson-Lindenstrauss tenant isolation + wave-interference similarity + HMAC-signed CHORUS Fabric transport.

---

## PrismCache — 5-line integration

```python
from prism.cache import PrismCache

cache = PrismCache.build(tenant_id="my-app", llm_model="gpt-4o")

def ask(question: str) -> str:
    return cache.get_or_call(
        query=question,
        call_fn=lambda: openai_client.chat.completions.create(...).choices[0].message.content,
    )
```

**Typical results:** 60-80% cache hit rate, 40-50× latency reduction on hits, $500-$2000/month saved per 1M queries.

---

## PrismDriver — replace your connection string

```python
# Before
conn = psycopg2.connect("postgresql://user:secret@db-host/mydb")

# After — no password, no hostname
from prism.ffi import PrismDriver, DriverConfig
driver = PrismDriver(DriverConfig(wrapper_host="db-proxy-1"))
await driver.connect()
```

The Server Wrapper (`prism-wrapper`) runs as a daemon on the DB node, intercepts WAL changes, vectorizes rows, and streams them to the DLL Driver over CHORUS Fabric. The app queries a local in-process cache — sub-millisecond, no DB round-trip.

---

## Architecture

```
┌─ DB Node ──────────────────────────────────────────┐
│  PostgreSQL / MySQL / CockroachDB / TiDB           │
│       │ WAL / binlog / changefeed                  │
│  ┌────▼─────────────────────────┐                  │
│  │  prism-wrapper (OS daemon)   │                  │
│  │  vectorizes rows → CHORUS    │                  │
│  └────────────┬─────────────────┘                  │
└───────────────┼────────────────────────────────────┘
                │ gRPC float32 stream (port 50051)
┌─ App Node ────┼────────────────────────────────────┐
│  ┌────────────▼─────────────────────────────────┐  │
│  │  Your Application                            │  │
│  │  ┌─────────────┐  ┌──────────────────────┐   │  │
│  │  │ PrismCache  │  │ PrismDriver (DLL)    │   │  │
│  │  │ LLM cache   │  │ local PrismResonance │   │  │
│  │  └─────────────┘  └──────────────────────┘   │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

---

## Benchmark

See [`benchmark/`](benchmark/) — Azure-based load test that measures:
- Cache hit rate over time
- P50 / P95 / P99 latency (hit vs miss)
- Throughput ceiling (req/s)
- Cost savings per hour at scale

```bash
cd benchmark
./azure/deploy.sh          # provision Azure infra
python load/seed_data.py   # pre-load 50k Q&A pairs
python load/run_benchmark.py --scenario mixed --duration 300
```

---

## Installation

```bash
# Semantic LLM cache only
pip install prismlib[cache]

# With OpenAI embeddings
pip install "prismlib[cache,cache-openai]"

# With Anthropic/Voyage embeddings
pip install "prismlib[cache,cache-anthropic]"

# Server Wrapper (DB node — Linux/macOS)
pip install "prismlib[wrapper]"
prism-wrapper --config /etc/prism/wrapper.toml
```

---

## License

Proprietary — InsightIts © 2026
