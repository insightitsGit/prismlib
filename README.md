# PrismLib

**Tensor-native semantic cache and distributed data plane.**

Two products, one mathematical core:

| Product | What it does | Install |
|---------|-------------|---------|
| **PrismCache** | Semantic LLM cache — serve repeated/paraphrased queries from memory instead of paying the LLM again | `pip install prismlib[cache]` |
| **PrismDriver** | Tensor-native DB driver — replaces SQL connection strings; queries hit an in-process vector cache seeded by WAL streaming | `pip install prismlib[fabric]` |

Both use the same core: Johnson-Lindenstrauss tenant isolation + wave-interference similarity + HMAC-signed CHORUS Fabric transport.

---

## Installation

```bash
# Semantic LLM cache only
pip install "prismlib[cache]"

# With OpenAI embeddings
pip install "prismlib[cache,cache-openai]"

# With Anthropic/Voyage embeddings
pip install "prismlib[cache,cache-anthropic]"

# With Ollama (local models)
pip install "prismlib[cache,cache-ollama]"

# DB driver (app node)
pip install "prismlib[fabric]"

# Server Wrapper daemon (DB node — Linux/macOS)
pip install "prismlib[wrapper]"
prism-wrapper --config /etc/prism/wrapper.toml

# Everything
pip install "prismlib[all]"
```

---

## Use Cases

### PrismCache

#### Drop-in LLM response cache

Save 60-80% of LLM API calls by serving semantically identical queries from cache.
Paraphrases hit the cache — "How do I reset my password?" and "I forgot my password, help" return the same answer without a second LLM call.

```python
from prism.cache import PrismCache

cache = PrismCache.build(tenant_id="my-app", llm_model="gpt-4o")

def ask(question: str) -> str:
    return cache.get_or_call(
        query=question,
        call_fn=lambda: openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
        ).choices[0].message.content,
    )
```

#### Multi-tenant SaaS — isolated caches per customer

Each tenant gets a mathematically isolated cache space (JL projection seeded by tenant ID).
One customer's cached answers never bleed into another's.

```python
from prism.cache import PrismCache

def get_cache(tenant_id: str) -> PrismCache:
    return PrismCache.build(tenant_id=tenant_id, llm_model="gpt-4o-mini")

# Tenant A and tenant B share no cache state
cache_a = get_cache("acme-corp")
cache_b = get_cache("globex-inc")

answer = cache_a.get_or_call(query="What is my plan limit?", call_fn=llm_call)
```

#### FastAPI / Django middleware — transparent caching

Wrap your existing LLM endpoint without changing any business logic.

```python
# FastAPI
from fastapi import FastAPI, Request
from prism.cache import PrismCache

app = FastAPI()
cache = PrismCache.build(tenant_id="api", llm_model="gpt-4o")

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    question = body["message"]
    answer = await cache.aget_or_call(
        query=question,
        call_fn=lambda: llm_client.ask(question),
    )
    return {"answer": answer}
```

```python
# Django — add to MIDDLEWARE in settings.py
# prism/middleware.py
from prism.cache import PrismCache

_cache = PrismCache.build(tenant_id="django-app", llm_model="gpt-4o")

class PrismCacheMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        return self.get_response(request)

    def process_llm_query(self, question: str, call_fn) -> str:
        return _cache.get_or_call(query=question, call_fn=call_fn)
```

#### Async batch queries

```python
import asyncio
from prism.cache import PrismCache

cache = PrismCache.build(tenant_id="batch", llm_model="gpt-4o-mini")

async def process_batch(questions: list[str]) -> list[str]:
    tasks = [
        cache.aget_or_call(query=q, call_fn=lambda q=q: llm_call(q))
        for q in questions
    ]
    return await asyncio.gather(*tasks)
```

#### Cost estimation

```python
from prism.cache import PrismCache

cache = PrismCache.build(tenant_id="finance", llm_model="gpt-4o")

# After processing queries...
metrics = cache.metrics()
print(f"Hit rate:          {metrics.hit_rate:.0%}")
print(f"Tokens saved:      {metrics.tokens_saved:,}")
print(f"Cost saved today:  ${metrics.cost_saved_usd:.2f}")
print(f"Projected monthly: ${metrics.cost_saved_usd * 30:.0f}")
```

---

### PrismDriver

#### Replace your DB connection string

No passwords in app config. The Server Wrapper on the DB node handles auth; the driver speaks CHORUS Fabric over gRPC.

```python
# Before
import psycopg2
conn = psycopg2.connect("postgresql://user:secret@db-host:5432/mydb")

# After — no password, no hostname in app config
from prism.ffi import PrismDriver, DriverConfig

async with PrismDriver(DriverConfig(wrapper_host="db-proxy-1")) as driver:
    results = await driver.query(
        embedding=my_embedding_vector,
        top_k=5,
        threshold=0.85,
    )
```

#### Sub-millisecond row lookups via local cache

The driver keeps a local PrismResonance cache warm via a background WAL subscription.
Reads never touch the DB — they hit the in-process float32 index.

```python
from prism.ffi import PrismDriver, DriverConfig
import numpy as np

config = DriverConfig(
    wrapper_host="10.0.1.50",
    wrapper_port=50051,
    tenant_id="products-service",
)

async with PrismDriver(config) as driver:
    # Typical hit: < 1ms, no network round-trip
    query_vec = np.array([...], dtype=np.float32)
    matches = await driver.query(embedding=query_vec, top_k=10)
    for m in matches:
        print(f"{m.row_id}  score={m.score:.3f}  {m.text_repr}")
```

#### Write through to DB

```python
async with PrismDriver(config) as driver:
    ack = await driver.write(
        row_id="product-42",
        data={"name": "Widget Pro", "price": 29.99, "stock": 150},
    )
    print(f"Written: event_id={ack.event_id}")
```

#### Go, C#, PHP, Java — same DLL, native bindings

```go
// Go
import prism "github.com/insightitsGit/prismlib/go"

driver, _ := prism.Connect("db-proxy-1:50051", "my-tenant")
defer driver.Close()
results, _ := driver.Query(embedding, prism.QueryOpts{TopK: 5, Threshold: 0.85})
```

```csharp
// C#
using InsightIts.Prism;

await using var driver = new PrismDriver("db-proxy-1:50051", tenantId: "my-tenant");
await driver.ConnectAsync();
var results = await driver.QueryAsync(embedding, topK: 5, threshold: 0.85f);
```

```php
// PHP 8.0+
$driver = new PrismDriver('db-proxy-1', 50051, 'my-tenant');
$driver->connect();
$results = $driver->query($embedding, topK: 5, threshold: 0.85);
```

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
python benchmark/load/run_benchmark.py \
  --host https://prism-benchmark.nicestone-720c6a9b.westus2.azurecontainerapps.io \
  --scenario mixed --duration 300
```

**Typical results:** 60-80% hit rate · 40-50× latency reduction on hits · $500-$2,000/month saved per 1M queries.

---

## License

Apache 2.0 — InsightIts © 2026
