# PrismLib

**Tensor-native semantic cache and distributed data plane.**

Two products, one mathematical core:

| Product | Component | Deployed on | Install |
|---------|-----------|-------------|---------|
| **PrismCache** | In-process LLM cache | App node | `pip install "prismlib[cache]"` |
| **PrismDriver** | **Server Wrapper** (daemon) | **DB node** | `pip install "prismlib[wrapper]"` |
| **PrismDriver** | **DLL Driver** (in-process) | **App node** | `pip install "prismlib[fabric]"` |

PrismDriver is a two-node system: the **Server Wrapper** runs as an OS daemon on the same machine as your database, intercepts WAL/binlog changes, vectorizes rows, and streams them over CHORUS Fabric to the **DLL Driver** on your app server. The driver keeps a local PrismResonance index warm so reads never leave the process.

Built on two open-source InsightIts libraries:
- **[PrismResonance](https://github.com/insightitsGit/prismresonance)** — the wave-memory similarity engine powering every cache lookup and local vector index
- **[CHORUS Fabric](https://github.com/insightitsGit/chorus_fabric)** — the gRPC binary streaming protocol that carries encrypted float32 tensor frames from the Server Wrapper to the DLL Driver

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

PrismDriver has two components that work together. Install each on the right machine.

**On the DB node — Server Wrapper**

The Server Wrapper is an OS daemon that sits next to your database. It reads WAL/binlog changes, vectorizes rows using `RowVectorizer`, encrypts them with `TensorCipher` (via CHORUS Fabric), and streams float32 frames to every connected DLL Driver.

```bash
# Install on the DB node (Linux or macOS)
pip install "prismlib[wrapper]"

# Configure and start
prism-wrapper --config /etc/prism/wrapper.toml
```

```toml
# /etc/prism/wrapper.toml
[database]
flavor = "postgresql"
dsn = "postgresql://user:pass@localhost/mydb"

[chorus]
listen_port = 50051
tenant_id = "products-service"
```

Supported databases: PostgreSQL (WAL / wal2json), MySQL (binlog), CockroachDB (EXPERIMENTAL CHANGEFEED), TiDB (push model).

**On the app node — DLL Driver**

The DLL Driver is an in-process library that replaces your DB connection string. On startup it connects to the Server Wrapper, subscribes to the CHORUS Fabric stream, and keeps a local PrismResonance index warm. All reads hit the in-process index — no network round-trip, sub-millisecond latency.

```bash
# Install on the app node
pip install "prismlib[fabric]"
```

#### Replace your DB connection string

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
┌─ DB Node ──────────────────────────────────────────────────────┐
│  PostgreSQL / MySQL / CockroachDB / TiDB                       │
│       │ WAL / binlog / changefeed                              │
│  ┌────▼────────────────────────────────────────────────────┐   │
│  │  prism-wrapper  (pip install "prismlib[wrapper]")       │   │
│  │  RowVectorizer → TensorCipher (V_enc = V @ K)          │   │
│  │  → HMAC-SHA256 watermark → CHORUSPublisher             │   │
│  └────────────────────────┬────────────────────────────────┘   │
└───────────────────────────┼────────────────────────────────────┘
                            │  CHORUS Fabric  (gRPC, port 50051)
                            │  encrypted float32 frames
┌─ App Node ────────────────┼────────────────────────────────────┐
│  ┌────────────────────────▼───────────────────────────────┐    │
│  │  PrismDriver DLL  (pip install "prismlib[fabric]")     │    │
│  │  Subscribe loop → decrypt → PrismResonance index       │    │
│  └──────────────────────────────────────────┬─────────────┘    │
│                                             │ sub-ms query     │
│  ┌──────────────────────────────────────────▼─────────────┐    │
│  │  Your Application                                       │    │
│  │  ┌──────────────────┐   ┌───────────────────────────┐  │    │
│  │  │  PrismCache      │   │  PrismDriver              │  │    │
│  │  │  LLM cache       │   │  local PrismResonance     │  │    │
│  │  │  pip install     │   │  (no DB round-trip)       │  │    │
│  │  │  prismlib[cache] │   │                           │  │    │
│  │  └──────────────────┘   └───────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

---

## Benchmark

### PrismCache — semantic LLM cache

Live results from Azure Container App (`westus2`, 1 vCPU / 2 GiB, mock LLM baseline):

| Scenario | Users | Duration | Hit rate | Queries | Tokens saved | Monthly est. |
|----------|-------|----------|----------|---------|-------------|-------------|
| Light    | 20    | 60s      | **91.0%** | 5,936  | 1,374,464   | **$594**    |
| Mixed    | 50    | 300s     | **95.9%** | 6,973  | 1,673,216   | **$723**    |

> Numbers use a mock LLM (80ms sleep). With real GPT-4o calls (1–3s), latency speedup is 4–13×; token savings are identical.

### PrismDriver — two-node baseline vs local index

Live two-node benchmark (Azure Container Apps `westus2`, 30 users × 60s per phase):

| Phase | Path | Avg latency | Queries |
|-------|------|-------------|---------|
| **Baseline** (no driver) | App → DB node, network | **142.8 ms** | 3,864 |
| **Driver** (local index) | App → in-process PrismResonance | **2.0 ms** | 1,479 |

**70.7× faster · 98.6% latency reduction**

The 98.6% reduction is a direct result of CHORUS Fabric doing its job. The subscription loop streamed 11,000 rows at **26,000 rows/s** from the DB node into the local PrismResonance index before the load test began. By the time the first `/driver/query` hit arrived, there were zero network hops — the answer was already in-process. This is what CHORUS Fabric was designed for: getting tensor data to where the query is, before the query arrives.

```bash
# Two-node benchmark (requires both container apps running)
python benchmark/load/run_driver_benchmark.py \
  --app-url https://prism-benchmark.nicestone-720c6a9b.westus2.azurecontainerapps.io \
  --db-url  https://prism-wrapper-sim.nicestone-720c6a9b.westus2.azurecontainerapps.io \
  --users 30 --duration 60

# PrismCache load test
python benchmark/load/run_benchmark.py \
  --host https://prism-benchmark.nicestone-720c6a9b.westus2.azurecontainerapps.io \
  --scenario mixed
```

See [`benchmark/`](benchmark/) for full results JSON, Locust CSV files, and the Azure deploy script.

---

## Core libraries

PrismLib is built on two InsightIts open-source libraries. You can use them directly if you need lower-level access.

### PrismResonance

> **[github.com/insightitsGit/prismresonance](https://github.com/insightitsGit/prismresonance)** · `pip install prismresonance`

The wave-memory similarity engine. Every cache lookup and local vector index in PrismLib goes through PrismResonance.

How it works:
- Receives a float32 embedding vector
- Johnson-Lindenstrauss reduces it to 64 dimensions using a projection matrix seeded by `SHA-256(tenant_id)` — this is what gives each tenant mathematically isolated address space
- Computes similarity as wave interference (cosine in projected space) in three lock-free phases: snapshot → ONNX MatMul → rank
- Returns ranked candidates in sub-millisecond time entirely in-process

PrismCache wraps this for LLM response caching. PrismDriver's local replica is a PrismResonance index kept warm by WAL streaming.

```python
from prismresonance import PrismProjector, WaveIndex

projector = PrismProjector(dim=64, tenant_id="my-tenant")
index = WaveIndex(projector)

index.add(vector=my_embedding, payload={"row_id": "product-1", "text": "Widget"})
results = index.query(vector=query_embedding, top_k=5, threshold=0.85)
```

### CHORUS Fabric

> **[github.com/insightitsGit/chorus_fabric](https://github.com/insightitsGit/chorus_fabric)** · `pip install chorus-fabric`

The secure gRPC binary streaming protocol for machine-to-machine tensor communication. PrismDriver uses CHORUS Fabric as its transport layer between the server wrapper on the DB node and the DLL driver on the app node.

How it works:
- `prism-wrapper` (DB node) vectorizes WAL row events via `RowVectorizer`, encrypts them with `TensorCipher` (`V_enc = V @ K`), appends an HMAC-SHA256 watermark, and publishes batches of raw float32 frames
- `PrismDriver` (app node) opens a persistent `WrapperService.Subscribe()` gRPC stream, receives encrypted frames, decrypts, and feeds them into the local PrismResonance index
- Transport is pure binary float32 over gRPC server-streaming — no JSON serialization, no REST overhead
- The `WrapperService` proto also exposes `Query`, `Write`, `Health`, and `Hello` RPCs for direct interaction

```python
from chorus_fabric import CHORUSPublisher, DriverEndpoint

publisher = CHORUSPublisher(config)
publisher.add_driver(DriverEndpoint(host="10.0.1.50", port=50051, tenant_id="prod"))
await publisher.run(event_queue)  # streams WAL events to all connected drivers
```

CHORUS Fabric is the same protocol used in the CHORUS M2M system — InsightIts' 4-container gRPC topology for tensor communication between AI agents. The 98.6% latency reduction in the PrismDriver benchmark is direct proof that the protocol works at production scale: 11,000 rows streamed at 26,000 rows/s across Azure inter-container networking, then served locally at 2ms.

---

## Enterprise

PrismLib is open source (Apache 2.0) and free to use. If your team needs any of the following, contact us for enterprise pricing:

- **On-premises deployment support** — air-gapped installs, hardened Docker images, SOC 2 documentation
- **SLA-backed support** — guaranteed response times, incident escalation, dedicated Slack channel
- **Custom embedding model integration** — fine-tuned domain-specific embedders for higher hit rates in specialized domains (legal, medical, finance, code)
- **Multi-region CHORUS Fabric topology** — active-active DB node clusters, cross-region WAL fan-out, geo-aware driver routing
- **Audit logging and compliance exports** — per-query access logs, tenant isolation attestation reports, GDPR data lineage
- **Professional services** — architecture review, migration from Redis/GPTCache, custom RowVectorizer schemas

**Contact: [insightits.info@gmail.com](mailto:insightits.info@gmail.com)**
**GitHub: [github.com/insightitsGit/prismlib](https://github.com/insightitsGit/prismlib)**

---

## Publishing to PyPI

```bash
pip install build twine
python -m build
python -m twine upload dist/* --username __token__ --password pypi-YOUR_TOKEN
```

---

## License

Apache 2.0 — InsightIts © 2026
