# prism.api — Vector-Native API Layer for AI Agents

PrismAPI lets an API owner serve content that is **already embedded and projected** into PrismResonance space, delivered over CHORUS as raw `float32` vectors.

The consuming agent retrieves results directly — **no JSON parsing, no re-embedding call**.

---

## The Problem

Every conventional agent retrieval cycle pays an embedding tax it doesn't need to:

```
agent → HTTP GET /search?q=... → JSON {"text": "..."} → embed(text) → float32 vector → use
```

If the provider already embedded that text when it indexed it, the consumer is **re-embedding content that is already a vector**. At 1,000 queries/second with 10 results each, that is **10,000 unnecessary embedding API calls per second**.

PrismAPI eliminates the consumer's embedding calls for result content:

```
agent → CHORUS API_REQUEST (query vector) → API_RESPONSE (float32 vectors) → use directly
```

The consumer still embeds its own **query** once. What it never does is embed the **results** it gets back.

---

## Semantic vs. Exact Boundary

This is the critical design constraint. **Never vectorize fields that carry exact meaning.**

| Field type | Examples | Handling |
|-----------|----------|----------|
| **Semantic** (vectorized) | `title`, `body`, `description`, `summary` | Concatenated, embedded, projected → `SemanticItem.vector` |
| **Exact** (sidecar) | `price`, `id`, `url`, `in_stock`, `category`, `date` | Passed as-is in `ExactSidecar.fields` — **never embedded** |

Embedding a price or a boolean is meaningless — the vector space has no notion of `$49.99` vs `$50.00`. Exact fields ride as a JSON sidecar alongside the vector payload.

---

## Quick Start

### Provider (one decorator line)

```python
from prism.api import PrismAPIProvider, SentenceTransformerEmbedder
from prism.lib.lang import PrismProjector, ProjectionConfig

projector = PrismProjector(ProjectionConfig(tenant_id="my-tenant"))
embedder  = SentenceTransformerEmbedder()   # all-MiniLM-L6-v2, 384-dim

provider = PrismAPIProvider(
    projector=projector,
    embedder=embedder,
    semantic_fields=["title", "body"],          # vectorized
    id_field="doc_id",
    exact_fields=["price", "url", "in_stock"],  # NEVER vectorized
)

@provider.expose                                # ← the one line
def search(query: str, top_k: int = 10) -> list[dict]:
    return db.search(query, top_k)              # your handler, unchanged
```

The original handler continues to serve its existing HTTP/REST clients unchanged. The `@provider.expose` decorator adds a parallel CHORUS channel.

### Consumer (agent side)

```python
from prism.api import PrismAPIClient

client = PrismAPIClient(
    projector=projector,
    embedder=embedder,
    loopback_provider=provider,   # or host="api.example.com", port=9100
    source_field="body",
)

response = client.query("how does inflation affect bond prices?", top_k=5)

# response.vectors  → np.ndarray (5, 64) — pre-projected, ready for retrieval
# response.sidecars → list[dict]         — exact fields: price, url, in_stock
# response.embedding_calls_saved → 5     — calls the consumer did NOT make
```

### LangGraph node

```python
from prism.api.consumer import LangGraphTool

tool = LangGraphTool(
    name="semantic_search",
    description="Search the knowledge base by meaning.",
    client=client,
)

# As a tool call:
result = tool.invoke({"query": "inflation and bond yields"})
# result["vectors"]  → np.ndarray (N, 64)
# result["sidecars"] → list[dict] with exact metadata

# As a LangGraph node:
node = tool.as_langgraph_node()
# graph.add_node("search", node)
```

### FastAPI + ASGI adapter

```python
from prism.api.provider import ASGIAdapter
from fastapi import FastAPI

app = FastAPI()

# Existing HTTP routes untouched.
# Adds POST /chorus/search accepting application/x-chorus-frame.
adapter = ASGIAdapter(handler=search, handler_name="search")
adapter.mount(app)
```

### MCP server

```python
from prism.api.mcp import PrismAPIMCPServer

server = PrismAPIMCPServer(
    provider=provider,
    handler=search,
    tool_name="semantic_search",
    tool_description="Search the knowledge base by semantic meaning.",
)
server.run()   # blocks, serves JSON-RPC 2.0 over stdio
```

The MCP tool returns both a **JSON summary** (for standard MCP clients / LLMs) and a `chorus_frame_b64` field (base64 CHORUSFrame) for CHORUS-native consumers. Standard clients ignore the CHORUS field; CHORUS-native clients get pre-projected vectors without a second call.

---

## Install

```bash
pip install sentence-transformers numpy   # for real embeddings
```

All other dependencies (`prism.lib.fabric`, `prism.lib.lang`, `prism.lib.resonance`) are within the PrismLib package. FastAPI is optional (ASGI adapter only). The `mcp` SDK is optional (falls back to built-in stdio transport).

---

## Benchmark Results

61-document corpus, 15 queries, top-5, all-MiniLM-L6-v2 (384-dim → JL projection).

### Embedding calls — structural, not tunable

| | Baseline (HTTP/REST) | PrismAPI (CHORUS) |
|-|---------------------|-------------------|
| Embedding calls — 15 queries | 90 | **15** |
| Per query | 6.0 (1 query + 5 results) | **1.0 (query only)** |
| **Saved** | — | **83.3%** |

This saving is structural: baseline pays `1 + top_k` embeds per query; PrismAPI always pays 1.  
At top_k=10 and 1,000 queries/second: **10,000 embedding API calls/second eliminated**.

### Retrieval quality vs. target_dim

| target_dim | Payload vs JSON | Jaccard@5 | Recall@5 | **Recall@10** |
|-----------|----------------|-----------|----------|--------------|
| **64** | **−38%** (1,726 B) | 0.42 | 0.57 | **0.77** |
| **128** | +7% (3,006 B) | 0.53 | 0.67 | **0.91** |

**Recall@10** ("Recall at 2K") is the production metric: fetch `top_k × 2` from PrismAPI, re-rank with PrismResonance. At 128-dim, **91% of the relevant results are found** in the over-fetched set — at 83% fewer embedding calls.

> Run `python benchmark/api/run_prismapi_benchmark.py --target-dim 64|128` to reproduce.  
> Raw JSON in `benchmark/api/results/`.

### Choosing target_dim

- **64-dim**: 38% smaller payloads, 77% Recall@10. Best when bandwidth is the constraint.
- **128-dim**: 91% Recall@10 at near-parity payload. Best when retrieval quality is the priority.

### On loopback latency

In these benchmarks, PrismAPI is slower per-query (~95 ms vs ~25 ms baseline) because loopback frame serialization is measured directly. **In production, this reverses:** each embedding API call (OpenAI, Gemini, Vertex) adds 20–100 ms and $0.00002. The baseline's `top_k` extra embedding calls per query dominate end-to-end latency at any meaningful scale.

---

## Wire Protocol

Frames use the CHORUS binary format defined in `prism.lib.fabric`:

```
header: [key_id:36][seq:8][watermark:32][frame_type:1][payload_len:4]
payload (API_REQUEST):  [dim:4][vec:dim*4][ctx_len:4][ctx_json:ctx_len]
payload (API_RESPONSE): [n:4][dim:4] then per result: [vec:dim*4][side_len:4][side_json]
```

`Content-Type: application/x-chorus-frame` on HTTP transport.  
`FrameType.API_REQUEST = 0x08`, `FrameType.API_RESPONSE = 0x09`.

---

## MCP Relationship

MCP (Model Context Protocol) and CHORUS are **complementary, not competing**:

- **MCP** carries the tool call — JSON-RPC 2.0 over stdio/SSE, standard across all MCP hosts.
- **CHORUS** carries the vector payload — binary float32 frames, consumed by CHORUS-aware agents.

`PrismAPIMCPServer` returns both in every tool response. This means:
- Claude Desktop (standard MCP) → reads the JSON summary, ignores `chorus_frame_b64`.
- A CHORUS-aware agent → decodes the frame, gets pre-projected vectors without a second call.

Same tool call. Same endpoint. No protocol fork.

---

## Module Map

```
prism/api/
    __init__.py      Public exports
    schema.py        Embedder protocol, SemanticItem, ExactSidecar, APIRequest, APIResponse
    provider.py      PrismAPIProvider, @expose decorator, ASGIAdapter
    consumer.py      PrismAPIClient (loopback + HTTP), LangGraphTool
    mcp.py           PrismAPIMCPServer (stdio JSON-RPC + mcp-sdk integration)

benchmark/api/
    run_prismapi_benchmark.py   Full corpus benchmark, baseline vs PrismAPI

examples/
    prismapi_quickstart.py      End-to-end demo, no network required
```
