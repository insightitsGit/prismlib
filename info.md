# PrismLib — Landing Page Brief

## Product overview

PrismLib is an open-source tensor-native library with two distinct products sharing one mathematical core:

- **PrismCache** — a semantic LLM response cache. Apps wrap any LLM call; repeated or paraphrased queries return a cached answer without touching the LLM again. Competes with GPTCache, Zep, Momento Semantic Cache, Redis Semantic Cache.
- **PrismDriver** — a tensor-native database driver. Replaces SQL connection strings in app config; an in-process vector cache (PrismResonance) is kept warm via WAL streaming from a server-side daemon. Sub-millisecond reads with no DB round-trip.

GitHub: https://github.com/insightitsGit/prismlib
License: Apache 2.0
Install: `pip install "prismlib[cache]"` or `pip install "prismlib[fabric]"`

---

## Positioning

Target buyers / users:
- Platform teams building multi-tenant SaaS on top of LLMs
- Backend engineers paying $500–$5,000/month in OpenAI API costs
- Data engineers replacing high-latency SQL reads with vector-native lookups
- DevOps / ML platform engineers who want a drop-in cache without a Redis cluster

Key differentiators vs competitors:
- **No external infra**: runs fully in-process, zero Redis/Pinecone/Qdrant required
- **Multi-tenant math**: Johnson-Lindenstrauss projection seeded by SHA-256(tenant_id) — cross-tenant isolation is a mathematical property, not a query filter
- **Wave-interference similarity**: PrismResonance's three-phase lock-free query is faster than cosine search on dense indexes
- **WAL-native**: Server Wrapper intercepts PostgreSQL WAL / MySQL binlog / CockroachDB changefeed / TiDB — no ORM changes needed
- **CHORUS Fabric transport**: HMAC-watermarked gRPC binary float32 stream with TensorCipher encryption — enterprise-grade security on the data plane

---

## Benchmark numbers (Azure, mock LLM baseline)

Run against live Azure Container App (`westus2`, 1 vCPU / 2 GiB):

| Scenario | Concurrent users | Duration | Cache hit rate | Queries served | Tokens saved | Monthly projection |
|----------|-----------------|----------|----------------|---------------|-------------|-------------------|
| Light    | 20              | 60s      | **91.0%**      | 5,936         | 1,374,464   | **$594/mo**       |
| Mixed    | 50              | 300s     | **95.9%**      | 6,973         | 1,673,216   | **$723/mo**       |

Note: these numbers use a mock LLM (80ms sleep). With a real GPT-4o call (1–3s), speedup factor on latency is 4–13×; token cost savings are identical.

---

## Core math (for technical readers)

1. **JL Projection**: query embedding → Johnson-Lindenstrauss reduction to 64-d using a random matrix seeded by SHA-256(tenant_id). Mathematically guarantees cross-tenant isolation without any filter clause.
2. **WavePacket similarity**: similarity computed as wave interference (cosine in projected space). Three-phase: snapshot under lock → ONNX MatMul lock-free → rank lock-free.
3. **CHORUS Fabric**: gRPC server-streaming of raw float32 arrays. Rows vectorized by `RowVectorizer` on DB node → `TensorCipher` encryption (V_enc = V @ K) → HMAC-SHA256 watermark → streamed to PrismDriver on app node.

---

## Landing page requirements

The landing page should be a single HTML file (no build tools, no Node.js) — plain HTML + Tailwind CDN + vanilla JS. It must be self-contained and deployable to GitHub Pages (`gh-pages` branch, `index.html`).

### Sections (in order)

1. **Hero** — headline, sub-headline, two CTA buttons (GitHub, pip install copy-to-clipboard)
2. **Problem / cost** — pain point: LLM API cost, DB latency; hook: "91% of your LLM calls are duplicates"
3. **Two products** — side-by-side cards: PrismCache (LLM cache) vs PrismDriver (DB driver)
4. **How it works** — data flow diagram or step illustration for each product (PrismCache: query → JL → wave lookup → cache hit/miss; PrismDriver: app → gRPC → PrismResonance → sub-ms result)
5. **Benchmark** — the numbers above in a visual table or stat cards; note the Azure live test context
6. **Code examples** — tabbed: Python / Go / C# / PHP; show the 5-line integration
7. **Use cases** — icon cards: SaaS multi-tenant, chatbot cost reduction, DB read acceleration, RAG pipeline caching
8. **Architecture** — ASCII-style or SVG: DB Node → prism-wrapper → CHORUS gRPC → PrismDriver → App
9. **Installation** — pip install variants with copy buttons
10. **Footer** — GitHub link, Apache 2.0 badge, InsightIts

### Design direction

- Clean, technical, dark-mode-first (like Vercel / Railway / Turso)
- Accent color: indigo/violet (#6366f1 or similar)
- Monospace font for code blocks (JetBrains Mono or similar via Google Fonts)
- No stock photos; use SVG illustrations or code snippets as hero visual
- Mobile-responsive

### Tone

Technical and confident. No buzzword soup. The copy should read like it was written by an engineer for engineers. Avoid "AI-powered", "next-gen", "game-changing". Prefer specific numbers and concrete claims.

### Key copy (suggested)

**Hero headline**: "Stop paying for the same LLM answer twice"
**Sub-headline**: "PrismCache intercepts repeated and paraphrased queries in-process — no Redis, no Pinecone, no infrastructure. 91% hit rate out of the box."
**PrismCache tagline**: "Semantic LLM cache · 5-line integration · zero infra"
**PrismDriver tagline**: "Tensor-native DB driver · WAL-streamed · sub-millisecond reads"

### Files to produce

- `index.html` — the landing page
- `assets/` — any local SVG illustrations if needed

The page should pass Lighthouse performance > 90 (no heavy JS frameworks).

---

## Competitive landscape (for copy reference)

| Competitor | What they do | PrismLib advantage |
|-----------|-------------|-------------------|
| GPTCache | LLM semantic cache | Requires Redis; no multi-tenant math |
| Zep | Memory layer for LLMs | Persistent graph; heavier infra |
| Momento Semantic Cache | Managed semantic cache | SaaS, not open-source; no WAL integration |
| Redis Semantic Cache | Redis + vector search | Requires Redis cluster; external network hop |
| Neon / PlanetScale | Serverless Postgres/MySQL | Still SQL; no vector-native driver |

---

## GitHub repo facts

- Repo: `insightitsGit/prismlib`
- Main branch: `master`
- CI: GitHub Actions, Python 3.11 + 3.12
- Live benchmark endpoint: `https://prism-benchmark.nicestone-720c6a9b.westus2.azurecontainerapps.io`
- Key files: `prism/cache/`, `prism/wrapper/`, `prism/ffi/`, `benchmark/`
