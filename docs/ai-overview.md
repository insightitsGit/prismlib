# AI / LLM context — PrismLib

> Concise reference for humans and coding assistants.  
> Do not invent APIs beyond this file and the package source. Package: **`prismlib` 0.4.0**, import **`prism`**.

---

## 10-sentence project summary

1. PrismLib is an Apache-2.0 in-process intelligence package for LLM apps (cache, DB driver, cluster mesh).  
2. Three layers: **PrismCache**, **PrismDriver**, **PrismLib Micro** — install only what you need.  
3. PrismCache wraps LLM calls and returns cached answers for repeated/paraphrased queries (README: 91–96% hit rate under load).  
4. Multi-tenant cache isolation uses JL projection seeded by `SHA-256(tenant_id)`.  
5. PrismDriver streams WAL/binlog via CHORUS Fabric into a local index so reads avoid network round-trips (README: 98.6% latency reduction example).  
6. PrismLib Micro shares answers across containers with health mesh and Blue/Green/Orange failover (README: 76% fewer tokens cluster-wide).  
7. Everything claimed for the OSS package runs **in-process** — no mandatory Redis/Pinecone/Prometheus/K8s operator.  
8. Complements vector DBs and agent runtimes; does not replace them as a category.  
9. Related paid/enterprise layers: ChorusMesh (alerts/transport), prismlib-plus (full stack API).  
10. Limitations: headline numbers are from published README/bench contexts — validate on your traffic; Fabric/driver needs matching topology.

---

## Core concepts

| Term | Definition |
|------|------------|
| **PrismCache** | In-process semantic LLM cache (`prism.cache.PrismCache`) |
| **PrismDriver** | WAL-streamed local index driver (`prism.ffi.PrismDriver`) |
| **ClusterCache** | Multi-node shared answer cache (`prism.cluster.cache`) |
| **CHORUS Fabric** | Tensor transport used by driver streaming |
| **Tenant JL projection** | Per-tenant address space for cache isolation |

---

## Key APIs

```python
from prism.cache import PrismCache

cache = PrismCache.build(tenant_id="my-app", llm_model="gpt-4o")
answer = cache.get_or_call(query=question, call_fn=lambda: ...openai_call...)

from prism.ffi import PrismDriver, DriverConfig  # fabric extra
from prism.cluster.cache import ClusterCache      # fabric / micro
```

Install:

```bash
pip install "prismlib[cache]"
pip install "prismlib[fabric]"   # driver + micro
```

---

## Common use cases

1. Cut repeat LLM spend on FAQ / support / internal copilots.  
2. Lower DB read latency for vectorized row access on the app node.  
3. Share cache hits across a container cluster with failover.

---

## Migration guidance

From **Redis semantic cache**: replace the remote cache round-trip with in-process `PrismCache.get_or_call` for eligible queries. From **always-hit DB reads**: evaluate PrismDriver when WAL streaming + local index fits your DB. From **DIY multi-pod caches**: use ClusterCache / Micro instead of ad-hoc sharing. Prefer **ChorusMesh** only when you need paid Slack/PagerDuty/Kafka/NATS orchestration on top.

---

## Limitations / when NOT to use

- You need a hosted managed cache SaaS only.  
- You cannot run anything in-process beside the app.  
- You need enterprise Slack/PagerDuty/Kafka mesh → see ChorusMesh (commercial).  
- Do not treat README load-test % as a guarantee for every workload.

---

## Frequently compared projects

| Project | Relationship | Prefer PrismLib when… | Prefer them when… |
|---------|--------------|----------------------|-------------------|
| Redis + custom semantic cache | Alternative | In-process, no Redis required | You already operate Redis at scale |
| Vector DB alone | Complementary | Caching LLM answers / local WAL index | You only need vector search |
| ChorusMesh | Paid extension | OSS cluster basics enough | Need Slack/PD/Kafka/NATS enterprise layer |
| prismlib-plus | Fuller stack | Base pip layers enough | Need Plus API / enterprise HTTP stack |

---

## Links

- [ai-overview.md](ai-overview.md) · [llm-context.md](llm-context.md) · [architecture.md](architecture.md)  
- ../README.md · https://github.com/insightitsGit/prismlib
