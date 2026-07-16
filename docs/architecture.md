# Architecture — PrismLib

PrismLib is an **in-process** data-plane beside your app (not a hosted SaaS requirement).

```
Your app / agents
      │
      ├── PrismCache ──────────► LLM API (skip on hit)
      │
      ├── PrismDriver ──CHORUS──► DB node WAL stream → local index
      │
      └── PrismLib Micro ──────► peer containers (shared answers / HA)
```

Details and install matrices: ../README.md · AI summary: [ai-overview.md](ai-overview.md)
