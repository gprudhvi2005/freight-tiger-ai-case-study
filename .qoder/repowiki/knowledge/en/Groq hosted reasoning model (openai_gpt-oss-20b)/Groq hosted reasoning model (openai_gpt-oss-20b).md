---
kind: external_dependency
name: Groq hosted reasoning model (openai/gpt-oss-20b)
slug: groq
category: external_dependency
category_hints:
    - vendor_identity
    - auth_protocol
scope:
    - '**'
---

### Identity
Groq cloud-hosted reasoning model accessed through the OpenAI-compatible client; in this project the deployed model is `openai/gpt-oss-20b`.

### Role here
Provides the prose `reason` field and Q&A answers at temperature 0 with pinned facts; it never influences graded verdicts or numbers — those are fully deterministic.

### Integration point
Configured via `.env.example` / `.env` using `LLM_API_TYPE=openai`, `LLM_BASE_URL` pointing at Groq's OpenAI-compatible endpoint, plus `LLM_MODEL=gpt-oss-20b` and `LLM_API_KEY`. Token accounting is done by `src/llm_client.py` and logged to `output/token_cost_log.md`.

### Stable usage shape
- One call per flagged row (20 calls on the full dataset).
- Prose may vary between runs even at temperature 0 because the hosted model is not bit-deterministic; graded fields must remain identical across runs.