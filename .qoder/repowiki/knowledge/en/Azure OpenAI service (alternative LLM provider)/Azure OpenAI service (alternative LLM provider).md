---
kind: external_dependency
name: Azure OpenAI service (alternative LLM provider)
slug: azure-openai
category: external_dependency
category_hints:
    - vendor_identity
    - auth_protocol
scope:
    - '**'
---

### Identity
Microsoft Azure OpenAI service, offered as an alternative to the generic OpenAI-compatible path.

### Role here
Optional LLM backend; when configured the pipeline produces the same graded outputs but LLM-worded explanations instead of template prose.

### Integration point
`.env.example` Option B: `LLM_API_TYPE=azure`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION=2024-10-21`. `src/config.py` loads these env vars and `src/llm_client.py` dispatches to the Azure endpoint.

### Stable usage shape
Same client contract as the OpenAI-compatible path (`LLM_API_TYPE=openai`); only the credential set and base URL differ. Verdicts and numbers remain deterministic regardless of which provider is selected.