---
kind: business_term
name: Business Glossary
category: business_term
scope:
    - '**'
---

### cost_per_tonne_km
- Definition：The graded metric computed per route-week as total freight cost divided by the sum of quantity-tonnes times distance-km over all shipments in that week, rounded to two decimals. It is the primary number against which both baselines (own history and similar routes) are compared.
- Aliases：cpk

### vs_own_history
- Definition：A baseline comparison measuring the current route-week's cost_per_tonne_km against the simple mean of the trailing 8 prior weekly cpk values of the same route, strictly before the current week (no look-ahead). If fewer than 8 prior weeks exist, all available prior weeks are used without padding.

### vs_similar_routes
- Definition：A baseline comparison measuring the current route-week's cost_per_tonne_km against the simple mean of the same-week cpk of every other route sharing the same route_type (Short / Medium / Long), excluding the route itself (leave-one-out peer average).

### candidate anomaly
- Definition：A route-week that breaches the flag threshold (≥ 20% above either vs_own_history or vs_similar_routes) and enters the adjudication stage. Not every candidate is ultimately flagged — some are marked justified after note matching.

### justified
- Definition：An adjudicated verdict meaning a retrieved context note passes all four rules (scope, window, direction, magnitude) for that route-week, so the cost rise is explained and the row is NOT flagged. In the sample data this applies to N002 festival surcharge and N001 flood within its date window.

### unexplained
- Definition：An adjudicated verdict meaning no retrieved note passes the four justification rules for that route-week; the row remains flagged as an unexplained cost spike. This is the default outcome when retrieval returns nothing or every note fails scope/window/direction/magnitude checks.

### adjudicator
- Definition：The deterministic rule engine (`src/adjudicator.py`) that decides whether a retrieved context note justifies a cost rise. It enforces four strict rules — scope, window, direction, magnitude — and is deliberately separate from the LLM to prevent hallucinated reasons.

### RAG
- Definition：Retrieval-Augmented Generation used here as semantic search over `context_notes.csv`: each note is embedded with the local `all-MiniLM-L6-v2` model, and the flagged route-week becomes a natural-language query returning cosine top-5 notes before they reach the adjudicator.

### pinned facts
- Definition：The fixed set of facts (route, week, cpk, baselines, matched note id/text) passed to the LLM at temperature 0 so it can word the explanation without inventing numbers or verdicts. The LLM has authority only over prose, never over graded fields.

### eval harness
- Definition：The evaluation script `eval/run_eval.py` that scores the pipeline against `eval/labeled_set.csv` (20 expected candidates with verdicts plus 7 negative controls), producing `eval/eval_report.md` with 36 checks covering sample replay, hallucination audit, retrieval recall, and output format.

### reproducibility check
- Definition：The script `scripts/run_reproducibility.py` that runs the full pipeline three times as fresh processes and compares outputs: deterministic mode requires byte-identical CSVs (same sha256); LLM mode requires identical graded fields while allowing prose wording to vary.

### token_cost_log.md
- Definition：The artifact produced by `src/llm_client.py` recording every LLM call of a full run: successful/failed call count, total input/output tokens, and estimated cost at the provider's published list price. Zero-cost for deterministic runs.
