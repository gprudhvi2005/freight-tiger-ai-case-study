# Freight Cost Watchdog

A smart assistant that watches weekly shipping costs per route, flags route-weeks
where the cost is **rising in a way that doesn't look justified**, checks a small
set of real-world context notes (fuel, tolls, weather, festivals) for a genuine
reason, and explains every verdict in plain English.

Built for the FreightTiger 24-hour AI-intern case study.

---

## TL;DR — what it does

```
shipment_records.csv ──> weekly cost per tonne-km per route
                         │
        ┌────────────────┴────────────────┐
   vs own history                 vs similar routes
   (trailing 8 prior weeks,       (same-week average of other
    no look-ahead)                 routes of the same route_type)
        └────────────────┬────────────────┘
                         │  breach >= 20% on EITHER comparison
                  candidate anomaly
                         │
              semantic retrieval (RAG) over context_notes.csv
              (all-MiniLM-L6-v2 embeddings, cosine top-5)
                         │
              deterministic adjudication (4 strict rules)
                         │
            ┌────────────┴────────────┐
      note passes all rules     no note passes
      -> justified, cites it    -> unexplained, flagged
                         │
        reason prose: LLM at temperature 0 over PINNED facts,
        with a deterministic template fallback (never invents)
                         │
              output/anomaly_report.csv  (exact sample format)
```

On the provided dataset it finds **20 candidate anomalies: 17 flagged as
unexplained, 3 justified** (festival surcharge N002 on Ahmedabad-Mumbai, flood
N001 on Chennai-Bangalore for the two weeks inside the flood window).

---

## How to run

```bash
pip install -r requirements.txt

# full pipeline, deterministic (no API key needed) -> output/anomaly_report.csv
python -m src.main run --no-llm

# full pipeline with LLM-worded explanations (needs env config, see below)
python -m src.main run

# hard check: reproduce the 3 sample_output_format_v2.csv rows exactly
python -m src.main verify-sample

# show how each context note was parsed (scope / window / direction / magnitude)
python -m src.main notes

# stretch goal: ask the assistant about its findings
python -m src.main ask "why did Mumbai-Pune get pricier in late 2025?"

# trust checks
python -m pytest tests/ -q          # 17 unit tests
python eval/run_eval.py             # eval harness, 36 checks
python scripts/run_reproducibility.py [--llm]   # 3 untouched runs + diff
python scripts/make_charts.py       # output/walkthrough_chart.png
```

LLM configuration is optional — copy `.env.example` to `.env` and fill in
either an OpenAI-compatible endpoint or Azure OpenAI. Without it the pipeline
runs fully deterministically with template explanations; verdicts, numbers
and note citations are identical either way (the LLM only words the `reason`
prose from pinned facts).

---

## Methodology (the graded numbers, exactly)

Definitions follow the brief precisely and were **reverse-verified against all
three rows of `sample_output_format_v2.csv` — every number reproduces exactly**
(`python -m src.main verify-sample`, also asserted in `tests/test_core.py`):

* **Weeks** run Monday–Sunday; `week_of` is the Monday date.
* **cost_per_tonne_km** for a route-week =
  `sum(freight_cost_inr) / sum(quantity_tonnes × distance_km)`
  pooled over the week's shipments (tonne-km weighted), rounded to 2 decimals.
* **vs_own_history** = current cpk vs the **simple mean of the trailing 8
  weekly cpk values** of the same route, using only weeks strictly before the
  current week (no look-ahead). If fewer than 8 prior weeks exist, all prior
  weeks are used — no padding, no extrapolation. *(In this dataset every
  candidate row has a full 8-week baseline, so the caveat never fires; the
  behavior is covered by a unit test and the `partial_history` flag is
  recorded in `output/run_diagnostics.json`.)*
* **vs_similar_routes** = current cpk vs the **simple mean of the same-week
  cpk of every other route sharing the route_type** (Short / Medium / Long),
  excluding the route itself (leave-one-out).

**Flag threshold.** The brief says "out of the ordinary" without a number, so
I picked and documented one: a route-week is a candidate when its cpk is
**≥ 20% above either baseline**. Rationale: weekly cpk noise on these routes
has a standard deviation of roughly 3–8% of the mean, so 20% is far above
noise; and it cleanly separates every planted event in the data from the
background (the largest non-event deviation in the file is Mumbai-Delhi's
+9.5% single-week blip). The threshold lives in `src/config.py`.

**"Rising"** is interpreted as *elevated vs either baseline by the threshold*
— matching the sample output, where Mumbai-Pune 2025-09-15 is flagged at
+9.2% vs own history but +23.6% vs similar routes.

---

## Where the AI is, and where it deliberately is not

| Decision | Made by | Why |
|---|---|---|
| Numbers, baselines, flags | Deterministic pandas code | These are graded fields; must never vary |
| Which notes to consider | **Embedding retrieval (RAG)**: each note embedded with `all-MiniLM-L6-v2` (local, free); the flagged route-week becomes a natural-language query; cosine top-5 | The case-study's "use AI meaningfully" ask; retrieval quality is measured (see eval) |
| Whether a note justifies the rise | **Deterministic rule adjudicator** (4 rules, below) | The anti-hallucination core — a rule cannot invent a reason |
| `reason` prose | LLM at **temperature 0, fixed seed, pinned facts** — or a deterministic template if no LLM is configured/fails | Prose quality without giving the LLM any authority over verdicts |
| Q&A (stretch) | Retrieval over the assistant's own findings + the same LLM-at-0 (or deterministic composer) | Answers cite the assistant's own rows and the shipped notes only |

### The four adjudication rules

A retrieved note justifies a cost **rise** on route R in week W only if **all**
of these pass (`src/adjudicator.py`, tested in `tests/test_core.py`):

1. **Scope** — the note applies to R or to all routes. (N004's toll applies to
   routes "not part of this dataset" → applies to no route.)
2. **Window** — the note's effective time window overlaps week W. N001's flood
   covers Feb 24 – Mar 8, 2025, so it justifies the weeks of Feb 24 and Mar 3
   but **not** Mar 10 / Mar 17 (roads were passable again by Mar 9).
3. **Direction** — the note describes a cost **increase**. Notes saying costs
   were "not significantly affected" (N005), demand "remained stable" (N006),
   "no significant disruptions" (N008), "returned to normal" (N009), costs
   "absorbed without a rate change" (N010), or road conditions "improved"
   (N007 — that would explain a *decrease*) can never justify a rise.
4. **Magnitude** — if the note quantifies the increase it must be big enough.
   N003 (diesel, "roughly 5-7%", **all routes**) cannot explain a 20–39%
   route-specific divergence, and because it affects all routes equally it
   cannot explain any divergence *vs similar routes* either (peers are equally
   affected) — it can only ever account for a small uniform rise.

Notes are parsed into structured facts (scope, window, direction, magnitude)
by generic, deterministic text patterns — dates, percentages, direction
phrases. **No note id is hard-coded anywhere**; run `python -m src.main notes`
to see the parsed table, which is also embedded in
`output/run_diagnostics.json` for every run.

A recall backfill double-checks the full note set when top-5 retrieval finds
no justification — with this data it never changes a verdict, which is exactly
what the eval's retrieval-recall check verifies (top-5 retrieval found every
justifying note at rank 1–4, no backfill rescues).

---

## How I convinced myself it was right

1. **Sample replay (hard gate).** All three rows of
   `sample_output_format_v2.csv` reproduce exactly — cpk to 2 decimals, both
   percentages to 1 decimal, verdict and note id. Checked by
   `python -m src.main verify-sample`, a unit test, and the eval harness.
2. **Hand-labeled eval set** (`eval/labeled_set.csv`): 20 expected candidates
   with expected verdicts + 7 negative controls (trap weeks that must NOT be
   flagged: N005's "costs not affected" week, N008's "no disruptions" week,
   N007's road-improvement week, Mumbai-Delhi's +9.5% noise blip, quiet
   distractor route-weeks). The eval harness scores the pipeline against it.
3. **Independent hallucination audit.** For every justified row, a separate,
   simpler code path (`eval/run_eval.py::check_hallucination_audit`) re-reads
   the raw `context_notes.csv` and re-verifies scope, temporal overlap and
   direction of the cited note — deliberately not reusing the adjudicator's
   logic.
4. **Unit tests** (17): sample replay, no-lookahead property on a synthetic
   series, leave-one-out peer math, note parsing of all 10 notes, each
   adjudication rule (including the flood-window boundary and the
   diesel-magnitude trap), detector behavior, output-format contract.
5. **Retrieval recall check**: the justifying note must appear in the top-5
   retrieval results (it did — rank 1 for N001, rank 4 for N002).

**Current result: 36/36 eval checks pass, 17/17 unit tests pass.** See
`eval/eval_report.md` for the full listing.

---

## Reproducibility (three untouched runs)

`python scripts/run_reproducibility.py` runs the full pipeline three times as
fresh processes and compares outputs:

* **Deterministic mode:** the three CSVs are **byte-identical** (same sha256).
* **LLM mode** (`--llm`, if configured): the graded fields
  (route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes,
  flagged, matched_note_id) must be identical across runs — they are. The
  `reason` wording is *pinned* by temperature 0 + fixed seed + pinned context,
  but the hosted reasoning model is not bit-deterministic even at temperature
  0, so prose may vary slightly between runs. That is acceptable here because
  verdicts and numbers never depend on the LLM — it only words explanations.

Result on this machine: `output/reproducibility_check.txt` — PASS
(byte-identical deterministic sha256 `45250680...`, identical graded fields in
LLM mode).

Nothing else in the pipeline is random: the embedding model is a fixed
local model with deterministic inference, pandas arithmetic is deterministic,
and all iteration orders are explicitly sorted.

---

## Token / cost log (one full run)

`output/token_cost_log.md` records every LLM call of a full run: total input
tokens, output tokens, call count, failures, and estimated cost at the
model's published list price. The deterministic run makes **0 LLM calls**
(0 tokens, $0) — the LLM only ever words explanations/Q&A, one call per
flagged row. Embeddings run locally (all-MiniLM-L6-v2) and cost nothing.

Measured for one full LLM run over the whole dataset (20 explanation calls,
Groq `openai/gpt-oss-20b` at $0.075/M input + $0.30/M output):

| metric | value |
|---|---|
| successful / failed calls | 20 / 0 |
| total input tokens | 13,015 |
| total output tokens | 5,951 |
| estimated cost | **$0.0028** (input $0.0010 + output $0.0018) |

At ~$0.003 per full dataset run, the LLM layer costs roughly a third of a
cent per flagged row — cheap enough to re-run freely during development.

---

## Output

`output/anomaly_report.csv` matches `sample_output_format_v2.csv` exactly
(column names, order, formatting):

| column | example |
|---|---|
| route | `Chennai-Bangalore` |
| week_of | `2025-02-24` |
| cost_per_tonne_km | `3.58` |
| vs_own_history | `+31.5% vs this route's past average` |
| vs_similar_routes | `+38.7% vs similar-length routes this week` |
| flagged | `Yes` / `No (justified)` |
| matched_note_id | `N001` / *(blank when unexplained)* |
| reason | plain-English verdict, citing the note and its dates when justified |

`output/run_diagnostics.json` keeps the full evidence trail per row (retrieved
notes with similarities, every adjudication decision and rejection reason,
backfill usage, config snapshot) — this is what the eval, charts and Q&A read.

---

## Q&A (stretch goal)

`python -m src.main ask "..."` answers plain-English questions using only the
assistant's own findings (report + diagnostics + note texts). The question is
resolved to a focus — route mentioned, and/or a time window ("late 2025",
"March", "H1 2025"...) — and the matching findings become pinned context.
If nothing matches, semantic retrieval fills in; if no LLM is configured, a
deterministic answer is composed from the same rows, so the feature always
works. A grounding check rejects any LLM answer that doesn't lean on a pinned
route, week or note id, falling back to the deterministic answer.

Real transcript (LLM configured):

```
$ python -m src.main ask "was the Chennai-Bangalore flood expensive?"
Yes. In the week of 2025-02-24 the cost rose to 3.58 INR per tonne-km, a
31.5% increase over the route's past average and 38.7% above similar-length
routes (note N001). The same spike was seen in the week of 2025-03-03, again
attributed to heavy flooding (note N001). The flood made the route
significantly more expensive.

Sources:
 - Chennai-Bangalore week of 2025-02-24
 - Chennai-Bangalore week of 2025-03-03
 - Chennai-Bangalore week of 2025-03-10
 - Chennai-Bangalore week of 2025-03-17
```

---

## Project layout

```
freight-watchdog/
├── data/                      the three input files (untouched copies)
├── src/
│   ├── config.py              every tunable + env/.env handling
│   ├── data_loader.py         load & validate CSVs
│   ├── weekly_metrics.py      weekly aggregation + the two graded baselines
│   ├── detector.py            candidate selection (>= 20% on either side)
│   ├── notes_kb.py            embedding retrieval over context notes (RAG)
│   ├── adjudicator.py         note parsing + the 4 justification rules
│   ├── llm_client.py          OpenAI-compatible client + token accounting
│   ├── explainer.py           pinned-facts LLM prose + template fallback
│   ├── qa.py                  stretch: Q&A over the assistant's findings
│   ├── pipeline.py            orchestration -> anomaly_report.csv
│   └── main.py                CLI (run / verify-sample / notes / ask)
├── eval/
│   ├── labeled_set.csv        20 expected verdicts + 7 negative controls
│   └── run_eval.py            the eval harness (36 checks)
├── tests/test_core.py         17 unit tests
├── scripts/
│   ├── run_reproducibility.py 3 untouched runs + diff + hashes
│   └── make_charts.py         walkthrough chart per route
└── output/                    anomaly_report.csv, diagnostics, token log,
                               reproducibility check, chart, eval artifacts
```

---

## Findings on this dataset (for the walkthrough)

* **Delhi-Jaipur, Nov 2024** (+35.5% / +21.0%): two flagged weeks, no
  applicable note → unexplained. (N007 is a road *improvement* on this route
  — wrong direction — and dated six months earlier.)
* **Ahmedabad-Mumbai, week of 2025-01-20** (+29.5% / +22.5%): justified by
  N002, the regional-festival surcharge week.
* **Chennai-Bangalore, Feb–Mar 2025** (+13.7–31.5% / +32.0–38.7%): N001's
  flooding justifies the weeks of Feb 24 and Mar 3; the weeks of Mar 10 and
  Mar 17 fall **outside** the flood window and stay flagged. (N009 confirms
  the route was back to normal from Mar 17.)
* **Mumbai-Pune, Jun–Dec 2025** (up to +38.9% vs peers): a sustained,
  unexplained drift — 13 flagged weeks. The diesel note N003 is retrieved and
  honestly rejected: it covers ~5–7% and applies to all routes, so it cannot
  explain a route-specific divergence from peers.
* Quiet routes (Kolkata-Bhubaneswar, Delhi-Chennai, Mumbai-Delhi): no flags —
  the mild all-route rise after May 2025 matches N003's 5–7% and stays below
  the bar.

## Trade-offs & what I'd build next

* **Determinism over cleverness for verdicts.** The LLM could have judged
  note applicability, but the brief explicitly penalizes confident wrong
  reasons harder than "couldn't find one", so adjudication is rules-first and
  the LLM is confined to wording. A natural next step: an *advisory* LLM
  cross-check that can veto (never create) a justification, logged but not
  authoritative.
* **Note parsing is pattern-based** (dates, percentages, direction phrases).
  It generalizes to similarly-phrased notes and is fully inspectable, but
  free-form notes would need a stronger parser (or an LLM extraction step at
  temperature 0 with the same strict validation on top).
* **Threshold (20%) is a documented judgment call**, calibrated against the
  noise floor of this dataset. With more history I'd make it adaptive
  (e.g. percentile-based per route_type) and expose it as a CLI flag.
* **Retrieval corpus is 10 notes** — tiny. Top-5 + backfill is the right
  shape at this size; at thousands of notes the backfill would go and I'd
  add a reranker and measure recall@k properly.
