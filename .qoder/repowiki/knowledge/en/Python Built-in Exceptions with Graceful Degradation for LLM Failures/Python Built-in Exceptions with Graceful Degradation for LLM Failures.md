---
kind: error_handling
name: Python Built-in Exceptions with Graceful Degradation for LLM Failures
category: error_handling
scope:
    - '**'
source_files:
    - freight-watchdog/src/data_loader.py
    - freight-watchdog/src/llm_client.py
    - freight-watchdog/src/main.py
    - freight-watchdog/tests/test_core.py
---

## Approach

The codebase uses plain Python built-in exceptions (`ValueError`) for input validation and a **graceful degradation** pattern for external dependencies (OpenAI / Azure OpenAI). There is no custom exception hierarchy, no error-code enum, no middleware layer, and no `panic`/`recover` equivalent — this is a standard Python script/pipeline.

## Key Files

- `src/data_loader.py` — raises `ValueError` when CSVs are missing required columns, contain nulls in required fields, or have non-positive numeric values. These are the only places where malformed input is rejected.
- `src/llm_client.py` — wraps all OpenAI/Azure calls in `try/except Exception as exc: # noqa: BLE001 - degrade gracefully`. On failure it logs the error into a `CallRecord.ok = False` entry and returns `None`; callers fall back to deterministic templates. The class also stores an `_init_error` string so `status()` reports why the client is unavailable.
- `src/main.py` — catches `AttributeError`/`ValueError` from `sys.stdout.reconfigure(encoding="utf-8")` on platforms that don't support it, treating stdout encoding issues as non-fatal.
- `tests/test_core.py` — asserts expected rejection reasons (`"window"`, `"scope"`, `"direction"`, `"peer_divergence"`, `"magnitude"`) returned by the adjudicator rather than asserting raised exceptions; errors in business logic are represented as structured results, not thrown.

## Architecture & Conventions

1. **Input validation via `ValueError`**: `load_shipments` and `load_notes` validate schema and value constraints immediately after `pd.read_csv`, raising descriptive `ValueError` messages (e.g. `"shipment file missing columns: [...]"`, `"quantity_tonnes and distance_km must be positive"`).

2. **External-service failures return `None` instead of propagating**: `LLMClient.chat` swallows every exception, records it in `TokenLog.records` with `ok=False` and a truncated message (`str(exc)[:300]`), and returns `None`. The docstring explicitly states: *"Returns None on failure (logged)"*. The module-level design goals comment says: *"failures degrade gracefully: the caller falls back to deterministic templates, so the pipeline NEVER depends on the LLM being reachable."*

3. **No custom exception types**: No `class *Error(Exception)` definitions exist anywhere under `freight-watchdog/src/`. All domain-level "errors" (e.g. a candidate not justified by a note) are modeled as data — the adjudicator returns a result object whose `passed` flag and `rejection` string encode the outcome.

4. **Top-level CLI never lets exceptions escape**: `main()` returns an `int` exit code; subcommands return `0`/`1` and are invoked through `argparse`, which converts uncaught exceptions into its own usage/error output. The only explicit `try/except` at the top level guards `sys.stdout.reconfigure` against unsupported environments.

5. **Lint suppression signals intentional broad catching**: Every `except Exception` block carries `# noqa: BLE001 - degrade gracefully`, marking these as deliberate trade-offs rather than accidental bare-except patterns.

## Conventions & Constraints

- Input-schema violations surface as `ValueError` with a human-readable message describing the exact problem (missing columns, nulls, non-positive values).
- External API calls are wrapped in `try/except Exception` and converted to a `None` return plus a logged record; callers must check for `None` and use deterministic fallbacks.
- Business-rule failures (adjudication) are represented as structured result objects with a `rejection` discriminator string, not raised exceptions.
- The test suite validates behavior by asserting returned result fields (`d.passed`, `d.rejection`) rather than expecting exceptions to be raised.