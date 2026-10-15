---
kind: configuration_system
name: Single-Module Config with .env Layering and CLI Flags
category: configuration_system
scope:
    - '**'
source_files:
    - freight-watchdog/src/config.py
    - freight-watchdog/.env.example
    - freight-watchdog/.env
    - freight-watchdog/src/main.py
---

# Configuration System

## Approach

The project uses a **single-module configuration** pattern centered on `freight-watchdog/src/config.py`. There is no external config framework (no Pydantic settings, no YAML/TOML loader, no Hydra). All tunables are Python module-level constants. Environment variables provide an override layer via a tiny custom `.env` parser.

## Key Files

- `freight-watchdog/src/config.py` — single source of truth for every runtime constant
- `freight-watchdog/.env` — local secrets / overrides (gitignored)
- `freight-watchdog/.env.example` — template documenting all supported env vars
- `freight-watchdog/src/main.py` — CLI entry point; the only place where `--no-llm` flags interact with config-derived behavior

## Architecture

### Loading order (enforced by `_load_dotenv`)

1. Module import triggers `_load_dotenv()` at import time.
2. It reads `freight-watchdog/.env` (resolved relative to `config.py`).
3. Each `KEY=VALUE` line is parsed (supports `#` comments, quoted values).
4. Values are written into `os.environ` **only if the key is not already set**.
5. Therefore: **real environment variables always win over `.env` values**.
6. After that, module-level constants read from `os.environ.get(...)` with defaults.

This gives three layers:

| Layer | Source | Override priority |
|---|---|---|
| Hard-coded defaults in `config.py` | e.g. `LLM_API_TYPE = os.environ.get("LLM_API_TYPE", "openai")` | Lowest |
| `.env` file | Parsed once at import | Middle |
| Real process environment | Shell / CI / Docker env | Highest |

### What is configured

- **Paths**: `PROJECT_ROOT`, `DATA_DIR`, `OUTPUT_DIR`, plus fixed data-file paths (`SHIPMENTS_CSV`, `NOTES_CSV`, `SAMPLE_OUTPUT_CSV`).
- **Detection thresholds**: `OWN_HISTORY_WEEKS=8`, `OWN_HISTORY_THRESHOLD_PCT=20.0`, `SIMILAR_ROUTES_THRESHOLD_PCT=20.0`, rounding decimals.
- **RAG**: `EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"`, `RETRIEVAL_TOP_K=5`.
- **Adjudication**: `MAGNITUDE_TOLERANCE_PCT=2.0`.
- **LLM**: `LLM_API_TYPE`, `LLM_MODEL`, `LLM_TEMPERATURE=0.0`, `LLM_MAX_TOKENS=2000`, `LLM_SEED=42`; Azure OpenAI variant keys (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`).
- **Output**: `TOKEN_LOG_PATH = OUTPUT_DIR / "token_cost_log.md"`.

### Third-party noise suppression

At import time the config sets:
```python
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
```
The use of `setdefault` means these can still be overridden by real env vars, but they suppress HF/transformers progress output so reproducibility diffs stay clean.

### CLI interaction

`main.py` defines a `--no-llm` flag on both `run` and `ask` subcommands. The flag does not change `config.py` directly; instead it is passed down as `use_llm=False` to `pipeline.run_pipeline()` and `qa.answer_question()`, which then bypass the LLM layer while still reading all other config values. This keeps the deterministic-vs-LLM toggle out of the central config module.

## Conventions and Constraints

- **All tunables live in one file.** The docstring states: "Every tunable lives here so the whole system is reproducible from one place." Modules import from `src.config` rather than redefining constants locally.
- **Third-party libraries are never imported at module level in `config.py`** — only `pathlib.Path` and `os` are used, keeping the config import side-effect-free beyond env loading.
- **Environment variable names are documented in `.env.example`**, which serves as the canonical list of supported keys. Keys not listed there (e.g. `LLM_BASE_URL`) exist in practice but are not formally declared in the example file.
- **Thresholds are pinned to dataset characteristics.** Comments document why `20.0` was chosen ("far above weekly noise (route-week stdev is ~3-8% of the mean) yet below every planted event in the dataset") and note that metric definitions must stay in sync with `data/sample_output_format_v2.csv`.
- **Determinism is enforced by config:** `LLM_TEMPERATURE=0.0` and `LLM_SEED=42` are hard-coded; the comment says "Verdicts, numbers and note citations are ALWAYS produced deterministically; the LLM only words the reason."
- **No schema validation or type checking** is performed on env values — they are consumed as strings and converted downstream (e.g. `LLM_API_TYPE.lower()`).
- **Secrets belong in `.env`** (which is gitignored); the example file contains only placeholders and comments.