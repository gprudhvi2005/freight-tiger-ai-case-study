---
kind: dependency_management
name: Python Dependencies via requirements.txt with Pinned Lower Bounds
category: dependency_management
scope:
    - '**'
source_files:
    - freight-watchdog/requirements.txt
---

## Approach

The repository uses the standard Python `pip` ecosystem. Dependency declarations are centralized in a single file: `freight-watchdog/requirements.txt`. There is no lockfile (no `requirements.lock`, `Pipfile.lock`, `poetry.lock`, or `uv.lock`) and no virtual-environment bootstrap script beyond the plain text manifest.

## Key Files

- `freight-watchdog/requirements.txt` — the only dependency manifest in the repository.

## Declared Dependencies

| Package | Version Constraint | Purpose |
|---|---:|---|
| `pandas` | `>=2.0` | Data loading / aggregation |
| `numpy` | `>=1.24` | Numerical operations |
| `scikit-learn` | `>=1.3` | Anomaly detection models |
| `sentence-transformers` | `>=2.2` | Embeddings for RAG context retrieval |
| `openai` | `>=1.30` | LLM client for adjudication |
| `matplotlib` | `>=3.7` | Chart generation |
| `pytest` | `>=7.4` | Unit testing (`tests/test_core.py`) |

## Conventions and Constraints

- **Single manifest**: All third-party dependencies are declared in `freight-watchdog/requirements.txt`; there are no per-package sub-manifests, no Poetry/Pipenv configuration, and no vendored packages under `vendor/`.
- **Lower-bound-only constraints**: Every package uses a `>=` lower bound rather than an exact pin (e.g., `pandas>=2.0`). This means `pip install -r requirements.txt` resolves to the latest compatible version at install time; reproducibility across environments is not enforced by a lockfile.
- **No private registry or authentication**: The manifest contains no `--index-url`, `--extra-index-url`, or environment-variable-based private index configuration. Dependencies are resolved from the default PyPI.
- **No build-time dependency tooling**: There is no `setup.py`, `pyproject.toml`, `setup.cfg`, `Pipfile`, or `poetry.toml`, so the project is not packaged as an installable distribution — it is consumed directly by running scripts under `freight-watchdog/src/`.
- **Test dependency separation**: `pytest` is listed alongside runtime dependencies in the same file rather than being split into a separate `requirements-dev.txt`.