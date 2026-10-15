"""Central configuration for the Freight Cost Watchdog pipeline.

Every tunable lives here so the whole system is reproducible from one place.
"""

from pathlib import Path
import os

# Keep third-party chatter (HF/transformers progress bars, load reports) out
# of run logs; they carry no signal for this pipeline and would pollute the
# reproducibility diffs.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def _load_dotenv() -> None:
    """Tiny .env loader (KEY=VALUE lines, # comments) -- no extra dependency.
    Real environment variables always win over .env values."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

SHIPMENTS_CSV = DATA_DIR / "shipment_records.csv"
NOTES_CSV = DATA_DIR / "context_notes.csv"
SAMPLE_OUTPUT_CSV = DATA_DIR / "sample_output_format_v2.csv"

# ---------------------------------------------------------------------------
# Metric definitions (graded fields -- do not change without re-verifying
# against data/sample_output_format_v2.csv)
# ---------------------------------------------------------------------------
# Week runs Monday..Sunday; week_of is that week's Monday date.
# cost_per_tonne_km(route, week) = sum(freight_cost) / sum(quantity * distance)
#   (pooled over all shipments of that route-week -- tonne-km weighted).
# vs_own_history: simple mean of the trailing OWN_HISTORY_WEEKS weekly
#   cost-per-tonne-km values, using only weeks STRICTLY before the current
#   week (no look-ahead). If fewer prior weeks exist, all prior weeks are used
#   and the row is tagged `partial_history=True`.
# vs_similar_routes: simple mean of the SAME-week cost-per-tonne-km of every
#   OTHER route sharing the same route_type (Short/Medium/Long).
OWN_HISTORY_WEEKS = 8

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
# A route-week is a *candidate anomaly* when its cost is rising and looks out
# of the ordinary on EITHER comparison:
#   own_pct  >= OWN_HISTORY_THRESHOLD   (vs its own trailing 8-week average)
#   sim_pct  >= SIMILAR_ROUTES_THRESHOLD (vs same-week similar-length routes)
# 20% is chosen because it is far above weekly noise (route-week stdev is
# ~3-8% of the mean) yet below every planted event in the dataset; see README.
OWN_HISTORY_THRESHOLD_PCT = 20.0
SIMILAR_ROUTES_THRESHOLD_PCT = 20.0

# Rounding for output CSV (matches sample_output_format_v2.csv).
CPK_DECIMALS = 2
PCT_DECIMALS = 1

# ---------------------------------------------------------------------------
# Retrieval (RAG)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 5  # notes retrieved per flagged route-week query

# ---------------------------------------------------------------------------
# Adjudication rules
# ---------------------------------------------------------------------------
# A retrieved note justifies a cost RISE on route R in week W only if ALL of:
#   1. route match: note.applies_to is R or "All Routes"
#      (notes whose text says the affected routes are not in the dataset
#       match no route at all),
#   2. window overlap: the note's effective window overlaps week W's 7 days,
#   3. direction: the note describes a cost INCREASE (surcharge, price rise,
#      disruption with higher trip costs). Notes that say costs were not
#      affected / demand stable / costs absorbed / conditions normal /
#      road improvement can never justify a rise,
#   4. magnitude: if the note quantifies the increase (e.g. "5-7%"), the
#      observed rise must fit within it. A note applying to ALL routes also
#      cannot explain a divergence vs peers (peers are equally affected), so
#      it can only ever justify a small same-week, cross-route rise.
#
# Tolerance added to a note's stated magnitude before calling it insufficient.
MAGNITUDE_TOLERANCE_PCT = 2.0

# ---------------------------------------------------------------------------
# LLM (explanation prose + Q&A). Verdicts, numbers and note citations are
# ALWAYS produced deterministically; the LLM only words the reason.
# ---------------------------------------------------------------------------
LLM_API_TYPE = os.environ.get("LLM_API_TYPE", "openai").lower()  # openai | azure
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0.0
# Reasoning models (e.g. Groq's gpt-oss) spend part of the output budget on
# reasoning tokens, so keep generous headroom; the final prose is 1-3 short
# sentences.
LLM_MAX_TOKENS = 2000
LLM_SEED = 42  # fixed seed passed alongside temperature 0

# Azure OpenAI variant
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")

# Token accounting is written here after a full run.
TOKEN_LOG_PATH = OUTPUT_DIR / "token_cost_log.md"
