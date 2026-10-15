"""Stretch goal: plain-English Q&A over the assistant's own findings.

How it works:
1. The findings (output/anomaly_report.csv + run_diagnostics.json) are turned
   into small documents -- one per flagged route-week, including its numbers,
   verdict and matched note text.
2. The question is resolved to a focus: any route mentioned, and any time
   window mentioned (month/year, "late 2025", H1/H2 ...). Findings matching
   the focus are always included; if the question has no recognizable focus,
   the most semantically similar findings are retrieved instead (RAG over our
   own results). Semantic hits fill any remaining room so context is never
   sparse.
3. The answer is either worded by the LLM (temperature 0, pinned context) or,
   if no LLM is available, composed deterministically from the same pinned
   rows -- so the assistant can always answer from its own results.

Example:
    python -m src.main ask "why did Mumbai-Pune get pricier in late 2025?"
"""

from __future__ import annotations

import json
import re

import pandas as pd

from . import config
from .explainer import normalize_ascii

_TOP_K = 8

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def load_findings() -> tuple[pd.DataFrame, dict]:
    """Load the last run's findings; regenerate them if absent."""
    report_path = config.OUTPUT_DIR / "anomaly_report.csv"
    diag_path = config.OUTPUT_DIR / "run_diagnostics.json"
    if not report_path.exists() or not diag_path.exists():
        from .pipeline import run_pipeline

        run_pipeline(use_llm=False, quiet=True)
    report = pd.read_csv(report_path, keep_default_na=False)
    with open(diag_path, encoding="utf-8") as fh:
        diagnostics = json.load(fh)
    return report, diagnostics


def _finding_docs(report: pd.DataFrame, notes_text: dict[str, str]) -> list[dict]:
    """One retrievable document per finding row."""
    docs = []
    for _, r in report.iterrows():
        note_id = str(r["matched_note_id"]).strip()
        note_part = ""
        if note_id:
            note_part = f" Justified by note {note_id}: {notes_text.get(note_id, '')}"
        docs.append({
            "route": r["route"],
            "week_of": r["week_of"],
            "text": (
                f"Route {r['route']}, week of {r['week_of']}: cost "
                f"{r['cost_per_tonne_km']} INR per tonne-km, "
                f"{r['vs_own_history']}, {r['vs_similar_routes']}. "
                f"Verdict: {'justified' if r['flagged'].startswith('No') else 'flagged unexplained'}.{note_part}"
            ),
        })
    return docs


def _time_filter(question: str) -> tuple[set[int] | None, set[int] | None]:
    """Extract referenced months and years from the question (or None)."""
    q = question.lower()
    months = {m for name, m in _MONTHS.items() if re.search(rf"\b{name}\b", q)} or None
    years = {int(y) for y in re.findall(r"\b(20\d{2})\b", q)} or None

    # "late YYYY" -> Oct-Dec, "early YYYY" -> Jan-Jun, "mid YYYY" -> Apr-Sep,
    # "H1/H2 YYYY" -> first/second half.
    m = re.search(r"\b(?:late|end of)\s*(20\d{2})\b", q)
    if m and months is None:
        months = {10, 11, 12}
        years = years or {int(m.group(1))}
    m = re.search(r"\bearly\s*(20\d{2})\b", q)
    if m and months is None:
        months = set(range(1, 7))
        years = years or {int(m.group(1))}
    m = re.search(r"\bmid\s*(20\d{2})\b", q)
    if m and months is None:
        months = set(range(4, 10))
        years = years or {int(m.group(1))}
    m = re.search(r"\bh2\s*(20\d{2})\b|\b(20\d{2})\s*h2\b", q)
    if m and months is None:
        months = set(range(7, 13))
        years = years or {int(m.group(1) or m.group(2))}
    m = re.search(r"\bh1\s*(20\d{2})\b|\b(20\d{2})\s*h1\b", q)
    if m and months is None:
        months = set(range(1, 7))
        years = years or {int(m.group(1) or m.group(2))}
    return months, years


def _select_findings(docs: list[dict], question: str, model) -> list[dict]:
    """Focused selection: explicit route/time mentions win; semantic fills."""
    months, years = _time_filter(question)
    ql = question.lower()

    focus = []
    for i, d in enumerate(docs):
        week = pd.Timestamp(d["week_of"])
        route_hit = d["route"].lower() in ql
        month_hit = months is None or week.month in months
        year_hit = years is None or week.year in years
        if route_hit and month_hit and year_hit:
            focus.append(i)

    if not focus:
        # no recognizable focus (or focus matched nothing): semantic top-k
        q_emb = model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
        corpus = [d["text"] for d in docs]
        emb = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
        sims = emb @ q_emb
        order = sorted(range(len(docs)), key=lambda i: -sims[i])
        return [docs[i] for i in order[:_TOP_K]]

    # a recognized route/time focus IS the answer context -- keep it tight
    return [docs[i] for i in sorted(focus)]


def _route_numbers(rows: list[dict]) -> str:
    """A compact, faithful numeric summary of the selected rows for a route."""
    cpks = [float(re.search(r"(\d+\.\d+)", r["text"]).group(1)) for r in rows]
    own = [float(re.search(r"([+-]\d+\.\d+)%", r["text"]).group(1)) for r in rows]
    sim = [float(re.findall(r"([+-]\d+\.\d+)%", r["text"])[1]) for r in rows]
    return (f"cost {min(cpks):.2f}-{max(cpks):.2f} INR/tonne-km, "
            f"{min(own):+.1f}% to {max(own):+.1f}% vs own history, "
            f"{min(sim):+.1f}% to {max(sim):+.1f}% vs similar routes")


def _deterministic_answer(rows: list[dict]) -> str:
    """Compose an answer from the selected finding rows (no LLM needed)."""
    if not rows:
        return ("I could not find anything in my findings matching that question. "
                "My current report covers the 2024-2025 shipment records; try naming "
                "a route (e.g. Mumbai-Pune) or a time range.")

    by_route: dict[str, list[dict]] = {}
    for r in rows:
        by_route.setdefault(r["route"], []).append(r)

    parts = []
    for route, items in sorted(by_route.items()):
        items = sorted(items, key=lambda r: r["week_of"])
        weeks = [r["week_of"] for r in items]
        justified = [r for r in items if "Justified by note" in r["text"]]
        flagged = [r for r in items if r not in justified]
        span = f"{weeks[0]} to {weeks[-1]}" if len(weeks) > 1 else weeks[0]
        part = (f"On {route}, {len(items)} flagged week"
                f"{'s' if len(items) != 1 else ''} ({span}), {_route_numbers(items)}. ")
        if justified:
            ids = sorted({re.search(r"Justified by note (\w+)", r["text"]).group(1)
                          for r in justified})
            first_note = re.search(r"Justified by note \w+: (.+)$", justified[0]["text"])
            part += (f"{len(justified)} {'were' if len(justified) != 1 else 'was'} "
                     f"justified by note {' and '.join(ids)} -- "
                     f"{first_note.group(1) if first_note else 'see the matched note text in the report'}. ")
        if flagged:
            part += (f"{len(flagged)} {'remain' if len(flagged) != 1 else 'remains'} "
                     f"unexplained and flagged for review -- no context note applies "
                     f"to that route and window. ")
        parts.append(part.strip())

    return (" ".join(parts)
            + " This comes straight from my own findings table; nothing beyond the "
              "shipped notes was used.")


_QA_SYSTEM = (
    "You are a freight-cost anomaly assistant. Answer the user's question "
    "ONLY using the FINDINGS provided -- these are your own verified results. "
    "Cite routes, weeks and note ids exactly as they appear. If the findings "
    "do not cover what is asked, say so plainly instead of guessing. "
    "Answer in 2-5 short plain-English sentences, no markdown."
)


def answer_question(question: str, use_llm: bool = True) -> tuple[str, list[str]]:
    """Answer a plain-English question from the assistant's own findings."""
    report, _ = load_findings()
    notes = pd.read_csv(config.NOTES_CSV)
    notes_text = {str(r["note_id"]): str(r["note"]) for _, r in notes.iterrows()}

    docs = _finding_docs(report, notes_text)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    selected = _select_findings(docs, question, model)

    sources = [f"{d['route']} week of {d['week_of']}" for d in selected]
    deterministic = _deterministic_answer(selected)

    if use_llm:
        from .llm_client import LLMClient

        llm = LLMClient()
        if llm.available:
            context = "\n".join(f"- {d['text']}" for d in selected) or "(no matching findings)"
            text = llm.chat(_QA_SYSTEM,
                            f"FINDINGS:\n{context}\n\nQUESTION: {question}",
                            purpose="qa")
            if text:
                text = " ".join(normalize_ascii(text).split())
                # grounding check: the answer must lean on at least one pinned
                # finding (a route name, a note id, or a week date from context)
                grounding = {r["route"] for r in selected}
                grounding |= {r["week_of"] for r in selected}
                grounding |= {m.group(1) for r in selected
                              for m in re.finditer(r"note (N\d+)", r["text"])}
                if not grounding or any(g in text for g in grounding):
                    return text, sources
            # fall through to deterministic on failure/empty/ungrounded
    return deterministic, sources
