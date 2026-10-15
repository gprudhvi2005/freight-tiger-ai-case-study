"""Explanation generation: deterministic templates, optionally worded by an LLM.

Division of labor (the anti-hallucination contract):
* The VERDICT, the matched note id and all NUMBERS come from the
  deterministic pipeline (metrics + adjudicator) and are pinned into the
  explanation context.
* The LLM (temperature 0, fixed seed) only words the ``reason`` prose from
  those pinned facts. If the LLM is unavailable or fails, a deterministic
  template produces the reason instead -- the pipeline never depends on it.

The template wording deliberately mirrors the style of
data/sample_output_format_v2.csv so deterministic runs read naturally.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .adjudicator import Verdict
from .llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You write the final 'reason' text for one row of a freight-cost anomaly "
    "report. You will be given VERIFIED facts computed by a deterministic "
    "pipeline. Follow these rules exactly:\n"
    "1. Never invent reasons, notes, dates, or numbers that are not in the facts.\n"
    "2. The verdict and the matched note id are FIXED. Do not change or question them.\n"
    "3. If the verdict is 'justified', name the note id and faithfully reflect its "
    "content and dates.\n"
    "4. If the verdict is 'unexplained', say that no genuine justification was "
    "found. You may mention why the closest considered note does not apply, but "
    "only using the rejection reasons given in the facts.\n"
    "5. Write 1-3 short plain-English sentences. No markdown, no quotation "
    "marks, no lists.\n"
    "6. End with a clear statement of the outcome: a clear explanation exists, "
    "or the rise is unexplained and worth a human review.\n"
    "7. Use plain ASCII characters only: regular hyphens, straight quotes."
)

# LLMs sometimes emit typographic Unicode (non-breaking hyphens, smart
# quotes, narrow spaces ...). Normalizing to ASCII keeps the CSV portable and
# byte-stable across platforms and console encodings.
_ASCII_MAP = {
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u00a0": " ", "\u2026": "...", "\u2212": "-",
    "\u202f": " ", "\u2009": " ", "\u2007": " ", "\u2060": "", "\ufeff": "",
}


def normalize_ascii(text: str) -> str:
    for bad, good in _ASCII_MAP.items():
        text = text.replace(bad, good)
    return text


def format_own_pct(own_pct: float, weeks_used: int) -> str:
    """Format the vs_own_history column exactly like the sample output."""
    base = f"{own_pct:+.{config.PCT_DECIMALS}f}% vs this route's past average"
    if weeks_used < config.OWN_HISTORY_WEEKS:
        base += f" (only {weeks_used} prior week{'s' if weeks_used != 1 else ''} available)"
    return base


def format_sim_pct(sim_pct: float | None) -> str:
    """Format the vs_similar_routes column exactly like the sample output."""
    if sim_pct is None or sim_pct != sim_pct:  # NaN check
        return "n/a (no similar routes this week)"
    return f"{sim_pct:+.{config.PCT_DECIMALS}f}% vs similar-length routes this week"


def _rejection_phrase(rejection: str, detail: str) -> str:
    """A short, faithful phrase for why a considered note did not apply."""
    if rejection == "scope":
        return f"it applies to a different route, not this one"
    if rejection == "window":
        return "its date window does not cover this week"
    if rejection == "direction":
        return "it does not describe a reason for a cost rise on this route"
    if rejection == "peer_divergence":
        return ("it affects all routes equally, so it cannot explain why this route "
                "is costlier than similar routes")
    if rejection == "magnitude":
        return "its stated effect is smaller than the observed rise"
    return "it does not justify this rise"


def _pinned_facts(route: str, week_of: pd.Timestamp, cpk: float, own_pct: float,
                  sim_pct: float | None, weeks_used: int, verdict: Verdict,
                  note_date: pd.Timestamp | None) -> dict:
    return {
        "route": route,
        "week_of": str(pd.Timestamp(week_of).date()),
        "cost_per_tonne_km": f"{cpk:.{config.CPK_DECIMALS}f}",
        "vs_own_history": f"{own_pct:+.1f}% above the route's trailing "
                          f"{weeks_used}-week average",
        "vs_similar_routes": (f"{sim_pct:+.1f}% above similar-length routes this week"
                              if sim_pct is not None and sim_pct == sim_pct else "not available"),
        "verdict": "justified" if verdict.justified else "unexplained",
        "matched_note_id": verdict.matched_note_id or "(none)",
        "matched_note_date": (str(note_date.date()) if note_date is not None else "(none)"),
        "matched_note_text": verdict.matched_note_text or "(none)",
        "considered_notes": [
            {
                "note_id": d.note_id,
                "rejection": d.rejection or "passed",
                "why": d.detail,
            }
            for d in verdict.decisions
        ],
    }


def _template_reason(route: str, week_of: pd.Timestamp, cpk: float, own_pct: float,
                     sim_pct: float | None, weeks_used: int, verdict: Verdict,
                     note_date: pd.Timestamp | None) -> str:
    """Deterministic reason text (mirrors sample_output_format_v2.csv style)."""
    if verdict.justified:
        note_id = verdict.matched_note_id
        date_str = str(note_date.date()) if note_date is not None else ""
        return (
            f"Matches note {note_id} dated {date_str}: \"{verdict.matched_note_text}\" "
            f"The cost rise has a clear explanation."
        )

    # Unexplained: mention the closest rejected candidate, if any.
    rejected = [d for d in verdict.decisions if not d.passed]
    near_miss = ""
    if rejected:
        best = rejected[0]
        phrase = _rejection_phrase(best.rejection, best.detail)
        near_miss = (
            f"The closest note considered ({best.note_id}) was ruled out: {phrase}. "
        )
    history_note = ""
    if weeks_used < config.OWN_HISTORY_WEEKS:
        history_note = (
            f"Baseline uses only {weeks_used} prior week"
            f"{'s' if weeks_used != 1 else ''} of history. "
        )
    return (
        f"{near_miss}{history_note}"
        f"No matching note found for this route and date range. "
        f"Cost rise looks unexplained and worth a human review."
    )


def build_reason(llm: LLMClient | None, route: str, week_of: pd.Timestamp,
                 cpk: float, own_pct: float, sim_pct: float | None,
                 weeks_used: int, verdict: Verdict,
                 note_date: pd.Timestamp | None,
                 note_dates: dict[str, pd.Timestamp]) -> str:
    """Produce the reason text: LLM-worded if possible, template otherwise.

    The LLM receives only pinned facts; the verdict and note id are fixed.
    Whatever the LLM returns, the deterministic verdict/numbers above are
    what get written to the CSV columns.
    """
    if llm is not None and llm.available:
        facts = _pinned_facts(route, week_of, cpk, own_pct, sim_pct,
                              weeks_used, verdict, note_date)
        user = (
            "FACTS (verified, do not contradict):\n"
            + "\n".join(f"- {k}: {v}" for k, v in facts.items() if k != "considered_notes")
            + "\n- considered_notes:\n"
            + "\n".join(
                f"  * {c['note_id']}: {c['rejection']} -- {c['why']}"
                for c in facts["considered_notes"]
            )
            + "\n\nWrite the reason text now."
        )
        text = llm.chat(_SYSTEM_PROMPT, user, purpose="explanation")
        if text:
            # Cheap post-condition: the reason must not contradict the pinned
            # verdict; if the LLM ignored instructions, fall back to template.
            if verdict.justified and verdict.matched_note_id not in text:
                text = None
            if not verdict.justified and text.lower().startswith("matches note"):
                text = None
            if text:
                return " ".join(normalize_ascii(text).split())

    return _template_reason(route, week_of, cpk, own_pct, sim_pct,
                            weeks_used, verdict, note_date)
