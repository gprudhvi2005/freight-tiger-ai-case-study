"""Deterministic adjudication: does a retrieved note REALLY justify a rise?

This module is the anti-hallucination core of the assistant. Retrieval
(embeddings) proposes candidate notes; this adjudicator disposes, using
strict, explainable rules -- never a language model guess:

  Rule 1 (scope)   The note must apply to this route (its ``applies_to`` is
                   the route or "All Routes"). Notes whose text says the
                   affected routes are "not part of this dataset" apply to
                   no route at all.
  Rule 2 (window)  The note's effective time window must overlap the flagged
                   week (Monday..Sunday). A flood that ended on Mar 8 does
                   not justify the week of Mar 10.
  Rule 3 (direction) The note must describe a cost INCREASE. Notes saying
                   costs were "not significantly affected", demand "remained
                   stable", costs "absorbed without a rate change", routes
                   "returned to normal", or road conditions "improved" can
                   never justify a rise.
  Rule 4 (magnitude) If the note quantifies the increase (e.g. "roughly
                   5-7%"), the observed rise must fit within it. A note that
                   applies to ALL routes additionally cannot explain a
                   divergence vs similar routes (peers are equally affected),
                   so it can only justify a small uniform rise -- never a
                   route-specific spike.

Notes are parsed into structured facts by generic, deterministic text
patterns (dates, percentages, direction keywords) -- no note id is
hard-coded anywhere. tests/test_core.py and eval/run_eval.py verify the
parsed facts of every note in this dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .notes_kb import RetrievedNote

# ---------------------------------------------------------------------------
# Parsing patterns
# ---------------------------------------------------------------------------

# Phrases that mean "this note does NOT describe a cost increase".
_NO_EFFECT_PATTERNS = [
    r"not significantly affected",
    r"remained stable",
    r"no major disruptions",
    r"no significant disruptions",
    r"returned to normal",
    r"remained normal",
    r"without a rate change",
    r"costs? (?:were )?absorbed",
]

# Phrases that mean "conditions got cheaper / easier" (wrong direction for a rise).
_COST_DOWN_PATTERNS = [
    r"improved road conditions",
    r"resurfacing",
    r"costs? (?:fell|dropped|decreased)",
    r"prices? (?:fell|dropped|decreased)",
]

# Phrases that describe a cost increase.
_COST_UP_PATTERNS = [
    r"surcharge",
    r"prices? rose",
    r"pushing up",
    r"higher trip costs",
    r"rose nationwide",
    r"toll plaza was commissioned",
    r"costs? (?:rose|increased|went up)",
]

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# "from Feb 24 to Mar 8" -> start/end month-day pair
_DATE_RANGE_RE = re.compile(
    r"from ([A-Za-z]{3,9})\.? (\d{1,2})(?:st|nd|rd|th)? to ([A-Za-z]{3,9})\.? (\d{1,2})(?:st|nd|rd|th)?",
    re.IGNORECASE,
)

# "roughly 5-7%" / "by 5 to 7 percent" / "about 6%"
_MAGNITUDE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*%")
_MAGNITUDE_SINGLE_RE = re.compile(r"(?:by|of|roughly|about|approximately)\s+(\d+(?:\.\d+)?)\s*%")

_SCOPE_EXCLUSION_RE = re.compile(r"not part of this dataset", re.IGNORECASE)

# An "ongoing regime change" phrase extends the window indefinitely.
_OPEN_ENDED_RE = re.compile(r"starting this week", re.IGNORECASE)

# Tolerances for Rule 4.
PEER_DIVERGENCE_TOL_PCT = 5.0        # all-routes notes can't explain > this much peer divergence
ALL_ROUTES_UNQUANTIFIED_CAP_PCT = 10.0  # safety cap when an all-routes note states no magnitude


@dataclass
class NoteFacts:
    """Structured facts parsed deterministically from one context note."""

    note_id: str
    routes: frozenset[str] | None  # None => applies to all routes; empty => no routes
    window_start: pd.Timestamp
    window_end: pd.Timestamp | None  # None => open-ended
    direction: str  # "cost_up" | "cost_down" | "no_effect"
    magnitude_upper_pct: float | None
    parse_rationale: str


@dataclass
class CandidateDecision:
    """How one retrieved note was judged for one flagged route-week."""

    note_id: str
    passed: bool
    rejection: str | None  # scope | window | direction | peer_divergence | magnitude
    detail: str


@dataclass
class Verdict:
    """Final adjudication outcome for one flagged route-week."""

    justified: bool
    matched_note_id: str
    matched_note_text: str
    matched_similarity: float | None
    decisions: list[CandidateDecision] = field(default_factory=list)
    used_backfill: bool = False


# ---------------------------------------------------------------------------
# Note parsing
# ---------------------------------------------------------------------------

def _match_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _parse_magnitude(text: str) -> float | None:
    m = _MAGNITUDE_RE.search(text)
    if m:
        return float(max(float(m.group(1)), float(m.group(2))))
    m = _MAGNITUDE_SINGLE_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def _parse_window(text: str, note_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp | None, str]:
    """Resolve the note's effective window from its text.

    Priority: explicit "from X to Y" range > open-ended "starting this week"
    > the calendar week containing the note date (the conservative default
    for week-scoped events like festival weeks or week-long maintenance).
    """
    m = _DATE_RANGE_RE.search(text)
    if m:
        mon1, day1, mon2, day2 = m.group(1)[:3].lower(), int(m.group(2)), m.group(3)[:3].lower(), int(m.group(4))
        if mon1 in _MONTHS and mon2 in _MONTHS:
            start = pd.Timestamp(year=note_date.year, month=_MONTHS[mon1], day=day1)
            end = pd.Timestamp(year=note_date.year, month=_MONTHS[mon2], day=day2)
            if end < start:  # range crosses a year boundary
                end = pd.Timestamp(year=note_date.year + 1, month=_MONTHS[mon2], day=day2)
            return start, end, f"explicit date range {start.date()}..{end.date()}"

    if _OPEN_ENDED_RE.search(text):
        return note_date, None, "ongoing change starting this week (open-ended)"

    week_start = note_date.normalize() - pd.Timedelta(days=note_date.weekday())
    week_end = week_start + pd.Timedelta(days=6)
    return week_start, week_end, f"calendar week of the note date ({week_start.date()}..{week_end.date()})"


def parse_note(note_id: str, applies_to: str, note_date: pd.Timestamp, text: str) -> NoteFacts:
    """Parse one context note into structured facts (fully deterministic)."""
    routes: frozenset[str] | None
    if _SCOPE_EXCLUSION_RE.search(text):
        routes = frozenset()  # the affected routes are not in this dataset
        scope_rationale = "text says affected routes are not part of this dataset"
    elif applies_to.strip().lower() == "all routes":
        routes = None
        scope_rationale = 'applies_to = "All Routes"'
    else:
        routes = frozenset({applies_to.strip()})
        scope_rationale = f"applies_to = {applies_to.strip()}"

    if _match_any(_NO_EFFECT_PATTERNS, text):
        direction = "no_effect"
    elif _match_any(_COST_DOWN_PATTERNS, text):
        direction = "cost_down"
    elif _match_any(_COST_UP_PATTERNS, text):
        direction = "cost_up"
    else:
        direction = "no_effect"  # unrecognised => assume it does not justify a rise

    magnitude = _parse_magnitude(text)
    window_start, window_end, window_rationale = _parse_window(text, note_date)

    rationale = (
        f"scope: {scope_rationale}; direction: {direction}; "
        f"magnitude_upper: {magnitude}; window: {window_rationale}"
    )
    return NoteFacts(
        note_id=note_id,
        routes=routes,
        window_start=window_start,
        window_end=window_end,
        direction=direction,
        magnitude_upper_pct=magnitude,
        parse_rationale=rationale,
    )


def parse_all_notes(notes: pd.DataFrame) -> dict[str, NoteFacts]:
    """Parse every note in the context file into NoteFacts, keyed by note_id."""
    return {
        str(row["note_id"]): parse_note(
            str(row["note_id"]), str(row["applies_to"]), pd.Timestamp(row["date"]), str(row["note"])
        )
        for _, row in notes.iterrows()
    }


# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------

def _week_span(week_of: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(week_of).normalize()
    return start, start + pd.Timedelta(days=6)


def _windows_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start <= b_end and (a_end is None or a_end >= b_start)


def judge_candidate(
    facts: NoteFacts,
    route: str,
    week_of: pd.Timestamp,
    own_pct: float,
    sim_pct: float,
) -> CandidateDecision:
    """Apply Rules 1-4 to one note for one flagged route-week."""
    wk_start, wk_end = _week_span(week_of)

    # Rule 1: scope.
    if facts.routes is not None and route not in facts.routes:
        routes_desc = ", ".join(sorted(facts.routes)) if facts.routes else "no route in this dataset"
        return CandidateDecision(
            facts.note_id, False, "scope",
            f"note applies to {routes_desc}, not {route}",
        )

    # Rule 2: window overlap with the flagged week.
    if not _windows_overlap(facts.window_start, facts.window_end, wk_start, wk_end):
        win_end = facts.window_end.date() if facts.window_end is not None else "ongoing"
        return CandidateDecision(
            facts.note_id, False, "window",
            f"note window {facts.window_start.date()}..{win_end} does not overlap week "
            f"{wk_start.date()}..{wk_end.date()}",
        )

    # Rule 3: direction.
    if facts.direction == "no_effect":
        return CandidateDecision(
            facts.note_id, False, "direction",
            "note describes normal/stable conditions or explicitly says costs were not affected",
        )
    if facts.direction == "cost_down":
        return CandidateDecision(
            facts.note_id, False, "direction",
            "note describes conditions that would lower costs, not raise them",
        )

    # Rule 4: magnitude / peer-divergence logic.
    all_routes_note = facts.routes is None
    if all_routes_note:
        # Peers are equally affected: an all-routes note cannot explain a
        # divergence vs similar routes beyond noise.
        if sim_pct is not None and sim_pct > PEER_DIVERGENCE_TOL_PCT:
            return CandidateDecision(
                facts.note_id, False, "peer_divergence",
                f"note applies to all routes, so it cannot explain a {sim_pct:.1f}% divergence "
                f"vs similar-length routes (peers are equally affected)",
            )
        cap = facts.magnitude_upper_pct
        if cap is None:
            cap = ALL_ROUTES_UNQUANTIFIED_CAP_PCT
        if own_pct is not None and own_pct > cap + config.MAGNITUDE_TOLERANCE_PCT:
            return CandidateDecision(
                facts.note_id, False, "magnitude",
                f"note's stated effect is at most ~{cap:.0f}%, but the route is "
                f"{own_pct:.1f}% above its own history",
            )
    else:
        # Route-specific note: if it quantifies the increase, the observed
        # rise must fit; unquantified route-specific events (festival
        # surcharge, flooding) justify any observed size.
        if facts.magnitude_upper_pct is not None:
            rise = max(
                own_pct if own_pct is not None else -float("inf"),
                sim_pct if sim_pct is not None else -float("inf"),
            )
            if rise > facts.magnitude_upper_pct + config.MAGNITUDE_TOLERANCE_PCT:
                return CandidateDecision(
                    facts.note_id, False, "magnitude",
                    f"note's stated effect is at most ~{facts.magnitude_upper_pct:.0f}%, "
                    f"but the observed rise is {rise:.1f}%",
                )

    return CandidateDecision(facts.note_id, True, None, "passes all rules: scope, window, direction, magnitude")


def adjudicate(
    retrieved: list[RetrievedNote],
    facts_by_id: dict[str, NoteFacts],
    route: str,
    week_of: pd.Timestamp,
    own_pct: float,
    sim_pct: float,
) -> Verdict:
    """Judge every retrieved candidate; the best passing note (highest
    retrieval similarity, note_id as deterministic tie-break) wins."""
    decisions = []
    passing: list[tuple[float, str, RetrievedNote]] = []

    for note in retrieved:
        facts = facts_by_id.get(note.note_id)
        if facts is None:
            continue
        decision = judge_candidate(facts, route, week_of, own_pct, sim_pct)
        decisions.append(decision)
        if decision.passed:
            passing.append((note.similarity if note.similarity == note.similarity else -1.0,
                            note.note_id, note))

    if passing:
        passing.sort(key=lambda t: (-t[0], t[1]))
        _, _, best = passing[0]
        return Verdict(
            justified=True,
            matched_note_id=best.note_id,
            matched_note_text=best.note,
            matched_similarity=best.similarity,
            decisions=decisions,
        )
    return Verdict(
        justified=False,
        matched_note_id="",
        matched_note_text="",
        matched_similarity=None,
        decisions=decisions,
    )
