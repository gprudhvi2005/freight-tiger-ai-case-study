"""Eval harness: score the pipeline against a hand-labeled set.

What it checks (and why each check builds trust):

1. SAMPLE REPLAY    The three rows of sample_output_format_v2.csv must be
                    reproduced number-for-number (cost_per_tonne_km,
                    vs_own_history, vs_similar_routes) and verdict-for-verdict.
2. LABELED VERDICTS Every hand-labeled candidate route-week must appear in
                    the report with the expected flagged value and
                    matched_note_id (exact note or blank). Missing = false
                    negative; unexpected extra rows = false positives.
3. NEGATIVE CONTROLSRoute-weeks that must NOT be flagged (trap notes, noise,
                    distractors) must be absent from the report.
4. HALLUCINATION    For every justified row in the report, an INDEPENDENT
    AUDIT           checker re-verifies the cited note against the raw
                    context_notes.csv: the note must exist, apply to the
                    route (or all routes), temporally overlap the flagged
                    week, and not be one of the notes that explicitly say
                    costs were unaffected. This is deliberately written as
                    a separate, simpler code path from the adjudicator.
5. RETRIEVAL RECALL For every justified row, the justifying note must have
                    been in the top-5 retrieval results (i.e. the recall
                    backfill never had to rescue a justification).

Run:  python eval/run_eval.py          (from the project root)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

LABELED_SET = Path(__file__).parent / "labeled_set.csv"

# Independent (from src/adjudicator.py) direction phrases for the audit.
_AUDIT_NO_EFFECT = [
    r"not significantly affected", r"remained stable", r"no major disruptions",
    r"no significant disruptions", r"returned to normal", r"remained normal",
    r"without a rate change", r"costs? (?:were )?absorbed", r"improved road conditions",
]
_AUDIT_DATE_RANGE = re.compile(
    r"from ([A-Za-z]{3,9})\.? (\d{1,2}) to ([A-Za-z]{3,9})\.? (\d{1,2})", re.IGNORECASE)
_AUDIT_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                 "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def _pct_value(text: str) -> float:
    return float(str(text).split("%")[0].replace("+", ""))


def _read_sample_loose(path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        for line in fh:
            line = line.rstrip("\n")
            if line:
                rows.append(line.split(",", len(header) - 1))
    return pd.DataFrame(rows, columns=header)


def check_sample_replay(report: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 1: reproduce sample_output_format_v2.csv rows exactly."""
    passed, failed = [], []
    sample = _read_sample_loose(config.SAMPLE_OUTPUT_CSV)
    for _, s in sample.iterrows():
        r = report[(report["route"] == s["route"]) & (report["week_of"] == s["week_of"])]
        if r.empty:
            failed.append(f"{s['route']} {s['week_of']}: row missing from report")
            continue
        r = r.iloc[0]
        checks = {
            "cost_per_tonne_km": f"{_num(r['cost_per_tonne_km']):.2f}" == s["cost_per_tonne_km"],
            "vs_own_history": abs(_pct_value(r["vs_own_history"]) - _pct_value(s["vs_own_history"])) < 0.05,
            "vs_similar_routes": abs(_pct_value(r["vs_similar_routes"]) - _pct_value(s["vs_similar_routes"])) < 0.05,
            "flagged": r["flagged"] == s["flagged"],
            "matched_note_id": str(r["matched_note_id"]) == str(s["matched_note_id"]),
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            failed.append(f"{s['route']} {s['week_of']}: {', '.join(bad)} mismatch")
        else:
            passed.append(f"{s['route']} {s['week_of']}: all fields match sample")
    return passed, failed


def _num(x) -> float:
    return float(x)


def check_labeled_verdicts(report: pd.DataFrame, labels: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 2: expected candidates appear with expected verdicts."""
    passed, failed = [], []
    expected = labels[labels["kind"] == "candidate"]
    report_keys = {(r["route"], r["week_of"]) for _, r in report.iterrows()}

    for _, lab in expected.iterrows():
        key = (lab["route"], lab["week_of"])
        if key not in report_keys:
            failed.append(f"{lab['route']} {lab['week_of']}: expected in report, missing (false negative)")
            continue
        row = report[(report["route"] == key[0]) & (report["week_of"] == key[1])].iloc[0]
        ok_flag = row["flagged"] == lab["expected_flagged"]
        ok_note = str(row["matched_note_id"]).strip() == str(lab["expected_matched_note_id"]).strip()
        if ok_flag and ok_note:
            passed.append(f"{lab['route']} {lab['week_of']}: verdict {lab['expected_flagged']}"
                          f"{f' via {lab['expected_matched_note_id']}' if str(lab['expected_matched_note_id']).strip() else ''}")
        else:
            failed.append(f"{lab['route']} {lab['week_of']}: expected "
                          f"{lab['expected_flagged']}/{lab['expected_matched_note_id'] or '(none)'}, "
                          f"got {row['flagged']}/{row['matched_note_id'] or '(none)'}")

    # Extra rows in the report that are not in the labeled candidate set:
    expected_keys = {(lab["route"], lab["week_of"]) for _, lab in expected.iterrows()}
    extras = report_keys - expected_keys
    for route, week in sorted(extras):
        failed.append(f"{route} {week}: in report but not in labeled set (unexpected flag)")
    return passed, failed


def check_negative_controls(report: pd.DataFrame, labels: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 3: negative-control route-weeks must be absent."""
    passed, failed = [], []
    negatives = labels[labels["kind"] == "negative"]
    for _, lab in negatives.iterrows():
        hit = report[(report["route"] == lab["route"]) & (report["week_of"] == lab["week_of"])]
        if hit.empty:
            passed.append(f"{lab['route']} {lab['week_of']}: correctly not flagged")
        else:
            failed.append(f"{lab['route']} {lab['week_of']}: flagged but should not be "
                          f"({lab['why']})")
    return passed, failed


def check_hallucination_audit(report: pd.DataFrame, notes: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Check 4: independent re-verification of every justified row."""
    passed, failed = [], []
    note_by_id = {str(r["note_id"]): r for _, r in notes.iterrows()}
    justified = report[report["flagged"].str.startswith("No")]

    if justified.empty:
        failed.append("no justified rows found in report (expected at least N002/N001 rows)")

    for _, row in justified.iterrows():
        note_id = str(row["matched_note_id"]).strip()
        if note_id not in note_by_id:
            failed.append(f"{row['route']} {row['week_of']}: cited note {note_id} does not exist")
            continue
        note = note_by_id[note_id]
        text = str(note["note"])
        applies_to = str(note["applies_to"])
        note_date = pd.Timestamp(note["date"])
        week_start = pd.Timestamp(row["week_of"])
        week_end = week_start + pd.Timedelta(days=6)

        # (a) route scope
        scope_ok = applies_to.strip().lower() == "all routes" or applies_to.strip() == row["route"]
        # (b) temporal compatibility: the note's effective window overlaps the week
        m = _AUDIT_DATE_RANGE.search(text)
        if m:
            mon1, d1, mon2, d2 = m.group(1)[:3].lower(), int(m.group(2)), m.group(3)[:3].lower(), int(m.group(4))
            win_start = pd.Timestamp(year=note_date.year, month=_AUDIT_MONTHS[mon1], day=d1)
            win_end = pd.Timestamp(year=note_date.year, month=_AUDIT_MONTHS[mon2], day=d2)
        else:
            win_start = note_date.normalize() - pd.Timedelta(days=note_date.weekday())
            win_end = win_start + pd.Timedelta(days=6)  # conservative: note's own week
        time_ok = win_start <= week_end and win_end >= week_start
        # (c) direction: not an explicitly no-effect note
        direction_ok = not any(re.search(p, text, re.IGNORECASE) for p in _AUDIT_NO_EFFECT)

        if scope_ok and time_ok and direction_ok:
            passed.append(f"{row['route']} {row['week_of']}: note {note_id} verified independently "
                          f"(scope, window overlap, direction)")
        else:
            why = []
            if not scope_ok:
                why.append(f"scope ({applies_to})")
            if not time_ok:
                why.append(f"window ({win_start.date()}..{win_end.date()} vs week {week_start.date()})")
            if not direction_ok:
                why.append("note says costs unaffected")
            failed.append(f"{row['route']} {row['week_of']}: note {note_id} fails independent audit: "
                          + "; ".join(why))
    return passed, failed


def check_retrieval_recall(diagnostics: dict) -> tuple[list[str], list[str]]:
    """Check 5: every justification was found by top-5 retrieval, not backfill."""
    passed, failed = [], []
    for row in diagnostics["rows"]:
        if not row["justified"]:
            continue
        retrieved_ids = [r["note_id"] for r in row["retrieved"]]
        if row["matched_note_id"] in retrieved_ids and not row["used_backfill"]:
            rank = retrieved_ids.index(row["matched_note_id"]) + 1
            passed.append(f"{row['route']} {row['week_of']}: note {row['matched_note_id']} "
                          f"retrieved at rank {rank}")
        elif row["used_backfill"] and row["justified"]:
            failed.append(f"{row['route']} {row['week_of']}: justification only found via "
                          f"backfill (retrieval recall miss)")
    return passed, failed


def main() -> int:
    print("running pipeline (deterministic mode)...")
    result = run_pipeline(use_llm=False, quiet=True)
    report, diagnostics = result.report, result.diagnostics
    labels = pd.read_csv(LABELED_SET, keep_default_na=False)
    notes = pd.read_csv(config.NOTES_CSV)

    sections = {
        "1. sample replay (3 rows from sample_output_format_v2.csv)": check_sample_replay(report),
        "2. labeled verdicts (hand-labeled candidate route-weeks)": check_labeled_verdicts(report, labels),
        "3. negative controls (must-not-flag route-weeks)": check_negative_controls(report, labels),
        "4. hallucination audit (independent re-verification of justified rows)": check_hallucination_audit(report, notes),
        "5. retrieval recall (justifying note in top-5, no backfill rescue)": check_retrieval_recall(diagnostics),
    }

    lines = ["# Eval report", "",
             f"- pipeline rows: {len(report)} "
             f"({(report['flagged'] == 'Yes').sum()} flagged, "
             f"{(report['flagged'].str.startswith('No')).sum()} justified)",
             f"- labeled set: {len(labels)} rows "
             f"({(labels['kind'] == 'candidate').sum()} candidates, "
             f"{(labels['kind'] == 'negative').sum()} negative controls)", ""]

    n_pass = n_fail = 0
    for title, (passed, failed) in sections.items():
        lines.append(f"## {title}")
        lines.append("")
        for p in passed:
            lines.append(f"- PASS {p}")
        for f_ in failed:
            lines.append(f"- FAIL {f_}")
        if not failed:
            lines.append(f"- all {len(passed)} checks passed")
        lines.append("")
        n_pass += len(passed)
        n_fail += len(failed)

    lines.append(f"**Total: {n_pass} passed, {n_fail} failed**")
    report_path = PROJECT_ROOT / "eval" / "eval_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {report_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
