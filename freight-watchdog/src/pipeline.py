"""Pipeline orchestration: shipments -> weekly metrics -> candidates ->
retrieval -> adjudication -> explanations -> output CSV.

One run produces:
  output/anomaly_report.csv      the graded deliverable (exact sample format)
  output/run_diagnostics.json    per-row evidence (retrieval, decisions, numbers)
  output/token_cost_log.md       LLM usage for the run (if LLM enabled)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from . import config
from . import detector, weekly_metrics
from .adjudicator import Verdict, adjudicate, parse_all_notes
from .data_loader import load_notes, load_shipments
from .explainer import build_reason, format_own_pct, format_sim_pct
from .llm_client import LLMClient
from .notes_kb import NotesKnowledgeBase

OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]


@dataclass
class PipelineResult:
    report: pd.DataFrame        # exact output format rows
    diagnostics: dict           # evidence for eval / charts / QA
    metrics: pd.DataFrame       # full route-week metrics table
    llm_status: str


def _row_for_candidate(
    row: pd.Series,
    verdict: Verdict,
    reason: str,
) -> dict:
    return {
        "route": row["route"],
        "week_of": str(pd.Timestamp(row["week_of"]).date()),
        "cost_per_tonne_km": f"{row['cost_per_tonne_km']:.{config.CPK_DECIMALS}f}",
        "vs_own_history": format_own_pct(row["own_pct"], int(row["own_history_weeks_used"])),
        "vs_similar_routes": format_sim_pct(
            None if pd.isna(row.get("sim_pct")) else row["sim_pct"]
        ),
        "flagged": "No (justified)" if verdict.justified else "Yes",
        "matched_note_id": verdict.matched_note_id,
        "reason": reason,
    }


def run_pipeline(use_llm: bool = True, quiet: bool = False) -> PipelineResult:
    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    # ---- 1. Load and aggregate ------------------------------------------
    shipments = load_shipments()
    notes = load_notes()
    metrics = weekly_metrics.build_metrics(shipments)
    log(f"[1/6] loaded {len(shipments)} shipments, {len(notes)} notes; "
        f"{len(metrics)} route-weeks")

    # ---- 2. Candidates ---------------------------------------------------
    candidates = detector.select_candidates(metrics)
    log(f"[2/6] {len(candidates)} candidate anomalies "
        f"(>={config.OWN_HISTORY_THRESHOLD_PCT:.0f}% vs own history or "
        f">={config.SIMILAR_ROUTES_THRESHOLD_PCT:.0f}% vs similar routes)")

    # ---- 3. Retrieval (RAG) + 4. adjudication ---------------------------
    kb = NotesKnowledgeBase(notes)
    facts_by_id = parse_all_notes(notes)
    note_dates = {str(r["note_id"]): pd.Timestamp(r["date"]) for _, r in notes.iterrows()}
    log(f"[3/6] notes embedded with {kb.model_name}; adjudication rules armed")

    llm = LLMClient() if use_llm else None
    if llm is not None:
        log(f"[4/6] {llm.status()}")
    else:
        log("[4/6] LLM disabled; explanations use deterministic templates")

    rows = []
    diag_rows = []
    n_backfills = 0

    for _, cand in candidates.iterrows():
        route = cand["route"]
        week_of = pd.Timestamp(cand["week_of"])
        own_pct = float(cand["own_pct"])
        sim_pct = None if pd.isna(cand["sim_pct"]) else float(cand["sim_pct"])

        retrieved = kb.retrieve(route, week_of, own_pct, sim_pct)
        verdict = adjudicate(retrieved, facts_by_id, route, week_of, own_pct, sim_pct)

        used_backfill = False
        if not verdict.justified:
            # Recall backfill for tiny corpora: give the adjudicator one look
            # at every note before concluding "unexplained". Retrieval-order
            # decisions are preserved so the near-miss explanation still cites
            # the most semantically similar rejected note.
            all_notes = kb.all_notes()
            backfill_verdict = adjudicate(all_notes, facts_by_id, route, week_of,
                                          own_pct, sim_pct)
            seen = {d.note_id for d in verdict.decisions}
            backfill_verdict.decisions = verdict.decisions + [
                d for d in backfill_verdict.decisions if d.note_id not in seen
            ]
            backfill_verdict.used_backfill = True
            verdict = backfill_verdict
            used_backfill = True
            n_backfills += 1

        note_date = note_dates.get(verdict.matched_note_id)
        reason = build_reason(
            llm, route, week_of, float(cand["cost_per_tonne_km"]),
            own_pct, sim_pct, int(cand["own_history_weeks_used"]),
            verdict, note_date, note_dates,
        )

        rows.append(_row_for_candidate(cand, verdict, reason))
        diag_rows.append({
            "route": route,
            "week_of": str(week_of.date()),
            "route_type": cand["route_type"],
            "n_shipments": int(cand["n_shipments"]),
            "cost_per_tonne_km": round(float(cand["cost_per_tonne_km"]), 4),
            "own_pct": round(own_pct, 2),
            "sim_pct": None if sim_pct is None else round(sim_pct, 2),
            "own_history_weeks_used": int(cand["own_history_weeks_used"]),
            "partial_history": bool(cand["partial_history"]),
            "justified": verdict.justified,
            "matched_note_id": verdict.matched_note_id,
            "matched_similarity": None if verdict.matched_similarity is None
                                  or verdict.matched_similarity != verdict.matched_similarity
                                  else round(verdict.matched_similarity, 4),
            "used_backfill": used_backfill,
            "retrieved": [
                {"note_id": n.note_id, "similarity": round(n.similarity, 4)}
                for n in retrieved
            ],
            "decisions": [
                {"note_id": d.note_id, "passed": d.passed,
                 "rejection": d.rejection, "detail": d.detail}
                for d in verdict.decisions
            ],
        })

    log(f"[5/6] adjudicated {len(candidates)} candidates: "
        f"{sum(1 for r in rows if r['flagged'] == 'Yes')} flagged (unexplained), "
        f"{sum(1 for r in rows if r['flagged'] == 'No (justified)')} justified; "
        f"backfill fired {n_backfills} time(s)")

    # ---- 5. Assemble report ----------------------------------------------
    report = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    diagnostics = {
        "config": {
            "own_history_weeks": config.OWN_HISTORY_WEEKS,
            "own_history_threshold_pct": config.OWN_HISTORY_THRESHOLD_PCT,
            "similar_routes_threshold_pct": config.SIMILAR_ROUTES_THRESHOLD_PCT,
            "embedding_model": kb.model_name,
            "retrieval_top_k": config.RETRIEVAL_TOP_K,
            "llm_enabled": llm is not None and llm.available,
            "llm_model": llm.model if llm is not None else None,
            "llm_temperature": config.LLM_TEMPERATURE if llm is not None else None,
            "llm_seed": config.LLM_SEED if llm is not None else None,
        },
        "n_route_weeks": int(len(metrics)),
        "n_candidates": int(len(candidates)),
        "n_flagged": int((report["flagged"] == "Yes").sum()) if len(report) else 0,
        "n_justified": int((report["flagged"] == "No (justified)").sum()) if len(report) else 0,
        "n_backfills": n_backfills,
        "notes_parsed": [
            {
                "note_id": fid,
                "routes": None if f.routes is None else sorted(f.routes),
                "window": [
                    str(f.window_start.date()),
                    None if f.window_end is None else str(f.window_end.date()),
                ],
                "direction": f.direction,
                "magnitude_upper_pct": f.magnitude_upper_pct,
                "rationale": f.parse_rationale,
            }
            for fid, f in sorted(facts_by_id.items())
        ],
        "rows": diag_rows,
    }

    # ---- 6. Persist -------------------------------------------------------
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.OUTPUT_DIR / "anomaly_report.csv", index=False)
    with open(config.OUTPUT_DIR / "run_diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2)
    if llm is not None:
        llm.write_token_log(mode="llm" if llm.available else "llm-unavailable-fallback")
    log(f"[6/6] wrote {config.OUTPUT_DIR / 'anomaly_report.csv'} "
        f"({len(report)} rows) + diagnostics + token log")

    llm_status = llm.status() if llm is not None else "llm disabled (--no-llm)"
    return PipelineResult(report=report, diagnostics=diagnostics, metrics=metrics,
                          llm_status=llm_status)
