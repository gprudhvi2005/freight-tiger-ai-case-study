"""Freight Cost Watchdog -- CLI entry point.

Usage:
    python -m src.main run [--no-llm]        full pipeline -> output/anomaly_report.csv
    python -m src.main verify-sample         reproduce the 3 sample rows exactly
    python -m src.main notes                 show how each context note was parsed
    python -m src.main ask \"<question>\"      ask the assistant about its findings
"""

from __future__ import annotations

import argparse
import sys

from . import config


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline

    result = run_pipeline(use_llm=not args.no_llm)
    if result.report.empty:
        print("no anomalies detected; nothing to write")
        return 0

    print("\n=== anomaly_report.csv ===")
    print(result.report.to_string(index=False))
    print(f"\n{result.llm_status}")
    return 0


def _read_sample_loose(path) -> "pd.DataFrame":
    """Read the sample CSV robustly: the first 7 fields never contain
    commas, so everything after the 7th comma is the reason text (the file
    has some unquoted reasons containing commas)."""
    import pandas as pd

    rows = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",", len(header) - 1)
            rows.append(parts)
    return pd.DataFrame(rows, columns=header)


def cmd_verify_sample(_: argparse.Namespace) -> int:
    """Hard check: our numbers must reproduce every sample output row."""
    import pandas as pd

    from .data_loader import load_shipments
    from .weekly_metrics import build_metrics

    sample = _read_sample_loose(config.SAMPLE_OUTPUT_CSV)
    metrics = build_metrics(load_shipments())

    failures = []
    for _, s in sample.iterrows():
        m = metrics[
            (metrics["route"] == s["route"])
            & (metrics["week_of"] == pd.Timestamp(s["week_of"]))
        ]
        if m.empty:
            failures.append(f"{s['route']} {s['week_of']}: no such route-week")
            continue
        m = m.iloc[0]
        checks = {
            "cost_per_tonne_km": abs(round(m["cost_per_tonne_km"], 2) - float(s["cost_per_tonne_km"])) < 5e-3,
            "own_pct": abs(round(m["own_pct"], 1) - float(_pct_value(s["vs_own_history"]))) < 5e-2,
            "sim_pct": abs(round(m["sim_pct"], 1) - float(_pct_value(s["vs_similar_routes"]))) < 5e-2,
        }
        for name, ok in checks.items():
            if not ok:
                failures.append(f"{s['route']} {s['week_of']}: {name} mismatch")

    if failures:
        print("SAMPLE VERIFICATION FAILED:")
        for f in failures:
            print(" -", f)
        return 1
    print(f"sample verification OK: all {len(sample)} rows reproduce exactly "
          "(cost_per_tonne_km, vs_own_history, vs_similar_routes)")
    return 0


def _pct_value(text: str) -> float:
    """'+35.5% vs this route's past average' -> 35.5"""
    return float(str(text).split("%")[0].replace("+", ""))


def cmd_notes(_: argparse.Namespace) -> int:
    from .adjudicator import parse_all_notes
    from .data_loader import load_notes

    facts = parse_all_notes(load_notes())
    print(f"{'id':6} {'direction':10} {'routes':22} {'window':26} {'mag':5} rationale")
    for fid, f in sorted(facts.items()):
        routes = "ALL" if f.routes is None else (",".join(sorted(f.routes)) or "(none in dataset)")
        win = f"{f.window_start.date()}..{f.window_end.date() if f.window_end is not None else 'ongoing'}"
        mag = f"{f.magnitude_upper_pct:.0f}%" if f.magnitude_upper_pct is not None else "-"
        print(f"{fid:6} {f.direction:10} {routes:22} {win:26} {mag:5} {f.parse_rationale}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .qa import answer_question

    answer, sources = answer_question(args.question, use_llm=not args.no_llm)
    print(answer)
    if sources:
        print("\nSources:")
        for s in sources:
            print(" -", s)
    return 0


def main(argv: list[str] | None = None) -> int:
    # Make console printing robust on Windows (cp1252) regardless of what the
    # LLM emits; files are always written UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(prog="freight-watchdog",
                                     description="Smart assistant that watches shipping costs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="full pipeline run")
    p_run.add_argument("--no-llm", action="store_true",
                       help="deterministic-only run (template explanations)")
    p_run.set_defaults(func=cmd_run)

    sub.add_parser("verify-sample", help="verify metrics against sample_output_format_v2.csv") \
        .set_defaults(func=cmd_verify_sample)
    sub.add_parser("notes", help="show parsed context-note facts") \
        .set_defaults(func=cmd_notes)

    p_ask = sub.add_parser("ask", help="ask a question about the findings")
    p_ask.add_argument("question")
    p_ask.add_argument("--no-llm", action="store_true",
                       help="deterministic answer only")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
