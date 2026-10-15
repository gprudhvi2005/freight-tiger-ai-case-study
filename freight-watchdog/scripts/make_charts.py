"""Walkthrough charts: weekly cost per tonne-km per route with flagged weeks.

Produces output/walkthrough_chart.png -- one panel per route showing:
  * the route's weekly cost per tonne-km (line),
  * its trailing 8-week own-history baseline (line),
  * the same-week peer average for its route_type (line),
  * flagged route-weeks (red dots),
  * justified route-weeks (green dots),
  * context-note windows for that route (shaded bands).

Run:  python scripts/make_charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data_loader import load_notes, load_shipments  # noqa: E402
from src.weekly_metrics import build_metrics  # noqa: E402


def main() -> int:
    metrics = build_metrics(load_shipments())
    notes = load_notes()
    with open(config.OUTPUT_DIR / "run_diagnostics.json", encoding="utf-8") as fh:
        diagnostics = json.load(fh)

    flagged = {(r["route"], r["week_of"]): r for r in diagnostics["rows"]}

    routes = sorted(metrics["route"].unique())
    fig, axes = plt.subplots(len(routes), 1, figsize=(13, 2.4 * len(routes)),
                             sharex=True)

    for ax, route in zip(axes, routes):
        df = metrics[metrics["route"] == route].sort_values("week_of")
        ax.plot(df["week_of"], df["cost_per_tonne_km"], lw=1.2,
                color="#1f77b4", label="cost / tonne-km")
        ax.plot(df["week_of"], df["own_avg_cpk"], lw=1.0, ls="--",
                color="#7f7f7f", label="own 8-week avg")
        ax.plot(df["week_of"], df["peer_avg_cpk"], lw=1.0, ls=":",
                color="#2ca02c", label="peer avg (same route_type)")

        # flagged / justified weeks
        for _, row in df.iterrows():
            key = (route, str(row["week_of"].date()))
            if key in flagged:
                d = flagged[key]
                color = "#d62728" if not d["justified"] else "#2ca02c"
                ax.plot(row["week_of"], row["cost_per_tonne_km"], "o",
                        ms=6, color=color, zorder=5,
                        markeredgecolor="black", markeredgewidth=0.5)

        # note windows touching this route
        for _, note in notes.iterrows():
            applies = str(note["applies_to"]).strip()
            if applies not in (route, "All Routes"):
                continue
            text = str(note["note"])
            import re

            m = re.search(r"from ([A-Za-z]{3,9})\.? (\d{1,2}) to ([A-Za-z]{3,9})\.? (\d{1,2})",
                          text, re.IGNORECASE)
            months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                      "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
            date = pd.Timestamp(note["date"])
            if m:
                start = pd.Timestamp(year=date.year, month=months[m.group(1)[:3].lower()],
                                     day=int(m.group(2)))
                end = pd.Timestamp(year=date.year, month=months[m.group(3)[:3].lower()],
                                   day=int(m.group(4)))
            else:
                start = date
                end = date + pd.Timedelta(days=6)
            alpha = 0.18 if applies == route else 0.08
            ax.axvspan(start, end, color="#ff7f0e", alpha=alpha)
            ax.text(start, ax.get_ylim()[1], f" {note['note_id']}", fontsize=7,
                    color="#ff7f0e", va="top")

        ax.set_title(f"{route}  ({df['route_type'].iloc[0]})", fontsize=10, loc="left")
        ax.set_ylabel("INR / tonne-km", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    handles = [
        plt.Line2D([], [], color="#1f77b4", label="cost / tonne-km"),
        plt.Line2D([], [], color="#7f7f7f", ls="--", label="own 8-week avg"),
        plt.Line2D([], [], color="#2ca02c", ls=":", label="peer avg (route_type)"),
        plt.Line2D([], [], color="#d62728", marker="o", ls="", label="flagged (unexplained)"),
        plt.Line2D([], [], color="#2ca02c", marker="o", ls="", label="justified (note found)"),
        plt.Line2D([], [], color="#ff7f0e", alpha=0.3, lw=6, label="context-note window"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Freight Cost Watchdog -- weekly cost per tonne-km, baselines and verdicts",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    out = config.OUTPUT_DIR / "walkthrough_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
