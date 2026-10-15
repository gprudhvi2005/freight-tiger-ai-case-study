"""Weekly cost metrics and the two graded comparison baselines.

Definitions (verified to reproduce every row of
data/sample_output_format_v2.csv exactly -- see tests/test_core.py):

* Weeks run Monday..Sunday. ``week_of`` is the Monday date.
* ``cost_per_tonne_km`` for a route-week is
      sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)
  i.e. pooled over the week's shipments (tonne-km weighted).
* ``vs_own_history`` compares the week's cost-per-tonne-km to the simple
  mean of the trailing 8 weekly cost-per-tonne-km values of the SAME route,
  using only weeks strictly before the current week (no look-ahead).
  If fewer than 8 prior weeks exist, all available prior weeks are used and
  ``own_history_weeks_used`` records how many (README documents this).
* ``vs_similar_routes`` compares to the simple mean of the same-week
  cost-per-tonne-km of every OTHER route with the same route_type.
"""

from __future__ import annotations

import pandas as pd

from . import config


def week_start(date: pd.Series) -> pd.Series:
    """Map each shipment_date to its week's Monday (week_of)."""
    return date - pd.to_timedelta(date.dt.weekday, unit="D")


def build_route_week_table(shipments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shipments to one row per (route, week_of).

    Columns: route, route_type, week_of, total_cost, total_tonne_km,
    n_shipments, cost_per_tonne_km.
    """
    df = shipments.copy()
    df["week_of"] = week_start(df["shipment_date"])

    weekly = (
        df.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(
            total_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            n_shipments=("shipment_id", "count"),
        )
        .sort_values(["route", "week_of"])
        .reset_index(drop=True)
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["total_tonne_km"]
    return weekly


def add_baselines(route_weeks: pd.DataFrame) -> pd.DataFrame:
    """Add own-history and similar-route baselines to the route-week table.

    Added columns:
      own_avg_cpk        trailing-average cost-per-tonne-km (strictly prior weeks)
      own_history_weeks_used  how many prior weeks fed that average (0 for the
                         first week of a route -- no baseline exists yet)
      own_pct            percent deviation vs own history (NaN if no history)
      peer_avg_cpk       same-week average of other routes with same route_type
      sim_pct            percent deviation vs similar routes (NaN if no peers)
    """
    df = route_weeks.copy()

    # --- own history: trailing N prior weeks, no look-ahead -----------------
    df = df.sort_values(["route", "week_of"]).reset_index(drop=True)
    trailing = (
        df.groupby("route")["cost_per_tonne_km"]
        .transform(lambda s: s.shift(1).rolling(config.OWN_HISTORY_WEEKS, min_periods=1).mean())
    )
    used = (
        df.groupby("route")["cost_per_tonne_km"]
        .transform(lambda s: s.shift(1).rolling(config.OWN_HISTORY_WEEKS, min_periods=1).count())
    )
    df["own_avg_cpk"] = trailing
    df["own_history_weeks_used"] = used.astype(int)
    df["own_pct"] = (df["cost_per_tonne_km"] / df["own_avg_cpk"] - 1.0) * 100.0

    # --- similar routes: same week, same route_type, excluding this route ---
    grp = df.groupby(["route_type", "week_of"])["cost_per_tonne_km"].agg(["mean", "count"])
    grp = grp.reset_index().rename(columns={"mean": "rt_mean", "count": "rt_count"})
    df = df.merge(grp, on=["route_type", "week_of"], how="left")
    # leave-one-out mean: (sum of peers) / (count - 1)
    df["peer_avg_cpk"] = (df["rt_mean"] * df["rt_count"] - df["cost_per_tonne_km"]) / (
        df["rt_count"] - 1
    )
    df["sim_pct"] = (df["cost_per_tonne_km"] / df["peer_avg_cpk"] - 1.0) * 100.0

    return df.drop(columns=["rt_mean", "rt_count"])


def build_metrics(shipments: pd.DataFrame) -> pd.DataFrame:
    """Full metrics table: route-weeks + both baselines."""
    return add_baselines(build_route_week_table(shipments))
