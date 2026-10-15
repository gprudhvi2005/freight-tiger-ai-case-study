"""Detection: which route-weeks are rising and out of the ordinary?

A route-week becomes a *candidate anomaly* when its cost-per-tonne-km is
rising and looks out of the ordinary on EITHER graded comparison:

    own_pct >= OWN_HISTORY_THRESHOLD_PCT        (vs own trailing history)
    OR
    sim_pct >= SIMILAR_ROUTES_THRESHOLD_PCT     (vs same-week peers)

The candidate set is what moves on to note retrieval / adjudication.
"""

from __future__ import annotations

import pandas as pd

from . import config


def select_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return route-week rows that breach a threshold on either comparison.

    Rows without any own history (a route's first week) cannot be judged as
    "rising" and are excluded. Weeks with no peers (route_type group of one)
    are judged on own history alone.
    """
    df = metrics.copy()

    own_ok = df["own_history_weeks_used"] > 0
    own_breach = df["own_pct"] >= config.OWN_HISTORY_THRESHOLD_PCT
    sim_breach = df["sim_pct"] >= config.SIMILAR_ROUTES_THRESHOLD_PCT

    candidates = df[own_ok & (own_breach | sim_breach.fillna(False))].copy()
    candidates["partial_history"] = (
        candidates["own_history_weeks_used"] < config.OWN_HISTORY_WEEKS
    )
    return candidates.sort_values(["week_of", "route"]).reset_index(drop=True)
