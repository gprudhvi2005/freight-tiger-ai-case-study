"""Unit tests for the deterministic core.

Run:  python -m pytest tests/ -q        (from the project root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.adjudicator import judge_candidate, parse_note  # noqa: E402
from src.data_loader import load_notes, load_shipments  # noqa: E402
from src.notes_kb import RetrievedNote  # noqa: E402
from src.weekly_metrics import build_metrics, week_start  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def shipments():
    return load_shipments()


@pytest.fixture(scope="session")
def notes():
    return load_notes()


@pytest.fixture(scope="session")
def metrics(shipments):
    return build_metrics(shipments)


# ---------------------------------------------------------------------------
# Weekly metrics
# ---------------------------------------------------------------------------

class TestWeeklyMetrics:
    def test_sample_rows_reproduce_exactly(self, metrics):
        """The three sample_output_format_v2.csv rows, number for number."""
        expected = [
            ("Delhi-Jaipur", "2024-11-11", 4.17, 35.5, 21.0),
            ("Ahmedabad-Mumbai", "2025-01-20", 3.29, 29.5, 22.5),
            ("Mumbai-Pune", "2025-09-15", 3.98, 9.2, 23.6),
        ]
        for route, week, cpk, own, sim in expected:
            row = metrics[(metrics["route"] == route)
                          & (metrics["week_of"] == pd.Timestamp(week))].iloc[0]
            assert round(row["cost_per_tonne_km"], 2) == cpk
            assert round(row["own_pct"], 1) == own
            assert round(row["sim_pct"], 1) == sim

    def test_week_of_is_always_monday(self, shipments):
        ws = week_start(shipments["shipment_date"])
        assert (ws.dt.weekday == 0).all()

    def test_own_history_has_no_lookahead(self):
        """own_avg for week N must be computable from weeks strictly before N."""
        weeks = pd.date_range("2024-01-01", periods=6, freq="W-MON")
        rw = pd.DataFrame({
            "route": ["R"] * 6,
            "route_type": ["Short"] * 6,
            "week_of": weeks,
            "total_cost": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "total_tonne_km": [50.0] * 6,
            "n_shipments": [1] * 6,
            "cost_per_tonne_km": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        })
        from src.weekly_metrics import add_baselines

        out = add_baselines(rw)
        # Week 0: no history -> NaN. Week 1: mean of [2.0]. Week 2: mean [2,4].
        assert pd.isna(out.loc[0, "own_avg_cpk"])
        assert out.loc[0, "own_history_weeks_used"] == 0
        assert out.loc[1, "own_avg_cpk"] == pytest.approx(2.0)
        assert out.loc[2, "own_avg_cpk"] == pytest.approx(3.0)
        # trailing 8 (fewer if unavailable) -- here weeks 1..5 use all prior
        assert out.loc[5, "own_avg_cpk"] == pytest.approx((2 + 4 + 6 + 8 + 10) / 5)

    def test_peer_average_excludes_self(self):
        """leave-one-out: peer avg must not contain the route's own value."""
        weeks = pd.date_range("2024-01-01", periods=1, freq="W-MON")
        rw = pd.DataFrame({
            "route": ["A", "B", "C"],
            "route_type": ["Medium"] * 3,
            "week_of": list(weeks) * 3,
            "total_cost": [100.0, 200.0, 300.0],
            "total_tonne_km": [50.0, 50.0, 50.0],
            "n_shipments": [1, 1, 1],
            "cost_per_tonne_km": [2.0, 4.0, 6.0],
        })
        from src.weekly_metrics import add_baselines

        out = add_baselines(rw)
        a, b, c = out.iloc[0], out.iloc[1], out.iloc[2]
        assert a["peer_avg_cpk"] == pytest.approx(5.0)   # (4+6)/2
        assert b["peer_avg_cpk"] == pytest.approx(4.0)   # (2+6)/2
        assert c["peer_avg_cpk"] == pytest.approx(3.0)   # (2+4)/2

    def test_full_coverage_728_route_weeks(self, metrics):
        assert len(metrics) == 728  # 7 routes x 104 weeks
        assert metrics["cost_per_tonne_km"].notna().all()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class TestDetector:
    def test_candidate_count_on_this_dataset(self, metrics):
        from src.detector import select_candidates

        candidates = select_candidates(metrics)
        assert len(candidates) == 20
        by_route = candidates.groupby("route").size().to_dict()
        assert by_route == {"Delhi-Jaipur": 2, "Ahmedabad-Mumbai": 1,
                            "Chennai-Bangalore": 4, "Mumbai-Pune": 13}

    def test_first_week_cannot_be_flagged(self, metrics):
        from src.detector import select_candidates

        candidates = select_candidates(metrics)
        # every route's first week is excluded (no history to compare to)
        firsts = metrics.groupby("route")["week_of"].min()
        for route, week in firsts.items():
            assert candidates[(candidates["route"] == route)
                              & (candidates["week_of"] == week)].empty


# ---------------------------------------------------------------------------
# Note parsing
# ---------------------------------------------------------------------------

class TestNoteParsing:
    def test_all_notes_parse_as_expected(self, notes):
        facts = {str(r["note_id"]): parse_note(str(r["note_id"]), str(r["applies_to"]),
                                               pd.Timestamp(r["date"]), str(r["note"]))
                 for _, r in notes.iterrows()}

        # scope
        assert facts["N001"].routes == frozenset({"Chennai-Bangalore"})
        assert facts["N002"].routes == frozenset({"Ahmedabad-Mumbai"})
        assert facts["N003"].routes is None           # all routes
        assert facts["N004"].routes == frozenset()    # not part of this dataset
        assert facts["N005"].routes == frozenset({"Mumbai-Delhi"})
        assert facts["N007"].routes == frozenset({"Delhi-Jaipur"})

        # direction
        assert facts["N001"].direction == "cost_up"
        assert facts["N002"].direction == "cost_up"
        assert facts["N003"].direction == "cost_up"
        assert facts["N004"].direction == "cost_up"   # but applies to no route
        assert facts["N005"].direction == "no_effect"
        assert facts["N006"].direction == "no_effect"
        assert facts["N007"].direction == "cost_down"
        assert facts["N008"].direction == "no_effect"
        assert facts["N009"].direction == "no_effect"
        assert facts["N010"].direction == "no_effect"

        # windows
        assert facts["N001"].window_start == pd.Timestamp("2025-02-24")
        assert facts["N001"].window_end == pd.Timestamp("2025-03-08")
        assert facts["N002"].window_start == pd.Timestamp("2025-01-20")
        assert facts["N002"].window_end == pd.Timestamp("2025-01-26")
        assert facts["N003"].window_start == pd.Timestamp("2025-05-05")
        assert facts["N003"].window_end is None       # ongoing
        assert facts["N005"].window_end == pd.Timestamp("2024-08-04")

        # magnitude
        assert facts["N003"].magnitude_upper_pct == pytest.approx(7.0)
        assert facts["N001"].magnitude_upper_pct is None
        assert facts["N002"].magnitude_upper_pct is None


# ---------------------------------------------------------------------------
# Adjudication rules
# ---------------------------------------------------------------------------

def _note_stub(facts, similarity=0.9) -> RetrievedNote:
    return RetrievedNote(note_id=facts.note_id, similarity=similarity,
                         applies_to="x", date=facts.window_start, note="stub")


FACTS = {
    nid: None for nid in ["N001", "N002", "N003", "N004", "N005",
                          "N006", "N007", "N008", "N009", "N010"]
}


@pytest.fixture(scope="class")
def parsed_notes(notes):
    return {str(r["note_id"]): parse_note(str(r["note_id"]), str(r["applies_to"]),
                                          pd.Timestamp(r["date"]), str(r["note"]))
            for _, r in notes.iterrows()}


class TestAdjudication:
    def test_flood_justifies_only_within_window(self, parsed_notes):
        f = parsed_notes["N001"]
        # weeks inside the flood window pass
        for week, own, sim in [("2025-02-24", 31.5, 38.7), ("2025-03-03", 22.8, 32.0)]:
            d = judge_candidate(f, "Chennai-Bangalore", pd.Timestamp(week), own, sim)
            assert d.passed, f"{week} should be justified by N001"
        # the week after the window ends must NOT pass
        for week in ["2025-03-10", "2025-03-17"]:
            d = judge_candidate(f, "Chennai-Bangalore", pd.Timestamp(week), 22.1, 33.8)
            assert not d.passed and d.rejection == "window"

    def test_flood_note_rejects_other_routes(self, parsed_notes):
        d = judge_candidate(parsed_notes["N001"], "Mumbai-Pune",
                            pd.Timestamp("2025-02-24"), 30.0, 30.0)
        assert not d.passed and d.rejection == "scope"

    def test_festival_justifies_its_week_only(self, parsed_notes):
        f = parsed_notes["N002"]
        ok = judge_candidate(f, "Ahmedabad-Mumbai", pd.Timestamp("2025-01-20"), 29.5, 22.5)
        assert ok.passed
        next_week = judge_candidate(f, "Ahmedabad-Mumbai", pd.Timestamp("2025-01-27"), 29.5, 22.5)
        assert not next_week.passed and next_week.rejection == "window"

    def test_diesel_cannot_explain_big_divergence(self, parsed_notes):
        f = parsed_notes["N003"]
        # +30% own / +25% vs peers: all-routes note is insufficient
        d = judge_candidate(f, "Mumbai-Pune", pd.Timestamp("2025-12-01"), 7.7, 37.4)
        assert not d.passed and d.rejection == "peer_divergence"
        # even a pure own-history rise beyond 7% (+tolerance) is too big
        d2 = judge_candidate(f, "Mumbai-Pune", pd.Timestamp("2025-12-01"), 25.0, 3.0)
        assert not d2.passed and d2.rejection == "magnitude"

    def test_no_effect_notes_never_justify(self, parsed_notes):
        for nid, route, week in [
            ("N005", "Mumbai-Delhi", "2024-07-29"),
            ("N006", "Mumbai-Pune", "2025-09-22"),
            ("N008", "Kolkata-Bhubaneswar", "2025-06-09"),
            ("N009", "Chennai-Bangalore", "2025-03-17"),
            ("N010", "Mumbai-Pune", "2025-10-27"),
        ]:
            d = judge_candidate(parsed_notes[nid], route, pd.Timestamp(week), 30.0, 30.0)
            assert not d.passed and d.rejection == "direction", nid

    def test_road_improvement_cannot_justify_rise(self, parsed_notes):
        # in the note's own week the window overlaps, so only the direction
        # rule can (and must) reject it
        d = judge_candidate(parsed_notes["N007"], "Delhi-Jaipur",
                            pd.Timestamp("2024-05-20"), 35.5, 21.0)
        assert not d.passed and d.rejection == "direction"
        # months later it is rejected on the window first (rule order)
        d2 = judge_candidate(parsed_notes["N007"], "Delhi-Jaipur",
                             pd.Timestamp("2024-11-11"), 35.5, 21.0)
        assert not d2.passed and d2.rejection == "window"

    def test_toll_note_applies_to_no_route(self, parsed_notes):
        d = judge_candidate(parsed_notes["N004"], "Mumbai-Pune",
                            pd.Timestamp("2024-03-11"), 30.0, 30.0)
        assert not d.passed and d.rejection == "scope"


# ---------------------------------------------------------------------------
# Output format contract
# ---------------------------------------------------------------------------

class TestOutputFormat:
    def test_columns_match_sample_header(self):
        sample_header = (config.DATA_DIR / "sample_output_format_v2.csv") \
            .read_text(encoding="utf-8").splitlines()[0].split(",")
        report_header = (config.OUTPUT_DIR / "anomaly_report.csv") \
            .read_text(encoding="utf-8").splitlines()[0].split(",")
        assert report_header == sample_header

    def test_flag_note_consistency(self):
        """A note id is present iff the verdict is justified."""
        report = pd.read_csv(config.OUTPUT_DIR / "anomaly_report.csv", keep_default_na=False)
        for _, r in report.iterrows():
            has_note = bool(str(r["matched_note_id"]).strip())
            is_justified = str(r["flagged"]).startswith("No")
            assert has_note == is_justified, f"{r['route']} {r['week_of']}"
            assert str(r["flagged"]) in ("Yes", "No (justified)")
