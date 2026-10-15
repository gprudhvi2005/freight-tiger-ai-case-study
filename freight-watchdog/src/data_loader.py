"""Load and validate the input CSVs."""

from __future__ import annotations

import pandas as pd

from . import config


def load_shipments(path=None) -> pd.DataFrame:
    """Load shipment_records.csv with basic validation.

    Returns a frame with an added `route` (origin-destination) and
    `tonne_km` (quantity_tonnes * distance_km) column, shipment_date
    parsed as datetime.
    """
    path = path or config.SHIPMENTS_CSV
    df = pd.read_csv(path, parse_dates=["shipment_date"])

    required = [
        "shipment_id", "origin", "destination", "route_type", "material",
        "quantity_tonnes", "distance_km", "freight_cost_inr",
        "shipment_date", "transporter",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"shipment file missing columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError("shipment file contains null values in required columns")
    if (df["quantity_tonnes"] <= 0).any() or (df["distance_km"] <= 0).any():
        raise ValueError("quantity_tonnes and distance_km must be positive")

    df["route"] = df["origin"] + "-" + df["destination"]
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def load_notes(path=None) -> pd.DataFrame:
    """Load context_notes.csv (note_id, date, applies_to, note)."""
    path = path or config.NOTES_CSV
    notes = pd.read_csv(path, parse_dates=["date"])
    required = ["note_id", "date", "applies_to", "note"]
    missing = [c for c in required if c not in notes.columns]
    if missing:
        raise ValueError(f"notes file missing columns: {missing}")
    return notes
