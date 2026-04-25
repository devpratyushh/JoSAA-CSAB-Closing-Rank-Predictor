"""
Prediction engine.

Given a student profile, returns a ranked list of (institute, program)
combinations they are likely to be admitted to in PREDICT_YEAR.

Result categories
─────────────────
  safe   → your rank is comfortably below predicted closing rank  (≤ 80 % of it)
  match  → your rank is within 80–100 % of predicted closing rank
  reach  → your rank exceeds predicted closing rank but is within 120 %
            (closing ranks shift year-to-year; could still be possible)
"""

import pickle
import numpy as np
import pandas as pd
from .config import MODEL_PATH, PREDICT_YEAR, COL_EXAM_TYPE
from .train import SLOT_COLS


SAFE_THRESHOLD  = 0.80   # rank ≤ 80 % of predicted close  → safe
REACH_THRESHOLD = 1.20   # rank ≤ 120 % of predicted close → reach (borderline)


def load_model(path: str = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _predict_close(slot_data: dict, year: int) -> float:
    """Predict closing rank for `year` from a slot's fitted data."""
    m = slot_data["trend_model"]
    if m is not None:
        pred = m.predict([[year]])[0]
        # Closing ranks must be positive integers
        return max(1.0, pred)
    # Fallback: median of historical closing ranks
    return slot_data["median_close"]


def predict(
    rank:       int,
    exam_type:  str,          # "advanced" or "mains"
    quota:      str,          # e.g. "AI", "HS", "OS"
    seat_type:  str,          # e.g. "OPEN", "OBC-NCL", "SC", "ST", "EWS"
    gender:     str,          # "Gender-Neutral" or "Female-only (including Supernumerary)"
    model:      dict | None = None,
    year:       int = PREDICT_YEAR,
    include_reach: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame sorted by predicted closing rank (ascending = more
    competitive / better college first):

        Institute | Academic Program Name | Quota | Seat Type | Gender |
        Predicted Close | Last Close | Years Seen | Category
    """
    if model is None:
        model = load_model()

    slots = model["slots"]
    results = []

    for key, data in slots.items():
        # key order matches SLOT_COLS:
        # (Institute, Program, Quota, SeatType, Gender, ExamType)
        inst, prog, q, st, g, et = key

        if et != exam_type:
            continue
        if q != quota:
            continue
        if st != seat_type:
            continue
        if g != gender:
            continue

        pred_close = _predict_close(data, year)

        ratio = rank / pred_close
        if ratio <= SAFE_THRESHOLD:
            category = "safe"
        elif ratio <= 1.0:
            category = "match"
        elif ratio <= REACH_THRESHOLD and include_reach:
            category = "reach"
        else:
            continue   # rank too high; not eligible

        results.append({
            "Institute":            inst,
            "Academic Program Name": prog,
            "Quota":                q,
            "Seat Type":            st,
            "Gender":               g,
            "Predicted Close":      int(round(pred_close)),
            "Last Close":           data["last_close"],
            "Years Seen":           data["years_seen"],
            "Category":             category,
        })

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    # Sort: safe first, then match, then reach; within each, ascending predicted close
    cat_order = {"safe": 0, "match": 1, "reach": 2}
    out["_cat_order"] = out["Category"].map(cat_order)
    out.sort_values(["_cat_order", "Predicted Close"], inplace=True)
    out.drop(columns=["_cat_order"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out
