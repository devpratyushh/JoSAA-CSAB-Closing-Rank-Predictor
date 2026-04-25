"""
Train and persist the prediction model.

Strategy
────────
For each unique (Institute, Program, Quota, Seat Type, Gender) combination
("slot"), fit a LinearRegression on (Year → Closing Rank).

Slots with fewer than MIN_YEARS_FOR_TREND data points get no trend model;
at prediction time they fall back to their historical median closing rank.

The entire fitted object is serialised with joblib so predict.py can load
it without re-reading the CSV.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import (
    COL_YEAR, COL_INSTITUTE, COL_PROGRAM,
    COL_QUOTA, COL_SEAT_TYPE, COL_GENDER, COL_EXAM_TYPE,
    COL_CLOSE_RANK,
    MIN_YEARS_FOR_TREND, MODEL_DIR, MODEL_PATH,
)
from .loader import load

# Columns that together identify one unique "seat slot"
SLOT_COLS = [COL_INSTITUTE, COL_PROGRAM, COL_QUOTA,
             COL_SEAT_TYPE, COL_GENDER, COL_EXAM_TYPE]


def train(csv_path: str) -> dict:
    """
    Returns a model dict:
    {
        "slots": {
            (institute, program, quota, seat_type, gender, exam_type): {
                "trend_model": LinearRegression | None,
                "median_close": float,
                "years_seen":   int,
                "last_close":   int,
            },
            ...
        }
    }
    """
    df = load(csv_path)
    print(f"Training on {len(df):,} rows across {df[COL_YEAR].nunique()} years.")

    slots = {}
    groups = df.groupby(SLOT_COLS)

    for key, grp in groups:
        grp = grp.sort_values(COL_YEAR)
        years  = grp[COL_YEAR].values.reshape(-1, 1)
        closes = grp[COL_CLOSE_RANK].values.astype(float)

        trend_model = None
        if len(grp) >= MIN_YEARS_FOR_TREND:
            trend_model = LinearRegression()
            trend_model.fit(years, closes)

        slots[key] = {
            "trend_model": trend_model,
            "median_close": float(np.median(closes)),
            "years_seen":   len(grp),
            "last_close":   int(closes[-1]),
        }

    model = {"slots": slots, "slot_cols": SLOT_COLS}

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained {len(slots):,} slot models → saved to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "josaa_ranks.csv"
    train(csv_path)
