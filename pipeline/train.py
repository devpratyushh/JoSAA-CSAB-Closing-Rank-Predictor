"""
Train the ensemble model that predicts closing rank for every round.

Two signals per slot (institute × program × quota × seat_type × gender):
─────────────────────────────────────────────────────────────────────────
  1. Year-trend per round
       For each round r, fit LinearRegression(year → closing_rank).
       Captures how the cutoff for *that specific round* has drifted year-over-year.

  2. Round-progression ratios
       For each year, record ratio[r] = close[r] / close[last_round_in_year].
       Average ratios across years to learn the typical R1→R2→...→Rfinal shape.
       At prediction time: predicted_close[r] = predicted_final_close × ratio[r].

Ensemble prediction for round r in target year Y:
       pred = w * direct_year_trend[r](Y)  +  (1-w) * (final_trend(Y) × ratio[r])
  where w = ENSEMBLE_WEIGHT (default 0.5).

Slots with < MIN_YEARS_FOR_TREND data points for a given round fall back to
the historical median for that round.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import (
    COL_YEAR, COL_ROUND, COL_INSTITUTE, COL_PROGRAM,
    COL_QUOTA, COL_SEAT_TYPE, COL_GENDER, COL_EXAM_TYPE,
    COL_CLOSE_RANK,
    MIN_YEARS_FOR_TREND, ENSEMBLE_WEIGHT,
    MODEL_DIR, MODEL_PATH,
)
from .loader import load

SLOT_COLS = [COL_INSTITUTE, COL_PROGRAM, COL_QUOTA,
             COL_SEAT_TYPE, COL_GENDER, COL_EXAM_TYPE]


class SlotModel:
    """
    Per-slot ensemble of year-trend models and round-progression ratios.
    """

    def __init__(self):
        # {round_no: LinearRegression}  — year → close_rank for that round
        self.round_year_models: dict[int, LinearRegression] = {}
        # {round_no: float}  — close[r] / close[last_round], averaged across years
        self.round_ratios:      dict[int, float] = {}
        # {round_no: float}  — fallback median closing rank per round
        self.round_medians:     dict[int, float] = {}
        self.max_round: int = 1
        self.n_years:   int = 0   # number of distinct years in training data

    def fit(self, slot_df: pd.DataFrame) -> None:
        """
        slot_df: all rows for one slot, all years, all rounds.
        """
        self.max_round = int(slot_df[COL_ROUND].max())
        self.n_years   = int(slot_df[COL_YEAR].nunique())

        # ── Per-round year-trend models ────────────────────────────────────
        for r, grp in slot_df.groupby(COL_ROUND):
            r = int(r)
            closes = grp[COL_CLOSE_RANK].values.astype(float)
            years  = grp[COL_YEAR].values.reshape(-1, 1)
            self.round_medians[r] = float(np.median(closes))
            if len(grp) >= MIN_YEARS_FOR_TREND:
                m = LinearRegression().fit(years, closes)
                self.round_year_models[r] = m

        # ── Round-progression ratios ───────────────────────────────────────
        # For each year, compute ratio[r] = close[r] / close[max_round_in_year].
        # Only use years that have *both* the round r and the max round.
        ratio_accum: dict[int, list[float]] = {}
        for year, year_grp in slot_df.groupby(COL_YEAR):
            year_rounds = year_grp.set_index(COL_ROUND)[COL_CLOSE_RANK].to_dict()
            final_r = int(year_grp[COL_ROUND].max())
            final_close = year_rounds.get(final_r)
            if not final_close or final_close == 0:
                continue
            for r, close in year_rounds.items():
                r = int(r)
                ratio_accum.setdefault(r, []).append(close / final_close)

        self.round_ratios = {r: float(np.mean(v)) for r, v in ratio_accum.items()}

    def predict_round(self, round_no: int, year: int) -> float:
        """
        Ensemble prediction for a single round.
        """
        # ── Signal 1: direct year trend for this round ─────────────────────
        if round_no in self.round_year_models:
            direct = float(self.round_year_models[round_no].predict([[year]])[0])
        else:
            direct = self.round_medians.get(round_no,
                     self.round_medians.get(self.max_round, 0))

        # ── Signal 2: scale from predicted final-round close ───────────────
        if self.max_round in self.round_year_models:
            pred_final = float(
                self.round_year_models[self.max_round].predict([[year]])[0]
            )
        else:
            pred_final = self.round_medians.get(self.max_round, direct)

        ratio = self.round_ratios.get(round_no,
                self.round_ratios.get(self.max_round, 1.0))
        via_ratio = pred_final * ratio

        pred = ENSEMBLE_WEIGHT * direct + (1 - ENSEMBLE_WEIGHT) * via_ratio
        return max(1.0, pred)

    def predict_all_rounds(self, year: int, rounds: list[int]) -> dict[int, int]:
        """Return {round_no: predicted_close_rank} for each requested round."""
        return {r: int(round(self.predict_round(r, year))) for r in rounds}


def train(csv_path: str, model_path: str = MODEL_PATH) -> dict:
    df = load(csv_path)
    print(f"Training on {len(df):,} rows  |  "
          f"{df[COL_YEAR].nunique()} years  |  "
          f"{df[COL_ROUND].nunique()} rounds")

    slots: dict[tuple, SlotModel] = {}
    for key, grp in df.groupby(SLOT_COLS):
        m = SlotModel()
        m.fit(grp)
        slots[tuple(key)] = m

    model = {"slots": slots, "slot_cols": SLOT_COLS}

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained {len(slots):,} slot models → {model_path}")
    return model


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "josaa_ranks.csv"
    train(csv_path)
