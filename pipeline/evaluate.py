"""
Backtesting: train on years 2016–(N-1), predict year N, measure accuracy.

Accuracy metric: for each slot, did the student's rank fall below the
actual closing rank when we predicted they would (and vice versa)?
We report Mean Absolute Error (MAE) of predicted vs actual closing rank.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from .config import (
    COL_YEAR, COL_CLOSE_RANK, MIN_YEARS_FOR_TREND,
)
from .loader import load
from .train import SLOT_COLS


def backtest(csv_path: str, test_year: int | None = None) -> dict:
    """
    If test_year is None, uses the most recent year in the data as the test year.
    Returns a dict with overall MAE and per-slot errors.
    """
    df = load(csv_path)
    all_years = sorted(df[COL_YEAR].unique())

    if test_year is None:
        test_year = all_years[-1]

    train_years = [y for y in all_years if y < test_year]
    print(f"Backtest: train on {train_years}, test on {test_year}")

    train_df = df[df[COL_YEAR].isin(train_years)]
    test_df  = df[df[COL_YEAR] == test_year]

    actuals    = []
    predictions = []

    for key, test_grp in test_df.groupby(SLOT_COLS):
        train_grp = train_df[
            (train_df[list(SLOT_COLS)] == pd.Series(dict(zip(SLOT_COLS, key)))).all(axis=1)
        ]
        if train_grp.empty:
            continue

        closes = train_grp[COL_CLOSE_RANK].values.astype(float)
        years  = train_grp[COL_YEAR].values.reshape(-1, 1)

        if len(train_grp) >= MIN_YEARS_FOR_TREND:
            m = LinearRegression().fit(years, closes)
            pred = max(1.0, m.predict([[test_year]])[0])
        else:
            pred = float(np.median(closes))

        actual = float(test_grp[COL_CLOSE_RANK].iloc[0])
        actuals.append(actual)
        predictions.append(pred)

    mae = mean_absolute_error(actuals, predictions) if actuals else float("nan")
    print(f"MAE on {test_year}: {mae:.1f} rank positions across {len(actuals):,} slots")

    return {
        "test_year":   test_year,
        "mae":         mae,
        "n_slots":     len(actuals),
        "actuals":     actuals,
        "predictions": predictions,
    }


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "josaa_ranks.csv"
    backtest(csv_path)
