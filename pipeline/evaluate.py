"""
Backtesting: train on years 2016–(N-1), predict year N, measure per-round MAE.

For each slot present in the test year:
  - Predict all rounds using only training-year data
  - Compare predicted vs actual closing rank per round
  - Report MAE per round and overall
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .config import (
    COL_YEAR, COL_ROUND, COL_CLOSE_RANK, ALL_ROUNDS,
)
from .loader import load
from .train import SLOT_COLS, SlotModel


def backtest(csv_path: str, test_year: int | None = None) -> dict:
    df = load(csv_path)
    all_years = sorted(df[COL_YEAR].unique())

    if test_year is None:
        test_year = all_years[-1]

    train_years = [y for y in all_years if y < test_year]
    print(f"Backtest  |  train: {train_years}  |  test: {test_year}")

    train_df = df[df[COL_YEAR].isin(train_years)]
    test_df  = df[df[COL_YEAR] == test_year]

    # Accumulators: {round_no: (actuals, predictions)}
    round_errors: dict[int, tuple[list, list]] = {r: ([], []) for r in ALL_ROUNDS}

    for key, test_grp in test_df.groupby(SLOT_COLS):
        slot_key = dict(zip(SLOT_COLS, key))
        train_grp = train_df[
            (train_df[list(SLOT_COLS)] == pd.Series(slot_key)).all(axis=1)
        ]
        if train_grp.empty:
            continue

        m = SlotModel()
        m.fit(train_grp)
        preds = m.predict_all_rounds(test_year, ALL_ROUNDS)

        test_by_round = test_grp.set_index(COL_ROUND)[COL_CLOSE_RANK].to_dict()
        for r in ALL_ROUNDS:
            if r not in test_by_round or r not in preds:
                continue
            round_errors[r][0].append(float(test_by_round[r]))
            round_errors[r][1].append(float(preds[r]))

    print(f"\n{'Round':<8} {'N slots':>8} {'MAE':>10}")
    print("─" * 30)
    all_act, all_pred = [], []
    for r in ALL_ROUNDS:
        act, pred = round_errors[r]
        if not act:
            continue
        mae = mean_absolute_error(act, pred)
        print(f"R{r:<7} {len(act):>8,} {mae:>10.1f}")
        all_act.extend(act)
        all_pred.extend(pred)

    overall_mae = mean_absolute_error(all_act, all_pred) if all_act else float("nan")
    print(f"{'Overall':<8} {len(all_act):>8,} {overall_mae:>10.1f}")

    return {
        "test_year":   test_year,
        "overall_mae": overall_mae,
        "round_errors": round_errors,
    }


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "josaa_ranks.csv"
    backtest(csv_path)
