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

from .config import COL_YEAR, COL_ROUND, COL_CLOSE_RANK, ALL_ROUNDS
from .loader import load
from .train import SLOT_COLS, SlotModel


def backtest(csv_path: str, test_year: int | None = None, rounds: list[int] | None = None) -> dict:
    df = load(csv_path)
    all_years = sorted(df[COL_YEAR].unique())

    if test_year is None:
        test_year = all_years[-1]

    train_years = [y for y in all_years if y < test_year]
    print(f"Backtest  |  train: {train_years}  |  test: {test_year}")

    if rounds is None:
        rounds = ALL_ROUNDS

    train_df = df[df[COL_YEAR].isin(train_years)]
    test_df  = df[df[COL_YEAR] == test_year]

    # Pre-group training data by slot key — O(1) lookup instead of O(n_rows) scan
    print("Grouping training data...")
    train_groups = {
        key: grp for key, grp in train_df.groupby(SLOT_COLS, sort=False)
    }
    print(f"Training slots: {len(train_groups):,}  |  Test slots: {test_df[SLOT_COLS].drop_duplicates().shape[0]:,}")

    # Accumulators: {round_no: (actuals, predictions)}
    round_errors: dict[int, tuple[list, list]] = {r: ([], []) for r in rounds}

    for i, (key, test_grp) in enumerate(test_df.groupby(SLOT_COLS, sort=False)):
        train_grp = train_groups.get(key)
        if train_grp is None:
            continue

        m = SlotModel()
        m.fit(train_grp)
        preds = m.predict_all_rounds(test_year, rounds)

        test_by_round = test_grp.set_index(COL_ROUND)[COL_CLOSE_RANK].to_dict()
        for r in rounds:
            if r not in test_by_round or r not in preds:
                continue
            round_errors[r][0].append(float(test_by_round[r]))
            round_errors[r][1].append(float(preds[r]))

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,} slots processed...")

    print(f"\n{'Round':<8} {'N slots':>8} {'MAE':>10}")
    print("─" * 30)
    all_act, all_pred = [], []
    for r in rounds:
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
        "test_year":    test_year,
        "overall_mae":  overall_mae,
        "round_errors": round_errors,
    }


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "josaa_ranks.csv"
    backtest(csv_path)
