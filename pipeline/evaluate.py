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

from .config import COL_YEAR, COL_ROUND, COL_CLOSE_RANK, ALL_ROUNDS, DEFAULT_TREND_MODEL
from .loader import load
from .train import SLOT_COLS, SlotModel


def backtest(
    csv_path:    str,
    test_year:   int | None  = None,
    rounds:      list[int] | None = None,
    trend_model: str = DEFAULT_TREND_MODEL,
    quiet:       bool = False,
    _df:         pd.DataFrame | None = None,
) -> dict:
    def log(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    df = _df if _df is not None else load(csv_path)
    all_years = sorted(df[COL_YEAR].unique())

    if test_year is None:
        test_year = all_years[-1]

    if rounds is None:
        rounds = ALL_ROUNDS

    train_years = [y for y in all_years if y < test_year]
    log(f"Backtest  |  train: {train_years}  |  test: {test_year}  |  trend={trend_model}")

    train_df = df[df[COL_YEAR].isin(train_years)]
    test_df  = df[df[COL_YEAR] == test_year]

    log("Grouping training data...")
    train_groups = {
        key: grp for key, grp in train_df.groupby(SLOT_COLS, sort=False)
    }
    log(f"Training slots: {len(train_groups):,}  |  "
        f"Test slots: {test_df[SLOT_COLS].drop_duplicates().shape[0]:,}")

    round_errors: dict[int, tuple[list, list]] = {r: ([], []) for r in rounds}

    for i, (key, test_grp) in enumerate(test_df.groupby(SLOT_COLS, sort=False)):
        train_grp = train_groups.get(key)
        if train_grp is None:
            continue

        m = SlotModel(trend_model=trend_model)
        m.fit(train_grp)
        preds = m.predict_all_rounds(test_year, rounds)

        test_by_round = test_grp.set_index(COL_ROUND)[COL_CLOSE_RANK].to_dict()
        for r in rounds:
            if r not in test_by_round or r not in preds:
                continue
            round_errors[r][0].append(float(test_by_round[r]))
            round_errors[r][1].append(float(preds[r]))

        if (i + 1) % 1000 == 0:
            log(f"  {i + 1:,} slots processed...")

    log(f"\n{'Round':<8} {'N slots':>8} {'MAE':>10}")
    log("-" * 30)
    all_act, all_pred = [], []
    round_maes: dict[int, float] = {}
    for r in rounds:
        act, pred = round_errors[r]
        if not act:
            continue
        mae = mean_absolute_error(act, pred)
        round_maes[r] = mae
        log(f"R{r:<7} {len(act):>8,} {mae:>10.1f}")
        all_act.extend(act)
        all_pred.extend(pred)

    overall_mae = mean_absolute_error(all_act, all_pred) if all_act else float("nan")
    log(f"{'Overall':<8} {len(all_act):>8,} {overall_mae:>10.1f}")

    return {
        "test_year":    test_year,
        "overall_mae":  overall_mae,
        "round_maes":   round_maes,
        "round_errors": round_errors,
    }


def tune_ensemble_weight(
    csv_path:    str,
    val_year:    int | None = None,
    rounds:      list[int] | None = None,
    trend_model: str = DEFAULT_TREND_MODEL,
    w_grid:      list[float] | None = None,
    quiet:       bool = False,
) -> dict:
    """
    Find the optimal ensemble weight w by grid search on a held-out year.

    Slot models are trained once on years < val_year; the weight sweep is then
    free (no retraining per candidate w).

    Returns:
        {
          "best_w":   float,
          "val_year": int,
          "results":  pd.DataFrame  (columns: w, overall_mae, R1, R2, ...),
        }
    """
    def log(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    if w_grid is None:
        w_grid = [round(x * 0.05, 2) for x in range(21)]  # 0.00 … 1.00 step 0.05

    df = load(csv_path)
    all_years = sorted(df[COL_YEAR].unique())

    if val_year is None:
        val_year = all_years[-1]

    if rounds is None:
        rounds = ALL_ROUNDS

    train_years = [y for y in all_years if y < val_year]
    log(f"Weight tune  |  train: {train_years}  |  val: {val_year}  |  trend={trend_model}")
    log(f"w grid: {w_grid}")

    train_df = df[df[COL_YEAR].isin(train_years)]
    val_df   = df[df[COL_YEAR] == val_year]

    # Train all slot models once
    log("Training slot models...")
    train_groups = {key: grp for key, grp in train_df.groupby(SLOT_COLS, sort=False)}
    slot_models: dict = {}
    for key, train_grp in train_groups.items():
        m = SlotModel(trend_model=trend_model)
        m.fit(train_grp)
        slot_models[key] = m
    log(f"Trained {len(slot_models):,} slot models.")

    # Pre-collect (actual, direct_signal, ratio_signal) per slot per round
    # so we can evaluate any w without re-running prediction
    actuals:      dict[int, list[float]] = {r: [] for r in rounds}
    directs:      dict[int, list[float]] = {r: [] for r in rounds}
    via_ratios:   dict[int, list[float]] = {r: [] for r in rounds}

    for key, val_grp in val_df.groupby(SLOT_COLS, sort=False):
        m = slot_models.get(key)
        if m is None:
            continue
        test_by_round = val_grp.set_index(COL_ROUND)[COL_CLOSE_RANK].to_dict()
        for r in rounds:
            if r not in test_by_round:
                continue
            # Decompose the two signals for this round
            if r in m.round_year_models:
                direct = float(m.round_year_models[r].predict([[val_year]])[0])
            else:
                direct = m.round_medians.get(r, m.round_medians.get(m.max_round, 0))

            if m.max_round in m.round_year_models:
                pred_final = float(m.round_year_models[m.max_round].predict([[val_year]])[0])
            else:
                pred_final = m.round_medians.get(m.max_round, direct)

            ratio     = m.round_ratios.get(r, m.round_ratios.get(m.max_round, 1.0))
            via_ratio = pred_final * ratio

            actuals[r].append(float(test_by_round[r]))
            directs[r].append(max(1.0, direct))
            via_ratios[r].append(max(1.0, via_ratio))

    # Sweep over w values
    rows = []
    for w in w_grid:
        all_act, all_pred = [], []
        row = {"w": w}
        for r in rounds:
            if not actuals[r]:
                continue
            preds = [max(1.0, w * d + (1 - w) * vr)
                     for d, vr in zip(directs[r], via_ratios[r])]
            mae = mean_absolute_error(actuals[r], preds)
            row[f"R{r}"] = round(mae, 1)
            all_act.extend(actuals[r])
            all_pred.extend(preds)
        row["overall_mae"] = round(mean_absolute_error(all_act, all_pred), 1) if all_act else float("nan")
        rows.append(row)

    results = pd.DataFrame(rows).set_index("w")
    best_w  = float(results["overall_mae"].idxmin())

    log(f"\n{'='*60}")
    log(f"  Ensemble weight tuning  |  val year {val_year}")
    log(f"{'='*60}")
    r_cols = [c for c in results.columns if c.startswith("R")]
    log(results[["overall_mae"] + r_cols].to_string())
    log(f"\n  Best w = {best_w}  (overall MAE {results.loc[best_w, 'overall_mae']:.1f})")
    log(f"  Default w = 0.5  (overall MAE {results.loc[0.5, 'overall_mae']:.1f})")
    improvement = (results.loc[0.5, "overall_mae"] - results.loc[best_w, "overall_mae"])
    log(f"  Improvement over default: {improvement:.1f} rank positions")

    return {"best_w": best_w, "val_year": val_year, "results": results}


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "josaa_ranks.csv"
    backtest(csv_path)
