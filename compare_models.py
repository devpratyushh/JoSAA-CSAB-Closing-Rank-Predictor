"""
Compare year-trend model variants for JoSAA and CSAB.

Runs backtest for every combination of source x trend_model and prints
a side-by-side MAE table so you can pick the best model for each source.

Usage:
    python compare_models.py                     # both sources, most recent year
    python compare_models.py josaa               # JoSAA only
    python compare_models.py josaa --year 2024   # test on 2024 instead
    python compare_models.py josaa josaa --year 2023 2024  # multiple years
"""

import sys
import os
import argparse
import pandas as pd

from pipeline.config import SOURCES
from pipeline.evaluate import backtest
from pipeline.train import TREND_MODELS

DESCRIPTIONS = {
    "ols":          "OLS — straight-line trend, equal weight all years",
    "theil_sen":    "Theil-Sen — robust OLS, median of pairwise slopes",
    "weighted_ols": "Weighted OLS — recent years weighted ~10x more",
    "median":       "Median — historical average, ignores trend direction",
}


def compare(source: str, test_year: int | None = None) -> pd.DataFrame:
    cfg = SOURCES[source]
    csv = cfg["csv"]
    rnds = cfg["rounds"]

    if not os.path.exists(csv):
        print(f"[{source}] CSV not found: {csv} — skipping.")
        return pd.DataFrame()

    rows = []
    for tm in TREND_MODELS:
        print(f"\n{'-'*60}")
        print(f"[{source.upper()}]  trend_model = {tm}")
        print(f"{'-'*60}")
        result = backtest(csv, test_year=test_year, rounds=rnds,
                          trend_model=tm, quiet=False)
        row = {"Model": tm, "Overall MAE": result["overall_mae"],
               "Test Year": result["test_year"]}
        for r in rnds:
            row[f"R{r}"] = result["round_maes"].get(r, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    mae_cols = ["Overall MAE"] + [c for c in df.columns if c.startswith("R")]
    df[mae_cols] = df[mae_cols].round(1)
    best_idx = df["Overall MAE"].idxmin()

    year_label = df["Test Year"].iloc[0]
    print(f"\n{'='*80}")
    print(f"  {source.upper()} — test year {year_label}  |  "
          f"best: {best_idx}  ({DESCRIPTIONS[best_idx]})")
    print(f"{'='*80}")
    print(df[["Overall MAE"] + [c for c in df.columns
              if c.startswith("R")]].to_string())
    print()
    for tm in TREND_MODELS:
        marker = " ◄ best" if tm == best_idx else ""
        print(f"  {tm:<14}  {DESCRIPTIONS[tm]}{marker}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="*", default=["josaa", "csab"],
                        choices=list(SOURCES), help="Sources to compare")
    parser.add_argument("--year", type=int, default=None,
                        help="Hold-out test year (default: most recent)")
    args = parser.parse_args()

    all_results = {}
    for src in args.sources:
        df = compare(src, test_year=args.year)
        if not df.empty:
            all_results[src] = df

    # If multiple sources, print a combined winner summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("  COMBINED SUMMARY")
        print(f"{'='*80}")
        for src, df in all_results.items():
            best = df["Overall MAE"].idxmin()
            best_mae = df.loc[best, "Overall MAE"]
            ols_mae  = df.loc["ols", "Overall MAE"]
            pct = (ols_mae - best_mae) / ols_mae * 100
            print(f"  {src.upper():<8}  best={best:<14}  "
                  f"MAE={best_mae:.1f}  ({pct:+.1f}% vs OLS)")
